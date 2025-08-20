import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import Optional, Tuple
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import NormalizationMode, FeatureType, PolicyFeature
from torch.utils.data import DataLoader, random_split
import numpy as np
from lerobot.common.policies.normalize import Normalize, Unnormalize

# Add imports for LeRobot integration
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineAnnealingSchedulerConfig
from dataclasses import dataclass, field

# Configuration class for the ResNet + Transformer model
@PreTrainedConfig.register_subclass("resnet_transformer")
@dataclass
class ResNetTransformerConfig(PreTrainedConfig):
    """Configuration for ResNet + Transformer policy with LoRA adaptation."""
    
    # Model architecture
    action_dim: int = 2
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    resnet_type: str = 'resnet18'
    lora_rank: int = 16
    lora_alpha: float = 32
    dropout: float = 0.05
    
    # Training parameters
    optimizer_lr: float = 1.0e-4
    optimizer_min_lr: float = 1.0e-4
    optimizer_lr_cycle_steps: int = 300000
    optimizer_weight_decay: float = 1e-5
    
    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineAnnealingSchedulerConfig(
            min_lr=self.optimizer_min_lr, T_max=self.optimizer_lr_cycle_steps
        )


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for efficient fine-tuning."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA adapters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Access the buffer as a tensor and slice it
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with LoRA adaptation."""
    
    def __init__(self, d_model: int, num_heads: int, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Base attention projections
        self.w_q_base = nn.Linear(d_model, d_model, bias=False)
        self.w_k_base = nn.Linear(d_model, d_model, bias=False)
        self.w_v_base = nn.Linear(d_model, d_model, bias=False)
        self.w_o_base = nn.Linear(d_model, d_model, bias=False)
        
        # LoRA adapters
        self.w_q_lora = LoRALayer(d_model, d_model, rank, alpha, dropout)
        self.w_k_lora = LoRALayer(d_model, d_model, rank, alpha, dropout)
        self.w_v_lora = LoRALayer(d_model, d_model, rank, alpha, dropout)
        self.w_o_lora = LoRALayer(d_model, d_model, rank, alpha, dropout)
        
        # Freeze base layers
        for param in [self.w_q_base, self.w_k_base, self.w_v_base, self.w_o_base]:
            for p in param.parameters():
                p.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Apply base + LoRA projections
        Q = self.w_q_base(query) + self.w_q_lora(query)
        K = self.w_k_base(key) + self.w_k_lora(key)
        V = self.w_v_base(value) + self.w_v_lora(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Apply output projection
        output = self.w_o_base(attention_output) + self.w_o_lora(attention_output)
        
        return self.layer_norm(output + query)


class FeedForward(nn.Module):
    """Feed-forward network with LoRA adaptation."""
    
    def __init__(self, d_model: int, d_ff: int, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.w1_base = nn.Linear(d_model, d_ff)
        self.w2_base = nn.Linear(d_ff, d_model)
        self.w1_lora = LoRALayer(d_model, d_ff, rank, alpha, dropout)
        self.w2_lora = LoRALayer(d_ff, d_model, rank, alpha, dropout)
        
        # Freeze base layers
        for param in [self.w1_base, self.w2_base]:
            for p in param.parameters():
                p.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply base + LoRA
        ff1 = self.w1_base(x) + self.w1_lora(x)
        ff1 = F.relu(ff1)
        ff1 = self.dropout(ff1)
        
        ff2 = self.w2_base(ff1) + self.w2_lora(ff1)
        ff2 = self.dropout(ff2)
        
        return self.layer_norm(ff2 + x)


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with LoRA adaptation."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, rank, alpha, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, rank, alpha, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, rank, alpha, dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = self.self_attention(x, x, x, tgt_mask)
        
        # Cross-attention to encoder output (ResNet + state features)
        x = self.cross_attention(x, encoder_output, encoder_output)
        
        # Feed-forward
        x = self.feed_forward(x)
        
        return x


class ResNetFeatureExtractor(nn.Module):
    """ResNet-based feature extractor with LoRA adaptation."""
    
    def __init__(self, resnet_type: str = 'resnet18', d_model: int = 512, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        
        # Load pre-trained ResNet
        if resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            self.feature_dim = 512
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
        
        # Remove classification head
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Freeze ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Projection layer with LoRA
        self.projection = LoRALinear(self.feature_dim, d_model, rank, alpha, dropout)
        
        # Spatial flattening layer - LeRobot images are already 96x96
        self.spatial_flatten = nn.AdaptiveAvgPool2d((3, 3))  # Force 3x3 output for 96x96 input
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch_size, 3, 96, 96] - LeRobot images are already the right size
        features = self.resnet(x)  # [batch_size, 512, H, W]
        
        # Force spatial dimensions to 3x3
        features = self.spatial_flatten(features)  # [batch_size, 512, 3, 3]
        
        # Flatten spatial dimensions
        batch_size = features.size(0)
        features = features.view(batch_size, self.feature_dim, -1)  # [batch_size, 512, 9]
        features = features.transpose(1, 2)  # [batch_size, 9, 512]
        
        # Project to d_model
        features = self.projection(features)  # [batch_size, 9, 512]
        
        return features


class StateEmbedding(nn.Module):
    """Embedding layer for robot state vectors with LoRA adaptation."""
    
    def __init__(self, d_model: int = 512, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.embedding = LoRALinear(2, d_model, rank, alpha, dropout)  # 2D state vector
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch_size, 2] - robot joint positions
        # Output: [batch_size, 1, 512] - add sequence dimension for transformer
        embedded = self.embedding(x)  # [batch_size, 512]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, 512]
        return embedded


class ResNetTransformerDecoder(nn.Module):
    """Main model: ResNet + Transformer Decoder with LoRA adaptation for action prediction."""
    
    def __init__(
        self,
        action_dim: int = 2,  # Changed from vocab_size to action_dim
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        resnet_type: str = 'resnet18',
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.05,
        normalization_mapping: Optional[dict[str, NormalizationMode]] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.action_dim = action_dim  # Robot action dimension
        
        # Default normalization mapping (min_max for all)
        if normalization_mapping is None:
            normalization_mapping = {
                "VISUAL": NormalizationMode.MIN_MAX,
                "STATE": NormalizationMode.MIN_MAX,
                "ACTION": NormalizationMode.MIN_MAX,
            }
        
        self.normalization_mapping = normalization_mapping
        
        # Data normalization using LeRobot's built-in classes
        # Define input features for normalization
        self.input_features = {
            "observation.image": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 96, 96),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(2,),
            ),
        }
        
        # Define output features for normalization
        self.output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(action_dim,),
            ),
        }
        
        # Create normalizers (will be initialized with dataset stats later)
        self.normalize_inputs = None
        self.normalize_targets = None
        self.unnormalize_outputs = None
        self._normalizers_initialized = False
        
        # Feature extraction
        self.resnet_extractor = ResNetFeatureExtractor(resnet_type, d_model, rank, alpha, dropout)
        self.state_embedding = StateEmbedding(d_model, rank, alpha, dropout)  # Changed from list embeddings
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, rank, alpha, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection for action prediction
        self.output_projection = LoRALinear(d_model, action_dim, rank, alpha, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def initialize_normalizers(self, train_loader):
        """Initialize normalizers with dataset statistics."""
        if self._normalizers_initialized:
            return
        
        print("Computing dataset statistics for normalization...")
        
        # Collect data for computing statistics
        all_images = []
        all_states = []
        all_actions = []
        
        for batch in train_loader:
            all_images.append(batch['images'])
            all_states.append(batch['states'])
            all_actions.append(batch['actions'])
        
        # Concatenate all batches
        all_images = torch.cat(all_images, dim=0)
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        
        # Compute statistics - avoid problematic min/max operations
        # Reshape images to [batch*height*width, channels] for easier computation
        batch_size, channels, height, width = all_images.shape
        images_flat = all_images.permute(0, 2, 3, 1).reshape(-1, channels)  # [batch*h*w, 3]
        
        img_mean = images_flat.mean(dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)  # [1, 3, 1, 1]
        img_std = images_flat.std(dim=0, keepdim=True).unsqueeze(-1).unsqueeze(-1)    # [1, 3, 1, 1]
        
        # Use tensor methods and extract values from tuples
        img_min_values, _ = images_flat.min(dim=0, keepdim=True)
        img_min = img_min_values.unsqueeze(-1).unsqueeze(-1)  # [1, 3, 1, 1]
        
        img_max_values, _ = images_flat.max(dim=0, keepdim=True)
        img_max = img_max_values.unsqueeze(-1).unsqueeze(-1)  # [1, 3, 1, 1]
        
        dataset_stats = {
            "observation.image": {
                "mean": img_mean,  # [1, 3, 1, 1]
                "std": img_std,    # [1, 3, 1, 1]
                "min": img_min,    # [1, 3, 1, 1]
                "max": img_max,    # [1, 3, 1, 1]
            },
            "observation.state": {
                "mean": all_states.mean(dim=0),  # [2]
                "std": all_states.std(dim=0),    # [2]
                "min": all_states.min(dim=0)[0],    # [2] - extract values from tuple
                "max": all_states.max(dim=0)[0],    # [2] - extract values from tuple
            },
            "action": {
                "mean": all_actions.mean(dim=0),  # [action_dim]
                "std": all_actions.std(dim=0),    # [action_dim]
                "min": all_actions.min(dim=0)[0],    # [action_dim] - extract values from tuple
                "max": all_actions.max(dim=0)[0],    # [action_dim] - extract values from tuple
            },
        }
        
        # Create normalizers with computed statistics
        self.normalize_inputs = Normalize(
            self.input_features, 
            self.normalization_mapping, 
            dataset_stats
        )
        
        self.normalize_targets = Normalize(
            self.output_features, 
            self.normalization_mapping, 
            dataset_stats
        )
        
        self.unnormalize_outputs = Unnormalize(
            self.output_features, 
            self.normalization_mapping, 
            dataset_stats
        )
        
        self._normalizers_initialized = True
        print("Normalizers initialized successfully!")
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,  # Changed from list1, list2
    ) -> torch.Tensor:
        """
        Forward pass of the model for action prediction.
        
        Args:
            images: [batch_size, 3, 96, 96] RGB images from LeRobot
            states: [batch_size, 2] robot joint state vectors
        
        Returns:
            actions: [batch_size, action_dim] predicted robot actions
        """
        batch_size = images.size(0)
        
        # Check if normalizers are initialized
        if not self._normalizers_initialized:
            raise RuntimeError("Normalizers not initialized. Call initialize_normalizers() first.")
        
        # Create batch dictionary for normalization
        batch = {
            "observation.image": images,
            "observation.state": states,
        }
        
        # Normalize inputs - ensure normalizers exist
        if self.normalize_inputs is not None:
            normalized_batch = self.normalize_inputs(batch)
            normalized_images = normalized_batch["observation.image"]
            normalized_states = normalized_batch["observation.state"]
        else:
            # Fallback: use raw inputs if normalizers not available
            normalized_images = images
            normalized_states = states
        
        # Extract ResNet features
        resnet_features = self.resnet_extractor(normalized_images)  # [batch_size, 9, 512]
        
        # Embed states
        state_features = self.state_embedding(normalized_states)  # [batch_size, 1, 512]
        
        # Concatenate all features
        combined_features = torch.cat([resnet_features, state_features], dim=1)
        # combined_features: [batch_size, 10, 512] (9 + 1)
        
        # Create a dummy sequence for the transformer (we only need the final output)
        # This is a bit of a hack, but allows us to reuse the transformer architecture
        dummy_sequence = torch.zeros(batch_size, 1, self.d_model, device=images.device)
        
        # Pass through transformer decoder layers
        x = dummy_sequence
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, combined_features)
        
        # Output projection for action prediction
        actions = self.output_projection(x.squeeze(1))  # [batch_size, action_dim]
        
        return actions


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def create_model(
    action_dim: int = 2,
    d_model: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    d_ff: int = 2048,
    resnet_type: str = 'resnet18',
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.05,
    normalization_mapping: Optional[dict[str, NormalizationMode]] = None
) -> ResNetTransformerDecoder:
    """
    Create a ResNet + Transformer Decoder model with LoRA adaptation for action prediction.
    
    Args:
        action_dim: Dimension of robot actions
        d_model: Hidden dimension of the transformer
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        resnet_type: Type of ResNet ('resnet18' or 'resnet50')
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout rate
    
    Returns:
        Configured model
    """
    model = ResNetTransformerDecoder(
        action_dim=action_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        resnet_type=resnet_type,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    return model


class LeRobotDatasetAdapter:
    """Adapter class to work with LeRobot dataset for training."""
    
    def __init__(self, dataset, batch_size=32, train_split=0.8):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Split dataset into train/val
        total_size = len(dataset)
        train_size = int(total_size * train_split)
        val_size = total_size - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        """Custom collate function for LeRobot dataset."""
        images = torch.stack([item['observation.image'] for item in batch])
        states = torch.stack([item['observation.state'] for item in batch])
        actions = torch.stack([item['action'] for item in batch])
        
        return {
            'images': images,
            'states': states,
            'actions': actions
        }


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4, device=None):
    """Training loop for the action prediction model."""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean squared error for action prediction
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['images'].to(device)
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            
            # Forward pass
            predicted_actions = model(images, states)
            loss = criterion(predicted_actions, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                
                predicted_actions = model(images, states)
                loss = criterion(predicted_actions, actions)
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_action_model.pth')
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 50)


# Example usage and testing
if __name__ == "__main__":
    # Load LeRobot dataset
    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(repo_id="lerobot/pusht")
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create data loaders
    data_adapter = LeRobotDatasetAdapter(dataset, batch_size=16)
    
    # Model configuration
    action_dim = 2  # LeRobot actions are 2D
    batch_size = 16
    seq_len = 20
    
    # Create model with min-max normalization for all modalities
    normalization_mapping = {
        "VISUAL": NormalizationMode.MIN_MAX,
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }
    
    model = create_model(
        action_dim=action_dim,
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=2048,
        resnet_type='resnet18',
        rank=16,
        alpha=32,
        dropout=0.05,
        normalization_mapping=normalization_mapping
    )
    
    # Initialize normalizers with dataset statistics
    model.initialize_normalizers(data_adapter.train_loader)
    
    # Print model info
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"Model Architecture:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters (LoRA only): {trainable_params:,}")
    print(f"  - Parameter efficiency: {trainable_params/total_params*100:.2f}%")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    sample_batch = next(iter(data_adapter.train_loader))
    images = sample_batch['images']
    states = sample_batch['states']
    actions = sample_batch['actions']
    
    print(f"  - Input images shape: {images.shape}")
    print(f"  - Input states shape: {states.shape}")
    print(f"  - Target actions shape: {actions.shape}")
    
    # Test model
    with torch.no_grad():
        predicted_actions = model(images, states)
        print(f"  - Predicted actions shape: {predicted_actions.shape}")
    
    # Train the model
    print(f"\nStarting training...")
    # Force CPU usage due to CUDA compatibility issues
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    train_model(
        model, 
        data_adapter.train_loader, 
        data_adapter.val_loader, 
        epochs=5,  # Start with fewer epochs for testing
        lr=1e-4,
        device=device
    )
    
    print(f"\nTraining completed!")
