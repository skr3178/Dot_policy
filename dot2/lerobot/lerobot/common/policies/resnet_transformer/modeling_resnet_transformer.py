import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.types import FeatureType, PolicyFeature
from .configuration_resnet_transformer import ResNetTransformerConfig


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
        nn.init.kaiming_uniform_(self.lora_A.weight, a=1.0)  # Use default value
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


class TransformerBlock(nn.Module):
    """Simple transformer block for processing concatenated features."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.05):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class ResNetTransformerPolicy(PreTrainedPolicy):
    """Main model: ResNet + Transformer with LoRA adaptation for action prediction."""
    
    name = "resnet_transformer"
    config_class = ResNetTransformerConfig
    
    def __init__(
        self,
        config,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        
        # 1. OBSERVATION EMBEDDING (2 -> 512)
        self.observation_embedding = LoRALinear(2, self.d_model, config.lora_rank, config.lora_alpha, config.dropout)
        
        # 2. IMAGE PROCESSING (ResNet + Linear Projection + Positional Encoding)
        if config.resnet_type == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet type: {config.resnet_type}")
        
        # Remove classification head and keep spatial features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Keep spatial dimensions
        
        # Freeze ResNet
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Linear projection (512 -> 512) with LoRA
        self.image_projection = LoRALinear(self.feature_dim, self.d_model, config.lora_rank, config.lora_alpha, config.dropout)
        
        # Positional encoding for spatial features
        self.pos_encoding = nn.Parameter(torch.randn(1, 9, self.d_model))  # 3x3 spatial grid = 9 positions
        
        # 3. ACTION EMBEDDING (2 -> 512)
        self.action_embedding = LoRALinear(2, self.d_model, config.lora_rank, config.lora_alpha, config.dropout)
        
        # 4. TRANSFORMER PROCESSING
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # 5. OUTPUT PROJECTION
        self.output_projection = LoRALinear(self.d_model, config.action_dim, config.lora_rank, config.lora_alpha, config.dropout)
        
        # Initialize normalizers if dataset_stats provided
        if dataset_stats is not None:
            self._initialize_normalizers(dataset_stats)
    
    def _initialize_normalizers(self, dataset_stats: dict[str, dict[str, torch.Tensor]]):
        """Initialize normalizers with provided dataset statistics."""
        self.normalize_inputs = Normalize(
            self.config.input_features, 
            self.config.normalization_mapping, 
            dataset_stats
        )
        
        self.normalize_targets = Normalize(
            self.config.output_features, 
            self.config.normalization_mapping, 
            dataset_stats
        )
        
        self.unnormalize_outputs = Unnormalize(
            self.config.output_features, 
            self.config.normalization_mapping, 
            dataset_stats
        )
    
    def forward(self, batch: dict[str, torch.Tensor]) -> dict:
        """Run the batch through the model and compute the loss for training or validation."""
        # Extract inputs from batch
        images = batch["observation.image"]
        states = batch["observation.state"]
        actions = batch["action"]
        
        # Forward pass
        predicted_actions = self._forward_internal(images, states, actions)
        
        # Compute loss - ensure actions have correct shape
        if actions.dim() == 3:
            actions = actions.squeeze(1)  # [batch, 1, 2] -> [batch, 2]
        loss = F.mse_loss(predicted_actions, actions)
        
        return {"loss": loss}
    
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one action to run in the environment (potentially in batch mode)."""
        # Extract inputs from batch
        images = batch["observation.image"]
        states = batch["observation.state"]
        
        # Forward pass (actions not needed for inference)
        predicted_actions = self._forward_internal(images, states, None)
        
        return predicted_actions
    
    def _forward_internal(self, images: torch.Tensor, states: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Internal forward pass for action prediction."""
        batch_size = images.size(0)
        
        # 1. PROCESS OBSERVATIONS (states): [batch, 1, 2] -> [batch, 1, 512]
        # Remove sequence dimension if present
        if states.dim() == 3:
            states = states.squeeze(1)  # [batch, 2]
        observation_features = self.observation_embedding(states)  # [batch, 512]
        observation_features = observation_features.unsqueeze(1)  # [batch, 1, 512]
        
        # 2. PROCESS IMAGES: [batch, 3, 96, 96] -> [batch, 9, 512]
        resnet_features = self.resnet(images)  # [batch, 512, H, W]
        
        # Force spatial dimensions to 3x3 (9 positions)
        resnet_features = F.adaptive_avg_pool2d(resnet_features, (3, 3))  # [batch, 512, 3, 3]
        
        # Reshape to [batch, 9, 512]
        resnet_features = resnet_features.permute(0, 2, 3, 1)  # [batch, 3, 3, 512]
        resnet_features = resnet_features.reshape(batch_size, 9, self.feature_dim)  # [batch, 9, 512]
        
        # Linear projection with LoRA
        image_features = self.image_projection(resnet_features)  # [batch, 9, 512]
        
        # Add positional encoding
        image_features = image_features + self.pos_encoding  # [batch, 9, 512]
        
        # 3. PROCESS ACTIONS (if provided for training)
        if actions is not None:
            # Remove sequence dimension if present
            if actions.dim() == 3:
                actions = actions.squeeze(1)  # [batch, 2]
            action_features = self.action_embedding(actions)  # [batch, 512]
            action_features = action_features.unsqueeze(1)  # [batch, 1, 512]
        else:
            # For inference, create dummy action features
            action_features = torch.zeros(batch_size, 1, self.d_model, device=images.device)
        
        # 4. CONCATENATE ALL FEATURES: [batch, 11, 512] (1+9+1)
        # observation_features: [batch, 1, 512] (1 feature)
        # image_features: [batch, 9, 512] (9 features) 
        # action_features: [batch, 1, 512] (1 feature)
        combined_features = torch.cat([observation_features, image_features, action_features], dim=1)  # [batch, 11, 512]
        
        # 5. PROCESS THROUGH TRANSFORMER
        x = combined_features
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 6. OUTPUT PROJECTION (take mean across sequence dimension)
        x = x.mean(dim=1)  # [batch, 512]
        predicted_actions = self.output_projection(x)  # [batch, action_dim]
        
        return predicted_actions
    
    def get_optim_params(self) -> dict:
        """Returns the policy-specific parameters dict to be passed on to the optimizer."""
        return {"lr": self.config.optimizer_lr, "weight_decay": self.config.optimizer_weight_decay}
    
    def reset(self):
        """To be called whenever the environment is reset. Does things like clearing caches."""
        pass
