import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple, Union
import math

class LoRALinear(nn.Module):
    """LoRA Linear layer for efficient fine-tuning"""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        # These are the weights from the original pretrained model
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        # These are the new LoRA params. In general rank << in_dim, out_dim
        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)

        # Rank and alpha are commonly-tuned hyperparameters
        self.rank = rank
        self.alpha = alpha

        # Most implementations also include some dropout
        self.dropout = nn.Dropout(p=dropout)

        # The original params are frozen, and only LoRA params are trainable.
        self.linear.weight.requires_grad = False
        self.lora_a.weight.requires_grad = True
        self.lora_b.weight.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This would be the output of the original model
        frozen_out = self.linear(x)

        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        lora_out = self.lora_b(self.lora_a(self.dropout(x)))

        # Finally, scale by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return frozen_out + (self.alpha / self.rank) * lora_out

class TextEncoder(nn.Module):
    """Text encoder using a pre-trained transformer model"""
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Freeze the transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Get the hidden size
        self.hidden_size = self.transformer.config.hidden_size
        
    def forward(self, text_inputs: Union[str, list]) -> torch.Tensor:
        """
        Encode text inputs to embeddings
        
        Args:
            text_inputs: String or list of strings
            
        Returns:
            Text embeddings of shape (batch_size, hidden_size)
        """
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]
            
        # Tokenize and encode
        encoded = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.transformer(**encoded)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings

class MultimodalFusion(nn.Module):
    """Fusion module to combine image and text features"""
    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        
        # Project image features to fusion dimension
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        
        # Project text features to fusion dimension
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Attention mechanism for cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse image and text features
        
        Args:
            image_features: Image features (batch_size, image_dim)
            text_features: Text features (batch_size, text_dim)
            
        Returns:
            Fused features (batch_size, fusion_dim)
        """
        # Project to common space
        image_proj = self.image_proj(image_features)  # (batch_size, fusion_dim)
        text_proj = self.text_proj(text_features)     # (batch_size, fusion_dim)
        
        # Reshape for attention (batch_size, 1, fusion_dim)
        image_proj = image_proj.unsqueeze(1)
        text_proj = text_proj.unsqueeze(1)
        
        # Cross-attention between image and text
        fused_features, _ = self.cross_attention(
            query=image_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Concatenate and fuse
        combined = torch.cat([image_proj, fused_features], dim=-1)  # (batch_size, 1, fusion_dim*2)
        combined = combined.squeeze(1)  # (batch_size, fusion_dim*2)
        
        # Final fusion
        fused = self.fusion_mlp(combined)
        fused = self.layer_norm(fused)
        
        return fused

class MultimodalResNet18LoRA(nn.Module):
    """Multimodal ResNet18 with LoRA fine-tuning"""
    
    def __init__(
        self,
        num_classes: int = 1000,
        text_model_name: str = "bert-base-uncased",
        lora_rank: int = 16,
        lora_alpha: float = 1.0,
        fusion_dim: int = 512,
        use_lora: bool = True
    ):
        super().__init__()
        
        # Load pre-trained ResNet18
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze the original ResNet parameters
        for param in self.resnet18.parameters():
            param.requires_grad = False
            
        # Get the feature dimension from ResNet
        self.image_feature_dim = self.resnet18.fc.in_features
        
        # Remove the final classification layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        
        # Text encoder
        self.text_encoder = TextEncoder(text_model_name)
        
        # Multimodal fusion
        self.fusion = MultimodalFusion(
            image_dim=self.image_feature_dim,
            text_dim=self.text_encoder.hidden_size,
            fusion_dim=fusion_dim
        )
        
        # Final classification layer with LoRA
        if use_lora:
            self.classifier = LoRALinear(
                in_dim=fusion_dim,
                out_dim=num_classes,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=0.1
            )
        else:
            self.classifier = nn.Linear(fusion_dim, num_classes)
            
        # Image transforms for ResNet
        self.transforms = ResNet18_Weights.DEFAULT.transforms(antialias=True)
        
    def forward(
        self, 
        images: torch.Tensor, 
        text_inputs: Union[str, list]
    ) -> torch.Tensor:
        """
        Forward pass with image and text inputs
        
        Args:
            images: Image tensor (batch_size, channels, height, width)
            text_inputs: Text string or list of strings
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Process images through ResNet
        if self.training:
            # Apply transforms during training
            images = self.transforms(images)
        
        # Extract image features
        image_features = self.resnet18(images)  # (batch_size, image_feature_dim, 1, 1)
        image_features = image_features.squeeze(-1).squeeze(-1)  # (batch_size, image_feature_dim)
        
        # Encode text
        text_features = self.text_encoder(text_inputs)  # (batch_size, text_hidden_size)
        
        # Fuse modalities
        fused_features = self.fusion(image_features, text_features)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters (LoRA + fusion + classifier)"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")
        
        return total_params, trainable_params

class MultimodalDataset(torch.utils.data.Dataset):
    """Simple dataset for multimodal data"""
    def __init__(self, image_paths, text_descriptions, labels, transform=None):
        self.image_paths = image_paths
        self.text_descriptions = text_descriptions
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = torch.load(self.image_paths[idx]) if self.image_paths[idx].endswith('.pt') else None
        
        # For now, return placeholder data
        # In practice, you'd load actual images and apply transforms
        text = self.text_descriptions[idx]
        label = self.labels[idx]
        
        return image, text, label

def create_multimodal_model(
    num_classes: int = 1000,
    text_model: str = "bert-base-uncased",
    lora_rank: int = 16,
    lora_alpha: float = 1.0,
    fusion_dim: int = 512
) -> MultimodalResNet18LoRA:
    """
    Create a multimodal ResNet18 model with LoRA
    
    Args:
        num_classes: Number of output classes
        text_model: Pre-trained text model name
        lora_rank: LoRA rank for efficient fine-tuning
        lora_alpha: LoRA alpha parameter
        fusion_dim: Dimension for fused features
        
    Returns:
        Configured multimodal model
    """
    model = MultimodalResNet18LoRA(
        num_classes=num_classes,
        text_model_name=text_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        fusion_dim=fusion_dim,
        use_lora=True
    )
    
    return model

# Example usage and training setup
if __name__ == "__main__":
    # Create model
    model = create_multimodal_model(
        num_classes=10,
        lora_rank=16,
        lora_alpha=1.0,
        fusion_dim=512
    )
    
    # Count parameters
    model.count_parameters()
    
    # Example forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    texts = ["A cute dog playing in the park", "A cat sitting on a chair"]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, texts)
        print(f"Output shape: {outputs.shape}")
        print(f"Output logits: {outputs}")
    
    # Get trainable parameters for optimizer
    trainable_params = model.get_trainable_parameters()
    print(f"Number of trainable parameter groups: {len(trainable_params)}")
    
    # Example optimizer setup
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    print("Model and optimizer created successfully!")
