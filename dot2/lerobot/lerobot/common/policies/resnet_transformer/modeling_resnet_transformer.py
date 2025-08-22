import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
from torch import Tensor
from typing import Type

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
        
        # LoRA adapters with better initialization
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Better weight initialization for LoRA
        # Initialize A with small random values for better gradient flow
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        # Initialize B with zeros to start with no change
        nn.init.zeros_(self.lora_B.weight)
        
        # Training step counter for warmup
        self.training_step = 0
        self.warmup_steps = 1000  # Warmup over first 1000 steps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply warmup scaling for better initial training stability
        if self.training:
            warmup_factor = min(1.0, self.training_step / self.warmup_steps)
            current_scaling = self.scaling * warmup_factor
        else:
            current_scaling = self.scaling
            
        return self.dropout(self.lora_B(self.lora_A(x))) * current_scaling
    
    def update_training_step(self, step: int):
        """Update the training step for warmup scaling."""
        self.training_step = step


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        # Initialize base layer with better weights
        nn.init.xavier_uniform_(self.base_layer.weight)
        if self.base_layer.bias is not None:
            nn.init.zeros_(self.base_layer.bias)
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


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
        
        # Better initialization for transformer blocks
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
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
    
    name: str = "resnet_transformer"
    config_class: Type[ResNetTransformerConfig] = ResNetTransformerConfig
    
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
        
        # 3. ACTION EMBEDDING (2 -> 512)
        self.action_embedding = LoRALinear(2, self.d_model, config.lora_rank, config.lora_alpha, config.dropout)
        
        # 4. POSITIONAL ENCODINGS - Updated to use new functions
        if config.use_positional_encoding:
            if config.pos_encoding_type == 'learnable':
                # Learnable positional encodings with better initialization
                self.obs_pos_encoding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
                self.image_pos_encoding = nn.Parameter(torch.randn(1, 9, self.d_model) * 0.02)
                self.action_pos_encoding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
                self.pos_encoding = None
            else:  # sinusoidal
                # Use the new create_sinusoidal_pos_embedding function
                self.pos_encoding = create_sinusoidal_pos_embedding(11, self.d_model)  # 11 total positions (1+9+1)
                # Initialize the individual positional encodings to None for sinusoidal case
                self.obs_pos_encoding = None
                self.image_pos_encoding = None
                self.action_pos_encoding = None
        else:
            self.obs_pos_encoding = None
            self.image_pos_encoding = None
            self.action_pos_encoding = None
            self.pos_encoding = None
        
        # 5. TRANSFORMER PROCESSING
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, config.num_heads, config.d_ff, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # 6. OUTPUT PROJECTION
        self.output_projection = LoRALinear(self.d_model, config.action_dim, config.lora_rank, config.lora_alpha, config.dropout)
        
        # Initialize normalizers if dataset_stats provided
        if dataset_stats is not None:
            self._initialize_normalizers(dataset_stats)
    
    def _initialize_normalizers(self, dataset_stats: dict[str, dict[str, torch.Tensor]]):
        """Initialize normalizers with provided dataset statistics."""
        # Normalize inputs (observations and actions) for training
        self.normalize_inputs = Normalize(
            self.config.input_features, 
            self.config.normalization_mapping, 
            dataset_stats
        )
        
        # Unnormalize outputs (actions) for inference
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
        
        # Normalize all inputs (including actions) if normalizers are available
        if hasattr(self, 'normalize_inputs'):
            normalized_batch = self.normalize_inputs(batch)
            images = normalized_batch["observation.image"]
            states = normalized_batch["observation.state"]
            actions = normalized_batch["action"]
        
        # Forward pass during training
        predicted_actions = self._forward_internal(images, states, actions)
        
        # Compute loss - ensure actions have correct shape
        if actions.dim() == 3:
            actions = actions.squeeze(1)  # [batch, 1, 2] -> [batch, 2]
        
        # Use L1 loss for better stability in early training
        loss = F.l1_loss(predicted_actions, actions)
        
        return {"loss": loss}
    
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return one action to run in the environment (potentially in batch mode)."""
        # Extract inputs from batch (used during inference/prediction)
        images = batch["observation.image"]
        states = batch["observation.state"]
        
        # Normalize inputs (observations) if normalizers are available
        if hasattr(self, 'normalize_inputs'):
            normalized_batch = self.normalize_inputs(batch)
            images = normalized_batch["observation.image"]
            states = normalized_batch["observation.state"]
        
        # Forward pass (actions not needed for inference)
        predicted_actions = self._forward_internal(images, states, None)
        
        # Unnormalize outputs (actions) if normalizers are available
        if hasattr(self, 'unnormalize_outputs'):
            # Create a batch with predicted actions for unnormalization
            action_batch = {"action": predicted_actions.unsqueeze(1)}  # Add sequence dimension
            unnormalized_batch = self.unnormalize_outputs(action_batch)
            predicted_actions = unnormalized_batch["action"].squeeze(1)  # Remove sequence dimension
        
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
        
        # Add positional encoding to observation features if enabled
        if self.config.use_positional_encoding and self.obs_pos_encoding is not None:
            observation_features = observation_features + self.obs_pos_encoding  # [batch, 1, 512]
        
        # 2. PROCESS IMAGES: [batch, 3, 96, 96] -> [batch, 9, 512]
        resnet_features = self.resnet(images)  # [batch, 512, H, W]
        
        # Force spatial dimensions to 3x3 (9 positions)
        resnet_features = F.adaptive_avg_pool2d(resnet_features, (3, 3))  # [batch, 512, 3, 3]
        
        # Reshape to [batch, 9, 512]
        resnet_features = resnet_features.permute(0, 2, 3, 1)  # [batch, 3, 3, 512]
        resnet_features = resnet_features.reshape(batch_size, 9, self.feature_dim)  # [batch, 9, 512]
        
        # Linear projection with LoRA
        image_features = self.image_projection(resnet_features)  # [batch, 9, 512]
        
        # Add positional encoding to image features if enabled
        if self.config.use_positional_encoding and self.image_pos_encoding is not None:
            image_features = image_features + self.image_pos_encoding  # [batch, 9, 512]
        
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
        
        # Add positional encoding to action features if enabled
        if self.config.use_positional_encoding and self.action_pos_encoding is not None:
            action_features = action_features + self.action_pos_encoding  # [batch, 1, 512]
        
        # 4. CONCATENATE ALL FEATURES: [batch, 11, 512] (1+9+1)
        # observation_features: [batch, 1, 512] (1 feature with pos encoding)
        # image_features: [batch, 9, 512] (9 features with pos encoding) 
        # action_features: [batch, 1, 512] (1 feature with pos encoding)
        combined_features = torch.cat([observation_features, image_features, action_features], dim=1)  # [batch, 11, 512]
        
        # 5. ADD SINUSOIDAL POSITIONAL ENCODING IF ENABLED - Updated to use new function
        if self.config.use_positional_encoding and self.pos_encoding is not None:
            # Apply sinusoidal encoding to the entire sequence using the new function
            # self.pos_encoding is now a tensor of shape [11, 512] from create_sinusoidal_pos_embedding
            combined_features = combined_features + self.pos_encoding.unsqueeze(0).to(combined_features.device)  # [batch, 11, 512]
        
        # 6. PROCESS THROUGH TRANSFORMER
        x = combined_features
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 7. OUTPUT PROJECTION (take mean across sequence dimension)
        x = x.mean(dim=1)  # [batch, 512]
        predicted_actions = self.output_projection(x)  # [batch, action_dim]
        
        return predicted_actions
    
    def get_optim_params(self) -> dict:
        """Returns the policy-specific parameters dict to be passed on to the optimizer."""
        return {"lr": self.config.optimizer_lr, "weight_decay": self.config.optimizer_weight_decay}
    
    def get_grad_clip_norm(self) -> float:
        """Returns the gradient clipping norm for training stability."""
        return 1.0  # Clip gradients to norm 1.0 for stability
    
    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Returns only the trainable LoRA parameters for efficient training."""
        trainable_params = []
        for name, param in self.named_parameters():
            if 'lora' in name and param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def update_training_step(self, step: int):
        """Update training step for all LoRA layers to enable warmup scaling."""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.update_training_step(step)
    
    def get_lora_scaling_info(self) -> dict:
        """Get information about LoRA scaling for monitoring."""
        scaling_info = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                warmup_factor = min(1.0, module.training_step / module.warmup_steps) if module.training else 1.0
                current_scaling = module.scaling * warmup_factor
                scaling_info[f"{name}.scaling"] = current_scaling
                scaling_info[f"{name}.warmup_factor"] = warmup_factor
        return scaling_info
    
    def reset(self):
        """To be called whenever the environment is reset. Does things like clearing caches."""
        pass