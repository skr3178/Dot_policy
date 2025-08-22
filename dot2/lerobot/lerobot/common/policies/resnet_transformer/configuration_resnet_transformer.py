from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineAnnealingSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode, PolicyFeature, FeatureType


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
    
    # Positional encoding settings
    use_positional_encoding: bool = True
    pos_encoding_type: str = 'sinusoidal'  # 'learnable' or 'sinusoidal'
    
    # Training parameters - Better defaults for stability
    optimizer_lr: float = 5.0e-5  # Reduced from 1e-4 for better stability
    optimizer_min_lr: float = 1.0e-6  # Lower minimum LR
    optimizer_lr_cycle_steps: int = 300000
    optimizer_weight_decay: float = 1e-4  # Increased from 1e-5 for better regularization
    
    # Observation and action steps
    n_obs_steps: int = 1
    
    # Normalization - Better defaults for stability
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    
    # Input and output features - These should be set explicitly by the user
    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            # Note: action is typically added explicitly for behavior cloning policies
        }
    )
    
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
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
    
    def validate_features(self) -> None:
        """Validate that the configuration has the required features."""
        # Check that we have the required input features
        if not self.input_features:
            raise ValueError("input_features must be set for ResNet transformer policy")
        
        # Check that we have the required output features
        if not self.output_features:
            raise ValueError("output_features must be set for ResNet transformer policy")
        
        # Check that we have image and state features
        has_image = any(feat.type == FeatureType.VISUAL for feat in self.input_features.values())
        has_state = any(feat.type == FeatureType.STATE for feat in self.input_features.values())
        has_action_output = any(feat.type == FeatureType.ACTION for feat in self.output_features.values())
        
        if not has_image:
            raise ValueError("ResNet transformer policy requires image input features")
        if not has_state:
            raise ValueError("ResNet transformer policy requires state input features")
        if not has_action_output:
            raise ValueError("ResNet transformer policy requires action output features")
        
        # For behavior cloning, actions should also be in input_features
        has_action_input = any(feat.type == FeatureType.ACTION for feat in self.input_features.values())
        if not has_action_input:
            print("Warning: Actions not found in input_features. This policy is designed for behavior cloning.")
    
    def print_features(self) -> None:
        """Print the current input and output features for debugging."""
        print("ResNet Transformer Configuration:")
        print(f"Input features: {list(self.input_features.keys())}")
        print(f"Output features: {list(self.output_features.keys())}")
        print(f"Normalization mapping: {self.normalization_mapping}")
    
    @property
    def observation_delta_indices(self) -> list | None:
        """Return the observation delta indices for this policy."""
        return list(range(1 - self.n_obs_steps, 1))
    
    @property
    def action_delta_indices(self) -> list | None:
        """Return the action delta indices for this policy."""
        return [0]  # Only predict current action
    
    @property
    def reward_delta_indices(self) -> list | None:
        """Return the reward delta indices for this policy."""
        return None