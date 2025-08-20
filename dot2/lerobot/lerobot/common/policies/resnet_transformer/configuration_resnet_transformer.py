from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.common.optim.schedulers import CosineAnnealingSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


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
    
    def validate_features(self) -> None:
        """Validate that the configuration has the required features."""
        # This model always has image and state features
        pass
    
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

