from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch

dataset = LeRobotDataset(repo_id="lerobot/pusht")
    
print("Dataset length:", len(dataset))
print("\nFirst sample:")
print(dataset[0])

print("\nSecond sample:")
print(dataset[1])

print("\nSample keys:")
sample = dataset[0]
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: {value.shape} - {value.dtype}")
    else:
        print(f"{key}: {type(value)} - {value}")

print("\nImage shape:", sample['observation.image'].shape)
print("State shape:", sample['observation.state'].shape)
print("Action shape:", sample['action'].shape)