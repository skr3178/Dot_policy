from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="lerobot/pusht")
    
print(dataset[0])