#!/usr/bin/env python

import sys
import os
sys.path.append('dot2/lerobot')

from lerobot.configs.train import TrainPipelineConfig

def test_defaults():
    """Test that the default values are set correctly."""
    cfg = TrainPipelineConfig()
    
    print("Testing default configuration values:")
    print(f"Policy type: {cfg.policy.type if cfg.policy else 'None'}")
    print(f"Dataset repo_id: {cfg.dataset.repo_id}")
    print(f"Env type: {cfg.env.type if cfg.env else 'None'}")
    print(f"Batch size: {cfg.batch_size}")
    
    # Test that the values match what we expect
    expected_values = {
        'policy_type': 'dot',
        'dataset_repo_id': 'lerobot/dot_pusht_keypoints',
        'env_type': 'pusht',
        'batch_size': 24
    }
    
    actual_values = {
        'policy_type': cfg.policy.type if cfg.policy else None,
        'dataset_repo_id': cfg.dataset.repo_id,
        'env_type': cfg.env.type if cfg.env else None,
        'batch_size': cfg.batch_size
    }
    
    print("\nExpected vs Actual:")
    for key, expected in expected_values.items():
        actual = actual_values[key]
        status = "✓" if actual == expected else "✗"
        print(f"{status} {key}: expected={expected}, actual={actual}")
    
    # Check if all values match
    all_match = all(actual_values[key] == expected_values[key] for key in expected_values)
    print(f"\nAll defaults correct: {'✓' if all_match else '✗'}")
    
    return all_match

if __name__ == "__main__":
    test_defaults()
