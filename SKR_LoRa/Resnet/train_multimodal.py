#!/usr/bin/env python3
"""
Training script for Multimodal ResNet18 with LoRA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
import os
import json
from pathlib import Path
from tqdm import tqdm
import argparse

from multimodal_resnet_lora import (
    MultimodalResNet18LoRA, 
    create_multimodal_model,
    MultimodalDataset
)

class SimpleMultimodalDataset(torch.utils.data.Dataset):
    """Simple dataset for demonstration purposes"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Create some dummy data for demonstration
        # In practice, you'd load your actual dataset
        self.samples = [
            {
                'image_path': 'dog1.jpg',
                'text': 'A cute dog playing in the park',
                'label': 0
            },
            {
                'image_path': 'dog2.jpg', 
                'text': 'A dog sitting on the grass',
                'label': 0
            }
        ]
        
        # Add more samples if images exist
        if (self.data_dir / 'dog1.jpg').exists():
            self.samples.append({
                'image_path': 'dog1.jpg',
                'text': 'A brown dog looking at the camera',
                'label': 0
            })
        
        if (self.data_dir / 'dog2.jpg').exists():
            self.samples.append({
                'image_path': 'dog2.jpg',
                'text': 'A dog in outdoor environment',
                'label': 0
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_dir / sample['image_path']
        if image_path.exists():
            try:
                image = read_image(str(image_path))
                # Convert to float and normalize to [0, 1]
                image = image.float() / 255.0
            except:
                # Create dummy image if loading fails
                image = torch.randn(3, 224, 224)
        else:
            # Create dummy image if file doesn't exist
            image = torch.randn(3, 224, 224)
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        text = sample['text']
        label = sample['label']
        
        return image, text, label

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, texts, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texts, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Train Multimodal ResNet18 with LoRA')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing images')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA alpha')
    parser.add_argument('--fusion_dim', type=int, default=512, help='Fusion dimension')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create model
    print("Creating multimodal model...")
    model = create_multimodal_model(
        num_classes=args.num_classes,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        fusion_dim=args.fusion_dim
    )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    print("\nModel parameters:")
    model.count_parameters()
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = SimpleMultimodalDataset(args.data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    trainable_params = model.get_trainable_parameters()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved with validation accuracy: {best_acc:.2f}%")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_acc': best_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved in: {args.save_dir}")

if __name__ == "__main__":
    main()
