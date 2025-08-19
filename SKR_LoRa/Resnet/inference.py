#!/usr/bin/env python3
"""
Inference script for Multimodal ResNet18 with LoRA
"""

import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import transforms
import argparse
from pathlib import Path
import json

from multimodal_resnet_lora import create_multimodal_model

def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    # Create model with same configuration
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    model = create_multimodal_model(
        num_classes=args['num_classes'],
        lora_rank=args['lora_rank'],
        lora_alpha=args['lora_alpha'],
        fusion_dim=args['fusion_dim']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint['best_acc']:.2f}%")
    
    return model, args

def preprocess_image(image_path, transform=None):
    """Preprocess image for inference"""
    try:
        image = read_image(str(image_path))
        # Convert to float and normalize to [0, 1]
        image = image.float() / 255.0
        
        if transform:
            image = transform(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def predict(model, image, text, device, class_names=None):
    """Make prediction with the model"""
    with torch.no_grad():
        # Move to device
        image = image.to(device)
        
        # Forward pass
        outputs = model(image, [text])
        
        # Get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # Get predicted class
        _, predicted = outputs.max(1)
        
        # Get confidence
        confidence = probs[0][predicted[0]].item()
        
        # Get top-k predictions
        top_k = 3
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        results = {
            'predicted_class': predicted[0].item(),
            'confidence': confidence,
            'top_predictions': []
        }
        
        for i in range(top_k):
            class_idx = top_indices[i].item()
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            prob = top_probs[i].item()
            
            results['top_predictions'].append({
                'class': class_idx,
                'class_name': class_name,
                'probability': prob
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Inference with Multimodal ResNet18 + LoRA')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--text', type=str, default='', help='Text description (optional)')
    parser.add_argument('--class_names', type=str, help='Path to JSON file with class names')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load class names if provided
    class_names = None
    if args.class_names:
        try:
            with open(args.class_names, 'r') as f:
                class_names = json.load(f)
            print(f"Loaded {len(class_names)} class names")
        except Exception as e:
            print(f"Error loading class names: {e}")
    
    # Load model
    print("Loading model...")
    model, model_args = load_model(args.checkpoint, device)
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    image = preprocess_image(args.image)
    
    if image is None:
        print("Failed to load image. Exiting.")
        return
    
    # Prepare text input
    if not args.text:
        # Extract filename as default text
        image_name = Path(args.image).stem
        args.text = f"An image of {image_name}"
    
    print(f"Text input: {args.text}")
    
    # Make prediction
    print("Making prediction...")
    results = predict(model, image, args.text, device, class_names)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted class: {results['predicted_class']}")
    if class_names:
        class_name = class_names[results['predicted_class']]
        print(f"Class name: {class_name}")
    print(f"Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
    
    print(f"\nTop {len(results['top_predictions'])} predictions:")
    for i, pred in enumerate(results['top_predictions']):
        print(f"{i+1}. Class {pred['class']}: {pred['class_name']} ({pred['probability']:.4f})")

def interactive_mode(checkpoint_path, device):
    """Interactive mode for multiple predictions"""
    print("Loading model for interactive mode...")
    model, model_args = load_model(checkpoint_path, device)
    
    print("\nInteractive mode started. Enter 'quit' to exit.")
    print("Format: <image_path> | <text_description>")
    
    while True:
        try:
            user_input = input("\nEnter image path and text (separated by |): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if '|' not in user_input:
                print("Please use format: <image_path> | <text_description>")
                continue
            
            image_path, text = user_input.split('|', 1)
            image_path = image_path.strip()
            text = text.strip()
            
            if not image_path or not text:
                print("Both image path and text are required")
                continue
            
            # Check if image exists
            if not Path(image_path).exists():
                print(f"Image not found: {image_path}")
                continue
            
            # Process image
            image = preprocess_image(image_path)
            if image is None:
                continue
            
            # Make prediction
            results = predict(model, image, text, device)
            
            # Display results
            print(f"\nPrediction: Class {results['predicted_class']} (Confidence: {results['confidence']:.4f})")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

