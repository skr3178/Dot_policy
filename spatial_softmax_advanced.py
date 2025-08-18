import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.special import softmax

def spatial_softmax_4d(features, temperature=1.0):
    """
    Apply spatial softmax to 4D features tensor [N, H, W, C].
    
    This implementation follows the TensorFlow approach:
    1. Transpose to [N, C, H, W]
    2. Reshape to [N*C, H*W] for joint softmax over spatial dimensions
    3. Apply softmax
    4. Reshape and transpose back to [N, H, W, C]
    
    Args:
        features: 4D numpy array of shape [N, H, W, C]
        temperature: Temperature parameter for softmax
    
    Returns:
        4D array with same shape as input, where each [H, W] slice sums to 1.0
    """
    N, H, W, C = features.shape
    
    # Transpose to [N, C, H, W]
    features_t = np.transpose(features, [0, 3, 1, 2])
    
    # Reshape to [N*C, H*W] for joint softmax over spatial dimensions
    features_reshaped = features_t.reshape(N * C, H * W)
    
    # Apply softmax over spatial dimensions
    softmaxed = softmax(features_reshaped / temperature, axis=1)
    
    # Reshape back to [N, C, H, W]
    softmaxed_reshaped = softmaxed.reshape(N, C, H, W)
    
    # Transpose back to [N, H, W, C]
    softmaxed_final = np.transpose(softmaxed_reshaped, [0, 2, 3, 1])
    
    return softmaxed_final

def create_image_coordinates(H, W):
    """
    Create image coordinate grid.
    
    Args:
        H, W: Height and width of the image
    
    Returns:
        Array of shape [H, W, 2] where each pixel contains [x, y] coordinates
    """
    x_coords = np.arange(W, dtype=np.float32)
    y_coords = np.arange(H, dtype=np.float32)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Stack coordinates
    coords = np.stack([X, Y], axis=-1)
    
    return coords

def spatial_soft_argmax(softmax_weights, image_coords):
    """
    Compute spatial soft argmax using softmax weights.
    
    This computes the mean pixel location for each channel using the softmax weights
    as attention weights over the spatial dimensions.
    
    Args:
        softmax_weights: Softmax weights of shape [N, H, W, C]
        image_coords: Image coordinates of shape [H, W, 2]
    
    Returns:
        Array of shape [N, C, 2] containing mean pixel locations for each channel
    """
    N, H, W, C = softmax_weights.shape
    
    # Expand softmax to [N, H, W, C, 1] for broadcasting
    softmax_expanded = np.expand_dims(softmax_weights, axis=-1)
    
    # Expand image coords to [H, W, 1, 2] for broadcasting
    coords_expanded = np.expand_dims(image_coords, axis=2)
    
    # Multiply softmax weights with coordinates and reduce over spatial dimensions
    # Result: [N, C, 2]
    spatial_argmax = np.sum(softmax_expanded * coords_expanded, axis=(1, 2))
    
    return spatial_argmax

def demonstrate_spatial_softmax_4d():
    """
    Demonstrate 4D spatial softmax with a simple example.
    """
    print("4D Spatial Softmax Function Demonstration")
    print("=" * 50)
    
    # Create a simple 4D feature tensor [N=2, H=3, W=3, C=2]
    features = np.array([
        # Batch 0
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0], [17.0, 18.0]]
        ],
        # Batch 1
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
            [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]
        ]
    ])
    
    print("Original 4D features shape:", features.shape)
    print("Features[0, :, :, 0]:")
    print(features[0, :, :, 0])
    print("Features[0, :, :, 1]:")
    print(features[0, :, :, 1])
    
    # Apply spatial softmax
    softmaxed = spatial_softmax_4d(features, temperature=1.0)
    
    print("\nSpatial softmax result shape:", softmaxed.shape)
    print("Softmaxed[0, :, :, 0]:")
    print(softmaxed[0, :, :, 0])
    print("Sum of softmaxed[0, :, :, 0]:", softmaxed[0, :, :, 0].sum())
    
    print("Softmaxed[0, :, :, 1]:")
    print(softmaxed[0, :, :, 1])
    print("Sum of softmaxed[0, :, :, 1]:", softmaxed[0, :, :, 1].sum())
    
    return features, softmaxed

def demonstrate_spatial_soft_argmax():
    """
    Demonstrate spatial soft argmax computation.
    """
    print("\n" + "="*50)
    print("Spatial Soft Argmax Demonstration")
    print("="*50)
    
    # Create image coordinates for 3x3 image
    H, W = 3, 3
    image_coords = create_image_coordinates(H, W)
    
    print("Image coordinates [H, W, 2]:")
    print(image_coords)
    
    # Use the softmaxed features from previous demonstration
    _, softmaxed = demonstrate_spatial_softmax_4d()
    
    # Compute spatial soft argmax
    spatial_argmax = spatial_soft_argmax(softmaxed, image_coords)
    
    print("\nSpatial soft argmax result [N, C, 2]:")
    print(spatial_argmax)
    
    print("\nInterpretation:")
    print("For batch 0, channel 0: mean location =", spatial_argmax[0, 0])
    print("For batch 0, channel 1: mean location =", spatial_argmax[0, 1])
    print("For batch 1, channel 0: mean location =", spatial_argmax[1, 0])
    print("For batch 1, channel 1: mean location =", spatial_argmax[1, 1])
    
    return spatial_argmax

def demonstrate_on_real_image():
    """
    Demonstrate spatial softmax on the DoT.jpg image with multiple channels.
    """
    print("\n" + "="*50)
    print("Demonstrating Spatial Softmax on DoT.jpg Image")
    print("="*50)
    
    # Load the image
    image_path = "DoT.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a 4D feature tensor [1, H, W, 3] from RGB channels
    features = np.expand_dims(image_rgb.astype(np.float64), axis=0)
    
    print(f"Image shape: {image.shape}")
    print(f"Features tensor shape: {features.shape}")
    
    # Apply spatial softmax
    softmaxed = spatial_softmax_4d(features, temperature=1.0)
    
    print(f"Softmaxed shape: {softmaxed.shape}")
    
    # Create image coordinates
    H, W = image.shape[:2]
    image_coords = create_image_coordinates(H, W)
    
    # Compute spatial soft argmax
    spatial_argmax = spatial_soft_argmax(softmaxed, image_coords)
    
    print(f"Spatial argmax shape: {spatial_argmax.shape}")
    print("Mean pixel locations for each channel:")
    print(f"  Red channel: {spatial_argmax[0, 0]}")
    print(f"  Green channel: {spatial_argmax[0, 1]}")
    print(f"  Blue channel: {spatial_argmax[0, 2]}")
    
    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spatial Softmax on DoT.jpg Image', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Individual channels
    channel_names = ['Red', 'Green', 'Blue']
    for i in range(3):
        axes[0, i+1].imshow(features[0, :, :, i], cmap='gray')
        axes[0, i+1].set_title(f'{channel_names[i]} Channel')
        axes[0, i+1].axis('off')
    
    # Softmaxed channels
    for i in range(3):
        im = axes[1, i].imshow(softmaxed[0, :, :, i], cmap='viridis')
        axes[1, i].set_title(f'Softmaxed {channel_names[i]}')
        axes[1, i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Plot the spatial argmax points on the image
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image_rgb)
    
    # Plot mean locations for each channel
    colors = ['red', 'green', 'blue']
    for i, (color, name) in enumerate(zip(colors, channel_names)):
        x, y = spatial_argmax[0, i]
        ax.plot(x, y, 'o', color=color, markersize=15, label=f'{name} channel')
        ax.text(x+10, y+10, f'{name}\n({x:.1f}, {y:.1f})', 
                color=color, fontsize=12, fontweight='bold')
    
    ax.set_title('Spatial Soft Argmax: Mean Pixel Locations')
    ax.legend()
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def demonstrate_temperature_effects():
    """
    Demonstrate how temperature affects spatial softmax and argmax.
    """
    print("\n" + "="*50)
    print("Temperature Effects on Spatial Softmax")
    print("="*50)
    
    # Create a simple 4D feature tensor
    features = np.random.randn(1, 5, 5, 2) * 2
    
    # Create image coordinates
    H, W = 5, 5
    image_coords = create_image_coordinates(H, W)
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(len(temperatures), 3, figsize=(15, 3*len(temperatures)))
    fig.suptitle('Temperature Effects on Spatial Softmax', fontsize=16)
    
    for i, temp in enumerate(temperatures):
        # Apply spatial softmax
        softmaxed = spatial_softmax_4d(features, temperature=temp)
        
        # Compute spatial argmax
        spatial_argmax = spatial_soft_argmax(softmaxed, image_coords)
        
        # Plot original features (only once)
        if i == 0:
            axes[i, 0].imshow(features[0, :, :, 0], cmap='viridis')
            axes[i, 0].set_title('Original Features (Channel 0)')
        else:
            axes[i, 0].axis('off')
        
        # Plot softmaxed result
        im = axes[i, 1].imshow(softmaxed[0, :, :, 0], cmap='viridis')
        axes[i, 1].set_title(f'Softmaxed (T={temp})')
        
        # Plot spatial argmax location
        axes[i, 2].imshow(softmaxed[0, :, :, 0], cmap='viridis')
        x, y = spatial_argmax[0, 0]
        axes[i, 2].plot(x, y, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2)
        axes[i, 2].set_title(f'Spatial Argmax: ({x:.1f}, {y:.1f})')
        
        # Turn off axes
        for j in range(3):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nTemperature effects on spatial argmax:")
    for temp in temperatures:
        softmaxed = spatial_softmax_4d(features, temperature=temp)
        spatial_argmax = spatial_soft_argmax(softmaxed, image_coords)
        
        print(f"  T={temp:4.1f}: Channel 0 = {spatial_argmax[0, 0]}, Channel 1 = {spatial_argmax[0, 1]}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_spatial_softmax_4d()
    demonstrate_spatial_soft_argmax()
    demonstrate_on_real_image()
    demonstrate_temperature_effects()
    
    print("\n" + "="*50)
    print("Advanced Spatial Softmax Demonstration Complete!")
    print("="*50)
