import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.special import softmax

def spatial_softmax(feature_map, temperature=1.0):
    """
    Apply spatial softmax to a 2D feature map.
    
    Args:
        feature_map: 2D numpy array (H, W) or (H, W, C)
        temperature: Temperature parameter for softmax (higher = sharper distribution)
    
    Returns:
        Softmaxed feature map with same shape as input
    """
    if len(feature_map.shape) == 3:
        # Handle multi-channel case
        H, W, C = feature_map.shape
        feature_map_reshaped = feature_map.reshape(-1, C)
        softmaxed = softmax(feature_map_reshaped / temperature, axis=0)
        return softmaxed.reshape(H, W, C)
    else:
        # Handle single-channel case
        H, W = feature_map.shape
        feature_map_reshaped = feature_map.reshape(-1)
        softmaxed = softmax(feature_map_reshaped / temperature)
        return softmaxed.reshape(H, W)

def create_sample_feature_map(image, feature_type='gradient'):
    """
    Create a sample feature map from an image.
    
    Args:
        image: Input image
        feature_type: Type of feature to extract ('gradient', 'intensity', 'random')
    
    Returns:
        Feature map as numpy array
    """
    if feature_type == 'gradient':
        # Compute gradient magnitude
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        feature_map = np.sqrt(grad_x**2 + grad_y**2)
        return feature_map
    
    elif feature_type == 'intensity':
        # Use image intensity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return gray.astype(np.float64)
    
    elif feature_type == 'random':
        # Generate random feature map
        H, W = image.shape[:2]
        return np.random.randn(H, W) * 10
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

def visualize_spatial_softmax(image_path, temperature=1.0):
    """
    Visualize the effect of spatial softmax on different feature maps.
    
    Args:
        image_path: Path to input image
        temperature: Temperature parameter for softmax
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create different feature maps
    feature_types = ['gradient', 'intensity', 'random']
    
    fig, axes = plt.subplots(2, len(feature_types), figsize=(15, 10))
    fig.suptitle(f'Spatial Softmax Demo (Temperature = {temperature})', fontsize=16)
    
    for i, feature_type in enumerate(feature_types):
        # Create feature map
        feature_map = create_sample_feature_map(image, feature_type)
        
        # Apply spatial softmax
        softmaxed_map = spatial_softmax(feature_map, temperature)
        
        # Plot original feature map
        axes[0, i].imshow(feature_map, cmap='viridis')
        axes[0, i].set_title(f'Original {feature_type.capitalize()}')
        axes[0, i].axis('off')
        
        # Plot softmaxed feature map
        axes[1, i].imshow(softmaxed_map, cmap='viridis')
        axes[1, i].set_title(f'Softmaxed {feature_type.capitalize()}')
        axes[1, i].axis('off')
        
        # Add colorbar
        plt.colorbar(axes[1, i].images[0], ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nFeature Map Statistics (Temperature = {temperature}):")
    for feature_type in feature_types:
        feature_map = create_sample_feature_map(image, feature_type)
        softmaxed_map = spatial_softmax(feature_map, temperature)
        
        print(f"\n{feature_type.capitalize()}:")
        print(f"  Original - Min: {feature_map.min():.4f}, Max: {feature_map.max():.4f}")
        print(f"  Softmaxed - Min: {softmaxed_map.min():.4f}, Max: {softmaxed_map.max():.4f}")
        print(f"  Softmaxed - Sum: {softmaxed_map.sum():.4f}")

def interactive_spatial_softmax(image_path):
    """
    Interactive demonstration with adjustable temperature.
    """
    import matplotlib.widgets as widgets
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Interactive Spatial Softmax Demo', fontsize=16)
    
    # Create feature map
    feature_map = create_sample_feature_map(image, 'gradient')
    
    # Initial plots
    im1 = ax1.imshow(feature_map, cmap='viridis')
    ax1.set_title('Original Feature Map (Gradient)')
    ax1.axis('off')
    
    im2 = ax2.imshow(spatial_softmax(feature_map, 1.0), cmap='viridis')
    ax2.set_title('Softmaxed Feature Map')
    ax2.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add slider for temperature
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = widgets.Slider(ax_slider, 'Temperature', 0.1, 5.0, valinit=1.0)
    
    def update(val):
        temperature = slider.val
        softmaxed = spatial_softmax(feature_map, temperature)
        im2.set_array(softmaxed)
        ax2.set_title(f'Softmaxed Feature Map (T={temperature:.2f})')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to the sample image
    image_path = "DoT.jpg"
    
    print("Spatial Softmax Function Implementation")
    print("=" * 50)
    
    # Basic demonstration
    print("\n1. Basic demonstration with different temperatures:")
    for temp in [0.5, 1.0, 2.0]:
        print(f"\nTemperature = {temp}")
        visualize_spatial_softmax(image_path, temp)
    
    # Interactive demonstration
    print("\n2. Interactive demonstration (adjust temperature with slider):")
    interactive_spatial_softmax(image_path)


