# Spatial Softmax Implementation

This repository contains implementations of spatial softmax functions in Python, demonstrating both basic and advanced approaches for computer vision applications.

## Overview

Spatial softmax is a technique that converts feature maps into probability distributions over spatial locations. It's commonly used in:

- **Attention mechanisms** in neural networks
- **Spatial localization** tasks
- **Keypoint detection** and **pose estimation**
- **Visual attention** models

## Files

### 1. `simple_spatial_softmax_demo.py`
Basic implementation with step-by-step demonstrations:
- 2D spatial softmax for single-channel feature maps
- 3D spatial softmax for multi-channel feature maps
- Simple examples with 3x3 and 2x2x3 tensors
- Visualization on the DoT.jpg image

### 2. `spatial_softmax_demo.py`
Advanced implementation with interactive features:
- Multiple feature map types (gradient, intensity, random)
- Interactive temperature adjustment with sliders
- Comprehensive visualization of different temperatures
- Statistical analysis of results

### 3. `spatial_softmax_advanced.py`
TensorFlow-style implementation following the user's specification:
- 4D tensor support [N, H, W, C] (batch, height, width, channels)
- Spatial soft argmax computation
- Coordinate grid generation
- Temperature effects demonstration

## Mathematical Formulation

### Basic Spatial Softmax
For a 2D feature map `F` of size `[H, W]`:

```
softmax(F)_ij = exp(F_ij / T) / Σ_{k,l} exp(F_kl / T)
```

where `T` is the temperature parameter.

### 4D Spatial Softmax (TensorFlow Style)
For a 4D feature tensor `F` of size `[N, H, W, C]`:

1. **Transpose**: `[N, H, W, C]` → `[N, C, H, W]`
2. **Reshape**: `[N, C, H, W]` → `[N*C, H*W]`
3. **Apply softmax**: Over spatial dimensions `H*W`
4. **Reshape back**: `[N*C, H*W]` → `[N, C, H, W]`
5. **Transpose back**: `[N, C, H, W]` → `[N, H, W, C]`

### Spatial Soft Argmax
Using softmax weights `W` and image coordinates `C`:

```
spatial_argmax_nc = Σ_{h,w} W_nhwc * C_hw
```

This computes the weighted mean location for each channel.

## Key Functions

### `spatial_softmax_2d(feature_map, temperature=1.0)`
- **Input**: 2D numpy array `[H, W]`
- **Output**: 2D array with same shape, values sum to 1.0
- **Use case**: Single-channel feature maps

### `spatial_softmax_3d(feature_map, temperature=1.0)`
- **Input**: 3D numpy array `[H, W, C]`
- **Output**: 3D array with same shape
- **Use case**: Multi-channel feature maps

### `spatial_softmax_4d(features, temperature=1.0)`
- **Input**: 4D numpy array `[N, H, W, C]`
- **Output**: 4D array with same shape
- **Use case**: Batch processing, TensorFlow compatibility

### `spatial_soft_argmax(softmax_weights, image_coords)`
- **Input**: Softmax weights `[N, H, W, C]`, coordinates `[H, W, 2]`
- **Output**: Mean locations `[N, C, 2]`
- **Use case**: Computing attention-weighted spatial locations

## Temperature Parameter

The temperature parameter `T` controls the sharpness of the softmax distribution:

- **Low temperature (T < 1)**: Sharper distribution, more focused attention
- **High temperature (T > 1)**: Softer distribution, more uniform attention
- **T = 1**: Standard softmax behavior

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `opencv-python>=4.5.0`
- `scipy>=1.7.0`

## Usage Examples

### Basic Usage
```python
import numpy as np
from simple_spatial_softmax_demo import spatial_softmax_2d

# Create a simple feature map
feature_map = np.array([[1.0, 2.0], [3.0, 4.0]])

# Apply spatial softmax
softmaxed = spatial_softmax_2d(feature_map, temperature=1.0)
print(softmaxed)
```

### Advanced Usage
```python
from spatial_softmax_advanced import spatial_softmax_4d, spatial_soft_argmax

# Create 4D features [batch, height, width, channels]
features = np.random.randn(2, 64, 64, 3)

# Apply spatial softmax
softmaxed = spatial_softmax_4d(features, temperature=1.0)

# Compute spatial argmax
image_coords = create_image_coordinates(64, 64)
spatial_argmax = spatial_soft_argmax(softmaxed, image_coords)
```

## Running the Demos

### Simple Demo
```bash
python simple_spatial_softmax_demo.py
```

### Advanced Demo
```bash
python spatial_softmax_advanced.py
```

### Interactive Demo
```bash
python spatial_softmax_demo.py
```

## Applications

1. **Keypoint Detection**: Use spatial softmax to predict keypoint locations
2. **Attention Mechanisms**: Weight spatial features based on importance
3. **Pose Estimation**: Localize body parts or object components
4. **Visual Question Answering**: Focus attention on relevant image regions
5. **Image Segmentation**: Generate attention maps for different regions

## Key Insights

- **Spatial softmax** converts arbitrary feature values to probabilities
- **Temperature scaling** controls attention sharpness
- **Spatial argmax** provides differentiable location prediction
- **Batch processing** enables efficient multi-image processing
- **Channel independence** allows separate attention for different features

## References

- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- [Soft Argmax](https://arxiv.org/abs/1603.08327)
- [Attention Mechanisms in Computer Vision](https://arxiv.org/abs/1805.08318)
