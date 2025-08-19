#%%
import torch
from torchvision.models import resnet18

# Assume resnet18 is defined globally
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18(pretrained=True).eval().to(device)  # Example initialization

#%%
def get_vector(t_img):
    t_img = t_img.to(device)  # Ensure input is on correct device
    my_embedding = torch.zeros(1, 512, 7, 7).to(device)  # Initialize on device
    
    def copy_data(m, i, o):
        my_embedding.copy_(o)  # No .data needed
    
    h = resnet18.layer4.register_forward_hook(copy_data)
    with torch.no_grad():  # Disable gradients for inference
        resnet18(t_img)
    h.remove()
    return my_embedding

#%%
from PIL import Image
from torchvision import transforms

# Load and preprocess image
image = Image.open("/home/skr3178/DOT_policy/SKR_LoRa/Resnet/dog1.jpg").convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Apply preprocessing to get tensor first, then add batch dimension
t_img = preprocess(image).unsqueeze(0)  # Add batch dimension: [1, 3, 224, 224]

# Get feature map
features = get_vector(t_img)
print(features.shape)  # Output: torch.Size([1, 512, 7, 7])
#%%