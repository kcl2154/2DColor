import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import sys
from model import CNN

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def save_output(output_tensor, save_path):
    #Save output tensor as image
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    vutils.save_image(output_tensor, save_path)
    
    return save_path

def main():
    # base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # paths, change as necessary
    checkpoint_path = os.path.join(base_dir, 'MODELS_ENHANCED_FIXED', 'checkpoints', 'best_model_clean.pth')
    reference_path = os.path.join(project_root, 'data', 'REFERENCE_IMAGE', 'ashitaka.jpg')
    lineart_path = os.path.join(project_root, 'data', 'ashitaka_image', 'ashitakaTARGET.jpg')
    output_dir = os.path.join(project_root, 'results', 'image_output')
    output_filename = 'output_ashitaka.png'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #arguments
    """reference_path = reference_image_path
    lineart_path = lineart_folder_path
    output_dir = output_directory"""

    
    # Load model
    model = CNN().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    lineart = load_image(lineart_path, transform).unsqueeze(0).to(device)
    reference = load_image(reference_path, transform).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(lineart, reference)
        output = output.clamp(0, 1)
        
        output_path = os.path.join(output_dir, output_filename)
        save_output(output[0], output_path)

        #return output_path
        
        #comparison visualization
        vis_path = os.path.join(output_dir, 'comparison_' + output_filename)
        vis_images = [
            lineart[0].cpu(),
            reference[0].cpu(),
            output[0].cpu()
        ]
        grid = vutils.make_grid(vis_images, nrow=3, padding=5, normalize=False)
        vutils.save_image(grid, vis_path)
        print(f"Saved visualization to {vis_path}")
        
        print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main() 