import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as vutils
import argparse
from PIL import Image
import sys
import numpy as np
from model import CNN

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def save_output(output_tensor, save_path):
    """Save output tensor as an image"""
    #print(f"Attempting to save to: {save_path}")
    save_path = os.path.normpath(save_path)
    # Create directory if no path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(output_tensor, save_path)
    print(f"Successfully saved to: {save_path}")
    
    return save_path


def main(lineart_folder_path, reference_image_path, output_directory, i ): #lineart_folder_path, reference_image_path, output_directory, i 
    
    # paths, change as necessary
    checkpoint_path = 'MODELS_ENHANCED_FIXED/checkpoints/best_model_clean.pth'
    
    # mostly for windows issues
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at: {checkpoint_path}")
        print(f"Current directory: {os.getcwd()}")
        return None
        
    output_filename = f'frame_{i:04d}.png'
    #output_filename = 'elephant_24.png'
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #arguments
    reference_path = reference_image_path
    lineart_path = lineart_folder_path
    output_dir = output_directory

    

    model = CNN().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    
    lineart = load_image(lineart_path, transform).unsqueeze(0).to(device)
    reference = load_image(reference_path, transform).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(lineart, reference)
        output = output.clamp(0, 1)
        
        output_path = os.path.join(output_dir, output_filename)
        save_output(output[0], output_path)

        return output_path
        
        """# Create comparison visualization
        vis_path = os.path.join(output_dir, 'comparison_' + output_filename)

        vis_images = [
            lineart[0].cpu(),
            reference[0].cpu(),
            output[0].cpu()
        ]
        grid = vutils.make_grid(vis_images, nrow=3, padding=5, normalize=False)
        vutils.save_image(grid, vis_path)
        print(f"Saved visualization to {vis_path}")
        
        print(f"Output saved to {output_path}")"""

if __name__ == "__main__":
    main() 