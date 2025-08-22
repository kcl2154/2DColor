import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import MobileNet_V2_Weights
import numpy as np
import random


#Model Architecture, Supervised Learning cross attentionish CNN
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        
    def forward(self, x):
        return F.leaky_relu(self.norm(self.conv(x)))


# Edge detection module
class EdgeDetector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        
        # create Sobel filters
        sobel_h = torch.tensor([[-1, -2, -1], 
                              [0, 0, 0], 
                              [1, 2, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
        sobel_v = torch.tensor([[-1, 0, 1], 
                              [-2, 0, 2], 
                              [-1, 0, 1]], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)
        
        # repeat
        sobel_h = sobel_h.repeat(self.in_channels, 1, 1, 1)
        sobel_v = sobel_v.repeat(self.in_channels, 1, 1, 1)
        
        # apply convolution with groups
        edge_h = F.conv2d(x, sobel_h, padding=1, groups=self.in_channels)
        edge_v = F.conv2d(x, sobel_v, padding=1, groups=self.in_channels)
        
        return torch.sqrt(edge_h**2 + edge_v**2 + 1e-6) # Calculate magnitude


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # encoder
        self.enc1 = ConvBlock(3, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 256)
        
        # Reference encoder with same architecture
        self.ref_enc1 = ConvBlock(3, 32)
        self.ref_enc2 = ConvBlock(32, 64)
        self.ref_enc3 = ConvBlock(64, 128)
        self.ref_enc4 = ConvBlock(128, 256)
        
        # Edge detectors
        self.edge_detect1 = EdgeDetector(32)
        self.edge_detect2 = EdgeDetector(64)
        self.edge_detect3 = EdgeDetector(128)
        self.edge_detect4 = EdgeDetector(256)
        
        #attention for feature fusion
        self.structure_extractor = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Color transfer attention
        self.color_extractor = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=1)
        )
        
        # decoder
        self.dec4 = ConvBlock(256, 128)
        self.dec3 = ConvBlock(128+128, 64)
        self.dec2 = ConvBlock(64+64, 32)
        self.dec1 = ConvBlock(32+32, 16)
        
        # Final output layer
        self.out_conv = nn.Conv2d(16, 3, kernel_size=1)
        
        
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, line_art, reference):
        # i kept on having torch sizing problems
        if line_art.shape[2:] != (256, 256):
            line_art = F.interpolate(line_art, size=(256, 256), mode='bilinear', align_corners=True)
            
        try:
            # Encode line art
            e1 = self.enc1(line_art)            # 256x256
            e1_edge = self.edge_detect1(e1)     
            e1_down = self.downsample(e1)     
            
            e2 = self.enc2(e1_down)           
            e2_edge = self.edge_detect2(e2)    
            e2_down = self.downsample(e2)      
            
            e3 = self.enc3(e2_down)            
            e3_edge = self.edge_detect3(e3)     
            e3_down = self.downsample(e3)      
            
            e4 = self.enc4(e3_down)            
            e4_edge = self.edge_detect4(e4)     
            
            # Encode reference
            r1 = self.ref_enc1(reference)       
            r1_down = self.downsample(r1)      
            r2 = self.ref_enc2(r1_down)        
            r2_down = self.downsample(r2)      
            r3 = self.ref_enc3(r2_down)        
            r3_down = self.downsample(r3)      
            r4 = self.ref_enc4(r3_down)        
            
            # Extract structure 
            structure_mask = self.structure_extractor(e4)
            
            # Extract color
            color_features = self.color_extractor(r4)
            
            # Combine 
            weighted_structure = e4 * (structure_mask * 0.6)  
            weighted_color = color_features * torch.clamp((1.0 - structure_mask * 0.6), min=0.2, max=0.9)  
            

            batch_size, channels, height, width = weighted_color.shape
            color_mean = weighted_color.mean(dim=[2, 3], keepdim=True)
            color_std = weighted_color.std(dim=[2, 3], keepdim=True) + 1e-6
            
            weighted_color = (weighted_color - color_mean) / color_std * 0.3 + weighted_color * 0.7  
            # bottleneck
            bottleneck = self.bottleneck(weighted_structure + weighted_color)
            
            # skip connections
            d4 = self.upsample(bottleneck)     
            d3 = self.dec4(d4)                  
            
            # Edge-aware combination
            e3_enhanced = e3 * (0.8 + e3_edge * 0.4)  
            d3_cat = torch.cat([d3, e3_enhanced], dim=1) 
            
            d3_up = self.upsample(d3_cat)      
            d2 = self.dec3(d3_up)             
            
            # Edge-aware feature combination 
            e2_enhanced = e2 * (0.8 + e2_edge * 0.4)  
            d2_cat = torch.cat([d2, e2_enhanced], dim=1)
            
            d2_up = self.upsample(d2_cat)       
            d1 = self.dec2(d2_up)            

            e1_enhanced = e1 * (0.8 + e1_edge * 0.4) 
            d1_cat = torch.cat([d1, e1_enhanced], dim=1)
            
            d0 = self.dec1(d1_cat)          
            
            pre_output = self.out_conv(d0)
            
            output_channels = pre_output.shape[1]
            channel_means = pre_output.mean(dim=[2, 3], keepdim=True)
            channel_stds = pre_output.std(dim=[2, 3], keepdim=True) + 1e-6
            
            # color bias
            mean_of_means = channel_means.mean(dim=1, keepdim=True)
            channel_bias = channel_means - mean_of_means
            bias_correction = torch.clamp(channel_bias * -0.5, min=-0.3, max=0.3)
            
            # bias correction tanh
            corrected_output = pre_output + bias_correction
            output = torch.tanh(corrected_output * 1.0)  # edges got a lil too sharp
            
            return output
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            print(f"Input shapes: line_art={line_art.shape}, reference={reference.shape}")
            
            # fallback
            batch_size = line_art.shape[0]
            return torch.zeros((batch_size, 3, 256, 256), device=line_art.device)

# Edge preservation loss function
class EdgePreservationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_detector = EdgeDetector(3)
        
    def forward(self, pred, line_art):
        # Extract edges
        line_edges = self.edge_detector(line_art)
        pred_edges = self.edge_detector(pred)
        
        # L1 loss between edges
        return F.l1_loss(pred_edges, line_edges) 