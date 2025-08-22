import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
import numpy as np
from PIL import Image
import os.path as osp


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
from model import CNN
from torchvision import transforms

class LineartDataset(Dataset):
    def __init__(self, lineart_dir, reference_dir, transform=None):
        self.lineart_dir = lineart_dir
        self.reference_dir = reference_dir
        self.transform = transform
        
        # Find image files, lineart
        self.lineart_files = []
        for root, _, files in os.walk(lineart_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # Store the full path
                    self.lineart_files.append(os.path.join(root, file))
        
        # Find all image files, reference
        self.reference_files = []
        for root, _, files in os.walk(reference_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # full path
                    self.reference_files.append(os.path.join(root, file))
        
        if len(self.lineart_files) == 0:
            raise ValueError(f"No image files found in {lineart_dir}")
        if len(self.reference_files) == 0:
            raise ValueError(f"No image files found in {reference_dir}")
        
        # Sort
        self.lineart_files.sort()
        self.reference_files.sort()
        
        # If lineart and reference directories have different number of files, use the smaller number to avoid index errors
        self.dataset_size = min(len(self.lineart_files), len(self.reference_files))
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # Get file paths
        lineart_path = self.lineart_files[idx]
        reference_path = self.reference_files[idx]
        
        # Load images
        lineart = Image.open(lineart_path).convert('RGB')
        reference = Image.open(reference_path).convert('RGB')

        if self.transform:
            lineart = self.transform(lineart)
            reference = self.transform(reference)
        
        return lineart, reference


class CombinedLoss(torch.nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.l1_loss = torch.nn.L1Loss()
    
    def edge_loss(self, pred, lineart):
        gray_lineart = 0.299 * lineart[:, 0] + 0.587 * lineart[:, 1] + 0.114 * lineart[:, 2]
        edges = gray_lineart > 0.5  

        edge_preservation = torch.nn.functional.mse_loss(
            pred[:, 0] * edges, lineart[:, 0] * edges
        ) + torch.nn.functional.mse_loss(
            pred[:, 1] * edges, lineart[:, 1] * edges
        ) + torch.nn.functional.mse_loss(
            pred[:, 2] * edges, lineart[:, 2] * edges
        )
        
        return edge_preservation
    
    def forward(self, pred, target, line_art=None):
        # Default
        l1 = self.l1_loss(pred, target)
        
        # Edge preservation if line art
        if line_art is not None:
            edge = self.edge_loss(pred, line_art)
        else:
            edge = torch.tensor(0.0, device=self.device)
        
        total_loss = l1 + edge
        
        return total_loss, {
            'l1': l1.item(),
            'edge': edge.item() if line_art is not None else 0.0
        }

def main():
    # Paths
    #if you run the training, be sure to change these as well.
    checkpoint_path = '/home/kellyliu/FinalProject/ModifiedCNN/MODELS_ENHANCED_FIXED/checkpoints/best_model.pth'
    lineart_dir = '/home/kellyliu/FinalProject/datasets/LineartPreprocessed'
    reference_dir = '/home/kellyliu/FinalProject/datasets/train_10k'
    lineart_test_dir = '/home/kellyliu/FinalProject/datasets/LineartPreprocessedTEST'
    reference_test_dir = '/home/kellyliu/FinalProject/datasets/test_2k_original'
    output_dir = '/home/kellyliu/FinalProject/ModifiedCNN/MODELS_ENHANCED_FIXED'
    
    # HYPERPARAMETERS
    batch_size = 4
    epochs = 15
    lr = 0.0001
    num_workers = 4
    log_freq = 10
    sample_freq = 100
    save_freq = 1
    early_stop = 10
    seed = 42
    
    # Loss weights
    edge_weight = 7.0       
    saturation_weight = 0.75    
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    model = CNN().to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Try to load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
    
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resuming training from epoch {start_epoch + 1}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
    ])
    
    # Load dataset 
    train_dataset = LineartDataset(
        lineart_dir=lineart_dir,
        reference_dir=reference_dir,
        transform=transform
    )

    #shifted shuffling
    class ShiftedPairsDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset
            self.dataset_size = len(base_dataset)
        
        def __len__(self):
            return self.dataset_size
        
        def __getitem__(self, idx):
            # Pair each lineart with the next reference image
            # The last one wraps around to the first reference
            lineart, _ = self.base_dataset[idx]
            next_idx = (idx + 1) % self.dataset_size
            _, reference = self.base_dataset[next_idx]
            
            return lineart, reference
    
    # Create a new dataset with shifted pairs
    shifted_dataset = ShiftedPairsDataset(train_dataset)
    print(f"Created shifted dataset with {len(shifted_dataset)} samples")
    
    # Combine regular dataset and shifted dataset
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([train_dataset, shifted_dataset])
    print(f"Final training dataset size: {len(train_dataset)}")
    
    # Load separate test dataset
    val_dataset = LineartDataset(
        lineart_dir=lineart_test_dir,
        reference_dir=reference_test_dir,
        transform=transform
    )
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,  # Force batch size=1 for validation
        shuffle=False,
        num_workers=1,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create criterion
    criterion = CombinedLoss(device)
    original_forward = criterion.forward
    
    def modified_forward(self, pred, target, line_art=None):
        if pred.shape[2:] != (256, 256):
            pred = torch.nn.functional.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=True)
        if target.shape[2:] != (256, 256):
            target = torch.nn.functional.interpolate(target, size=(256, 256), mode='bilinear', align_corners=True)
            
        # saturation control
        pred_r, pred_g, pred_b = pred[:, 0], pred[:, 1], pred[:, 2]
        
        #value (max RGB)
        pred_v = torch.max(torch.max(pred_r, pred_g), pred_b)
        
        # chroma (max - min RGB)
        pred_min = torch.min(torch.min(pred_r, pred_g), pred_b)
        pred_c = pred_v - pred_min
        
        #saturation reduction
        pred_c_reduced = pred_c * saturation_weight
        
        #RGB with reduced saturation
        eps = 1e-8
        pred_r_norm = torch.where(pred_v > eps, (pred_r - pred_min) / (pred_v - pred_min + eps), torch.zeros_like(pred_r))
        pred_g_norm = torch.where(pred_v > eps, (pred_g - pred_min) / (pred_v - pred_min + eps), torch.zeros_like(pred_g))
        pred_b_norm = torch.where(pred_v > eps, (pred_b - pred_min) / (pred_v - pred_min + eps), torch.zeros_like(pred_b))
        
        #reduced chroma
        pred_r_new = pred_min + pred_r_norm * pred_c_reduced
        pred_g_new = pred_min + pred_g_norm * pred_c_reduced
        pred_b_new = pred_min + pred_b_norm * pred_c_reduced
        
        # Stack back to tensor
        pred_adjusted = torch.stack([pred_r_new, pred_g_new, pred_b_new], dim=1)
        
        # Combined loss
        l1 = self.l1_loss(pred_adjusted, target) * 20.0
        
        # Add MSE
        mse = torch.nn.functional.mse_loss(pred_adjusted, target) * 10.0
        
        # Edge preservation loss
        if line_art is not None:
            if line_art.shape[2:] != (256, 256):
                line_art = torch.nn.functional.interpolate(line_art, size=(256, 256), mode='bilinear', align_corners=True)
            edge = self.edge_loss(pred_adjusted, line_art) * edge_weight
        else:
            edge = torch.tensor(0.0, device=self.device)
        
        total_loss = l1 + mse + edge
        
        return total_loss, {
            'l1': l1.item(),
            'mse': mse.item(),
            'edge': edge.item() if line_art is not None else 0.0
        }


    criterion.forward = modified_forward.__get__(criterion, CombinedLoss)
    print(f"Using edge preservation weight of {edge_weight} and saturation control of {saturation_weight}")
    
    #scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    log_file = open(os.path.join(output_dir, 'continued_training_log.txt'), 'a')
    log_file.write(f"Continuing training from epoch {start_epoch + 1} with edge weight {edge_weight}, saturation {saturation_weight}\n")
    log_file.flush()
    
    print("Starting training loop...")
    from tqdm import tqdm
    import time
    import torch.nn.functional as F
    import torchvision.utils as vutils
    
    # Fixed size
    img_size = 256
    clip_grad = 0
    
    # Early stopping
    early_stop_counter = 0
    early_stop_patience = early_stop
    
    # Training loop
    total_steps = 0
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_metrics = {'l1': 0, 'mse': 0, 'edge': 0}
        start_time = time.time()
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs} [Train]")
        for i, (lineart, reference) in enumerate(train_progress):
            try: 
                if lineart.shape[2:] != (256, 256) or reference.shape[2:] != (256, 256):
                    # Force reshape to 256x256
                    lineart = F.interpolate(lineart, size=(256, 256), mode='bilinear', align_corners=True)
                    reference = F.interpolate(reference, size=(256, 256), mode='bilinear', align_corners=True)
                
                # Move to device
                lineart = lineart.to(device)
                reference = reference.to(device)
                if i == 0:
                    print(f"First batch dimensions: lineart={lineart.shape}, reference={reference.shape}")
                
                # Forward pass
                outputs = model(lineart, reference)
                if outputs.shape[2:] != (256, 256):
                    outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=True)
                
                # Calculate loss
                loss, metrics = criterion(outputs, reference, lineart)
                
                # NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at step {i}. Skipping batch.")
                    continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                train_progress.set_postfix({
                    'loss': loss.item(),
                    'l1': metrics['l1'],
                    'edge': metrics['edge'],
                })
                
                
                if (i + 1) % log_freq == 0:
                    log_step = f"E{epoch+1} | Step {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                    for k, v in metrics.items():
                        log_step += f" | {k}: {v:.4f}"
                    print(log_step)
                    log_file.write(log_step + "\n")
                    log_file.flush()
                
                # Sample every sample_freq steps
                if (i + 1) % sample_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        # Generate a sample
                        sample_lineart = lineart[0:1]
                        sample_reference = reference[0:1]
                        sample_output = model(sample_lineart, sample_reference)
                        
                        # Apply saturation control to output for visualization
                        sample_r, sample_g, sample_b = sample_output[:, 0], sample_output[:, 1], sample_output[:, 2]
                        sample_v = torch.max(torch.max(sample_r, sample_g), sample_b)
                        sample_min = torch.min(torch.min(sample_r, sample_g), sample_b)
                        sample_c = sample_v - sample_min
                        sample_c_reduced = sample_c * saturation_weight
                        
                        eps = 1e-8
                        sample_r_norm = torch.where(sample_v > eps, (sample_r - sample_min) / (sample_v - sample_min + eps), torch.zeros_like(sample_r))
                        sample_g_norm = torch.where(sample_v > eps, (sample_g - sample_min) / (sample_v - sample_min + eps), torch.zeros_like(sample_g))
                        sample_b_norm = torch.where(sample_v > eps, (sample_b - sample_min) / (sample_v - sample_min + eps), torch.zeros_like(sample_b))
                        
                        sample_r_new = sample_min + sample_r_norm * sample_c_reduced
                        sample_g_new = sample_min + sample_g_norm * sample_c_reduced
                        sample_b_new = sample_min + sample_b_norm * sample_c_reduced
                        
                        sample_adjusted = torch.stack([sample_r_new, sample_g_new, sample_b_new], dim=1)
                        
                        grid = torch.cat([
                            sample_lineart,
                            sample_reference,
                            sample_output
                        ], dim=0)
                        
                        sample_path = os.path.join(
                            output_dir, 
                            'samples', 
                            f'sample_epoch{epoch+1}_step{i+1}.png'
                        )
                        vutils.save_image(
                            grid, 
                            sample_path, 
                            nrow=4, 
                            normalize=True
                        )
                        print(f"Saved sample to {sample_path}")
                    model.train()
                
                total_steps += 1
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in epoch_metrics.items()}
        
        log_epoch = f"Epoch {epoch+1}/{start_epoch + epochs} completed in {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f}"
        for k, v in avg_metrics.items():
            log_epoch += f" | {k}: {v:.4f}"
        print(log_epoch)
        log_file.write(log_epoch + "\n")
        log_file.flush()
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_metrics = {'l1': 0, 'mse': 0, 'edge': 0}
            
            print(f"Validating...")
            with torch.no_grad():
                for lineart, reference in tqdm(val_loader, desc=f"Epoch {epoch+1}/{start_epoch + epochs} [Val]"):
                    try:
                        # Verifying tensor shapes
                        if lineart.shape[2:] != (256, 256) or reference.shape[2:] != (256, 256):
                            lineart = F.interpolate(lineart, size=(256, 256), mode='bilinear', align_corners=True)
                            reference = F.interpolate(reference, size=(256, 256), mode='bilinear', align_corners=True)
                        
                        lineart = lineart.to(device)
                        reference = reference.to(device)
                        
                        # Forward pass
                        outputs = model(lineart, reference)
                        
                        if outputs.shape[2:] != (256, 256):
                            outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=True)
                        
                        #loss
                        loss, metrics = criterion(outputs, reference, lineart)
                        val_loss += loss.item()
                        for k, v in metrics.items():
                            val_metrics[k] += v
                    
                    except Exception as e:
                        print(f"Error in validation step: {e}")
                        continue
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
            
            log_val = f"Validation | Avg Loss: {avg_val_loss:.4f}"
            for k, v in avg_val_metrics.items():
                log_val += f" | {k}: {v:.4f}"
            print(log_val)
            log_file.write(log_val + "\n")
            log_file.flush()
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'img_size': img_size
                }
                torch.save(checkpoint, os.path.join(output_dir, 'checkpoints', 'best_model.pth'))
                print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            else:
                early_stop_counter += 1
                print(f"Validation loss did not improve. Early stopping counter: {early_stop_counter}/{early_stop_patience}")
                
                if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered after {early_stop_counter} epochs without improvement")
                    break
        
        #checkpoint saving
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'img_size': img_size
            }
            torch.save(checkpoint, os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pth'))
            print(f"Saved checkpoint for epoch {epoch+1}")

    log_file.write(f"Training completed after {epoch+1} epochs\n")
    log_file.close()
    print(f"Training completed after {epoch+1} epochs")

if __name__ == "__main__":
    main() 