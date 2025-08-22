# Final Project Coding Environment Guide

## Quick Start

1. **Download and Extract**
   ```bash
   unzip Final_Project_kcl2154.zip
   cd Final_Project_kcl2154
   ```

2. **Set Up Python Environment**
   ```bash
   # Create and activate virtual environment
   python3 -m venv .venv
   
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .\.venv\Scripts\activate
   
   # Install dependencies
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```

3. **Prepare Your Files**
   - Place your line art frames in `tests/ANIMATION_INPUT/`
   - Place your reference image in `tests/REFERENCE_IMAGE/`
   - The output will be saved in `tests/image_output/` or `tests/output/`

4. **Run the Script**
   ```bash
   # For single image:
   python ModifiedCNN/predict_image.py
   
   # For animation:
   python ModifiedCNN/animation_loop.py
   ```

## Project Structure
```
Final_Project_kcl2154/
├── ModifiedCNN/              # Main model and processing code
│   ├── predict_image.py      # Script for processing single images
│   ├── animation_loop.py     # Script for processing multiple frames
│   ├── predict.py            # Core processing script (used by animation_loop.py)
│   ├── model.py             # CNN model architecture
│   ├── training.py          # Training script
│   └── MODELS_ENHANCED_FIXED/  # Pre-trained model
│       └── checkpoints/
│           └── best_model_clean.pth
├── tests/                   # Test images and outputs
│   ├── ANIMATION_INPUT/     # Put your line art frames here
│   ├── REFERENCE_IMAGE/     # Put your reference image here
│   └── image_output/       # Colored frames will be saved here
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Requirements
- Python 3.8-3.11 (Python 3.12+ is not supported)
- pip (Python package installer)
- Virtual environment (recommended)

## Troubleshooting

### Common Issues

1. **"No module named 'torch'" Error**
   - Make sure you're in the virtual environment (you should see `(.venv)` in your prompt)
   - Try reinstalling PyTorch:
     ```bash
     pip install torch torchvision torchaudio
     ```

2. **File Not Found Errors**
   - Make sure you're in the project root directory
   - Check that your input files are in the correct directories
   - Verify file permissions

3. **Python Version Issues**
   - Make sure you're using Python 3.8-3.11
   - Check your Python version:
     ```bash
     python --version
     ```

4. **Environment Issues**
   If you need to recreate your environment:
   ```bash
   # Remove old environment
   rm -rf .venv  # On macOS/Linux
   # OR
   rmdir /s /q .venv  # On Windows
   
   # Create new environment
   python3 -m venv .venv
   
   # Activate and install dependencies
   source .venv/bin/activate  # On macOS/Linux
   # OR
   .\.venv\Scripts\activate   # On Windows
   
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```

## Support
If you encounter any issues:
1. Check the troubleshooting section above
2. Make sure you're following all setup steps
3. Verify your Python version is compatible
4. Ensure all dependencies are installed correctly