from PIL import Image
import os
import sys
import glob
import predict
import re


def natural_sort_key(s):
    #Sort strings that contain numbers in order.
    #ie frame1.png before frame10.png
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', os.path.basename(s))]


def main():
    # paths, change as necessary
    base_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(base_dir)
    
    animation_folder_path = os.path.join(project_root, 'data', 'ANIMATION_INPUT')
    colored_animation_outputDIR = os.path.join(project_root, 'results', 'output')
    reference_image_path = os.path.join(project_root, 'data', 'REFERENCE_IMAGE', 'kaya.png')
    
    print(f"Animation folder: {animation_folder_path}")
    print(f"Output directory: {colored_animation_outputDIR}")
    print(f"Reference image: {reference_image_path}")
    
    # Create output directory
    os.makedirs(colored_animation_outputDIR, exist_ok=True)
    print(f"Created output directory: {colored_animation_outputDIR}")
    
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    frames = []
    for ext in image_extensions:
        frames.extend(glob.glob(os.path.join(animation_folder_path, ext)))
    
    frames.sort(key=natural_sort_key)
    
    if not frames:
        print(f"No images found in {animation_folder_path}")
        return
    
    print(f"Animation Received: Processing {len(frames)} frames")
    print("Frame order:")
    for i, frame in enumerate(frames):
        print(f"  {i+1}: {os.path.basename(frame)}")

        #og image sizes
    reference_img = Image.open(reference_image_path)
    reference_size = reference_img.size
    print(f"Reference image size: {reference_size}")

    # Process each frame
    for i, frame_path in enumerate(frames):
        print(f"Processing frame {i+1}/{len(frames)}: {os.path.basename(frame_path)}")
        
        # Process the frame
        output_path = predict.main(frame_path, reference_image_path, colored_animation_outputDIR, i)
        
        # final resizing
        processed_img = Image.open(output_path)
        resized_img = processed_img.resize(reference_size, Image.LANCZOS)
        resized_img.save(output_path)
        
        print(f"Resized frame {i+1} to {reference_size}")

    print("The Colored Animation is saved in " + colored_animation_outputDIR)

if __name__ == "__main__":
    main()

