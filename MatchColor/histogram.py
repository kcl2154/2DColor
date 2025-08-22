import os
import cv2
import numpy as np
from PIL import Image

class MatchColor:

    def match_and_transfer_color(segmented_target, segmented_ref, colored_ref):
        
        """
        Histogram matching
        Match segments between target and reference, then transfer colors.
        
        Args:
            segmented_target: 2D numpy array of segment IDs
            segmented_ref: 2D numpy array of segment IDs
            colored_ref: 3D BGR reference image
            aug_idx: Index for output filename
            
        Returns:
            output: Color-transferred image
        """
        os.makedirs("ColorOutputs", exist_ok=True)
        segmented_target = cv2.resize(segmented_target, (colored_ref.shape[1], colored_ref.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmented_ref = cv2.resize(segmented_ref, (colored_ref.shape[1], colored_ref.shape[0]), interpolation=cv2.INTER_NEAREST)



        output = np.zeros_like(colored_ref)
        target_segments = np.unique(segmented_target) #segment ids extracted
        ref_segments = np.unique(segmented_ref)
        
        for target_id in target_segments:
            if target_id == 0:  # Skip background
                continue
            
            mask_target = (segmented_target == target_id)
            hist_target = cv2.calcHist([colored_ref], [0,1,2], mask_target.astype(np.uint8), 
                                   [8,8,8], [0,256,0,256,0,256])
            
            best_match_id = None
            best_score = -np.inf
            
            for ref_id in ref_segments:
                if ref_id == 0:
                    continue
                
                mask_ref = (segmented_ref == ref_id)
                hist_ref = cv2.calcHist([colored_ref], [0,1,2], mask_ref.astype(np.uint8),
                                     [8,8,8], [0,256,0,256,0,256])
                
                score = cv2.compareHist(hist_target, hist_ref, cv2.HISTCMP_CORREL)
                
                if score > best_score:
                    best_score = score
                    best_match_id = ref_id
            
            if best_match_id is not None:  # Added safety check
                mean_color = cv2.mean(colored_ref, 
                                   mask=(segmented_ref == best_match_id).astype(np.uint8))[:3]
                output[mask_target] = mean_color

        # Convert and save
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(output_rgb)
        img.save(f"ColorOutputs/output_{1}.png")
        
        return output
    
    
    if __name__=='__main__':

        segmented_target = cv2.imread("../CNNlineattempt/segmentation/segframes/sawaguyoutput1.jpg", cv2.IMREAD_GRAYSCALE)
        segmented_ref = cv2.imread("../CNNlineattempt/segmentation/segframes/sawaguyoutput2.jpg", cv2.IMREAD_GRAYSCALE)
        colored_ref = cv2.imread("../CNNlineattempt/segmentation/segframes/sawaguy1.jpg")
        match_and_transfer_color(segmented_target, segmented_ref, colored_ref)


        match_and_transfer_color(segmented_target, segmented_ref, colored_ref)
        print("Finished matching and coloring")