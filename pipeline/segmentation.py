import torch
import numpy as np
import cv2
from transformers import Sam2Processor, Sam2Model
import torch

class SAM2BoxSegmenter:
    """
    Uses SAM2 (hiera-large) to generate masks
    from bounding boxes.
    """

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(self.device)
        self.processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")


    def generate_mask_from_boxes(self, image_bgr, boxes):
        """
        image_bgr: OpenCV image
        boxes: list of [x1, y1, x2, y2]
        returns: combined uint8 mask (0/255)
        """
        image_rgb = image_bgr.convert("RGB")
        #image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=[image_rgb], input_boxes=[boxes], return_tensors="pt").to(self.device)


        #H, W = image_bgr.shape[:2]
        #final_mask = np.zeros((H, W), dtype=np.uint8)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        all_masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
        '''
        for box in boxes:
            box_array = np.array(box)

            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=False
            )

            mask = masks[0]  # (H, W) boolean

            mask_uint8 = mask.astype(np.uint8) * 255
            final_mask = np.maximum(final_mask, mask_uint8)
        '''
        return all_masks
