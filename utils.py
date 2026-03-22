import numpy as np, os
from PIL import Image
import torch
from core.config_loader import ConfigLoader
from pathlib import Path

from core.logger import logger

def crop_and_resize(image, mask_np, margin=20, target_size=512):
    """
    Crop image & mask based on mask bounding box, add optional margin.
    Resize to target_size (square) for SD inpainting.
    """
    #mask_np = np.array(mask)
    mask = Image.fromarray(mask_np)
    ys, xs = np.where(mask_np > 127)
    y1, y2 = max(0, ys.min() - margin), min(mask_np.shape[0], ys.max() + margin)
    x1, x2 = max(0, xs.min() - margin), min(mask_np.shape[1], xs.max() + margin)

    crop_img = image.crop((x1, y1, x2, y2))
    crop_mask = mask.crop((x1, y1, x2, y2))

    # Resize to target_size
    crop_img = crop_img.resize((target_size, target_size))
    crop_mask = crop_mask.resize((target_size, target_size))

    return crop_img, crop_mask, (x1, y1, x2, y2)


def paste_back(final_img, inpainted_crop, bbox, original_size):
    """
    Resize inpainted crop back to original bbox size and paste into final image.
    """
    x1, y1, x2, y2 = bbox
    inpaint_resized = inpainted_crop.resize((x2 - x1, y2 - y1))
    final_img.paste(inpaint_resized, (x1, y1))
    return final_img

def get_cropped_mask_images(final_masks, image, margin=20):
    crop_images = []
    crop_masks = []
    bboxes = []

    mask_tensor = final_masks.squeeze(1)

    for count in range(len(final_masks)):

        mask_t = mask_tensor[count]
        mask_np = mask_t.cpu().numpy().astype(np.uint8) * 255    
        #mask = Image.open(mask_path)
        crop_img, crop_mask, bbox = crop_and_resize(image, mask_np, margin=margin)
        crop_images.append(crop_img)
        crop_masks.append(crop_mask)
        bboxes.append(bbox)

    return crop_images, crop_masks, bboxes

def combine_masks(final_mask):

    mask_tensor = final_mask[0]          # shape: (N, 1, H, W)
    mask_tensor = mask_tensor.squeeze(1) # shape: (N, H, W)
    combined_mask = torch.any(mask_tensor, dim=0)  # shape: (H, W)
    mask_np = combined_mask.cpu().numpy().astype(np.uint8) * 255

    return mask_np

def syncing_prompts_with_detected_objects(detected_labels, objects_to_edit):

    # Align prompts with detected instances sequentially
    # Track how many times we've used a prompt for each label
    prompt_counters = {}
    final_prompts = []

    for det_label in detected_labels:
    # Find all matching prompts from state for this specific label
        matching_objs = [o for o in objects_to_edit if o['label'] == det_label]
        
        if matching_objs:
            # Get the current index for this label, defaulting to 0
            idx = prompt_counters.get(det_label, 0)
            
            # If we have a specific prompt for this instance, use it
            if idx < len(matching_objs):
                final_prompts.append(matching_objs[idx]['edit_prompt'])
            else:
                # Fallback to the last available prompt if we have more objects than prompts
                final_prompts.append(matching_objs[-1]['edit_prompt'])
            
            # Increment the counter for this label
            prompt_counters[det_label] = idx + 1
        else:
            final_prompts.append("keep as is")

    return final_prompts

def save_output(input_image_path, session_id, image, message_index, prefix):
    image_name = Path(input_image_path).name
    output_folder = ConfigLoader("config/pipeline_config.yaml").get('paths')['outputs']
    thread_folder = os.path.join(output_folder, f'{image_name}_{session_id}')
    Path(thread_folder).mkdir(parents=True, exist_ok=True)

    final_path = os.path.join(thread_folder, f'{prefix}_{message_index}.jpg')
    image.save(final_path)

    logger.info(f"{prefix} image has been saved to {final_path}, for thread id : {session_id} and message index {message_index}")
    return final_path

def wrong_object_class_by_planner(planner_prompts):

    detectable_list = ConfigLoader("config/model_config.yaml").get('models')['detector']['detectable_objects']
    # Check if any label is NOT in the detectable_list
    if any(item['label'] not in detectable_list for item in planner_prompts):
        # Concatenate all edit_prompts separated by a space
        final_prompt = " ".join(item['edit_prompt'] for item in planner_prompts)
        return True, final_prompt
    else:
        return False, None