import numpy as np
from PIL import Image

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
