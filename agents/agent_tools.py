from utils import syncing_prompts_with_detected_objects, combine_masks, get_cropped_mask_images, paste_back
from PIL import Image
from agents.agent_types import AgentState
import cv2

def local_editor_node(state: AgentState, model_registry):
    """
    Use this when the user wants to change specific objects
    like sofa, table, chair, lamp, etc.
    """

    image_path = state['image_path']
    objects_to_edit = state['objects'] # List of {'label': ..., 'edit_prompt': ...}
    target_labels = list(set([obj['label'] for obj in objects_to_edit]))

    img = Image.open(image_path)
    # 1. Detect all instances of the requested labels
    bboxes, detected_labels = model_registry.object_detector.get_bounding_boxes(img, target_labels)
    
    if not bboxes:
        print("No matching objects detected.")
        return state

    # 2. Align prompts with detected instances sequentially
    # Track how many times we've used a prompt for each label
    final_prompts = syncing_prompts_with_detected_objects(detected_labels, objects_to_edit)
    
    print('final prompts assigned:', final_prompts)

    # 3. Generate masks for all detected boxes
    final_masks = model_registry.sam_segmenter.generate_mask_from_boxes(img, bboxes)
    #combined mask for quality checker node
    combined_mask = combine_masks(final_masks)
    state['last_mask'] = combined_mask

    img = img.convert('RGB')
    # 4. Crop regions based on masks
    crop_images, crop_masks, bboxes_out = get_cropped_mask_images(final_masks[0], img)
    
    # 5. Batch inpainting with aligned prompts
    results = model_registry.inpainter.inpaint_batch(crop_images, crop_masks, final_prompts)
    
    final_image = img.copy()
    for result, bbox in zip(results, bboxes_out):
        final_image = paste_back(final_image, result, bbox, img.size)

    # save output
    result_path = "agent_output_local.png"
    final_image.save(result_path)
    state['current_image_path'] = result_path
    state['previous_image_path'] = image_path

    return state

def global_editor_node(state: AgentState, model_registry):
    """
    Use this when the user wants to change the entire room style
    or overall look.
    """
    image_path = state['image_path']
    img = cv2.imread(image_path)
    
    # Estimators need NumPy/OpenCV format
    _, depth_norm = model_registry.depth_estimator.predict(img)
    _, line_map_img = model_registry.line_detector.compute(img)

    # Generator needs PIL
    depth_map_img = Image.fromarray(depth_norm)

    # Use dynamic strength from state, default to 0.35 if not set
    strength_val = state.get('strength', 0.35)
    print('prompt and strength', state['global_prompt'], strength_val)
    result = model_registry.global_generator.generate(
        prompt=state['global_prompt'],
        original_image=Image.open(image_path),
        line_map=line_map_img,
        depth_map=depth_map_img,
        strength = strength_val,
        resizing=True, image_size = (1024, 1024)
    )

    result_path = "agent_output_global.png"
    result.save(result_path)
    state['current_image_path'] = result_path
    state['previous_image_path'] = image_path

    state['depth_original'] = depth_norm
    _, state['depth_generated'] = model_registry.depth_estimator.predict(cv2.imread(result_path))
    return state