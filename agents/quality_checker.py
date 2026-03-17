import torch
from transformers import CLIPProcessor, CLIPModel
from core.config_loader import ConfigLoader
import cv2
import numpy as np

class CLIPSemanticChecker:

    def __init__(self, model_id):

        self.model = CLIPModel.from_pretrained(
            model_id
        )

        self.processor = CLIPProcessor.from_pretrained(
           model_id
        )

    def score(self, image, prompt):

        inputs = self.processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        )

        outputs = self.model(**inputs)

        image_emb = outputs.image_embeds
        text_emb = outputs.text_embeds

        similarity = torch.cosine_similarity(
            image_emb,
            text_emb
        )

        return similarity.item()


class StructureChecker:

    def score(self, original, edited):

        orig = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2GRAY)
        edit = cv2.cvtColor(np.array(edited), cv2.COLOR_RGB2GRAY)

        edge_orig = cv2.Canny(orig, 100, 200)
        edge_edit = cv2.Canny(edit, 100, 200)

        diff = np.abs(edge_orig - edge_edit)

        change_ratio = diff.sum() / edge_orig.sum()

        structure_score = 1 - change_ratio

        return max(0, structure_score)
    

class MaskLeakageChecker:

    def score(self, original, edited, mask):

        orig = np.array(original)
        edit = np.array(edited)

        mask = mask.astype(bool)

        outside_mask = ~mask

        diff = np.abs(orig - edit).sum(axis=2)

        leakage = diff[outside_mask].mean()

        score = 1 / (1 + leakage)

        return score

class DepthConsistencyChecker:

    def score(self, depth_original, depth_generated):

        depth_original = np.array(depth_original)
        depth_generated = np.array(depth_generated)

        diff = np.abs(depth_original - depth_generated)

        change_ratio = diff.mean()

        score = 1 / (1 + change_ratio)

        return score


class ImageQualityValidator:

    def __init__(self, clip_model_id='openai/clip-vit-base-patch32'):

        self.semantic = CLIPSemanticChecker(model_id=clip_model_id)
        self.structure = StructureChecker()
        self.mask_check = MaskLeakageChecker()
        self.depth_check = DepthConsistencyChecker()
        self.config = ConfigLoader("config/model_config.yaml")

    def evaluate_local(self, original, edited, mask, prompt):

        semantic = self.semantic.score(edited, prompt)
        structure = self.structure.score(original, edited)
        mask_score = self.mask_check.score(original, edited, mask)

        methods_weights = self.config.get('models')['quality']['inpaint_scoring_weights']

        final = (
            methods_weights['semantic'] * semantic
            + structure * methods_weights['structure']
            + mask_score * methods_weights['segmentation']
        )

        return {
            "semantic": semantic,
            "structure": structure,
            "mask": mask_score,
            "final": final
        }

    def evaluate_global(
        self,
        original,
        edited,
        depth_original,
        depth_generated,
        prompt
    ):
        methods_weights = self.config.get('models')['quality']['inpaint_scoring_weights']

        semantic = self.semantic.score(edited, prompt)
        structure = self.structure.score(original, edited)

        depth_score = self.depth_check.score(
            depth_original,
            depth_generated
        )

        final = (
            semantic * methods_weights['semantic']
            + structure * methods_weights['structure']
            + depth_score * methods_weights['depth']
        )

        return {
            "semantic": semantic,
            "structure": structure,
            "depth": depth_score,
            "final": final
        }