from core.config_loader import ConfigLoader
import torch

class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, device="cuda"):
        if self._initialized:
            return

        self.config = ConfigLoader("config/model_config.yaml").get('models')

        from ultralytics import YOLO
        from transformers import DPTForDepthEstimation, DPTFeatureExtractor, CLIPModel, CLIPProcessor
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        
        from vision.segmentation import SAM2BoxSegmenter
        from vision.depth import MiDaSDepthEstimator
        from vision.edges import StructuralLineDetector

        from editors.room_decor_inpainter import DiffusionInpainter
        from editors.global_editor import GeometryAwareImg2ImgGenerator

        from agents.quality_checker import ImageQualityValidator

        self.device = device

        # YOLO Detector
        self.object_detector = YOLO(self.config['detector']['model_id'])

        # SAM Segmentation
        self.sam_segmenter = SAM2BoxSegmenter(model_id=self.config['segmentation']['model_id'], device=device)

        self.inpainter = DiffusionInpainter(model_id=self.config['inpaint_model']['model_id'])
        self.global_generator = GeometryAwareImg2ImgGenerator(self.config['global_editor']['model_id'])

        # Depth Estimation
        self.depth_estimator = MiDaSDepthEstimator(model_type=self.config['depth']['model_type'])
        self.line_detector = StructuralLineDetector(self.config['edges']['model_id']))

        # CLIP for quality check
        self.validator = ImageQualityValidator(clip_model_id=self.config['quality']['clip_model_id'])

        self._initialized = True

    def get(self, name):
        return getattr(self, name)
    

#from core.model_registry import ModelRegistry
# registry = ModelRegistry()
# registry.initialize(device="cuda")

# builder.add_node("local_editor", lambda state: local_editor_node(state, registry))
# builder.add_node("global_editor", lambda state: global_editor_node(state, registry))