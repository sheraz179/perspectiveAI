from core.config_loader import ConfigLoader
import torch
from vision.segmentation import SAM2BoxSegmenter
from vision.depth import MiDaSDepthEstimator
from vision.edges import StructuralLineDetector
from vision.decor_detector import YOLOObjectoxDetector

from editors.room_decor_inpainter import DiffusionInpainter
from editors.global_editor import GeometryAwareImg2ImgGenerator

from agents.quality_checker import ImageQualityValidator
from core.logger import logger

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

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
        self.device = device

        # YOLO Detector
        self.object_detector = YOLOObjectoxDetector(self.config['detector']['model_id'])
        logger.info(f'The room decor objetct  detector has been loaded')

        # SAM Segmentation
        self.sam_segmenter = SAM2BoxSegmenter(model_id=self.config['segmentation']['model_id'], device=device)
        logger.info('The Sam segmentation model has been loaded')

        self.inpainter = DiffusionInpainter(model_id=self.config['diffusion']['inpaint_model']['model_id'])
        logger.info('Inpainting diffuser model has been loaded')

        self.global_generator = GeometryAwareImg2ImgGenerator(self.config['diffusion']['global_editor']['model_id'])
        logger.info('Global editor model has been loaded')

        # Depth Estimation
        self.depth_estimator = MiDaSDepthEstimator(model_type=self.config['depth']['model_type'])
        logger.info('Depth estimator has been loaded')

        self.line_detector = StructuralLineDetector(self.config['edges']['model_id'])
        logger.info('Edge model has been loaded')

        # CLIP for quality check
        self.validator = ImageQualityValidator(clip_model_id=self.config['quality']['clip_model_id'])


        llm_endpoint = HuggingFaceEndpoint(
            repo_id=self.config['llm']['model_id'],
            temperature=self.config['llm']['temperature'],
            task="text-generation"
        )

        self.llm  = ChatHuggingFace(llm=llm_endpoint)
        self._initialized = True
        logger.info('LLM model has been initilized')
        logger.info('All models have been loaded.')

    def get(self, name):
        return getattr(self, name)
    

#from core.model_registry import ModelRegistry
# registry = ModelRegistry()
# registry.initialize(device="cuda")

# builder.add_node("local_editor", lambda state: local_editor_node(state, registry))
# builder.add_node("global_editor", lambda state: global_editor_node(state, registry))