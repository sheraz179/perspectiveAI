import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import cv2

from core.logger import logger

class YOLOObjectoxDetector:
    def __init__(self, model_id, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_id)
        self.model.to(self.device)

        self.class_names = self.model.names  # {id: name}

    def _filter_class_ids(self, target_classes):
        """
        Convert class names to class IDs
        """
        target_classes = [c.lower() for c in target_classes]
        class_ids = []

        for class_id, name in self.class_names.items():
            if name.lower() in target_classes:
                class_ids.append(class_id)

        return class_ids

    def get_bounding_boxes(self, image, target_classes, conf_threshold=0.25):
        """
        image: PIL image or numpy image
        target_classes: list of class names to keep
        returns: list of filtered bounding boxes [x1,y1,x2,y2]
        """

        results = self.model(image, verbose=False)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy()

        target_ids = self._filter_class_ids(target_classes)

        filtered_boxes = []
        filtered_labels=[]

        for bbox, score, cls in zip(boxes, scores, labels):
            if score < conf_threshold:
                continue

            if int(cls) in target_ids:
                filtered_boxes.append(list(map(int, bbox)))
                filtered_labels.append(self.class_names[int(cls)])

        logger.info(f'Have detected labels {filtered_labels} with room decor detector')
        return filtered_boxes, filtered_labels
