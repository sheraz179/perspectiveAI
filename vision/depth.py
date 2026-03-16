import torch
import torch.nn.functional as F
import cv2
import numpy as np


class MiDaSDepthEstimator:
    def __init__(self, model_type="MiDaS_Hybrid", device=None):
        """
        model_type:
            - "MiDaS_small" (fast, lighter)
            - "DPT_Large" (higher quality, slower)
            - "DPT_Hybrid"
        """

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, image_bgr):
        """
        image_bgr: numpy array (OpenCV format)
        returns:
            depth_raw: raw depth values (float32)
            depth_norm: normalized depth [0,1]
        """

        # Convert BGR -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Transform
        input_batch = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_raw = prediction.cpu().numpy()

        # Normalize depth to [0,1]
        depth_norm = (depth_raw - depth_raw.min()) / (
            depth_raw.max() - depth_raw.min() + 1e-8
        )

        return depth_raw, depth_norm

    def save_visualization(self, depth_norm, output_path):
        """
        Save normalized depth map as 8-bit image
        """
        depth_vis = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite(output_path, depth_vis)
