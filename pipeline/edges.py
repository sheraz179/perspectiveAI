import numpy as np
import cv2
from controlnet_aux import MLSDdetector
from sklearn.cluster import KMeans
from PIL import Image

class StructuralLineDetector:
    def __init__(self):
        self.detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    def group_orientations_kmeans(self, lines, n_clusters=3):

        #Manhattan world assumption 3 clusters(vertical, horizontal, depth)
        angles = np.array([self.line_orientation(line) for line in lines]).reshape(-1,1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(angles)
        groups = {}
        for idx, label in enumerate(kmeans.labels_):
            groups.setdefault(label, []).append(lines[idx])
        return groups, kmeans.cluster_centers_

    def line_orientation(self, line):

        x1, y1, x2, y2 = line
        angle = np.arctan2((y2 - y1), (x2 - x1))  # radians
        angle_deg = np.degrees(angle)
        if angle_deg < 0:
            angle_deg += 180
        return angle_deg

    def extract_lines_from_mlsd(self, mlsd_map):
        """
        mlsd_map: MLSD output as numpy array (binary / grayscale)
        returns: list of [x1, y1, x2, y2]
        """

        # Convert PIL Image → NumPy array
        if isinstance(mlsd_map, Image.Image):
            mlsd_map = np.array(mlsd_map)

        if len(mlsd_map.shape) == 3:
            mlsd_map = cv2.cvtColor(mlsd_map, cv2.COLOR_RGB2GRAY)
        
        # Threshold to binary
        _, binary = cv2.threshold(mlsd_map, 127, 255, cv2.THRESH_BINARY)
        
        # Use HoughLinesP to get line segments
        lines_p = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=20,
            maxLineGap=1
        )
        
        # Convert to list of [x1,y1,x2,y2]
        lines = []
        if lines_p is not None:
            for l in lines_p:
                lines.append(l[0].tolist())
        
        return lines

    def extract_lines(self, detected_map):
        return self.extract_lines_from_mlsd(detected_map)

    def compute(self, image, lines_out_image_path):
        """
        image: numpy array (BGR OpenCV format)
        returns: structural line map (numpy array)
        """

        # Convert BGR (OpenCV) → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run MLSD
        detected_map = self.detector(image_rgb)
        w,h = image_rgb.shape[:2]
        detected_map = detected_map.resize((h, w))
        #print(detected_map.size, 'shape')
        cv2.imwrite(lines_out_image_path, np.array(detected_map))
        lines = self.extract_lines(detected_map)
        # Convert back to numpy array
        #detected_map = np.array(detected_map)

        return lines #detected_map

    def scaling_groups_to_original_shape(self, groups, original_shape):

        orig_h, orig_w = original_shape #original_image.shape[:2]  # original image
        mlsd_w, mlsd_h = 896, 512  # detected MLSD image size

        scale_x = orig_w / mlsd_w
        scale_y = orig_h / mlsd_h

        # Scale grouped lines
        groups_scaled = {}
        for idx, group in groups.items():
            scaled_group = []
            for x1, y1, x2, y2 in group:
                scaled_group.append([
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ])
            groups_scaled[idx] = scaled_group
        return groups_scaled

    def visualize_groups(self, image, output_image_path, grouped_lines):

        #grouped_lines = self.scaling_groups_to_original_shape(grouped_lines, image.shape[:2])
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        img = image.copy()
        for idx, group in enumerate(grouped_lines.values()):
            color = colors[idx % len(colors)]
            for x1, y1, x2, y2 in group:
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
          
        cv2.imwrite(output_image_path, img)
        return img
