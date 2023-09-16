import os
import cv2
import logging
import easyocr
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ObjectDetection:
    def __init__(self, model_path, language='en', yolo_threshold=0.25, ocr_threshold=0.25):
        """Initialize the object detection model and OCR reader."""
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader([language])
        self.yolo_threshold = yolo_threshold
        self.ocr_threshold = ocr_threshold
        self.class_colors = None

    def detect_objects(self, image_path):
        """Detect objects in the given image."""
        img = cv2.imread(image_path)
        results = self.model(image_path)
        
        for result in results:
            boxes = result.boxes
            names = result.names

        xywh = boxes.xywh.tolist()
        conf = boxes.conf.tolist()
        cls = boxes.cls.tolist()
        
        bbox_preds = [[xywh[i][0], xywh[i][1], xywh[i][2], xywh[i]
                       [3], conf[i], names[int(cls[i])]] for i in range(len(xywh))]
        bounding_boxes = [((int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)),
                           (int(box[0] + box[2] / 2),
                            int(box[1] - box[3] / 2)),
                           (int(box[0] + box[2] / 2),
                            int(box[1] + box[3] / 2)),
                           (int(box[0] - box[2] / 2), int(box[1] + box[3] / 2))) for box in bbox_preds]
        
        return img, bounding_boxes, xywh, conf, cls

    def draw_bounding_boxes(self, image_path):
        """Draw bounding boxes on detected objects."""
        img, bounding_boxes, _, _, classes = self.detect_objects(image_path)

        if self.class_colors is None:
            unique_classes = np.unique(classes)
            self.class_colors = np.random.uniform(0, 255, size=(len(unique_classes), 3))

        for box, cls_id in zip(bounding_boxes, classes):
            start_point = box[0]
            end_point = box[2]
            color = self.class_colors[np.where(unique_classes == cls_id)[0][0]].tolist()
            img = cv2.rectangle(img, start_point, end_point, color, 2)

        return img

    def extract_text(self, image, bbox):
        """Extract text from the specified bounding box in the image."""
        x1, y1 = bbox[0]
        x3, y3 = bbox[2]
        roi = image[y1:y3, x1:x3]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
        results = self.reader.readtext(binary_roi, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        text = ' '.join(results)
        
        return text

    def process_bounding_boxes(self, image_path):
        """Process detected bounding boxes and extract related information."""
        image, bounding_boxes, _, conf, cls = self.detect_objects(image_path)
        detected_elements = []
        for bbox, conf, cls_name in zip(bounding_boxes, conf, cls):
            if conf > self.yolo_threshold:
                text = self.extract_text(image, bbox)
                element = {'bbox': bbox, 'text': text, 'class': cls_name, 'confidence': conf}
                detected_elements.append(element)
                self.draw_bbox_with_text(image, bbox, text)
        
        return detected_elements

    def draw_bbox_with_text(self, image, bbox, text):
        """Draw bounding box and associated text on the image."""
        bbox = [(int(x), int(y)) for x, y in bbox]
        x, y = bbox[0]
        x3, y3 = bbox[2]
        cv2.rectangle(image, (x, y), (x3, y3), (0, 255, 0), 2)
        text_position = (x, y - 10)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

class UIElementCluster:
    CLASS_MAP = {
        0.0: "button",
        1.0: "field",
        2.0: "heading",
        3.0: "iframe",
        4.0: "image",
        5.0: "label",
        6.0: "link",
        7.0: "text"
    }

    def __init__(self, detection_results, eps=50):
        """Initialize clusters using detected results."""
        self.detection_results = detection_results
        self.eps = eps
        self.clusters = self.cluster_by_position()

    @staticmethod
    def get_centroids(detection_results):
        """Calculate centroids for detection results."""
        centroids = []
        for element in detection_results:
            x_center = (element['bbox'][0][0] + element['bbox'][2][0]) / 2
            y_center = (element['bbox'][0][1] + element['bbox'][2][1]) / 2
            centroids.append([x_center, y_center])
        
        return centroids

    def cluster_by_position(self):
        """Cluster detection results by position."""
        centroids = self.get_centroids(self.detection_results)
        clustering = DBSCAN(eps=self.eps, min_samples=1).fit(centroids)
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.detection_results[idx])
        
        return clusters

    def describe_cluster(self, cluster):
        """Describe the cluster using class names."""
        descriptions = [f"{self.CLASS_MAP[element['class']]} {element['text']}" for element in cluster]
        return ', '.join(descriptions)

    def get_cluster_descriptions(self):
        """Get descriptions for all clusters."""
        return {label: self.describe_cluster(cluster) for label, cluster in self.clusters.items()}
