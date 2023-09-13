import cv2
import easyocr
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


class ObjectDetection:
    def __init__(self, model_path, language='en', yolo_threshold=0.25, ocr_threshold=0.25):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader([language])
        self.yolo_threshold = yolo_threshold
        self.ocr_threshold = ocr_threshold
        self.class_colors = None  # Array to store colors for each class

    def detect_objects(self, image_path):

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

        img, bounding_boxes, _, _, classes = self.detect_objects(image_path)

        # If the class colors haven't been initialized, generate them
        if self.class_colors is None:
            unique_classes = np.unique(classes)
            self.class_colors = np.random.uniform(
                0, 255, size=(len(unique_classes), 3))

        for box, cls_id in zip(bounding_boxes, classes):
            start_point = box[0]
            end_point = box[2]

            color = self.class_colors[np.where(
                unique_classes == cls_id)[0][0]].tolist()
            img = cv2.rectangle(img, start_point, end_point, color, 2)

        return img

    def extract_text(self, image, bbox):
        x1, y1 = bbox[0]
        x3, y3 = bbox[2]

        # Get the region of interest (ROI) for the bounding box
        roi = image[y1:y3, x1:x3]

        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale ROI to get a binary image
        _, binary_roi = cv2.threshold(
            gray_roi, 128, 255, cv2.THRESH_BINARY_INV)

        # Use EasyOCR to extract text from the binary image
        results = self.reader.readtext(
            binary_roi, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        text = ' '.join(results)

        return text

    def process_bounding_boxes(self, image_path):

        image, bounding_boxes, _, conf, cls = self.detect_objects(image_path)
        detected_elements = []
        for bbox, conf, cls_name in zip(bounding_boxes, conf, cls):

            if conf > self.yolo_threshold:
                text = self.extract_text(image, bbox)
                element = {
                    'bbox': bbox,
                    'text': text,
                    'class': cls_name,
                    'confidence': conf
                }
                detected_elements.append(element)
                self.draw_bbox_with_text(image, bbox, text)

        return detected_elements

    def draw_bbox_with_text(self, image, bbox, text):
        # Convert the bounding box points to integer values
        bbox = [(int(x), int(y)) for x, y in bbox]

        # Draw the bounding box on the image
        x, y = bbox[0]
        x3, y3 = bbox[2]
        cv2.rectangle(image, (x, y), (x3, y3), (0, 255, 0), 2)

        # Calculate the text position and put it on top of the bounding box
        text_position = (x, y - 10)
        cv2.putText(image, text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# # Example usage
# model_path = 'best.pt'
# image_path = 'test.png'

# Initialize ObjectDetection class
# detector = ObjectDetection(model_path)

# Detect objects and get bounding boxes, confidence scores, and class indices
# img, bounding_boxes, xywh, conf, cls = detector.detect_objects(image_path)

# # Process the bounding boxes to extract text and draw on the image
# fu = detector.process_bounding_boxes(img, bounding_boxes, conf, cls)


# # Display the image with bounding boxes and extracted text
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()


class UIElementCluster:

    # Mapping of class numbers to human-readable class names
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
        self.detection_results = detection_results
        self.eps = eps
        self.clusters = self.cluster_by_position()

    @staticmethod
    def get_centroids(detection_results):
        centroids = []
        for element in detection_results:
            x_center = (element['bbox'][0][0] + element['bbox'][2][0]) / 2
            y_center = (element['bbox'][0][1] + element['bbox'][2][1]) / 2
            centroids.append([x_center, y_center])
        return centroids

    def cluster_by_position(self):
        centroids = self.get_centroids(self.detection_results)
        clustering = DBSCAN(eps=self.eps, min_samples=1).fit(centroids)

        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.detection_results[idx])
        return clusters

    def describe_cluster(self, cluster):
        descriptions = [
            f"{self.CLASS_MAP[element['class']]} {element['text']}" for element in cluster]
        return ', '.join(descriptions)

    def get_cluster_descriptions(self):
        return {label: self.describe_cluster(cluster) for label, cluster in self.clusters.items()}


# Create an instance and get the cluster descriptions
# cluster_instance = UIElementCluster(fu)
# cluster_descriptions = cluster_instance.get_cluster_descriptions()
# print(cluster_descriptions)
