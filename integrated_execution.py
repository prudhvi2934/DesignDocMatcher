
import logging
from testCaseGen import TestCaseRetriever
from webUIObjectDetection import ObjectDetection, UIElementCluster
from integrate import EmbeddingComparator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


def main(requirements_pdf, website_screenshot, model_path, thresold):
    logger.info("Starting the integrated process...")

    # Step 1: Test Case Generation
    logger.info("Generating test cases from the requirements document...")
    testcase_gen = TestCaseRetriever([requirements_pdf])
    user_input = "what are the web UI test cases to check the design of the Home Page"
    test_cases = testcase_gen.retrieve_test_cases(user_input)
    print(type(test_cases))
    # Step 2: Web UI Element Detection
    logger.info("Detecting and extracting text from the web UI image...")
    obj_detector = ObjectDetection(model_path)
    # Detect objects and get bounding boxes, confidence scores, and class indices
    # img, bounding_boxes, xywh, conf, cls = obj_detector.detect_objects(
    #     website_screenshot)

   # Process the bounding boxes to extract text and draw on the image
    detections = obj_detector.process_bounding_boxes(website_screenshot)
    ui_cluster = UIElementCluster(detections)
    cluster_descriptions = ui_cluster.get_cluster_descriptions()

    # Step 3: Semantic Validation
    logger.info(
        "Semantically validating the detected UI elements against test cases...")
    comparator = EmbeddingComparator()
    results = comparator.validate_test_cases_semantically(
        test_cases, cluster_descriptions, thresold)

    # logger.info("Process completed.")
    print(results)
    return results


if __name__ == "__main__":
    # Modify these paths as per your setup
    requirements_pdf = "Ripple_tocuch.pdf"
    website_screenshot = "Front cover.png"
    model_path = 'Yolov8_weights/best.pt'
    thresold = 0.8
    main(requirements_pdf, website_screenshot, model_path, thresold)
