import streamlit as st
from PIL import Image
from testCaseGen import TestCaseRetriever
from webUIObjectDetection import ObjectDetection, UIElementCluster
from integrate import EmbeddingComparator
from langchain import OpenAI, LLMChain
from tes2 import get_pages_from_pdf
from PyPDF2 import PdfReader
from io import BytesIO
import tempfile
import os
import cv2

# Importing the necessary methods from your scripts
# from integrate import extract_text_from_pdf, get_ui_elements_from_image, compare_test_cases_with_ui

# Streamlit App
import streamlit as st
from PIL import Image
# from integrated_execution import TestCaseRetriever

# Streamlit App
st.title("UI Test Case Extractor")

retriever = None
# uploaded_pdf_content = None  # Initialize the variable


# Initial upload
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

# If PDF is uploaded, save its content to session state
if uploaded_pdf:
    st.session_state.uploaded_pdf_content = uploaded_pdf.read()

# Check if PDF content exists in session state
if "uploaded_pdf_content" in st.session_state:
    # Use a button to proceed after PDF is uploaded
    if "proceed_clicked" not in st.session_state or not st.session_state.proceed_clicked:
        if st.button("Proceed"):
            st.session_state.proceed_clicked = True
    if st.session_state.get("proceed_clicked"):
        # Save the uploaded file's content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(st.session_state.uploaded_pdf_content)
            temp_file_path = tmpfile.name

        pages = get_pages_from_pdf(temp_file_path, 'Website Structure')
        retriever = TestCaseRetriever([temp_file_path])

        selected_page = st.selectbox("Select a page:", [""] + pages)
        if selected_page:
            uploaded_screenshot = st.file_uploader(
                f"Upload a screenshot for {selected_page}", type=["png", "jpg", "jpeg"])
            if uploaded_screenshot:
                # Process the uploaded screenshot here
                st.image(
                    uploaded_screenshot, caption=f"Screenshot for {selected_page}.", use_column_width=True)
    # # 3. Upload a screenshot and detect UI elements using YOLO.
    # uploaded_image = st.file_uploader(
    #     "Upload a screenshot of the page", type=["png", "jpg", "jpeg"])

                if uploaded_screenshot:
                    # Extract the extension from the uploaded file's name
                    file_extension = os.path.splitext(
                        uploaded_screenshot.name)[1]

                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as img_tmpfile:
                        img_tmpfile.write(uploaded_screenshot.read())
                        temp_image_path = img_tmpfile.name  # Store the path for later use

                    obj_detector = ObjectDetection('Yolov8_weights/best.pt')
                    # Get the image with bounding boxes drawn
                    img_with_boxes = obj_detector.draw_bounding_boxes(
                        temp_image_path)

                    # Convert to RGB and display in Streamlit
                    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, channels="RGB", use_column_width=True)

                    user_input = f"I need visual UI test cases for {selected_page}"
                    test_cases = retriever.retrieve_test_cases(user_input)
                    st.write("Test Cases:")
                    st.write(test_cases)

                    # Process the bounding boxes to extract text and draw on the image

                    detections = obj_detector.process_bounding_boxes(
                        temp_image_path)
                    ui_cluster = UIElementCluster(detections)
                    cluster_descriptions = ui_cluster.get_cluster_descriptions()

                    comparator = EmbeddingComparator()
                    results = comparator.validate_test_cases_semantically(
                        test_cases, cluster_descriptions, 0.8)
                    st.write(results)
                    # # Detect objects and get bounding boxes, confidence scores, and class indices
                    # img, bounding_boxes, xywh, conf, cls = obj_detector.detect_objects(
                    #     temp_image_path)
                    # # ui_elements = get_ui_elements_from_image(image)

                    # # Process the bounding boxes to extract text and draw on the image
                    # detections = obj_detector.process_bounding_boxes(
                    #     img, bounding_boxes, conf, cls)
                    # ui_cluster = UIElementCluster(detections)
                    # cluster_descriptions = ui_cluster.get_cluster_descriptions()

            # 4. Display the detected UI elements and allow users to add additional test cases.
            # Logic to display and add test cases

            # # 5. Display the comparison results of the test cases against the detected UI elements.
            # comparator = EmbeddingComparator()
            # results = comparator.validate_test_cases_semantically(
            #     test_cases, cluster_descriptions, 0.5)
            # # results = compare_test_cases_with_ui(test_cases, ui_elements)
            # st.write(results)
