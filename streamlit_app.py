import tempfile
import os
import cv2
import streamlit as st
from testCaseGen import TestCaseRetriever, PagesExtractor
from webUIObjectDetection import ObjectDetection, UIElementCluster
from integrate import EmbeddingComparator



# Streamlit App Setup
st.title("UI Test Case Extractor")

# Initial upload
uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

# If PDF is uploaded, save its content to session state
if uploaded_pdf:
    st.session_state.uploaded_pdf_content = uploaded_pdf.read()

# Check if PDF content exists in session state and manage the workflow
if "uploaded_pdf_content" in st.session_state:
    if "proceed_clicked" not in st.session_state or not st.session_state.proceed_clicked:
        if st.button("Proceed"):
            st.session_state.proceed_clicked = True

    if st.session_state.get("proceed_clicked"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(st.session_state.uploaded_pdf_content)
            temp_file_path = tmpfile.name

        pages = PagesExtractor.get_pages_from_pdf(temp_file_path, 'Website Structure')
        retriever = TestCaseRetriever([temp_file_path])
        selected_page = st.selectbox("Select a page:", [""] + pages)

        if selected_page:
            uploaded_screenshot = st.file_uploader(f"Upload a screenshot for {selected_page}", type=["png", "jpg", "jpeg"])

            if uploaded_screenshot:
                st.image(uploaded_screenshot, caption=f"Screenshot for {selected_page}.", use_column_width=True)
                file_extension = os.path.splitext(uploaded_screenshot.name)[1]

                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as img_tmpfile:
                    img_tmpfile.write(uploaded_screenshot.read())
                    temp_image_path = img_tmpfile.name

                obj_detector = ObjectDetection('Yolov8_weights/best.pt')
                img_with_boxes = obj_detector.draw_bounding_boxes(temp_image_path)
                img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, channels="RGB", use_column_width=True)

                user_input = f"I need visual UI test cases for {selected_page}"
                test_cases = retriever.retrieve_test_cases(user_input)
                st.write("Test Cases:")
                st.write(test_cases)

                detections = obj_detector.process_bounding_boxes(temp_image_path)
                ui_cluster = UIElementCluster(detections)
                cluster_descriptions = ui_cluster.get_cluster_descriptions()

                comparator = EmbeddingComparator()
                results = comparator.validate_test_cases_semantically(test_cases, cluster_descriptions, 0.8)
                st.write(results)
