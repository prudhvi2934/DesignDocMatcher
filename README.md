# DesignDocMatcher

# Web UI Testing Automation Tool

A comprehensive tool that integrates test case generation, web UI element detection, and semantic validation.

## Features

- **Test Case Generation**: Extracts and generates test cases from provided requirement documents.
- **Web UI Element Detection**: Uses advanced object detection models to identify and classify elements in a website screenshot.
- **Semantic Validation**: Compares the detected UI elements semantically against the test cases to ensure design consistency.

## Prerequisites

1. Python 3.x
2. Dependencies from `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/prudhvi2934/DesignDocMatcher.git
   ```

2. Navigate to the project directory:

   ```bash
   cd [directory-name]
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Place your requirement document in the root directory.
2. Take a screenshot of the web UI you want to test and place it in the root directory.
3. If using a custom model for object detection, place it in the respective directory.

You can execute the program using either of the following commands:

For the Streamlit app:

```bash
streamlit run streamlit_app.py

```
For the integrated execution:

```bash
python integrated_execution.py
```

## Configuration
You can configure the thresholds and model paths in the respective execution files.

## Contributing
1. Fork the repository.
2. Create a new branch.
3. Commit your changes and push to your branch.
4. Create a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- OpenAI for the semantic comparison tool.
- YOLO for the object detection model.
