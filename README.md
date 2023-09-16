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
   git clone [repository-link]
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
   echo "OPENAI_API_KEY=your_api_key_here" > .env
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
