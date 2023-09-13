import os
import logging
import re

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class TestCaseRetriever:
    def __init__(self, pdf_docs):
        self.stored_results = []
        self.pdf_docs = pdf_docs

    def get_pdf_text(self):
        text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(self, text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(self, text_chunks):
        embeddings = OpenAIEmbeddings()
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(self, vectorstore):

        llm = ChatOpenAI()
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            chain_type="stuff"
        )

        return conversation_chain

    def retrieve_test_cases(self, user_input):
        # Using the conversation chain to retrieve relevant chunks based on user input
        text = self.get_pdf_text()
        text_chunks = self.get_text_chunks(text)
        vectorstore = self.get_vectorstore(text_chunks)
        conversation_chain = self.get_conversation_chain(vectorstore)
        retrieved_chunks = conversation_chain(user_input)
        # Assuming retrieved chunks are test cases
        # print(retrieved_chunks)
        ai_message_content = retrieved_chunks['answer']
        # print(ai_message_content)
        test_cases = ai_message_content.split(':\n\n')[1]
        pattern = re.compile(r'\n\n[A-Za-z].*$', re.S)
        # Remove matched content
        modified_tcs = re.sub(pattern, '', test_cases)
        print(modified_tcs)
        text = self.adaptive_extract_as_list_v2(modified_tcs)
        print(text)
        # Storing the results
        self.stored_results.extend(text)

        return text

    def adaptive_extract_as_list_v2(self, text):
        """
        Adaptively extracts points from the given text based on its format and returns them as a list.
        This version ensures that section headers are not included in the result.
        """
        # Define patterns for two types of points
        pattern1 = re.compile(
            r'^\d+\.\s(.*?)(?=\n\d+\.|\n\n\d+\.|$)', re.S | re.M)
        pattern2 = re.compile(r'^-\s(.*?)(?=\n-|\n\n-|$)', re.S | re.M)

        points = []

        # Extract points based on the patterns found in the text
        if pattern1.search(text):
            points.extend([match.strip() for match in pattern1.findall(
                text) if not match.strip().endswith(':')])
        if "\n\n" in text:  # If the text contains dashed points
            points.extend([match.strip() for match in pattern2.findall(
                text) if not match.strip().endswith(':')])

        return points


def get_stored_results(self):
    return self.stored_results


class ChatBotInterface:
    def __init__(self, test_case_manager):
        self.test_case_manager = test_case_manager

    def start_chat(self):
        while True:
            # Get user input
            user_input = input("You: ")

            # If the user wants to exit the chat
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            response = self.test_case_manager.retrieve_test_cases(
                user_input)
            # print(f"Bot: {response['answer']}\\n")

            if "test cases to check the webui elements" in user_input.lower():
                ai_message_content = response['answer']
                # test_cases = "\n".join([line for line in ai_message_content.split(
                #     '\n') if line.strip() and line[0].isdigit()])
                return ai_message_content


class KeywordExtractor:

    def __init__(self):
        # Dynamically loaded keywords related to UI elements.
        # This is just a sample. In real implementation, this can be loaded dynamically or updated as needed.
        self.dynamic_keywords = {
            'button': ['click', 'press', 'navigate', 'tab', 'link'],
            'text': ['information', 'description', 'tagline', 'section', 'content', 'details', 'overview'],
            'image': ['picture', 'photo', 'banner', 'background', 'icon', 'logo', 'thumbnail'],
            'field': ['input', 'type', 'fill', 'submit', 'form', 'textbox'],
            'iframe': ['embed', 'video', 'widget', 'map', 'third-party', 'external content'],
            'link': ['hyperlink', 'url', 'clickable'],
            'label': ['indicator', 'signifier', 'denote'],
            'heading': ['title', 'header', 'headline', 'subheading']
        }

    def process_test_cases(self, test_cases_str):
        """
        Process multiple test cases to extract associated web UI elements.
        """
        test_cases = [line.strip() for line in test_cases_str.split(
            '\n') if line.strip() and line[0].isdigit()]
        test_case_to_elements = {}

        print(test_cases)
        for test_case in test_cases:
            elements = self.extract_keywords_from_test_case(test_case)
            test_case_to_elements[test_case] = elements

        return test_case_to_elements

    def extract_keywords_from_test_case(self, test_case):
        """
        Extract keywords related to UI element types from a given test case.
        """
        extracted_elements = set()

        for element_type, keywords in self.dynamic_keywords.items():
            if any(keyword in test_case.lower() for keyword in keywords):
                extracted_elements.add(element_type)

        return extracted_elements


# # Example of usage:
# Initialize the TestCaseRetriever with a list of PDFs
# retriever = TestCaseRetriever(["Ripple_tocuch.pdf"])
# # # User provides input requesting specific test cases
# user_input = "what are the web UI test cases to check the design of the Home Page"
# test_cases = retriever.retrieve_test_cases(user_input)
# print(test_cases)
# Starting the chatbot interaction using the new class
# retriever = TestCaseRetriever(["Lighthouse_Law_AI_generated_spec.pdf"])
# chatbot = ChatBotInterface(retriever)
# sample_test_case = chatbot.start_chat()
# print(sample_test_case)


# Sample Usage:
# extractor = KeywordExtractor()
# sample_test_case = "Test case: Verify that the banner section on the home page contains a background image."
# logger.warning("Using hardcoded test case.")
# extracted_keywords = extractor.process_test_cases(sample_test_case)
# print(extracted_keywords)
