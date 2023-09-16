import os
import logging
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class TestCaseRetriever:
    def __init__(self, pdf_docs):
        self.stored_results = []
        self.pdf_docs = pdf_docs

    def get_pdf_text(self):
        """
        Extracts and returns the combined text content from all the PDF documents
        in the pdf_docs attribute of the object.

        Returns:
        text (str): A concatenated string containing the text content from all 
        the PDF pages in the pdf_docs.
        """

        text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

        return text

    def get_text_chunks(self, text):
        """
        Splits the given text into overlapping chunks of a defined size using 
        CharacterTextSplitter.

        Args:
            text (str): Text to split.

        Returns:
            list: Chunks of split text.
        """

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        return chunks

    def get_vectorstore(self, text_chunks):
        """
        Converts a list of text chunks into a vector store using 
        OpenAIEmbeddings and FAISS.

        Args:
            text_chunks (list): List of text segments.

        Returns:
            vectorstore: FAISS vector storage representation of the
            text embeddings.
        """

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        return vectorstore

    def get_conversation_chain(self, vectorstore):
        """
        Constructs a conversational retrieval chain using OpenAI chat model,
        a vector store retriever, and a buffer memory.

        Args:
            vectorstore: FAISS vector storage representation of text embeddings.

        Returns:
            conversation_chain: Conversational retrieval mechanism for engaging
            with the ChatOpenAI model.
        """

        llm = ChatOpenAI()

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
        """
        Retrieves relevant test cases from PDF content based on user input using 
        a conversational retrieval mechanism.

        Args:
            user_input (str): User's query to retrieve relevant test cases.

        Returns:
            list: A list of relevant test cases extracted from the retrieval process.
        """

        text = self.get_pdf_text()
        text_chunks = self.get_text_chunks(text)
        vectorstore = self.get_vectorstore(text_chunks)
        conversation_chain = self.get_conversation_chain(vectorstore)
        retrieved_chunks = conversation_chain(user_input)
        ai_message_content = retrieved_chunks['answer']
        test_cases = ai_message_content.split(':\n\n')[1]
        pattern = re.compile(r'\n\n[A-Za-z].*$', re.S)
        modified_tcs = re.sub(pattern, '', test_cases)
        text = self.extract_as_list(modified_tcs)
        self.stored_results.extend(text)

        return text

    def extract_as_list(self, text):
        """
        Extracts list items from the provided text based on specific number
        and hyphen patterns.

        Args:
            text (str): The input text containing items to extract.

        Returns:
            list: A list of extracted items without trailing colons.
        """

        pattern1 = re.compile(
            r'^\d+\.\s(.*?)(?=\n\d+\.|\n\n\d+\.|$)', re.S | re.M)
        pattern2 = re.compile(r'^-\s(.*?)(?=\n-|\n\n-|$)', re.S | re.M)

        points = []

        if pattern1.search(text):
            points.extend([match.strip() for match in pattern1.findall(
                text) if not match.strip().endswith(':')])
        if "\n\n" in text:
            points.extend([match.strip() for match in pattern2.findall(
                text) if not match.strip().endswith(':')])

        return points

    def get_stored_results(self):
        return self.stored_results


class KeywordExtractor:

    def __init__(self):

        self.keywords = {
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

        for element_type, keywords in self.keywords.items():
            if any(keyword in test_case.lower() for keyword in keywords):
                extracted_elements.add(element_type)

        return extracted_elements


class PagesExtractor:

    def get_pages_from_pdf(pdf_path, section):
        """
        Retrieves content from a specified section of a PDF using a Q&A
        retrieval chain.

        Args:
            pdf_path (str): Path to the target PDF file.
            section (str): The name/identifier of the section in the PDF
            from which to retrieve content.

        Returns:
            list: A list of items extracted from the specified section of the PDF.
        """

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        output_parser = CommaSeparatedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        prompt = PromptTemplate(
            template="what are there in section {subject}.\n{format_instructions}",
            input_variables=["subject"],
            partial_variables={"format_instructions": format_instructions}
        )
        input = prompt.format(subject=section)
        chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")

        chain_output = chain.run(input_documents=documents, question=input)
        items = output_parser.parse(chain_output)

        return items
