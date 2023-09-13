import os
import PyPDF2
import re

from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, ListOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List


load_dotenv()


def new_test(pdf_path, section):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    class page_tc(BaseModel):
        page: str = Field(description="name of the web page")
        test_cases: List[str] = Field(
            description="list of web UI test cases for the page")
        # Set up a parser + inject instructions into the prompt template.

    pydantic_parser = PydanticOutputParser(pydantic_object=page_tc)

    format_instructions = pydantic_parser.get_format_instructions()

    # template_string = """You are a QA test testing only web UI design, not its functionality. \

    # Take the name of the page below delimited by triple backticks and use it to create a list of test cases.

    # page: ```{page}```
    # """
    question = "what are the web UI design test cases for the {page}.\n{format_instructions}"
    prompt = ChatPromptTemplate.from_template(template=question)

    messages = prompt.format_messages(page=section,
                                      format_instructions=format_instructions)
    print(messages)
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff", prompt=messages)
    s = chain.run(input_documents=documents, question=messages)
    print(s)
    # items = pydantic_parser.parse(s)
    # print(items)
    # return items


def get_pages_from_pdf(pdf_path, section):
    # load document
    load_dotenv()
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="what are there in section {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )
    _input = prompt.format(subject=section)
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")

    s = chain.run(input_documents=documents, question=_input)
    items = output_parser.parse(s)
    # print(items)
    return items


def get_testCases_from_pdf(pdf_path, section):
    # load document
    load_dotenv()
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="What will be web design testcases for the {subject}.\n{format_instructions}",
        input_variables=["subject"],
        partial_variables={"format_instructions": format_instructions}
    )
    _input = prompt.format(subject=section)
    chain = load_qa_chain(llm=OpenAI(), chain_type='map_reduce')

    s = chain.run(input_documents=documents, question=_input)
    items = output_parser.parse(s)
    print(s)
    print(len(items))
    # return items


def get_subsection(pdf_path, section):
    # load document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
    query = f"Extract the text under the subsection titled 'Design the {section}:?"
    x = chain.run(input_documents=documents, question=query)
    return x


def extract_points_from_pdf(pdf_path, section_title):
    # Extract text from the PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        full_text = ''.join([reader.pages[i].extract_text()
                            for i in range(len(reader.pages))])

    # Regular expression pattern to match the specific section and its associated points
    pattern = re.compile(
        rf'{section_title}:[\s\S]*?([\s\S]*?)(?=\d+\.\s*\w+ Page:|$)', re.M)
    match = pattern.search(full_text)
    if not match:
        return []

    section_content = match.group(1)
    points = [point.strip('').strip() for point in section_content.split(
        '\n') if point.strip().startswith('')]

    return points


# Example usage:
pdf_path = "Ripple_tocuch.pdf"
section_title = "Home Page"
points = extract_points_from_pdf(pdf_path, section_title)
print(points)


# Usage
# subsection = get_subsection(
#     "Ripple_tocuch.pdf", "Case Studies")
# print(subsection.split('\n'))


# s = "The pages are: Home Page, About Us, Our Services, Contact, and Case Studies."
# page_list = convert_to_list(s)
# print(page_list)
# get_testCases_from_pdf(
#     'Ripple_tocuch.pdf', 'Contact Page')
# print(get_pages_from_pdf(
#     'Ripple_tocuch.pdf', 'Website Structure'))

# new_test('Ripple_tocuch.pdf', 'Contact Page')
