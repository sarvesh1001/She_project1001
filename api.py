from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import os

load_dotenv()

app = FastAPI(title="SHL Assessment API")

# Pydantic models
class Assessment(BaseModel):
    assessment_name: str
    url: str
    remote_testing: str
    adaptive_irt_support: str
    test_type: str
    time_duration: int

class AssessmentList(BaseModel):
    assessments: List[Assessment]

# Load and prepare once at startup
loader1 = CSVLoader(file_path="./shl_individual_updated.csv")
loader2 = CSVLoader(file_path="./shl_prepackaged_updated.csv")

doc1 = loader1.load()
doc2 = loader2.load()

combined_text = "\n\n".join([d.page_content for d in (doc1 + doc2)])
doc = Document(page_content=combined_text)

embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
chunker = SemanticChunker(embedding_model)
chunks = chunker.split_documents([doc])
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
retriever = vectorstore.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
parser = PydanticOutputParser(pydantic_object=AssessmentList)

prompt = PromptTemplate(
    template="""
    You are an SHL assessment expert.

    Based on the user query, recommend up to 10 relevant SHL assessments from the given context.
    Return only the result in the following JSON format between triple backticks:

    {format_instructions}

    User Query:
    {user_query}

    Only return the JSON inside triple backticks. No explanations.
    """,
    input_variables=["user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

@app.get("/recommend", response_model=AssessmentList)
def recommend_assessments(query: str = Query(..., description="Query about SHL assessments")):
    formatted_prompt = prompt.format(user_query=query)
    response = qa_chain.invoke(formatted_prompt)
    return parser.parse(response["result"].strip())
