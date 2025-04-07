from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_experimental.text_splitter import SemanticChunker 
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import os
from pydantic import BaseModel
import streamlit as st 
from typing import List
#Pydantic model
class Assessment(BaseModel):
    assessment_name: str
    url: str
    remote_testing: str
    adaptive_irt_support: str
    test_type: str
    time_duration:int

class AssessmentList(BaseModel):
    assessments:List[Assessment]    
load_dotenv()

st.title("SHL ASSIGNMENT")
user_query=st.text_input("Ask a question (eg. test under 50 minutes )")

if user_query: 
    with st.spinner("Processing  "):
        loader1 = CSVLoader(file_path='./shl_individual_updated.csv')
        loader2 = CSVLoader(file_path='./shl_prepackaged_updated.csv')

        doc1 = loader1.load()
        doc2 = loader2.load()

        combined_text = "\n\n".join([d.page_content for d in (doc1 + doc2)])
        doc = Document(page_content=combined_text)

        #embedding model semantic chunker
        embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        chunker = SemanticChunker(embedding_model)

        # Split 
        chunks = chunker.split_documents([doc])

        # Build chunks
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        retriever = vectorstore.as_retriever()

        # LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"
        )

        parser = PydanticOutputParser(pydantic_object=AssessmentList)

        prompt = PromptTemplate(
            template="""
                    
                        You are an SHL assessment expert. Given the user query, recommend up to 10 relevant SHL assessments.

                        Only return this JSON between triple backticks:
                        {format_instructions}

                        User Query:
                        {user_query}
                        

                    """,
            input_variables=["user_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
                )   

        try:
            formatted_prompt = prompt.format(user_query=user_query)
            response = qa_chain.invoke(formatted_prompt)  # ✅ This must be a plain string!
            st.subheader("🔍 Raw Model Output")
            st.code(response["result"], language="json")
            validated = parser.parse(response['result'].strip())

            st.success("Recommended Assessments:")
            for i, assess in enumerate(validated.assessments, start=1):
                with st.expander(f"📘 Assessment {i}: {assess.assessment_name}"):
                    st.write(f"**URL**: [{assess.url}]({assess.url})")
                    st.write(f"**Remote Testing**: {assess.remote_testing}")
                    st.write(f"**Adaptive/IRT Support**: {assess.adaptive_irt_support}")
                    st.write(f"**Test Type**: {assess.test_type}")
                    st.write(f"**Time Duration**: {assess.time_duration} minutes")

        except Exception as e:
             st.error(f"❌ Error: {e}")