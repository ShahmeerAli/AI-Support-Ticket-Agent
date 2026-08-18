import os
import requests
import uuid
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings,ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

endpoint=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.2,  
)

model=ChatHuggingFace(
    llm=endpoint
)


#class to verify the confidence if its less than 8 contact helpline
class ConfidenceState(TypedDict):
    confidence:int

class AIBOTState(TypedDict):
    message:str
    ragResponse:str
    

def RAGNode():
    #Loading the document
    loader=PyPDFLoader(
        file_path="DOC.pdf"
    )
    docs=loader.load()
    #Splitting the Document content
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    splitter_docs=splitter.split_documents(docs)
    #Embeddings creation
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    #Now for the Embeddings to be stored in the VectorStore Chroma
    vectorSore=Chroma.from_documents(
        splitter_docs,embeddings
    )
    #Similarity Search-----
    retriever_docs=vectorSore.as_retriever(
        search_type='similarity',kwargs={'k':3}
    )
    

