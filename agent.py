import os
import requests
import uuid
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings,ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser

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


#class to verify the confidence if its less than 8 contact helpline and the main node
class AIBOTState(TypedDict):
    message:str
    ragResponse:str
    confidence:int
    humanResponse:bool
    

def RAGNode(state:AIBOTState):
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
    retriever_docs=vectorSore.similarity_search_with_relevance_scores(
        search_type='similarity',search_kwargs={'k':3}
    )

    documents = [
        doc for doc, score in retriever_docs
    ]

    # -----------------------------
    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt=ChatPromptTemplate.from_template(
        """
        Answer the Question only from the provided context.
        Question: {question}
        Context:  {context}
        If the context is not enough you simply say contact helpline.
        """
    )

    parser=StrOutputParser()
    
    chain=prompt | model|parser
    output=chain.invoke({
        "context":context,
        "question":state['message']
    })
    return {'ragResponse':output}



def ConfidenceScore(state:AIBOTState):
    prompt=ChatPromptTemplate.from_template(
        """
        Evaluate the following answer.
        Question:
        {question}
        Answer:
        {answer}
        Determine how well the answer is supported by the available
        information.
        Return a confidence score from 0 to 10.
        10 = completely supported
        8  = strongly supported
        5  = partially supported
        0  = unsupported

        Return ONLY the number.
        """
    )
    chain=prompt|model|StrOutputParser()
    output=chain.invoke({
        'question':state["message"],
        'answer':state['ragResponse']
    })
    return {'confidence':output}

