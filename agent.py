import os
import requests
import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings,ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import HumanMessage,AIMessage
from dotenv import load_dotenv
from typing import Annotated, Optional, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, StateGraph,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel
import os


load_dotenv()

################################CONFIGURATIONS############################################
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##########################FIRST LLM THAT GENERATS AN ANSWER#################################
endpoint=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.2,  
)

model=ChatHuggingFace(
    llm=endpoint
)

########################SECOND LLM FOR THE CONFIDENCE EVALUATION#############################

endpointEvaluator=HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.4,  
)

modelEvaluator=ChatHuggingFace(
    llm=endpointEvaluator
)


 #Loading the document
loader=PyPDFLoader(file_path="DOC.pdf")
docs=loader.load()
#Splitting the Document content
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100,
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


#class to verify the confidence if its less than 8 contact helpline and the main node
class AIBOTState(TypedDict):
    message:str
    ragResponse:str
    confidence:int
    context: str
  

class ChatRequest(BaseModel):
    message: str    

def RAGNode(state:AIBOTState):
    #Similarity Search-----
    retriever_docs=vectorSore.similarity_search(
        state['message'],k=3
    )

    documents = retriever_docs
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
    return {'ragResponse':output,'context':context}



def ConfidenceScore(state:AIBOTState):
    prompt=ChatPromptTemplate.from_template(
        """
        Evaluate the following answer.
        Question:
        {question}
         Context:
        {context}
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
    chain=prompt|modelEvaluator|StrOutputParser()
    output=chain.invoke({
        'question':state["message"],
        'context':state['context'],
        'answer':state['ragResponse']
    })
    return {'confidence':int(output.strip())}


#now the conditional Node
def CheckCondition(state:AIBOTState):
    if state["confidence"]>=8:
        return "answer"
    else:
        return 'human'


#answer node
def answerNode(state:AIBOTState):
    return {'ragResponse':state["ragResponse"]}


#human escalation Node
def human_node(state:AIBOTState):
    email_data=human_email.invoke({"question":state['message'],"ai_message":state['ragResponse'],"confidence":state['confidence']})
    return {
        'ragResponse':(
            "I am not sure about this scenario."
            "This Scenario has been forwarded to a human."
            "Contact Helpline for further details."
        )
    }



@tool
def human_email(
    question:str,
    ai_message:str,
    confidence:int
):
    """AI SENDS AN EMAIL TO THE HUMAN IF THE CONFIDENCE SCORE IS
        LOW.
    """
    result=send_email(
        question,
        ai_message,
        confidence
    )
    return result

def send_email(question:str,ai_message:str,confidence:int):
    sender_email=os.getenv("EMAIL_ADDRESS")
    sender_password = os.getenv("EMAIL_PASSWORD")
    receiver_email=os.getenv("HUMAN_REVIEW_EMAIL")
    message=MIMEMultipart()
    message['FROM']=sender_email
    message['To']=receiver_email
    message['Subject']="Human Review Required"

    body=f"""
    AI confidence is low , requires Human Intervention
    User_Question:
    {question}
    AI_Response:
    {ai_message}
    Confidence Score:
    {confidence}/10

    """

    message.attach(MIMEText(body,'plain'))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:

        server.login(
            sender_email,
            sender_password
        )

        server.sendmail(
            sender_email,
            receiver_email,
            message.as_string()
        )
    return "Your Case has been sent!"



graph=StateGraph(AIBOTState)
graph.add_node('rag',RAGNode)
graph.add_node('confidence',ConfidenceScore)
graph.add_node('answer',answerNode)
graph.add_node('human',human_node)

graph.add_edge(START,'rag')
graph.add_edge('rag','confidence')
graph.add_conditional_edges(
    'confidence',
    CheckCondition,
    {
        'answer':'answer',
        'human':'human'
    }
)

graph.add_edge('answer',END)
graph.add_edge('human',END)

app=graph.compile()

@api.get("/health")
def health():
    return {
        "status": "ok"
    }


@api.post("/chat")
def chat(request: ChatRequest):

    result = app.invoke({
        "message": request.message
    })

    return {
        "response": result["ragResponse"],
        "confidence": result["confidence"]
    }