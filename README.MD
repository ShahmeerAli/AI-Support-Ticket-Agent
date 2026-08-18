# 🤖 AI Customer Support Agent

An **AI-powered customer support system** that combines **Hugging Face LLMs, RAG (Retrieval-Augmented Generation), LangGraph, ChromaDB, and FastAPI** to provide reliable answers from a predefined knowledge base while automatically escalating uncertain cases to human support.

The goal of this project is not simply to build a chatbot. It is designed as an **agentic customer-support workflow** where the AI retrieves relevant information, generates an answer, evaluates its own confidence, and decides whether the request can be answered automatically or requires human intervention.

## 🚀 How It Works

The system follows this pipeline:

```text
User
 │
 ▼
Next.js Frontend
 │
 │ HTTP Request
 ▼
FastAPI Backend
 │
 ▼
LangGraph Agent
 │
 ▼
RAG Retrieval
 │
 ├── PDF Knowledge Base
 ├── Text Splitting
 ├── Hugging Face Embeddings
 └── ChromaDB Vector Store
 │
 ▼
Hugging Face LLM
 │
 ▼
Generated Answer
 │
 ▼
Confidence Evaluation
 │
 ┌───────────────┴───────────────┐
 │                               │
Confidence >= 8              Confidence < 8
 │                               │
 ▼                               ▼
Answer User                 Human Escalation
                                 │
                                 ▼
                            Email Support
```

### 1. Knowledge Base

The system uses a PDF document as its knowledge base.

The document is:

1. Loaded using `PyPDFLoader`
2. Split into smaller chunks using `RecursiveCharacterTextSplitter`
3. Converted into vector embeddings using a **Hugging Face Sentence Transformer**
4. Stored in **ChromaDB**

When a user asks a question, the system performs a similarity search to retrieve the most relevant pieces of information from the knowledge base.

### 2. Retrieval-Augmented Generation

Instead of allowing the LLM to answer completely from its own knowledge, the system provides the retrieved information as context.

The LLM is instructed to answer the user's question **only using the retrieved context**.

This reduces unsupported answers and makes the system more suitable for customer-support scenarios where answers should be grounded in a defined knowledge base.

### 3. Hugging Face LLM

The project uses **Hugging Face** for the language-model layer.

The current implementation uses:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
```

Two LLM instances are used:

* **Answer Generation LLM** — generates the customer-facing response.
* **Confidence Evaluation LLM** — evaluates how strongly the generated answer is supported by the retrieved knowledge.

The model is accessed through LangChain's Hugging Face integration.

### 4. LangGraph Agent

**LangGraph** controls the workflow of the AI agent.

The graph contains several nodes:

```text
RAG
 │
 ▼
Confidence Evaluation
 │
 ├── High Confidence ──► Answer
 │
 └── Low Confidence ───► Human Review
```

This makes the system deterministic and easier to control than a simple chatbot chain.

### 5. Confidence-Based Human Escalation

After generating an answer, the second LLM evaluates its confidence on a scale from **0 to 10**.

The current decision rule is:

```text
Confidence >= 8
        │
        ▼
   Answer User


Confidence < 8
        │
        ▼
 Human Intervention
```

When the confidence is below the threshold, the system sends the case to a human reviewer through email instead of providing an unreliable answer.

The email contains:

* User question
* AI-generated response
* Confidence score

This creates a **human-in-the-loop support system**.

## ⚡ FastAPI Backend

The AI agent is exposed through a **FastAPI REST API**.

The frontend communicates with the backend through HTTP.

### Health Check

```http
GET /health
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "message": "What is the refund policy?"
}
```

Response:

```json
{
  "response": "AI generated response...",
  "confidence": 9
}
```

This allows the AI system to be consumed by a web frontend without exposing the internal LangGraph, RAG, ChromaDB, or LLM implementation to the client.

## 🖥️ Frontend

The frontend is built using **Next.js** and communicates with the FastAPI backend.

The frontend is responsible for:

* Chat interface
* Sending user questions
* Displaying AI responses
* Showing loading states
* Handling API errors
* Displaying human-escalation responses

The AI processing remains on the backend.

## 🧰 Tech Stack

| Technology       | Purpose                                     |
| ---------------- | ------------------------------------------- |
| **Hugging Face** | LLMs and embeddings                         |
| **DeepSeek**     | Answer generation and confidence evaluation |
| **LangChain**    | LLM and RAG integration                     |
| **LangGraph**    | Agent workflow and decision routing         |
| **ChromaDB**     | Vector database                             |
| **FastAPI**      | Backend REST API                            |
| **Next.js**      | Frontend                                    |
| **Python**       | AI/backend implementation                   |
| **PyPDFLoader**  | PDF knowledge-base ingestion                |
| **SMTP / Gmail** | Human escalation emails                     |

## 🎯 Project Goal

The project demonstrates how to build a production-oriented AI support workflow rather than a basic LLM chatbot.

It combines:

* **LLM integration**
* **Retrieval-Augmented Generation**
* **Vector search**
* **Agentic workflows**
* **Confidence evaluation**
* **Conditional routing**
* **Human-in-the-loop escalation**
* **REST APIs**
* **Frontend/backend integration**

The architecture is designed so that the AI can answer straightforward support questions automatically while recognizing situations where it should **stop and involve a human**.

## 🔮 Future Improvements

Planned improvements include:

* Conversation memory
* Persistent ChromaDB storage
* Authentication
* Conversation history
* Admin dashboard for human reviewers
* Better confidence evaluation
* Source/citation display
* Streaming responses
* Docker containerization
* Production deployment
* Monitoring and logging
