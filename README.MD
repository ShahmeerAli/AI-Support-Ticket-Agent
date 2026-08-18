A full-stack AI-powered customer support system that automatically resolves tickets using a stateful multi-step agent. Built with LangGraph for agentic workflow control and LangChain for RAG-based knowledge retrieval.

The agent classifies incoming tickets, searches a knowledge base, evaluates its own confidence, and either drafts an automatic response or escalates to a human agent — without hallucinating uncertain answers.

How it works
- Customer submits a ticket via React frontend
- LangGraph agent classifies the ticket and searches the knowledge base using LangChain RAG + ChromaDB
- If confidence is high → agent drafts and sends a response automatically
- If confidence is low → ticket is escalated to the human agent dashboard
- Human agent reviews, resolves manually, LangGraph resumes and logs the outcome
- Every ticket is logged with full reasoning trace, confidence score, and final response

Tech Stack
- **Agent:** LangGraph, LangChain, OpenAI / Anthropic API
- **Vector Store:** ChromaDB
- **Backend:** FastAPI
- **Frontend:** React
- **Database:** PostgreSQL / SQLite

Key concepts
- Stateful agentic workflows with LangGraph
- Retrieval Augmented Generation (RAG)
- Human-in-the-loop with confidence-based routing
- Full-stack AI system integration
- Production-grade logging and audit trail
