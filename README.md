# AI Support Ticket Triage & Auto-Response System

A production-grade Agentic RAG solution for enterprise helpdesks. This system uses a combined tech stack involving Hugging Face tokenizers, LangChain, LangGraph orchestrations, ChromaDB vectors, and Google's Gemini models.

## Deployment on Streamlit Cloud

### 1. Secrets Setup
Before deploying to Streamlit Cloud, ensure your Gemini API key is configured in the **Secrets** section of the Streamlit Advanced Settings panel:
```toml
GEMINI_API_KEY = "your_actual_google_gemini_api_key_here"