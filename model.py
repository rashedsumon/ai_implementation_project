import os
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field

# LangChain and Graph requirements
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

# Pydantic schema for strict JSON evaluation structured output
class TicketAnalysis(BaseModel):
    sentiment: str = Field(description="The emotional tone of the customer email")
    urgency_level: str = Field(description="Urgency graded as low, medium, or high")
    category: str = Field(description="Classified business category (e.g., billing, technical)")
    suggested_action: str = Field(description="Actionable internal routing suggestion")
    draft_response: str = Field(description="Professional customer resolution email reply draft")

# Graph State definition
class AgentState(TypedDict):
    ticket_text: str
    retrieved_context: str
    structured_output: Dict[str, Any]

class SupportAIModelPipeline:
    def __init__(self):
        # Automatically draws the configured global OS environment flag from streamlit_app.py
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1
        )
        
        # FIX: Swapped to a local Hugging Face embedding model. 
        # This executes purely on the Streamlit server container, bypassing Google's async API bugs.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vector_db = None

    def build_vector_store(self, processed_data: List[Dict[str, Any]]):
        """Builds an in-memory FAISS database instance using local embeddings."""
        sample_size = min(50, len(processed_data))
        documents = [
            Document(page_content=item["text"], metadata=item["metadata"])
            for item in processed_data[:sample_size]
        ]
        
        if documents:
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

    def get_retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node designed to run embedding matches against FAISS index."""
        query = state["ticket_text"]
        
        if self.vector_db:
            try:
                matching_docs = self.vector_db.similarity_search(query, k=1)
                context_string = "\n\n".join([doc.page_content for doc in matching_docs])
            except Exception:
                context_string = "Standard Operating Procedures apply."
        else:
            context_string = "Default Enterprise billing and customer disputes parameters apply."
            
        return {"retrieved_context": context_string}

    def get_llm_processor_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node executing LLM structural analysis backed by fine-tuned instructions."""
        system_prompt = (
            "You are an elite corporate AI Implementation Specialist agent. Analyze incoming support requests.\n"
            "Use the provided context from company manuals to structure the answer accurately.\n\n"
            f"--- Context Knowledge Base ---\n{state['retrieved_context']}\n\n"
            f"--- Customer Ticket ---\n{state['ticket_text']}"
        )
        
        structured_llm = self.llm.with_structured_output(TicketAnalysis)
        response = structured_llm.invoke(system_prompt)
        
        return {"structured_output": response.model_dump()}

    def compile_graph(self):
        """Assembles LangGraph workflow sequence framework."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve_knowledge", self.get_retriever_node)
        workflow.add_node("analyze_and_draft", self.get_llm_processor_node)
        
        workflow.set_entry_point("retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "analyze_and_draft")
        workflow.add_edge("analyze_and_draft", END)
        
        return workflow.compile()