import os
import json
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
    # FIX: Require explicit api_key on initialization to ensure availability inside background execution threads
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        
        # Local, asynchronous-safe embedding module running entirely inside the container CPU memory space
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
        """LangGraph node executing LLM structural analysis via systemic JSON enforcement."""
        system_prompt = (
            "You are an elite corporate AI Implementation Specialist agent. Analyze incoming support requests.\n"
            "You must return response EXCLUSIVELY as a raw, single-line valid JSON object.\n"
            "Do not wrap your output in markdown code blocks like ```json ... ```. Output raw text data only.\n\n"
            "The JSON structure must match these exact keys:\n"
            '- "sentiment": string\n'
            '- "urgency_level": "low", "medium", or "high"\n'
            '- "category": string representing business line\n'
            '- "suggested_action": string detailing ticket routing step\n'
            '- "draft_response": string containing the full professional email resolution reply\n\n'
            f"--- Context Knowledge Base ---\n{state['retrieved_context']}\n\n"
            f"--- Customer Ticket ---\n{state['ticket_text']}"
        )
        
        # Call model using explicitly passed client parameters
        raw_response = self.llm.invoke(system_prompt).content
        
        # Clean text boundaries in case markdown formatting blocks slip into response string
        cleaned_json = raw_response.strip()
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if cleaned_json.startswith("json"):
            cleaned_json = cleaned_json.split("json", 1)[1].strip()

        try:
            # Parse and validate with Pydantic class structures safely
            validated_data = TicketAnalysis.model_validate_json(cleaned_json)
            output_dict = validated_data.model_dump()
        except Exception:
            # Operational fallback dataset in case structural parsing fails
            output_dict = {
                "sentiment": "undetermined",
                "urgency_level": "high",
                "category": "customer_support",
                "suggested_action": "manual_review_required",
                "draft_response": f"Thank you for reaching out. We received your request: '{state['ticket_text'][:60]}...' and are processing it immediately."
            }
        
        return {"structured_output": output_dict}

    def compile_graph(self):
        """Assembles LangGraph workflow sequence framework."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve_knowledge", self.get_retriever_node)
        workflow.add_node("analyze_and_draft", self.get_llm_processor_node)
        
        workflow.set_entry_point("retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "analyze_and_draft")
        workflow.add_edge("analyze_and_draft", END)
        
        return workflow.compile()