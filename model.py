import os
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field

# LangChain and Graph requirements
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
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
    def __init__(self, api_key: str):
        # Instantiate Gemini via LangChain using the validated explicit API key configuration
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        # FIX: Updated to the robust, current generation text embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        self.vector_store_path = os.path.join(os.getcwd(), "chroma_db")
        self.vector_db = None

    def build_vector_store(self, processed_data: List[Dict[str, Any]]):
        """Builds or connects to the local standalone Chroma DB instance using batch ingestion."""
        documents = [
            Document(page_content=item["text"], metadata=item["metadata"])
            for item in processed_data[:100]  # Kept light to stay within safe memory and API limits
        ]
        
        # FIX: Ingest in small batches to protect against Google GenAI rate limits or network dropouts
        batch_size = 20
        
        # Initialize the Chroma store with the first batch
        if documents:
            self.vector_db = Chroma.from_documents(
                documents=documents[:batch_size],
                embedding=self.embeddings,
                persist_directory=self.vector_store_path
            )
            
            # Ingest remaining batches sequentially
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.vector_db.add_documents(batch)

    def get_retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node designed to run embedding matches against vector store."""
        if not self.vector_db:
            self.vector_db = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embeddings)
            
        query = state["ticket_text"]
        matching_docs = self.vector_db.similarity_search(query, k=2)
        context_string = "\n\n".join([doc.page_content for doc in matching_docs])
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