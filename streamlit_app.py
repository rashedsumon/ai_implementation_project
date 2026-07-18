import streamlit as st
import os

# Import pipeline components
from data_loader import HelpdeskDataLoader
from model import SupportAIModelPipeline

st.set_page_colortheme = "dark"
st.title("Enterprise AI Ticket Triage & Auto-Responder")
st.caption("Powered by LangGraph, Hugging Face Data Engine, ChromaDB, and Gemini")

# 1. Verification of Cloud Deployment Secret Keys
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("Missing Gemini API Key! Please configure the GEMINI_API_KEY parameter in your Streamlit Cloud Workspace Secrets panel.")
    st.stop()

# Initialize state objects in application engine memory spaces
if "pipeline" not in st.session_state:
    with st.spinner("Initializing AI Orchestration Stacks (Downloading Core Datasets)..."):
        # Setup pipeline models
        pipeline = SupportAIModelPipeline(api_key=api_key)
        
        # Pull data via Hugging Face loader engine logic
        loader = HelpdeskDataLoader()
        raw_hf_data = loader.fetch_and_prepare_data()
        
        # Populate Vector Stores
        pipeline.build_vector_store(raw_hf_data)
        
        st.session_state.pipeline = pipeline
        st.session_state.graph = pipeline.compile_graph()
    st.success("AI Infrastructure successfully mounted!")

# 2. User Input UI Area
st.subheader("Input Customer Ticket Profile")
default_email = (
    "I was charged twice for my subscription renewal this morning ($49.99 x 2). "
    "I need an immediate refund for the second charge, and I want to make sure my account isn't canceled. "
    "This is urgent!"
)
user_ticket = st.text_area("Paste Incoming Support Text Email:", value=default_email, height=150)

# 3. Execution Action Trigger
if st.button("Run Automated Triage Pipeline", type="primary"):
    initial_graph_input = {
        "ticket_text": user_ticket,
        "retrieved_context": "",
        "structured_output": {}
    }
    
    with st.spinner("Executing LangGraph Processing Node Routing Loops..."):
        # Run agent graph
        output_state = st.session_state.graph.invoke(initial_graph_input)
        structured_results = output_state.get("structured_output", {})

    # 4. Model Output Visualization Interface
    st.subheader("Automated Operational Diagnostics (Model Output)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Calculated Urgency Level", value=structured_results.get("urgency_level", "N/A").upper())
    with col2:
        st.metric(label="Identified Sentiment", value=structured_results.get("sentiment", "N/A").capitalize())
    with col3:
        st.metric(label="Assigned Business Routing Category", value=structured_results.get("category", "N/A").replace("_", " ").capitalize())
        
    st.markdown("### Suggested Operational Action Route")
    st.info(structured_results.get("suggested_action", "No internal routing flags raised."))
    
    st.markdown("### Generated Automated Email Reply Draft")
    st.text_area("Review or Approve Communication Output:", value=structured_results.get("draft_response", ""), height=200)