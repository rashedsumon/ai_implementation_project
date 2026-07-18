import os
from datasets import load_dataset

class HelpdeskDataLoader:
    """
    Manages automated downloading and caching of helpdesk support data
    from the Hugging Face Hub for fine-tuning context and RAG vectors.
    """
    def __init__(self, dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"):
        self.dataset_name = dataset_name
        # Resolves path neatly for both local development and virtual machines on Streamlit Cloud
        self.cache_dir = os.path.join(os.getcwd(), "dataset_cache")

    def fetch_and_prepare_data(self):
        """
        Downloads the dataset from Hugging Face Hub using the datasets library,
        caches it locally, and parses it into a standard vector-ingestion format.
        """
        print(f"Initializing connection to Hugging Face Hub dataset: {self.dataset_name}")
        
        # Load dataset from Hugging Face with local disk caching
        dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
        
        # Convert training split into standardized documents for our downstream vector store
        raw_data = []
        for row in dataset['train']:
            # Safe fallbacks to extract content even if upstream schema keys slightly fluctuate
            instruction = row.get("instruction", row.get("utterance", ""))
            response = row.get("response", row.get("intent", ""))
            
            # Combine fields to build a robust context packet for the RAG retriever engine
            document_text = f"Customer Request: {instruction}\nSuggested Solution: {response}"
            
            raw_data.append({
                "text": document_text,
                "metadata": {"category": row.get("intent", "general")}
            })
            
        return raw_data

if __name__ == "__main__":
    # Test block to verify pipeline logic standalone
    loader = HelpdeskDataLoader()
    data = loader.fetch_and_prepare_data()
    print(f"Successfully processed {len(data)} documents.")