import os
from dotenv import load_dotenv

def load_config():
    """Loads API keys and Azure configuration from .env file."""
    load_dotenv()
    config = {
        # Azure OpenAI Config
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "azure_chat_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        "azure_embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    }
    # Basic validation
    if not all([config["azure_endpoint"], config["azure_api_key"], config["azure_api_version"], config["azure_chat_deployment"], config["azure_embedding_deployment"]]):
         print("Warning: One or more Azure OpenAI environment variables are missing in .env. Please check.")
         # You might want to raise an error here depending on desired strictness
    return config