"""Configuration for the inference service."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class."""
    
    # Model configuration
    MODEL_CHECKPOINT_DIR = os.getenv(
        "MODEL_CHECKPOINT_DIR",
        "ravifission/MyLlamaNPC_0910"
    )
    
    # Service configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Generation defaults
    DEFAULT_TEMPERATURE = 0.01
    DEFAULT_MAX_NEW_TOKENS = 5
    
    # Prompt formatting
    # Options: 'llama_evaluation', 'plain', 'custom'
    PROMPT_FORMATTER = os.getenv("PROMPT_FORMATTER", "llama_evaluation")
    
    # Custom template (only used if PROMPT_FORMATTER='custom')
    CUSTOM_PROMPT_TEMPLATE = os.getenv(
        "CUSTOM_PROMPT_TEMPLATE",
        "{question}"
    )
    
    # Benchmarking dataset configuration
    BENCHMARK_DATASET_PATH = os.getenv(
        "BENCHMARK_DATASET_PATH",
        "dataset/ground_truth.csv"
    )
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        # Check if it's a local path
        checkpoint_path = Path(cls.MODEL_CHECKPOINT_DIR)
        
        # If it looks like a local path (starts with . or /), verify it exists
        if cls.MODEL_CHECKPOINT_DIR.startswith(('.', '/')):
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Local model checkpoint directory not found: {cls.MODEL_CHECKPOINT_DIR}"
                )
        else:
            # Assume it's a Hugging Face model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
            # We'll let transformers handle validation when loading
            pass
        
        return True


# Global config instance
config = Config()

