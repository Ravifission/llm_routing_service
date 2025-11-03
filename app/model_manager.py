"""Model manager for Llama 3.2 1B Instruct inference."""

import torch
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import Config
from app.prompt_formatter import get_formatter

logger = logging.getLogger(__name__)


class LlamaModelManager:
    """Manages the Llama model loading and inference."""
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize the model manager.
        
        Args:
            checkpoint_dir: Path to the model checkpoint directory
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        # Initialize prompt formatter based on config
        formatter_kwargs = {}
        if Config.PROMPT_FORMATTER == 'custom':
            formatter_kwargs['template'] = Config.CUSTOM_PROMPT_TEMPLATE
        
        self.prompt_formatter = get_formatter(Config.PROMPT_FORMATTER, **formatter_kwargs)
        logger.info(f"Using prompt formatter: {Config.PROMPT_FORMATTER}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the model, tokenizer, and pipeline."""
        import os
        
        logger.info(f"Loading model from: {self.checkpoint_dir}")
        logger.info("This works with both local paths and Hugging Face model IDs")
        
        # Get HF token if available
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        
        try:
            # Load Model
            # This supports both:
            # - Local paths: "./model_checkpoint" or "/path/to/model"
            # - Hugging Face IDs: "ravifission/MyLlamaNPC_0910"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint_dir,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="eager",
                token=hf_token  # Use token for private/gated models
            )
            
            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_dir,
                token=hf_token  # Use token for private/gated models
            )
            
            # Create Pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            logger.info("Model loaded successfully")
            logger.info(f"Model device: {self.get_device()}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def generate(
        self,
        input_text: str,
        temperature: float = 0.01,
        max_new_tokens: int = 5,
        **kwargs
    ) -> str:
        """
        Generate text using the Llama model.
        
        Args:
            input_text: The input text/prompt
            temperature: Temperature for generation (default: 0.01)
            max_new_tokens: Maximum number of new tokens to generate (default: 5)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text content
        """
        if self.pipe is None:
            raise RuntimeError("Model pipeline not initialized")
        
        try:
            # Format the input text using the configured prompt formatter
            formatted_input = self.prompt_formatter.format(input_text)


            
            # Generate output
            outputs = self.pipe(
                formatted_input,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                disable_compile=True,
                **kwargs
            )

            # Extract the generated text
            # Based on your code: outputs[0]['generated_text'][-1]['content']
            generated_content = outputs[0]['generated_text'][-1]['content']
            
            return generated_content
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise
    
    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract number from generated text.
        
        Args:
            text: Text containing a number
        
        Returns:
            Extracted number as float, or None if no number found
        """
        import re
        
        # Remove whitespace
        text = text.strip()
        
        # Try to find a number in the text
        # Match integers and floats (including negative numbers)
        match = re.search(r'-?\d+\.?\d*', text)
        
        if match:
            try:
                return float(match.group())
            except ValueError:
                logger.warning(f"Could not convert '{match.group()}' to number")
                return float(5)
        
        logger.warning(f"No number found in text: '{text}'")
        return float(5)
    
    def generate_and_extract_number(
        self,
        input_text: str,
        temperature: float = 0.01,
        max_new_tokens: int = 5,
        **kwargs
    ) -> Optional[float]:
        """
        Generate text and extract number from it.
        
        Args:
            input_text: The input text/prompt
            temperature: Temperature for generation
            max_new_tokens: Maximum number of new tokens
            **kwargs: Additional generation parameters
        
        Returns:
            Extracted number or None
        """
        generated_text = self.generate(
            input_text,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return self.extract_number(generated_text)
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None and self.pipe is not None
    
    def get_device(self) -> str:
        """Get the device the model is running on."""
        if self.model is not None:
            return str(self.model.device)
        return "unknown"


# Global model manager instance
_model_manager: Optional[LlamaModelManager] = None


def get_model_manager(checkpoint_dir: Optional[str] = None) -> LlamaModelManager:
    """
    Get or create the global model manager instance.
    
    Args:
        checkpoint_dir: Path to model checkpoint (required on first call)
    
    Returns:
        LlamaModelManager instance
    """
    global _model_manager
    
    if _model_manager is None:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required for first initialization")
        _model_manager = LlamaModelManager(checkpoint_dir)
    
    return _model_manager

