"""FastAPI application for Llama 3.2 1B inference service."""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.models import InferenceRequest, InferenceResponse, HealthResponse
from app.model_manager import get_model_manager

# Import accuracy calculation function
from app.accuracy_calculator import calculate_accuracy_from_excel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager (private variable following Python conventions)
_model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global _model_manager
    
    # Startup
    logger.info("Starting Llama 3.2 1B Inference Service...")
    
    try:
        # Get checkpoint directory from environment variable or config
        # This can be either:
        # - A local path: "./model_checkpoint" or "/path/to/model"
        # - A Hugging Face model ID: "ravifission/MyLlamaNPC_0910"
        checkpoint_dir = os.getenv("MODEL_CHECKPOINT_DIR", "ravifission/MyLlamaNPC_0910")
        
        # Only validate if it looks like a local path
        if checkpoint_dir.startswith(('.', '/')):
            if not os.path.exists(checkpoint_dir):
                logger.error(f"Local model checkpoint directory not found: {checkpoint_dir}")
                logger.error("Please set MODEL_CHECKPOINT_DIR to a valid local path or Hugging Face model ID")
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_dir}")
        
        logger.info(f"Loading model from: {checkpoint_dir}")
        _model_manager = get_model_manager(checkpoint_dir)
        
        logger.info("Service started successfully!")
        logger.info(f"Model device: {_model_manager.get_device()}")
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Llama 3.2 1B Inference Service...")


# Create FastAPI app
app = FastAPI(
    title="Llama 3.2 1B Inference Service",
    description="Inference service for Llama 3.2 1B Instruct model",
    version=__version__,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "Llama 3.2 1B Inference Service",
        "version": __version__,
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    if _model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return HealthResponse(
        status="healthy" if _model_manager.is_ready() else "unhealthy",
        model_loaded=_model_manager.is_ready(),
        model_device=_model_manager.get_device(),
        version=__version__
    )


@app.post("/generate", response_model=InferenceResponse, tags=["Inference"])
async def generate_text(request: InferenceRequest):
    """
    Generate text using the Llama model.
    
    Returns only the generated text without extracting numbers.
    """
    if _model_manager is None or not _model_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    
    try:
        generated_text = _model_manager.generate(
            input_text=request.input_text,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens
        )
        
        return InferenceResponse(
            generated_text=generated_text,
            extracted_number=None,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return InferenceResponse(
            generated_text="",
            extracted_number=None,
            success=False,
            error_message=str(e)
        )


@app.post("/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer_number(request: InferenceRequest):
    """
    Generate text and extract number from it.
    Then calculates accuracy for that score using ground truth Excel file.
    
    This is the main endpoint that matches your use case:
    - Generates text using the model
    - Extracts the number from the generated text
    - Uses that number as target_score to calculate accuracy
    - Returns both the text and the extracted number
    """
    if _model_manager is None or not _model_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready"
        )
    
    try:
        # Generate text
        generated_text = _model_manager.generate(
            input_text=request.input_text,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens
        )
        
        # Extract number from generated text
        extracted_number = _model_manager.extract_number(generated_text)
        
        # Calculate accuracy using the extracted number as target_score
        # Default path to the ground truth Excel file
        excel_path = "dataset/inference_on_pretrained_model.xlsx"
        
        accuracy_results = {}
        if extracted_number is not None and os.path.exists(excel_path):
            try:
                logger.info(f"Calculating accuracy for target_score={extracted_number}")
                accuracy_results = calculate_accuracy_from_excel(
                    excel_path=excel_path,
                    target_score=int(extracted_number)
                )
                logger.info(f"Accuracy calculation completed: {len(accuracy_results)} models analyzed")
            except Exception as accuracy_error:
                logger.warning(f"Accuracy calculation failed: {accuracy_error}")
        
        # Create response with accuracy results in the generated_text field
        # Format the results for better presentation
        sorted_results = None
        if accuracy_results:
            result_summary = f"\n\nAccuracy Results (target_score={extracted_number}):\n"
            result_summary += "-" * 60 + "\n"
            
            # Sort by accuracy and convert to list of dicts for JSON serialization
            sorted_model_results = sorted(accuracy_results.items(), key=lambda x: x[1], reverse=True)
            sorted_results = [
                {"model_name": name, "accuracy": acc} 
                for name, acc in sorted_model_results
            ]
            
            for model_name, accuracy in sorted_model_results:
                result_summary += f"{model_name:50s} {accuracy:6.2f}%\n"
            
            response = InferenceResponse(
                generated_text=generated_text + result_summary,
                extracted_number=extracted_number,
                sorted_results=sorted_results,
                success=True
            )
        else:
            response = InferenceResponse(
                generated_text=generated_text,
                extracted_number=extracted_number,
                sorted_results=sorted_results,
                success=True
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return InferenceResponse(
            generated_text="",
            extracted_number=None,
            sorted_results=None,
            success=False,
            error_message=str(e)
        )


# Utility function that can be imported and used directly
def infer_number_from_text(
    input_text: str,
    temperature: float = 0.01,
    max_new_tokens: int = 5
) -> Optional[float]:
    """
    Utility function to generate text and extract number.
    
    This function can be imported and used in other parts of your code.
    
    Args:
        input_text: Input text/prompt
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Extracted number or None
    
    Example:
        from app.main import infer_number_from_text
        
        result = infer_number_from_text("What is 2 + 2?")
        print(result)  # 4.0
    """
    if _model_manager is None:
        raise RuntimeError("Model not initialized")
    
    return _model_manager.generate_and_extract_number(
        input_text=input_text,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )

