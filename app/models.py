"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class InferenceRequest(BaseModel):
    """Request model for inference."""
    
    input_text: str = Field(
        ...,
        description="Input text/prompt for the model",
        min_length=1
    )
    temperature: float = Field(
        default=0.01,
        ge=0.0,
        le=2.0,
        description="Temperature for generation"
    )
    max_new_tokens: int = Field(
        default=5,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "input_text": "What is 2 + 2?",
                "temperature": 0.01,
                "max_new_tokens": 5
            }
        }


class InferenceResponse(BaseModel):
    """Response model for inference."""
    
    generated_text: str = Field(..., description="Generated text from the model")
    extracted_number: Optional[float] = Field(None, description="Extracted number from generated text")
    success: bool = Field(..., description="Whether the inference was successful")
    error_message: Optional[str] = Field(None, description="Error message if inference failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    sorted_results: Optional[List[Dict[str, Any]]] = Field(None, description="Sorted accuracy results for models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "4",
                "extracted_number": 4.0,
                "success": True,
                "error_message": None,
                "timestamp": "2025-10-21T12:00:00",
                "sorted_results": None
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_device: str = Field(..., description="Device model is running on")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

