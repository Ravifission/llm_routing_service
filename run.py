#!/usr/bin/env python
"""Run the Llama inference service."""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import config


def main():
    """Run the service."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║        Llama 3.2 1B Inference Service                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Validate configuration
    try:
        config.validate()
        if config.MODEL_CHECKPOINT_DIR.startswith(('.', '/')):
            print(f"✓ Model checkpoint path: {config.MODEL_CHECKPOINT_DIR}")
        else:
            print(f"✓ Hugging Face model ID: {config.MODEL_CHECKPOINT_DIR}")
            print("  (Model will be downloaded from HF Hub on first run)")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nPlease ensure:")
        print("1. Your model checkpoint is in the correct location (for local models)")
        print("2. MODEL_CHECKPOINT_DIR in .env is set correctly")
        print("\nExamples:")
        print("  Local path:  MODEL_CHECKPOINT_DIR=/path/to/llama-3.2-1b-instruct")
        print("  HF model ID: MODEL_CHECKPOINT_DIR=meta-llama/Llama-3.2-1B-Instruct")
        sys.exit(1)
    
    print(f"""
Starting service...
  Host: {config.HOST}
  Port: {config.PORT}

API Documentation:
  - Swagger UI: http://{config.HOST}:{config.PORT}/docs
  - ReDoc:      http://{config.HOST}:{config.PORT}/redoc

Health Check:
  - http://{config.HOST}:{config.PORT}/health

Main Endpoints:
  - POST /infer     - Generate text and extract number
  - POST /generate  - Generate text only

Press CTRL+C to stop the service
""")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=config.HOST,
            port=config.PORT,
            reload=False,  # Set to False in production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nService stopped by user.")
    except Exception as e:
        print(f"\n✗ Error starting service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

