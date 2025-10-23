#!/usr/bin/env python
"""Test client for the inference service."""

import requests
import sys
import json


def test_service():
    """Test the inference service."""
    
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Testing Llama 3.2 1B Inference Service")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✓ Service is healthy")
            print(f"   ✓ Model loaded: {health['model_loaded']}")
            print(f"   ✓ Device: {health['model_device']}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Cannot connect to service: {e}")
        print("\n   Please make sure the service is running:")
        print("   python run.py")
        return False
    
    # Test 2: Generate text
    print("\n2. Testing text generation...")
    test_input = "What is 2 + 2?"
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "input_text": test_input,
                "temperature": 0.01,
                "max_new_tokens": 5
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✓ Generation successful")
                print(f"   Input: {test_input}")
                print(f"   Generated: {result['generated_text']}")
            else:
                print(f"   ✗ Generation failed: {result.get('error_message')}")
        else:
            print(f"   ✗ Request failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Infer number
    print("\n3. Testing number inference...")
    test_cases = [
        "What is 5 * 3?",
        "Calculate 10 + 7",
        "100 divided by 4"
    ]
    
    for test_input in test_cases:
        try:
            response = requests.post(
                f"{base_url}/infer",
                json={
                    "input_text": test_input,
                    "temperature": 0.01,
                    "max_new_tokens": 5
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print(f"   ✓ Input: {test_input}")
                    print(f"     Generated: {result['generated_text']}")
                    print(f"     Extracted number: {result['extracted_number']}")
                else:
                    print(f"   ✗ Failed: {result.get('error_message')}")
            else:
                print(f"   ✗ Request failed: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)
    return True


def example_usage():
    """Show example usage."""
    
    print("\n\nExample Usage:")
    print("=" * 60)
    
    print("\n# Python example:")
    print("""
import requests

response = requests.post(
    "http://localhost:8000/infer",
    json={
        "input_text": "What is 2 + 2?",
        "temperature": 0.01,
        "max_new_tokens": 5
    }
)

result = response.json()
numb = result['extracted_number']
print(f"Result: {numb}")
""")
    
    print("\n# cURL example:")
    print("""
curl -X POST "http://localhost:8000/infer" \\
  -H "Content-Type: application/json" \\
  -d '{
    "input_text": "Calculate 5 * 3",
    "temperature": 0.01,
    "max_new_tokens": 5
  }'
""")


if __name__ == "__main__":
    success = test_service()
    
    if success:
        example_usage()
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)

