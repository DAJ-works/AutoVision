#!/usr/bin/env python3
"""
Comprehensive test for the chatbot with CA Vehicle Code data
Tests multiple aspects:
1. Basic speed law (CVC 22350)
2. Stop sign requirements (CVC 22450)
3. Right-of-way rules (CVC 21800)
4. Integration with YOLO detection data
"""
import requests
import json

def test_legal_chat(question):
    """Test the legal chat endpoint"""
    try:
        response = requests.post(
            'http://localhost:5001/api/legal-chat',
            json={'message': question},
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"

def main():
    print("=" * 80)
    print("COMPREHENSIVE CA VEHICLE CODE CHATBOT TEST")
    print("=" * 80)
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Basic Speed Law",
            "question": "What is California Vehicle Code 22350 and what does it say about speed limits?"
        },
        {
            "name": "Stop Sign Requirements",
            "question": "What are the requirements for stopping at a stop sign in California?"
        },
        {
            "name": "Right of Way",
            "question": "What is CVC 21800 about?"
        },
        {
            "name": "Following Too Closely",
            "question": "What does California Vehicle Code say about following too closely?"
        },
        {
            "name": "Integration Test",
            "question": "If YOLO detects a vehicle going 65 mph in a 45 mph zone, what CA Vehicle Code would apply?"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"TEST {i}: {test['name']}")
        print("-" * 80)
        print(f"Question: {test['question']}")
        print()
        print("Response:")
        response = test_legal_chat(test['question'])
        print(response)
        print()
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()
