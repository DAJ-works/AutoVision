#!/usr/bin/env python3
"""
Test the chatbot with a query that matches loaded rules
"""

import requests
import json

url = "http://localhost:5001/api/legal-chat"

# Test with a stop sign question (which should match CVC 22450)
payload = {
    "message": "What are the requirements for stopping at a stop sign in California?"
}

print("\n" + "="*60)
print("Testing California Vehicle Code Chat")
print("="*60)
print(f"\nQuestion: {payload['message']}\n")

try:
    response = requests.post(url, json=payload, timeout=120)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Response:")
        print("-" * 60)
        print(data.get('response', 'No response'))
        print("-" * 60)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60 + "\n")
