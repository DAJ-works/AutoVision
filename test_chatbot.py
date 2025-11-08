#!/usr/bin/env python3
"""
Test script to verify the chatbot/Ollama integration is working
"""

import requests
import json

# Test the legal assistant chat endpoint
def test_legal_chat():
    print("\n" + "="*60)
    print("Testing Legal Assistant Chat Endpoint")
    print("="*60 + "\n")
    
    url = "http://localhost:5001/api/legal-chat"
    
    payload = {
        "message": "What is California Vehicle Code 21453?"
    }
    
    print(f"Sending request to: {url}")
    print(f"Message: {payload['message']}\n")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS! Response received:")
            print("-" * 60)
            print(data.get('response', 'No response field'))
            print("-" * 60)
        else:
            print(f"\n❌ ERROR! Status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\n❌ ERROR: Request timed out after 120 seconds")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to backend. Is it running on port 5001?")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

# Test if backend is responding
def test_backend_alive():
    print("\n" + "="*60)
    print("Testing Backend Connection")
    print("="*60 + "\n")
    
    try:
        response = requests.get("http://localhost:5001/api/cases", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and responding!")
            return True
        else:
            print(f"⚠️  Backend responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not responding. Please start it with:")
        print("   cd /Users/Jayanth/Desktop/idnaraiytuk")
        print("   source venv/bin/activate")
        print("   python backend/api/app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🤖 AutoVision Chatbot Integration Test")
    print("="*60)
    
    # First check if backend is alive
    if test_backend_alive():
        # Then test the chat
        test_legal_chat()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60 + "\n")
