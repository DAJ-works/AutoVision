"""
Test script for RAG-based incident analyzer.

This script validates the entire RAG pipeline:
1. Loads example YOLOv8 detections
2. Embeds CA vehicle rules
3. Retrieves relevant rules
4. Calls Ollama Llama 3.1
5. Returns fault analysis
"""

import sys
import json
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from rag_utils import analyze_incident, RAGIncidentAnalyzer


def test_basic_components():
    """Test individual components before full integration."""
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*70 + "\n")
    
    analyzer = RAGIncidentAnalyzer(
        rules_path="backend/data/ca_vehicle_rules.jsonl",
        ollama_model="llama3.1"
    )
    
    # Test 1: Parse YOLOv8 JSON
    print("1. Testing YOLOv8 JSON parsing...")
    try:
        yolo_data = analyzer.parse_yolo_json("backend/data/example_yolo_output.json")
        print(f"   ✓ Successfully parsed. Found {len(yolo_data.get('detections', []))} detections")
        print(f"   Camera: {yolo_data.get('camera_id')}, Frame: {yolo_data.get('frame_id')}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: Build incident summary
    print("\n2. Testing incident summary generation...")
    try:
        summary = analyzer.build_incident_summary(yolo_data)
        print(f"   ✓ Generated summary ({len(summary)} chars)")
        print("\n   Preview:")
        for line in summary.split('\n')[:8]:
            print(f"   {line}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 3: Load and embed rules
    print("\n3. Testing rule loading and embedding...")
    try:
        analyzer.load_and_embed_rules()
        print(f"   ✓ Rules loaded and embedded successfully")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        print(f"   Note: Ensure sentence-transformers and chromadb are installed:")
        print(f"   pip install sentence-transformers chromadb")
        return False
    
    # Test 4: Retrieve relevant rules
    print("\n4. Testing rule retrieval...")
    try:
        rules = analyzer.retrieve_relevant_rules(summary, top_k=3)
        print(f"   ✓ Retrieved {len(rules)} relevant rules")
        for i, rule in enumerate(rules, 1):
            code = rule['metadata'].get('code', 'N/A')
            title = rule['document'].split('\n')[0][:50]
            print(f"   {i}. [{code}] {title}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 5: Build prompt
    print("\n5. Testing prompt building...")
    try:
        prompt = analyzer.build_prompt(summary, rules)
        print(f"   ✓ Built prompt ({len(prompt)} chars)")
        print("\n   Prompt preview (first 300 chars):")
        print("   " + "-"*60)
        for line in prompt[:300].split('\n'):
            print(f"   {line}")
        print("   " + "-"*60)
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("ALL COMPONENT TESTS PASSED ✓")
    print("="*70 + "\n")
    return True


def test_full_pipeline():
    """Test the complete end-to-end pipeline."""
    print("\n" + "="*70)
    print("TESTING FULL RAG PIPELINE WITH OLLAMA")
    print("="*70 + "\n")
    
    print("Note: This requires Ollama to be running with llama3.1 model.")
    print("If not installed:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Run: ollama pull llama3.1")
    print("  3. Verify: ollama list\n")
    
    response = input("Proceed with Ollama test? (y/n): ").strip().lower()
    if response != 'y':
        print("\nSkipping Ollama test. You can test manually with:")
        print("  python backend/rag_utils.py backend/data/example_yolo_output.json")
        return True
    
    print("\nRunning full analysis...")
    try:
        result = analyze_incident(
            yolo_json_path="backend/data/example_yolo_output.json",
            rules_path="backend/data/ca_vehicle_rules.jsonl",
            top_k_rules=5
        )
        
        print("\n" + "="*70)
        print("ANALYSIS RESULT FROM LLAMA 3.1")
        print("="*70)
        print(result)
        print("="*70 + "\n")
        
        print("✓ Full pipeline test PASSED")
        return True
        
    except FileNotFoundError as e:
        if "ollama" in str(e).lower():
            print(f"\n✗ Ollama not found: {e}")
            print("\nTo fix:")
            print("  1. Install Ollama from https://ollama.ai")
            print("  2. Run: ollama pull llama3.1")
            print("  3. Verify with: ollama list")
        else:
            print(f"\n✗ File not found: {e}")
        return False
        
    except TimeoutError as e:
        print(f"\n✗ Timeout: {e}")
        print("The model may be processing. Try increasing timeout or simplify the prompt.")
        return False
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_detection():
    """Test with a simple custom detection scenario."""
    print("\n" + "="*70)
    print("TESTING WITH CUSTOM DETECTION SCENARIO")
    print("="*70 + "\n")
    
    # Create a simple test case
    custom_detection = {
        "camera_id": "test_cam",
        "timestamp": "2025-11-08T15:00:00Z",
        "frame_id": 100,
        "detections": [
            {
                "label": "car",
                "confidence": 0.95,
                "bbox": [100, 200, 300, 400],
                "color": "silver",
                "direction": "southbound",
                "speed_estimate": 55
            },
            {
                "label": "stop_sign",
                "confidence": 0.92,
                "bbox": [500, 100, 600, 250]
            }
        ],
        "incident_notes": "Silver car appears to pass through intersection without stopping at stop sign"
    }
    
    # Save to temp file
    temp_file = "backend/data/test_custom_detection.json"
    with open(temp_file, 'w') as f:
        json.dump(custom_detection, f, indent=2)
    
    print(f"Created test detection file: {temp_file}")
    print("\nDetection summary:")
    print(json.dumps(custom_detection, indent=2))
    
    print("\nAnalyzing...")
    try:
        result = analyze_incident(
            yolo_json_path=temp_file,
            rules_path="backend/data/ca_vehicle_rules.jsonl",
            top_k_rules=3
        )
        
        print("\n" + "="*70)
        print("CUSTOM SCENARIO ANALYSIS")
        print("="*70)
        print(result)
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAG INCIDENT ANALYZER - VALIDATION SUITE")
    print("="*70)
    
    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import sentence_transformers
        print("  ✓ sentence-transformers installed")
    except ImportError:
        print("  ✗ sentence-transformers NOT installed")
        print("    Install: pip install sentence-transformers")
    
    try:
        import chromadb
        print("  ✓ chromadb installed")
    except ImportError:
        print("  ✗ chromadb NOT installed")
        print("    Install: pip install chromadb")
    
    # Run tests
    success = True
    
    # Test 1: Components
    if not test_basic_components():
        success = False
        print("\n⚠ Component tests failed. Fix issues before proceeding.")
        sys.exit(1)
    
    # Test 2: Full pipeline (optional - requires Ollama)
    if not test_full_pipeline():
        print("\n⚠ Full pipeline test skipped or failed.")
        print("You can test manually once Ollama is set up.")
    
    # Test 3: Custom scenario (optional)
    response = input("\nTest custom detection scenario? (y/n): ").strip().lower()
    if response == 'y':
        test_custom_detection()
    
    print("\n" + "="*70)
    if success:
        print("VALIDATION COMPLETE ✓")
    else:
        print("VALIDATION COMPLETED WITH WARNINGS")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("1. Ensure Ollama is installed and llama3.1 model is pulled")
    print("2. Run analysis: python backend/rag_utils.py backend/data/example_yolo_output.json")
    print("3. Integrate into your YOLOv8 pipeline by calling analyze_incident()")
    print()
