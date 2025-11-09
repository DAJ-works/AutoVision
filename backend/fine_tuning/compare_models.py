"""
Compare base Llama 3.1 vs custom traffic-analyst model.
"""

import sys
sys.path.insert(0, 'backend')
from rag_utils import RAGIncidentAnalyzer

def compare_models(yolo_json: str = "backend/data/example_yolo_output.json"):
    """Compare base vs custom model on same incident."""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: llama3.1 vs traffic-analyst")
    print("="*70 + "\n")
    
    # Test with base model
    print("1️⃣  Testing BASE MODEL (llama3.1)...")
    print("-" * 70)
    analyzer_base = RAGIncidentAnalyzer(ollama_model="llama3.1")
    result_base = analyzer_base.analyze_incident(yolo_json, top_k_rules=3)
    print("\nBase Model Result:")
    print(result_base)
    
    print("\n" + "="*70 + "\n")
    
    # Test with custom model
    print("2️⃣  Testing CUSTOM MODEL (traffic-analyst)...")
    print("-" * 70)
    analyzer_custom = RAGIncidentAnalyzer(ollama_model="traffic-analyst")
    result_custom = analyzer_custom.analyze_incident(yolo_json, top_k_rules=3)
    print("\nCustom Model Result:")
    print(result_custom)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")
    
    # Simple metrics
    print("Metrics:")
    print(f"  Base model length: {len(result_base)} chars")
    print(f"  Custom model length: {len(result_custom)} chars")
    print(f"  Base CVC citations: {result_base.count('CVC')}")
    print(f"  Custom CVC citations: {result_custom.count('CVC')}")
    print()

if __name__ == "__main__":
    compare_models()
