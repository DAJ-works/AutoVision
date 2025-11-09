"""
Create a custom Ollama Modelfile for traffic incident analysis.

This script generates a Modelfile that customizes Llama 3.1 with:
- Specialized system prompt for CA Vehicle Code reasoning
- Optimized parameters for incident analysis
- Example few-shot demonstrations

Usage:
    python backend/fine_tuning/create_ollama_modelfile.py
    ollama create traffic-analyst -f backend/fine_tuning/TrafficAnalystModelfile
    ollama run traffic-analyst
"""

import json
from pathlib import Path
from typing import List, Dict


def load_ca_rules(rules_path: str = "backend/data/ca_vehicle_rules.jsonl") -> List[Dict]:
    """Load CA Vehicle Code rules."""
    rules = []
    with open(rules_path, 'r') as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))
    return rules


def create_system_prompt(rules: List[Dict], include_examples: bool = True) -> str:
    """Create specialized system prompt for traffic analysis."""
    
    prompt_parts = [
        "You are an expert traffic incident analyst specializing in California Vehicle Code.",
        "Your role is to analyze traffic incidents from detection data and determine fault with precise legal reasoning.",
        "",
        "# Your Expertise",
        "- Deep knowledge of California Vehicle Code",
        "- Ability to map visual detections (vehicles, traffic signals, pedestrians) to legal violations",
        "- Evidence-based reasoning with specific rule citations",
        "- Clear, concise communication in 3-bullet format",
        "",
        "# California Vehicle Code Rules You Must Know",
        ""
    ]
    
    # Add top rules for quick reference
    key_rules = rules[:10]  # First 10 most important
    for rule in key_rules:
        code = rule.get('code', 'N/A')
        title = rule.get('title', 'Unknown')
        desc = rule.get('description', '')[:150] + '...' if len(rule.get('description', '')) > 150 else rule.get('description', '')
        prompt_parts.append(f"- **{code} ({title})**: {desc}")
    
    prompt_parts.extend([
        "",
        "# Analysis Format",
        "When analyzing incidents, you MUST:",
        "1. Identify all parties and their actions from detection data",
        "2. Match actions to specific CVC violations",
        "3. Provide exactly 3 bullet points, each with:",
        "   - Who is at fault (or not at fault)",
        "   - Specific CVC code violated",
        "   - Evidence from detection data (confidence, bbox, labels, states)",
        "   - Legal reasoning",
        "",
        "# Output Template",
        "• **[Party] is at fault for [action] ([CVC Code])**: [Evidence from detections] [Legal reasoning]",
        "• **[Party] violated [rule] ([CVC Code])**: [Evidence from detections] [Legal reasoning]",
        "• **[Party] is not at fault but [consideration]**: [Evidence] [Legal reasoning]",
        ""
    ])
    
    if include_examples:
        prompt_parts.extend([
            "# Example Analysis",
            "",
            "Incident: Car detected (confidence 0.94) approaching intersection. Traffic light shows 'red' (confidence 0.89). Pedestrian in crosswalk (confidence 0.87).",
            "",
            "Your response:",
            "• **Driver is at fault for running red light (CVC 21453(a))**: Detection shows car (confidence 0.94) approaching intersection while traffic light state is 'red' (confidence 0.89). Driver failed to stop before entering intersection as required by CVC 21453(a).",
            "• **Driver violated pedestrian right-of-way (CVC 21950(a))**: Pedestrian detected in crosswalk (confidence 0.87). Driver must yield to pedestrians in marked crosswalks per CVC 21950(a), but proceeded into intersection.",
            "• **Pedestrian followed traffic laws**: No evidence of violation. Pedestrian was lawfully using crosswalk during permitted crossing time.",
            ""
        ])
    
    prompt_parts.extend([
        "# Key Principles",
        "- Always cite specific CVC codes",
        "- Reference detection confidence scores and labels",
        "- Consider all parties (vehicles, pedestrians, cyclists)",
        "- If uncertain, state limitations in evidence",
        "- Be precise: specify colors, directions, positions from detection data",
        "",
        "# Remember",
        "Your analysis determines legal liability. Be thorough, precise, and evidence-based."
    ])
    
    return "\n".join(prompt_parts)


def create_modelfile(
    base_model: str = "llama3.1",
    custom_name: str = "traffic-analyst",
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 40,
    rules_path: str = "backend/data/ca_vehicle_rules.jsonl",
    output_path: str = "backend/fine_tuning/TrafficAnalystModelfile"
):
    """
    Create Ollama Modelfile for traffic incident analysis.
    
    Parameters:
        base_model: Base Ollama model to customize
        custom_name: Name for the custom model
        temperature: Lower = more deterministic (0.1-0.5 recommended)
        top_p: Nucleus sampling (0.9 recommended)
        top_k: Top-K sampling (40 recommended)
        rules_path: Path to CA Vehicle Code rules
        output_path: Where to save Modelfile
    """
    
    # Load rules
    print(f"Loading CA Vehicle Code rules from {rules_path}...")
    rules = load_ca_rules(rules_path)
    print(f"Loaded {len(rules)} rules")
    
    # Create system prompt
    print("Creating specialized system prompt...")
    system_prompt = create_system_prompt(rules, include_examples=True)
    
    # Build Modelfile
    modelfile_lines = [
        f"# Traffic Incident Analyst - Custom Llama 3.1 Model",
        f"# Base model: {base_model}",
        f"# Created: {Path().cwd()}",
        f"# Purpose: Specialized reasoning for traffic incident fault analysis",
        "",
        f"FROM {base_model}",
        "",
        "# Set parameters for deterministic, precise reasoning",
        f"PARAMETER temperature {temperature}",
        f"PARAMETER top_p {top_p}",
        f"PARAMETER top_k {top_k}",
        "PARAMETER num_ctx 4096",  # Context window
        "PARAMETER stop \"\\n\\n---\"",  # Stop sequence
        "",
        "# System prompt with CA Vehicle Code expertise",
        f'SYSTEM """',
        system_prompt,
        '"""',
        "",
        "# Optional: Add a template for consistent formatting",
        'TEMPLATE """{{ if .System }}System: {{ .System }}{{ end }}',
        "",
        'User: {{ .Prompt }}',
        "",
        'Assistant:"""',
        ""
    ]
    
    # Write Modelfile
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(modelfile_lines))
    
    print(f"\n✓ Modelfile created: {output_path}")
    print(f"\nTo create your custom model, run:")
    print(f"  ollama create {custom_name} -f {output_path}")
    print(f"\nTo use it:")
    print(f"  ollama run {custom_name}")
    print(f"\nOr in Python:")
    print(f"  analyzer = RAGIncidentAnalyzer(ollama_model='{custom_name}')")
    
    return output_path


def create_comparison_script(output_path: str = "backend/fine_tuning/compare_models.py"):
    """Create a script to compare base vs custom model."""
    
    script_content = '''"""
Compare base Llama 3.1 vs custom traffic-analyst model.
"""

import sys
sys.path.insert(0, 'backend')
from rag_utils import RAGIncidentAnalyzer

def compare_models(yolo_json: str = "backend/data/example_yolo_output.json"):
    """Compare base vs custom model on same incident."""
    
    print("\\n" + "="*70)
    print("MODEL COMPARISON: llama3.1 vs traffic-analyst")
    print("="*70 + "\\n")
    
    # Test with base model
    print("1️⃣  Testing BASE MODEL (llama3.1)...")
    print("-" * 70)
    analyzer_base = RAGIncidentAnalyzer(ollama_model="llama3.1")
    result_base = analyzer_base.analyze_incident(yolo_json, top_k_rules=3)
    print("\\nBase Model Result:")
    print(result_base)
    
    print("\\n" + "="*70 + "\\n")
    
    # Test with custom model
    print("2️⃣  Testing CUSTOM MODEL (traffic-analyst)...")
    print("-" * 70)
    analyzer_custom = RAGIncidentAnalyzer(ollama_model="traffic-analyst")
    result_custom = analyzer_custom.analyze_incident(yolo_json, top_k_rules=3)
    print("\\nCustom Model Result:")
    print(result_custom)
    
    print("\\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\\n")
    
    # Simple metrics
    print("Metrics:")
    print(f"  Base model length: {len(result_base)} chars")
    print(f"  Custom model length: {len(result_custom)} chars")
    print(f"  Base CVC citations: {result_base.count('CVC')}")
    print(f"  Custom CVC citations: {result_custom.count('CVC')}")
    print()

if __name__ == "__main__":
    compare_models()
'''
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"✓ Comparison script created: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OLLAMA MODELFILE GENERATOR")
    print("="*70 + "\n")
    
    # Create Modelfile
    modelfile_path = create_modelfile(
        base_model="llama3.1",
        custom_name="traffic-analyst",
        temperature=0.3,  # Lower for more consistent reasoning
        output_path="backend/fine_tuning/TrafficAnalystModelfile"
    )
    
    print("\n" + "-"*70)
    
    # Create comparison script
    create_comparison_script()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Create the custom model:")
    print("   ollama create traffic-analyst -f backend/fine_tuning/TrafficAnalystModelfile")
    print("\n2. Test it:")
    print("   ollama run traffic-analyst")
    print("\n3. Compare with base model:")
    print("   /Users/Jayanth/Desktop/idnaraiytuk/.venv/bin/python backend/fine_tuning/compare_models.py")
    print("\n4. Use in your pipeline:")
    print("   from backend.rag_utils import analyze_incident")
    print("   result = analyze_incident('detections.json', ollama_model='traffic-analyst')")
    print("\n" + "="*70 + "\n")
