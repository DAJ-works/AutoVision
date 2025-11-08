"""
Generate supervised fine-tuning dataset from YOLOv8 detections + CA rules.

This creates JSONL datasets in the format needed for:
- Ollama fine-tuning (if/when supported)
- External fine-tuning (Hugging Face, Axolotl, etc.)
- Few-shot prompt engineering

Output format:
{
  "instruction": "Analyze this traffic incident...",
  "input": "Incident details with detections...",
  "output": "3-bullet fault analysis with CVC citations..."
}
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class TrainingDataGenerator:
    """Generate fine-tuning datasets for traffic incident analysis."""
    
    def __init__(
        self,
        rules_path: str = "backend/data/ca_vehicle_rules.jsonl",
        output_dir: str = "backend/fine_tuning/data"
    ):
        self.rules_path = rules_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load rules
        self.rules = self._load_rules()
        print(f"Loaded {len(self.rules)} CA Vehicle Code rules")
    
    def _load_rules(self) -> List[Dict]:
        """Load CA rules from JSONL."""
        rules = []
        with open(self.rules_path, 'r') as f:
            for line in f:
                if line.strip():
                    rules.append(json.loads(line))
        return rules
    
    def create_synthetic_scenarios(self, num_scenarios: int = 50) -> List[Dict]:
        """
        Create synthetic traffic scenarios for training.
        
        These are realistic combinations of detections + violations.
        """
        scenarios = []
        
        # Scenario templates
        templates = [
            # Red light violations
            {
                "description": "Vehicle runs red light",
                "detections": [
                    {"label": "car", "confidence": 0.95, "bbox": [100, 200, 300, 400], "color": "red"},
                    {"label": "traffic_light", "confidence": 0.92, "bbox": [50, 50, 80, 120], "state": "red"}
                ],
                "applicable_rules": ["CVC 21453(a)"],
                "fault": "car",
                "reasoning": "Car approached intersection while traffic light showed red signal. Driver failed to stop before entering intersection."
            },
            
            # Pedestrian violations
            {
                "description": "Vehicle fails to yield to pedestrian in crosswalk",
                "detections": [
                    {"label": "car", "confidence": 0.93, "bbox": [200, 300, 400, 500], "color": "blue"},
                    {"label": "person", "confidence": 0.89, "bbox": [350, 400, 400, 600], "location": "crosswalk"}
                ],
                "applicable_rules": ["CVC 21950(a)"],
                "fault": "car",
                "reasoning": "Pedestrian was lawfully in crosswalk. Driver failed to yield right-of-way."
            },
            
            # Stop sign violations
            {
                "description": "Vehicle fails to stop at stop sign",
                "detections": [
                    {"label": "car", "confidence": 0.96, "bbox": [150, 250, 350, 450], "speed_estimate": 35},
                    {"label": "stop_sign", "confidence": 0.94, "bbox": [50, 100, 100, 200]}
                ],
                "applicable_rules": ["CVC 21802(a)", "CVC 22450(a)"],
                "fault": "car",
                "reasoning": "Car approached stop sign but no indication of stopping before entering intersection."
            },
            
            # Following too closely
            {
                "description": "Vehicle following too closely",
                "detections": [
                    {"label": "car", "confidence": 0.97, "bbox": [100, 200, 300, 400], "color": "silver", "direction": "northbound"},
                    {"label": "car", "confidence": 0.95, "bbox": [95, 380, 295, 580], "color": "black", "direction": "northbound"}
                ],
                "applicable_rules": ["CVC 21703"],
                "fault": "car (black)",
                "reasoning": "Insufficient distance between vehicles given speed and road conditions."
            },
            
            # Unsafe lane change
            {
                "description": "Vehicle makes unsafe lane change",
                "detections": [
                    {"label": "car", "confidence": 0.94, "bbox": [200, 300, 400, 500], "color": "white"},
                    {"label": "car", "confidence": 0.92, "bbox": [380, 320, 580, 520], "color": "red"}
                ],
                "applicable_rules": ["CVC 22107", "CVC 21658(a)"],
                "fault": "car (white)",
                "reasoning": "Lane change made without reasonable safety and no signal given."
            },
            
            # Green light with pedestrian
            {
                "description": "Vehicle with green light must still yield to pedestrian",
                "detections": [
                    {"label": "car", "confidence": 0.96, "bbox": [100, 200, 300, 400], "color": "blue"},
                    {"label": "traffic_light", "confidence": 0.91, "bbox": [50, 50, 80, 120], "state": "green"},
                    {"label": "person", "confidence": 0.87, "bbox": [250, 350, 300, 550], "location": "crosswalk"}
                ],
                "applicable_rules": ["CVC 21451(a)", "CVC 21950(a)"],
                "fault": "car",
                "reasoning": "Despite green signal, driver must yield to pedestrians lawfully in intersection."
            },
            
            # Multiple violations
            {
                "description": "Reckless driving through intersection",
                "detections": [
                    {"label": "car", "confidence": 0.98, "bbox": [150, 250, 350, 450], "color": "black", "speed_estimate": 65},
                    {"label": "traffic_light", "confidence": 0.89, "bbox": [50, 50, 80, 120], "state": "red"},
                    {"label": "person", "confidence": 0.85, "bbox": [300, 400, 350, 600], "location": "crosswalk"}
                ],
                "applicable_rules": ["CVC 21453(a)", "CVC 21950(a)", "CVC 22350", "CVC 23103(a)"],
                "fault": "car",
                "reasoning": "Driver ran red light at excessive speed while pedestrian was in crosswalk, showing willful disregard for safety."
            }
        ]
        
        # Generate variations
        for _ in range(num_scenarios):
            template = random.choice(templates)
            
            # Create variation
            scenario = {
                "camera_id": f"cam_{random.randint(1, 10):02d}",
                "timestamp": datetime.now().isoformat(),
                "frame_id": random.randint(1000, 9999),
                "detections": template["detections"],
                "description": template["description"],
                "applicable_rules": template["applicable_rules"],
                "expected_fault": template["fault"],
                "reasoning": template["reasoning"]
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def scenario_to_training_example(self, scenario: Dict) -> Dict:
        """Convert scenario to training example format."""
        
        # Build input (what the model sees)
        input_parts = [
            f"Camera: {scenario['camera_id']}, Frame: {scenario['frame_id']}, Time: {scenario['timestamp']}",
            "",
            "Detections:"
        ]
        
        for i, det in enumerate(scenario['detections'], 1):
            det_str = f"{i}. {det['label']} (confidence: {det['confidence']:.2f}) at bbox {det['bbox']}"
            if 'color' in det:
                det_str += f", color: {det['color']}"
            if 'state' in det:
                det_str += f", state: {det['state']}"
            if 'location' in det:
                det_str += f", location: {det['location']}"
            if 'speed_estimate' in det:
                det_str += f", speed: ~{det['speed_estimate']} mph"
            input_parts.append(det_str)
        
        input_text = "\n".join(input_parts)
        
        # Build expected output (3-bullet analysis)
        # Get rule details
        rule_texts = []
        for rule_code in scenario['applicable_rules']:
            for rule in self.rules:
                if rule.get('code') == rule_code:
                    rule_texts.append(f"{rule_code}: {rule.get('title', 'Unknown')}")
                    break
        
        output_parts = [
            f"• **{scenario['expected_fault']} is at fault for {scenario['description'].lower()} ({', '.join(scenario['applicable_rules'])})**: {scenario['reasoning']}"
        ]
        
        # Add 2 more contextual bullets
        if len(scenario['detections']) > 1:
            output_parts.append(
                "• **Other parties showed no violations**: Based on available detection data, no other vehicles or pedestrians exhibited behavior inconsistent with traffic laws."
            )
        
        output_parts.append(
            f"• **Applicable rules**: {'; '.join(rule_texts)}"
        )
        
        output_text = "\n".join(output_parts)
        
        # Standard instruction
        instruction = (
            "You are a traffic incident analyst. Analyze the following incident and determine fault. "
            "Provide exactly 3 bullet points with: who is at fault, applicable CVC code, evidence from detections, and legal reasoning."
        )
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
    
    def generate_training_dataset(
        self,
        num_examples: int = 100,
        output_file: str = "traffic_incidents_training.jsonl",
        format: str = "alpaca"  # "alpaca", "sharegpt", "ollama"
    ):
        """
        Generate full training dataset.
        
        Formats:
        - alpaca: {"instruction": ..., "input": ..., "output": ...}
        - sharegpt: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
        - ollama: {"prompt": ..., "completion": ...}
        """
        print(f"\nGenerating {num_examples} training examples...")
        
        # Create scenarios
        scenarios = self.create_synthetic_scenarios(num_examples)
        
        # Convert to training examples
        training_data = []
        for scenario in scenarios:
            example = self.scenario_to_training_example(scenario)
            
            # Convert to requested format
            if format == "alpaca":
                formatted = example
            elif format == "sharegpt":
                formatted = {
                    "conversations": [
                        {"from": "system", "value": example["instruction"]},
                        {"from": "human", "value": example["input"]},
                        {"from": "gpt", "value": example["output"]}
                    ]
                }
            elif format == "ollama":
                formatted = {
                    "prompt": f"{example['instruction']}\n\n{example['input']}",
                    "completion": example["output"]
                }
            else:
                raise ValueError(f"Unknown format: {format}")
            
            training_data.append(formatted)
        
        # Write to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"✓ Wrote {len(training_data)} examples to {output_path}")
        print(f"  Format: {format}")
        print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return output_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING DATA GENERATOR")
    print("="*70 + "\n")
    
    generator = TrainingDataGenerator()
    
    # Generate datasets in multiple formats
    print("Generating training datasets...")
    print("-" * 70)
    
    # Alpaca format (most common)
    generator.generate_training_dataset(
        num_examples=100,
        output_file="traffic_incidents_alpaca.jsonl",
        format="alpaca"
    )
    
    # ShareGPT format (for some trainers)
    generator.generate_training_dataset(
        num_examples=100,
        output_file="traffic_incidents_sharegpt.jsonl",
        format="sharegpt"
    )
    
    # Ollama format
    generator.generate_training_dataset(
        num_examples=100,
        output_file="traffic_incidents_ollama.jsonl",
        format="ollama"
    )
    
    print("\n" + "="*70)
    print("DATASETS CREATED")
    print("="*70)
    print("\nNext steps:")
    print("1. For Ollama (when supported):")
    print("   ollama create traffic-analyst-finetuned -f <training_file>")
    print("\n2. For external fine-tuning:")
    print("   - Use Hugging Face Trainer with traffic_incidents_alpaca.jsonl")
    print("   - Use Axolotl with traffic_incidents_sharegpt.jsonl")
    print("   - Export to GGUF and load back to Ollama")
    print("\n3. For few-shot prompting:")
    print("   - Use examples from JSONL as in-context demonstrations")
    print("\n" + "="*70 + "\n")
