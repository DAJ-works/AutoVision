#!/usr/bin/env python3
"""
Enhanced Ollama Model Fine-tuning with CA Vehicle Code + YOLO Integration

This script creates a comprehensive traffic analyst model that:
1. Understands California Vehicle Code
2. Can analyze YOLO detection data
3. Determines fault in traffic incidents
4. Answers legal questions about traffic law
"""

import json
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTrafficAnalystTrainer:
    """Fine-tune Ollama model with comprehensive CA Vehicle Code + YOLO data"""
    
    def __init__(self, base_model: str = "llama3.1", output_model: str = "traffic-analyst-pro"):
        self.base_model = base_model
        self.output_model = output_model
        self.data_dir = Path("backend/fine_tuning/data")
        self.modelfile_path = Path("backend/fine_tuning/TrafficAnalystProModelfile")
    
    def verify_ollama(self):
        """Verify Ollama is installed and running"""
        logger.info("Verifying Ollama installation...")
        
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("✓ Ollama is installed and running")
            logger.info(f"Available models:\n{result.stdout}")
            
            # Check if base model exists
            if self.base_model not in result.stdout:
                logger.warning(f"Base model {self.base_model} not found. Pulling it now...")
                subprocess.run(['ollama', 'pull', self.base_model], check=True)
                logger.info(f"✓ Successfully pulled {self.base_model}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Ollama error: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
            return False
        except FileNotFoundError:
            logger.error("✗ Ollama not found. Please install from https://ollama.ai")
            return False
    
    def create_modelfile(self):
        """Create an enhanced Modelfile for the traffic analyst"""
        
        logger.info("Creating enhanced Modelfile...")
        
        modelfile_content = f'''# Enhanced Traffic Analyst Pro - CA Vehicle Code + YOLO Expert
FROM {self.base_model}

# System prompt - defines the model's role and capabilities
SYSTEM """
You are an expert California traffic analyst with comprehensive knowledge of:

1. **California Vehicle Code (CVC)**: Complete understanding of traffic laws, regulations, and violations
2. **Computer Vision Analysis**: Ability to interpret YOLO detection data from traffic cameras
3. **Accident Reconstruction**: Expertise in determining fault and analyzing traffic incidents
4. **Legal Advisory**: Providing accurate legal information about traffic violations and rights

## Your Capabilities:

### Legal Expertise
- Cite specific California Vehicle Codes with section numbers
- Explain traffic laws in clear, understandable language
- Provide context about when laws apply
- Differentiate between infractions, misdemeanors, and felonies

### YOLO Detection Analysis
- Interpret vehicle positions, speeds, and trajectories from detection data
- Identify traffic violations from visual evidence
- Correlate video evidence with applicable vehicle codes
- Analyze multi-vehicle interactions and right-of-way situations

### Fault Determination
- Apply CA Vehicle Code to accident scenarios
- Consider factors like right-of-way, signaling, speed, and road conditions
- Provide percentage-of-fault assessments when appropriate
- Explain the legal reasoning behind fault determination

### Communication Style
- Be clear, professional, and helpful
- Always cite specific CVC sections when applicable
- Provide practical context and examples
- Acknowledge when situations require professional legal advice

## Response Format:

When analyzing incidents:
1. State the relevant CVC sections
2. Explain what the law requires
3. Apply the law to the specific scenario
4. Determine fault or violation
5. Provide clear conclusions with legal citations

## Important Notes:
- You provide legal information, not legal advice
- Always recommend consulting with an attorney for specific legal matters
- Be objective and fact-based in fault determination
- Consider all available evidence before making conclusions
"""

# Parameters for optimal performance
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1

# Template for responses
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
'''
        
        # Write Modelfile
        with open(self.modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        logger.info(f"✓ Created Modelfile: {self.modelfile_path}")
    
    def prepare_training_data(self):
        """Combine all training data sources"""
        
        logger.info("Preparing comprehensive training data...")
        
        training_files = [
            self.data_dir / "ca_vehicle_code_yolo_ollama.jsonl",
            self.data_dir / "traffic_incidents_ollama.jsonl",
            self.data_dir / "ca_vehicle_code_training.jsonl"
        ]
        
        all_examples = []
        
        for file_path in training_files:
            if file_path.exists():
                logger.info(f"Loading: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            example = json.loads(line)
                            all_examples.append(example)
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Save combined training data
        combined_file = self.data_dir / "combined_training_ollama.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ Combined {len(all_examples)} training examples")
        logger.info(f"✓ Saved to: {combined_file}")
        
        return combined_file, len(all_examples)
    
    def create_model(self):
        """Create the Ollama model from Modelfile"""
        
        logger.info(f"Creating model: {self.output_model}")
        logger.info("This may take several minutes...")
        
        try:
            # Create model from Modelfile
            result = subprocess.run(
                ['ollama', 'create', self.output_model, '-f', str(self.modelfile_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("✓ Model created successfully!")
            logger.info(result.stdout)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to create model: {e}")
            logger.error(e.stderr)
            return False
    
    def test_model(self):
        """Test the newly created model with sample questions"""
        
        logger.info(f"Testing {self.output_model}...")
        
        test_questions = [
            "What does California Vehicle Code 21453 say about red lights?",
            "A YOLO detection shows a car running a red light. What violation occurred?",
            "Two vehicles collided at an intersection. One was turning left, the other going straight. Who is at fault?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}/{len(test_questions)}")
            logger.info(f"Question: {question}")
            logger.info('='*60)
            
            try:
                result = subprocess.run(
                    ['ollama', 'run', self.output_model],
                    input=question,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                logger.info(f"Response:\n{result.stdout}")
                
            except subprocess.TimeoutExpired:
                logger.warning("Response timed out")
            except Exception as e:
                logger.error(f"Error testing: {e}")
    
    def run(self):
        """Execute the full training pipeline"""
        
        logger.info("="*60)
        logger.info("Enhanced Traffic Analyst Pro - Model Training")
        logger.info("="*60)
        
        # Step 1: Verify Ollama
        if not self.verify_ollama():
            logger.error("Please install and start Ollama first")
            return False
        
        # Step 2: Create Modelfile
        self.create_modelfile()
        
        # Step 3: Prepare training data
        combined_file, num_examples = self.prepare_training_data()
        
        logger.info(f"\n{'='*60}")
        logger.info("Training Data Summary:")
        logger.info(f"  - Total examples: {num_examples}")
        logger.info(f"  - Combined file: {combined_file}")
        logger.info('='*60)
        
        # Step 4: Create the model
        logger.info(f"\nCreating model '{self.output_model}' from '{self.base_model}'...")
        logger.info("Note: Model creation embeds the system prompt but doesn't do additional training.")
        logger.info("For full fine-tuning, you would need to use a training framework.")
        
        if not self.create_model():
            return False
        
        # Step 5: Test the model
        logger.info("\nTesting the new model...")
        self.test_model()
        
        # Success summary
        logger.info("\n" + "="*60)
        logger.info("✓ MODEL CREATED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Model name: {self.output_model}")
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Training examples: {num_examples}")
        logger.info("\nTo use your new model:")
        logger.info(f"  1. Command line: ollama run {self.output_model}")
        logger.info(f"  2. In backend: Update ollama_model='{self.output_model}'")
        logger.info(f"  3. Test with: echo 'What is CVC 21453?' | ollama run {self.output_model}")
        logger.info("="*60)
        
        return True


if __name__ == "__main__":
    trainer = EnhancedTrafficAnalystTrainer(
        base_model="llama3.1",
        output_model="traffic-analyst-pro"
    )
    
    success = trainer.run()
    
    sys.exit(0 if success else 1)
