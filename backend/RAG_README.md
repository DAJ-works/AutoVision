# RAG-Based Incident Analyzer

## Overview

This module integrates **Llama 3.1** (via Ollama) with **YOLOv8** detections to perform automated traffic incident fault analysis using California Vehicle Code rules. It uses RAG (Retrieval-Augmented Generation) to retrieve relevant legal rules and generate evidence-based analysis.

## Architecture

```
YOLOv8 Detections (JSON)
          ↓
    Parse & Summarize
          ↓
    Embed CA Vehicle Rules ──→ ChromaDB Vector Store
          ↓
    Retrieve Top-K Rules
          ↓
    Build Contextual Prompt
          ↓
    Llama 3.1 (Ollama) ──→ Fault Analysis Output
```

## Files Created

- **`backend/rag_utils.py`** - Main RAG module with `analyze_incident()` function
- **`backend/data/ca_vehicle_rules.jsonl`** - 20 CA Vehicle Code rules (JSONL format)
- **`backend/data/example_yolo_output.json`** - Sample YOLOv8 detection output
- **`backend/test_rag.py`** - Comprehensive test suite
- **`requirements.txt`** - Updated with RAG dependencies

## Setup

### 1. Install Dependencies

```bash
cd /Users/Jayanth/Desktop/idnaraiytuk
pip install -r requirements.txt
```

**New dependencies added:**
- `sentence-transformers>=2.2.0` - For embedding CA rules and queries
- `chromadb>=0.4.0` - Vector database for semantic search
- `pydantic>=2.0.0` - Data validation

### 2. Install Ollama and Llama 3.1

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull Llama 3.1 model
ollama pull llama3.1

# Verify installation
ollama list
```

Expected output:
```
NAME            ID              SIZE    MODIFIED
llama3.1:latest abc123def456    4.7 GB  X days ago
```

### 3. Verify Setup

Run the test suite to validate all components:

```bash
python backend/test_rag.py
```

This will:
1. Test YOLOv8 JSON parsing
2. Test incident summary generation
3. Load and embed CA vehicle rules into ChromaDB
4. Test rule retrieval
5. Test prompt building
6. (Optional) Test full Ollama integration

## Usage

### Option 1: As a Python Module

```python
from backend.rag_utils import analyze_incident

# Analyze a YOLOv8 detection file
result = analyze_incident("path/to/yolo_detections.json")
print(result)
```

### Option 2: Command Line

```bash
python backend/rag_utils.py backend/data/example_yolo_output.json
```

### Option 3: Advanced (Custom Configuration)

```python
from backend.rag_utils import RAGIncidentAnalyzer

analyzer = RAGIncidentAnalyzer(
    rules_path="backend/data/ca_vehicle_rules.jsonl",
    embedding_model="all-MiniLM-L6-v2",  # Fast, lightweight
    ollama_model="llama3.1",
    chroma_persist_dir="backend/data/chroma_db"
)

# Analyze incident
result = analyzer.analyze_incident(
    "backend/data/example_yolo_output.json",
    top_k_rules=5
)
print(result)
```

## YOLOv8 JSON Format

The module expects YOLOv8 detections in this format:

```json
{
  "camera_id": "cam_01",
  "timestamp": "2025-11-08T14:32:15.234Z",
  "frame_id": 1847,
  "detections": [
    {
      "label": "car",
      "confidence": 0.96,
      "bbox": [120, 340, 380, 580],
      "color": "blue",
      "direction": "eastbound"
    },
    {
      "label": "traffic_light",
      "confidence": 0.89,
      "bbox": [450, 50, 490, 120],
      "state": "red"
    }
  ]
}
```

**Required fields:**
- `detections` (list): Array of detection objects
- Each detection: `label`, `confidence`, `bbox`

**Optional fields:**
- `camera_id`, `timestamp`, `frame_id`
- Detection metadata: `color`, `direction`, `speed_estimate`, `state`

## How It Works

### 1. Load YOLOv8 Detections
Parses the JSON file and extracts vehicle, pedestrian, and traffic control detections.

### 2. Build Incident Summary
Converts raw detections into natural language:
```
Incident Analysis - Camera: cam_01, Frame: 1847, Time: 2025-11-08T14:32:15Z

Detected Objects and Events:
Object Summary:
  - car: 2 detected
  - traffic_light: 2 detected
  - person: 1 detected

High-Confidence Detections (>0.5):
  1. car (confidence: 0.96) at bbox [120, 340, 380, 580]
  2. car (confidence: 0.94) at bbox [620, 180, 850, 420]
  ...
```

### 3. Retrieve Relevant Rules
Uses sentence-transformers to embed the incident summary and searches ChromaDB for the top-k most relevant CA Vehicle Code rules.

Example retrieved rules:
- CVC 21453(a): Red Light Violation
- CVC 21950(a): Pedestrian Right-of-Way in Crosswalk
- CVC 22350: Basic Speed Law

### 4. Build Prompt
Combines incident summary + retrieved rules into a structured prompt:

```
You are a traffic incident analyst with expertise in California Vehicle Code.
Analyze the following traffic incident and determine fault based on the provided rules.

=== INCIDENT DETAILS ===
[Incident summary here]

=== RELEVANT CALIFORNIA VEHICLE CODE RULES ===
1. [CVC 21453(a)] Red Light Violation
   A driver facing a steady circular red signal alone shall stop...
2. [CVC 21950(a)] Pedestrian Right-of-Way in Crosswalk
   The driver of a vehicle shall yield the right-of-way to a pedestrian...

=== QUESTION ===
Based on the incident details and the California Vehicle Code rules above,
determine who is at fault. Provide your analysis in exactly 3 bullet points,
each with specific evidence from the incident and reference to the applicable rule code.
```

### 5. Query Llama 3.1
Sends the prompt to Ollama via subprocess and receives the model's fault analysis.

## Integration with Your YOLOv8 Pipeline

### Option A: Direct Integration

Modify your YOLOv8 processing script to call `analyze_incident()`:

```python
# your_yolo_processor.py
from backend.rag_utils import analyze_incident

# After YOLOv8 generates detections
detections = run_yolo_model(video_path)

# Save to JSON
with open("detections.json", "w") as f:
    json.dump(detections, f)

# Analyze
fault_analysis = analyze_incident("detections.json")
print(fault_analysis)
```

### Option B: Backend API Endpoint

Add an endpoint to `backend/api/app.py`:

```python
from fastapi import FastAPI, UploadFile
from backend.rag_utils import analyze_incident
import tempfile
import json

app = FastAPI()

@app.post("/api/analyze-incident")
async def analyze_incident_endpoint(file: UploadFile):
    # Save uploaded YOLOv8 JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        content = await file.read()
        f.write(content.decode())
        temp_path = f.name
    
    # Analyze
    result = analyze_incident(temp_path)
    
    return {"analysis": result}
```

### Option C: Real-Time Stream Processing

For live video analysis:

```python
from backend.rag_utils import RAGIncidentAnalyzer

# Initialize once
analyzer = RAGIncidentAnalyzer()
analyzer.load_and_embed_rules()  # Pre-load rules

# Process each frame batch
for frame_batch in video_stream:
    detections = yolo_model(frame_batch)
    
    # Trigger analysis on incident detection
    if is_incident(detections):
        # Create temp JSON
        incident_data = format_detections(detections)
        with open("temp_incident.json", "w") as f:
            json.dump(incident_data, f)
        
        # Analyze
        result = analyzer.analyze_incident("temp_incident.json")
        alert_system(result)
```

## Example Output

```
=== ANALYSIS RESULT ===

Based on the incident and California Vehicle Code:

• **Red car is at fault for running the red light (CVC 21453(a))**: 
  The northbound red car was detected approaching the intersection while 
  the traffic light state shows "red" for northbound traffic. The driver 
  failed to stop before entering the intersection, directly violating 
  CVC 21453(a).

• **Red car failed to yield to pedestrian in crosswalk (CVC 21950(a))**: 
  Detection shows a person (confidence 0.87) located in the crosswalk 
  with activity "crossing". The red car entered the intersection while 
  the pedestrian had right-of-way, violating CVC 21950(a).

• **Blue car is not at fault but must remain vigilant**: 
  The eastbound blue car has a green light (detected state "green") and 
  legal right to proceed. However, the driver must still yield to the 
  pedestrian in the crosswalk per CVC 21950(a) before continuing through 
  the intersection.
```

## Performance Notes

- **First run**: Downloads sentence-transformer model (~80MB) and builds ChromaDB index (~2-5 seconds)
- **Subsequent runs**: Rules are cached in ChromaDB, startup is instant
- **Ollama latency**: 2-10 seconds depending on prompt complexity and hardware
- **Memory**: ~500MB for embeddings + ChromaDB + Ollama model

## Troubleshooting

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### "chromadb not installed"
```bash
pip install chromadb
```

### "Ollama command not found"
1. Install Ollama: https://ollama.ai
2. Add to PATH or restart terminal
3. Verify: `ollama --version`

### "Model llama3.1 not found"
```bash
ollama pull llama3.1
ollama list  # Verify it's downloaded
```

### Slow ChromaDB initialization
The first run downloads the embedding model. Subsequent runs use cached embeddings.

### Empty or generic responses from Llama
- Ensure prompt includes specific detection details
- Increase `top_k_rules` to provide more context
- Check if YOLOv8 JSON has sufficient metadata (colors, directions, etc.)

## Customization

### Add More CA Vehicle Code Rules

Edit `backend/data/ca_vehicle_rules.jsonl` and add lines:

```jsonl
{"code": "CVC 21654", "title": "Slow-Moving Vehicles", "description": "Any vehicle proceeding upon a highway at a speed less than the normal speed of traffic moving in the same direction shall be driven in the right-hand lane...", "category": "lane_usage"}
```

Then force-reload:

```python
analyzer = RAGIncidentAnalyzer()
analyzer.load_and_embed_rules(force_reload=True)
```

### Use a Different Embedding Model

```python
analyzer = RAGIncidentAnalyzer(
    embedding_model="all-mpnet-base-v2"  # Higher quality, slower
)
```

### Use a Different Ollama Model

```bash
ollama pull llama3.1:70b  # Larger, more accurate
```

```python
analyzer = RAGIncidentAnalyzer(
    ollama_model="llama3.1:70b"
)
```

## License

This module integrates with:
- **Ollama** (MIT License)
- **ChromaDB** (Apache 2.0)
- **Sentence-Transformers** (Apache 2.0)
- **YOLOv8** (AGPL-3.0)

Ensure compliance with all licenses in production use.
