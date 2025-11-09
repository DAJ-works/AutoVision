"""
Example integration of RAG incident analyzer into FastAPI backend.

Add this to your backend/api/app.py or create a new endpoint file.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import tempfile
import os
from pathlib import Path

# Import the RAG analyzer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_utils import analyze_incident, RAGIncidentAnalyzer

app = FastAPI()

# Initialize analyzer once at startup (singleton pattern)
_analyzer = None

def get_analyzer() -> RAGIncidentAnalyzer:
    """Get or create singleton RAG analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = RAGIncidentAnalyzer(
            rules_path="backend/data/ca_vehicle_rules.jsonl",
            ollama_model="llama3.1"
        )
        # Pre-load rules at startup
        _analyzer.load_and_embed_rules()
    return _analyzer


class AnalysisRequest(BaseModel):
    """Request model for incident analysis."""
    yolo_json_path: Optional[str] = None
    detections: Optional[dict] = None  # Direct JSON payload
    camera_id: Optional[str] = None
    top_k_rules: int = 5


class AnalysisResponse(BaseModel):
    """Response model for incident analysis."""
    analysis: str
    camera_id: Optional[str] = None
    frame_id: Optional[int] = None
    rules_used: int
    status: str


@app.on_event("startup")
async def startup_event():
    """Pre-load models on startup."""
    print("Initializing RAG analyzer...")
    get_analyzer()
    print("✓ RAG analyzer ready")


@app.post("/api/ollama/analyze-incident", response_model=AnalysisResponse)
async def analyze_incident_endpoint(request: AnalysisRequest):
    """
    Analyze traffic incident using YOLOv8 detections + Llama 3.1.
    
    Two modes:
    1. Provide yolo_json_path - path to existing JSON file
    2. Provide detections - inline JSON payload
    
    Example request (inline):
    {
      "detections": {
        "camera_id": "cam_01",
        "frame_id": 1847,
        "detections": [
          {"label": "car", "confidence": 0.96, "bbox": [120, 340, 380, 580]},
          {"label": "traffic_light", "confidence": 0.89, "bbox": [450, 50, 490, 120], "state": "red"}
        ]
      },
      "top_k_rules": 5
    }
    
    Example request (file path):
    {
      "yolo_json_path": "backend/data/example_yolo_output.json",
      "top_k_rules": 3
    }
    """
    try:
        analyzer = get_analyzer()
        
        # Determine input mode
        if request.yolo_json_path:
            # Mode 1: File path
            if not os.path.exists(request.yolo_json_path):
                raise HTTPException(status_code=404, detail=f"File not found: {request.yolo_json_path}")
            
            yolo_path = request.yolo_json_path
            
        elif request.detections:
            # Mode 2: Inline JSON
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(request.detections, f)
                yolo_path = f.name
        else:
            raise HTTPException(
                status_code=400, 
                detail="Must provide either 'yolo_json_path' or 'detections'"
            )
        
        # Analyze
        analysis_text = analyzer.analyze_incident(yolo_path, top_k_rules=request.top_k_rules)
        
        # Extract metadata
        camera_id = request.camera_id or request.detections.get('camera_id') if request.detections else None
        frame_id = request.detections.get('frame_id') if request.detections else None
        
        # Clean up temp file if created
        if request.detections:
            try:
                os.unlink(yolo_path)
            except:
                pass
        
        return AnalysisResponse(
            analysis=analysis_text,
            camera_id=camera_id,
            frame_id=frame_id,
            rules_used=request.top_k_rules,
            status="success"
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/ollama/analyze-upload")
async def analyze_uploaded_file(file: UploadFile = File(...)):
    """
    Upload YOLOv8 JSON file and analyze.
    
    Example usage:
    curl -X POST "http://localhost:8000/api/ollama/analyze-upload" \
         -F "file=@detections.json"
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Parse JSON
        try:
            detections = json.loads(contents.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(detections, f)
            temp_path = f.name
        
        # Analyze
        analyzer = get_analyzer()
        analysis_text = analyzer.analyze_incident(temp_path, top_k_rules=5)
        
        # Clean up
        os.unlink(temp_path)
        
        return AnalysisResponse(
            analysis=analysis_text,
            camera_id=detections.get('camera_id'),
            frame_id=detections.get('frame_id'),
            rules_used=5,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {str(e)}")


@app.get("/api/ollama/status")
async def ollama_status():
    """Check if Ollama and RAG system are operational."""
    try:
        analyzer = get_analyzer()
        
        # Check ChromaDB
        _, collection = analyzer._init_chroma()
        rules_count = collection.count()
        
        # Test Ollama with simple prompt
        test_response = analyzer.call_ollama("Say 'OK' if you can read this.", timeout=10)
        ollama_ok = len(test_response) > 0
        
        return {
            "status": "operational",
            "ollama_available": ollama_ok,
            "chromadb_rules": rules_count,
            "embedding_model": analyzer.embedding_model_name,
            "ollama_model": analyzer.ollama_model
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/api/ollama/rules")
async def list_rules():
    """List all CA Vehicle Code rules in the database."""
    try:
        analyzer = get_analyzer()
        _, collection = analyzer._init_chroma()
        
        # Get all rules
        results = collection.get()
        
        rules = []
        for i in range(len(results['ids'])):
            rules.append({
                'id': results['ids'][i],
                'code': results['metadatas'][i].get('code', 'N/A'),
                'title': results['metadatas'][i].get('title', 'N/A'),
                'category': results['metadatas'][i].get('category', 'general'),
                'document': results['documents'][i][:200] + '...' if len(results['documents'][i]) > 200 else results['documents'][i]
            })
        
        return {
            "total_rules": len(rules),
            "rules": rules
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Test endpoint (development only)
@app.get("/api/ollama/test")
async def test_analysis():
    """
    Test the RAG analyzer with example data.
    """
    try:
        example_path = "backend/data/example_yolo_output.json"
        if not os.path.exists(example_path):
            raise HTTPException(status_code=404, detail="Example file not found")
        
        result = analyze_incident(example_path, top_k_rules=3)
        
        return {
            "test": "success",
            "example_file": example_path,
            "analysis": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("RAG Incident Analyzer API")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /api/ollama/analyze-incident  - Analyze from path or JSON")
    print("  POST /api/ollama/analyze-upload    - Upload file and analyze")
    print("  GET  /api/ollama/status            - Check system status")
    print("  GET  /api/ollama/rules             - List all rules")
    print("  GET  /api/ollama/test              - Test with example data")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
