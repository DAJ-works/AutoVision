# Setup Guide - LLaMA 3.1 & YOLO RAG Integration

## ✅ What Was Just Merged

I've successfully pulled the **chatbot and LLaMA 3.1/YOLO RAG integration** from the `llama-yolo-rag-integration` branch while **keeping all your local UI changes intact**.

### Backend Files Added:
- ✅ `backend/rag_utils.py` - Main RAG analyzer module
- ✅ `backend/test_rag.py` - Test suite for RAG system
- ✅ `backend/RAG_README.md` - Comprehensive RAG documentation
- ✅ `backend/api/ollama_endpoints.py` - FastAPI endpoints for Ollama integration
- ✅ `backend/data/ca_vehicle_rules.jsonl` - CA Vehicle Code rules database
- ✅ `backend/data/example_yolo_output.json` - Example YOLO detections
- ✅ `backend/fine_tuning/` - Training data and model fine-tuning scripts
- ✅ Dependencies installed: `sentence-transformers`, `chromadb`, `pydantic`

### Frontend Files (Your UI Changes Preserved):
- ✅ All your UI improvements kept intact
- ✅ Centered content layout
- ✅ Professional styling
- ✅ "Incidents" renamed from "Cases"
- ✅ AI Assistant button added to homepage
- ✅ Menu improvements

---

## 🚀 Next Steps - Install Ollama

To complete the setup, you need to install Ollama and pull the LLaMA 3.1 model:

### 1. Install Ollama

Visit **https://ollama.ai** and download the installer for macOS, or use Homebrew:

```bash
brew install ollama
```

### 2. Start Ollama Service

```bash
# Start Ollama in the background
ollama serve
```

### 3. Pull LLaMA 3.1 Model

Open a new terminal and run:

```bash
ollama pull llama3.1
```

This will download the LLaMA 3.1 model (about 4.7 GB).

### 4. Verify Installation

```bash
ollama list
```

You should see:
```
NAME            ID              SIZE    MODIFIED
llama3.1:latest abc123def456    4.7 GB  X days ago
```

---

## 🧪 Test the RAG System

Once Ollama is installed, test the system:

```bash
cd /Users/aaravgoel/Desktop/idnaraiytuk
python backend/test_rag.py
```

This will:
1. ✓ Parse YOLOv8 detections
2. ✓ Load CA Vehicle Code rules into ChromaDB
3. ✓ Test rule retrieval
4. ✓ Test prompt building
5. ✓ Test full Ollama integration

---

## 🔧 How to Use the RAG System

### Option 1: Python Module

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

### Option 3: FastAPI Endpoints

The `backend/api/ollama_endpoints.py` file provides REST API endpoints:

- `POST /api/ollama/analyze-incident` - Analyze incident from path or JSON
- `POST /api/ollama/analyze-upload` - Upload YOLO file and analyze
- `GET /api/ollama/status` - Check system status
- `GET /api/ollama/rules` - List all CA Vehicle Code rules
- `GET /api/ollama/test` - Test with example data

---

## 📊 What the RAG System Does

1. **Parses YOLOv8 Detections** - Reads vehicle, pedestrian, traffic light data
2. **Builds Incident Summary** - Converts detections to natural language
3. **Retrieves Relevant Rules** - Uses semantic search to find applicable CA Vehicle Code
4. **Generates Analysis** - LLaMA 3.1 analyzes the incident and determines fault

**Example Output:**
```
INCIDENT ANALYSIS

Based on the provided evidence and California Vehicle Code:

FAULT DETERMINATION:
Vehicle 1 (Blue sedan) - 75% at fault
- Violated CVC 21453(a): Red Light Violation
- Failed to yield right-of-way

Vehicle 2 (White SUV) - 25% at fault
- Violated CVC 22350: Excessive speed for conditions

LEGAL JUSTIFICATION:
[Detailed analysis using CA Vehicle Code...]
```

---

## 🎯 Current Status

### ✅ Completed
- Backend RAG files merged
- Dependencies installed
- Your UI changes preserved
- Frontend and backend servers can run independently

### ⏳ Needs Installation
- Ollama (https://ollama.ai)
- LLaMA 3.1 model (`ollama pull llama3.1`)

---

## 🏃 Running the Project

### Terminal 1 - Backend (Flask)
```bash
cd /Users/aaravgoel/Desktop/idnaraiytuk/backend/api
/Users/aaravgoel/Desktop/idnaraiytuk/.venv/bin/python /Users/aaravgoel/Desktop/idnaraiytuk/backend/api/app.py
```

### Terminal 2 - Frontend (React)
```bash
cd /Users/aaravgoel/Desktop/idnaraiytuk/frontend
npm start
```

Access the app at: **http://localhost:3000**

---

## 📚 Additional Resources

- **RAG System Details**: See `backend/RAG_README.md`
- **Training Data**: `backend/fine_tuning/data/`
- **Example Detections**: `backend/data/example_yolo_output.json`
- **Ollama Docs**: https://github.com/ollama/ollama

---

## 🐛 Troubleshooting

### "Ollama not found"
- Install from https://ollama.ai
- Make sure `ollama serve` is running

### "Model not found"
- Run `ollama pull llama3.1`

### "ChromaDB error"
- Delete `backend/data/chroma_db/` and re-run test

### "Port already in use"
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

---

## 🎉 Summary

You now have:
- ✅ Your beautiful UI (all changes preserved)
- ✅ LLaMA 3.1 RAG integration (backend only)
- ✅ YOLO detection analysis
- ✅ CA Vehicle Code rule database
- ⏳ Ollama installation pending

**Next**: Install Ollama and test the system!
