#!/bin/bash
# Enhanced Traffic Analyst Setup Script
# Scrapes CA Vehicle Code data and trains an Ollama model with YOLO integration

set -e  # Exit on error

echo "=========================================="
echo "Enhanced Traffic Analyst Pro Setup"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not detected"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# Install required packages
echo "Installing required Python packages..."
pip install beautifulsoup4 requests lxml --quiet

echo "✓ Dependencies installed"
echo ""

# Step 1: Scrape CA Vehicle Code data
echo "=========================================="
echo "Step 1: Scraping CA Vehicle Code Data"
echo "=========================================="
echo ""
echo "Collecting data from:"
echo "  - leginfo.legislature.ca.gov"
echo "  - catsip.berkeley.edu"
echo ""

python backend/fine_tuning/scrape_ca_vehicle_code.py

echo ""
echo "✓ CA Vehicle Code data scraped successfully"
echo ""

# Step 2: Copy scraped data to RAG directory
echo "=========================================="
echo "Step 2: Updating RAG Database"
echo "=========================================="
echo ""

if [ -f "backend/fine_tuning/data/ca_vehicle_code_complete.jsonl" ]; then
    cp backend/fine_tuning/data/ca_vehicle_code_complete.jsonl backend/data/ca_vehicle_rules.jsonl
    echo "✓ Updated RAG database with comprehensive CA Vehicle Code"
else
    echo "⚠️  Scraped data not found, using existing data"
fi

echo ""

# Step 3: Train the enhanced Ollama model
echo "=========================================="
echo "Step 3: Training Enhanced Ollama Model"
echo "=========================================="
echo ""
echo "Creating 'traffic-analyst-pro' model..."
echo "Base model: llama3.1"
echo "Training data: CA Vehicle Code + YOLO integration"
echo ""

python backend/fine_tuning/train_enhanced_model.py

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Your enhanced traffic analyst is ready!"
echo ""
echo "Features:"
echo "  ✓ Comprehensive CA Vehicle Code knowledge"
echo "  ✓ YOLO detection integration"
echo "  ✓ Fault determination capabilities"
echo "  ✓ Legal Q&A with code citations"
echo ""
echo "Usage:"
echo "  1. Command line test:"
echo "     ollama run traffic-analyst-pro"
echo ""
echo "  2. Backend integration:"
echo "     Update backend/api/app.py:"
echo "     ollama_model='traffic-analyst-pro'"
echo ""
echo "  3. Quick test:"
echo "     echo 'What is CVC 21453?' | ollama run traffic-analyst-pro"
echo ""
echo "  4. Restart backend:"
echo "     pkill -f 'python.*app.py'"
echo "     nohup python backend/api/app.py > backend.log 2>&1 &"
echo ""
echo "Generated files:"
echo "  - backend/fine_tuning/data/ca_vehicle_code_yolo_alpaca.jsonl"
echo "  - backend/fine_tuning/data/ca_vehicle_code_yolo_ollama.jsonl"
echo "  - backend/fine_tuning/data/ca_vehicle_code_complete.jsonl"
echo "  - backend/data/ca_vehicle_rules.jsonl (RAG database)"
echo ""
echo "=========================================="
