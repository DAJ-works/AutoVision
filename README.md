# AutoVision

#### Developers: Dhruv Kothari, Aarav Goel, Jayanth Veerappa
#### Los Altos Hacks 2025 - Winners: 1st Place

AutoVision is an advanced AI-powered system for **intelligent accident reconstruction and traffic flow analysis**. By combining computer vision, physics-based modeling, and large language models (LLMs), this tool automatically analyzes dashcam and traffic camera footage to reconstruct incidents, determine fault, and provide actionable insights for improving road safety.

## What It Does
Simply upload traffic or dashcam footage, and the application will:

**Detect and Track Vehicles**: Identifies every vehicle in the footage, tracking their movements across frames with unique IDs and timestamps.

**Estimate Speed and Trajectories**: Calculates vehicle speeds and reconstructs movement paths to understand traffic flow and collision dynamics.

**Detect Violations and Unsafe Behavior**: Automatically flags red-light violations, speeding, unsafe lane changes, and other traffic violations.

**Reconstruct Collisions**: Performs frame-by-frame accident reconstruction, identifying the exact sequence of events leading to a collision.

**Determine Fault**: Applies traffic law-based logic and physics models to determine which driver violated rules or caused impact.

**Detect Near-Misses**: Identifies potential collisions that didn't occur - valuable data for proactive safety improvements.

**Chat with the AI Analyst**: Users can interact with a language model to ask questions about incidents, get fault determinations, and understand what happened using natural language.

## Features
### Feature	Description
**Traffic Video Upload**	Upload dashcam or traffic camera footage securely and privately.
**AI-Powered Analysis**	Uses deep learning to detect vehicles, track movement, estimate speeds, and classify violations.
**Interactive Analytics Dashboard**	Visualizes vehicles, violations, speed data, and incident timeline in a user-friendly interface.
**Automatic Fault Determination**	Applies traffic laws and physics to determine which driver(s) caused the incident.
**Evidence Logging**	Vehicles and violations are automatically logged and time-tagged for legal documentation.
**LLM Accident Analyst**	Chat interface to ask the AI about the incident, fault determination, or speed analysis.
**Secure & Private**	Keeps footage confidential and processed with privacy in mind.

# Use Cases
**Accident Investigation** – Reconstruct collisions and determine fault automatically with AI-verified evidence.

**Insurance Claims Processing** – Speed up claim processing with objective, data-driven analysis.

**Traffic Safety Analysis** – Identify dangerous intersections and driving patterns for city planning.

**Fleet Management** – Monitor driver safety and detect risky driving behavior in commercial vehicles.

**Smart City Integration** – Provide real-time insights for traffic management and signal optimization.

# Tech Stack
**Frontend**: React, CSS, Material-UI

**Backend**: Python (Flask), Node.js

**Video Processing**: OpenCV, YOLOv8, PyTorch

**LLM Integration**: HuggingFace, GPT-based models


# Getting Started
**Clone the repository**

```bash
git clone https://github.com/DAJ-works/AutoVision.git
cd AutoVision
```

**Install dependencies**

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

**Start the application**

```bash
# Run backend (from project root)
cd backend/api
python app.py

# Run frontend (from project root)
cd frontend
npm start
```

## RAG Implementation for LLM Accident Analyst

Our RAG system processes traffic analysis results to enable context-aware conversations:

- Video analysis results (vehicles, speeds, violations, trajectories) are structured into specialized document formats
- Each document includes metadata and context for efficient retrieval
- Analysis results are chunked into semantically meaningful segments for vehicle tracking and incident reconstruction

- User queries are analyzed for intent recognition (fault determination, speed analysis, violation detection)
- A specialized prompt template incorporates retrieved context about the specific incident
- Multi-stage retrieval ensures the most relevant information is provided for accurate accident analysis

## Custom YOLOv8 Vehicle Detection Model

- We use YOLOv8 optimized for vehicle detection and tracking
- The model detects cars, trucks, buses, motorcycles, bicycles, and pedestrians
- Training optimized for traffic scenarios with various lighting and weather conditions

## Speed Estimation & Trajectory Analysis

- Uses optical flow and perspective mapping to estimate vehicle speeds from 2D footage
- Kalman filtering for smooth trajectory prediction and collision risk assessment
- Physics-based models calculate impact forces and angles for accident reconstruction

## Future Work
- Real-time traffic monitoring and accident prevention
- Integration with smart city infrastructure for dynamic traffic signal control
- Advanced violation detection (tailgating, unsafe merges, distracted driving)
- Mobile application for on-scene accident documentation
- Enhanced 3D reconstruction for complex multi-vehicle collisions
- Integration with insurance and law enforcement databases for automated reporting
