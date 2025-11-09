"""
RAG-based Incident Analysis Module

This module provides RAG (Retrieval-Augmented Generation) functionality to analyze
traffic incidents from YOLOv8 detections using California Vehicle Code rules and
Llama 3.1 via Ollama.

Main function: analyze_incident(yolo_json_path: str) -> str
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    logger.warning("chromadb not installed. Install with: pip install chromadb")
    CHROMADB_AVAILABLE = False


class RAGIncidentAnalyzer:
    """
    RAG-based analyzer that combines YOLOv8 detections with CA vehicle rules
    and uses Llama 3.1 for fault analysis.
    """
    
    def __init__(
        self,
        rules_path: str = "backend/data/ca_vehicle_rules.jsonl",
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.1",
        chroma_persist_dir: str = "backend/data/chroma_db"
    ):
        """
        Initialize the RAG analyzer.
        
        Args:
            rules_path: Path to CA vehicle rules JSONL file
            embedding_model: Sentence transformer model name
            ollama_model: Ollama model name (must be downloaded locally)
            chroma_persist_dir: Directory to persist ChromaDB
        """
        self.rules_path = rules_path
        self.embedding_model_name = embedding_model
        self.ollama_model = ollama_model
        self.chroma_persist_dir = chroma_persist_dir
        
        # Lazy initialization
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None
        self._rules_loaded = False
    
    def _check_dependencies(self):
        """Verify all required dependencies are installed."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "pip install sentence-transformers"
            )
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required. Install with:\n"
                "pip install chromadb"
            )
    
    def _init_embedding_model(self):
        """Lazily initialize the sentence transformer model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
    
    def _init_chroma(self):
        """Lazily initialize ChromaDB client and collection."""
        if self._chroma_client is None:
            logger.info(f"Initializing ChromaDB at: {self.chroma_persist_dir}")
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            
            # Get or create collection
            self._collection = self._chroma_client.get_or_create_collection(
                name="ca_vehicle_rules",
                metadata={"description": "California Vehicle Code rules for traffic analysis"}
            )
            logger.info(f"ChromaDB collection initialized. Documents: {self._collection.count()}")
        
        return self._chroma_client, self._collection
    
    def load_and_embed_rules(self, force_reload: bool = False):
        """
        Load CA vehicle rules from JSONL and embed them into ChromaDB.
        
        Args:
            force_reload: If True, clear existing collection and reload
        """
        self._check_dependencies()
        self._init_embedding_model()
        _, collection = self._init_chroma()
        
        # Check if rules already loaded
        if collection.count() > 0 and not force_reload:
            logger.info(f"Rules already loaded ({collection.count()} documents). Skipping.")
            self._rules_loaded = True
            return
        
        # Clear collection if force reload
        if force_reload and collection.count() > 0:
            logger.info("Force reload: clearing existing collection")
            self._chroma_client.delete_collection("ca_vehicle_rules")
            self._collection = self._chroma_client.create_collection(
                name="ca_vehicle_rules",
                metadata={"description": "California Vehicle Code rules for traffic analysis"}
            )
            collection = self._collection
        
        # Load rules from JSONL
        rules_path = Path(self.rules_path)
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_path}")
        
        logger.info(f"Loading rules from: {self.rules_path}")
        rules = []
        with open(rules_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rule = json.loads(line)
                    rules.append(rule)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        if not rules:
            raise ValueError(f"No valid rules found in {self.rules_path}")
        
        logger.info(f"Loaded {len(rules)} rules. Embedding and storing in ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, rule in enumerate(rules):
            # Create searchable text from rule
            code = rule.get('code', f'Rule-{i}')
            title = rule.get('title', '')
            description = rule.get('description', '')
            
            # Combine for embedding
            doc_text = f"{title}\n{description}"
            
            ids.append(f"rule_{i}_{code}")
            documents.append(doc_text)
            metadatas.append({
                'code': code,
                'title': title,
                'category': rule.get('category', 'general')
            })
        
        # Batch add to ChromaDB (it handles embedding automatically if we provide an embedding function)
        # But we'll embed explicitly for more control
        embeddings = self._embedding_model.encode(documents, show_progress_bar=True)
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )
        
        logger.info(f"Successfully embedded {len(rules)} rules into ChromaDB")
        self._rules_loaded = True
    
    def retrieve_relevant_rules(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant rules based on query.
        
        Args:
            query: Search query (e.g., incident description)
            top_k: Number of rules to retrieve
            
        Returns:
            List of rule dictionaries with metadata
        """
        if not self._rules_loaded:
            self.load_and_embed_rules()
        
        # Embed query
        query_embedding = self._embedding_model.encode([query])[0]
        
        # Search ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        retrieved_rules = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                rule = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_rules.append(rule)
        
        logger.info(f"Retrieved {len(retrieved_rules)} rules for query: {query[:100]}...")
        return retrieved_rules
    
    def parse_yolo_json(self, yolo_json_path: str) -> Dict[str, Any]:
        """
        Parse YOLOv8 JSON output file.
        
        Args:
            yolo_json_path: Path to YOLOv8 JSON file
            
        Returns:
            Parsed detection data
        """
        path = Path(yolo_json_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLOv8 JSON not found: {yolo_json_path}")
        
        logger.info(f"Loading YOLOv8 detections from: {yolo_json_path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def build_incident_summary(self, yolo_data: Dict[str, Any]) -> str:
        """
        Build a natural language summary of the incident from YOLOv8 data.
        
        Args:
            yolo_data: Parsed YOLOv8 detection data
            
        Returns:
            Human-readable incident summary
        """
        summary_parts = []
        
        # Handle different possible YOLOv8 JSON structures
        if 'detections' in yolo_data:
            detections = yolo_data['detections']
        elif 'frames' in yolo_data:
            # Multi-frame format
            detections = []
            for frame in yolo_data['frames']:
                detections.extend(frame.get('detections', []))
        elif isinstance(yolo_data, list):
            detections = yolo_data
        else:
            detections = [yolo_data]
        
        # Extract metadata
        camera_id = yolo_data.get('camera_id', 'unknown')
        timestamp = yolo_data.get('timestamp', 'unknown')
        frame_id = yolo_data.get('frame_id', 'N/A')
        
        summary_parts.append(f"Incident Analysis - Camera: {camera_id}, Frame: {frame_id}, Time: {timestamp}")
        summary_parts.append("\nDetected Objects and Events:")
        
        # Group detections by label
        label_counts = {}
        high_conf_detections = []
        
        for det in detections:
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)
            bbox = det.get('bbox', [])
            
            label_counts[label] = label_counts.get(label, 0) + 1
            
            if confidence >= 0.5:  # High confidence threshold
                high_conf_detections.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        # Summary of objects
        summary_parts.append(f"\nObject Summary:")
        for label, count in sorted(label_counts.items()):
            summary_parts.append(f"  - {label}: {count} detected")
        
        # Detailed high-confidence detections
        if high_conf_detections:
            summary_parts.append(f"\nHigh-Confidence Detections (>0.5):")
            for i, det in enumerate(high_conf_detections[:10], 1):  # Limit to top 10
                summary_parts.append(
                    f"  {i}. {det['label']} (confidence: {det['confidence']:.2f}) at bbox {det['bbox']}"
                )
        
        return "\n".join(summary_parts)
    
    def build_prompt(self, incident_summary: str, relevant_rules: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for Llama 3.1 with incident data and relevant rules.
        
        Args:
            incident_summary: Natural language incident summary
            relevant_rules: Retrieved CA vehicle rules
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a traffic incident analyst with expertise in California Vehicle Code.",
            "Analyze the following traffic incident and determine fault based on the provided rules.\n",
            "=== INCIDENT DETAILS ===",
            incident_summary,
            "\n=== RELEVANT CALIFORNIA VEHICLE CODE RULES ===\n"
        ]
        
        for i, rule in enumerate(relevant_rules, 1):
            metadata = rule['metadata']
            doc = rule['document']
            code = metadata.get('code', 'N/A')
            
            prompt_parts.append(f"{i}. [{code}] {doc}\n")
        
        prompt_parts.append("\n=== QUESTION ===")
        prompt_parts.append(
            "Based on the incident details and the California Vehicle Code rules above, "
            "determine who is at fault. Provide your analysis in exactly 3 bullet points, "
            "each with specific evidence from the incident and reference to the applicable rule code."
        )
        prompt_parts.append("\nYour response:")
        
        return "\n".join(prompt_parts)
    
    def call_ollama(self, prompt: str, timeout: int = 60) -> str:
        """
        Call Llama 3.1 via Ollama subprocess.
        
        Args:
            prompt: The prompt to send to Ollama
            timeout: Timeout in seconds
            
        Returns:
            Model's response text
        """
        logger.info(f"Calling Ollama model: {self.ollama_model}")
        
        try:
            # Build Ollama command
            cmd = [
                'ollama',
                'run',
                self.ollama_model
            ]
            
            # Call Ollama via subprocess with prompt on stdin
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise RuntimeError(f"Ollama command failed (exit code {result.returncode}): {error_msg}")
            
            response = result.stdout.strip()
            
            if not response:
                raise ValueError("Ollama returned empty response")
            
            logger.info(f"Received response from Ollama ({len(response)} chars)")
            return response
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Ollama call timed out after {timeout} seconds")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Ollama command not found. Make sure Ollama is installed and in PATH.\n"
                "Install from: https://ollama.ai\n"
                f"Then run: ollama pull {self.ollama_model}"
            )
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise
    
    def analyze_incident(self, yolo_json_path: str, top_k_rules: int = 5) -> str:
        """
        Main analysis function: RAG-based incident fault analysis.
        
        This function:
        1. Loads YOLOv8 detections from JSON
        2. Loads and embeds CA vehicle rules (if not already done)
        3. Retrieves top-k relevant rules
        4. Builds a prompt with incident + rules
        5. Calls Llama 3.1 via Ollama
        6. Returns the model's reasoning
        
        Args:
            yolo_json_path: Path to YOLOv8 JSON output
            top_k_rules: Number of rules to retrieve
            
        Returns:
            Llama 3.1's fault analysis response
        """
        try:
            # Step 1: Check dependencies
            self._check_dependencies()
            
            # Step 2: Parse YOLOv8 JSON
            yolo_data = self.parse_yolo_json(yolo_json_path)
            
            # Step 3: Build incident summary
            incident_summary = self.build_incident_summary(yolo_data)
            logger.info(f"Built incident summary:\n{incident_summary[:200]}...")
            
            # Step 4: Load and embed rules (if needed)
            self.load_and_embed_rules()
            
            # Step 5: Retrieve relevant rules
            # Use the incident summary as the query
            relevant_rules = self.retrieve_relevant_rules(incident_summary, top_k=top_k_rules)
            
            # Step 6: Build prompt
            prompt = self.build_prompt(incident_summary, relevant_rules)
            logger.info(f"Built prompt ({len(prompt)} chars)")
            
            # Step 7: Call Ollama
            response = self.call_ollama(prompt)
            
            logger.info("Analysis complete")
            return response
            
        except Exception as e:
            logger.error(f"Error in analyze_incident: {e}", exc_info=True)
            raise


# Convenience function for direct use
def analyze_incident(
    yolo_json_path: str,
    rules_path: str = "backend/data/ca_vehicle_rules.jsonl",
    embedding_model: str = "all-MiniLM-L6-v2",
    ollama_model: str = "llama3.1",
    top_k_rules: int = 5
) -> str:
    """
    Analyze a traffic incident using RAG + Llama 3.1.
    
    Args:
        yolo_json_path: Path to YOLOv8 detection JSON
        rules_path: Path to CA vehicle rules JSONL
        embedding_model: Sentence transformer model
        ollama_model: Ollama model name
        top_k_rules: Number of rules to retrieve
        
    Returns:
        Llama 3.1's fault analysis (string)
        
    Example:
        >>> result = analyze_incident("data/incident_001.json")
        >>> print(result)
    """
    analyzer = RAGIncidentAnalyzer(
        rules_path=rules_path,
        embedding_model=embedding_model,
        ollama_model=ollama_model
    )
    
    return analyzer.analyze_incident(yolo_json_path, top_k_rules=top_k_rules)


if __name__ == "__main__":
    # CLI interface
    if len(sys.argv) < 2:
        print("Usage: python rag_utils.py <yolo_json_path> [rules_path]")
        print("\nExample:")
        print("  python rag_utils.py backend/data/example_yolo_output.json")
        sys.exit(1)
    
    yolo_path = sys.argv[1]
    rules_path = sys.argv[2] if len(sys.argv) > 2 else "backend/data/ca_vehicle_rules.jsonl"
    
    print(f"\n{'='*60}")
    print("RAG-Based Incident Analyzer")
    print(f"{'='*60}\n")
    
    try:
        result = analyze_incident(yolo_path, rules_path=rules_path)
        print("\n" + "="*60)
        print("ANALYSIS RESULT")
        print("="*60)
        print(result)
        print("\n" + "="*60 + "\n")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
