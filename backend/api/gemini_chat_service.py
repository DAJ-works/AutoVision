"""
FREE CA Vehicle Code Expert using Google Gemini Flash 2.0
Perfect for M2 Mac - uses Google's FREE API!
No local models needed, generous free tier
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeminiCALawExpert:
    def __init__(self):
        """
        Initialize Gemini-based CA law expert
        Uses Google's FREE Gemini Flash 2.0 API
        
        Free tier includes:
        - 15 requests per minute
        - 1 million tokens per day
        - 1500 requests per day
        
        More than enough for your use case!
        """
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Get one free at https://aistudio.google.com/apikey")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash (free and fast!)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        logger.info("✅ Gemini Flash 2.0 initialized (FREE API)")
        
        # Load CA Vehicle Code data - try multiple possible paths
        possible_paths = [
            Path("./data/training_data/ca_vehicle_code_training.jsonl"),
            Path("../fine_tuning/data/training_data/ca_vehicle_code_training.jsonl"),
            Path(__file__).parent.parent / "fine_tuning/data/training_data/ca_vehicle_code_training.jsonl"
        ]
        
        self.ca_code_path = None
        self.ca_code_data = []
        
        for path in possible_paths:
            if path.exists():
                self.ca_code_path = path
                logger.info(f"✅ Found CA Vehicle Code data at: {path}")
                break
        
        if self.ca_code_path:
            self._load_ca_code()
        else:
            logger.warning(f"⚠️ CA Vehicle Code data not found. Tried: {[str(p) for p in possible_paths]}")
    
    def _load_ca_code(self):
        """Load CA Vehicle Code into memory"""
        logger.info(f"Loading CA Vehicle Code from {self.ca_code_path}")
        
        with open(self.ca_code_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.ca_code_data.append(item)
        
        logger.info(f"✅ Loaded {len(self.ca_code_data)} CA Vehicle Code sections")
    
    def search_relevant_laws(self, query: str, n_results: int = 8) -> List[Dict]:
        """Smart keyword search for relevant CA Vehicle Code sections"""
        if not self.ca_code_data:
            return []
        
        query_lower = query.lower()
        scored_results = []
        
        # Enhanced keyword matching
        query_words = [w for w in query_lower.split() if len(w) > 2]
        
        # Common synonyms and related terms
        keyword_map = {
            'speed': ['speeding', 'velocity', 'mph', 'fast', 'racing'],
            'drunk': ['dui', 'dwi', 'intoxicated', 'impaired', 'alcohol', 'bac'],
            'accident': ['crash', 'collision', 'hit', 'struck'],
            'red': ['light', 'signal', 'stop', 'traffic light'],
            'turn': ['turning', 'lane', 'intersection'],
            'park': ['parking', 'parked', 'vehicle'],
            'license': ['permit', 'registration', 'dmv'],
            'reckless': ['dangerous', 'careless', 'negligent'],
            'phone': ['cell', 'mobile', 'texting', 'distracted'],
            'pedestrian': ['crosswalk', 'sidewalk', 'walking'],
        }
        
        for item in self.ca_code_data:
            score = 0
            text = f"{item['instruction']} {item.get('input', '')} {item['output']}".lower()
            
            # Score based on direct keyword matches
            for word in query_words:
                # Exact word match (highest value)
                if word in text.split():
                    score += 15
                
                # Substring match (medium value)
                if word in text:
                    score += text.count(word) * 3
                
                # Check synonyms
                if word in keyword_map:
                    for synonym in keyword_map[word]:
                        if synonym in text:
                            score += 8
            
            # Boost score for CVC section numbers if mentioned
            import re
            cvc_numbers = re.findall(r'cvc\s*(\d+)', query_lower)
            for num in cvc_numbers:
                if num in text:
                    score += 100  # Exact CVC match is most important
            
            if score > 0:
                scored_results.append((score, item))
        
        # Sort by score and return top results
        scored_results.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scored_results[:n_results]]
    
    def generate_response(self, user_message: str, case_context: str = None) -> str:
        """
        Generate response using Gemini Flash 2.0 (FREE!)
        """
        # Get relevant CA Vehicle Code sections
        relevant_laws = self.search_relevant_laws(user_message, n_results=8)
        
        # Build comprehensive prompt with smart formatting
        system_prompt = """You are an expert California Vehicle Code attorney and legal advisor. You provide clear, conversational, and highly accurate legal guidance.

COMMUNICATION STYLE:
- Write naturally like a knowledgeable lawyer explaining things to a client
- NO markdown asterisks (*), NO bullet points with stars
- Use clear paragraphs and natural formatting
- Be conversational but professional
- Get straight to the answer without unnecessary pleasantries

EXPERTISE AREAS:
- California traffic laws and enforcement
- Accident liability and fault determination
- Traffic violations, citations, and penalties
- Vehicle operation and equipment standards
- DUI laws and consequences
- Insurance requirements and claims
- Reckless vs negligent driving distinctions

RESPONSE REQUIREMENTS:
1. Answer directly and comprehensively
2. Always cite specific CVC section numbers (e.g., "Under CVC 22350..." or "According to Vehicle Code Section 23152(a)...")
3. Explain the law's practical application
4. Include relevant penalties, fines, or consequences
5. Mention any exceptions or special circumstances
6. If analyzing accidents, clearly explain legal fault and liability

Format your response as clean paragraphs. Use "Under CVC [number]" or "Vehicle Code Section [number]" when citing laws. NO asterisks or markdown formatting symbols."""
        
        # Add relevant law sections
        context = ""
        if relevant_laws:
            context += "\n\n=== RELEVANT CA VEHICLE CODE DATABASE ===\n"
            for i, law in enumerate(relevant_laws, 1):
                context += f"\n[Section {i}]\n"
                context += f"Topic: {law['instruction']}\n"
                if law.get('input'):
                    context += f"Context: {law['input']}\n"
                context += f"Legal Info: {law['output']}\n"
            context += "\n=== END DATABASE ===\n"
        
        # Add case context if available
        if case_context:
            context += f"\n\n=== CURRENT CASE DATA ===\n{case_context}\n=== END CASE DATA ===\n"
        
        # Build full prompt
        full_prompt = f"""{system_prompt}

{context}

User Question: {user_message}

Provide a detailed, accurate answer based on California Vehicle Code. Be conversational and direct. NO markdown formatting symbols."""
        
        try:
            # Generate response with Gemini
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.3,  # Lower = more consistent/accurate
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            
            # Clean up any markdown formatting that slips through
            text = response.text
            # Remove bold/italic markers
            text = text.replace('**', '').replace('__', '').replace('*', '').replace('_', '')
            # Clean up bullet points
            text = text.replace('•', '-')
            
            return text
        
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            if "API_KEY" in str(e):
                return "Gemini API key issue. Please set GEMINI_API_KEY environment variable with a valid key from https://aistudio.google.com/apikey"
            else:
                return f"Sorry, I encountered an error: {str(e)}"
    
    def chat(self, user_message: str, case_context: str = None) -> str:
        """Main chat function"""
        return self.generate_response(user_message, case_context)

# Global instance
_gemini_expert = None

def get_gemini_expert():
    """Get or create Gemini expert instance"""
    global _gemini_expert
    
    if _gemini_expert is None:
        _gemini_expert = GeminiCALawExpert()
    
    return _gemini_expert

def general_law_chat(user_message: str) -> str:
    """FREE chat about California Vehicle Code using Gemini"""
    expert = get_gemini_expert()
    return expert.chat(user_message)

def chat_with_case(case_id: str, user_message: str, case_data: Dict = None) -> str:
    """FREE chat about a specific case using Gemini"""
    expert = get_gemini_expert()
    
    # Format case context
    case_context = None
    if case_data:
        context_parts = []
        if 'video_path' in case_data:
            context_parts.append(f"Video: {case_data['video_path']}")
        if 'detections' in case_data:
            context_parts.append(f"Detected {len(case_data['detections'])} objects")
        if 'violations' in case_data:
            violations = case_data['violations']
            context_parts.append(f"Violations: {len(violations)}")
            for v in violations[:5]:
                context_parts.append(f"  - {v.get('type', 'Unknown')}: {v.get('description', 'N/A')}")
        if 'speeds' in case_data:
            speeds = case_data['speeds']
            if speeds:
                max_speed = max([s.get('speed_mph', 0) for s in speeds])
                context_parts.append(f"Max Speed: {max_speed:.1f} mph")
        case_context = "\n".join(context_parts)
    
    return expert.chat(user_message, case_context)
