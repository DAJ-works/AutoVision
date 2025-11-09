"""
Evaluation framework to measure traffic-analyst model accuracy.

This script:
1. Creates a labeled test set of traffic incidents
2. Runs both base and custom models
3. Calculates accuracy metrics:
   - CVC code accuracy (correct rule citations)
   - Fault determination accuracy (correct party identification)
   - Evidence usage score (references detection data)
   - Format compliance (3-bullet structure)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.insert(0, 'backend')
from rag_utils import RAGIncidentAnalyzer


class AccuracyEvaluator:
    """Evaluate traffic incident analysis model accuracy."""
    
    def __init__(self):
        self.test_cases = self._create_test_set()
        self.base_analyzer = RAGIncidentAnalyzer(ollama_model="llama3.1")
        self.custom_analyzer = RAGIncidentAnalyzer(ollama_model="traffic-analyst")
    
    def _create_test_set(self) -> List[Dict]:
        """
        Create labeled test cases with ground truth.
        
        Each case has:
        - incident: YOLOv8-style detections
        - ground_truth: Expected CVC codes, fault determination
        """
        return [
            {
                "name": "Red light violation",
                "incident": {
                    "camera_id": "test_cam_01",
                    "frame_id": 100,
                    "detections": [
                        {"label": "car", "confidence": 0.95, "bbox": [100, 200, 300, 400], "color": "red"},
                        {"label": "traffic_light", "confidence": 0.92, "bbox": [50, 50, 80, 120], "state": "red"}
                    ]
                },
                "ground_truth": {
                    "cvc_codes": ["CVC 21453(a)", "21453"],
                    "at_fault": ["car", "driver", "red car"],
                    "not_at_fault": [],
                    "key_evidence": ["red", "traffic light", "state"]
                }
            },
            {
                "name": "Pedestrian right-of-way violation",
                "incident": {
                    "camera_id": "test_cam_02",
                    "frame_id": 200,
                    "detections": [
                        {"label": "car", "confidence": 0.93, "bbox": [200, 300, 400, 500], "color": "blue"},
                        {"label": "person", "confidence": 0.89, "bbox": [350, 400, 400, 600], "location": "crosswalk"}
                    ]
                },
                "ground_truth": {
                    "cvc_codes": ["CVC 21950(a)", "21950"],
                    "at_fault": ["car", "driver", "blue car"],
                    "not_at_fault": ["pedestrian", "person"],
                    "key_evidence": ["crosswalk", "pedestrian", "person"]
                }
            },
            {
                "name": "Stop sign violation",
                "incident": {
                    "camera_id": "test_cam_03",
                    "frame_id": 300,
                    "detections": [
                        {"label": "car", "confidence": 0.96, "bbox": [150, 250, 350, 450], "speed_estimate": 40},
                        {"label": "stop_sign", "confidence": 0.94, "bbox": [50, 100, 100, 200]}
                    ]
                },
                "ground_truth": {
                    "cvc_codes": ["CVC 21802(a)", "CVC 22450(a)", "21802", "22450"],
                    "at_fault": ["car", "driver"],
                    "not_at_fault": [],
                    "key_evidence": ["stop sign", "stop_sign"]
                }
            },
            {
                "name": "Following too closely",
                "incident": {
                    "camera_id": "test_cam_04",
                    "frame_id": 400,
                    "detections": [
                        {"label": "car", "confidence": 0.97, "bbox": [100, 200, 300, 400], "color": "silver"},
                        {"label": "car", "confidence": 0.95, "bbox": [95, 380, 295, 580], "color": "black"}
                    ]
                },
                "ground_truth": {
                    "cvc_codes": ["CVC 21703", "21703"],
                    "at_fault": ["black car", "rear", "following"],
                    "not_at_fault": ["silver car", "lead"],
                    "key_evidence": ["distance", "following", "closely"]
                }
            },
            {
                "name": "Green light with pedestrian",
                "incident": {
                    "camera_id": "test_cam_05",
                    "frame_id": 500,
                    "detections": [
                        {"label": "car", "confidence": 0.96, "bbox": [100, 200, 300, 400], "color": "blue"},
                        {"label": "traffic_light", "confidence": 0.91, "bbox": [50, 50, 80, 120], "state": "green"},
                        {"label": "person", "confidence": 0.87, "bbox": [250, 350, 300, 550], "location": "crosswalk"}
                    ]
                },
                "ground_truth": {
                    "cvc_codes": ["CVC 21950(a)", "21950", "CVC 21451(a)", "21451"],
                    "at_fault": ["car", "driver"],
                    "not_at_fault": ["pedestrian", "person"],
                    "key_evidence": ["crosswalk", "pedestrian", "yield"]
                }
            }
        ]
    
    def _extract_cvc_codes(self, text: str) -> List[str]:
        """Extract CVC codes from analysis text."""
        # Match patterns like "CVC 21453(a)", "21453(a)", "CVC 21950"
        patterns = [
            r'CVC\s*\d+(?:\([a-z]\))?',
            r'\b\d{5}(?:\([a-z]\))?(?=\s|,|\.|\))',
            r'\b\d{4}(?:\([a-z]\))?(?=\s|,|\.|\))'
        ]
        
        codes = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            codes.extend([m.strip() for m in matches])
        
        return list(set(codes))  # Remove duplicates
    
    def _check_fault_identification(self, text: str, ground_truth: Dict) -> Dict:
        """Check if fault was correctly identified."""
        text_lower = text.lower()
        
        # Check at-fault identification
        at_fault_correct = any(
            fault_term in text_lower and ("at fault" in text_lower or "violated" in text_lower)
            for fault_term in ground_truth["at_fault"]
        )
        
        # Check not-at-fault identification
        not_at_fault_correct = True
        if ground_truth["not_at_fault"]:
            not_at_fault_correct = any(
                term in text_lower and ("not at fault" in text_lower or "no violation" in text_lower or "followed" in text_lower)
                for term in ground_truth["not_at_fault"]
            )
        
        return {
            "at_fault_correct": at_fault_correct,
            "not_at_fault_correct": not_at_fault_correct,
            "overall_correct": at_fault_correct and not_at_fault_correct
        }
    
    def _check_evidence_usage(self, text: str, ground_truth: Dict) -> float:
        """Score evidence usage (0-1)."""
        text_lower = text.lower()
        evidence_count = sum(
            1 for evidence in ground_truth["key_evidence"]
            if evidence.lower() in text_lower
        )
        return evidence_count / len(ground_truth["key_evidence"]) if ground_truth["key_evidence"] else 0
    
    def _check_format_compliance(self, text: str) -> Dict:
        """Check if output follows 3-bullet format."""
        lines = text.strip().split('\n')
        bullet_lines = [l for l in lines if l.strip().startswith('•') or l.strip().startswith('-') or l.strip().startswith('*')]
        
        return {
            "has_bullets": len(bullet_lines) > 0,
            "has_three_bullets": len(bullet_lines) == 3,
            "bullet_count": len(bullet_lines)
        }
    
    def evaluate_case(self, case: Dict, model_name: str) -> Dict:
        """Evaluate a single test case."""
        # Save incident to temp file
        temp_file = Path("backend/data/temp_eval_case.json")
        with open(temp_file, 'w') as f:
            json.dump(case["incident"], f)
        
        try:
            # Run analysis
            analyzer = self.custom_analyzer if "traffic" in model_name else self.base_analyzer
            result = analyzer.analyze_incident(str(temp_file), top_k_rules=3)
            
            # Extract metrics
            cvc_codes = self._extract_cvc_codes(result)
            fault_check = self._check_fault_identification(result, case["ground_truth"])
            evidence_score = self._check_evidence_usage(result, case["ground_truth"])
            format_check = self._check_format_compliance(result)
            
            # Check CVC code accuracy
            ground_truth_codes = set(case["ground_truth"]["cvc_codes"])
            extracted_codes_normalized = set()
            for code in cvc_codes:
                # Normalize: "CVC 21453(a)" -> "21453"
                normalized = re.sub(r'CVC\s*', '', code, flags=re.IGNORECASE)
                normalized = re.sub(r'\([a-z]\)', '', normalized)
                extracted_codes_normalized.add(normalized)
            
            ground_truth_normalized = set()
            for code in ground_truth_codes:
                normalized = re.sub(r'CVC\s*', '', code, flags=re.IGNORECASE)
                normalized = re.sub(r'\([a-z]\)', '', normalized)
                ground_truth_normalized.add(normalized)
            
            cvc_correct = len(extracted_codes_normalized & ground_truth_normalized) > 0
            
            return {
                "case_name": case["name"],
                "model": model_name,
                "result": result,
                "cvc_codes_found": cvc_codes,
                "cvc_correct": cvc_correct,
                "fault_correct": fault_check["overall_correct"],
                "evidence_score": evidence_score,
                "format_compliant": format_check["has_three_bullets"],
                "bullet_count": format_check["bullet_count"],
                "details": {
                    "fault_check": fault_check,
                    "format_check": format_check
                }
            }
        
        except Exception as e:
            return {
                "case_name": case["name"],
                "model": model_name,
                "error": str(e),
                "cvc_correct": False,
                "fault_correct": False,
                "evidence_score": 0,
                "format_compliant": False
            }
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
    
    def run_evaluation(self) -> Dict:
        """Run full evaluation on both models."""
        print("\n" + "="*70)
        print("TRAFFIC ANALYST MODEL ACCURACY EVALUATION")
        print("="*70 + "\n")
        
        results = {
            "base_model": [],
            "custom_model": []
        }
        
        # Evaluate base model
        print("Evaluating BASE MODEL (llama3.1)...")
        print("-" * 70)
        for i, case in enumerate(self.test_cases, 1):
            print(f"  {i}/{len(self.test_cases)} - {case['name']}...", end=" ", flush=True)
            result = self.evaluate_case(case, "llama3.1")
            results["base_model"].append(result)
            status = "✓" if result.get("cvc_correct") and result.get("fault_correct") else "✗"
            print(status)
        
        print("\n" + "="*70 + "\n")
        
        # Evaluate custom model
        print("Evaluating CUSTOM MODEL (traffic-analyst)...")
        print("-" * 70)
        for i, case in enumerate(self.test_cases, 1):
            print(f"  {i}/{len(self.test_cases)} - {case['name']}...", end=" ", flush=True)
            result = self.evaluate_case(case, "traffic-analyst")
            results["custom_model"].append(result)
            status = "✓" if result.get("cvc_correct") and result.get("fault_correct") else "✗"
            print(status)
        
        return results
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate aggregate metrics."""
        metrics = {}
        
        for model_key in ["base_model", "custom_model"]:
            model_results = results[model_key]
            
            # Filter out errors
            valid_results = [r for r in model_results if "error" not in r]
            
            if not valid_results:
                metrics[model_key] = {"error": "No valid results"}
                continue
            
            metrics[model_key] = {
                "total_cases": len(valid_results),
                "cvc_accuracy": sum(r["cvc_correct"] for r in valid_results) / len(valid_results),
                "fault_accuracy": sum(r["fault_correct"] for r in valid_results) / len(valid_results),
                "evidence_score": sum(r["evidence_score"] for r in valid_results) / len(valid_results),
                "format_compliance": sum(r["format_compliant"] for r in valid_results) / len(valid_results),
                "overall_accuracy": sum(
                    r["cvc_correct"] and r["fault_correct"] 
                    for r in valid_results
                ) / len(valid_results)
            }
        
        return metrics
    
    def print_report(self, results: Dict, metrics: Dict):
        """Print evaluation report."""
        print("\n" + "="*70)
        print("ACCURACY REPORT")
        print("="*70 + "\n")
        
        for model_key, model_name in [("base_model", "BASE (llama3.1)"), ("custom_model", "CUSTOM (traffic-analyst)")]:
            if "error" in metrics[model_key]:
                print(f"{model_name}: Error in evaluation")
                continue
            
            m = metrics[model_key]
            print(f"{model_name}:")
            print(f"  CVC Code Accuracy:    {m['cvc_accuracy']*100:.1f}%")
            print(f"  Fault Accuracy:       {m['fault_accuracy']*100:.1f}%")
            print(f"  Evidence Usage:       {m['evidence_score']*100:.1f}%")
            print(f"  Format Compliance:    {m['format_compliance']*100:.1f}%")
            print(f"  Overall Accuracy:     {m['overall_accuracy']*100:.1f}%")
            print()
        
        # Improvement calculation
        if "error" not in metrics["base_model"] and "error" not in metrics["custom_model"]:
            print("IMPROVEMENT (Custom vs Base):")
            improvements = {
                "CVC Accuracy": metrics["custom_model"]["cvc_accuracy"] - metrics["base_model"]["cvc_accuracy"],
                "Fault Accuracy": metrics["custom_model"]["fault_accuracy"] - metrics["base_model"]["fault_accuracy"],
                "Evidence Usage": metrics["custom_model"]["evidence_score"] - metrics["base_model"]["evidence_score"],
                "Format Compliance": metrics["custom_model"]["format_compliance"] - metrics["base_model"]["format_compliance"],
                "Overall": metrics["custom_model"]["overall_accuracy"] - metrics["base_model"]["overall_accuracy"]
            }
            
            for metric, improvement in improvements.items():
                sign = "+" if improvement >= 0 else ""
                print(f"  {metric:20s} {sign}{improvement*100:.1f}%")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING ACCURACY EVALUATION")
    print("="*70)
    print("\nThis will test both models on 5 traffic scenarios:")
    print("  1. Red light violation")
    print("  2. Pedestrian right-of-way")
    print("  3. Stop sign violation")
    print("  4. Following too closely")
    print("  5. Green light with pedestrian")
    print("\nMetrics measured:")
    print("  - CVC Code Accuracy (correct rule citations)")
    print("  - Fault Accuracy (correct party identification)")
    print("  - Evidence Usage (references detection data)")
    print("  - Format Compliance (3-bullet structure)")
    print("  - Overall Accuracy (all correct)")
    
    # Run evaluation
    evaluator = AccuracyEvaluator()
    results = evaluator.run_evaluation()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    
    # Print report
    evaluator.print_report(results, metrics)
    
    # Save detailed results
    output_file = Path("backend/fine_tuning/data/evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "results": results,
            "metrics": metrics
        }, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    print()
