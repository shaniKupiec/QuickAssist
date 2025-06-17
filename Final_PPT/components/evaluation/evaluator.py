"""Main evaluator class combining automatic and human evaluation metrics."""

from typing import List, Dict, Any
from .metrics import calculate_bert_score, calculate_rouge, calculate_bleu
from .human_eval import HumanEvaluator
from .intent_eval import calculate_intent_accuracy
import csv
import json

class Evaluator:
    def __init__(self, api_key: str, model_name: str, main_config):
        """Initialize evaluator with optional human evaluation.
        """
        self.main_config = main_config
        self.human_evaluator = HumanEvaluator(api_key, model_name, max_samples=self.main_config['max_eval_samples'])

    async def evaluate(self, results: List[Dict[str, str]], save_intent_to_calc_accuracy, test_data) -> Dict[str, float]:
        """Evaluate generated responses using multiple metrics.
        
        Args:
            results: List of dictionaries containing 'query', 'generated_response',
                    and 'reference_response' keys
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Extract generated and reference responses
        generated = [r['generated_response'] for r in results]
        references = [r['reference_response'] for r in results]
        queries = [r['query'] for r in results]
        intent = [r.get('intent', '') for r in results]
        
        # Calculate automatic metrics
        metrics = {}
        
        # BERTScore
        bert_scores = calculate_bert_score(generated, references)
        metrics.update(bert_scores)

        # ROUGE scores
        rouge_scores = calculate_rouge(generated, references)
        metrics.update(rouge_scores)
        
        # BLEU score
        bleu_score = calculate_bleu(generated, references)
        metrics.update(bleu_score)
        
        intent_accuracy = None
        if save_intent_to_calc_accuracy:
            accuracy_value = calculate_intent_accuracy(test_data, save_intent_to_calc_accuracy)
            intent_accuracy = {"intent_accuracy": accuracy_value}
        
        # Human evaluation if enabled
        human_scores = await self.human_evaluator.evaluate_batch(queries, generated)
        human_metrics = self.human_evaluator.avgMetricsHumanScore(human_scores)
        metrics.update(human_metrics)

        if self.main_config["save_results"]:
            self.save_results(results, intent, human_scores, metrics, intent_accuracy)
        
        return {
            "metrics": metrics,
            "intent_accuracy": intent_accuracy
        }

    def save_results(
        self,
        results: List[Dict[str, str]],
        intent: List[str],
        human_scores: List[Dict[str, int] | None],
        metrics: Dict[str, float],
        intent_accuracy: Dict[str, float],
        human_scores_csv_path: str = "human_scores.csv",
        metrics_file_path: str = "metrics.json",
        intent_accuracy_file_path: str = "intent_accuracy.json"
    ):
        """Save human evaluation scores alongside original data to CSV and metrics dict to file.
        
        Args:
            results: Original results with query, generated_response, reference_response
            intent: List of intent (aligned with results)
            human_scores: List of human evaluation dicts or None
            metrics: Aggregated evaluation metrics dict
            human_scores_csv_path: Path to save human scores CSV
            metrics_file_path: Path to save aggregated metrics (JSON)
        """
        # Prepare rows for the CSV
        rows = []
        # Pad or truncate human_scores to match the length of results
        full_human_scores = human_scores + [None] * (len(results) - len(human_scores))

        # Now iterate using zip with padded scores
        for i, (res, intent_val, score) in enumerate(zip(results, intent, full_human_scores)):
            if score is None:
                score = {
                    "Helpfulness": "",
                    "Fluency": "",
                    "Appropriateness": "",
                    "average_score": ""
                }
            rows.append({
                "query": res['query'],
                "intent": intent_val,
                "reference_response": res['reference_response'],
                "generated_response": res['generated_response'],
                "Helpfulness": score.get("Helpfulness", ""),
                "Fluency": score.get("Fluency", ""),
                "Appropriateness": score.get("Appropriateness", ""),
                "avg_score": score.get("average_score", "")
            })


        # Write human_scores CSV
        with open(human_scores_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                "query", "intent", "reference_response", "generated_response",
                "Helpfulness", "Fluency", "Appropriateness", "avg_score"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        # Save metrics dict as JSON
        with open(metrics_file_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        # Save metrics dict as JSON
        if intent_accuracy:
            with open(intent_accuracy_file_path, "w", encoding="utf-8") as f:
                json.dump(intent_accuracy, f, indent=4)

        print(f"✅ Saved")
