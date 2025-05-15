"""Main evaluator class combining automatic and human evaluation metrics."""

from typing import List, Dict, Any
from .metrics import calculate_bert_score, calculate_rouge, calculate_bleu
from .human_eval import HumanEvaluator

class Evaluator:
    def __init__(self, use_human_eval: bool = True):
        """Initialize evaluator with optional human evaluation.
        
        Args:
            use_human_eval: Whether to include human-like evaluation using LLMs
        """
        self.use_human_eval = use_human_eval
        if use_human_eval:
            self.human_evaluator = HumanEvaluator()

    async def evaluate(self, results: List[Dict[str, str]]) -> Dict[str, float]:
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
        
        # Human evaluation if enabled
        if self.use_human_eval:
            human_scores = await self.human_evaluator.evaluate_batch(queries, generated)
            metrics.update(human_scores)
        
        return metrics

    def save_results(self, metrics: Dict[str, float], output_dir: str):
        """Save evaluation results to files.
        
        Args:
            metrics: Dictionary of evaluation metrics
            output_dir: Directory to save results
        """
        import os
        import json
        import pandas as pd
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([metrics])
        csv_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {output_dir}")
        print("\nMetrics Summary:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}") 