"""Automatic evaluation metrics for text generation."""

from typing import List, Dict, Any
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
import numpy as np

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

def calculate_bert_score(generated: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore for generated responses.
    
    Args:
        generated: List of generated responses
        references: List of reference responses
    
    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    P, R, F1 = score(generated, references, lang="en", verbose=False)
    return {
        'bert_score_precision': P.mean().item(),
        'bert_score_recall': R.mean().item(),
        'bert_score_f1': F1.mean().item()
    }

def calculate_rouge(generated: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for generated responses.
    
    Args:
        generated: List of generated responses
        references: List of reference responses
    
    Returns:
        Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1_f': [],
        'rouge2_f': [],
        'rougeL_f': []
    }
    
    for gen, ref in zip(generated, references):
        score = scorer.score(ref, gen)
        scores['rouge1_f'].append(score['rouge1'].fmeasure)
        scores['rouge2_f'].append(score['rouge2'].fmeasure)
        scores['rougeL_f'].append(score['rougeL'].fmeasure)
    
    return {
        'rouge1_f': np.mean(scores['rouge1_f']),
        'rouge2_f': np.mean(scores['rouge2_f']),
        'rougeL_f': np.mean(scores['rougeL_f'])
    }

def calculate_bleu(generated: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU score for generated responses.
    
    Args:
        generated: List of generated responses
        references: List of reference responses
    
    Returns:
        Dictionary containing BLEU score
    """
    # Tokenize the texts
    tokenized_gen = [nltk.word_tokenize(gen.lower()) for gen in generated]
    tokenized_ref = [[nltk.word_tokenize(ref.lower())] for ref in references]
    
    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(
        tokenized_ref,
        tokenized_gen,
        smoothing_function=smoothing
    )
    
    return {'bleu': bleu_score} 