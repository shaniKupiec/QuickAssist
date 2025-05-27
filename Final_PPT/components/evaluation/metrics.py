"""Automatic evaluation metrics for text generation."""

from typing import List, Dict
from bert_score import score
from rouge_score import rouge_scorer
import numpy as np
from collections import Counter
import math


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

def tokenize(text: str) -> List[str]:
    return text.lower().split()

def ngramify(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def calculate_bleu(generated: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU score for generated responses using math and Counter.

    Args:
        generated: List of generated responses.
        references: List of reference responses (same length as generated).

    Returns:
        Dictionary containing BLEU score.
    """
    max_n = 4
    weights = [1.0 / max_n] * max_n
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    cand_len_total = 0
    ref_len_total = 0

    for cand_str, ref_str in zip(generated, references):
        cand = tokenize(cand_str)
        ref = tokenize(ref_str)
        cand_len_total += len(cand)
        ref_len_total += len(ref)

        for n in range(1, max_n + 1):
            cand_ngrams = Counter(ngramify(cand, n))
            ref_ngrams = Counter(ngramify(ref, n))

            max_ref = ref_ngrams
            for ng in cand_ngrams:
                clipped = min(cand_ngrams[ng], max_ref.get(ng, 0))
                clipped_counts[n-1] += clipped
                total_counts[n-1] += cand_ngrams[ng]

    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(1e-9)  # smoothing
        else:
            precision = clipped_counts[i] / total_counts[i]
            if precision == 0:
                precision = 1e-9  # smoothing
            precisions.append(precision)

    log_precisions = [math.log(p) for p in precisions]
    geometric_mean = math.exp(sum(w * lp for w, lp in zip(weights, log_precisions)))

    # Brevity penalty
    if cand_len_total == 0:
        bp = 0.0
    elif cand_len_total > ref_len_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len_total / cand_len_total)

    bleu_score = bp * geometric_mean
    return {'bleu': bleu_score}