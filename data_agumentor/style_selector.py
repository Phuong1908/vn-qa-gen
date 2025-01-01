from collections import Counter
from typing import List, Tuple, Dict, Set
from underthesea import ner, pos_tag
import pickle
import os
import numpy as np

from data_agumentor.answer_selector import select_answers
from data_agumentor.config import QUESTION_TYPES, QUESTION_PATTERNS


def get_question_type(question: str) -> Tuple[str, int]:
    """
    Determine question type for Vietnamese questions
    Returns: (question_type, type_id)
    """
    question = question.lower()

    # Check each question type pattern
    for qtype, patterns in QUESTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in question:
                return qtype, QUESTION_TYPES.index(qtype)

    return "OTHER", len(QUESTION_TYPES) - 1


def get_question_type_probs(answer_tag: str, sample_probs: Dict) -> List[float]:
    """Get probabilities for each question type given answer tag"""
    probs = []
    for qtype in QUESTION_TYPES:
        key = f"{qtype}_{answer_tag}"
        prob = sample_probs.get("s|c,a", {}).get(key, 1.0)
        probs.append(prob)
    return probs
