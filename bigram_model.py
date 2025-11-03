"""
Sparse bigram model loader for OpenWebText bigram predictions.
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tiktoken


class SparseBigramModel:
    """Sparse bigram probability model for GPT-2 vocabulary."""

    def __init__(
        self,
        bigram_counts: Dict[Tuple[int, int], int],
        row_sums: Dict[int, int],
        vocab_size: int,
        smoothing: float = 0.1
    ):
        self.bigram_counts = bigram_counts
        self.row_sums = row_sums
        self.V = vocab_size
        self.smoothing = smoothing
        self.enc = tiktoken.get_encoding("gpt2")

    def get_top_k_next(self, token_id: int, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k most likely next tokens with their probabilities.

        Args:
            token_id: The input token ID
            k: Number of top predictions to return

        Returns:
            List of (next_token_id, probability) tuples, sorted by probability descending
        """
        total = self.row_sums.get(token_id, 0) + self.smoothing * self.V

        if total == 0:
            # Token never seen, return empty
            return []

        # Build probability distribution (sparse)
        probs = {}
        for (i, j), count in self.bigram_counts.items():
            if i == token_id:
                probs[j] = (count + self.smoothing) / total

        if not probs:
            return []

        # Sort by probability descending
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]


def load_bigram_model(path: Path) -> Optional[SparseBigramModel]:
    """
    Load bigram model from pickle file.

    Args:
        path: Path to the pickle file containing bigram data

    Returns:
        SparseBigramModel instance or None if file doesn't exist
    """
    if not path.exists():
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return SparseBigramModel(
        bigram_counts=data['bigram_counts'],
        row_sums=data['row_sums'],
        vocab_size=data['vocab_size'],
        smoothing=0.1
    )
