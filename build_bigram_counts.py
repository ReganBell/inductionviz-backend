#!/usr/bin/env python3
"""Build sparse bigram counts from corpus and save to JSON."""

import json
from collections import defaultdict
from typing import Dict, List
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def encode(text: str) -> List[int]:
    return enc.encode(text)

def build_bigram_counts() -> Dict[int, Dict[int, int]]:
    """Build bigram counts, only storing transitions that actually exist."""
    with open('words.txt', 'r') as f:
        content = f.read()
        counts = defaultdict(lambda: defaultdict(int))
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            ids = encode(line)
            for a, b in zip(ids[:-1], ids[1:]):
                counts[a][b] += 1

    # Convert to regular dict (only stores actual transitions)
    return {prev: dict(next_counts) for prev, next_counts in counts.items()}

if __name__ == "__main__":
    print("Building bigram counts from corpus...")
    bigram_counts = build_bigram_counts()

    # Calculate stats
    total_transitions = sum(len(next_tokens) for next_tokens in bigram_counts.values())
    print(f"Total unique transitions: {total_transitions:,}")
    print(f"Tokens that have transitions: {len(bigram_counts):,}")

    # Save to JSON
    print("Saving to bigram_counts.json...")
    with open('bigram_counts.json', 'w') as f:
        json.dump(bigram_counts, f)

    print("Done!")
