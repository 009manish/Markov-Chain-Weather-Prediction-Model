from __future__ import annotations
import json
import os
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable

@dataclass
class SerializableModel:
    states: List[str]
    transition_counts: Dict[str, Dict[str, int]]

class MarkovModel:
    def __init__(self):
        self.states: List[str] = []
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.transition_probs: Dict[str, Dict[str, float]] = {}

    # ------------------ Training ------------------
    def fit(self, seq: Iterable[str]) -> None:
        """Learn transitions from a sequence of states."""
        seq = [s.title() for s in seq]  # Normalize states
        if len(seq) < 2:
            raise ValueError("Need at least 2 observations to train the Markov model")
        
        self.states = sorted(set(seq))
        self.transition_counts = defaultdict(Counter)
        
        for a, b in zip(seq[:-1], seq[1:]):
            self.transition_counts[a][b] += 1
        
        # Compute probabilities
        self.transition_probs = {}
        for s in self.states:
            counter = self.transition_counts.get(s, Counter())
            total = sum(counter.values())
            if total == 0:
                self.transition_probs[s] = {t: 1.0 / len(self.states) for t in self.states}
            else:
                self.transition_probs[s] = {t: counter.get(t, 0) / total for t in self.states}

    # ------------------ Forecasting ------------------
    def predict_next_prob(self, current_state: str) -> Dict[str, float]:
        """Return next-state probabilities for a given state."""
        state = current_state.title()
        if state not in self.transition_probs:
            return {s: 1.0 / len(self.states) for s in self.states}
        return dict(self.transition_probs[state])

    def sample_next(self, current_state: str) -> str:
        """Randomly sample the next state based on transition probabilities."""
        probs = self.predict_next_prob(current_state)
        states = list(probs.keys())
        weights = list(probs.values())
        if sum(weights) <= 0:
            return random.choice(self.states)
        return random.choices(states, weights=weights, k=1)[0]

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Return the full transition probability matrix."""
        mat: Dict[str, Dict[str, float]] = {}
        for s in self.states:
            row = self.transition_probs.get(s, {})
            mat[s] = {t: row.get(t, 0.0) for t in self.states}
        return mat

    # ------------------ Serialization ------------------
    def to_serializable(self) -> SerializableModel:
        counts = {s: dict(self.transition_counts.get(s, {})) for s in self.transition_counts}
        return SerializableModel(states=self.states, transition_counts=counts)

    def save_to_json(self, filepath: str) -> None:
        serial = self.to_serializable()
        with open(filepath, 'w') as fh:
            json.dump(asdict(serial), fh, indent=2)
        print(f"Model saved to {filepath}")

    def load_from_json(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)
        with open(filepath, 'r') as fh:
            data = json.load(fh)

        self.states = data.get("states", [])
        counts = data.get("transition_counts", {})
        self.transition_counts = defaultdict(Counter)
        for s, d in counts.items():
            self.transition_counts[s] = Counter(d)
        
        # Recompute probabilities
        self.transition_probs = {}
        for s in self.states:
            counter = self.transition_counts.get(s, Counter())
            total = sum(counter.values())
            if total == 0:
                self.transition_probs[s] = {t: 1.0 / len(self.states) for t in self.states}
            else:
                self.transition_probs[s] = {t: counter.get(t, 0) / total for t in self.states}
        
        print(f"Model loaded from {filepath}")
