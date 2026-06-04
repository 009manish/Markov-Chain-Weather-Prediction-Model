from __future__ import annotations
from M2 import MarkovModel
import csv
import json
import os
import random
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Tuple, Optional

class Predictor:
    def __init__(self, model: MarkovModel):
        self.model = model

    def most_likely_next(self, current_state: str) -> str:
        probs = self.model.predict_next_prob(current_state)
        if not probs:
            raise ValueError('Model is not trained or has no states')
        return max(probs.items(), key=lambda kv: kv[1])[0]

    def most_likely_path(self, start_state: str, days: int) -> List[str]:
        path = []
        cur = start_state
        for _ in range(days):
            nxt = self.most_likely_next(cur)
            path.append(nxt)
            cur = nxt
        return path

    def simulate(self, start_state: str, days: int, sims: int = 1000) -> List[List[str]]:
        sims = max(1, int(sims))
        results = []
        for i in range(sims):
            cur = start_state
            traj = []
            for _ in range(days):
                cur = self.model.sample_next(cur)
                traj.append(cur)
            results.append(traj)
        return results

    def forecast_distribution(self, start_state: str, days: int, sims: int = 500) -> List[Dict[str, float]]:
        sims_res = self.simulate(start_state, days, sims=sims)
        distributions: List[Dict[str, float]] = []
        for d in range(days):
            cnt = Counter(sim[d] for sim in sims_res)
            total = sum(cnt.values())
            dist = {s: (cnt.get(s, 0) / total) for s in self.model.states}
            distributions.append(dist)
        return distributions

    def accuracy_next_day(self, seq: List[str]) -> float:
        if len(seq) < 2:
            return float('nan')
        correct = 0
        total = 0
        for a, b in zip(seq[:-1], seq[1:]):
            probs = self.model.predict_next_prob(a)
            if not probs:
                continue
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
            if pred == b:
                correct += 1
            total += 1
        return (correct / total) if total > 0 else float('nan')
