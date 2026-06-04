from __future__ import annotations
from M1 import DataLoader
from M2 import MarkovModel
from M3 import Predictor
import csv
import json
import os
import random
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Tuple, Optional

class Menu:
    def __init__(self):
        self.data_sequence: Optional[List[str]] = None
        self.data_filepath: Optional[str] = None
        self.model = MarkovModel()
        self.trained = False

    def run(self) -> None:
        print('----------------------------------')
        print('----------------------------------')
        print('Markov Weather Prediction System')
        print('----------------------------------')
        print('----------------------------------')
        while True:
            try:
                self.print_menu()
                choice = input('Choose an option (1-11): ').strip()
                if not choice:
                    continue
                if choice == '1':
                    self.option_load_csv()
                elif choice == '2':
                    self.option_generate_sample()
                elif choice == '3':
                    self.option_train()
                elif choice == '4':
                    self.option_view_matrix()
                elif choice == '5':
                    self.option_forecast()
                elif choice == '6':
                    self.option_simulate()
                elif choice == '7':
                    self.option_evaluate()
                elif choice == '8':
                    self.option_save_model()
                elif choice == '9':
                    self.option_load_model()
                elif choice == '10':
                    self.option_show_data_summary()
                elif choice == '11':
                    print('Goodbye!')
                    break
                else:
                    print('Invalid choice — enter a number between 1 and 11')
            except Exception as e:
                print(f'Error: {e}')

    def print_menu(self) -> None:
        print('\nMenu:')
        print(' 1) Load CSV (weather data)')
        print(' 2) Generate sample CSV')
        print(' 3) Train Markov model on loaded data')
        print(' 4) View transition probability matrix')
        print(' 5) Forecast (most-likely / Monte Carlo distribution)')
        print(' 6) Run Monte Carlo simulation and show sample trajectories')
        print(' 7) Evaluate next-day greedy accuracy (train/test split)')
        print(' 8) Save model to JSON')
        print(' 9) Load model from JSON')
        print('10) Show loaded data summary (counts, sample rows)')
        print('11) Exit')

    # ------------------ Features ------------------
    def option_load_csv(self) -> None:
        path = input('Enter path to CSV (e.g., file.csv): ').strip()
        if not path:
            print('No path entered — cancelled')
            return
        try:
            seq = DataLoader.load_csv(path)
            self.data_sequence = seq
            self.data_filepath = path
            print(f'Loaded {len(seq)} rows from {path}.')
        except Exception as e:
            print(f'Failed to load CSV: {e}')

    def option_generate_sample(self) -> None:
        path = input('Write sample CSV to (default: sample_weather.csv): ').strip() or 'sample_weather.csv'
        days_str = input('Days to generate (default 365): ').strip()
        days = 365
        if days_str:
            try:
                days = int(days_str)
            except Exception:
                print('Invalid days value — using default 365')
        DataLoader.generate_sample(path, days=days)

    def option_train(self) -> None:
        if not self.data_sequence:
            print('No data loaded. Use option 1 to load a CSV first.')
            return
        try:
            self.model = MarkovModel()
            self.model.fit(self.data_sequence)
            self.trained = True
            print('Model trained successfully.')
            print(f'States: {self.model.states}')
        except Exception as e:
            print(f'Failed to train model: {e}')

    def option_view_matrix(self) -> None:
        if not self.trained:
            print('Model is not trained yet. Train the model first (option 3).')
            return
        mat = self.model.get_transition_matrix()
        self.pretty_print_matrix(mat)

    def option_forecast(self) -> None:
        if not self.trained:
            print('Model is not trained. Train it first.')
            return
        start = input("Enter current state (or press Enter to use most frequent state): ").strip()
        if not start:
            start = self.most_frequent_state() or (self.model.states[0] if self.model.states else None)
            print(f"Using most frequent state as start: {start}")
        if start not in self.model.states:
            print(f"Start state '{start}' not known in model states: {self.model.states}")
            return
        days = self.read_int('Days to forecast (default 7): ', default=7, minval=1)
        sims = self.read_int('Monte Carlo sims for distribution (default 500): ', default=500, minval=1)
        predictor = Predictor(self.model)
        ml_path = predictor.most_likely_path(start, days)
        print('\nMost-likely (greedy) path:')
        print(' -> '.join(ml_path))
        print('\nEstimating distribution by Monte Carlo...')
        dist = predictor.forecast_distribution(start, days, sims=sims)
        for i, d in enumerate(dist, start=1):
            print(f"Day +{i} distribution:")
            for s, p in sorted(d.items(), key=lambda kv: -kv[1]):
                print(f"  {s}: {p:.3f}")

    def option_simulate(self) -> None:
        if not self.trained:
            print('Train model first.')
            return
        start = input('Enter start state (or press Enter to use most frequent): ').strip()
        if not start:
            start = self.most_frequent_state() or self.model.states[0]
            print(f"Using {start}")
        if start not in self.model.states:
            print(f"Unknown start '{start}'")
            return
        days = self.read_int('Days per simulation (default 7): ', default=7, minval=1)
        sims = self.read_int('Number of simulations to run (default 10): ', default=10, minval=1)
        predictor = Predictor(self.model)
        sims_res = predictor.simulate(start, days, sims=sims)
        print('\nSample trajectories (showing up to 10):')
        for i, traj in enumerate(sims_res[:10], start=1):
            print(f" {i}) {start} -> " + ' -> '.join(traj))

    def option_evaluate(self) -> None:
        if not self.data_sequence:
            print('Load data first to evaluate.')
            return
        test_frac = None
        while test_frac is None:
            s = input('Fraction to hold out for testing (e.g., 0.2) [default 0.2]: ').strip() or '0.2'
            try:
                test_frac = float(s)
                if not (0.0 < test_frac < 1.0):
                    print('Enter a number between 0 and 1 (exclusive).')
                    test_frac = None
            except Exception:
                print('Invalid number, try again.')
        n = len(self.data_sequence)
        split = max(1, int(n * (1.0 - test_frac)))
        train_seq = self.data_sequence[:split]
        test_seq = self.data_sequence[split:]
        if len(test_seq) < 2:
            print('Not enough data for testing after split; reduce test fraction or provide more data.')
            return
        model = MarkovModel(); model.fit(train_seq)
        pred = Predictor(model)
        acc = pred.accuracy_next_day(test_seq)
        print(f'Train size: {len(train_seq)}, Test size: {len(test_seq)}')
        if acc != acc:  # NaN check
            print('Accuracy not defined (insufficient transitions).')
        else:
            print(f'Next-day greedy accuracy on held-out test set: {acc:.3%}')

    def option_save_model(self) -> None:
        if not self.trained:
            print('Train model first before saving.')
            return
        path = input('Enter JSON filepath to save model [default: model.json]: ').strip() or 'model.json'
        try:
            self.model.save_to_json(path)
        except Exception as e:
            print(f'Failed to save model: {e}')

    def option_load_model(self) -> None:
        path = input('Enter JSON filepath to load model from: ').strip()
        if not path:
            print('No path provided.')
            return
        try:
            self.model = MarkovModel()
            self.model.load_from_json(path)
            self.trained = True
        except Exception as e:
            print(f'Failed to load model: {e}')

    def option_show_data_summary(self) -> None:
        if not self.data_sequence:
            print('No data loaded.')
            return
        seq = self.data_sequence
        print(f'Loaded data from: {self.data_filepath} (count = {len(seq)})')
        ctr = Counter(seq)
        print('Top states by frequency:')
        for s, c in ctr.most_common(10):
            print(f'  {s}: {c}')
        print('\nSample (first 20 rows):')
        for i, s in enumerate(seq[:20], start=1):
            print(f' {i:3d}) {s}')

    # ------------------ Helpers ------------------
    def pretty_print_matrix(self, mat: Dict[str, Dict[str, float]]) -> None:
        states = list(mat.keys())
        # header
        header = 'From \ To'.ljust(16) + ''.join(s.ljust(12) for s in states)
        print('\n' + header)
        print('-' * len(header))
        for r in states:
            row = r.ljust(16)
            for c in states:
                row += f"{mat[r].get(c, 0.0):.3f}".ljust(12)
            print(row)

    def most_frequent_state(self) -> Optional[str]:
        if not self.data_sequence:
            return None
        return Counter(self.data_sequence).most_common(1)[0][0]

    @staticmethod
    def read_int(prompt: str, default: int = 0, minval: Optional[int] = None) -> int:
        while True:
            s = input(prompt).strip()
            if not s:
                val = default
                return val
            try:
                v = int(s)
                if minval is not None and v < minval:
                    print(f'Enter a value >= {minval}')
                    continue
                return v
            except Exception:
                print('Please enter a valid integer')


#----Entrypoint-----------
def main():
    menu = Menu()
    menu.run()


if __name__ == '__main__':
    main()
