from __future__ import annotations
import csv
import os
import random
from typing import List, Iterable, Optional

class DataLoader:
    @staticmethod
    def find_weather_column(fieldnames: Iterable[str]) -> Optional[str]:
        """Return the first header name that matches 'weather' """
        if not fieldnames:
            return None
        for name in fieldnames:
            if name and name.strip().lower() == 'weather':
                return name
        for name in fieldnames:
            if 'weather' in name.lower():
                return name
        return None

    @staticmethod
    def load_csv(filepath: str, weather_col: str = 'weather') -> List[str]:
        """Load weather column from CSV and return a list of weather states """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        seq: List[str] = []
        with open(filepath, newline='') as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row / field names")
            
            col_to_use = weather_col if weather_col in reader.fieldnames else DataLoader.find_weather_column(reader.fieldnames)
            if col_to_use is None:
                raise ValueError(f"Could not find a 'weather' column. Found: {reader.fieldnames}")
            
            for row in reader:
                raw = row.get(col_to_use, '')
                if raw is None:
                    continue
                val = str(raw).strip().title() 
                if val:
                    seq.append(val)
        
        if not seq:
            raise ValueError(f"No weather data read from '{filepath}' (column '{col_to_use}').")
        
        return seq

    @staticmethod
    def generate_sample(filepath: str, days: int = 365) -> None:
        """Generate a sample weather CSV for testing/training."""
        states = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
        base_probs = {'Sunny': 0.5, 'Cloudy': 0.25, 'Rainy': 0.18, 'Snowy': 0.07}

        with open(filepath, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['date', 'weather'])
            
            for i in range(days):
                p = base_probs.copy()
                if 90 <= (i % 365) <= 270:
                    p['Sunny'] += 0.12
                    p['Snowy'] = max(p['Snowy'] - 0.05, 0.0)
                
                states_list = list(p.keys())
                weights = list(p.values())
                choice = random.choices(states_list, weights=weights, k=1)[0]
                writer.writerow([f'2020-01-01+{i}', choice.title()]) 

        print(f"Sample weather CSV generated at: {filepath} ({days} rows)")