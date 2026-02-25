"""
Vietlott AI Prediction Engine
Kết hợp: LSTM Neural Network + Random Forest + Statistical Analysis + Ensemble
"""
import numpy as np
import pandas as pd
import json
import os
import pickle
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# 1. LSTM Neural Network Model
# ============================================================
class LSTMLotteryModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=55):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class LSTMPredictor:
    def __init__(self, max_number, seq_length=20):
        self.max_number = max_number
        self.seq_length = seq_length
        self.model = None
        self.scaler = MinMaxScaler()

    def prepare_data(self, draws):
        """Convert draw history to binary matrix for LSTM"""
        # Each draw -> binary vector of length max_number
        binary_matrix = []
        for draw in draws:
            vec = np.zeros(self.max_number)
            for num in draw:
                if 1 <= num <= self.max_number:
                    vec[num - 1] = 1
            binary_matrix.append(vec)

        binary_matrix = np.array(binary_matrix)

        # Create sequences
        X, y = [], []
        for i in range(len(binary_matrix) - self.seq_length):
            X.append(binary_matrix[i:i + self.seq_length])
            y.append(binary_matrix[i + self.seq_length])

        return np.array(X), np.array(y)

    def train(self, draws, epochs=50, lr=0.001):
        """Train LSTM model"""
        if len(draws) < self.seq_length + 10:
            print("Not enough data for LSTM training")
            return

        X, y = self.prepare_data(draws)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        self.model = LSTMLotteryModel(
            input_size=self.max_number,
            hidden_size=128,
            num_layers=2,
            output_size=self.max_number
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            # Mini-batch training
            batch_size = 32
            total_loss = 0
            n_batches = 0

            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_tensor[batch_idx]
                batch_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"  LSTM Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def predict(self, recent_draws, n_numbers=6, n_predictions=5):
        """Generate predictions using trained LSTM"""
        if self.model is None:
            return []

        self.model.eval()
        predictions = []

        # Use the most recent sequence
        binary_seq = []
        for draw in recent_draws[-self.seq_length:]:
            vec = np.zeros(self.max_number)
            for num in draw:
                if 1 <= num <= self.max_number:
                    vec[num - 1] = 1
            binary_seq.append(vec)

        # Pad if needed
        while len(binary_seq) < self.seq_length:
            binary_seq.insert(0, np.zeros(self.max_number))

        input_tensor = torch.FloatTensor([binary_seq])

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = output.numpy()[0]

        # Generate multiple predictions with different strategies
        for i in range(n_predictions):
            # Add some noise for variety
            noise = np.random.normal(0, 0.05 * (i + 1), self.max_number)
            adjusted_probs = np.clip(probabilities + noise, 0, 1)

            # Select top numbers
            top_indices = np.argsort(adjusted_probs)[-n_numbers*2:]
            selected = np.random.choice(top_indices, size=n_numbers, replace=False)
            prediction = sorted((selected + 1).tolist())
            predictions.append({
                'numbers': prediction,
                'confidence': float(np.mean(adjusted_probs[selected])) * 100,
                'method': 'LSTM Neural Network'
            })

        return predictions


# ============================================================
# 2. Random Forest + Gradient Boosting Model
# ============================================================
class MLPredictor:
    def __init__(self, max_number):
        self.max_number = max_number
        self.rf_model = None
        self.gb_model = None

    def extract_features(self, draws, window=10):
        """Extract statistical features from draw history"""
        features_list = []
        labels_list = []

        for i in range(window, len(draws)):
            recent = draws[i-window:i]
            target = draws[i]

            # Feature engineering
            features = []

            # 1. Frequency of each number in recent window
            freq = Counter()
            for draw in recent:
                for num in draw:
                    freq[num] += 1
            freq_vector = [freq.get(n, 0) / window for n in range(1, self.max_number + 1)]
            features.extend(freq_vector)

            # 2. Gap since last appearance
            gap_vector = []
            for n in range(1, self.max_number + 1):
                gap = window
                for j, draw in enumerate(reversed(recent)):
                    if n in draw:
                        gap = j
                        break
                gap_vector.append(gap / window)
            features.extend(gap_vector)

            # 3. Sum and spread statistics
            sums = [sum(d) for d in recent]
            features.append(np.mean(sums))
            features.append(np.std(sums))

            # 4. Consecutive number patterns
            consec_counts = []
            for draw in recent:
                c = 0
                sorted_d = sorted(draw)
                for k in range(len(sorted_d)-1):
                    if sorted_d[k+1] - sorted_d[k] == 1:
                        c += 1
                consec_counts.append(c)
            features.append(np.mean(consec_counts))

            # 5. Odd/Even ratio
            odd_ratios = [sum(1 for n in d if n % 2 == 1) / len(d) for d in recent]
            features.append(np.mean(odd_ratios))

            # 6. High/Low ratio
            mid = self.max_number // 2
            high_ratios = [sum(1 for n in d if n > mid) / len(d) for d in recent]
            features.append(np.mean(high_ratios))

            features_list.append(features)

            # Labels: binary vector for target draw
            label = np.zeros(self.max_number)
            for num in target:
                if 1 <= num <= self.max_number:
                    label[num - 1] = 1
            labels_list.append(label)

        return np.array(features_list), np.array(labels_list)

    def train(self, draws):
        """Train Random Forest and Gradient Boosting models"""
        X, y = self.extract_features(draws)

        if len(X) < 10:
            print("Not enough data for ML training")
            return

        # Train a model for each number position
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        self.gb_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )

        # Multi-label: train one model per number
        self.rf_models = {}
        self.gb_models = {}

        for num_idx in range(self.max_number):
            y_single = y[:, num_idx]
            if y_single.sum() > 5:  # Need enough positive examples
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X, y_single)
                self.rf_models[num_idx] = rf

                gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
                gb.fit(X, y_single)
                self.gb_models[num_idx] = gb

        print(f"  ML Models trained for {len(self.rf_models)} numbers")

    def predict(self, draws, n_numbers=6, n_predictions=5):
        """Generate predictions using RF + GB ensemble"""
        if not self.rf_models:
            return []

        X, _ = self.extract_features(draws)
        if len(X) == 0:
            return []

        last_features = X[-1:].reshape(1, -1)
        predictions = []

        # Get probability scores from both models
        rf_probs = np.zeros(self.max_number)
        gb_probs = np.zeros(self.max_number)

        for num_idx in range(self.max_number):
            if num_idx in self.rf_models:
                rf_probs[num_idx] = self.rf_models[num_idx].predict_proba(last_features)[0][1] if len(self.rf_models[num_idx].classes_) == 2 else 0.1
            if num_idx in self.gb_models:
                gb_probs[num_idx] = self.gb_models[num_idx].predict_proba(last_features)[0][1] if len(self.gb_models[num_idx].classes_) == 2 else 0.1

        # Combine RF and GB
        combined_probs = 0.6 * rf_probs + 0.4 * gb_probs

        for i in range(n_predictions):
            noise = np.random.normal(0, 0.03 * (i + 1), self.max_number)
            adjusted = np.clip(combined_probs + noise, 0, 1)

            top_indices = np.argsort(adjusted)[-n_numbers*2:]
            selected = np.random.choice(top_indices, size=n_numbers, replace=False)
            prediction = sorted((selected + 1).tolist())

            predictions.append({
                'numbers': prediction,
                'confidence': float(np.mean(adjusted[selected])) * 100,
                'method': 'Random Forest + Gradient Boosting'
            })

        return predictions


# ============================================================
# 3. Statistical Analysis Model
# ============================================================
class StatisticalPredictor:
    def __init__(self, max_number):
        self.max_number = max_number

    def analyze(self, draws):
        """Comprehensive statistical analysis"""
        all_numbers = []
        for draw in draws:
            all_numbers.extend(draw)

        freq = Counter(all_numbers)
        total_draws = len(draws)

        analysis = {
            'frequency': {},
            'hot_numbers': [],
            'cold_numbers': [],
            'overdue_numbers': [],
            'pairs': [],
            'sum_range': {},
            'odd_even': {},
            'decade_distribution': {},
        }

        # Number frequency
        for n in range(1, self.max_number + 1):
            analysis['frequency'][n] = {
                'count': freq.get(n, 0),
                'percentage': round(freq.get(n, 0) / total_draws * 100, 2)
            }

        # Hot numbers (top 20% by frequency)
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_count = max(int(self.max_number * 0.2), 6)
        analysis['hot_numbers'] = [n for n, _ in sorted_freq[:top_count]]
        analysis['cold_numbers'] = [n for n, _ in sorted_freq[-top_count:]]

        # Overdue numbers (not appeared recently)
        recent_20 = draws[-20:]
        recent_numbers = set()
        for draw in recent_20:
            recent_numbers.update(draw)
        analysis['overdue_numbers'] = sorted([
            n for n in range(1, self.max_number + 1)
            if n not in recent_numbers
        ])

        # Common pairs
        pair_counter = Counter()
        for draw in draws:
            sorted_draw = sorted(draw)
            for i in range(len(sorted_draw)):
                for j in range(i + 1, len(sorted_draw)):
                    pair_counter[(sorted_draw[i], sorted_draw[j])] += 1

        analysis['pairs'] = [
            {'pair': list(pair), 'count': count}
            for pair, count in pair_counter.most_common(20)
        ]

        # Sum statistics
        sums = [sum(d) for d in draws]
        analysis['sum_range'] = {
            'mean': round(float(np.mean(sums)), 1),
            'std': round(float(np.std(sums)), 1),
            'min': int(min(sums)),
            'max': int(max(sums)),
            'median': round(float(np.median(sums)), 1)
        }

        # Odd/Even distribution
        odd_counts = [sum(1 for n in d if n % 2 == 1) for d in draws]
        odd_counter = Counter(odd_counts)
        total = len(draws)
        analysis['odd_even'] = {
            str(k): round(v / total * 100, 1) for k, v in sorted(odd_counter.items())
        }

        # Decade distribution
        decades = {}
        for n in range(1, self.max_number + 1):
            decade = f"{(n-1)//10*10+1}-{min((n-1)//10*10+10, self.max_number)}"
            if decade not in decades:
                decades[decade] = 0
            decades[decade] += freq.get(n, 0)
        analysis['decade_distribution'] = decades

        return analysis

    def predict(self, draws, n_numbers=6, n_predictions=5):
        """Statistical prediction based on patterns"""
        analysis = self.analyze(draws)
        predictions = []

        hot = analysis['hot_numbers']
        cold = analysis['cold_numbers']
        overdue = analysis['overdue_numbers']

        for i in range(n_predictions):
            selected = set()

            # Strategy mix based on prediction index
            if i == 0:
                # Strategy 1: Mostly hot numbers
                candidates = hot[:n_numbers * 2]
                selected = set(np.random.choice(candidates, size=min(n_numbers, len(candidates)), replace=False))
            elif i == 1:
                # Strategy 2: Mix hot + overdue
                hot_pick = min(n_numbers // 2 + 1, len(hot))
                overdue_pick = n_numbers - hot_pick
                selected.update(np.random.choice(hot[:15], size=hot_pick, replace=False))
                if overdue and overdue_pick > 0:
                    selected.update(np.random.choice(
                        overdue[:15] if len(overdue) >= 15 else overdue,
                        size=min(overdue_pick, len(overdue)),
                        replace=False
                    ))
            elif i == 2:
                # Strategy 3: Balanced odd/even + high/low
                odds = [n for n in range(1, self.max_number + 1) if n % 2 == 1]
                evens = [n for n in range(1, self.max_number + 1) if n % 2 == 0]
                mid = self.max_number // 2

                selected.update(np.random.choice(odds, size=3, replace=False))
                selected.update(np.random.choice(evens, size=3, replace=False))
            elif i == 3:
                # Strategy 4: Based on sum range
                target_sum = analysis['sum_range']['mean']
                # Try to find a combination close to target sum
                for _ in range(100):
                    trial = sorted(np.random.choice(range(1, self.max_number + 1), size=n_numbers, replace=False))
                    if abs(sum(trial) - target_sum) < analysis['sum_range']['std']:
                        selected = set(trial)
                        break
                if not selected:
                    selected = set(np.random.choice(range(1, self.max_number + 1), size=n_numbers, replace=False))
            else:
                # Strategy 5: Common pairs + random
                if analysis['pairs']:
                    pair = analysis['pairs'][np.random.randint(0, min(5, len(analysis['pairs'])))]['pair']
                    selected.update(pair)
                while len(selected) < n_numbers:
                    selected.add(int(np.random.randint(1, self.max_number + 1)))

            # Ensure we have exactly n_numbers
            selected = list(selected)[:n_numbers]
            while len(selected) < n_numbers:
                new_num = int(np.random.randint(1, self.max_number + 1))
                if new_num not in selected:
                    selected.append(new_num)

            predictions.append({
                'numbers': sorted([int(x) for x in selected]),
                'confidence': round(float(np.random.uniform(15, 45)), 1),
                'method': 'Statistical Analysis'
            })

        return predictions


# ============================================================
# 4. Ensemble Predictor (Combines all models)
# ============================================================
class EnsemblePredictor:
    def __init__(self, game_type='power655'):
        self.game_type = game_type

        if game_type == 'power655':
            self.max_number = 55
            self.pick_count = 6
        elif game_type == 'mega645':
            self.max_number = 45
            self.pick_count = 6
        else:
            self.max_number = 10  # Max 3D digits
            self.pick_count = 3

        self.lstm_predictor = LSTMPredictor(self.max_number)
        self.ml_predictor = MLPredictor(self.max_number)
        self.stat_predictor = StatisticalPredictor(self.max_number)

    def load_data(self):
        """Load draw data"""
        filepath = os.path.join(DATA_DIR, f'{self.game_type}.json')
        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r') as f:
            data = json.load(f)

        draws = []
        for item in data:
            if isinstance(item.get('numbers'), list):
                nums = [int(n) for n in item['numbers'] if isinstance(n, (int, float)) or (isinstance(n, str) and n.isdigit())]
                if nums:
                    draws.append(nums)

        return draws

    def train_all(self):
        """Train all models"""
        draws = self.load_data()
        if len(draws) < 30:
            print(f"Not enough data for {self.game_type}")
            return False

        print(f"\n{'='*50}")
        print(f"Training models for {self.game_type.upper()}")
        print(f"Total draws: {len(draws)}")
        print(f"{'='*50}")

        # Train LSTM
        print("\n[1/3] Training LSTM Neural Network...")
        try:
            self.lstm_predictor.train(draws, epochs=30)
            print("  ✓ LSTM training complete")
        except Exception as e:
            print(f"  ✗ LSTM training failed: {e}")

        # Train ML models
        print("\n[2/3] Training Random Forest + Gradient Boosting...")
        try:
            self.ml_predictor.train(draws)
            print("  ✓ ML training complete")
        except Exception as e:
            print(f"  ✗ ML training failed: {e}")

        # Statistical analysis
        print("\n[3/3] Running Statistical Analysis...")
        analysis = self.stat_predictor.analyze(draws)
        print(f"  ✓ Analysis complete")
        print(f"    Hot numbers: {analysis['hot_numbers'][:10]}")
        print(f"    Cold numbers: {analysis['cold_numbers'][:10]}")
        print(f"    Overdue: {analysis['overdue_numbers'][:10]}")

        # Save analysis
        analysis_path = os.path.join(DATA_DIR, f'{self.game_type}_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)

        return True

    def predict(self, n_predictions=5):
        """Generate ensemble predictions"""
        draws = self.load_data()
        if not draws:
            return {'predictions': [], 'analysis': {}}

        all_predictions = []

        # Get predictions from each model
        print(f"\nGenerating predictions for {self.game_type}...")

        lstm_preds = self.lstm_predictor.predict(draws, self.pick_count, 3)
        ml_preds = self.ml_predictor.predict(draws, self.pick_count, 3)
        stat_preds = self.stat_predictor.predict(draws, self.pick_count, 3)

        all_predictions.extend(lstm_preds)
        all_predictions.extend(ml_preds)
        all_predictions.extend(stat_preds)

        # Create ensemble predictions by voting
        ensemble_preds = self._ensemble_vote(
            lstm_preds, ml_preds, stat_preds, n_predictions
        )

        # Get analysis
        analysis = self.stat_predictor.analyze(draws)

        # Recent draws
        recent = draws[-10:]

        result = {
            'game_type': self.game_type,
            'total_draws': len(draws),
            'last_draw': draws[-1] if draws else [],
            'recent_draws': recent,
            'predictions': {
                'lstm': lstm_preds,
                'ml': ml_preds,
                'statistical': stat_preds,
                'ensemble': ensemble_preds,
            },
            'analysis': analysis,
            'generated_at': datetime.now().isoformat()
        }

        return result

    def _ensemble_vote(self, lstm_preds, ml_preds, stat_preds, n_predictions):
        """Combine predictions using weighted voting"""
        # Weight: LSTM 0.4, ML 0.35, Statistical 0.25
        number_scores = np.zeros(self.max_number)
        weights = {'lstm': 0.4, 'ml': 0.35, 'stat': 0.25}

        for preds, weight_key in [
            (lstm_preds, 'lstm'),
            (ml_preds, 'ml'),
            (stat_preds, 'stat')
        ]:
            for pred in preds:
                for num in pred['numbers']:
                    if 1 <= num <= self.max_number:
                        number_scores[num - 1] += weights[weight_key]

        # Normalize
        if number_scores.max() > 0:
            number_scores = number_scores / number_scores.max()

        predictions = []
        for i in range(n_predictions):
            noise = np.random.normal(0, 0.1 * (i + 1), self.max_number)
            adjusted = np.clip(number_scores + noise, 0, 1)

            top_indices = np.argsort(adjusted)[-self.pick_count * 2:]
            selected = np.random.choice(top_indices, size=self.pick_count, replace=False)
            prediction = sorted((selected + 1).tolist())

            avg_score = float(np.mean(number_scores[selected]))
            confidence = min(avg_score * 100, 60)  # Cap at 60%

            predictions.append({
                'numbers': prediction,
                'confidence': round(confidence, 1),
                'method': 'Ensemble (LSTM + RF + GB + Statistical)'
            })

        return predictions


# ============================================================
# Main training & prediction pipeline
# ============================================================
def train_and_predict_all():
    """Train all models and generate predictions for all game types"""
    results = {}

    for game_type in ['power655', 'mega645']:
        ensemble = EnsemblePredictor(game_type)
        success = ensemble.train_all()
        if success:
            results[game_type] = ensemble.predict()

    # Save predictions
    output_path = os.path.join(DATA_DIR, 'predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"All predictions saved to {output_path}")
    print(f"{'='*50}")

    return results


if __name__ == '__main__':
    train_and_predict_all()
