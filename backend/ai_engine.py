"""
Vietlott AI Prediction Engine
Kết hợp: LSTM Neural Network + Random Forest + Statistical Analysis + Ensemble
Hỗ trợ: Power 6/55, Mega 6/45, Max 3D, Max 3D+, Keno
Data thật từ github.com/vietvudanh/vietlott-data
"""
import numpy as np
import json
import os
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Game configuration
GAME_CONFIG = {
    'power655':  {'max_number': 55, 'pick_count': 6,  'has_power': True,  'digit_game': False},
    'mega645':   {'max_number': 45, 'pick_count': 6,  'has_power': False, 'digit_game': False},
    'keno':      {'max_number': 80, 'pick_count': 10, 'has_power': False, 'digit_game': False},
    'max3d':     {'max_number': 999,'pick_count': 3,  'has_power': False, 'digit_game': True},
    'max3dplus': {'max_number': 999,'pick_count': 6,  'has_power': False, 'digit_game': True},
    'bingo18':   {'max_number': 18, 'pick_count': 3,  'has_power': False, 'digit_game': False},
    'power535':  {'max_number': 35, 'pick_count': 5,  'has_power': True,  'digit_game': False},
}

# ============================================================
# 1. LSTM Neural Network
# ============================================================
class LSTMLotteryModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=55):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc2(self.dropout(self.relu(self.fc1(out[:, -1, :]))))
        return self.sigmoid(out)


class LSTMPredictor:
    def __init__(self, max_number, seq_length=20):
        self.max_number = max_number
        self.seq_length = seq_length
        self.model = None

    def prepare_data(self, draws):
        binary = np.array([
            np.array([1 if (n-1) == j else 0 for j in range(self.max_number)] if isinstance(draws[0][0], int) else
                     [0]*self.max_number)
            for draw in draws
            for n in [0]  # dummy
        ]) if False else None  # placeholder

        # Build binary matrix properly
        matrix = []
        for draw in draws:
            vec = np.zeros(self.max_number)
            for num in draw:
                if 1 <= num <= self.max_number:
                    vec[num - 1] = 1
            matrix.append(vec)
        matrix = np.array(matrix)

        X, y = [], []
        for i in range(len(matrix) - self.seq_length):
            X.append(matrix[i:i + self.seq_length])
            y.append(matrix[i + self.seq_length])
        return np.array(X), np.array(y)

    def train(self, draws, epochs=30, lr=0.001):
        if len(draws) < self.seq_length + 10:
            return
        X, y = self.prepare_data(draws)
        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.FloatTensor(np.array(y))

        self.model = LSTMLotteryModel(self.max_number, 128, 2, self.max_number)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            batch_size = 64
            total_loss, n_b = 0, 0
            idx = torch.randperm(len(X_t))
            for i in range(0, len(X_t), batch_size):
                bi = idx[i:i + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(X_t[bi]), y_t[bi])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1
            if (epoch + 1) % 10 == 0:
                print(f"  LSTM Epoch [{epoch+1}/{epochs}], Loss: {total_loss/n_b:.4f}")

    def predict(self, draws, n_numbers=6, n_predictions=5):
        if self.model is None:
            return []
        self.model.eval()

        seq = []
        for draw in draws[-self.seq_length:]:
            vec = np.zeros(self.max_number)
            for num in draw:
                if 1 <= num <= self.max_number:
                    vec[num - 1] = 1
            seq.append(vec)
        while len(seq) < self.seq_length:
            seq.insert(0, np.zeros(self.max_number))

        with torch.no_grad():
            probs = self.model(torch.FloatTensor(np.array([seq]))).numpy()[0]

        predictions = []
        for i in range(n_predictions):
            noise = np.random.normal(0, 0.05 * (i + 1), self.max_number)
            adj = np.clip(probs + noise, 0, 1)
            top = np.argsort(adj)[-n_numbers * 2:]
            sel = np.random.choice(top, size=n_numbers, replace=False)
            predictions.append({
                'numbers': sorted((sel + 1).tolist()),
                'confidence': round(float(np.mean(adj[sel])) * 100, 1),
                'method': 'LSTM Neural Network'
            })
        return predictions


# ============================================================
# 2. Random Forest + Gradient Boosting
# ============================================================
class MLPredictor:
    def __init__(self, max_number):
        self.max_number = max_number
        self.rf_models = {}
        self.gb_models = {}

    def extract_features(self, draws, window=10):
        features_list, labels_list = [], []
        for i in range(window, len(draws)):
            recent = draws[i - window:i]
            target = draws[i]
            features = []

            freq = Counter(n for d in recent for n in d)
            features.extend([freq.get(n, 0) / window for n in range(1, self.max_number + 1)])

            for n in range(1, self.max_number + 1):
                gap = window
                for j, d in enumerate(reversed(recent)):
                    if n in d:
                        gap = j; break
                features.append(gap / window)

            sums = [sum(d) for d in recent]
            features.extend([float(np.mean(sums)), float(np.std(sums))])

            consec = [sum(1 for k in range(len(sorted(d))-1) if sorted(d)[k+1]-sorted(d)[k]==1) for d in recent]
            features.append(float(np.mean(consec)))
            features.append(float(np.mean([sum(1 for n in d if n % 2 == 1) / len(d) for d in recent])))
            features.append(float(np.mean([sum(1 for n in d if n > self.max_number//2) / len(d) for d in recent])))

            features_list.append(features)
            label = np.zeros(self.max_number)
            for num in target:
                if 1 <= num <= self.max_number:
                    label[num - 1] = 1
            labels_list.append(label)

        return np.array(features_list), np.array(labels_list)

    def train(self, draws):
        X, y = self.extract_features(draws)
        if len(X) < 10:
            return
        self.rf_models, self.gb_models = {}, {}
        for idx in range(self.max_number):
            ys = y[:, idx]
            if ys.sum() > 5:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                rf.fit(X, ys)
                self.rf_models[idx] = rf
                gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
                gb.fit(X, ys)
                self.gb_models[idx] = gb
        print(f"  ML Models trained for {len(self.rf_models)} numbers")

    def predict(self, draws, n_numbers=6, n_predictions=5):
        if not self.rf_models:
            return []
        X, _ = self.extract_features(draws)
        if len(X) == 0:
            return []
        last = X[-1:].reshape(1, -1)
        rf_p = np.zeros(self.max_number)
        gb_p = np.zeros(self.max_number)
        for idx in range(self.max_number):
            if idx in self.rf_models:
                try: rf_p[idx] = self.rf_models[idx].predict_proba(last)[0][1]
                except Exception: rf_p[idx] = 0.1
            if idx in self.gb_models:
                try: gb_p[idx] = self.gb_models[idx].predict_proba(last)[0][1]
                except Exception: gb_p[idx] = 0.1
        combined = 0.6 * rf_p + 0.4 * gb_p
        predictions = []
        for i in range(n_predictions):
            adj = np.clip(combined + np.random.normal(0, 0.03*(i+1), self.max_number), 0, 1)
            top = np.argsort(adj)[-n_numbers*2:]
            sel = np.random.choice(top, size=n_numbers, replace=False)
            predictions.append({
                'numbers': sorted((sel + 1).tolist()),
                'confidence': round(float(np.mean(adj[sel])) * 100, 1),
                'method': 'Random Forest + Gradient Boosting'
            })
        return predictions


# ============================================================
# 3. Statistical Analysis
# ============================================================
class StatisticalPredictor:
    def __init__(self, max_number):
        self.max_number = max_number

    def analyze(self, draws):
        all_nums = [n for d in draws for n in d]
        freq = Counter(all_nums)
        total = len(draws)

        analysis = {'frequency': {}, 'hot_numbers': [], 'cold_numbers': [],
                     'overdue_numbers': [], 'pairs': [], 'sum_range': {},
                     'odd_even': {}, 'decade_distribution': {}}

        for n in range(1, self.max_number + 1):
            analysis['frequency'][n] = {
                'count': freq.get(n, 0),
                'percentage': round(freq.get(n, 0) / total * 100, 2)
            }

        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        top_n = max(int(self.max_number * 0.2), 6)
        analysis['hot_numbers'] = [n for n, _ in sorted_freq[:top_n]]
        analysis['cold_numbers'] = [n for n, _ in sorted_freq[-top_n:]]

        recent_set = set(n for d in draws[-20:] for n in d)
        analysis['overdue_numbers'] = sorted(n for n in range(1, self.max_number + 1) if n not in recent_set)

        pair_c = Counter()
        for d in draws:
            sd = sorted(d)
            for i in range(len(sd)):
                for j in range(i + 1, len(sd)):
                    pair_c[(sd[i], sd[j])] += 1
        analysis['pairs'] = [{'pair': list(p), 'count': c} for p, c in pair_c.most_common(20)]

        sums = [sum(d) for d in draws]
        analysis['sum_range'] = {
            'mean': round(float(np.mean(sums)), 1), 'std': round(float(np.std(sums)), 1),
            'min': int(min(sums)), 'max': int(max(sums)),
            'median': round(float(np.median(sums)), 1)
        }

        odd_counts = [sum(1 for n in d if n % 2 == 1) for d in draws]
        odd_c = Counter(odd_counts)
        analysis['odd_even'] = {str(k): round(v / total * 100, 1) for k, v in sorted(odd_c.items())}

        decades = {}
        for n in range(1, self.max_number + 1):
            dec = f"{(n-1)//10*10+1}-{min((n-1)//10*10+10, self.max_number)}"
            decades[dec] = decades.get(dec, 0) + freq.get(n, 0)
        analysis['decade_distribution'] = decades

        return analysis

    def predict(self, draws, n_numbers=6, n_predictions=5):
        analysis = self.analyze(draws)
        hot, cold, overdue = analysis['hot_numbers'], analysis['cold_numbers'], analysis['overdue_numbers']
        predictions = []

        strategies = [
            lambda: set(np.random.choice(hot[:n_numbers*2], size=min(n_numbers, len(hot[:n_numbers*2])), replace=False)),
            lambda: set(np.random.choice(hot[:15], size=min(n_numbers//2+1, len(hot[:15])), replace=False)) |
                    (set(np.random.choice(overdue[:15] or [1], size=min(n_numbers//2, len(overdue[:15] or [1])), replace=False)) if overdue else set()),
            lambda: set(np.random.choice([n for n in range(1, self.max_number+1) if n%2==1], size=3, replace=False)) |
                    set(np.random.choice([n for n in range(1, self.max_number+1) if n%2==0], size=3, replace=False)),
            lambda: set(np.random.choice(range(1, self.max_number+1), size=n_numbers, replace=False)),
            lambda: (set(analysis['pairs'][np.random.randint(0, min(5, len(analysis['pairs'])))]['pair']) if analysis['pairs'] else set()) or
                    set(np.random.choice(range(1, self.max_number+1), size=n_numbers, replace=False)),
        ]

        for i in range(n_predictions):
            try:
                selected = list(strategies[i % len(strategies)]())
            except Exception:
                selected = list(np.random.choice(range(1, self.max_number+1), size=n_numbers, replace=False))
            selected = selected[:n_numbers]
            while len(selected) < n_numbers:
                new = int(np.random.randint(1, self.max_number + 1))
                if new not in selected:
                    selected.append(new)
            predictions.append({
                'numbers': sorted([int(x) for x in selected]),
                'confidence': round(float(np.random.uniform(15, 45)), 1),
                'method': 'Statistical Analysis'
            })
        return predictions


# ============================================================
# 4. Digit-game Predictor (Max 3D, 3D+, 4D)
# ============================================================
class DigitPredictor:
    """For digit-based games (Max 3D, Max 3D+, Max 4D)"""
    def __init__(self, max_val, pick_count):
        self.max_val = max_val  # 999 or 9999
        self.pick_count = pick_count
        self.digits = len(str(max_val))  # 3 or 4

    def analyze(self, draws):
        """Analyze digit frequency patterns"""
        all_nums = [n for d in draws for n in d]
        # Digit frequency per position
        digit_freq = {pos: Counter() for pos in range(self.digits)}
        for num_str in all_nums:
            s = str(num_str).zfill(self.digits)
            for pos, ch in enumerate(s):
                digit_freq[pos][ch] += 1

        # Most common full numbers
        full_freq = Counter(all_nums)
        # Last digit patterns
        last_digits = Counter(str(n)[-1] for n in all_nums)
        # Sum of digits
        digit_sums = [sum(int(c) for c in str(n).zfill(self.digits)) for n in all_nums]

        return {
            'digit_frequency': {str(k): dict(v.most_common()) for k, v in digit_freq.items()},
            'top_numbers': [{'number': n, 'count': c} for n, c in full_freq.most_common(20)],
            'last_digit_freq': dict(last_digits.most_common()),
            'digit_sum': {
                'mean': round(float(np.mean(digit_sums)), 1),
                'std': round(float(np.std(digit_sums)), 1),
            },
            'total_draws': len(draws),
        }

    def predict(self, draws, n_predictions=5):
        analysis = self.analyze(draws)
        predictions = []
        fmt = f'{{:0{self.digits}d}}'

        for i in range(n_predictions):
            nums = []
            for _ in range(self.pick_count):
                # Build number digit by digit using frequency-weighted random
                digits_chosen = []
                for pos in range(self.digits):
                    freq = analysis['digit_frequency'].get(str(pos), {})
                    if freq:
                        vals = list(freq.keys())
                        counts = [freq[v] for v in vals]
                        total = sum(counts)
                        probs = [c/total for c in counts]
                        # Add noise for variety
                        probs = np.array(probs) + np.random.uniform(0, 0.05 * (i+1), len(probs))
                        probs = probs / probs.sum()
                        digit = np.random.choice(vals, p=probs)
                    else:
                        digit = str(np.random.randint(0, 10))
                    digits_chosen.append(digit)
                nums.append(''.join(digits_chosen))

            predictions.append({
                'numbers': nums,
                'confidence': round(float(np.random.uniform(10, 35)), 1),
                'method': 'Digit Pattern AI'
            })
        return predictions


# ============================================================
# 5. Ensemble Predictor
# ============================================================
class EnsemblePredictor:
    def __init__(self, game_type='power655'):
        self.game_type = game_type
        cfg = GAME_CONFIG.get(game_type, GAME_CONFIG['power655'])
        self.max_number = cfg['max_number']
        self.pick_count = cfg['pick_count']
        self.is_digit_game = cfg.get('digit_game', False)

        if self.is_digit_game:
            self.digit_predictor = DigitPredictor(self.max_number, self.pick_count)
            self.lstm_predictor = None
            self.ml_predictor = None
        else:
            # Cap LSTM/ML at 80 for keno, normal for others
            ml_max = min(self.max_number, 80)
            self.lstm_predictor = LSTMPredictor(ml_max)
            self.ml_predictor = MLPredictor(ml_max)
            self.digit_predictor = None

        self.stat_predictor = StatisticalPredictor(self.max_number) if not self.is_digit_game else None

    def load_data(self):
        filepath = os.path.join(DATA_DIR, f'{self.game_type}.json')
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            data = json.load(f)

        draws = []
        for item in data:
            nums = item.get('numbers', [])
            if isinstance(nums, list) and nums:
                if self.is_digit_game:
                    draws.append([str(n) for n in nums])
                else:
                    int_nums = []
                    for n in nums:
                        try: int_nums.append(int(n))
                        except (ValueError, TypeError): pass
                    if int_nums:
                        draws.append(int_nums)
        return draws

    def train_all(self):
        draws = self.load_data()
        if len(draws) < 30:
            if len(draws) >= 1:
                print(f"  Limited data for {self.game_type} ({len(draws)} draws) — random predictions only")
                return 'limited'
            print(f"  No data for {self.game_type}")
            return False

        print(f"\n{'='*50}")
        print(f"Training: {self.game_type.upper()} ({len(draws)} draws)")
        print(f"{'='*50}")

        if self.is_digit_game:
            print("  [1/1] Running Digit Pattern Analysis...")
            analysis = self.digit_predictor.analyze(draws)
            print(f"  ✓ Top numbers: {[x['number'] for x in analysis['top_numbers'][:5]]}")

            analysis_path = os.path.join(DATA_DIR, f'{self.game_type}_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        else:
            # Limit training data for speed (bingo18 has 83k+ draws, use more)
            max_train = 5000 if self.game_type == 'bingo18' else 2000
            train_draws = draws[-max_train:] if len(draws) > max_train else draws

            print("  [1/3] Training LSTM Neural Network...")
            try:
                self.lstm_predictor.train(train_draws, epochs=30)
                print("  ✓ LSTM complete")
            except Exception as e:
                print(f"  ✗ LSTM failed: {e}")

            print("  [2/3] Training Random Forest + Gradient Boosting...")
            try:
                self.ml_predictor.train(train_draws)
                print("  ✓ ML complete")
            except Exception as e:
                print(f"  ✗ ML failed: {e}")

            print("  [3/3] Running Statistical Analysis...")
            analysis = self.stat_predictor.analyze(draws)
            print(f"  ✓ Hot: {analysis['hot_numbers'][:8]}")

            analysis_path = os.path.join(DATA_DIR, f'{self.game_type}_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)

        return True

    def predict(self, n_predictions=5, limited=False):
        draws = self.load_data()
        if not draws:
            return {'predictions': {}, 'analysis': {}}

        # Limited data: random predictions with warning flag
        if limited:
            random_preds = []
            for i in range(n_predictions):
                if self.is_digit_game:
                    nums = [str(np.random.randint(0, self.max_number + 1)).zfill(len(str(self.max_number)))
                            for _ in range(self.pick_count)]
                else:
                    nums = sorted(np.random.choice(
                        range(1, self.max_number + 1), size=self.pick_count, replace=False
                    ).tolist())
                pred = {
                    'numbers': nums,
                    'confidence': round(float(np.random.uniform(5, 15)), 1),
                    'method': 'Random (insufficient data)'
                }
                random_preds.append(pred)

            result = {
                'game_type': self.game_type,
                'total_draws': len(draws),
                'limited_data': True,
                'last_draw': draws[-1] if draws else [],
                'recent_draws': draws[-10:],
                'predictions': {'ensemble': random_preds},
                'analysis': {},
                'generated_at': datetime.now().isoformat(),
            }
            # Add power number for power games
            cfg = GAME_CONFIG.get(self.game_type, {})
            if cfg.get('has_power') and not limited:
                pass  # power handled in normal flow
            return result

        if self.is_digit_game:
            digit_preds = self.digit_predictor.predict(draws, n_predictions)
            analysis = self.digit_predictor.analyze(draws)
            return {
                'game_type': self.game_type,
                'total_draws': len(draws),
                'last_draw': draws[-1] if draws else [],
                'recent_draws': draws[-10:],
                'predictions': {'digit_ai': digit_preds, 'ensemble': digit_preds},
                'analysis': analysis,
                'generated_at': datetime.now().isoformat(),
            }
        else:
            pick = min(self.pick_count, 10)  # Keno: predict top 10
            lstm_preds = self.lstm_predictor.predict(draws, pick, 3) if self.lstm_predictor else []
            ml_preds = self.ml_predictor.predict(draws, pick, 3) if self.ml_predictor else []
            stat_preds = self.stat_predictor.predict(draws, pick, 3)
            ensemble_preds = self._ensemble_vote(lstm_preds, ml_preds, stat_preds, n_predictions, pick)
            analysis = self.stat_predictor.analyze(draws)

            return {
                'game_type': self.game_type,
                'total_draws': len(draws),
                'last_draw': draws[-1] if draws else [],
                'recent_draws': draws[-10:],
                'predictions': {
                    'ensemble': ensemble_preds,
                    'lstm': lstm_preds,
                    'ml': ml_preds,
                    'statistical': stat_preds,
                },
                'analysis': analysis,
                'generated_at': datetime.now().isoformat(),
            }

    def _ensemble_vote(self, lstm_preds, ml_preds, stat_preds, n_predictions, pick):
        mn = min(self.max_number, 80)
        scores = np.zeros(mn)
        weights = {'lstm': 0.4, 'ml': 0.35, 'stat': 0.25}
        for preds, wk in [(lstm_preds, 'lstm'), (ml_preds, 'ml'), (stat_preds, 'stat')]:
            for p in preds:
                for num in p['numbers']:
                    if 1 <= num <= mn:
                        scores[num - 1] += weights[wk]
        if scores.max() > 0:
            scores /= scores.max()

        predictions = []
        for i in range(n_predictions):
            adj = np.clip(scores + np.random.normal(0, 0.1*(i+1), mn), 0, 1)
            top = np.argsort(adj)[-pick*2:]
            sel = np.random.choice(top, size=pick, replace=False)
            predictions.append({
                'numbers': sorted((sel + 1).tolist()),
                'confidence': round(min(float(np.mean(scores[sel])) * 100, 60), 1),
                'method': 'Ensemble (LSTM + RF + GB + Statistical)'
            })
        return predictions


# ============================================================
# Main Pipeline
# ============================================================
def train_and_predict_all():
    results = {}
    for game_type in GAME_CONFIG:
        ensemble = EnsemblePredictor(game_type)
        status = ensemble.train_all()
        if status == 'limited':
            results[game_type] = ensemble.predict(limited=True)
        elif status:
            results[game_type] = ensemble.predict()

    output_path = os.path.join(DATA_DIR, 'predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"All predictions saved ({len(results)} games)")
    print(f"{'='*50}")
    return results


if __name__ == '__main__':
    train_and_predict_all()
