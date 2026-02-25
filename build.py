#!/usr/bin/env python3
"""
Build script: Generates static site with pre-computed predictions.
Used by GitHub Actions to build the site before deploying to GitHub Pages.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from scraper import collect_all_data, load_data, DATA_DIR
from ai_engine import EnsemblePredictor, StatisticalPredictor, train_and_predict_all

BUILD_DIR = os.path.join(os.path.dirname(__file__), 'dist')
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')


def build():
    print("=" * 60)
    print("  VIETLOTT AI PREDICTOR - STATIC BUILD")
    print("=" * 60)

    # Step 1: Collect data
    print("\n[1/4] Collecting data...")
    all_data = collect_all_data()

    # Step 2: Train models & generate predictions
    print("\n[2/4] Training AI models & generating predictions...")
    predictions = train_and_predict_all()

    # Step 3: Prepare static data files
    print("\n[3/4] Preparing static data...")
    os.makedirs(BUILD_DIR, exist_ok=True)
    data_out = os.path.join(BUILD_DIR, 'data')
    os.makedirs(data_out, exist_ok=True)

    # Copy predictions
    pred_path = os.path.join(DATA_DIR, 'predictions.json')
    if os.path.exists(pred_path):
        shutil.copy2(pred_path, os.path.join(data_out, 'predictions.json'))

    # Generate per-game data files
    for game_type in ['power655', 'mega645', 'max3d']:
        game_data = load_data(f'{game_type}.json')
        if game_data:
            # Full history (reversed, newest first)
            history = list(reversed(game_data))
            with open(os.path.join(data_out, f'{game_type}_history.json'), 'w') as f:
                json.dump(history, f, ensure_ascii=False)

            # Analysis
            analysis_path = os.path.join(DATA_DIR, f'{game_type}_analysis.json')
            if os.path.exists(analysis_path):
                shutil.copy2(analysis_path, os.path.join(data_out, f'{game_type}_analysis.json'))

    # Step 4: Copy frontend files
    print("\n[4/4] Building static site...")
    for item in os.listdir(FRONTEND_DIR):
        src = os.path.join(FRONTEND_DIR, item)
        dst = os.path.join(BUILD_DIR, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # Create .nojekyll for GitHub Pages
    with open(os.path.join(BUILD_DIR, '.nojekyll'), 'w') as f:
        pass

    print(f"\n{'=' * 60}")
    print(f"  BUILD COMPLETE!")
    print(f"  Output: {BUILD_DIR}")
    print(f"  Files: {len(os.listdir(BUILD_DIR))}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    build()
