#!/usr/bin/env python3
"""
Build script: Generates static site with pre-computed predictions for all Vietlott games.
"""
import json, os, shutil, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
from scraper import collect_all_data, load_data, DATA_DIR, GAME_CONFIG
from ai_engine import train_and_predict_all
from ai_autoplay_engine import generate_all_autoplay

BUILD_DIR = os.path.join(os.path.dirname(__file__), 'dist')
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')


def build():
    print("=" * 60)
    print("  VIETLOTT AI PREDICTOR - STATIC BUILD")
    print("=" * 60)

    print("\n[1/5] Collecting data...")
    collect_all_data()

    print("\n[2/5] Training AI models & generating predictions...")
    train_and_predict_all()

    print("\n[3/5] Generating AI Auto-Play backtests...")
    generate_all_autoplay(n_plays=200)

    print("\n[4/5] Preparing static data...")
    os.makedirs(BUILD_DIR, exist_ok=True)
    data_out = os.path.join(BUILD_DIR, 'data')
    os.makedirs(data_out, exist_ok=True)

    # Copy predictions
    for fname in ['predictions.json', 'game_config.json']:
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(data_out, fname))

    # Per-game files (trim large datasets for static site performance)
    MAX_HISTORY = 1000
    for game_type in GAME_CONFIG:
        gd = load_data(f'{game_type}.json')
        if gd:
            history = list(reversed(gd))[:MAX_HISTORY]
            with open(os.path.join(data_out, f'{game_type}_history.json'), 'w') as f:
                json.dump(history, f, ensure_ascii=False)
        for suffix in ['_analysis.json', '_autoplay.json']:
            fpath = os.path.join(DATA_DIR, f'{game_type}{suffix}')
            if os.path.exists(fpath):
                shutil.copy2(fpath, os.path.join(data_out, f'{game_type}{suffix}'))

    # Copy frontend
    print("\n[5/5] Building static site...")
    for item in os.listdir(FRONTEND_DIR):
        src = os.path.join(FRONTEND_DIR, item)
        dst = os.path.join(BUILD_DIR, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

    open(os.path.join(BUILD_DIR, '.nojekyll'), 'w').close()

    total_files = sum(len(f) for _, _, f in os.walk(BUILD_DIR))
    print(f"\n{'=' * 60}")
    print(f"  BUILD COMPLETE! ({total_files} files in dist/)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    build()
