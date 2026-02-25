"""
Vietlott Data Scraper - Thu thập dữ liệu THẬT từ vietvudanh/vietlott-data
Repo: https://github.com/vietvudanh/vietlott-data (cập nhật hàng ngày)
Hỗ trợ: Power 6/55, Mega 6/45, Max 3D, Max 3D+, Keno
"""
import json
import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

REPO_URL = 'https://github.com/vietvudanh/vietlott-data.git'
REPO_DIR = '/tmp/vietlott-data'

# Mapping from repo filenames to our game types
GAME_FILES = {
    'power655':  'power655.jsonl',
    'mega645':   'power645.jsonl',
    'max3d':     '3d.jsonl',
    'max3dplus': '3d_pro.jsonl',
    'keno':      'keno.jsonl',
}

GAME_CONFIG = {
    'power655': {
        'name': 'Power 6/55',
        'max_number': 55,
        'pick_count': 6,
        'has_power': True,
        'digit_game': False,
    },
    'mega645': {
        'name': 'Mega 6/45',
        'max_number': 45,
        'pick_count': 6,
        'has_power': False,
        'digit_game': False,
    },
    'max3d': {
        'name': 'Max 3D',
        'max_number': 999,
        'pick_count': 3,
        'has_power': False,
        'digit_game': True,
    },
    'max3dplus': {
        'name': 'Max 3D+',
        'max_number': 999,
        'pick_count': 6,
        'has_power': False,
        'digit_game': True,
    },
    'keno': {
        'name': 'Keno',
        'max_number': 80,
        'pick_count': 20,
        'has_power': False,
        'digit_game': False,
    },
}


def clone_or_pull_repo():
    """Clone or update the vietlott-data repo"""
    if os.path.exists(os.path.join(REPO_DIR, '.git')):
        print("  Updating vietlott-data repo...")
        subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'],
                       capture_output=True, timeout=60)
    else:
        print("  Cloning vietlott-data repo...")
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR],
                       capture_output=True, timeout=120)


def parse_power655(filepath):
    """Parse Power 6/55 JSONL → our format
    Format: {"date":"2017-08-01","id":"00001","result":[5,10,14,23,24,38,35]}
    result = [6 main numbers, power_number]
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = item.get('result', [])
            if len(result) >= 7:
                main_numbers = sorted(result[:6])
                power = result[6]
            elif len(result) == 6:
                main_numbers = sorted(result)
                power = 0
            else:
                continue

            records.append({
                'draw_id': f"#{item.get('id', '?')}",
                'date': item.get('date', ''),
                'numbers': main_numbers,
                'power': power,
            })
    return records


def parse_mega645(filepath):
    """Parse Mega 6/45 JSONL → our format
    Format: {"date":"2017-10-25","id":"00198","result":[12,17,23,25,34,38]}
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = item.get('result', [])
            if len(result) < 6:
                continue
            records.append({
                'draw_id': f"#{item.get('id', '?')}",
                'date': item.get('date', ''),
                'numbers': sorted(result[:6]),
            })
    return records


def parse_max3d(filepath):
    """Parse Max 3D JSONL → our format
    Format: {"date":"...","id":"...","result":{"Giải Đặc biệt":["015","517"],...}}
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = item.get('result', {})
            if isinstance(result, dict):
                # Collect all prize numbers
                all_nums = []
                for prize_name in ['Giải Đặc biệt', 'Giải Nhất', 'Giải Nhì', 'Giải ba']:
                    nums = result.get(prize_name, [])
                    all_nums.extend(nums)
                # Take the special prize as main numbers
                special = result.get('Giải Đặc biệt', [])
                first = result.get('Giải Nhất', [])
                records.append({
                    'draw_id': f"#{item.get('id', '?')}",
                    'date': item.get('date', ''),
                    'numbers': special + first[:1],  # Top 3 prize numbers
                    'prize_1st': special[0] if special else '',
                    'prize_2nd': special[1] if len(special) > 1 else '',
                    'prize_3rd': first[0] if first else '',
                    'all_prizes': result,
                })
    return records


def parse_max3dplus(filepath):
    """Parse Max 3D+ JSONL → our format (same structure as Max 3D)"""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = item.get('result', {})
            if isinstance(result, dict):
                special = result.get('Giải Đặc biệt', [])
                first = result.get('Giải Nhất', [])
                second = result.get('Giải Nhì', [])
                # Max 3D+ has more numbers
                nums = special + first[:2] + second[:2]
                records.append({
                    'draw_id': f"#{item.get('id', '?')}",
                    'date': item.get('date', ''),
                    'numbers': nums[:6],
                    'prize_1st': special[0] if special else '',
                    'prize_2nd': special[1] if len(special) > 1 else '',
                    'prize_3rd': first[0] if first else '',
                    'all_prizes': result,
                })
    return records


def parse_keno(filepath):
    """Parse Keno JSONL → our format
    Format: {"date":"2022-12-04","id":"#0110271","result":[3,8,10,...20 numbers]}
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = item.get('result', [])
            if len(result) < 20:
                continue
            records.append({
                'draw_id': item.get('id', '?'),
                'date': item.get('date', ''),
                'numbers': sorted(result[:20]),
            })
    return records


PARSERS = {
    'power655': parse_power655,
    'mega645': parse_mega645,
    'max3d': parse_max3d,
    'max3dplus': parse_max3dplus,
    'keno': parse_keno,
}


def save_data(data, filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data):>6} records → {filename}")
    return filepath


def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def collect_all_data():
    """Main: clone repo + parse all games → save JSON"""
    print("=" * 60)
    print("VIETLOTT REAL DATA COLLECTOR")
    print("  Source: github.com/vietvudanh/vietlott-data")
    print("=" * 60)

    # Clone / pull
    clone_or_pull_repo()

    if not os.path.exists(os.path.join(REPO_DIR, 'data')):
        print("ERROR: Could not access vietlott-data repo")
        sys.exit(1)

    all_data = {}
    print("\nParsing real Vietlott data:")

    for game_type, jsonl_file in GAME_FILES.items():
        filepath = os.path.join(REPO_DIR, 'data', jsonl_file)
        if not os.path.exists(filepath):
            print(f"  ⚠ {jsonl_file} not found, skipping {game_type}")
            continue

        parser = PARSERS.get(game_type)
        if not parser:
            continue

        records = parser(filepath)
        # Sort by date
        records.sort(key=lambda x: x.get('date', ''))
        all_data[game_type] = records
        save_data(records, f'{game_type}.json')

    # Save game config
    config_for_frontend = {}
    for gt, cfg in GAME_CONFIG.items():
        config_for_frontend[gt] = {
            **cfg,
            'total_draws': len(all_data.get(gt, [])),
        }
    save_data(config_for_frontend, 'game_config.json')

    total = sum(len(v) for v in all_data.values())
    print(f"\n✓ Total: {total:,} REAL draws across {len(all_data)} game types")

    # Show date ranges
    for gt, records in all_data.items():
        if records:
            first_date = records[0].get('date', '?')
            last_date = records[-1].get('date', '?')
            print(f"  {gt:12s}: {len(records):>6} draws  ({first_date} → {last_date})")

    return all_data


if __name__ == '__main__':
    collect_all_data()
