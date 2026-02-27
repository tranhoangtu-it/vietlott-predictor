"""
AI Autoplay Engine - Backtests AI predictions against real Vietlott lottery data.
Reuses StatisticalPredictor / DigitPredictor from ai_engine and data helpers from scraper.
"""
import os
import sys

# Allow sibling-module imports when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import load_data, DATA_DIR, GAME_CONFIG, save_data
from ai_engine import StatisticalPredictor, DigitPredictor

# ---------------------------------------------------------------------------
# Prize table (VND) keyed by match count
# ---------------------------------------------------------------------------
PRIZE_TABLE = {
    'power655': {6: 30_000_000_000, 5: 40_000_000, 4: 500_000, 3: 50_000},
    'mega645':  {6: 12_000_000_000, 5: 10_000_000, 4: 300_000, 3: 30_000},
    'keno':     {10: 2_000_000_000, 9: 500_000_000, 8: 50_000_000,
                 7: 5_000_000, 6: 500_000, 5: 50_000, 4: 10_000},
    'max3d':    {3: 1_000_000_000, 2: 10_000_000, 1: 100_000},
    'max3dplus':{3: 1_000_000_000, 2: 10_000_000, 1: 100_000},
    'bingo18':  {3: 2_000_000_000, 2: 500_000, 1: 20_000},
    'power535': {6: 30_000_000_000, 5: 40_000_000, 4: 500_000, 3: 50_000},
}

TICKET_PRICE = 10_000  # VND per play


def _lookup_prize(game_type: str, matches: int) -> int:
    """Return prize amount for given match count; 0 if no prize."""
    table = PRIZE_TABLE.get(game_type, {})
    return table.get(matches, 0)


def _count_matches_digit(predicted: list, actual: list) -> int:
    """Exact string match count for digit games."""
    actual_set = set(actual)
    return sum(1 for n in predicted if n in actual_set)


def _count_matches_regular(predicted: list, actual: list) -> int:
    """Intersection count for regular number games."""
    return len(set(predicted) & set(actual))


def generate_autoplay(game_type: str, n_plays: int = 200) -> dict | None:
    """
    Backtest AI predictions for a single game type against historical draws.

    Returns a result dict (suitable for JSON serialisation) or None when
    there are insufficient draws.
    """
    draws = load_data(f"{game_type}.json")
    if not draws or len(draws) < 50:
        print(f"  [{game_type}] insufficient data ({len(draws) if draws else 0} draws, need 50)")
        return None

    cfg = GAME_CONFIG[game_type]
    is_digit = cfg.get('digit_game', False)
    max_num = cfg['max_number']
    # Keno: player picks 10 numbers; house draws 20
    pick_count = 10 if game_type == 'keno' else cfg['pick_count']
    strategy_label = 'Digit Pattern AI' if is_digit else 'Statistical AI (Hot/Cold/Overdue)'

    start_idx = max(30, len(draws) - n_plays)
    plays = []
    total_won = 0
    cumulative_profit = 0
    best_match = 0
    wins = 0

    for i in range(start_idx, len(draws)):
        history = draws[:i]
        actual = draws[i]
        actual_nums = actual.get('numbers', [])

        # Build history of numbers only (list of lists/strings)
        history_nums = [d['numbers'] for d in history]

        # Generate one prediction using historical data
        try:
            if is_digit:
                pred = DigitPredictor(max_num, pick_count).predict(history_nums, n_predictions=1)[0]
            else:
                pred = StatisticalPredictor(max_num).predict(history_nums, pick_count, n_predictions=1)[0]
        except Exception as exc:
            print(f"  [{game_type}] prediction error at draw {i}: {exc}")
            pred = {'numbers': []}

        ai_numbers = pred.get('numbers', [])

        # Count matches
        if is_digit:
            matches = _count_matches_digit(ai_numbers, actual_nums)
        else:
            matches = _count_matches_regular(ai_numbers, actual_nums)

        prize = _lookup_prize(game_type, matches)
        cost = TICKET_PRICE

        total_won += prize
        cumulative_profit += prize - cost
        if matches > best_match:
            best_match = matches
        if prize > 0:
            wins += 1

        plays.append({
            'draw_id': actual.get('draw_id', ''),
            'date': actual.get('date', ''),
            'ai_numbers': ai_numbers,
            'actual_numbers': actual_nums,
            'actual_power': actual.get('power', None),
            'matches': matches,
            'prize': prize,
            'cost': cost,
            'cumulative_profit': cumulative_profit,
        })

    total_plays = len(plays)
    total_spent = total_plays * TICKET_PRICE
    profit = total_won - total_spent
    roi = round(profit / total_spent * 100, 2) if total_spent else 0.0
    win_rate = round(wins / total_plays * 100, 2) if total_plays else 0.0

    return {
        'game_type': game_type,
        'strategy': strategy_label,
        'ticket_price': TICKET_PRICE,
        'total_plays': total_plays,
        'total_spent': total_spent,
        'total_won': total_won,
        'profit': profit,
        'roi': roi,
        'win_rate': win_rate,
        'best_match': best_match,
        'plays': plays,
    }


def generate_all_autoplay(n_plays: int = 200) -> dict:
    """
    Run backtesting for every configured game type, save results to JSON,
    and return a summary dict keyed by game_type.
    """
    results = {}
    for game_type in GAME_CONFIG:
        print(f"Running autoplay backtest: {game_type} ...")
        result = generate_autoplay(game_type, n_plays=n_plays)
        if result is None:
            print(f"  [{game_type}] skipped.")
            continue
        save_data(result, f"{game_type}_autoplay.json")
        # Store summary without per-play list to keep memory lean
        results[game_type] = {k: v for k, v in result.items() if k != 'plays'}
        print(
            f"  [{game_type}] done: {result['total_plays']} plays, "
            f"ROI={result['roi']}%, win_rate={result['win_rate']}%"
        )
    return results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Vietlott AI Autoplay Backtester')
    parser.add_argument('--game', default='all', help='Game type or "all"')
    parser.add_argument('--plays', type=int, default=200, help='Number of plays to backtest')
    args = parser.parse_args()

    if args.game == 'all':
        summary = generate_all_autoplay(n_plays=args.plays)
        print("\nSummary:")
        for gt, r in summary.items():
            print(f"  {gt}: ROI={r['roi']}%, win_rate={r['win_rate']}%, best_match={r['best_match']}")
    else:
        res = generate_autoplay(args.game, n_plays=args.plays)
        if res:
            print(f"Result: {res['total_plays']} plays, ROI={res['roi']}%, win_rate={res['win_rate']}%")
