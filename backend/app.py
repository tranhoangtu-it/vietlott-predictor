"""
Vietlott Predictor - Flask API Server
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraper import collect_all_data, load_data, DATA_DIR
from ai_engine import EnsemblePredictor, train_and_predict_all

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

PREDICTIONS_CACHE = {}


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/status')
def status():
    """Check system status"""
    has_data = {
        'power655': os.path.exists(os.path.join(DATA_DIR, 'power655.json')),
        'mega645': os.path.exists(os.path.join(DATA_DIR, 'mega645.json')),
        'max3d': os.path.exists(os.path.join(DATA_DIR, 'max3d.json')),
    }
    has_predictions = os.path.exists(os.path.join(DATA_DIR, 'predictions.json'))

    return jsonify({
        'status': 'running',
        'data_available': has_data,
        'predictions_available': has_predictions,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/collect-data', methods=['POST'])
def api_collect_data():
    """Trigger data collection"""
    try:
        data = collect_all_data()
        return jsonify({
            'success': True,
            'counts': {k: len(v) for k, v in data.items()},
            'message': 'Data collected successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    """Train all AI models and generate predictions"""
    try:
        results = train_and_predict_all()
        return jsonify({
            'success': True,
            'games_trained': list(results.keys()),
            'message': 'Models trained and predictions generated'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predictions/<game_type>')
def get_predictions(game_type):
    """Get predictions for a specific game type"""
    # Try loading from cache or file
    pred_path = os.path.join(DATA_DIR, 'predictions.json')
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            all_predictions = json.load(f)

        if game_type in all_predictions:
            return jsonify(all_predictions[game_type])

    return jsonify({'error': f'No predictions for {game_type}'}), 404


@app.route('/api/predictions')
def get_all_predictions():
    """Get all predictions"""
    pred_path = os.path.join(DATA_DIR, 'predictions.json')
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({}), 404


@app.route('/api/history/<game_type>')
def get_history(game_type):
    """Get historical draw data"""
    data = load_data(f'{game_type}.json')
    if data:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))

        # Reverse to show newest first
        data_reversed = list(reversed(data))
        start = (page - 1) * per_page
        end = start + per_page

        return jsonify({
            'data': data_reversed[start:end],
            'total': len(data),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(data) + per_page - 1) // per_page
        })

    return jsonify({'error': 'No data available'}), 404


@app.route('/api/analysis/<game_type>')
def get_analysis(game_type):
    """Get statistical analysis for a game type"""
    analysis_path = os.path.join(DATA_DIR, f'{game_type}_analysis.json')
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            return jsonify(json.load(f))

    # Generate analysis on-the-fly
    data = load_data(f'{game_type}.json')
    if data:
        from ai_engine import StatisticalPredictor
        max_num = 55 if game_type == 'power655' else 45
        predictor = StatisticalPredictor(max_num)
        draws = [item['numbers'] for item in data if isinstance(item.get('numbers'), list)]
        analysis = predictor.analyze(draws)
        return jsonify(analysis)

    return jsonify({'error': 'No data available'}), 404


@app.route('/api/quick-predict/<game_type>', methods=['POST'])
def quick_predict(game_type):
    """Generate a quick new prediction without full retraining"""
    try:
        ensemble = EnsemblePredictor(game_type)
        draws = ensemble.load_data()

        if not draws:
            return jsonify({'error': 'No data available'}), 404

        # Use statistical predictor for quick prediction
        predictions = ensemble.stat_predictor.predict(draws, ensemble.pick_count, 5)

        return jsonify({
            'predictions': predictions,
            'game_type': game_type,
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  VIETLOTT AI PREDICTOR SERVER")
    print("=" * 60)

    # Auto-collect data if not exists
    if not os.path.exists(os.path.join(DATA_DIR, 'power655.json')):
        print("\nFirst run - collecting data...")
        collect_all_data()

    # Auto-train if predictions don't exist
    if not os.path.exists(os.path.join(DATA_DIR, 'predictions.json')):
        print("\nTraining AI models...")
        train_and_predict_all()

    print("\n🚀 Server starting at http://localhost:5000")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
