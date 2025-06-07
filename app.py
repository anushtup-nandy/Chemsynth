# app.py
from flask import Flask, render_template, request, jsonify
from core_logic.literature_analysis import perform_literature_search
from core_logic.synthesis_planner import plan_synthesis_route

# Initialize Flask App
app = Flask(__name__)

# --- Primary Route to Serve the Frontend ---

@app.route('/')
def index():
    """
    Renders the main single-page application UI.
    """
    # The HTML file should be in the 'templates' directory
    return render_template('index.html')

# --- API Endpoints ---

@app.route('/api/literature_search', methods=['POST'])
def api_literature_search():
    """
    API endpoint for the Literature Analysis feature.
    Accepts search terms and returns aggregated results.
    """
    data = request.json
    search_query = data.get('query')
    max_results = data.get('max_results', 5)

    if not search_query:
        return jsonify({"error": "Search query is required."}), 400

    try:
        # This function will orchestrate calls to utils modules
        results = perform_literature_search(search_query, max_results)
        return jsonify(results)
    except Exception as e:
        # In a production environment, log the error properly
        print(f"An error occurred during literature search: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/api/plan_synthesis', methods=['POST'])
def api_plan_synthesis():
    """
    API endpoint for the Synthesis Design feature.
    Accepts a target molecule SMILES and returns planned routes.
    """
    data = request.json
    target_smiles = data.get('smiles')

    if not target_smiles:
        return jsonify({"error": "Target SMILES string is required."}), 400

    try:
        # This function orchestrates AiZynthFinder and LLM calls
        results = plan_synthesis_route(target_smiles)
        if "error" in results:
            return jsonify(results), 500 # Pass through specific errors
        return jsonify(results)
    except Exception as e:
        print(f"An unhandled error occurred during synthesis planning: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Main Execution ---

if __name__ == '__main__':
    # Flask will automatically use the settings from the .flaskenv file
    # for host and port during development.
    app.run(debug=True)