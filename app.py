# app.py
from flask import Flask, render_template, request, jsonify
from core_logic.literature_analysis import perform_literature_search
# MODIFIED imports
from core_logic.synthesis_planner import plan_synthesis_route, generate_hypothetical_route

# Initialize Flask App
app = Flask(__name__)

# --- Primary Route to Serve the Frontend ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoints ---
@app.route('/api/literature_search', methods=['POST'])
def api_literature_search():
    # ... (no changes here)
    data = request.json
    search_query = data.get('query')
    max_results = data.get('max_results', 5)
    if not search_query:
        return jsonify({"error": "Search query is required."}), 400
    try:
        results = perform_literature_search(search_query, max_results)
        return jsonify(results)
    except Exception as e:
        print(f"An error occurred during literature search: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/api/plan_synthesis', methods=['POST'])
def api_plan_synthesis():
    """
    API endpoint for Synthesis Design. Accepts a molecule identifier (SMILES, Name, CAS).
    """
    data = request.json
    # RENAMED for clarity
    target_identifier = data.get('identifier') 

    if not target_identifier:
        return jsonify({"error": "Target molecule identifier is required."}), 400

    try:
        # MODIFIED: Pass the raw identifier to the backend function
        results = plan_synthesis_route(target_identifier)
        if "error" in results:
            # Pass through specific errors (e.g., "Could not resolve identifier")
            return jsonify(results), 400 
        return jsonify(results)
    except Exception as e:
        print(f"An unhandled error occurred during synthesis planning: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- NEW API ENDPOINT ---
@app.route('/api/generate_new_route', methods=['POST'])
def api_generate_new_route():
    """
    API endpoint to generate a new hypothetical route based on user suggestion.
    """
    data = request.json
    suggestion = data.get('suggestion')
    existing_routes = data.get('existing_routes', [])
    target_smiles = data.get('target_smiles')

    if not suggestion or not target_smiles:
        return jsonify({"error": "A suggestion and target SMILES are required."}), 400

    try:
        results = generate_hypothetical_route(suggestion, existing_routes, target_smiles)
        if results and "new_route" in results:
            return jsonify(results)
        else:
            return jsonify({"error": "The AI could not generate a valid route from the suggestion."}), 500
    except Exception as e:
        print(f"An error occurred during hypothetical route generation: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)

# # app.py
# from flask import Flask, render_template, request, jsonify
# from core_logic.literature_analysis import perform_literature_search
# from core_logic.synthesis_planner import plan_synthesis_route

# # Initialize Flask App
# app = Flask(__name__)

# # --- Primary Route to Serve the Frontend ---

# @app.route('/')
# def index():
#     """
#     Renders the main single-page application UI.
#     """
#     # The HTML file should be in the 'templates' directory
#     return render_template('index.html')

# # --- API Endpoints ---

# @app.route('/api/literature_search', methods=['POST'])
# def api_literature_search():
#     """
#     API endpoint for the Literature Analysis feature.
#     Accepts search terms and returns aggregated results.
#     """
#     data = request.json
#     search_query = data.get('query')
#     max_results = data.get('max_results', 5)

#     if not search_query:
#         return jsonify({"error": "Search query is required."}), 400

#     try:
#         # This function will orchestrate calls to utils modules
#         results = perform_literature_search(search_query, max_results)
#         return jsonify(results)
#     except Exception as e:
#         # In a production environment, log the error properly
#         print(f"An error occurred during literature search: {e}")
#         return jsonify({"error": "An internal error occurred."}), 500

# @app.route('/api/plan_synthesis', methods=['POST'])
# def api_plan_synthesis():
#     """
#     API endpoint for the Synthesis Design feature.
#     Accepts a target molecule SMILES and returns planned routes.
#     """
#     data = request.json
#     target_smiles = data.get('smiles')

#     if not target_smiles:
#         return jsonify({"error": "Target SMILES string is required."}), 400

#     try:
#         # This function orchestrates AiZynthFinder and LLM calls
#         results = plan_synthesis_route(target_smiles)
#         if "error" in results:
#             return jsonify(results), 500 # Pass through specific errors
#         return jsonify(results)
#     except Exception as e:
#         print(f"An unhandled error occurred during synthesis planning: {e}")
#         return jsonify({"error": "An internal server error occurred."}), 500

# # --- Main Execution ---

# if __name__ == '__main__':
#     # Flask will automatically use the settings from the .flaskenv file
#     # for host and port during development.
#     app.run(debug=True)