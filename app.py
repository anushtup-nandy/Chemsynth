# app.py
from flask import Flask, render_template, request, jsonify
from core_logic.literature_analysis import perform_literature_search
# MODIFIED imports
from core_logic.synthesis_planner import plan_synthesis_route, generate_hypothetical_route, resolve_molecule_identifier
from core_logic.sourcing_analysis import analyze_route_cost_and_sourcing # <<< NEW IMPORT

# Initialize Flask App
app = Flask(__name__)

# --- Primary Route to Serve the Frontend ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API Endpoints ---

# --- UTILITY ENDPOINT ---
@app.route('/api/resolve_identifier', methods=['POST'])
def api_resolve_identifier():
    """
    Resolves a molecule identifier to its canonical SMILES and common name.
    """
    data = request.json
    identifier = data.get('identifier')
    if not identifier:
        return jsonify({"error": "Identifier is required."}), 400
    
    try:
        # We can reuse the resolver from the synthesis planner
        smiles = resolve_molecule_identifier(identifier)
        if not smiles:
            return jsonify({"error": f"Could not resolve identifier: '{identifier}'"}), 404
        
        # Additionally, let's try to get a common name for better search queries
        import pubchempy as pcp
        compounds = pcp.get_compounds(smiles, 'smiles')
        name = compounds[0].iupac_name if compounds and compounds[0].iupac_name else identifier
        
        return jsonify({"smiles": smiles, "name": name})
    except Exception as e:
        print(f"An error occurred during identifier resolution: {e}")
        return jsonify({"error": "An internal error occurred during resolution."}), 500


@app.route('/api/literature_search', methods=['POST'])
def api_literature_search():
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
    target_identifier = data.get('identifier') 

    if not target_identifier:
        return jsonify({"error": "Target molecule identifier is required."}), 400

    try:
        results = plan_synthesis_route(target_identifier)
        if "error" in results:
            return jsonify(results), 400 
        return jsonify(results)
    except Exception as e:
        print(f"An unhandled error occurred during synthesis planning: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

# --- NEW SOURCING & COST ANALYSIS ENDPOINT ---
@app.route('/api/analyze_sourcing', methods=['POST'])
def api_analyze_sourcing():
    """
    API endpoint for Sourcing and Cost analysis.
    Accepts synthesis route steps and a target amount.
    """
    data = request.json
    route_steps = data.get('route_steps')
    target_amount_g = float(data.get('target_amount_g', 1.0))

    if not route_steps:
        return jsonify({"error": "Route steps are required for analysis."}), 400

    try:
        # This function comes from core_logic/sourcing_analysis.py
        analysis_results = analyze_route_cost_and_sourcing(route_steps, target_amount_g)
        return jsonify(analysis_results)
    except Exception as e:
        print(f"An unhandled error occurred during sourcing analysis: {e}")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500


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