# app.py
from flask import Flask, render_template, request, jsonify
import sys
import os
import traceback
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from core_logic.literature_analysis import perform_literature_search
from core_logic.synthesis_planner import plan_synthesis_route, generate_hypothetical_route, resolve_molecule_identifier
from core_logic.sourcing_analysis import analyze_route_cost_and_sourcing
from core_logic.chemfm_synthesis_engine import generate_route_with_chemfm
from utils.pubchem_processor import search_pubchem_literature
try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False
    print("Warning: pubchempy not available. Some functionality may be limited.")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
        name = identifier  # Default fallback
        synonyms = []
        
        if PUBCHEM_AVAILABLE:
            try:
                compounds = pcp.get_compounds(smiles, 'smiles')
                if compounds and compounds[0]:
                    compound = compounds[0]
                    
                    # Try to get synonyms
                    if hasattr(compound, 'synonyms') and compound.synonyms:
                        synonyms = compound.synonyms[:5]  # Get first 5 synonyms
                        # Find the shortest reasonable name
                        short_names = [s for s in synonyms if len(s) < 50 and not any(char in s for char in ['(', ')', '[', ']'])]
                        if short_names:
                            name = short_names[0]
                        else:
                            name = synonyms[0]
                    
                    # Fallback to IUPAC name
                    if name == identifier and hasattr(compound, 'iupac_name') and compound.iupac_name:
                        name = compound.iupac_name
            except Exception as e:
                print(f"Error getting compound details: {e}")
        
        return jsonify({
            "smiles": smiles, 
            "name": name,
            "synonyms": synonyms
        })
        
    except Exception as e:
        print(f"An error occurred during identifier resolution: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An internal error occurred during resolution."}), 500


@app.route('/api/literature_search', methods=['POST'])
def api_literature_search():
    """
    Enhanced literature search endpoint with better error handling and debugging.
    """
    data = request.json
    search_query = data.get('query')
    max_results = data.get('max_results', 5)
    
    if not search_query:
        return jsonify({"error": "Search query is required."}), 400
    
    try:
        print(f"=== Starting Literature Search ===")
        print(f"Original query: '{search_query}'")
        print(f"Max results: {max_results}")
        
        # Try to resolve the search query to get both SMILES and common name
        compound_name = None
        resolved_smiles = None
        
        try:
            # First, try to resolve the identifier
            resolved_smiles = resolve_molecule_identifier(search_query)
            if resolved_smiles:
                print(f"Resolved SMILES: {resolved_smiles}")
                
                # If we got SMILES, try to get the common name and synonyms
                if PUBCHEM_AVAILABLE:
                    compounds = pcp.get_compounds(resolved_smiles, 'smiles')
                    if compounds and compounds[0]:
                        compound = compounds[0]
                        
                        compound_name = None                        
                        try:
                            if hasattr(compound, 'synonyms') and compound.synonyms:
                                synonyms = compound.synonyms
                                common_names = [s for s in synonyms if len(s) < 30 and not any(char in s for char in ['(', ')', '[', ']', ','])]
                                if common_names:
                                    compound_name = common_names[0]
                                else:
                                    compound_name = synonyms[0]
                        except:
                            pass
                        
                        # Fallback to IUPAC name
                        if not compound_name:
                            try:
                                compound_name = getattr(compound, 'iupac_name', None)
                            except:
                                pass
                        
                        # Final fallback
                        if not compound_name:
                            compound_name = search_query
                        
                        print(f"Resolved compound name: '{compound_name}'")
                else:
                    print("PubChem not available - using original query as compound name")
                    compound_name = search_query
                    
        except Exception as e:
            print(f"Could not resolve identifier '{search_query}': {e}")
            # Continue with original query if resolution fails
        
        # Perform literature search with original query, resolved compound name, and SMILES
        print("=== Calling perform_literature_search ===")
        results = perform_literature_search(
            query=search_query,
            compound_name=compound_name,
            resolved_smiles=resolved_smiles,
            max_results=max_results
        )
        
        if 'search_info' not in results:
            results['search_info'] = {}
        
        results['search_info'].update({
            'original_query': search_query,
            'resolved_compound_name': compound_name,
            'resolved_smiles': resolved_smiles,
            'pubchem_available': PUBCHEM_AVAILABLE
        })
        
        print(f"=== Literature Search Complete ===")
        print(f"Found {len(results.get('papers', []))} papers")
        print(f"Errors: {results.get('search_info', {}).get('errors', [])}")
        
        return jsonify(results)
        
    except Exception as e:
        print(f"An error occurred during literature search: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": "An internal error occurred during literature search.",
            "details": str(e)
        }), 500


@app.route('/api/pubchem_literature', methods=['POST'])
def api_pubchem_literature():
    """
    Direct PubChem literature search endpoint for testing/debugging.
    """
    data = request.json
    query = data.get('query')
    max_results = data.get('max_results', 5)
    
    if not query:
        return jsonify({"error": "Query is required."}), 400
    
    if not PUBCHEM_AVAILABLE:
        return jsonify({"error": "PubChem functionality not available. Please install pubchempy."}), 503
    
    try:
        print(f"Direct PubChem search for: '{query}'")
        results = search_pubchem_literature(query, max_results)
        
        return jsonify({
            "results": results,
            "count": len(results),
            "query": query
        })
        
    except Exception as e:
        print(f"Error in direct PubChem search: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": "PubChem search failed.",
            "details": str(e)
        }), 500


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
        print(f"Planning synthesis for: '{target_identifier}'")
        results = plan_synthesis_route(target_identifier)
        if "error" in results:
            return jsonify(results), 400 
        return jsonify(results)
    except Exception as e:
        print(f"An unhandled error occurred during synthesis planning: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred."}), 500


# --- SOURCING & COST ANALYSIS ENDPOINT ---
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
        print(f"Analyzing sourcing for {len(route_steps)} route steps, target amount: {target_amount_g}g")
        analysis_results = analyze_route_cost_and_sourcing(route_steps, target_amount_g)
        return jsonify(analysis_results)
    except Exception as e:
        print(f"An unhandled error occurred during sourcing analysis: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred during analysis."}), 500


@app.route('/api/generate_new_route', methods=['POST'])
def api_generate_new_route():
    """
    API endpoint to generate alternative synthesis routes using the ChemFM engine.
    """
    data = request.json
    # The frontend sends 'target_smiles' and 'suggestion' now
    target_smiles = data.get('target_smiles')
    suggestion = data.get('suggestion', 'Any reasonable synthesis.') # Default suggestion
    
    if not target_smiles:
        return jsonify({"error": "Target molecule SMILES is required."}), 400
    
    try:
        print(f"Generating new ChemFM route for: '{target_smiles}' with suggestion: '{suggestion}'")
        # Call the new function from our new engine
        results = generate_route_with_chemfm(target_smiles, suggestion)
        
        if "error" in results:
             # Pass the specific error from the engine to the frontend
            return jsonify(results), 500
            
        return jsonify(results)
    except Exception as e:
        print(f"An error occurred during ChemFM route generation: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred during route generation."}), 500


# --- HEALTH CHECK ENDPOINT ---
@app.route('/api/health', methods=['GET'])
def api_health():
    """
    Health check endpoint to verify system status.
    """
    try:
        # Test basic imports
        status = {
            "status": "healthy",
            "pubchem_available": PUBCHEM_AVAILABLE,
            "modules": {
                "literature_analysis": True,
                "synthesis_planner": True,
                "sourcing_analysis": True,
                "pubchem_processor": True
            }
        }
        
        # Test basic functionality
        try:
            from utils import arxiv_processor, web_searcher
            status["modules"]["arxiv_processor"] = True
            status["modules"]["web_searcher"] = True
        except ImportError as e:
            status["modules"]["arxiv_processor"] = False
            status["modules"]["web_searcher"] = False
            status["warnings"] = status.get("warnings", []) + [f"Import warning: {str(e)}"]
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# --- ERROR HANDLERS ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("=== Flask App Starting ===")
    print(f"PubChem available: {PUBCHEM_AVAILABLE}")
    
    # Test imports on startup
    try:
        from utils import arxiv_processor, web_searcher
        print("Successfully imported arxiv_processor and web_searcher")
    except ImportError as e:
        print(f"Warning: Could not import utils modules: {e}")
    
    print("Available endpoints:")
    print("  GET  / - Frontend")
    print("  POST /api/resolve_identifier - Resolve molecule identifiers")
    print("  POST /api/literature_search - Comprehensive literature search")
    print("  POST /api/pubchem_literature - Direct PubChem literature search")
    print("  POST /api/plan_synthesis - Synthesis route planning")
    print("  POST /api/analyze_sourcing - Cost and sourcing analysis")
    print("  POST /api/generate_new_route - Alternative route generation")
    print("  GET  /api/health - System health check")
    print("=========================")
    
    app.run(debug=True, host='0.0.0.0', port=5000)