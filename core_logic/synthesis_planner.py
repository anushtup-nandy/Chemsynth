# chem_mvp/core_logic/synthesis_planner.py
import os
import json
import logging
from aizynthfinder.aizynthfinder import AiZynthFinder
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from utils.llm_interface import generate_text
from utils.prompt_loader import format_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- RDKit Helper Functions ---
def get_smiles_from_molecule(mol):
    """
    Extract SMILES string from a molecule object.
    Handles both RDKit ROMol objects and AiZynthFinder UniqueMolecule objects.
    """
    if mol is None:
        return None
    
    try:
        # Check if it's an AiZynthFinder UniqueMolecule object
        if hasattr(mol, 'mol'):
            # UniqueMolecule has a .mol attribute that contains the RDKit molecule
            rdkit_mol = mol.mol
        elif hasattr(mol, 'rd_mol'):
            # Alternative attribute name in some versions
            rdkit_mol = mol.rd_mol
        else:
            # Assume it's already an RDKit molecule
            rdkit_mol = mol
        
        # Now convert the RDKit molecule to SMILES
        return Chem.MolToSmiles(rdkit_mol) if rdkit_mol else None
        
    except Exception as e:
        logger.warning(f"Could not convert molecule to SMILES: {e}")
        return None

def get_formula_from_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return rdMolDescriptors.CalcMolFormula(mol)
    except Exception as e:
        logger.warning(f"Could not calculate formula for SMILES {smiles}: {e}")
    return "N/A"

def validate_config_file(config_path: str) -> dict:
    """
    Validate that the config file exists and contains required sections.
    It checks for model/stock files relative to the config file's location.
    Returns dict with 'valid' boolean and 'error' message if invalid.
    """
    if not os.path.exists(config_path):
        return {"valid": False, "error": f"Config file not found at {config_path}"}
    
    config_dir = os.path.dirname(config_path)

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = ['policy', 'stock']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            return {"valid": False, "error": f"Missing required sections in config: {missing_sections}"}
        
        # Check if policy files exist (relative to the config file)
        if 'files' in config.get('policy', {}):
            for policy_file in config['policy']['files']:
                abs_policy_path = os.path.join(config_dir, policy_file)
                if not os.path.exists(abs_policy_path):
                    return {"valid": False, "error": f"Policy file not found: {abs_policy_path}"}
        
        # Check if stock files exist (relative to the config file)
        # Note: AiZynthFinder stock can be configured in many ways. This checks a simple file path.
        stock_config = config.get('stock', {})
        for stock_name, stock_details in stock_config.items():
            if 'path' in stock_details:
                 abs_stock_path = "zinc_stock.hdf5"
                 if not os.path.exists(abs_stock_path):
                      return {"valid": False, "error": f"Stock file not found: {abs_stock_path}"}

        return {"valid": True, "error": None}
        
    except Exception as e:
        return {"valid": False, "error": f"Error reading or validating config file: {str(e)}"}


def initialize_aizynthfinder(config_path: str) -> dict:
    """
    Initialize AiZynthFinder with proper error handling.
    Returns dict with 'finder' object and 'error' message if failed.
    """
    try:
        logger.info(f"Initializing AiZynthFinder with config: {config_path}")
        
        # Validate config first
        config_validation = validate_config_file(config_path)
        if not config_validation["valid"]:
            return {"finder": None, "error": config_validation["error"]}
        
        # Initialize AiZynthFinder
        # AiZynthFinder resolves paths inside the config file relative to the config file's location.
        finder = AiZynthFinder(configfile=config_path)
        
        logger.info("AiZynthFinder initialized successfully")
        return {"finder": finder, "error": None}
        
    except ImportError as e:
        return {"finder": None, "error": f"Missing required dependencies for AiZynthFinder: {str(e)}"}
    except Exception as e:
        logger.error(f"Error initializing AiZynthFinder: {e}", exc_info=True)
        return {"finder": None, "error": f"An unexpected error occurred during AiZynthFinder initialization: {str(e)}"}

# --- Main Synthesis Planning Logic (with the CORRECTED NAMES) ---
def plan_synthesis_route(target_smiles: str) -> dict:
    """
    Plans synthesis routes for a target molecule using AiZynthFinder
    and elaborates the steps using an LLM.
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol:
        return {"error": "Invalid target SMILES string provided."}

    try:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yml'))
        logger.info(f"Using AiZynthFinder config path: {config_path}")
        if not os.path.exists(config_path):
            return {"error": f"AiZynthFinder config.yml not found at {config_path}"}
    except Exception as e:
        return {"error": f"Could not determine project path: {e}"}

    try:
        # Initialize AiZynthFinder from the config file
        finder = AiZynthFinder(configfile=config_path)
        finder.target_smiles = target_smiles
        logger.info(f"Set target molecule: {target_smiles}")
        finder.stock.select("zinc")  # Matches the stock name in config.yml
        finder.expansion_policy.select(["uspto", "ringbreaker"])  # Select both policies at once
        finder.filter_policy.select("uspto")  # Matches filter policy name in config.yml
        # ***************************************************************
        
        logger.info("Policies and stock selected. Preparing search tree.")
        finder.prepare_tree()
        logger.info("Tree prepared. Starting search.")
        
        # Check if analysis object exists before setting max_time
        if hasattr(finder, 'analysis') and finder.analysis is not None:
            # finder.analysis.max_time = 120  # 120 seconds = 2 minutes
            logger.info("Set analysis max_time to 120 seconds")
        else:
            logger.warning("Analysis object not available, proceeding without time limit")
        
        # Perform the tree search with proper error handling
        search_stats = finder.tree_search()
        logger.info(f"Tree search completed. Stats: {search_stats}")
        
        finder.build_routes()
        
        routes = list(finder.routes)
        if not routes:
            logger.warning(f"No synthesis routes found for {target_smiles} within the time limit.")
            return {"routes": []}
            
        logger.info(f"Found {len(routes)} synthesis routes.")
        
    except Exception as e:
        logger.error(f"Error running AiZynthFinder search: {e}", exc_info=True)
        # More specific error for the frontend
        if "No expansion policy selected" in str(e):
            return {"error": "AiZynthFinder Error: No expansion policy was loaded. Check that policy names in config.yml are correct and the model files exist."}
        if "is not a known" in str(e):
             return {"error": f"AiZynthFinder Error: A specified policy or stock name in the config is unknown. Details: {e}"}
        return {"error": f"Failed to run synthesis planning: {str(e)}"}

    # --- Route Processing Logic (Fixed for dict routes) ---
    routes_data = []
    try:
        # Handle both dict and object routes
        def get_route_score(route):
            try:
                if isinstance(route, dict):
                    score = route.get('score', 0.0)
                else:
                    score = getattr(route, 'score', 0.0)
                
                # Ensure score is numeric
                if isinstance(score, (int, float)):
                    return float(score)
                else:
                    logger.warning(f"Non-numeric score found: {score}, type: {type(score)}")
                    return 0.0
            except Exception as e:
                logger.warning(f"Error getting route score: {e}")
                return 0.0
        
        def get_reaction_tree(route):
            if isinstance(route, dict):
                return route.get('reaction_tree', None)
            else:
                return getattr(route, 'reaction_tree', None)
        
        # FIX: Create a stable sort key that includes both score and index
        # This prevents the comparison of dict objects when scores are equal
        routes_with_index = [(route, idx) for idx, route in enumerate(routes)]
        sorted_routes_with_index = sorted(
            routes_with_index, 
            key=lambda x: (get_route_score(x[0]), -x[1]),  # Sort by score desc, then by index asc
            reverse=True
        )
        sorted_routes = [route for route, _ in sorted_routes_with_index]
        
        for index, route in enumerate(sorted_routes[:3]):
            route_score = get_route_score(route)
            logger.info(f"Processing route {index + 1} with score {route_score}")
            route_info = { 
                "id": f"route_{chr(97 + index)}", 
                "overall_yield": route_score * 100 if route_score else 0.0, 
                "steps": [], 
                "evaluation": {} 
            }
            route_description_for_eval = []
            
            try:
                reaction_tree = get_reaction_tree(route)
                if reaction_tree is None:
                    logger.warning(f"No reaction tree found for route {index}")
                    continue
                reaction_steps = list(reaction_tree.reactions())
            except Exception as e:
                logger.warning(f"Could not extract reactions from route {index}: {e}")
                continue
            
            for step_index, reaction in enumerate(reversed(reaction_steps)):
                try:
                    reactants_smiles = [get_smiles_from_molecule(mol) for group in reaction.reactants for mol in group if mol]
                    product_smiles = get_smiles_from_molecule(reaction.mol)
                    
                    if not reactants_smiles or not product_smiles: continue
                    
                    reaction_smiles_str = f"{'.'.join(reactants_smiles)}>>{product_smiles}"
                    step_details = {"title": "Reaction", "conditions": "N/A", "notes": "N/A"}
                    
                    try:
                        elaboration_prompt = format_prompt("elaborate_synthesis_step", reaction_smiles=reaction_smiles_str)
                        if elaboration_prompt:
                            llm_response = generate_text(elaboration_prompt, temperature=0.5)
                            if llm_response:
                                cleaned_response = llm_response.strip().replace('```json', '').replace('```', '').strip()
                                step_details = json.loads(cleaned_response)
                    except Exception as e:
                        logger.warning(f"LLM elaboration failed for step {step_index}: {e}")

                    yield_value = reaction.metadata.get('template_score', 0.0) * 100
                    route_description_for_eval.append(f"Step {step_index + 1}: {step_details.get('title')} (Yield: {yield_value:.0f}%)")

                    route_info["steps"].append({
                        "step_number": step_index + 1,
                        "title": step_details.get("title", "Unnamed Reaction"),
                        "yield": yield_value,
                        "reagents_conditions": step_details.get("conditions", "Not specified"),
                        "source_notes": step_details.get("notes", "No specific notes available."),
                        "product": {"smiles": product_smiles, "formula": get_formula_from_smiles(product_smiles)},
                        "reactants": [{"smiles": smi, "formula": get_formula_from_smiles(smi)} for smi in reactants_smiles]
                    })
                except Exception as e:
                    logger.error(f"Error processing step {step_index} in route {index}: {e}", exc_info=True)
                    continue
            
            if route_description_for_eval:
                try:
                    eval_prompt = format_prompt("evaluate_synthesis_route", route_description=" -> ".join(route_description_for_eval))
                    if eval_prompt:
                        eval_response = generate_text(eval_prompt, temperature=0.5)
                        if eval_response:
                            cleaned_response = eval_response.strip().replace('```json', '').replace('```', '').strip()
                            route_info["evaluation"] = json.loads(cleaned_response)
                except Exception as e:
                    logger.warning(f"Route evaluation failed: {e}")
            
            if route_info["steps"]:
                routes_data.append(route_info)
    except Exception as e:
        logger.error(f"A critical error occurred while processing the found routes: {e}", exc_info=True)
        return {"error": f"An error occurred while processing the synthesis routes: {str(e)}"}

    logger.info(f"Successfully processed and returning {len(routes_data)} routes")
    return {"routes": routes_data}