import os
import json
import logging
import pubchempy as pcp
from aizynthfinder.aizynthfinder import AiZynthFinder
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from utils.llm_interface import generate_text
from utils.prompt_loader import format_prompt
from utils.reaction_utils import map_reaction, generate_reaction_image
from core_logic.sourcing_analysis import analyze_route_cost_and_sourcing
from utils.yield_optimizer import AdvancedYieldPredictor
import requests
import json
from urllib.parse import quote
from utils.balance_reaction import balance_chemical_equation, count_atoms
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resolve_molecule_identifier(identifier: str) -> str | None:
    """
    Resolves a molecule identifier (SMILES, Name, or CAS) to a SMILES string.
    Returns the SMILES string or None if not found.
    """
    identifier = identifier.strip()
    logger.info(f"Attempting to resolve identifier: {identifier}")

    # 1. Check if it's already a valid SMILES
    mol = Chem.MolFromSmiles(identifier, sanitize=False)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
            logger.info("Identifier is a valid SMILES string.")
            return identifier
        except Exception:
            pass # Invalid SMILES, proceed to lookup

    # 2. If not a valid SMILES, query PubChem by name/CAS
    try:
        logger.info(f"Querying PubChem for '{identifier}'...")
        # PubChemPy's 'name' namespace searches names, synonyms, and CAS RNs
        results = pcp.get_compounds(identifier, 'name')
        if results and len(results) > 0:  # Check if results exist and are not empty
            compound = results[0]
            logger.info(f"Found compound CID: {compound.cid}")
            
            # Try to get a fresh compound object by CID to ensure all properties are loaded
            try:
                logger.info(f"Fetching fresh compound data for CID: {compound.cid}")
                fresh_compound = pcp.Compound.from_cid(compound.cid)
                compound = fresh_compound
                logger.info(f"Successfully fetched fresh compound data")
            except Exception as e:
                logger.warning(f"Could not fetch fresh compound data: {e}, using original")
            
            # Try different SMILES properties in order of preference
            smiles = None
            smiles_sources = [
                ('isomeric_smiles', getattr(compound, 'isomeric_smiles', None)),
                ('canonical_smiles', getattr(compound, 'canonical_smiles', None)),
                ('smiles', getattr(compound, 'smiles', None))
            ]
            
            for source_name, smiles_value in smiles_sources:
                logger.info(f"Checking {source_name}: {smiles_value}")
                if smiles_value and str(smiles_value).strip():
                    smiles = str(smiles_value).strip()
                    logger.info(f"Found SMILES from {source_name}: {smiles}")
                    break
            
            if smiles:
                logger.info(f"Resolved '{identifier}' to SMILES: {smiles}")
                return smiles
            else:
                logger.warning(f"PubChem found compound for '{identifier}' (CID: {compound.cid}) but no SMILES available")
                # try a direct API call with correct property names
                try:
                    logger.info(f"Attempting direct API call for CID: {compound.cid}")
                    import requests
                    
                    # Try multiple property names that are valid in PubChem API
                    property_names = [
                        "IsomericSMILES",
                        "CanonicalSMILES", 
                        "SMILES"
                    ]
                    
                    for prop_name in property_names:
                        try:
                            api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/property/{prop_name}/JSON"
                            logger.info(f"Trying API URL: {api_url}")
                            response = requests.get(api_url, timeout=10)
                            logger.info(f"API response status: {response.status_code}")
                            
                            if response.status_code == 200:
                                data = response.json()
                                logger.info(f"API response data: {data}")
                                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                                    properties = data['PropertyTable']['Properties']
                                    if properties and prop_name in properties[0]:
                                        smiles = properties[0][prop_name]
                                        logger.info(f"Got SMILES from direct API ({prop_name}): {smiles}")
                                        return smiles
                                    else:
                                        logger.info(f"No {prop_name} in API response properties")
                                else:
                                    logger.warning(f"Unexpected API response structure: {data}")
                            else:
                                logger.info(f"API call failed with status {response.status_code} for {prop_name}")
                                
                        except Exception as e:
                            logger.warning(f"API call failed for {prop_name}: {e}")
                            continue
                    
                    # If all property API calls failed, try the SDF format approach
                    logger.info("Trying SDF format approach")
                    try:
                        sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/SDF"
                        response = requests.get(sdf_url, timeout=10)
                        if response.status_code == 200:
                            sdf_content = response.text
                            # Parse SDF to extract SMILES using RDKit
                            from io import StringIO
                            sdf_supplier = Chem.SDMolSupplier(StringIO(sdf_content), sanitize=False)
                            for mol in sdf_supplier:
                                if mol is not None:
                                    try:
                                        Chem.SanitizeMol(mol)
                                        smiles = Chem.MolToSmiles(mol)
                                        logger.info(f"Got SMILES from SDF: {smiles}")
                                        return smiles
                                    except Exception as e:
                                        logger.warning(f"Could not sanitize molecule from SDF: {e}")
                                        continue
                        else:
                            logger.warning(f"SDF API call failed with status {response.status_code}")
                    except Exception as e:
                        logger.warning(f"SDF approach failed: {e}")
                    
                except Exception as e:
                    logger.warning(f"All direct API calls failed: {e}")
                    import traceback
                    logger.warning(f"Traceback: {traceback.format_exc()}")
                
                return None
        else:
            logger.warning(f"Identifier '{identifier}' could not be resolved by PubChem.")
            return None
    except Exception as e:
        logger.error(f"An error occurred during PubChem lookup for '{identifier}': {e}")
        return None

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
    
def initialize_yield_predictor() -> AdvancedYieldPredictor | None:
    """
    Initialize the Advanced Yield Predictor with proper error handling.
    Returns the predictor instance or None if initialization fails.
    """
    try:
        this_dir = os.path.dirname(__file__)
        # Update these paths to match your actual model locations
        CONDITION_MODEL_PATH = os.path.join(this_dir, '..', 'models', 'model 2')
        YIELD_MODEL_PATH = os.path.join(this_dir, '..', 'models', 'model')
        
        # Check if model directories exist
        if not os.path.exists(CONDITION_MODEL_PATH):
            logger.warning(f"Condition model path not found: {CONDITION_MODEL_PATH}")
            return None
        if not os.path.exists(YIELD_MODEL_PATH):
            logger.warning(f"Yield model path not found: {YIELD_MODEL_PATH}")
            return None
        
        logger.info("Initializing Advanced Yield Predictor...")
        predictor = AdvancedYieldPredictor(
            condition_model_dir=CONDITION_MODEL_PATH,
            yield_model_dir=YIELD_MODEL_PATH,
            cuda_device=-1  # Use CPU for now
        )
        logger.info("Advanced Yield Predictor initialized successfully")
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to initialize Advanced Yield Predictor: {e}", exc_info=True)
        return None

def plan_synthesis_route(target_identifier: str) -> dict:
    """
    Plans synthesis routes for a target molecule using AiZynthFinder
    and elaborates the steps using an LLM with advanced yield prediction.
    """
    target_smiles = resolve_molecule_identifier(target_identifier)
    if not target_smiles:
        return {"error": f"Invalid or unknown molecule identifier: '{target_identifier}'. Could not resolve to a SMILES string."}

    try:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yml'))
        logger.info(f"Using AiZynthFinder config path: {config_path}")
        if not os.path.exists(config_path):
            return {"error": f"AiZynthFinder config.yml not found at {config_path}"}
    except Exception as e:
        return {"error": f"Could not determine project path: {e}"}

    # Initialize Advanced Yield Predictor
    yield_predictor = initialize_yield_predictor()
    if yield_predictor is None:
        logger.warning("Advanced Yield Predictor not available, falling back to basic yield calculation")

    try:
        # Initialize AiZynthFinder from the config file
        finder = AiZynthFinder(configfile=config_path)
        finder.target_smiles = target_smiles
        logger.info(f"Set target molecule: {target_smiles}")
        finder.stock.select("zinc")  # Matches the stock name in config.yml
        finder.expansion_policy.select(["uspto", "ringbreaker"])  # Select both policies at once
        finder.filter_policy.select("uspto")  # Matches filter policy name in config.yml
        
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

    # --- MODIFIED Route Processing Logic with Advanced Yield Prediction ---
    routes_data = []
    try:
        # Helper function to get route score
        def get_route_score(route):
            try:
                if isinstance(route, dict):
                    score_obj = route.get('score', 0.0)
                else:
                    score_obj = getattr(route, 'score', 0.0)
                
                # Check if the score object is a dictionary and extract the numerical score
                if isinstance(score_obj, dict):
                    score = score_obj.get('state score', 0.0)
                else:
                    score = score_obj

                # Ensure the final score is numeric
                if isinstance(score, (int, float)):
                    return float(score)
                else:
                    logger.warning(f"Could not extract a numeric score from: {score_obj}, type: {type(score_obj)}")
                    return 0.0
            except Exception as e:
                logger.warning(f"Error getting route score: {e}")
                return 0.0
        
        def get_reaction_tree(route):
            """Get reaction tree from route, handling both dict and object formats"""
            if isinstance(route, dict):
                return route.get('reaction_tree', None)
            else:
                return getattr(route, 'reaction_tree', None)
        
        # Create a stable sort key that includes both score and index
        routes_with_index = [(route, idx) for idx, route in enumerate(routes)]
        sorted_routes_with_index = sorted(
            routes_with_index, 
            key=lambda x: (get_route_score(x[0]), -x[1]),  # Sort by score desc, then by index asc
            reverse=True
        )
        sorted_routes = [route for route, _ in sorted_routes_with_index]
        
        for index, route in enumerate(sorted_routes[:3]):
            route_score = get_route_score(route)
            route_id = f"route_{chr(97 + index)}"
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
                if hasattr(reaction_tree, 'reactions') and callable(getattr(reaction_tree, 'reactions')):
                    reaction_steps = list(reaction_tree.reactions())
                else:
                    logger.warning(f"Reaction tree for route {index} does not have a reactions() method")
                    continue
                    
            except Exception as e:
                logger.warning(f"Could not extract reactions from route {index}: {e}")
                continue

            # Collect all reaction SMILES for batch yield prediction
            reaction_smiles_list = []
            step_metadata = []
            
            for step_index, reaction in enumerate(reversed(reaction_steps)):
                try:
                    reactants_smiles = [get_smiles_from_molecule(mol) for group in reaction.reactants for mol in group if mol]
                    product_smiles = get_smiles_from_molecule(reaction.mol)
                    
                    if not reactants_smiles or not product_smiles: 
                        continue
                    
                    reaction_smiles_str = f"{'.'.join(reactants_smiles)}>>{product_smiles}"
                    reaction_smiles_list.append(reaction_smiles_str)
                    step_metadata.append({
                        'step_index': step_index,
                        'reaction': reaction,
                        'reactants_smiles': reactants_smiles,
                        'product_smiles': product_smiles,
                        'reaction_smiles_str': reaction_smiles_str
                    })
                    
                except Exception as e:
                    logger.error(f"Error preparing reaction for step {step_index} in route {index}: {e}")
                    continue
            
            # Predict yields for all reactions in this route using the advanced predictor
            predicted_yields = {}
            predicted_conditions = {}
            if yield_predictor and reaction_smiles_list:
                try:
                    logger.info(f"Predicting yields for {len(reaction_smiles_list)} reactions in route {index + 1}")
                    yield_results = yield_predictor.predict(reaction_smiles_list)
                    
                    for i, result in enumerate(yield_results):
                        if 'error' not in result:
                            # Extract the numerical yield value
                            yield_str = result.get('best_yield', '0.0%')
                            yield_value = float(yield_str.replace('%', ''))
                            predicted_yields[reaction_smiles_list[i]] = yield_value
                            optimal_conditions = result.get('optimal_conditions', {})
                            predicted_conditions[reaction_smiles_list[i]] = optimal_conditions
                            logger.info(f"Predicted yield for reaction {i+1}: {yield_value:.1f}%")
                            logger.info(f"Predicted conditions: {optimal_conditions}")
                        else:
                            logger.warning(f"Yield prediction failed for reaction {i+1}: {result['error']}")
                            
                except Exception as e:
                    logger.error(f"Error during yield prediction for route {index + 1}: {e}")
            
            # Process each step with the predicted yields
            for metadata in step_metadata:
                step_index = metadata['step_index']
                reaction = metadata['reaction']
                reactants_smiles = metadata['reactants_smiles']
                product_smiles = metadata['product_smiles']
                main_product_smiles = metadata['product_smiles'] 
                reaction_smiles_str = metadata['reaction_smiles_str']

                # Default to the original, incomplete reaction
                all_reactants_smiles = metadata['reactants_smiles']
                all_products_smiles = [metadata['product_smiles']]

                # Get the full reaction details from the yield prediction result
                yield_result = next((r for i, r in enumerate(yield_results) if reaction_smiles_list[i] == reaction_smiles_str), None)

                if yield_result and 'optimized_reaction_smiles' in yield_result:
                    optimized_reaction = yield_result['optimized_reaction_smiles']
                    logger.info(f"Using optimized reaction for balancing: {optimized_reaction}")
                    try:
                        reactants_part, products_part = optimized_reaction.split('>>')
                        all_reactants_smiles = [smi for smi in reactants_part.split('.') if smi]
                        all_products_smiles = [smi for smi in products_part.split('.') if smi]
                    except ValueError:
                        logger.warning(f"Could not parse optimized reaction: {optimized_reaction}. Falling back to original.")
                
                # Initialize coefficients with a default of 1
                reactant_coeffs = {smi: 1 for smi in all_reactants_smiles}
                product_coeffs = {smi: 1 for smi in all_products_smiles}

                # PRE-EMPTIVE CHECK: Verify atom conservation before attempting to balance
                try:
                    reactant_atoms = Counter()
                    for smi in all_reactants_smiles:
                        reactant_atoms.update(count_atoms(smi))

                    product_atoms = Counter()
                    for smi in all_products_smiles:
                        product_atoms.update(count_atoms(smi))

                    # If atoms are conserved, attempt to find integer coefficients
                    if reactant_atoms.keys() == product_atoms.keys():
                        logger.info("Atom sets match. Attempting to balance.")
                        balanced_reactants, balanced_products = balance_chemical_equation(all_reactants_smiles, all_products_smiles)
                        
                        reactant_coeffs = {smi: coeff for coeff, smi in zip([br[0] for br in balanced_reactants], all_reactants_smiles)}
                        product_coeffs = {smi: coeff for coeff, smi in zip([bp[0] for bp in balanced_products], all_products_smiles)}
                        
                        logger.info(f"Successfully balanced reaction for step {step_index + 1}.")
                    else:
                        # Atoms are not conserved; the equation is incomplete. Use 1:1 stoichiometry.
                        missing_in_products = set(reactant_atoms.keys()) - set(product_atoms.keys())
                        missing_in_reactants = set(product_atoms.keys()) - set(reactant_atoms.keys())
                        logger.warning(
                            f"Cannot balance: predicted reaction is incomplete. "
                            f"Missing from products: {missing_in_products}. "
                            f"Missing from reactants: {missing_in_reactants}. "
                            f"Defaulting to 1:1 stoichiometry for all components."
                        )
                except Exception as e:
                    logger.warning(f"Balancing check failed: {e}. Defaulting to 1:1 stoichiometry.")
                
                predicted_condition_info = predicted_conditions.get(reaction_smiles_str, {})
                
                # Format conditions string
                conditions_parts = []
                if predicted_condition_info.get('catalyst'):
                    conditions_parts.append(f"Catalyst: {predicted_condition_info['catalyst']}")
                if predicted_condition_info.get('solvents'):
                    conditions_parts.append(f"Solvent: {predicted_condition_info['solvents']}")
                if predicted_condition_info.get('reagents'):
                    conditions_parts.append(f"Reagents: {predicted_condition_info['reagents']}")
                if predicted_condition_info.get('temperature'):
                    conditions_parts.append(f"Temperature: {predicted_condition_info['temperature']}")
                predicted_conditions_str = "; ".join(conditions_parts) if conditions_parts else "Conditions not predicted"
                
                # Use predicted yield if available, otherwise fall back to original method
                if reaction_smiles_str in predicted_yields:
                    yield_value = predicted_yields[reaction_smiles_str]
                    logger.info(f"Using predicted yield for step {step_index + 1}: {yield_value:.1f}%")
                else:
                    # Fallback to original yield calculation
                    yield_fraction = reaction.metadata.get('plausibility', reaction.metadata.get('template_score', 0.0))
                    if yield_fraction == 0.0:
                        num_steps = len(step_metadata)
                        if num_steps > 0 and route_score > 0:
                            yield_fraction = route_score**(1 / num_steps)
                    yield_value = yield_fraction * 100
                    logger.info(f"Using fallback yield for step {step_index + 1}: {yield_value:.1f}%")
                
                step_details = {"title": "Reaction", "conditions": "N/A", "notes": "N/A"}
                step_id = f"{route_id}_step_{step_index + 1}"
                image_url = generate_reaction_image(reaction_smiles_str, step_id)

                try:
                    elaboration_prompt = format_prompt("elaborate_synthesis_step", reaction_smiles=reaction_smiles_str)
                    if elaboration_prompt:
                        llm_response = generate_text(elaboration_prompt, temperature=0.5)
                        if llm_response:
                            cleaned_response = llm_response.strip().replace('```json', '').replace('```', '').strip()
                            step_details = json.loads(cleaned_response)
                except Exception as e:
                    logger.warning(f"LLM elaboration failed for step {step_index}: {e}")

                route_description_for_eval.append(f"Step {step_index + 1}: {step_details.get('title')} (Yield: {yield_value:.1f}%)")

                byproducts = [smi for smi in all_products_smiles if smi != main_product_smiles]

                route_info["steps"].append({
                    "step_number": step_index + 1,
                    "title": step_details.get("title", "Unnamed Reaction"),
                    "yield": yield_value,
                    "reagents_conditions": predicted_conditions_str, 
                    "predicted_conditions": predicted_condition_info, 
                    "source_notes": step_details.get("notes", "No specific notes available."),
                    # The main product of this step
                    "product": {
                        "smiles": main_product_smiles, 
                        "formula": get_formula_from_smiles(main_product_smiles),
                        "coeff": product_coeffs.get(main_product_smiles, 1)
                    },
                    # All reactants needed for the reaction
                    "reactants": [{
                        "smiles": smi, 
                        "formula": get_formula_from_smiles(smi),
                        "coeff": reactant_coeffs.get(smi, 1)
                    } for smi in all_reactants_smiles],
                    "byproducts": [{
                        "smiles": smi,
                        "formula": get_formula_from_smiles(smi),
                        "coeff": product_coeffs.get(smi, 1)
                    } for smi in byproducts],
                    "reaction_image_url": image_url,
                    "optimized_reaction_smiles": yield_result.get('optimized_reaction_smiles', reaction_smiles_str) if yield_result else reaction_smiles_str,
                    "naked_reaction_smiles": reaction_smiles_str
                })
            
            # Calculate overall yield from individual step yields
            if route_info["steps"]:
                overall_yield = 1.0
                for step in route_info["steps"]:
                    overall_yield *= (step["yield"] / 100.0)
                route_info["overall_yield"] = overall_yield * 100
                logger.info(f"Calculated overall yield for route {index + 1}: {route_info['overall_yield']:.1f}%")
            
            if route_description_for_eval:
                try:
                    # Create a more detailed description for the LLM to improve its evaluation.
                    full_description_for_eval = (
                        f"Overall Predicted Yield: {route_info.get('overall_yield', 0.0):.1f}%. "
                        f"Individual Steps: {' -> '.join(route_description_for_eval)}"
                    )
                    # NOTE: The 'evaluate_synthesis_route' prompt should instruct the LLM to consider yields.
                    # Example prompt instruction: "Analyze the route's advantages and challenges.
                    # Pay close attention to the overall yield and the yield of each individual step.
                    # A low-yield step is a significant challenge, while high-yield steps are advantageous."
                    eval_prompt = format_prompt(
                        "evaluate_synthesis_route",
                        route_description=full_description_for_eval
                    )
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
        logger.error(f"Critical error processing routes: {e}", exc_info=True)
        return {"error": f"An error occurred while processing synthesis routes: {str(e)}"}

    logger.info(f"Successfully processed and returning {len(routes_data)} routes")
    return {"routes": routes_data}