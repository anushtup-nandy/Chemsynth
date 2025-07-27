import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import json
import logging
import os
from utils.llm_interface import generate_text
from utils.prompt_loader import format_prompt
from utils.reaction_utils import generate_reaction_image
from utils.yield_optimizer import AdvancedYieldPredictor
from transformers.models.llama.modeling_llama import LlamaAttention

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Caching ---
# We cache models and tokenizers in memory to avoid reloading them on every request.
_chemfm_model = None
_chemfm_tokenizer = None
_yield_predictor = None
_device = "mps" if torch.mps.is_available() else "cpu" 

def _get_chemfm_model():
    """
    Loads and caches the ChemFM model and tokenizer.
    On subsequent calls, it returns the cached objects.
    """
    global _chemfm_model, _chemfm_tokenizer
    if _chemfm_model is None or _chemfm_tokenizer is None:
        logger.info("Initializing ChemFM model... This may take a moment.")
        try:
            model_name = "ChemFM/ChemFM-3B"
            _chemfm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _chemfm_model = AutoModelForCausalLM.from_pretrained(model_name).to(_device)
            # _chemfm_model = AutoModel.from_pretrained(model_name, ignore_mismatched_sizes=True).to(_device)
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            _chemfm_model.eval() # Set model to evaluation mode
            logger.info(f"ChemFM model '{model_name}' loaded successfully on device '{_device}'.")
        except Exception as e:
            logger.error(f"Failed to load ChemFM model: {e}", exc_info=True)
            raise RuntimeError(f"Could not load ChemFM model. Please check your internet connection and Hugging Face credentials. Error: {e}")
    return _chemfm_model, _chemfm_tokenizer

def _get_yield_predictor():
    """
    Loads and caches the AdvancedYieldPredictor.
    On subsequent calls, it returns the cached object. Returns None if models are not found.
    """
    global _yield_predictor
    if _yield_predictor is None:
        logger.info("Initializing Advanced Yield Predictor...")
        try:
            this_dir = os.path.dirname(__file__)
            # Model paths are relative to the parent directory of 'core_logic'
            CONDITION_MODEL_PATH = os.path.join(this_dir, '..', 'models', 'model 2')
            YIELD_MODEL_PATH = os.path.join(this_dir, '..', 'models', 'model')

            if not os.path.exists(CONDITION_MODEL_PATH) or not os.path.exists(YIELD_MODEL_PATH):
                logger.warning(f"Yield predictor model paths not found. Searched for '{CONDITION_MODEL_PATH}' and '{YIELD_MODEL_PATH}'. Yield optimization will be skipped.")
                return None

            _yield_predictor = AdvancedYieldPredictor(
                condition_model_dir=CONDITION_MODEL_PATH,
                yield_model_dir=YIELD_MODEL_PATH,
                cuda_device=-1  # Use CPU for compatibility
            )
            logger.info("Advanced Yield Predictor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Advanced Yield Predictor: {e}", exc_info=True)
            # Continue without the predictor, so we don't block the main functionality
            return None
    return _yield_predictor


def _calculate_properties(smiles: str) -> dict:
    """Calculates key molecular properties using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"error": "Invalid SMILES for property calculation"}
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "formula": rdMolDescriptors.CalcMolFormula(mol)
        }
    except Exception as e:
        logger.warning(f"Could not calculate properties for {smiles}: {e}")
        return {}

def generate_route_with_chemfm(target_smiles: str, user_suggestion: str) -> dict:
    """
    Generates a new, single-step synthesis route using ChemFM, optimizes its yield,
    and then uses a general LLM to format and evaluate the result.

    Args:
        target_smiles: The SMILES string of the target molecule.
        user_suggestion: The user's text prompt for guidance.

    Returns:
        A dictionary representing the new synthesis route, formatted for the frontend.
    """
    logger.info(f"Generating new route for '{target_smiles}' with suggestion: '{user_suggestion}'")
    
    # 1. Load the ChemFM model
    try:
        model, tokenizer = _get_chemfm_model()
    except RuntimeError as e:
        return {"error": str(e)}

    # 2. Formulate the prompt for ChemFM for retrosynthesis
    retrosynthesis_prompt = f"Product>>{target_smiles}"
    logger.info(f"ChemFM prompt for retrosynthesis: '{retrosynthesis_prompt}'")

    # 3. Perform inference with ChemFM
    try:
        inputs = tokenizer(
            retrosynthesis_prompt, 
            return_tensors="pt", 
            add_special_tokens=False
        ).to(_device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256, 
                num_beams=5,
                early_stopping=True,
                num_return_sequences=1,
                repetition_penalty=1.2,       
                no_repeat_ngram_size=3  
            )
        
        raw_prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Raw ChemFM prediction: {raw_prediction}")
        
        cleaned_prediction = raw_prediction.replace(' ', '') 
        parts = cleaned_prediction.split('>>')
        if len(parts) < 2 or not parts[1]:
            raise ValueError("ChemFM did not return a parsable reaction string.")
            
        predicted_reactants_smiles = parts[1].split('.')
        predicted_reactants_smiles = [smi for smi in predicted_reactants_smiles if Chem.MolFromSmiles(smi)]
        if not predicted_reactants_smiles:
            raise ValueError("ChemFM returned invalid reactant SMILES after parsing.")

    except Exception as e:
        logger.error(f"Error during ChemFM inference or parsing: {e}", exc_info=True)
        return {"error": f"ChemFM model failed to generate a valid retrosynthesis pathway. Details: {e}"}

    # 4. Construct the naked reaction and optimize yield
    naked_reaction_smiles = f"{'.'.join(predicted_reactants_smiles)}>>{target_smiles}"
    logger.info(f"Naked reaction for yield optimization: '{naked_reaction_smiles}'")
    
    yield_predictor = _get_yield_predictor()
    optimized_yield_data = None
    if yield_predictor:
        try:
            optimization_results = yield_predictor.predict([naked_reaction_smiles])
            if optimization_results and 'error' not in optimization_results[0]:
                optimized_yield_data = optimization_results[0]
                logger.info(f"Yield optimization successful: {optimized_yield_data}")
            else:
                logger.warning(f"Yield optimization failed or returned an error: {optimization_results}")
        except Exception as e:
            logger.error(f"An error occurred during yield optimization: {e}", exc_info=True)
    else:
        logger.info("Yield predictor not available, skipping optimization.")

    # 5. Calculate Molecular Properties for context
    target_properties = _calculate_properties(target_smiles)

    # 6. Use a general LLM to format the output and provide evaluation
    logger.info("Using LLM to elaborate and format the ChemFM prediction...")
    try:
        # This prompt is now enhanced with yield and condition data
        elaboration_prompt = format_prompt(
            "propose_chemfm_route",
            "synthesis_prompts.yaml",
            target_molecule_smiles=target_smiles,
            target_molecule_properties=json.dumps(target_properties),
            user_suggestion=user_suggestion,
            chemfm_predicted_reactants=json.dumps(predicted_reactants_smiles),
            # Pass the optimized data to the LLM for it to use in its response
            optimized_yield_info=json.dumps(optimized_yield_data) if optimized_yield_data else "Not available."
        )
        if not elaboration_prompt:
            return {"error": "Failed to format the 'propose_chemfm_route' prompt."}

        llm_response = generate_text(elaboration_prompt, temperature=0.5)
        if not llm_response:
            return {"error": "LLM failed to generate an elaboration for the route."}
        
        cleaned_response = llm_response.strip().replace('```json', '').replace('```', '').strip()
        new_route_data = json.loads(cleaned_response)

        # 7. Post-process the LLM output (add image URL, and fallback yield/conditions)
        if "steps" in new_route_data and new_route_data["steps"]:
            step = new_route_data["steps"][0]
            reaction_smiles_for_image = None

            # If we have optimized data, use it as the source of truth
            if optimized_yield_data:
                # Use the full reaction SMILES from the optimizer for the image
                reaction_smiles_for_image = optimized_yield_data.get('optimized_reaction_smiles')
                
                # As a fallback, if the LLM missed details, inject them here
                if 'yield' not in step or not step.get('yield'):
                    step['yield'] = float(optimized_yield_data.get('best_yield', '0.0%').replace('%', ''))
                
                if 'reagents_conditions' not in step or not step.get('reagents_conditions'):
                    conditions = optimized_yield_data.get('optimal_conditions', {})
                    conditions_parts = [
                        f"Catalyst: {conditions['catalyst']}" if conditions.get('catalyst') else None,
                        f"Reagents: {conditions['reagents']}" if conditions.get('reagents') else None,
                        f"Solvents: {conditions['solvents']}" if conditions.get('solvents') else None,
                        f"Temperature: {conditions['temperature']}" if conditions.get('temperature') else None,
                    ]
                    step['reagents_conditions'] = "; ".join(filter(None, conditions_parts)) or "Conditions not specified"

            # If no optimization data, use the original ChemFM prediction for the image
            else:
                reactants = [r['smiles'] for r in step.get('reactants', [])]
                product = step.get('product', {}).get('smiles')
                if reactants and product:
                    reaction_smiles_for_image = f"{'.'.join(reactants)}>>{product}"
            
            # Generate the reaction image using the best available reaction string
            if reaction_smiles_for_image:
                step_id = f"{new_route_data.get('id', 'hypothetical')}_step_1"
                image_url = generate_reaction_image(reaction_smiles_for_image, step_id)
                step["reaction_image_url"] = image_url

        logger.info("Successfully generated and formatted a new route using ChemFM and LLM.")
        return {"new_route": new_route_data}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode LLM response into JSON. Response was: {llm_response}")
        return {"error": f"The LLM returned a malformed JSON object. Please try again. Details: {e}"}
    except Exception as e:
        logger.error(f"An error occurred during LLM elaboration: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while finalizing the route. Details: {e}"}