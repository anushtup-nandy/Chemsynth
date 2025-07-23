# core_logic/chemfm_synthesis_engine.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import json
import logging

from utils.llm_interface import generate_text
from utils.prompt_loader import format_prompt
from utils.reaction_utils import generate_reaction_image

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Caching ---
# We cache the model and tokenizer in memory to avoid reloading them on every request.
_chemfm_model = None
_chemfm_tokenizer = None
_device = "mps" if torch.mps.is_available() else "cpu" 
# _device = "cuda" if torch.cuda.is_available() else "cpu" 

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
            _chemfm_model.eval() # Set model to evaluation mode
            logger.info(f"ChemFM model '{model_name}' loaded successfully on device '{_device}'.")
        except Exception as e:
            logger.error(f"Failed to load ChemFM model: {e}", exc_info=True)
            raise RuntimeError(f"Could not load ChemFM model. Please check your internet connection and Hugging Face credentials. Error: {e}")
    return _chemfm_model, _chemfm_tokenizer

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
    Generates a new, single-step synthesis route using ChemFM, guided by user input,
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
    # This prompt asks the model to predict the reactants for the given product.
    retrosynthesis_prompt = f"Product>>{target_smiles}"
    
    logger.info(f"ChemFM prompt for retrosynthesis: '{retrosynthesis_prompt}'")

    # 3. Perform inference with ChemFM
    try:
        # MODIFIED: Tokenize without adding special tokens, which can sometimes confuse generation.
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
            raise ValueError("ChemFM did not return a parsable reaction string after cleaning.")
            
        predicted_reactants_smiles = parts[1].split('.')
        
        predicted_reactants_smiles = [smi for smi in predicted_reactants_smiles if Chem.MolFromSmiles(smi)]
        if not predicted_reactants_smiles:
            raise ValueError("ChemFM returned invalid reactant SMILES after cleaning and parsing.")

    except Exception as e:
        logger.error(f"Error during ChemFM inference or parsing: {e}", exc_info=True)
        return {"error": f"ChemFM model failed to generate a valid retrosynthesis pathway. Details: {e}"}

    # 4. Calculate Molecular Properties for context
    target_properties = _calculate_properties(target_smiles)

    # 5. Use a general LLM to format the output and provide evaluation
    # This step translates the raw SMILES from ChemFM into the rich JSON format the frontend needs.
    logger.info("Using LLM to elaborate and format the ChemFM prediction...")
    try:
        # We need a new prompt that instructs the LLM on how to use the ChemFM output
        elaboration_prompt = format_prompt(
            "propose_chemfm_route",  # This is a NEW prompt template you must add
            "synthesis_prompts.yaml",
            target_molecule_smiles=target_smiles,
            target_molecule_properties=json.dumps(target_properties),
            user_suggestion=user_suggestion,
            chemfm_predicted_reactants=json.dumps(predicted_reactants_smiles)
        )
        if not elaboration_prompt:
            return {"error": "Failed to format the 'propose_chemfm_route' prompt."}

        llm_response = generate_text(elaboration_prompt, temperature=0.5)
        if not llm_response:
            return {"error": "LLM failed to generate an elaboration for the route."}
        
        # Clean and parse the JSON output from the LLM
        cleaned_response = llm_response.strip().replace('```json', '').replace('```', '').strip()
        new_route_data = json.loads(cleaned_response)

        # 6. Post-process the LLM output (add image URL, etc.)
        if "steps" in new_route_data and new_route_data["steps"]:
            step = new_route_data["steps"][0]
            reactants = [r['smiles'] for r in step.get('reactants', [])]
            product = step.get('product', {}).get('smiles')
            
            if reactants and product:
                reaction_smiles_str = f"{'.'.join(reactants)}>>{product}"
                step_id = f"{new_route_data.get('id', 'hypothetical')}_step_1"
                # Generate the reaction image using your existing utility
                image_url = generate_reaction_image(reaction_smiles_str, step_id)
                new_route_data["steps"][0]["reaction_image_url"] = image_url

        logger.info("Successfully generated and formatted a new route using ChemFM and LLM.")
        return {"new_route": new_route_data}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode LLM response into JSON. Response was: {llm_response}")
        return {"error": f"The LLM returned a malformed JSON object. Please try again. Details: {e}"}
    except Exception as e:
        logger.error(f"An error occurred during LLM elaboration: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while finalizing the route. Details: {e}"}