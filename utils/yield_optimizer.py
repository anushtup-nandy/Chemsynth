import os
import sys
import pandas as pd
import numpy as np
import torch
from rdkit import Chem, RDLogger
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from models.packages.rcr_torch_version.baseline_condition_model import NeuralNetContextRecommender
from models.packages.yield_predictor.models import SmilesClassificationModel
from torch.nn import MSELoss 

class SmilesClassificationModelSelf(SmilesClassificationModel):
    """A simple wrapper to handle the specific loading arguments we need."""
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False, verbose=True, silent=False):
        return super().load_and_cache_examples(examples, evaluate, no_cache, multi_label, verbose, silent)

# --- Helper Functions ---
def canonicalize_smi(smi):
    """Generates a canonical SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return ""

def merge_reaction_with_conditions(naked_reaction: str, conditions: dict):
    """
    Combines a naked reaction (A>>B) with a dictionary of predicted conditions.
    """
    reactants, products = naked_reaction.split('>>')
    
    all_reactants = [reactants]
    for key in ['solvent', 'reagent', 'catalyst']:
        if conditions[key] and conditions[key] != '':
            all_reactants.append(conditions[key])
            
    # Create the full reaction string with canonicalized reactants
    full_reactant_string = ".".join(sorted(all_reactants))
    canonical_reactant_string = canonicalize_smi(full_reactant_string)
    
    if not canonical_reactant_string:
        return None # Failed to process reactants
        
    return f"{canonical_reactant_string}>>{products}"


class AdvancedYieldPredictor:
    """
    An orchestrator that predicts reaction conditions and then predicts the yield
    for the most promising conditions.
    """
    def __init__(self, condition_model_dir: str, yield_model_dir: str, cuda_device: int = -1):
        """
        Initializes the predictor by loading the condition and yield models.
        
        Args:
            condition_model_dir (str): Path to the 'model 2' directory (RCR model).
            yield_model_dir (str): Path to the 'model' directory (Transformer model).
            cuda_device (int): The GPU device to use (-1 for CPU).
        """
        print("--- Initializing Advanced Yield Predictor ---")
        self.condition_model_dir = condition_model_dir
        self.yield_model_dir = yield_model_dir
        
        # Determine if CUDA is available and should be used
        use_cuda = torch.mps.is_available() and cuda_device != -1

        # 1. Load the Condition Predictor (RCR Model)
        print(f"Loading Condition Predictor from: {self.condition_model_dir}")
        self.condition_predictor = NeuralNetContextRecommender()
        self.condition_predictor.load_nn_model(
            info_path=self.condition_model_dir,
            weights_path=os.path.join(self.condition_model_dir, 'dict_weights.npy')
        )
        print("Condition Predictor loaded successfully.")

        # 2. Load the Yield Predictor (Transformer Model)
        print(f"Loading Yield Predictor from: {self.yield_model_dir}")
        self.yield_predictor = SmilesClassificationModelSelf(
            "bert", 
            self.yield_model_dir, 
            num_labels=1, 
            use_cuda=use_cuda, 
            cuda_device=cuda_device,
            args={'silent': True}
        )
        self.yield_predictor.loss_fct = MSELoss()
        # Constants for de-normalizing the yield prediction
        self.yield_mean = 79.29119663076209
        self.yield_std = 18.858441890553195
        print("Yield Predictor loaded successfully.")
        print("--- Initialization Complete ---\n")

    def predict(self, naked_reaction_list: list, top_n_conditions: int = 10):
        """
        Takes a list of naked reactions (A>>B), predicts conditions, and finds the
        combination with the highest predicted yield.

        Args:
            naked_reaction_list (list): A list of reaction SMILES strings (e.g., ['A.B>>C']).
            top_n_conditions (int): The number of condition sets to evaluate per reaction.

        Returns:
            list: A list of dictionaries, one for each input reaction, detailing the
                  best predicted conditions and the resulting yield.
        """
        final_results = []
        
        print(f"Processing {len(naked_reaction_list)} reaction(s)...")

        for i, naked_rxn in enumerate(naked_reaction_list):
            print(f"\nAnalyzing Reaction {i+1}: {naked_rxn}")

            # Step 1: Predict top N conditions using the RCR model
            try:
                predicted_conditions, condition_scores = self.condition_predictor.get_n_conditions(
                    naked_rxn, n=top_n_conditions, return_scores=True
                )
                if not predicted_conditions:
                    print("  - Could not predict conditions for this reaction.")
                    final_results.append({'input_reaction': naked_rxn, 'error': 'Condition prediction failed.'})
                    continue
            except Exception as e:
                print(f"  - Error during condition prediction: {e}")
                final_results.append({'input_reaction': naked_rxn, 'error': str(e)})
                continue

            # Step 2: Augment reactions with predicted conditions
            augmented_reactions = []
            condition_details = []
            for conditions in predicted_conditions:
                temp, solvent, reagent, catalyst, _, _ = conditions
                condition_dict = {'solvent': solvent, 'reagent': reagent, 'catalyst': catalyst, 'temperature': temp}
                
                full_rxn_smiles = merge_reaction_with_conditions(naked_rxn, condition_dict)
                if full_rxn_smiles:
                    augmented_reactions.append(full_rxn_smiles)
                    condition_details.append(condition_dict)
            
            if not augmented_reactions:
                print("  - Could not generate valid full reactions from predicted conditions.")
                final_results.append({'input_reaction': naked_rxn, 'error': 'Reaction augmentation failed.'})
                continue

            print(f"  - Generated {len(augmented_reactions)} potential full reactions.")

            # Step 3: Predict yield for all augmented reactions in a batch
            raw_yields = self.yield_predictor.predict(augmented_reactions)[0]
            
            # De-normalize and clip yields
            scaled_yields = (raw_yields * self.yield_std) + self.yield_mean
            predicted_yields = np.clip(scaled_yields, 0, 100)

            # Step 4: Find the best condition set (highest yield)
            best_yield_index = np.argmax(predicted_yields)
            best_yield = predicted_yields[best_yield_index]
            best_conditions = condition_details[best_yield_index]
            best_full_reaction = augmented_reactions[best_yield_index]

            print(f"  - Optimal yield found: {best_yield:.2f}%")
            
            # Step 5: Collate results
            result = {
                'input_reaction': naked_rxn,
                'best_yield': f"{best_yield:.2f}%",
                'optimal_conditions': {
                    'catalyst': best_conditions['catalyst'],
                    'reagents': best_conditions['reagent'],
                    'solvents': best_conditions['solvent'],
                    'temperature': f"{best_conditions['temperature']:.1f}°C"
                },
                'optimized_reaction_smiles': best_full_reaction
            }
            final_results.append(result)

        return final_results

if __name__ == '__main__':
    # Define paths relative to the script location
    this_dir = os.path.dirname(__file__)
    CONDITION_MODEL_PATH = '/Users/anushtupnandy/Documents/projects/git-repos/chemsynth/chem_mvp/models/model 2'
    YIELD_MODEL_PATH = '/Users/anushtupnandy/Documents/projects/git-repos/chemsynth/chem_mvp/models/model'
    # --- Initialize the predictor ---
    predictor = AdvancedYieldPredictor(
        condition_model_dir=CONDITION_MODEL_PATH,
        yield_model_dir=YIELD_MODEL_PATH,
        cuda_device=-1  # Use -1 for CPU, or 0, 1, etc., for a specific GPU
    )

    # --- Define some test reactions (without conditions) ---
    test_reactions = [
            # Condition 1: In-situ Grignard formation
            "Cl[Si](Cl)(Cl)Cl >> Cl[Si](Cl)(C(C)(C)C)C(C)(C)C",
            
            # # Condition 2: Pre-formed Grignard reagent
            # "Cl[Si](Cl)(Cl)Cl.C(C)(C)[Mg]Cl>>CC(C)(C)[Si](Cl)(C(C)(C)C)C(C)(C)C",
            
            # # Condition 3: Pre-formed Grignard with additives
            # "Cl[Si](Cl)(Cl)Cl.C(C)(C)[Mg]Cl.[Li]Cl.[Cu]Br>>CC(C)(C)[Si](Cl)(C(C)(C)C)C(C)(C)C"
        ]

    optimized_results = predictor.predict(test_reactions)

    import json
    print("\n--- OPTIMIZATION RESULTS ---")
    print(json.dumps(optimized_results, indent=2))