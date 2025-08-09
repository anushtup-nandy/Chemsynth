import os
import logging
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any

from baybe import Campaign
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
from baybe.recommenders import BotorchRecommender, RandomRecommender
from baybe.surrogates import GaussianProcessSurrogate
from baybe.acquisition import ExpectedImprovement, UpperConfidenceBound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from utils.yield_optimizer import AdvancedYieldPredictor, merge_reaction_with_conditions
except ImportError:
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.yield_optimizer import AdvancedYieldPredictor, merge_reaction_with_conditions

ORACLE = None

def safe_predict_yield(oracle: AdvancedYieldPredictor, full_reaction_smiles: str) -> float:
    """
    Safely predict yield based on the actual format returned by yield_optimizer.py
    """
    try:
        if not full_reaction_smiles or not full_reaction_smiles.strip():
            logger.warning("Empty reaction SMILES provided")
            return 0.0
            
        logger.debug(f"Predicting yield for: {full_reaction_smiles}")
        
        prediction_result = oracle.yield_predictor.predict([full_reaction_smiles])
        logger.debug(f"Raw prediction result: {prediction_result}")
        
        raw_yield = None
        
        if isinstance(prediction_result, (tuple, list)) and len(prediction_result) >= 1:
            first_element = prediction_result[0]
            
            if isinstance(first_element, np.ndarray):
                if first_element.size > 0:
                    raw_yield = first_element.item()
                else:
                    logger.warning("Empty numpy array in prediction")
                    return 0.0
            elif hasattr(first_element, 'item'):
                raw_yield = first_element.item()
            elif isinstance(first_element, (int, float)):
                raw_yield = first_element
            else:
                logger.warning(f"Unexpected first element type: {type(first_element)}")
                return 0.0
        else:
            logger.warning(f"Unexpected prediction result format: {type(prediction_result)}")
            return 0.0
        
        if raw_yield is None:
            logger.warning("Could not extract raw yield from prediction")
            return 0.0
            
        # De-normalize using oracle's constants
        scaled_yield = (raw_yield * oracle.yield_std) + oracle.yield_mean
        final_yield = max(0.0, min(100.0, float(scaled_yield)))
        
        return final_yield
        
    except Exception as e:
        logger.error(f"Yield prediction failed for '{full_reaction_smiles}': {e}", exc_info=True)
        return 0.0

def create_bayesian_campaign(searchspace: SearchSpace, use_random_for_initial: bool = True) -> Optional[Campaign]:
    """
    Create a proper Bayesian optimization campaign.
    """
    target = NumericalTarget(name="Yield", mode="MAX", bounds=(0, 100))
    objective = SingleTargetObjective(target)
    
    try:
        if use_random_for_initial:
            # Start with random recommender for initial exploration
            recommender = RandomRecommender()
            logger.info("Created campaign with Random recommender for initial sampling")
        else:
            # Use Bayesian recommender
            recommender = BotorchRecommender(
                surrogate_model=GaussianProcessSurrogate(),
                acquisition_function=UpperConfidenceBound(beta=2.0)  # More exploration
            )
            logger.info("Created campaign with Bayesian recommender")
        
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=recommender
        )
        
        return campaign
        
    except Exception as e:
        logger.error(f"Failed to create campaign: {e}")
        return None

def switch_to_bayesian_recommender(campaign: Campaign) -> bool:
    """
    Switch the campaign to use Bayesian optimization after initial random sampling.
    """
    try:
        # Create new Bayesian recommender
        bayesian_recommender = BotorchRecommender(
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function=UpperConfidenceBound(beta=2.0)
        )
        
        # Create new campaign with same search space and objective but new recommender
        new_campaign = Campaign(
            searchspace=campaign.searchspace,
            objective=campaign.objective,
            recommender=bayesian_recommender
        )
        
        # Transfer all previous measurements to new campaign
        if hasattr(campaign, 'measurements') and not campaign.measurements.empty:
            new_campaign.add_measurements(campaign.measurements)
            logger.info(f"Transferred {len(campaign.measurements)} previous measurements to Bayesian campaign")
        
        # Replace the campaign object's internals
        campaign._recommender = bayesian_recommender
        campaign.measurements = new_campaign.measurements if hasattr(new_campaign, 'measurements') else campaign.measurements
        
        logger.info("Successfully switched to Bayesian recommender")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to switch to Bayesian recommender: {e}")
        return False

def initialize_oracle() -> Optional[AdvancedYieldPredictor]:
    """
    Initializes the AdvancedYieldPredictor.
    """
    global ORACLE
    if ORACLE is None:
        logger.info("Initializing the simulation oracle (AdvancedYieldPredictor)...")
        try:
            this_dir = os.path.dirname(__file__)
            condition_model_path = os.path.abspath(os.path.join(this_dir, '..', 'models', 'model 2'))
            yield_model_path = os.path.abspath(os.path.join(this_dir, '..', 'models', 'model'))
            
            if not os.path.exists(condition_model_path):
                logger.error(f"Condition model path does not exist: {condition_model_path}")
                return None
            if not os.path.exists(yield_model_path):
                logger.error(f"Yield model path does not exist: {yield_model_path}")
                return None

            ORACLE = AdvancedYieldPredictor(
                condition_model_dir=condition_model_path,
                yield_model_dir=yield_model_path,
                cuda_device=-1
            )
            
            logger.info("Oracle initialized successfully.")
            return ORACLE
            
        except Exception as e:
            logger.error(f"Failed to initialize the oracle: {e}", exc_info=True)
            return None
    return ORACLE

def run_true_bayesian_optimization(
    naked_reaction_smiles: str,
    num_initial_candidates: int = 20,
    num_iterations: int = 15,
    num_random_init: int = 5
) -> Dict[str, Any]:
    """
    Run TRUE Bayesian optimization with explicit phase switching.
    """
    logger.info(f"Starting TRUE Bayesian Optimization for: {naked_reaction_smiles}")
    
    if not naked_reaction_smiles or not naked_reaction_smiles.strip():
        return {"error": "Invalid or empty reaction SMILES provided"}
    
    # 1. Initialize Oracle
    oracle = initialize_oracle()
    if oracle is None:
        return {"error": "Could not initialize the simulation oracle"}

    # 2. Generate Search Space
    try:
        logger.info(f"Generating {num_initial_candidates} initial candidate conditions...")
        initial_conditions, _ = oracle.condition_predictor.get_n_conditions(
            naked_reaction_smiles, n=num_initial_candidates, return_scores=True
        )
        
        if not initial_conditions:
            return {"error": "Could not generate initial candidate conditions"}

        # Process conditions
        temps = []
        solvents = set()
        reagents = set()
        catalysts = set()
        
        for cond in initial_conditions:
            if len(cond) >= 4:
                temp, solvent, reagent, catalyst = cond[0], cond[1], cond[2], cond[3]
                
                if temp is not None and isinstance(temp, (int, float)):
                    temps.append(float(temp))
                if solvent and isinstance(solvent, str) and solvent.strip():
                    solvents.add(solvent.strip())
                if reagent and isinstance(reagent, str) and reagent.strip():
                    reagents.add(reagent.strip())
                if catalyst and isinstance(catalyst, str) and catalyst.strip():
                    catalysts.add(catalyst.strip())
        
        # Convert to sorted lists with fallbacks
        solvents = sorted(list(solvents)) or ["THF", "DCM"]
        reagents = sorted(list(reagents)) or ["None"]
        catalysts = sorted(list(catalysts)) or ["None"]
        
        if not temps:
            temps = [20.0, 80.0]
        
        temp_min, temp_max = min(temps), max(temps)
        if temp_max - temp_min < 5.0:
            temp_max = temp_min + 20.0
        
        logger.info(f"Search space: {len(solvents)} solvents, {len(reagents)} reagents, "
                   f"{len(catalysts)} catalysts, temp [{temp_min:.1f}, {temp_max:.1f}]°C")

    except Exception as e:
        logger.error(f"Error generating search space: {e}", exc_info=True)
        return {"error": f"Failed to generate search space: {e}"}

    # 3. Create BayBE Parameters and Search Space
    try:
        parameters = []
        
        if len(solvents) > 1:
            parameters.append(CategoricalParameter(name="Solvent", values=solvents, encoding="OHE"))
        if len(reagents) > 1:
            parameters.append(CategoricalParameter(name="Reagent", values=reagents, encoding="OHE"))
        if len(catalysts) > 1:
            parameters.append(CategoricalParameter(name="Catalyst", values=catalysts, encoding="OHE"))
            
        parameters.append(NumericalContinuousParameter(
            name="Temperature",
            bounds=(temp_min, temp_max)
        ))

        searchspace = SearchSpace.from_product(parameters)

    except Exception as e:
        logger.error(f"Error creating search space: {e}", exc_info=True)
        return {"error": f"Failed to create BayBE search space: {e}"}

    # 4. Phase 1: Random Exploration
    logger.info(f"Phase 1: Random exploration ({num_random_init} iterations)")
    random_campaign = create_bayesian_campaign(searchspace, use_random_for_initial=True)
    if random_campaign is None:
        return {"error": "Failed to create random exploration campaign"}

    optimization_history = []
    
    # Random exploration phase
    for i in range(num_random_init):
        try:
            recommendations = random_campaign.recommend(batch_size=1)
            
            if recommendations.empty:
                logger.warning(f"No recommendations at iteration {i+1}")
                continue
                
            rec = recommendations.iloc[0]
            
            # Extract parameters
            temp = rec['Temperature']
            solvent = rec.get('Solvent', solvents[0])
            reagent = rec.get('Reagent', reagents[0])
            catalyst = rec.get('Catalyst', catalysts[0])
            
            # Create reaction SMILES
            condition_dict = {
                'solvent': solvent,
                'reagent': reagent if reagent != "None" else "",
                'catalyst': catalyst if catalyst != "None" else ""
            }
            
            full_reaction_smiles = merge_reaction_with_conditions(naked_reaction_smiles, condition_dict)
            yield_value = safe_predict_yield(oracle, full_reaction_smiles)
            
            logger.info(f"RANDOM {i+1}/{num_random_init} | "
                       f"[T:{temp:.1f}°C, S:{solvent}, R:{reagent}, C:{catalyst}] -> "
                       f"Yield: {yield_value:.2f}%")
            
            # Add measurement
            measurements = recommendations.copy()
            measurements["Yield"] = yield_value
            random_campaign.add_measurements(measurements)
            
            optimization_history.append({
                "iteration": i + 1,
                "phase": "random_exploration",
                "conditions": {
                    "Temperature": round(float(temp), 2),
                    "Solvent": solvent,
                    "Reagent": reagent,
                    "Catalyst": catalyst,
                },
                "yield": round(float(yield_value), 2)
            })
            
        except Exception as e:
            logger.warning(f"Random iteration {i+1} failed: {e}")
            continue

    # 5. Phase 2: Bayesian Optimization
    if len(optimization_history) < 2:
        return {"error": "Insufficient random samples for Bayesian optimization"}
    
    logger.info(f"Phase 2: Bayesian optimization ({num_iterations - num_random_init} iterations)")
    
    # Create Bayesian campaign
    bayesian_campaign = create_bayesian_campaign(searchspace, use_random_for_initial=False)
    if bayesian_campaign is None:
        return {"error": "Failed to create Bayesian campaign"}
    
    # Transfer random exploration data to Bayesian campaign
    try:
        if hasattr(random_campaign, 'measurements') and not random_campaign.measurements.empty:
            bayesian_campaign.add_measurements(random_campaign.measurements)
            logger.info(f"Transferred {len(random_campaign.measurements)} random samples to Bayesian campaign")
    except Exception as e:
        logger.warning(f"Failed to transfer measurements: {e}")
    
    # Bayesian optimization phase
    for i in range(num_random_init, num_iterations):
        try:
            recommendations = bayesian_campaign.recommend(batch_size=1)
            
            if recommendations.empty:
                logger.warning(f"No recommendations at iteration {i+1}")
                continue
                
            rec = recommendations.iloc[0]
            
            # Extract parameters
            temp = rec['Temperature']
            solvent = rec.get('Solvent', solvents[0])
            reagent = rec.get('Reagent', reagents[0])
            catalyst = rec.get('Catalyst', catalysts[0])
            
            # Create reaction SMILES
            condition_dict = {
                'solvent': solvent,
                'reagent': reagent if reagent != "None" else "",
                'catalyst': catalyst if catalyst != "None" else ""
            }
            
            full_reaction_smiles = merge_reaction_with_conditions(naked_reaction_smiles, condition_dict)
            yield_value = safe_predict_yield(oracle, full_reaction_smiles)
            
            logger.info(f"BAYESIAN {i+1-num_random_init}/{num_iterations-num_random_init} | "
                       f"[T:{temp:.1f}°C, S:{solvent}, R:{reagent}, C:{catalyst}] -> "
                       f"Yield: {yield_value:.2f}%")
            
            # Add measurement
            measurements = recommendations.copy()
            measurements["Yield"] = yield_value
            bayesian_campaign.add_measurements(measurements)
            
            optimization_history.append({
                "iteration": i + 1,
                "phase": "bayesian_optimization",
                "conditions": {
                    "Temperature": round(float(temp), 2),
                    "Solvent": solvent,
                    "Reagent": reagent,
                    "Catalyst": catalyst,
                },
                "yield": round(float(yield_value), 2)
            })
            
        except Exception as e:
            logger.warning(f"Bayesian iteration {i+1} failed: {e}")
            continue

    # 6. Process Results
    if not optimization_history:
        return {"error": "No successful optimization iterations"}
    
    try:
        best_result = max(optimization_history, key=lambda x: x['yield'])
        
        # Calculate improvement metrics
        random_yields = [h['yield'] for h in optimization_history if h['phase'] == 'random_exploration']
        bayesian_yields = [h['yield'] for h in optimization_history if h['phase'] == 'bayesian_optimization']
        
        avg_random_yield = np.mean(random_yields) if random_yields else 0
        avg_bayesian_yield = np.mean(bayesian_yields) if bayesian_yields else 0
        
        return {
            "best_conditions": best_result['conditions'],
            "max_yield": best_result['yield'],
            "optimization_history": optimization_history,
            "search_space_summary": {
                "solvents": solvents,
                "reagents": reagents,
                "catalysts": catalysts,
                "temperature_range": [round(temp_min, 1), round(temp_max, 1)]
            },
            "performance_metrics": {
                "total_evaluations": len(optimization_history),
                "random_phase_evaluations": len(random_yields),
                "bayesian_phase_evaluations": len(bayesian_yields),
                "avg_random_yield": round(avg_random_yield, 2),
                "avg_bayesian_yield": round(avg_bayesian_yield, 2),
                "improvement_over_random": round(avg_bayesian_yield - avg_random_yield, 2) if bayesian_yields else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing results: {e}", exc_info=True)
        return {"error": f"Failed to process results: {e}"}

# Test function
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    test_reaction = "Cl[Si](Cl)(Cl)Cl.c1ccccc1[Mg]Br>>Cl[Si](Cl)(Cl)c1ccccc1"
    
    print("\n" + "="*60)
    print(f"RUNNING TRUE BAYESIAN OPTIMIZATION FOR:\n{test_reaction}")
    print("="*60 + "\n")
    
    results = run_true_bayesian_optimization(
        naked_reaction_smiles=test_reaction,
        num_initial_candidates=20,
        num_iterations=12,
        num_random_init=5  # 5 random, then 7 Bayesian
    )

    import json
    print(json.dumps(results, indent=2))