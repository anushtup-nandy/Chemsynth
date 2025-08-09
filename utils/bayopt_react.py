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
from baybe.acquisition import (
    ExpectedImprovement, 
    UpperConfidenceBound, 
    ProbabilityOfImprovement,
    qExpectedImprovement,  # For batch acquisition
    qUpperConfidenceBound   # For batch acquisition
)

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

def create_bayesian_campaign(searchspace: SearchSpace, use_random_for_initial: bool = True, 
                           acquisition_type: str = "EI", batch_size: int = 1) -> Optional[Campaign]:
    """
    Create a Bayesian optimization campaign with advanced acquisition functions.
    
    Args:
        acquisition_type: "EI", "UCB", "PI", "qEI", "qUCB" for different strategies
        batch_size: Number of parallel recommendations (>1 enables batch optimization)
    """
    target = NumericalTarget(name="Yield", mode="MAX", bounds=(0, 100))
    objective = SingleTargetObjective(target)
    
    try:
        if use_random_for_initial:
            recommender = RandomRecommender()
            logger.info("Created campaign with Random recommender for initial sampling")
        else:
            # Advanced acquisition function selection
            if acquisition_type == "EI":
                acq_func = ExpectedImprovement()
            elif acquisition_type == "UCB":
                acq_func = UpperConfidenceBound(beta=2.0)
            elif acquisition_type == "PI":
                acq_func = ProbabilityOfImprovement()
            elif acquisition_type == "qEI" and batch_size > 1:
                acq_func = qExpectedImprovement()
            elif acquisition_type == "qUCB" and batch_size > 1:
                acq_func = qUpperConfidenceBound(beta=2.0)
            else:
                # Fallback to EI for invalid combinations
                acq_func = ExpectedImprovement()
                logger.warning(f"Invalid acquisition type '{acquisition_type}' for batch_size {batch_size}, using EI")
            
            recommender = BotorchRecommender(
                surrogate_model=GaussianProcessSurrogate(),
                acquisition_function=acq_func
            )
            logger.info(f"Created campaign with {acquisition_type} acquisition function, batch_size={batch_size}")
        
        campaign = Campaign(
            searchspace=searchspace,
            objective=objective,
            recommender=recommender
        )
        
        return campaign
        
    except Exception as e:
        logger.error(f"Failed to create campaign: {e}")
        return None

def switch_to_bayesian_recommender(campaign: Campaign, acquisition_type: str = "EI", 
                                 batch_size: int = 1) -> bool:
    """
    Switch to Bayesian optimization with specified acquisition function and batch size.
    """
    try:
        # Select acquisition function
        if acquisition_type == "EI":
            acq_func = ExpectedImprovement()
        elif acquisition_type == "UCB":
            acq_func = UpperConfidenceBound(beta=2.0)
        elif acquisition_type == "PI":
            acq_func = ProbabilityOfImprovement()
        elif acquisition_type == "qEI" and batch_size > 1:
            acq_func = qExpectedImprovement()
        elif acquisition_type == "qUCB" and batch_size > 1:
            acq_func = qUpperConfidenceBound(beta=2.0)
        else:
            acq_func = ExpectedImprovement()
            logger.warning(f"Using EI fallback for {acquisition_type} with batch_size {batch_size}")
        
        bayesian_recommender = BotorchRecommender(
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function=acq_func
        )
        
        new_campaign = Campaign(
            searchspace=campaign.searchspace,
            objective=campaign.objective,
            recommender=bayesian_recommender
        )
        
        if hasattr(campaign, 'measurements') and not campaign.measurements.empty:
            new_campaign.add_measurements(campaign.measurements)
            logger.info(f"Transferred {len(campaign.measurements)} measurements to {acquisition_type} campaign")
        
        campaign._recommender = bayesian_recommender
        campaign.measurements = new_campaign.measurements if hasattr(new_campaign, 'measurements') else campaign.measurements
        
        logger.info(f"Switched to {acquisition_type} acquisition with batch_size={batch_size}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to switch to Bayesian recommender: {e}")
        return False
    
def get_adaptive_acquisition_strategy(iteration: int, total_iterations: int, 
                                    current_best_yield: float) -> Tuple[str, int]:
    """
    Dynamically select acquisition function and batch size based on optimization progress.
    
    Returns: (acquisition_type, batch_size)
    """
    progress = iteration / total_iterations
    
    # Early stage: exploration with batch sampling
    if progress < 0.3:
        return ("qEI", 2)  # Batch Expected Improvement for exploration
    # Middle stage: balanced exploration-exploitation
    elif progress < 0.7:
        if current_best_yield < 50:  # Low yields suggest need for more exploration
            return ("UCB", 1)  # Upper Confidence Bound for exploration
        else:
            return ("EI", 1)   # Expected Improvement for balanced approach
    # Late stage: exploitation
    else:
        return ("PI", 1)

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
    num_random_init: int = 5,
    use_adaptive_strategy: bool = True,
    initial_batch_size: int = 2
) -> Dict[str, Any]:
    """
    Enhanced Bayesian optimization with advanced acquisition functions and batch optimization.
    """
    logger.info(f"Starting ADVANCED Bayesian Optimization for: {naked_reaction_smiles}")
    
    if not naked_reaction_smiles or not naked_reaction_smiles.strip():
        return {"error": "Invalid or empty reaction SMILES provided"}
    
    # Initialize Oracle (unchanged)
    oracle = initialize_oracle()
    if oracle is None:
        return {"error": "Could not initialize the simulation oracle"}

    # Generate Search Space (unchanged section)
    try:
        logger.info(f"Generating {num_initial_candidates} initial candidate conditions...")
        initial_conditions, _ = oracle.condition_predictor.get_n_conditions(
            naked_reaction_smiles, n=num_initial_candidates, return_scores=True
        )
        
        if not initial_conditions:
            return {"error": "Could not generate initial candidate conditions"}

        # Process conditions (unchanged)
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
        
        solvents = sorted(list(solvents)) or ["THF", "DCM"]
        reagents = sorted(list(reagents)) or ["None"]
        catalysts = sorted(list(catalysts)) or ["None"]
        
        if not temps:
            temps = [20.0, 80.0]
        
        temp_min, temp_max = min(temps), max(temps)
        if temp_max - temp_min < 5.0:
            temp_max = temp_min + 20.0

    except Exception as e:
        logger.error(f"Error generating search space: {e}", exc_info=True)
        return {"error": f"Failed to generate search space: {e}"}

    # Create Search Space (unchanged)
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

    # ENHANCED Phase 1: Random Exploration with Batch Sampling
    logger.info(f"Phase 1: Batch random exploration ({num_random_init} iterations)")
    random_campaign = create_bayesian_campaign(searchspace, use_random_for_initial=True, batch_size=initial_batch_size)
    if random_campaign is None:
        return {"error": "Failed to create random exploration campaign"}

    optimization_history = []
    current_best_yield = 0.0
    
    # Random exploration with batch evaluation
    for i in range(num_random_init):
        try:
            batch_size = min(initial_batch_size, max(1, (num_random_init - i)))  # Reduce batch size near end
            recommendations = random_campaign.recommend(batch_size=batch_size)
            
            if recommendations.empty:
                logger.warning(f"No recommendations at iteration {i+1}")
                continue
            
            batch_yields = []
            batch_measurements = recommendations.copy()
            
            # Evaluate entire batch
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                temp = rec['Temperature']
                solvent = rec.get('Solvent', solvents[0])
                reagent = rec.get('Reagent', reagents[0])
                catalyst = rec.get('Catalyst', catalysts[0])
                
                condition_dict = {
                    'solvent': solvent,
                    'reagent': reagent if reagent != "None" else "",
                    'catalyst': catalyst if catalyst != "None" else ""
                }
                
                full_reaction_smiles = merge_reaction_with_conditions(naked_reaction_smiles, condition_dict)
                yield_value = safe_predict_yield(oracle, full_reaction_smiles)
                batch_yields.append(yield_value)
                
                current_best_yield = max(current_best_yield, yield_value)
                
                optimization_history.append({
                    "iteration": i + 1,
                    "batch_index": idx,
                    "phase": "random_exploration",
                    "conditions": {
                        "Temperature": round(float(temp), 2),
                        "Solvent": solvent,
                        "Reagent": reagent,
                        "Catalyst": catalyst,
                    },
                    "yield": round(float(yield_value), 2)
                })
            
            # Add all batch measurements at once
            batch_measurements["Yield"] = batch_yields
            random_campaign.add_measurements(batch_measurements)
            
            logger.info(f"BATCH RANDOM {i+1}/{num_random_init} | "
                       f"Batch size: {len(batch_yields)} | "
                       f"Best in batch: {max(batch_yields):.2f}% | "
                       f"Overall best: {current_best_yield:.2f}%")
            
        except Exception as e:
            logger.warning(f"Random batch iteration {i+1} failed: {e}")
            continue

    # ENHANCED Phase 2: Adaptive Bayesian Optimization
    if len(optimization_history) < 2:
        return {"error": "Insufficient random samples for Bayesian optimization"}
    
    logger.info(f"Phase 2: Adaptive Bayesian optimization ({num_iterations - num_random_init} iterations)")
    
    # Start with initial acquisition strategy
    current_acquisition = "EI"
    current_batch_size = 1
    
    bayesian_campaign = create_bayesian_campaign(searchspace, use_random_for_initial=False, 
                                                acquisition_type=current_acquisition, batch_size=current_batch_size)
    if bayesian_campaign is None:
        return {"error": "Failed to create Bayesian campaign"}
    
    # Transfer data
    try:
        if hasattr(random_campaign, 'measurements') and not random_campaign.measurements.empty:
            bayesian_campaign.add_measurements(random_campaign.measurements)
            logger.info(f"Transferred {len(random_campaign.measurements)} random samples")
    except Exception as e:
        logger.warning(f"Failed to transfer measurements: {e}")
    
    # Adaptive Bayesian optimization phase
    for i in range(num_random_init, num_iterations):
        try:
            # Adaptive strategy selection
            if use_adaptive_strategy:
                new_acquisition, new_batch_size = get_adaptive_acquisition_strategy(
                    i - num_random_init + 1, num_iterations - num_random_init, current_best_yield
                )
                
                # Switch strategy if needed
                if new_acquisition != current_acquisition or new_batch_size != current_batch_size:
                    logger.info(f"Switching to {new_acquisition} with batch_size={new_batch_size}")
                    switch_success = switch_to_bayesian_recommender(
                        bayesian_campaign, new_acquisition, new_batch_size
                    )
                    if switch_success:
                        current_acquisition = new_acquisition
                        current_batch_size = new_batch_size
            
            # Get recommendations
            recommendations = bayesian_campaign.recommend(batch_size=current_batch_size)
            
            if recommendations.empty:
                logger.warning(f"No recommendations at iteration {i+1}")
                continue
            
            batch_yields = []
            batch_measurements = recommendations.copy()
            
            # Evaluate batch
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                temp = rec['Temperature']
                solvent = rec.get('Solvent', solvents[0])
                reagent = rec.get('Reagent', reagents[0])
                catalyst = rec.get('Catalyst', catalysts[0])
                
                condition_dict = {
                    'solvent': solvent,
                    'reagent': reagent if reagent != "None" else "",
                    'catalyst': catalyst if catalyst != "None" else ""
                }
                
                full_reaction_smiles = merge_reaction_with_conditions(naked_reaction_smiles, condition_dict)
                yield_value = safe_predict_yield(oracle, full_reaction_smiles)
                batch_yields.append(yield_value)
                
                current_best_yield = max(current_best_yield, yield_value)
                
                optimization_history.append({
                    "iteration": i + 1,
                    "batch_index": idx,
                    "phase": "bayesian_optimization",
                    "acquisition_function": current_acquisition,
                    "conditions": {
                        "Temperature": round(float(temp), 2),
                        "Solvent": solvent,
                        "Reagent": reagent,
                        "Catalyst": catalyst,
                    },
                    "yield": round(float(yield_value), 2)
                })
            
            # Add measurements
            batch_measurements["Yield"] = batch_yields
            bayesian_campaign.add_measurements(batch_measurements)
            
            logger.info(f"BAYESIAN {i+1-num_random_init}/{num_iterations-num_random_init} | "
                       f"{current_acquisition} batch={len(batch_yields)} | "
                       f"Best: {max(batch_yields):.2f}% | Overall: {current_best_yield:.2f}%")
            
        except Exception as e:
            logger.warning(f"Bayesian iteration {i+1} failed: {e}")
            continue

    # Enhanced results processing (ADD acquisition strategy info)
    if not optimization_history:
        return {"error": "No successful optimization iterations"}
    
    try:
        best_result = max(optimization_history, key=lambda x: x['yield'])
        
        # Calculate metrics per acquisition function
        acq_functions_used = list(set(h.get('acquisition_function', 'Random') for h in optimization_history))
        acq_performance = {}
        
        for acq_func in acq_functions_used:
            acq_yields = [h['yield'] for h in optimization_history if h.get('acquisition_function') == acq_func]
            if acq_yields:
                acq_performance[acq_func] = {
                    "avg_yield": round(np.mean(acq_yields), 2),
                    "max_yield": round(max(acq_yields), 2),
                    "evaluations": len(acq_yields)
                }
        
        random_yields = [h['yield'] for h in optimization_history if h['phase'] == 'random_exploration']
        bayesian_yields = [h['yield'] for h in optimization_history if h['phase'] == 'bayesian_optimization']
        
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
                "avg_random_yield": round(np.mean(random_yields), 2) if random_yields else 0,
                "avg_bayesian_yield": round(np.mean(bayesian_yields), 2) if bayesian_yields else 0,
                "acquisition_performance": acq_performance,
                "improvement_over_random": round(np.mean(bayesian_yields) - np.mean(random_yields), 2) if bayesian_yields and random_yields else 0
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
        naked_reaction_smiles="Cl[Si](Cl)(Cl)Cl.c1ccccc1[Mg]Br>>Cl[Si](Cl)(Cl)c1ccccc1",
        num_initial_candidates=20,
        num_iterations=15,
        num_random_init=5,
        use_adaptive_strategy=True,  # Enable adaptive acquisition
        initial_batch_size=2        # Start with batch optimization
    )

    import json
    print(json.dumps(results, indent=2))