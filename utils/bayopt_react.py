import os
import logging
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
import json
from baybe import Campaign
from baybe.parameters import (
    CategoricalParameter, 
    NumericalContinuousParameter,
    SubstanceParameter
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
from baybe.recommenders import (
    BotorchRecommender, 
    RandomRecommender,
    TwoPhaseMetaRecommender,
    FPSRecommender
)
from baybe.surrogates import GaussianProcessSurrogate
from scipy.stats import norm
from scipy.spatial.distance import cdist
import math

from rxn_insight.reaction import Reaction
from utils.llm_interface import generate_text
from utils.prompt_loader import format_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("rxn-insight successfully imported")

LLM_AVAILABLE = True
logger.info("LLM interface successfully imported for optimization analysis.")
RXN_INSIGHT_AVAILABLE = True

try:
    from utils.yield_optimizer import AdvancedYieldPredictor, merge_reaction_with_conditions
except ImportError:
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from utils.yield_optimizer import AdvancedYieldPredictor, merge_reaction_with_conditions

ORACLE = None

def sanitize_for_json(obj):
    """Recursively replace NaN and Inf values with None for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def is_valid_smiles(smiles: str) -> bool:
    """Validate if a string is valid SMILES"""
    if not smiles or not isinstance(smiles, str):
        return False
    if smiles.strip().lower() == 'none':
        return False
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

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


def get_rxn_insight_conditions(naked_reaction_smiles: str, n_conditions: int = 20) -> tuple:
    """
    Advanced rxn-insight-based condition recommendation system.
    Uses comprehensive reaction analysis, similarity search, and statistical methods
    to dynamically generate optimal reaction conditions without hardcoding.
    
    Returns: (conditions_dict, metadata)
    """
    if not RXN_INSIGHT_AVAILABLE:
        return {}, {"error": "rxn-insight not available"}
    
    try:
        logger.info(f"Advanced rxn-insight analysis for: {naked_reaction_smiles}")
        
        # Initialize comprehensive results storage
        conditions_aggregator = {
            'solvents': {},
            'reagents': {},
            'catalysts': {},
            'temperatures': [],
            'additives': {},
            'atmospheres': {},
            'times': [],
            'pressures': []
        }
        
        # Create Reaction object with enhanced analysis
        rxn = Reaction(naked_reaction_smiles)
        
        # 1. COMPREHENSIVE REACTION ANALYSIS
        logger.info("Performing comprehensive reaction analysis...")
        reaction_info = rxn.get_reaction_info()
        functional_groups = getattr(rxn, 'get_functional_groups', lambda: {})()
        ring_analysis = getattr(rxn, 'get_ring_changes', lambda: {})()
        descriptors = getattr(rxn, 'get_descriptors', lambda: {})()
        classification = getattr(rxn, 'classify_reaction', lambda: {})()
        
        logger.info(f"Reaction analysis complete - Class: {classification.get('class', 'Unknown')}")
        
        # 2. SIMILARITY-BASED CONDITION MINING
        logger.info("Mining conditions from similar reactions...")
        
        try:
            similar_reactions = []
            similarity_methods = ['tanimoto', 'structural', 'functional_group']
            
            for method in similarity_methods:
                try:
                    similar_batch = getattr(rxn, 'find_similar_reactions', lambda **kwargs: [])(
                        method=method, 
                        limit=min(50, n_conditions * 3),
                        min_similarity=0.3
                    )
                    similar_reactions.extend(similar_batch)
                except (AttributeError, Exception) as e:
                    logger.debug(f"Similarity method {method} not available: {e}")
            
            if not similar_reactions:
                similar_reactions = getattr(rxn, 'find_similar', lambda n=50: [])(n=min(50, n_conditions * 3))
            
            logger.info(f"Found {len(similar_reactions)} similar reactions for analysis")
            
            # Process similar reactions for condition extraction
            for idx, similar_rxn in enumerate(similar_reactions):
                try:
                    similarity_score = similar_rxn.get('similarity', 0.5)
                    rxn_conditions = similar_rxn.get('conditions', {})
                    confidence = similarity_score
                    source = f"similar_rxn_{idx}"
                    
                    # Process solvents
                    solvents_list = []
                    if 'solvent' in rxn_conditions:
                        solvents_list.append(rxn_conditions['solvent'])
                    if 'solvents' in rxn_conditions:
                        solvents_list.extend(rxn_conditions['solvents'])
                    
                    for solvent in solvents_list:
                        if solvent and isinstance(solvent, str) and len(solvent.strip()) > 0:
                            solvent = solvent.strip()
                            if solvent not in conditions_aggregator['solvents']:
                                conditions_aggregator['solvents'][solvent] = {
                                    'count': 0, 'confidence': 0, 'sources': []
                                }
                            conditions_aggregator['solvents'][solvent]['count'] += 1
                            conditions_aggregator['solvents'][solvent]['confidence'] += confidence
                            conditions_aggregator['solvents'][solvent]['sources'].append(source)
                    
                    # Process catalysts
                    catalysts_list = []
                    if 'catalyst' in rxn_conditions:
                        catalysts_list.append(rxn_conditions['catalyst'])
                    if 'catalysts' in rxn_conditions:
                        catalysts_list.extend(rxn_conditions['catalysts'])
                    
                    for catalyst in catalysts_list:
                        if catalyst and isinstance(catalyst, str) and len(catalyst.strip()) > 0:
                            catalyst = catalyst.strip()
                            if catalyst not in conditions_aggregator['catalysts']:
                                conditions_aggregator['catalysts'][catalyst] = {
                                    'count': 0, 'confidence': 0, 'sources': []
                                }
                            conditions_aggregator['catalysts'][catalyst]['count'] += 1
                            conditions_aggregator['catalysts'][catalyst]['confidence'] += confidence
                            conditions_aggregator['catalysts'][catalyst]['sources'].append(source)
                    
                    # Process reagents/bases/acids
                    reagents_list = []
                    for key in ['reagent', 'reagents', 'base', 'bases', 'acid', 'acids', 'additive', 'additives']:
                        if key in rxn_conditions:
                            item = rxn_conditions[key]
                            if isinstance(item, list):
                                reagents_list.extend(item)
                            elif item:
                                reagents_list.append(item)
                    
                    for reagent in reagents_list:
                        if reagent and isinstance(reagent, str) and len(reagent.strip()) > 0:
                            reagent = reagent.strip()
                            if reagent not in conditions_aggregator['reagents']:
                                conditions_aggregator['reagents'][reagent] = {
                                    'count': 0, 'confidence': 0, 'sources': []
                                }
                            conditions_aggregator['reagents'][reagent]['count'] += 1
                            conditions_aggregator['reagents'][reagent]['confidence'] += confidence
                            conditions_aggregator['reagents'][reagent]['sources'].append(source)
                    
                    # Process temperatures
                    temp_keys = ['temperature', 'temp', 'Temperature', 'reaction_temperature']
                    for temp_key in temp_keys:
                        if temp_key in rxn_conditions:
                            try:
                                temp_val = float(rxn_conditions[temp_key])
                                if 0 <= temp_val <= 500:
                                    conditions_aggregator['temperatures'].append((temp_val, confidence, source))
                            except (ValueError, TypeError):
                                pass
                    
                    # Process atmospheres
                    if 'atmosphere' in rxn_conditions:
                        atm = rxn_conditions['atmosphere']
                        if atm not in conditions_aggregator['atmospheres']:
                            conditions_aggregator['atmospheres'][atm] = {'count': 0, 'confidence': 0}
                        conditions_aggregator['atmospheres'][atm]['count'] += 1
                        conditions_aggregator['atmospheres'][atm]['confidence'] += confidence
                    
                    # Process reaction times
                    if 'time' in rxn_conditions:
                        try:
                            time_val = float(rxn_conditions['time'])
                            conditions_aggregator['times'].append((time_val, confidence, source))
                        except (ValueError, TypeError):
                            pass
                            
                except Exception as e:
                    logger.debug(f"Error processing similar reaction {idx}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Similarity-based condition mining failed: {e}")
        
        # 3. FUNCTIONAL GROUP-BASED CONDITION ENHANCEMENT
        logger.info("Enhancing conditions based on functional group analysis...")
        
        try:
            reactant_fg = functional_groups.get('reactants', {})
            product_fg = functional_groups.get('products', {})
            
            fg_changes = {}
            for fg in set(list(reactant_fg.keys()) + list(product_fg.keys())):
                reactant_count = reactant_fg.get(fg, 0)
                product_count = product_fg.get(fg, 0)
                if reactant_count != product_count:
                    fg_changes[fg] = {
                        'reactants': reactant_count,
                        'products': product_count,
                        'net_change': product_count - reactant_count
                    }
            
            fg_condition_db = getattr(rxn, 'get_fg_conditions', lambda fg_dict: {})(fg_changes)
            
            for condition_type, suggestions in fg_condition_db.items():
                if condition_type in conditions_aggregator:
                    for suggestion in suggestions:
                        name = suggestion.get('name', '')
                        confidence = suggestion.get('confidence', 0.3)
                        if name and name not in conditions_aggregator[condition_type]:
                            conditions_aggregator[condition_type][name] = {
                                'count': 1, 'confidence': confidence, 'sources': ['functional_group_analysis']
                            }
                        
        except Exception as e:
            logger.debug(f"Functional group enhancement failed: {e}")
        
        # 4. STATISTICAL ANALYSIS AND RANKING
        logger.info("Performing statistical analysis and ranking...")
        
        def rank_conditions(condition_dict, min_count=1, max_results=None):
            ranked = []
            for name, data in condition_dict.items():
                avg_confidence = data['confidence'] / max(data['count'], 1)
                popularity_score = min(data['count'] / 10.0, 1.0)
                composite_score = (0.7 * avg_confidence) + (0.3 * popularity_score)
                
                if data['count'] >= min_count:
                    ranked.append({
                        'name': name,
                        'score': composite_score,
                        'count': data['count'],
                        'avg_confidence': avg_confidence,
                        'sources': len(set(data['sources']))
                    })
            
            ranked.sort(key=lambda x: x['score'], reverse=True)
            
            if max_results:
                ranked = ranked[:max_results]
            
            return ranked
        
        ranked_solvents = rank_conditions(conditions_aggregator['solvents'], min_count=1, max_results=15)
        ranked_reagents = rank_conditions(conditions_aggregator['reagents'], min_count=1, max_results=12)
        ranked_catalysts = rank_conditions(conditions_aggregator['catalysts'], min_count=1, max_results=12)
        ranked_atmospheres = rank_conditions(conditions_aggregator['atmospheres'], min_count=1, max_results=5)
        
        # Statistical temperature analysis
        temp_analysis = {}
        if conditions_aggregator['temperatures']:
            temps = [t[0] for t in conditions_aggregator['temperatures']]
            confidences = [t[1] for t in conditions_aggregator['temperatures']]
            
            weighted_temps = np.average(temps, weights=confidences)
            temp_std = np.std(temps)
            
            temp_analysis = {
                'mean': float(weighted_temps),
                'std': float(temp_std),
                'range': [float(min(temps)), float(max(temps))],
                'recommended_range': [
                    max(0, float(weighted_temps - temp_std)), 
                    min(300, float(weighted_temps + temp_std))
                ],
                'data_points': len(temps)
            }
        
        # 5. COMPILE FINAL RESULTS
        logger.info("Compiling final condition recommendations...")
        
        final_conditions = {
            'solvents': [item['name'] for item in ranked_solvents],
            'reagents': [item['name'] for item in ranked_reagents],
            'catalysts': [item['name'] for item in ranked_catalysts],
            'temperatures': list(temp_analysis.get('recommended_range', [])),
            'atmospheres': [item['name'] for item in ranked_atmospheres],
            'temperature_analysis': temp_analysis
        }
        
        metadata = {
            'reaction_class': classification.get('class', 'Unknown'),
            'reaction_confidence': classification.get('confidence', 0),
            'functional_groups': functional_groups,
            'ring_changes': ring_analysis,
            'similar_reactions_analyzed': len(similar_reactions),
            'condition_statistics': {
                'solvents_found': len(ranked_solvents),
                'reagents_found': len(ranked_reagents),
                'catalysts_found': len(ranked_catalysts),
                'temperature_data_points': len(conditions_aggregator['temperatures'])
            },
            'top_ranked_conditions': {
                'solvents': ranked_solvents[:5],
                'reagents': ranked_reagents[:5], 
                'catalysts': ranked_catalysts[:5]
            },
            'analysis_successful': True,
            'data_sources': len(similar_reactions),
            'methodology': 'similarity_mining + functional_group_analysis + statistical_ranking'
        }
        
        logger.info(f"Advanced analysis complete - Found {len(final_conditions['solvents'])} solvents, "
                   f"{len(final_conditions['reagents'])} reagents, {len(final_conditions['catalysts'])} catalysts")
        
        return final_conditions, metadata
        
    except Exception as e:
        logger.error(f"Advanced rxn-insight analysis failed: {e}", exc_info=True)
        return {}, {
            'analysis_successful': False, 
            'error': str(e),
            'methodology': 'advanced_rxn_insight_failed'
        }

def get_adaptive_batch_size(
    iteration: int, 
    total_iterations: int, 
    current_best_yield: float,
    remaining_budget: int,
    recent_improvements: List[float]
) -> int:
    """
    Budget-aware adaptive batch sizing with exploration-exploitation balance.
    
    Args:
        iteration: Current iteration number
        total_iterations: Total planned iterations
        current_best_yield: Best yield found so far
        remaining_budget: Number of evaluations remaining in budget
        recent_improvements: List of yield improvements in recent iterations
        
    Returns:
        Optimal batch size for current iteration
    """
    progress = iteration / max(total_iterations, 1)
    
    # Calculate recent improvement trend
    if len(recent_improvements) >= 3:
        avg_recent_improvement = np.mean(recent_improvements[-3:])
        improvement_trend = avg_recent_improvement > 1.0  # >1% improvement
    else:
        improvement_trend = True  # Assume positive in early stages
    
    # Base batch size on phase
    if progress < 0.2:
        # Early exploration: larger batches
        base_batch = 3
    elif progress < 0.5:
        # Mid exploration-exploitation
        base_batch = 2
    elif progress < 0.8:
        # Late exploitation
        base_batch = 1 if current_best_yield > 70 else 2
    else:
        # Final refinement
        base_batch = 1
    
    # Adjust based on yield performance
    if current_best_yield < 30:
        # Poor performance: increase exploration
        base_batch = min(base_batch + 1, 4)
    elif current_best_yield > 80:
        # Excellent performance: focus exploitation
        base_batch = 1
    
    # Adjust based on improvement trend
    if not improvement_trend and progress > 0.3:
        # Stagnation: try larger batches for diversity
        base_batch = min(base_batch + 1, 3)
    
    # Budget constraint: ensure we don't exceed remaining budget
    iterations_remaining = max(1, total_iterations - iteration)
    max_sustainable_batch = max(1, remaining_budget // iterations_remaining)
    
    # Conservative buffer for final iterations
    if progress > 0.9:
        max_sustainable_batch = min(max_sustainable_batch, remaining_budget)
    
    final_batch = min(base_batch, max_sustainable_batch, remaining_budget)
    
    logger.info(f"Adaptive batch: base={base_batch}, budget_limit={max_sustainable_batch}, "
                f"final={final_batch} (budget remaining: {remaining_budget})")
    
    return max(1, final_batch)

def enforce_diversity_constraint(
    recommendations: pd.DataFrame,
    campaign: Campaign,
    min_distance_threshold: float = 0.15,
    max_attempts: int = 10
) -> pd.DataFrame:
    """
    Enforce diversity in recommendations by rejecting points too similar to existing data.
    
    Args:
        recommendations: Candidate recommendations from campaign
        campaign: BayBE campaign with measurement history
        min_distance_threshold: Minimum normalized distance required (0-1)
        max_attempts: Maximum re-sampling attempts
        
    Returns:
        Diversified recommendations DataFrame
    """
    if not hasattr(campaign, 'measurements') or campaign.measurements.empty:
        return recommendations
    
    try:
        from scipy.spatial.distance import cdist
        from baybe.utils.dataframe import to_tensor
        
        # Transform existing measurements to model space
        existing_data = campaign.measurements.drop(columns=['Yield'], errors='ignore')
        X_existing = campaign.searchspace.transform(existing_data)
        X_existing_array = to_tensor(X_existing).numpy()
        
        diverse_recs = []
        attempts = 0
        
        for idx, (_, rec) in enumerate(recommendations.iterrows()):
            rec_df = pd.DataFrame([rec])
            X_rec = campaign.searchspace.transform(rec_df)
            X_rec_array = to_tensor(X_rec).numpy()
            
            # Calculate minimum distance to existing points
            distances = cdist(X_rec_array, X_existing_array, metric='euclidean')
            min_distance = distances.min()
            
            # Normalize by feature space dimensionality
            normalized_distance = min_distance / np.sqrt(X_existing_array.shape[1])
            
            if normalized_distance >= min_distance_threshold or attempts >= max_attempts:
                diverse_recs.append(rec)
                # Add this point to existing data for next comparisons
                X_existing_array = np.vstack([X_existing_array, X_rec_array])
            else:
                logger.debug(f"Recommendation {idx} too similar (distance={normalized_distance:.3f}), "
                            f"threshold={min_distance_threshold}")
                attempts += 1
                
                # Request replacement recommendation
                try:
                    new_rec = campaign.recommend(batch_size=1)
                    if not new_rec.empty:
                        diverse_recs.append(new_rec.iloc[0])
                        X_new = campaign.searchspace.transform(new_rec)
                        X_existing_array = np.vstack([X_existing_array, to_tensor(X_new).numpy()])
                except:
                    # Fallback: keep original if re-sampling fails
                    diverse_recs.append(rec)
        
        result = pd.DataFrame(diverse_recs)
        logger.info(f"Diversity enforcement: {len(recommendations)} -> {len(result)} unique recommendations")
        return result
        
    except Exception as e:
        logger.warning(f"Diversity constraint failed: {e}, returning original recommendations")
        return recommendations


def create_adaptive_recommender(phase_config: Dict[str, Any]) -> BotorchRecommender:
    """
    Create BotorchRecommender with dynamically selected acquisition function.
    
    Args:
        phase_config: Dictionary containing acquisition_function_cls and acquisition_function_kwargs
        
    Returns:
        Configured BotorchRecommender instance
    """
    return BotorchRecommender(
        surrogate_model=GaussianProcessSurrogate(),
        acquisition_function_cls=phase_config["acquisition_function_cls"],
        acquisition_function_kwargs=phase_config.get("acquisition_function_kwargs", {})
    )

def switch_to_bayesian_recommender_advanced(campaign: Campaign, batch_size: int = 1) -> Campaign:
    """
    Switch to Bayesian optimization with BotorchRecommender.
    Returns a new campaign object with the transferred measurements.
    """
    try:
        # Create standard BotorchRecommender (no acquisition function parameters)
        bayesian_recommender = BotorchRecommender()
        
        # Store existing measurements
        existing_measurements = None
        if hasattr(campaign, 'measurements') and not campaign.measurements.empty:
            existing_measurements = campaign.measurements.copy()
        
        # Create completely new campaign
        new_campaign = Campaign(
            searchspace=campaign.searchspace,
            objective=campaign.objective,
            recommender=bayesian_recommender
        )
        
        # Transfer measurements to new campaign if they exist
        if existing_measurements is not None and not existing_measurements.empty:
            new_campaign.add_measurements(existing_measurements)
            logger.info(f"Transferred {len(existing_measurements)} measurements to Bayesian campaign")
        
        return new_campaign
        
    except Exception as e:
        logger.warning(f"Failed to switch to Bayesian recommender: {e}")
        return campaign

def analyze_recommendations_with_uncertainty(
    campaign: Campaign, 
    recommendations: pd.DataFrame,
    current_best: float
) -> List[Dict[str, Any]]:
    """
    Provide Gaussian Process-based uncertainty metrics for each recommendation.
    
    Returns:
        List of dicts containing uncertainty metrics for each recommendation
    """
    uncertainty_analysis = []
    
    try:
        # Access the surrogate model from the campaign
        if not hasattr(campaign, 'recommender'):
            return [{"error": "No recommender available"} for _ in range(len(recommendations))]
        
        recommender = campaign.recommender
        
        # Handle TwoPhaseMetaRecommender
        if hasattr(recommender, 'recommender'):
            actual_recommender = recommender.recommender
        else:
            actual_recommender = recommender
            
        # Check if we have a Gaussian Process surrogate
        has_gp_surrogate = (
            hasattr(actual_recommender, 'surrogate_model') and
            actual_recommender.surrogate_model is not None and
            hasattr(campaign, 'measurements') and 
            len(campaign.measurements) > 0
        )
        
        if not has_gp_surrogate:
            logger.debug("No GP surrogate available for uncertainty quantification")
            return [{"error": "GP not available yet"} for _ in range(len(recommendations))]
        
        surrogate = actual_recommender.surrogate_model
        
        # Transform recommendations to the model's input space
        from baybe.utils.dataframe import to_tensor
        
        for idx, (_, rec) in enumerate(recommendations.iterrows()):
            try:
                # Convert recommendation to proper format
                rec_df = pd.DataFrame([rec])
                
                # Transform to model space using campaign's searchspace
                X_test = campaign.searchspace.transform(rec_df)
                X_test_tensor = to_tensor(X_test)
                
                # Get GP posterior predictions
                with torch.no_grad():
                    posterior = surrogate._model.posterior(X_test_tensor)
                    mean = posterior.mean.squeeze().item()
                    variance = posterior.variance.squeeze().item()
                    std = np.sqrt(variance)
                
                # De-normalize predictions if target is normalized
                if hasattr(campaign.objective.targets[0], 'transformation'):
                    # Assuming yield is in [0, 100] range
                    predicted_mean = mean * 100
                    predicted_std = std * 100
                else:
                    predicted_mean = mean
                    predicted_std = std
                
                # Calculate confidence intervals
                ci_95_lower = max(0, predicted_mean - 1.96 * predicted_std)
                ci_95_upper = min(100, predicted_mean + 1.96 * predicted_std)
                ci_68_lower = max(0, predicted_mean - predicted_std)
                ci_68_upper = min(100, predicted_mean + predicted_std)
                
                # Calculate probability of improvement over current best
                from scipy.stats import norm
                if predicted_std > 1e-6:
                    z_score = (predicted_mean - current_best) / predicted_std
                    poi = float(norm.cdf(z_score))
                else:
                    poi = 1.0 if predicted_mean > current_best else 0.0
                
                # Calculate expected improvement
                if predicted_std > 1e-6:
                    improvement = predicted_mean - current_best
                    z = improvement / predicted_std
                    ei = improvement * norm.cdf(z) + predicted_std * norm.pdf(z)
                else:
                    ei = max(0, predicted_mean - current_best)
                
                # Coefficient of variation
                cv = predicted_std / (abs(predicted_mean) + 1e-9)
                
                # Quality assessment based on uncertainty
                if cv < 0.1 and predicted_std < 5.0:
                    quality = 'high'
                elif cv < 0.3 and predicted_std < 15.0:
                    quality = 'medium'
                else:
                    quality = 'low'
                
                analysis = {
                    'conditions': rec.to_dict(),
                    'predicted_yield_mean': round(float(predicted_mean), 2),
                    'predicted_yield_std': round(float(predicted_std), 2),
                    'confidence_interval_95': [round(ci_95_lower, 2), round(ci_95_upper, 2)],
                    'confidence_interval_68': [round(ci_68_lower, 2), round(ci_68_upper, 2)],
                    'probability_of_improvement': round(poi, 3),
                    'expected_improvement': round(float(ei), 2),
                    'coefficient_of_variation': round(cv, 3),
                    'recommendation_quality': quality,
                    'uncertainty_source': 'gaussian_process'
                }
                
                uncertainty_analysis.append(analysis)
                
            except Exception as e:
                logger.debug(f"Error in GP uncertainty for recommendation {idx}: {e}")
                uncertainty_analysis.append({
                    'conditions': rec.to_dict(),
                    'error': str(e),
                    'uncertainty_source': 'failed'
                })
                
    except Exception as e:
        logger.warning(f"GP uncertainty analysis failed: {e}")
        return [{"error": str(e), "uncertainty_source": "failed"} for _ in range(len(recommendations))]
    
    return uncertainty_analysis

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
    initial_batch_size: int = 2,
    use_rxn_insight: bool = True,
    prior_knowledge: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced Bayesian optimization with chemistry-aware parameters and adaptive acquisition.
    """
    logger.info(f"Starting ENHANCED Bayesian Optimization for: {naked_reaction_smiles}")
    
    if not naked_reaction_smiles or not naked_reaction_smiles.strip():
        return {"error": "Invalid or empty reaction SMILES provided"}
    
    # Initialize Oracle
    oracle = initialize_oracle()
    if oracle is None:
        return {"error": "Could not initialize the simulation oracle"}
    
    # ========================================================================
    # EARLY CONSTRAINT EXTRACTION
    # ========================================================================
    user_constraints = {
        'temperature_range': None,
        'solvents': None,
        'reagents': None,
        'starting_materials': None,
        'catalysts': None,
        'reagent_eq_range': None,
        'catalyst_mol_percent_range': None,
    }

    reactants_smiles = naked_reaction_smiles.split(">>")[0].split('.')
    num_reactants = len(reactants_smiles)
    
    if prior_knowledge and 'constraints' in prior_knowledge:
        constraints = prior_knowledge['constraints']
        logger.info("User constraints detected - extracting and preserving them")
        
        if 'temperature_range' in constraints and len(constraints['temperature_range']) == 2:
            user_constraints['temperature_range'] = constraints['temperature_range']
            logger.info(f"USER CONSTRAINT: Temperature range = {constraints['temperature_range']}")
        
        if 'solvents' in constraints and constraints['solvents']:
            user_constraints['solvents'] = constraints['solvents']
            logger.info(f"USER CONSTRAINT: Solvents = {constraints['solvents']}")
            
        if 'reagents' in constraints and constraints['reagents']:
            user_constraints['reagents'] = constraints['reagents']
            logger.info(f"USER CONSTRAINT: Reagents = {constraints['reagents']}")

        if 'starting_materials' in constraints and constraints['starting_materials']:
            user_constraints['starting_materials'] = constraints['starting_materials']
            logger.info(f"USER CONSTRAINT: Starting Materials = {constraints['starting_materials']}")
            
        if 'catalysts' in constraints and constraints['catalysts']:
            user_constraints['catalysts'] = constraints['catalysts']
            logger.info(f"USER CONSTRAINT: Catalysts = {constraints['catalysts']}")

        if 'reagent_eq_range' in constraints and len(constraints['reagent_eq_range']) == 2:
            user_constraints['reagent_eq_range'] = constraints['reagent_eq_range']
            logger.info(f"USER CONSTRAINT: Reagent Equiv. Range = {constraints['reagent_eq_range']}")

        if 'catalyst_mol_percent_range' in constraints and len(constraints['catalyst_mol_percent_range']) == 2:
            user_constraints['catalyst_mol_percent_range'] = constraints['catalyst_mol_percent_range']
            logger.info(f"USER CONSTRAINT: Catalyst Mol % Range = {constraints['catalyst_mol_percent_range']}")
            
        if num_reactants > 1:
            for i in range(1, num_reactants):
                param_key = f"reactant_{i+1}_eq_range"
                if param_key in constraints and len(constraints[param_key]) == 2:
                    user_constraints[param_key] = constraints[param_key]
                    logger.info(f"USER CONSTRAINT: Reactant {i+1} Equiv. = {constraints[param_key]}")

    # ========================================================================
    # GENERATE SEARCH SPACE WITH RXN-INSIGHT INTEGRATION
    # ========================================================================
    try:
        logger.info(f"Generating enhanced search space...")
        
        # Get initial conditions from RCR model
        initial_conditions, _ = oracle.condition_predictor.get_n_conditions(
            naked_reaction_smiles, n=num_initial_candidates, return_scores=True
        )
        
        if not initial_conditions:
            return {"error": "Could not generate initial candidate conditions"}

        # Process RCR model conditions
        model_temps = []
        model_solvents = set()
        model_reagents = set()
        model_catalysts = set()
        
        for cond in initial_conditions:
            if len(cond) >= 4:
                temp, solvent, reagent, catalyst = cond[0], cond[1], cond[2], cond[3]
                
                if temp is not None and isinstance(temp, (int, float)):
                    model_temps.append(float(temp))
                if solvent and isinstance(solvent, str) and solvent.strip():
                    model_solvents.add(solvent.strip())
                if reagent and isinstance(reagent, str) and reagent.strip():
                    model_reagents.add(reagent.strip())
                if catalyst and isinstance(catalyst, str) and catalyst.strip():
                    model_catalysts.add(catalyst.strip())
        
        # Advanced rxn-insight integration
        rxn_insight_metadata = {}
        if use_rxn_insight:
            try:
                rxn_insight_conditions, rxn_insight_metadata = get_rxn_insight_conditions(
                    naked_reaction_smiles, n_conditions=num_initial_candidates + 20
                )
                
                if rxn_insight_conditions and rxn_insight_metadata.get('analysis_successful', False):
                    logger.info(f"Advanced rxn-insight analysis successful - "
                               f"Class: {rxn_insight_metadata.get('reaction_class', 'Unknown')} "
                               f"(confidence: {rxn_insight_metadata.get('reaction_confidence', 0):.2f})")
                    
                    # Intelligent condition merging with priority to rxn-insight
                    rxn_solvents = set(rxn_insight_conditions.get('solvents', []))
                    rxn_reagents = set(rxn_insight_conditions.get('reagents', []))
                    rxn_catalysts = set(rxn_insight_conditions.get('catalysts', []))
                    
                    # Priority merge: rxn-insight first, then RCR model conditions
                    model_solvents = list(rxn_solvents) + [s for s in model_solvents if s not in rxn_solvents]
                    model_reagents = list(rxn_reagents) + [r for r in model_reagents if r not in rxn_reagents]
                    model_catalysts = list(rxn_catalysts) + [c for c in model_catalysts if c not in rxn_catalysts]
                    
                    # Advanced temperature integration
                    if not user_constraints['temperature_range']:
                        temp_analysis = rxn_insight_conditions.get('temperature_analysis', {})
                        if temp_analysis and 'recommended_range' in temp_analysis:
                            rxn_temp_range = temp_analysis['recommended_range']
                            if len(rxn_temp_range) == 2:
                                data_points = temp_analysis.get('data_points', 0)
                                if data_points >= 3:
                                    model_temps = rxn_temp_range
                                    logger.info(f"Using rxn-insight temperature range: {rxn_temp_range[0]:.1f}-{rxn_temp_range[1]:.1f}°C "
                                               f"(based on {data_points} similar reactions)")
                    
                    logger.info(f"Enhanced search space: {len(model_solvents)} solvents, "
                               f"{len(model_reagents)} reagents, {len(model_catalysts)} catalysts")
                    logger.info(f"Data sources: {rxn_insight_metadata.get('data_sources', 0)} similar reactions analyzed")
                else:
                    logger.warning("rxn-insight analysis unsuccessful, using RCR-only conditions")
                
            except Exception as e:
                logger.warning(f"Advanced rxn-insight integration failed: {e}")
                rxn_insight_metadata = {'analysis_successful': False, 'error': str(e)}
        
        # ========================================================================
        # CONSTRAINT APPLICATION
        # ========================================================================
        
        # SOLVENTS: User constraints take absolute priority
        if user_constraints['solvents'] is not None:
            final_solvents = [s for s in user_constraints['solvents'] if s]
            logger.info(f"USING USER SOLVENT CONSTRAINTS: {len(final_solvents)} solvents")
        else:
            if use_rxn_insight and rxn_insight_metadata.get('analysis_successful', False):
                top_conditions = rxn_insight_metadata.get('top_ranked_conditions', {})
                if 'solvents' in top_conditions:
                    priority_solvents = [item['name'] for item in top_conditions['solvents']]
                    other_solvents = [s for s in model_solvents if s not in priority_solvents and s and len(s) < 50]
                    final_solvents = priority_solvents + other_solvents[:max(0, 25-len(priority_solvents))]
                else:
                    final_solvents = sorted([s for s in model_solvents if s and len(s) < 50])[:25]
            else:
                final_solvents = sorted([s for s in model_solvents if s and len(s) < 50])[:25]
            logger.info(f"USING MODEL-BASED SOLVENTS: {len(final_solvents)} solvents")
        
        # REAGENTS: User constraints take absolute priority
        if user_constraints['reagents'] is not None:
            final_reagents = [r for r in user_constraints['reagents'] if r]
            logger.info(f"USING USER REAGENT CONSTRAINTS: {len(final_reagents)} reagents")
        else:
            if use_rxn_insight and rxn_insight_metadata.get('analysis_successful', False):
                top_conditions = rxn_insight_metadata.get('top_ranked_conditions', {})
                if 'reagents' in top_conditions:
                    priority_reagents = [item['name'] for item in top_conditions['reagents']]
                    other_reagents = [r for r in model_reagents if r not in priority_reagents and r and len(r) < 50]
                    final_reagents = priority_reagents + other_reagents[:max(0, 20-len(priority_reagents))]
                else:
                    final_reagents = sorted([r for r in model_reagents if r and len(r) < 50])[:20]
            else:
                final_reagents = sorted([r for r in model_reagents if r and len(r) < 50])[:30]
            logger.info(f"USING MODEL-BASED REAGENTS: {len(final_reagents)} reagents")
        
        # CATALYSTS: User constraints take absolute priority
        if user_constraints['catalysts'] is not None:
            final_catalysts = [c for c in user_constraints['catalysts'] if c]
            logger.info(f"USING USER CATALYST CONSTRAINTS: {len(final_catalysts)} catalysts")
        else:
            if use_rxn_insight and rxn_insight_metadata.get('analysis_successful', False):
                top_conditions = rxn_insight_metadata.get('top_ranked_conditions', {})
                if 'catalysts' in top_conditions:
                    priority_catalysts = [item['name'] for item in top_conditions['catalysts']]
                    other_catalysts = [c for c in model_catalysts if c not in priority_catalysts and c and len(c) < 50]
                    final_catalysts = priority_catalysts + other_catalysts[:max(0, 20-len(priority_catalysts))]
                else:
                    final_catalysts = sorted([c for c in model_catalysts if c and len(c) < 50])[:20]
            else:
                final_catalysts = sorted([c for c in model_catalysts if c and len(c) < 50])[:25]
            logger.info(f"USING MODEL-BASED CATALYSTS: {len(final_catalysts)} catalysts")
            
        # TEMPERATURE: User constraints take absolute priority
        if user_constraints['temperature_range'] is not None:
            temp_min, temp_max = user_constraints['temperature_range']
            # Validate user constraints
            if temp_min >= temp_max:
                logger.warning(f"Invalid user temperature range: {temp_min} >= {temp_max}. Swapping values.")
                temp_min, temp_max = temp_max, temp_min
            temp_min = max(-100.0, min(temp_min, 400.0))
            temp_max = max(-100.0, min(temp_max, 400.0))
            logger.info(f"USING USER TEMPERATURE CONSTRAINTS: {temp_min}°C to {temp_max}°C")
        else:
            if model_temps:
                model_temps = [t for t in model_temps if -100 <= t <= 400]
                if model_temps:
                    temp_min, temp_max = min(model_temps), max(model_temps)
                    # Ensure minimum range of 10°C
                    if temp_max - temp_min < 10.0:
                        temp_center = (temp_min + temp_max) / 2
                        temp_min = max(-100, temp_center - 25)
                        temp_max = min(400, temp_center + 25)
                    # Clamp to physical limits
                    temp_min = max(-100.0, temp_min)
                    temp_max = min(400.0, temp_max)
                else:
                    temp_min, temp_max = 0.0, 200.0
            else:
                temp_min, temp_max = 0.0, 200.0
            logger.info(f"USING MODEL-BASED TEMPERATURE RANGE: {temp_min:.1f}°C to {temp_max:.1f}°C")

        # Final validation
        if temp_min >= temp_max:
            logger.error(f"Invalid final temperature range: [{temp_min}, {temp_max}]. Using safe default.")
            temp_min, temp_max = 0.0, 200.0
            
        # Add "None" options for optional components if not constrained
        if "None" not in final_reagents:
            final_reagents.append("None")
        if "None" not in final_catalysts:
            final_catalysts.append("None")
            
        # Ensure fallback options when no model data available
        if not final_solvents:
            final_solvents = [
                "C1CCOC1", "CCOCC", "ClCCl", "c1ccccc1", "CC(C)=O",
                "CN(C)C=O", "CCCCCC", "CCO", "CO", "ClC(Cl)Cl"
            ]

        if not final_reagents or final_reagents == ["None"]:
            final_reagents = [
                "None", "O", "Cl", "F", "Br", "I", "[Li+]", "[Na+]", "[K+]",
                "CCN(CC)CC", "C1CCC2=NCCCN2CC1", "ClC(=O)C(=O)Cl",
                "CC(C)(C)OC(=O)N", "ClS(=O)(=O)c1ccc(C)cc1"
            ]

        if not final_catalysts or final_catalysts == ["None"]:
            final_catalysts = [
                "None", "[Pd]", "[Ni]", "[Cu]",
                "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21"
            ]
        
        logger.info(f"FINAL SEARCH SPACE: {len(final_solvents)} solvents, {len(final_reagents)} reagents, "
                   f"{len(final_catalysts)} catalysts, temp range: {temp_min:.1f}-{temp_max:.1f}°C")

    except Exception as e:
        logger.error(f"Error generating enhanced search space: {e}", exc_info=True)
        return {"error": f"Failed to generate search space: {e}"}
    
    # Filter out invalid SMILES
    final_reagents = [r for r in final_reagents if is_valid_smiles(r)]
    final_catalysts = [c for c in final_catalysts if is_valid_smiles(c)]
    final_solvents = [s for s in final_solvents if is_valid_smiles(s)]

    # Add "None" option ONLY after validation
    if not any(r.lower() == 'none' for r in final_reagents):
        final_reagents.append("None")
    if not any(c.lower() == 'none' for c in final_catalysts):
        final_catalysts.append("None")

    # ========================================================================
    # CREATE SEARCH SPACE WITH CHEMISTRY-AWARE PARAMETERS
    # ========================================================================
    try:
        parameters = []
        
        # Track which chemical parameters use SubstanceParameter vs structured numerical
        use_substance_params = {
            'solvents': True,
            'reagents': True,
            'catalysts': True
        }
        
        # REAGENTS: Check for structured constraints
        if (user_constraints.get('reagents') and 
            isinstance(user_constraints['reagents'], list) and 
            all(isinstance(r, dict) and 'smiles' in r and 'eq_range' in r for r in user_constraints['reagents'])):
            
            logger.info("Detected structured reagent constraints. Creating numerical parameters.")
            use_substance_params['reagents'] = False
            
            for reagent_info in user_constraints['reagents']:
                reagent_smiles = reagent_info['smiles']
                sanitized_smiles = "".join(c for c in reagent_smiles if c.isalnum())
                param_name = f"Reagent_{sanitized_smiles}_Equivalents"
                bounds = tuple(reagent_info['eq_range'])
                parameters.append(NumericalContinuousParameter(name=param_name, bounds=bounds))
                logger.info(f"Added numerical parameter for Reagent '{reagent_smiles}' with bounds {bounds}")
        else:
            # Use SubstanceParameter with chemical encoding
            # For reagents
            if len(final_reagents) > 1:
                # Exclude "None" from SubstanceParameter data dict
                reagent_smiles_dict = {
                    f"Reagent_{i}": smiles 
                    for i, smiles in enumerate(final_reagents) 
                    if smiles.lower() != "none"
                }
                if reagent_smiles_dict:  # Only create if we have valid SMILES
                    parameters.append(
                        SubstanceParameter(
                            name="Reagent",
                            data=reagent_smiles_dict,
                            encoding="MORDRED",
                            decorrelate=True
                        )
                    )
                logger.info(f"Created SubstanceParameter for Reagent with RDKIT encoding ({len(final_reagents)} options)")
            
            reagent_eq_bounds = user_constraints['reagent_eq_range'] or (0.8, 5.0)
            parameters.append(NumericalContinuousParameter(name="Reagent_Equivalents", bounds=reagent_eq_bounds))

        # STARTING MATERIALS: Check for structured constraints
        if (user_constraints.get('starting_materials') and
            isinstance(user_constraints['starting_materials'], list) and
            all(isinstance(sm, dict) and 'smiles' in sm and 'eq_range' in sm for sm in user_constraints['starting_materials'])):
            
            logger.info("Detected structured starting material constraints. Creating numerical parameters.")
            for sm_info in user_constraints['starting_materials']:
                sm_smiles = sm_info['smiles']
                sanitized_smiles = "".join(c for c in sm_smiles if c.isalnum())
                param_name = f"Starting_Material_{sanitized_smiles}_Equivalents"
                bounds = tuple(sm_info['eq_range'])
                parameters.append(NumericalContinuousParameter(name=param_name, bounds=bounds))
                logger.info(f"Added numerical parameter for Starting Material '{sm_smiles}' with bounds {bounds}")
            
        # CATALYSTS: Check for structured constraints
        if (user_constraints.get('catalysts') and 
            isinstance(user_constraints['catalysts'], list) and 
            all(isinstance(c, dict) and 'smiles' in c and 'mol_percent_range' in c for c in user_constraints['catalysts'])):

            logger.info("Detected structured catalyst constraints. Creating numerical parameters.")
            use_substance_params['catalysts'] = False
            
            for catalyst_info in user_constraints['catalysts']:
                catalyst_smiles = catalyst_info['smiles']
                sanitized_smiles = "".join(c for c in catalyst_smiles if c.isalnum())
                param_name = f"Catalyst_{sanitized_smiles}_mol_percent"
                bounds = tuple(catalyst_info['mol_percent_range'])
                parameters.append(NumericalContinuousParameter(name=param_name, bounds=bounds))
                logger.info(f"Added numerical parameter for Catalyst '{catalyst_smiles}' with bounds {bounds}")
        else:
            # Use SubstanceParameter with chemical encoding
            if len(final_catalysts) > 1:
                catalyst_smiles_dict = {f"Catalyst_{i}": smiles 
                                        for i, smiles in enumerate(final_catalysts) 
                                        if smiles.lower() != "none"}
                parameters.append(
                    SubstanceParameter(
                        name="Catalyst",
                        data=catalyst_smiles_dict,
                        encoding="MORDRED",
                        decorrelate=True
                    )
                )
                logger.info(f"Created SubstanceParameter for Catalyst with MORDRED encoding ({len(final_catalysts)} options)")
            
            catalyst_mol_percent_bounds = user_constraints['catalyst_mol_percent_range'] or (0.1, 10.0)
            parameters.append(NumericalContinuousParameter(name="Catalyst_mol_percent", bounds=catalyst_mol_percent_bounds))

        # SOLVENTS: Always use SubstanceParameter unless explicitly constrained to single solvent
        if len(final_solvents) > 1:
            solvent_smiles_dict = {f"Solvent_{i}": smiles for i, smiles in enumerate(final_solvents)}
            parameters.append(
                SubstanceParameter(
                    name="Solvent",
                    data=solvent_smiles_dict,
                    encoding="MORDRED",
                    decorrelate=True
                )
            )
            logger.info(f"Created SubstanceParameter for Solvent with MORDRED encoding ({len(final_solvents)} options)")
            
        # TEMPERATURE
        parameters.append(NumericalContinuousParameter(
            name="Temperature",
            bounds=(temp_min, temp_max)
        ))

        # ADDITIONAL REACTANT EQUIVALENTS
        if num_reactants > 1:
            for i in range(1, num_reactants):
                param_name = f"Reactant_{i+1}_Equivalents"
                constraint_key = f"reactant_{i+1}_eq_range"
                bounds = user_constraints.get(constraint_key) or (0.8, 3.0)
                parameters.append(NumericalContinuousParameter(name=param_name, bounds=bounds))
                logger.info(f"Added dynamic parameter: {param_name} with bounds {bounds}")

        searchspace = SearchSpace.from_product(parameters)
        logger.info(f"Created CHEMISTRY-AWARE search space with {len(parameters)} parameters")

    except Exception as e:
        logger.error(f"Error creating chemistry-aware search space: {e}", exc_info=True)
        return {"error": f"Failed to create BayBE search space: {e}"}

    # ========================================================================
    # PHASE 1: INITIAL SAMPLING WITH FPS
    # ========================================================================
    logger.info(f"Phase 1: FPS-based initial sampling ({num_random_init} iterations)")

    # Create TwoPhaseMetaRecommender for better initial sampling
    initial_recommender = RandomRecommender()
    bayesian_recommender = BotorchRecommender()

    meta_recommender = TwoPhaseMetaRecommender(
        initial_recommender=initial_recommender,
        recommender=bayesian_recommender,
        switch_after=num_random_init  # Switch after initial random phase
    )   

    target = NumericalTarget(name="Yield", mode="MAX", bounds=(0, 100))
    objective = SingleTargetObjective(target)
    campaign = Campaign(searchspace, objective, meta_recommender)

    optimization_history = []
    current_best_yield = 0.0

    # Initialize adaptive batching variables HERE (before the loop)
    recent_improvements = []
    remaining_budget = num_random_init * initial_batch_size  # Total budget estimate

    # Helper function to extract SMILES from SubstanceParameter recommendations
    def get_smiles_from_substance_param(param_name: str, param_value: Any, searchspace: SearchSpace) -> str:
        """
        Extract actual SMILES string from SubstanceParameter recommendation.
        BayBE returns the dictionary KEY, we need to look up the VALUE (SMILES).
        """
        try:
            # Find the parameter in searchspace
            param = next((p for p in searchspace.parameters if p.name == param_name), None)
            
            if param and hasattr(param, 'data'):
                # param_value is something like "Solvent_0"
                # param.data is {"Solvent_0": "C1CCOC1", "Solvent_1": "ClCCl", ...}
                if param_value in param.data:
                    return param.data[param_value]
            
            # Fallback: check if it's already a valid SMILES
            if is_valid_smiles(param_value):
                return param_value
                
        except Exception as e:
            logger.debug(f"Error extracting SMILES for {param_name}={param_value}: {e}")
        
        return "None"

    # Random/FPS exploration with batch evaluation
    for i in range(num_random_init):
        try:
            # Calculate adaptive batch size with proper checks
            if i > 0 and len(optimization_history) > 0:
                # Calculate recent improvement safely
                if len(optimization_history) >= 5:
                    recent_yield_change = optimization_history[-1]['yield'] - optimization_history[-5]['yield']
                else:
                    recent_yield_change = optimization_history[-1]['yield'] - optimization_history[0]['yield']
                recent_improvements.append(recent_yield_change)
            
            # Get adaptive batch size
            batch_size = get_adaptive_batch_size(
                iteration=i,
                total_iterations=num_random_init,
                current_best_yield=current_best_yield,
                remaining_budget=max(1, remaining_budget),
                recent_improvements=recent_improvements if recent_improvements else [0.0]
            )
            
            recommendations = campaign.recommend(batch_size=batch_size)
            
            # Apply diversity constraint (after initial phase)
            if i >= num_random_init // 2:
                recommendations = enforce_diversity_constraint(
                    recommendations=recommendations,
                    campaign=campaign,
                    min_distance_threshold=0.12,
                    max_attempts=5
                )
            
            if recommendations.empty:
                logger.warning(f"No recommendations at iteration {i+1}")
                continue
            
            batch_yields = []
            batch_measurements = recommendations.copy()
            
            # Evaluate entire batch
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                temp = rec['Temperature']
                
                # Extract SMILES from SubstanceParameters using the helper function
                solvent = get_smiles_from_substance_param('Solvent', rec.get('Solvent'), searchspace)
                
                condition_dict = {'solvent': solvent}
                reagent_parts = []
                reagent_eq_parts = []
                sm_parts = []
                sm_eq_parts = []
                catalyst_parts = []
                catalyst_mol_per_parts = []
                
                # Check for structured numerical parameters first
                for param_name, value in rec.items():
                    # For reagents
                    if param_name.startswith('Reagent_') and param_name.endswith('_Equivalents'):
                        reagent_constraints = user_constraints.get('reagents') or []
                        original_smiles = None
                        for r in reagent_constraints:
                            if isinstance(r, dict) and 'smiles' in r:
                                if "".join(ch for ch in r['smiles'] if ch.isalnum()) in param_name:
                                    original_smiles = r['smiles']
                                    break
                            elif isinstance(r, str):
                                if "".join(ch for ch in r if ch.isalnum()) in param_name:
                                    original_smiles = r
                                    break
                        if original_smiles:
                            reagent_parts.append(original_smiles)
                            reagent_eq_parts.append(str(round(float(value), 3)))

                    # For starting materials  
                    elif param_name.startswith('Starting_Material_') and param_name.endswith('_Equivalents'):
                        sm_constraints = user_constraints.get('starting_materials') or []
                        original_smiles = None
                        for sm in sm_constraints:
                            if isinstance(sm, dict) and 'smiles' in sm:
                                if "".join(ch for ch in sm['smiles'] if ch.isalnum()) in param_name:
                                    original_smiles = sm['smiles']
                                    break
                            elif isinstance(sm, str):
                                if "".join(ch for ch in sm if ch.isalnum()) in param_name:
                                    original_smiles = sm
                                    break
                        if original_smiles:
                            sm_parts.append(original_smiles)
                            sm_eq_parts.append(str(round(float(value), 3)))
                            
                    elif param_name.startswith('Catalyst_') and param_name.endswith('_mol_percent'):
                        catalyst_constraints = user_constraints.get('catalysts') or []
                        
                        original_smiles = None
                        for c in catalyst_constraints:
                            if isinstance(c, dict) and 'smiles' in c:
                                if "".join(ch for ch in c['smiles'] if ch.isalnum()) in param_name:
                                    original_smiles = c['smiles']
                                    break
                            elif isinstance(c, str):
                                if "".join(ch for ch in c if ch.isalnum()) in param_name:
                                    original_smiles = c
                                    break
                        
                        if original_smiles:
                            catalyst_parts.append(original_smiles)
                            catalyst_mol_per_parts.append(str(round(float(value), 3)))

                # Fallback to SubstanceParameter if structured params weren't found
                if not reagent_parts and 'Reagent' in rec:
                    reagent = get_smiles_from_substance_param('Reagent', rec.get('Reagent'), searchspace)
                    reagent_equivalents = round(float(rec.get('Reagent_Equivalents', 1.0)), 3)
                    if reagent.lower() != "none":
                        condition_dict['reagent'] = reagent
                        condition_dict['reagent_eq'] = reagent_equivalents
                elif reagent_parts:
                    condition_dict['reagent'] = ".".join(reagent_parts)
                    condition_dict['reagent_eq'] = ",".join(reagent_eq_parts)

                if sm_parts:
                    condition_dict['starting_material'] = ".".join(sm_parts)
                    condition_dict['starting_material_eq'] = ",".join(sm_eq_parts)

                if not catalyst_parts and 'Catalyst' in rec:
                    catalyst = get_smiles_from_substance_param('Catalyst', rec.get('Catalyst'), searchspace)
                    catalyst_mol_per = round(float(rec.get('Catalyst_mol_percent', 1.0)), 3)
                    if catalyst.lower() != "none":
                        condition_dict['catalyst'] = catalyst
                        condition_dict['catalyst_mol_per'] = catalyst_mol_per
                elif catalyst_parts:
                    condition_dict['catalyst'] = ".".join(catalyst_parts)
                    condition_dict['catalyst_mol_per'] = ",".join(catalyst_mol_per_parts)
                    
                if num_reactants > 1:
                    for j in range(1, num_reactants):
                        param_name = f"Reactant_{j+1}_Equivalents"
                        condition_dict[param_name] = round(float(rec.get(param_name, 1.0)), 3)
                
                full_reaction_smiles = merge_reaction_with_conditions(naked_reaction_smiles, condition_dict)
                
                logger.info(f"Iteration {i+1}.{idx+1}: {condition_dict}")
                
                if not full_reaction_smiles:
                    logger.warning(f"Failed to merge conditions, using 0 yield")
                    yield_value = 0.0
                else:
                    yield_value = safe_predict_yield(oracle, full_reaction_smiles)
                    
                batch_yields.append(yield_value)
                current_best_yield = max(current_best_yield, yield_value)
                
                # Store results - use actual SMILES not labels
                hist_entry = {
                    "iteration": i + 1,
                    "batch_index": idx,
                    "phase": "bayesian_optimization",
                    "acquisition_function": "BotorchDefault",
                    "conditions": {
                        "Temperature": round(float(temp), 2),
                        "Solvent": solvent,
                        "Reagent": condition_dict.get('reagent', 'None'),
                        "Catalyst": condition_dict.get('catalyst', 'None'),
                        "Reagent_Equivalents": condition_dict.get('reagent_eq', 0),
                        "Catalyst_mol_percent": condition_dict.get('catalyst_mol_per', 0),
                    },
                    "yield": round(float(yield_value), 2)
                }
                
                optimization_history.append(hist_entry)
            
            # Add measurements
            batch_measurements["Yield"] = batch_yields
            campaign.add_measurements(batch_measurements)
            
            # Update remaining budget AFTER batch evaluation
            remaining_budget -= len(batch_yields)
            
            logger.info(f"Iteration {i+1}/{num_random_init} | "
                f"batch={len(batch_yields)} | "
                f"Best: {max(batch_yields):.2f}% | Overall: {current_best_yield:.2f}%")
            
        except Exception as e:
            logger.error(f"Bayesian iteration {i+1} failed: {e}", exc_info=True)
            continue

    # ========================================================================
    # ENHANCED RESULTS PROCESSING
    # ========================================================================
    if not optimization_history:
        return {"error": "No successful optimization iterations"}

    try:
        best_result = max(optimization_history, key=lambda x: x['yield'])
        
        # Calculate metrics per acquisition function
        acq_functions_used = list(set(h.get('acquisition_function', 'FPS') for h in optimization_history))
        acq_performance = {}
        
        for acq_func in acq_functions_used:
            acq_yields = [h['yield'] for h in optimization_history if h.get('acquisition_function') == acq_func]
            if acq_yields:
                acq_performance[acq_func] = {
                    "avg_yield": round(np.mean(acq_yields), 2),
                    "max_yield": round(max(acq_yields), 2),
                    "evaluations": len(acq_yields)
                }
        
        initial_yields = [h['yield'] for h in optimization_history if h['phase'] == 'initial_sampling']
        bayesian_yields = [h['yield'] for h in optimization_history if h['phase'] == 'bayesian_optimization']

        # Calculate final uncertainty estimate for best result
        uncertainty_estimate = None
        best_uncertainty_metrics = None
        
        try:
            # Get uncertainty for best conditions
            best_conditions_df = pd.DataFrame([best_result['conditions']])
            uncertainty_results = analyze_recommendations_with_uncertainty(
                campaign=campaign,
                recommendations=best_conditions_df,
                current_best=current_best_yield
            )
            
            if uncertainty_results and len(uncertainty_results) > 0 and 'error' not in uncertainty_results[0]:
                best_uncertainty_metrics = uncertainty_results[0]
                uncertainty_estimate = {
                    "mean": best_uncertainty_metrics.get('predicted_yield_mean'),
                    "std": best_uncertainty_metrics.get('predicted_yield_std'),
                    "confidence_interval_95": best_uncertainty_metrics.get('confidence_interval_95'),
                    "coefficient_of_variation": best_uncertainty_metrics.get('coefficient_of_variation'),
                    "recommendation_quality": best_uncertainty_metrics.get('recommendation_quality'),
                    "source": best_uncertainty_metrics.get('uncertainty_source', 'gaussian_process')
                }
                logger.info(f"Final uncertainty: mean={uncertainty_estimate['mean']:.2f}, "
                        f"std={uncertainty_estimate['std']:.2f}, "
                        f"quality={uncertainty_estimate['recommendation_quality']}")
        except Exception as unc_e:
            logger.warning(f"Could not calculate final uncertainty: {unc_e}")

        # LLM analysis
        improvement_suggestions = "LLM analysis was not available or failed."
        if LLM_AVAILABLE:
            try:
                best_cond_dict = best_result['conditions']
                best_conditions_str = (
                    f"- **Temperature:** {best_cond_dict.get('Temperature', 'N/A')}°C\n"
                    f"- **Solvent:** {best_cond_dict.get('Solvent', 'N/A')}\n"
                    f"- **Reagent:** {best_cond_dict.get('Reagent', 'N/A')}\n"
                    f"- **Catalyst:** {best_cond_dict.get('Catalyst', 'N/A')}"
                )

                history_df = pd.DataFrame([h['conditions'] for h in optimization_history])
                history_df['yield'] = [h['yield'] for h in optimization_history]
                history_summary_df = pd.concat([
                    history_df.head(5), 
                    history_df.nlargest(5, 'yield'),
                    history_df.tail(5)
                ]).drop_duplicates().to_string(index=False)
                
                prompt_text = format_prompt(
                    "analyze_bo_results",
                    "task_specific_prompts.yaml",
                    reaction_smiles=naked_reaction_smiles,
                    best_conditions_str=best_conditions_str,
                    max_yield=best_result['yield'],
                    history_summary=history_summary_df
                )

                if prompt_text:
                    logger.info("Generating LLM-based improvement suggestions...")
                    llm_response = generate_text(prompt_text, temperature=0.6)
                    if llm_response:
                        improvement_suggestions = llm_response
                    else:
                        improvement_suggestions = "The LLM could not generate suggestions for this reaction."
                else:
                    improvement_suggestions = "Failed to format the analysis prompt for the LLM."

            except Exception as llm_e:
                logger.error(f"LLM analysis step failed: {llm_e}", exc_info=True)
                improvement_suggestions = f"An error occurred during LLM analysis: {llm_e}"
        
        # Construct the final result object
        final_result = {
            "best_conditions": best_result['conditions'],
            "max_yield": best_result['yield'],
            "improvement_suggestions": improvement_suggestions,
            "optimization_history": optimization_history,
            "search_space_summary": {
                "solvents": final_solvents,
                "reagents": final_reagents,
                "catalysts": final_catalysts,
                "temperature_range": [round(temp_min, 1), round(temp_max, 1)],
                "reagent_eq_range": user_constraints['reagent_eq_range'] or (0.8, 5.0),
                "catalyst_mol_percent_range": user_constraints['catalyst_mol_percent_range'] or (0.1, 10.0),
                "chemistry_aware_encoding": {
                    "solvents": "MORDRED" if use_substance_params['solvents'] else "Structured",
                    "reagents": "RDKIT" if use_substance_params['reagents'] else "Structured",
                    "catalysts": "MORDRED" if use_substance_params['catalysts'] else "Structured"
                }
            },
            "constraints_applied": {
                "temperature_constrained": user_constraints['temperature_range'] is not None,
                "solvents_constrained": user_constraints['solvents'] is not None,
                "reagents_constrained": user_constraints['reagents'] is not None,
                "starting_materials_constrained": user_constraints['starting_materials'] is not None,
                "catalysts_constrained": user_constraints['catalysts'] is not None,
                "reagent_eq_range": user_constraints['reagent_eq_range'] is not None,
                "catalyst_mol_percent_range": user_constraints['catalyst_mol_percent_range'] is not None,
                "user_constraints": user_constraints
            },
            "rxn_insight_analysis": {
                **rxn_insight_metadata,
                "condition_rankings": rxn_insight_metadata.get('top_ranked_conditions', {}),
                "statistical_confidence": rxn_insight_metadata.get('condition_statistics', {}),
                "data_quality": {
                    "similar_reactions_count": rxn_insight_metadata.get('data_sources', 0),
                    "reaction_classification_confidence": rxn_insight_metadata.get('reaction_confidence', 0),
                    "temperature_data_points": rxn_insight_metadata.get('condition_statistics', {}).get('temperature_data_points', 0)
                }
            },
            "performance_metrics": {
                "total_evaluations": len(optimization_history),
                "initial_phase_evaluations": len(initial_yields),
                "bayesian_phase_evaluations": len(bayesian_yields),
                "avg_initial_yield": round(np.mean(initial_yields), 2) if initial_yields else 0,
                "avg_bayesian_yield": round(np.mean(bayesian_yields), 2) if bayesian_yields else 0,
                "acquisition_performance": acq_performance,
                "improvement_over_initial": round(np.mean(bayesian_yields) - np.mean(initial_yields), 2) if bayesian_yields and initial_yields else 0,
                "search_space_enhancement": f"rxn-insight {'enabled' if use_rxn_insight and RXN_INSIGHT_AVAILABLE else 'disabled'}",
                "adaptive_acquisition": "enabled" if use_adaptive_strategy else "disabled",
                "chemistry_aware_descriptors": "enabled",
                "diversity_constraint": "enabled",
                "adaptive_batching": "budget_aware"
            },
            "uncertainty_quantification": {
                "final_uncertainty_estimate": uncertainty_estimate,
                "best_result_uncertainty_metrics": best_uncertainty_metrics,
                "best_result_has_uncertainty": best_uncertainty_metrics is not None,
                "uncertainty_tracked": any('uncertainty_metrics' in h for h in optimization_history),
                "gp_based_uncertainty": uncertainty_estimate is not None and uncertainty_estimate.get('source') == 'gaussian_process'
            }
        }
        return sanitize_for_json(final_result)
        
    except Exception as e:
        logger.error(f"Error processing results: {e}", exc_info=True)
        return {"error": f"Failed to process results: {e}"}