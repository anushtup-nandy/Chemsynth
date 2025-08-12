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
from rxn_insight.reaction import Reaction
RXN_INSIGHT_AVAILABLE = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("rxn-insight successfully imported")

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
            'solvents': {},      # {solvent: {count, confidence, sources}}
            'reagents': {},      # {reagent: {count, confidence, sources}}  
            'catalysts': {},     # {catalyst: {count, confidence, sources}}
            'temperatures': [],   # [(temp, confidence, source)]
            'additives': {},     # Additional reagents/bases/acids
            'atmospheres': {},   # Reaction atmosphere conditions
            'times': [],         # Reaction time data
            'pressures': []      # Pressure conditions
        }
        
        # Create Reaction object with enhanced analysis
        rxn = Reaction(naked_reaction_smiles)
        
        # 1. COMPREHENSIVE REACTION ANALYSIS
        logger.info("Performing comprehensive reaction analysis...")
        
        # Get full reaction information
        reaction_info = rxn.get_reaction_info()
        
        # Get functional group analysis
        functional_groups = getattr(rxn, 'get_functional_groups', lambda: {})()
        
        # Get ring analysis
        ring_analysis = getattr(rxn, 'get_ring_changes', lambda: {})()
        
        # Get molecular descriptors
        descriptors = getattr(rxn, 'get_descriptors', lambda: {})()
        
        # Get reaction classification with confidence
        classification = getattr(rxn, 'classify_reaction', lambda: {})()
        
        logger.info(f"Reaction analysis complete - Class: {classification.get('class', 'Unknown')}")
        
        # 2. SIMILARITY-BASED CONDITION MINING
        logger.info("Mining conditions from similar reactions...")
        
        try:
            # Find similar reactions with multiple similarity metrics
            similar_reactions = []
            
            # Try different similarity methods if available
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
            
            # Fallback to basic similarity search
            if not similar_reactions:
                similar_reactions = getattr(rxn, 'find_similar', 
                                          lambda n=50: [])(n=min(50, n_conditions * 3))
            
            logger.info(f"Found {len(similar_reactions)} similar reactions for analysis")
            
            # Process similar reactions for condition extraction
            for idx, similar_rxn in enumerate(similar_reactions):
                try:
                    similarity_score = similar_rxn.get('similarity', 0.5)
                    rxn_conditions = similar_rxn.get('conditions', {})
                    rxn_data = similar_rxn.get('reaction_data', {})
                    
                    # Extract and weight conditions by similarity score
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
                                if 0 <= temp_val <= 500:  # Reasonable temperature range
                                    conditions_aggregator['temperatures'].append((temp_val, confidence, source))
                            except (ValueError, TypeError):
                                pass
                    
                    # Process additional conditions
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
            # Get reaction center functional groups
            reactant_fg = functional_groups.get('reactants', {})
            product_fg = functional_groups.get('products', {})
            
            # Analyze functional group changes for mechanism insights
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
            
            # Use functional group database to suggest conditions
            fg_condition_db = getattr(rxn, 'get_fg_conditions', lambda fg_dict: {})(fg_changes)
            
            # Integrate functional group-based conditions
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
        
        # Rank and filter conditions by statistical significance
        def rank_conditions(condition_dict, min_count=1, max_results=None):
            """Rank conditions by weighted score combining count and confidence"""
            ranked = []
            for name, data in condition_dict.items():
                # Calculate composite score
                avg_confidence = data['confidence'] / max(data['count'], 1)
                popularity_score = min(data['count'] / 10.0, 1.0)  # Normalize count
                composite_score = (0.7 * avg_confidence) + (0.3 * popularity_score)
                
                if data['count'] >= min_count:
                    ranked.append({
                        'name': name,
                        'score': composite_score,
                        'count': data['count'],
                        'avg_confidence': avg_confidence,
                        'sources': len(set(data['sources']))
                    })
            
            # Sort by composite score
            ranked.sort(key=lambda x: x['score'], reverse=True)
            
            if max_results:
                ranked = ranked[:max_results]
            
            return ranked
        
        # Rank all condition types
        ranked_solvents = rank_conditions(conditions_aggregator['solvents'], min_count=1, max_results=15)
        ranked_reagents = rank_conditions(conditions_aggregator['reagents'], min_count=1, max_results=12)
        ranked_catalysts = rank_conditions(conditions_aggregator['catalysts'], min_count=1, max_results=12)
        ranked_atmospheres = rank_conditions(conditions_aggregator['atmospheres'], min_count=1, max_results=5)
        
        # Statistical temperature analysis
        temp_analysis = {}
        if conditions_aggregator['temperatures']:
            temps = [t[0] for t in conditions_aggregator['temperatures']]
            confidences = [t[1] for t in conditions_aggregator['temperatures']]
            
            # Weighted statistics
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
        
        # Extract top conditions
        final_conditions = {
            'solvents': [item['name'] for item in ranked_solvents],
            'reagents': [item['name'] for item in ranked_reagents],
            'catalysts': [item['name'] for item in ranked_catalysts],
            'temperatures': list(temp_analysis.get('recommended_range', [])),
            'atmospheres': [item['name'] for item in ranked_atmospheres],
            'temperature_analysis': temp_analysis
        }
        
        # Comprehensive metadata
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
                                 batch_size: int = 1) -> Campaign:
    """
    Switch to Bayesian optimization with specified acquisition function and batch size.
    Returns a new campaign object with the transferred measurements.
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
        
        # Create new recommender
        bayesian_recommender = BotorchRecommender(
            surrogate_model=GaussianProcessSurrogate(),
            acquisition_function=acq_func
        )
        
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
            logger.info(f"Transferred {len(existing_measurements)} measurements to {acquisition_type} campaign")
        
        logger.info(f"Successfully created new {acquisition_type} campaign with batch_size={batch_size}")
        return new_campaign
        
    except Exception as e:
        logger.warning(f"Failed to switch to Bayesian recommender: {e}")
        return campaign  # Return original campaign on failure
    
def get_adaptive_acquisition_strategy(iteration: int, total_iterations: int, 
                                    current_best_yield: float) -> Tuple[str, int]:
    """
    Dynamically select acquisition function and batch size based on optimization progress.
    
    Returns: (acquisition_type, batch_size)
    """
    progress = iteration / total_iterations
    
    # Early stage: exploration with batch sampling
    if progress < 0.4:          # MORE EXPLORATION TIME
        return ("qEI", 3)       # LARGER BATCHES
    elif progress < 0.8:        # LONGER BALANCED PHASE
        if current_best_yield < 60:  # LOWER THRESHOLD
            return ("UCB", 2)   # BATCH UCB FOR MORE EXPLORATION
        else:
            return ("EI", 1)
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
    initial_batch_size: int = 2,
    use_rxn_insight: bool = True,
    prior_knowledge: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced Bayesian optimization with rxn-insight integration for expanded condition space.
    """
    logger.info(f"Starting ENHANCED Bayesian Optimization for: {naked_reaction_smiles}")
    
    if not naked_reaction_smiles or not naked_reaction_smiles.strip():
        return {"error": "Invalid or empty reaction SMILES provided"}
    
    # Initialize Oracle
    oracle = initialize_oracle()
    if oracle is None:
        return {"error": "Could not initialize the simulation oracle"}
    
    constraint_temp_range = None
    constraint_solvents = None
    constraint_reagents = None
    constraint_catalysts = None

    # ENHANCED Generate Search Space with rxn-insight integration
    try:
        logger.info(f"Generating enhanced search space...")
        
        # Get initial conditions from existing RCR model
        initial_conditions, _ = oracle.condition_predictor.get_n_conditions(
            naked_reaction_smiles, n=num_initial_candidates, return_scores=True
        )
        
        if not initial_conditions:
            return {"error": "Could not generate initial candidate conditions"}

        # Process RCR model conditions
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
        
        # ENHANCED: Advanced rxn-insight integration with intelligent merging
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
                    solvents = list(rxn_solvents) + [s for s in solvents if s not in rxn_solvents]
                    reagents = list(rxn_reagents) + [r for r in reagents if r not in rxn_reagents]
                    catalysts = list(rxn_catalysts) + [c for c in catalysts if c not in rxn_catalysts]
                    
                    # Advanced temperature integration
                    temp_analysis = rxn_insight_conditions.get('temperature_analysis', {})
                    if temp_analysis and 'recommended_range' in temp_analysis:
                        rxn_temp_range = temp_analysis['recommended_range']
                        if len(rxn_temp_range) == 2:
                            # Use rxn-insight temperature range if statistically significant
                            data_points = temp_analysis.get('data_points', 0)
                            if data_points >= 3:  # Sufficient data
                                temp_min, temp_max = rxn_temp_range
                                logger.info(f"Using rxn-insight temperature range: {temp_min:.1f}-{temp_max:.1f}°C "
                                           f"(based on {data_points} similar reactions)")
                    
                    logger.info(f"Enhanced search space: {len(solvents)} solvents, "
                               f"{len(reagents)} reagents, {len(catalysts)} catalysts")
                    logger.info(f"Data sources: {rxn_insight_metadata.get('data_sources', 0)} similar reactions analyzed")
                else:
                    logger.warning("rxn-insight analysis unsuccessful, using RCR-only conditions")
                
            except Exception as e:
                logger.warning(f"Advanced rxn-insight integration failed: {e}")
                rxn_insight_metadata = {'analysis_successful': False, 'error': str(e)}
        
        # Apply intelligent filtering based on rxn-insight rankings
        if use_rxn_insight and rxn_insight_metadata.get('analysis_successful', False):
            # Use statistical ranking from rxn-insight instead of arbitrary limits
            top_conditions = rxn_insight_metadata.get('top_ranked_conditions', {})
            
            # Prioritize statistically significant conditions
            if 'solvents' in top_conditions:
                priority_solvents = [item['name'] for item in top_conditions['solvents']]
                other_solvents = [s for s in solvents if s not in priority_solvents and s and len(s) < 50]
                solvents = priority_solvents + other_solvents[:max(0, 25-len(priority_solvents))]
            else:
                solvents = sorted([s for s in solvents if s and len(s) < 50])[:25]
            
            if 'reagents' in top_conditions:
                priority_reagents = [item['name'] for item in top_conditions['reagents']]
                other_reagents = [r for r in reagents if r not in priority_reagents and r and len(r) < 50]
                reagents = priority_reagents + other_reagents[:max(0, 20-len(priority_reagents))]
            else:
                reagents = sorted([r for r in reagents if r and len(r) < 50])[:20]
            
            if 'catalysts' in top_conditions:
                priority_catalysts = [item['name'] for item in top_conditions['catalysts']]
                other_catalysts = [c for c in catalysts if c not in priority_catalysts and c and len(c) < 50]
                catalysts = priority_catalysts + other_catalysts[:max(0, 20-len(priority_catalysts))]
            else:
                catalysts = sorted([c for c in catalysts if c and len(c) < 50])[:20]
        else:
            # Fallback to larger limits when rxn-insight unavailable
            solvents = sorted([s for s in solvents if s and len(s) < 50])[:25]   
            reagents = sorted([r for r in reagents if r and len(r) < 50])[:30]  
            catalysts = sorted([c for c in catalysts if c and len(c) < 50])[:25]

        if prior_knowledge and 'constraints' in prior_knowledge:
            constraints = prior_knowledge['constraints']
            logger.info("Applying user-defined search space constraints.")
            
            if 'solvents' in constraints and constraints['solvents']:
                solvents = constraints['solvents']
                logger.info(f"Overriding solvents with: {solvents}")
            if 'reagents' in constraints and constraints['reagents']:
                reagents = constraints['reagents']
                logger.info(f"Overriding reagents with: {reagents}")
            if 'catalysts' in constraints and constraints['catalysts']:
                catalysts = constraints['catalysts']
                logger.info(f"Overriding catalysts with: {catalysts}")
            if 'temperature_range' in constraints and len(constraints['temperature_range']) == 2:
                temp_min, temp_max = constraints['temperature_range']
                logger.info(f"Overriding temperature range with: {temp_min}°C to {temp_max}°C")
        
        # Add "None" options for optional components
        if "None" not in reagents:
            reagents.append("None")
        if "None" not in catalysts:
            catalysts.append("None")
            
        # Ensure we have fallback options
        if not solvents:
            solvents = [
                "C1CCOC1",      # THF
                "CCOCC",        # Diethyl ether  
                "ClCCl",        # DCM
                "c1ccccc1",     # Benzene/Toluene
                "CC(C)=O",      # Acetone
                "CN(C)C=O",     # DMF
                "CCCCCC",       # Hexane
                "CCO",          # Ethanol
                "CO",           # Methanol
                "ClC(Cl)Cl"     # Chloroform
            ]

        if not reagents:
            reagents = [
                "None", "O", "Cl", "F", "Br", "I",           # Basic reagents
                "[Li+]", "[Na+]", "[K+]",                    # Metal cations
                "CCN(CC)CC",                                  # Triethylamine
                "C1CCC2=NCCCN2CC1",                         # DBU
                "ClC(=O)C(=O)Cl",                           # Oxalyl chloride
                "CC(C)(C)OC(=O)N",                          # Boc anhydride
                "ClS(=O)(=O)c1ccc(C)cc1"                    # TsCl
            ]

        if not catalysts:
            catalysts = [
                "None",
                "[Pd]",                    # Palladium
                "[Ni]", 
                "[Cu]",
                "c1ccc(P(c2ccccc2)c2ccccc2)cc1",  # PPh3
                "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21"  # BINAP
            ]
        
        # Enhanced temperature range
        if temps:
            temps = [t for t in temps if -100 <= t <= 400]  # Filter reasonable temperatures
            temp_min, temp_max = min(temps), max(temps)
            if temp_max - temp_min < 10.0:  # Ensure reasonable range
                temp_center = (temp_min + temp_max) / 2
                temp_min = max(0, temp_center - 25)
                temp_max = min(300, temp_center + 25)
        else:
            temp_min, temp_max = -100.0, 200.0
        
        logger.info(f"Final search space: {len(solvents)} solvents, {len(reagents)} reagents, "
                   f"{len(catalysts)} catalysts, temp range: {temp_min:.1f}-{temp_max:.1f}°C")

    except Exception as e:
        logger.error(f"Error generating enhanced search space: {e}", exc_info=True)
        return {"error": f"Failed to generate search space: {e}"}

    # Create Search Space (keep existing logic but log the enhanced space)
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
        logger.info(f"Created search space with {len(parameters)} parameters")

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

    if prior_knowledge and 'data_points' in prior_knowledge and prior_knowledge['data_points']:
        logger.info(f"Adding {len(prior_knowledge['data_points'])} prior data points to the campaign.")
        try:
            prior_data_list = []
            for point in prior_knowledge['data_points']:
                row = point['conditions'].copy()
                row['Yield'] = point['yield']
                prior_data_list.append(row)
            
            # Ensure all required columns are present, filling with 'None' if necessary
            for param in searchspace.continuous.keys():
                for row in prior_data_list:
                    row.setdefault(param, np.nan)
            for param in searchspace.categorical.keys():
                 for row in prior_data_list:
                    row.setdefault(param, "None") # A reasonable default

            prior_df = pd.DataFrame(prior_data_list)
            
            # Add these known data points to the campaign
            bayesian_campaign.add_measurements(prior_df)
            logger.info("Successfully added prior data points.")
            
            # Also add to history for tracking
            for i, row in prior_df.iterrows():
                optimization_history.append({
                    "iteration": 0, "batch_index": i, "phase": "prior_knowledge",
                    "conditions": {k: v for k, v in row.items() if k != 'Yield'},
                    "yield": row['Yield']
                })
                current_best_yield = max(current_best_yield, row['Yield'])

        except Exception as e:
            logger.error(f"Failed to add prior data points: {e}", exc_info=True)
    
    # Adaptive Bayesian optimization phase
    for i in range(num_random_init, num_iterations):
        try:
            # Adaptive strategy selection
            if use_adaptive_strategy:
                new_acquisition, new_batch_size = get_adaptive_acquisition_strategy(
                    i - num_random_init + 1, num_iterations - num_random_init, current_best_yield
                )
                if new_acquisition != current_acquisition or new_batch_size != current_batch_size:
                    logger.info(f"Switching to {new_acquisition} with batch_size={new_batch_size}")
                    bayesian_campaign = switch_to_bayesian_recommender(
                        bayesian_campaign, new_acquisition, new_batch_size
                    )
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
            "rxn_insight_analysis": {
                **rxn_insight_metadata,  # Include all the rich metadata
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
                "random_phase_evaluations": len(random_yields),
                "bayesian_phase_evaluations": len(bayesian_yields),
                "avg_random_yield": round(np.mean(random_yields), 2) if random_yields else 0,
                "avg_bayesian_yield": round(np.mean(bayesian_yields), 2) if bayesian_yields else 0,
                "acquisition_performance": acq_performance,
                "improvement_over_random": round(np.mean(bayesian_yields) - np.mean(random_yields), 2) if bayesian_yields and random_yields else 0,
                "search_space_enhancement": f"rxn-insight {'enabled' if use_rxn_insight and RXN_INSIGHT_AVAILABLE else 'disabled'}"
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing results: {e}", exc_info=True)
        return {"error": f"Failed to process results: {e}"}

# Test function
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    test_reaction = "Cl[Si](Cl)(Cl)Cl.c1ccccc1[Mg]Br>>Cl[Si](Cl)(Cl)c1ccccc1"
    
    print("\n" + "="*70)
    print(f"RUNNING ENHANCED BAYESIAN OPTIMIZATION WITH RXN-INSIGHT FOR:\n{test_reaction}")
    print("="*70 + "\n")
    
    results = run_true_bayesian_optimization(
        naked_reaction_smiles=test_reaction,
        num_initial_candidates=40,     
        num_iterations=30,             
        num_random_init=10,             
        use_adaptive_strategy=True,
        initial_batch_size=4,           
        use_rxn_insight=True
    )

    import json
    print(json.dumps(results, indent=2))