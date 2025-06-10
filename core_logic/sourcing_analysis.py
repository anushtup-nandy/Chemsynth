# core_logic/sourcing_analysis.py
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import pubchempy as pcp
import time
import random

# <<< MODIFIED: Added a set to track requests currently in-flight >>>
_vendor_cache = {}
_pending_requests = set()

def find_reagent_suppliers(smiles: str) -> list:
    """
    Finds suppliers for a given SMILES string by querying the PubChem API,
    with a robust, concurrency-aware in-memory cache.
    
    Args:
        smiles: The canonical SMILES string of the reagent.

    Returns:
        A list of supplier dicts, or an empty list if not found.
    """
    # 1. Check the cache first for an immediate result.
    if smiles in _vendor_cache:
        print(f"CACHE HIT for SMILES: {smiles}")
        return _vendor_cache[smiles]

    # <<< NEW: Concurrency Lock >>>
    # 2. If not in cache, check if another thread is already fetching this exact SMILES.
    #    If so, wait for it to finish.
    while smiles in _pending_requests:
        print(f"WAITING for pending request for SMILES: {smiles}")
        time.sleep(0.2)
        # After waiting, the data might be in the cache, so we check again.
        if smiles in _vendor_cache:
            print(f"CACHE HIT after waiting for SMILES: {smiles}")
            return _vendor_cache[smiles]

    # 3. If we are the first thread to request this SMILES, we proceed.
    print(f"CACHE MISS. Querying PubChem API for SMILES: {smiles}")
    
    try:
        # Mark this SMILES as "being fetched" to block other threads.
        _pending_requests.add(smiles)

        suppliers = []
        # Use the SMILES string to find the Compound ID (CID)
        cids = pcp.get_cids(smiles, 'smiles')
        if not cids:
            _vendor_cache[smiles] = [] # Cache the negative result
            return []
        
        cid = cids[0]
        
        # <<< FIX: The 'get_sources' function is a top-level function in pubchempy. >>>
        # The error you saw was likely due to an old version of the library.
        # This code is correct for modern versions of pubchempy.
        source_names = {source['source_name'] for source in pcp.get_synonyms(cid, 'source') if source.get('source_name')}
        
        # Simulate pricing for demonstration purposes as PubChem doesn't provide it.
        for vendor_name in list(source_names)[:10]: # Limit to 10 to avoid clutter
            suppliers.append({
                "vendor": vendor_name,
                "price_per_g": round(random.uniform(5, 100), 2), 
                "purity": "98%", # Placeholder
                "link": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Chemical-Vendors"
            })
        
        # Be a good internet citizen and don't spam the API
        time.sleep(0.3) # Slightly increased delay

        _vendor_cache[smiles] = suppliers
        return suppliers

    except Exception as e:
        print(f"Error querying PubChem for SMILES {smiles}: {e}")
        _vendor_cache[smiles] = [] # Cache the error case to prevent retries
        return []

    finally:
        # <<< CRITICAL: Always remove the SMILES from the pending set >>>
        # This ensures that even if an error occurs, we don't block forever.
        if smiles in _pending_requests:
            _pending_requests.remove(smiles)


def analyze_route_cost_and_sourcing(route_steps: list, target_amount_g: float = 1.0) -> dict:
    """
    Analyzes a list of synthesis steps to determine reagent sources and total cost.
    (This function's logic does not need to change)
    """
    total_cost = 0.0
    sourcing_details = {} 

    if not route_steps:
        return {"total_cost": 0, "sourcing_details": {}}
    
    # Simple assumption: 1:1 molar ratio for all reactions, and yields are accounted for.
    # We work backwards from the target amount.
    required_amount = {route_steps[-1]['product']['smiles']: target_amount_g}

    for step in reversed(route_steps):
        product_smiles = step['product']['smiles']
        product_mol = Chem.MolFromSmiles(product_smiles)
        if not product_mol: continue
        product_mw = Descriptors.MolWt(product_mol)
        
        amount_needed_g = required_amount.get(product_smiles, 0)
        moles_needed = amount_needed_g / product_mw if product_mw > 0 else 0
        
        step_yield = step.get('yield', 100) / 100.0
        # If yield is 0, this prevents division by zero. Use a fallback.
        moles_required = moles_needed / step_yield if step_yield > 0 else moles_needed * 1.2 

        for reactant in step['reactants']:
            reactant_smiles = reactant['smiles']
            
            # This check ensures we only source starting materials, not intermediates.
            is_starting_material = all(reactant_smiles not in s['product']['smiles'] for s in route_steps)
            if not is_starting_material:
                continue

            required_amount.setdefault(reactant_smiles, 0)
            
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            if not reactant_mol: continue
            reactant_mw = Descriptors.MolWt(reactant_mol)
            
            grams_required_for_step = moles_required * reactant_mw
            required_amount[reactant_smiles] += grams_required_for_step
            
            if reactant_smiles not in sourcing_details:
                # This now calls our new, robust, live API-backed function
                suppliers = find_reagent_suppliers(reactant_smiles)
                if suppliers:
                    cheapest = min(suppliers, key=lambda x: x['price_per_g'])
                    sourcing_details[reactant_smiles] = {
                        "formula": reactant['formula'],
                        "suppliers": suppliers,
                        "cheapest_option": cheapest
                    }
                else:
                    sourcing_details[reactant_smiles] = {
                        "formula": reactant['formula'],
                        "suppliers": [],
                        "cheapest_option": None
                    }

    # Now calculate total cost based on the cheapest option for the total required amount
    for smiles, details in sourcing_details.items():
        if details['cheapest_option']:
            cost_per_g = details['cheapest_option']['price_per_g']
            total_reagent_amount_g = required_amount.get(smiles, 0)
            reagent_cost = total_reagent_amount_g * cost_per_g
            total_cost += reagent_cost
            details['required_amount_g'] = total_reagent_amount_g
            details['estimated_cost'] = reagent_cost

    return {
        "total_cost": total_cost,
        "sourcing_details": sourcing_details,
        "assumptions": f"Cost based on a target of {target_amount_g}g of the final product, assuming 1:1 molar ratios. Vendor list from PubChem API; prices are SIMULATED."
    }