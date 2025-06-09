# core_logic/sourcing_analysis.py
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- Constants ---
# Assuming the script runs from the project root or PYTHONPATH is set correctly
DATA_DIR = Path(__file__).parent.parent / "data"
VENDOR_DB_PATH = DATA_DIR / "vendor_database.json"

# --- Caching ---
_vendor_data = None

def load_vendor_data():
    """Loads the vendor database from JSON, with in-memory caching."""
    global _vendor_data
    if _vendor_data is None:
        try:
            with open(VENDOR_DB_PATH, 'r') as f:
                _vendor_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading vendor database: {e}")
            _vendor_data = {} # Prevent retrying on every call
    return _vendor_data

def find_reagent_suppliers(smiles: str) -> list:
    """
    Finds suppliers for a given SMILES string from the database.
    
    Args:
        smiles: The SMILES string of the reagent.

    Returns:
        A list of supplier dicts, or an empty list if not found.
    """
    db = load_vendor_data()
    # In a real system, you'd canonicalize SMILES before lookup
    return db.get(smiles, [])

def analyze_route_cost_and_sourcing(route_steps: list, target_amount_g: float = 1.0) -> dict:
    """
    Analyzes a list of synthesis steps to determine reagent sources and total cost.

    Args:
        route_steps: A list of step dictionaries from the synthesis plan.
        target_amount_g: The desired amount of the final product in grams.

    Returns:
        A dictionary containing 'total_cost' and 'sourcing_details'.
    """
    total_cost = 0.0
    sourcing_details = {} # Keyed by SMILES

    if not route_steps:
        return {"total_cost": 0, "sourcing_details": {}}
    
    # Simple assumption: 1:1 molar ratio for all reactions, and yields are accounted for.
    # We work backwards from the target amount.
    required_amount = {route_steps[-1]['product']['smiles']: target_amount_g}

    for step in reversed(route_steps):
        product_smiles = step['product']['smiles']
        product_mol = Chem.MolFromSmiles(product_smiles)
        product_mw = Descriptors.MolWt(product_mol)
        
        # Amount of product needed for this step (either final target or intermediate for next step)
        amount_needed_g = required_amount.get(product_smiles, 0)
        
        # Calculate moles of product needed
        moles_needed = amount_needed_g / product_mw if product_mw > 0 else 0
        
        # Based on yield, calculate moles of reactants required
        step_yield = step.get('yield', 100) / 100.0
        moles_required = moles_needed / step_yield if step_yield > 0 else moles_needed * 1.2 # Assume 20% excess if no yield

        for reactant in step['reactants']:
            reactant_smiles = reactant['smiles']
            
            # Update the total amount required for this reactant (in case it's used in multiple steps)
            required_amount.setdefault(reactant_smiles, 0)
            
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            if not reactant_mol: continue
            reactant_mw = Descriptors.MolWt(reactant_mol)
            
            # Amount of this reactant required in grams for this step
            grams_required_for_step = moles_required * reactant_mw
            required_amount[reactant_smiles] += grams_required_for_step
            
            # Find suppliers only if we haven't already
            if reactant_smiles not in sourcing_details:
                suppliers = find_reagent_suppliers(reactant_smiles)
                if suppliers:
                    # Simple strategy: pick the cheapest supplier
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
            # Add cost info to the details dict
            details['required_amount_g'] = total_reagent_amount_g
            details['estimated_cost'] = reagent_cost

    return {
        "total_cost": total_cost,
        "sourcing_details": sourcing_details,
        "assumptions": f"Cost based on a target of {target_amount_g}g of the final product, assuming 1:1 molar ratios and using the cheapest available supplier for each starting material."
    }