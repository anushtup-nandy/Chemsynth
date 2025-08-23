import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict, Counter
import requests
import base64
from chemicals import CAS_from_any, Tb, Tm, Tc, Hfs, Hfl, Hfg, S0s, S0l, S0g
import logging
from chempy import balance_stoichiometry, Substance
from chempy.chemistry import Reaction
CHEMPY_AVAILABLE = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_smiles_from_name(name):
    """
    Fetch the SMILES string of a molecule by its common name from PubChem.

    Parameters:
    name (str): The common name of the molecule.

    Returns:
    str: The SMILES string of the molecule, or an error message if not found.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return smiles
    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}"
    except requests.exceptions.RequestException as err:
        return f"Request Exception: {err}"
    except (KeyError, IndexError):
        return "No data found or error occurred."


def count_atoms(smiles):
    """
    Count the number of each type of atom in a SMILES string, including hydrogen atoms.

    Parameters:
    smiles (str): The SMILES string of the molecule.

    Returns:
    dict: A dictionary with atom symbols as keys and their counts as values.
    """
    mol = Chem.MolFromSmiles(smiles)
    atom_counts = defaultdict(int)
    if mol:
        mol = Chem.AddHs(mol)
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1
    return dict(atom_counts)


def get_molecular_formula(smiles):
    """
    Get the molecular formula of a molecule from its SMILES string.

    Parameters:
    smiles (str): The SMILES string of the molecule.

    Returns:
    str: The molecular formula of the molecule, or an error message if invalid.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        return rdMolDescriptors.CalcMolFormula(molecule)
    else:
        return "Invalid SMILES string"


def smiles_to_formula(smiles):
    """
    Convert SMILES to molecular formula for chempy.
    
    Parameters:
    smiles (str): The SMILES string
    
    Returns:
    str: Molecular formula compatible with chempy
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except:
        return None


def balance_chemical_equation(reactant_smiles, product_smiles):
    """
    Balance a chemical equation given reactants and products as SMILES strings using ChemPy.
    Can handle both determined and underdetermined systems.

    Parameters:
    reactant_smiles (list): A list of SMILES strings for the reactants.
    product_smiles (list): A list of SMILES strings for the products.

    Returns:
    tuple: Two lists containing tuples of stoichiometric coefficients and molecular formulas for reactants and products.
    """
    if not CHEMPY_AVAILABLE:
        raise ImportError("ChemPy is required for advanced balancing. Install with: pip install chempy")
    
    try:
        # Convert SMILES to molecular formulas
        reactant_formulas = []
        for smiles in reactant_smiles:
            formula = smiles_to_formula(smiles)
            if formula is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            reactant_formulas.append(formula)
        
        product_formulas = []
        for smiles in product_smiles:
            formula = smiles_to_formula(smiles)
            if formula is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            product_formulas.append(formula)
        
        logger.info(f"Balancing reaction: {' + '.join(reactant_formulas)} -> {' + '.join(product_formulas)}")
        
        # Use chempy to balance the equation
        # Convert to sets as required by chempy
        reactants_set = set(reactant_formulas)
        products_set = set(product_formulas)
        
        # Try balancing with underdetermined=True to handle incomplete reactions
        try:
            balanced_reactants, balanced_products = balance_stoichiometry(
                reactants_set, 
                products_set, 
                underdetermined=True
            )
            
            logger.info("Successfully balanced using underdetermined=True")
            
        except Exception as e:
            logger.warning(f"Underdetermined balancing failed: {e}")
            # Fall back to regular balancing
            try:
                balanced_reactants, balanced_products = balance_stoichiometry(
                    reactants_set, 
                    products_set, 
                    underdetermined=False
                )
                logger.info("Successfully balanced using regular method")
            except Exception as e2:
                logger.error(f"Both balancing methods failed: {e2}")
                # Return 1:1 stoichiometry as fallback
                reactant_data = [(1, get_molecular_formula(smiles)) for smiles in reactant_smiles]
                product_data = [(1, get_molecular_formula(smiles)) for smiles in product_smiles]
                logger.warning("Falling back to 1:1 stoichiometry")
                return reactant_data, product_data
        
        # Convert the results back to the expected format
        reactant_data = []
        for smiles, formula in zip(reactant_smiles, reactant_formulas):
            coeff = balanced_reactants.get(formula, 1)
            # Handle symbolic coefficients from underdetermined systems
            if hasattr(coeff, 'evalf'):
                # If it's a symbolic expression, evaluate it numerically
                try:
                    coeff_val = float(coeff.evalf())
                    if coeff_val <= 0:
                        coeff_val = 1  # Ensure positive coefficient
                    coeff = int(round(coeff_val))
                except:
                    coeff = 1
            elif isinstance(coeff, (int, float)):
                coeff = int(max(1, round(coeff)))
            else:
                coeff = 1
            
            reactant_data.append((coeff, formula))
        
        product_data = []
        for smiles, formula in zip(product_smiles, product_formulas):
            coeff = balanced_products.get(formula, 1)
            # Handle symbolic coefficients from underdetermined systems
            if hasattr(coeff, 'evalf'):
                try:
                    coeff_val = float(coeff.evalf())
                    if coeff_val <= 0:
                        coeff_val = 1
                    coeff = int(round(coeff_val))
                except:
                    coeff = 1
            elif isinstance(coeff, (int, float)):
                coeff = int(max(1, round(coeff)))
            else:
                coeff = 1
                
            product_data.append((coeff, formula))
        
        logger.info(f"Final balanced coefficients - Reactants: {reactant_data}, Products: {product_data}")
        return reactant_data, product_data
        
    except Exception as e:
        logger.error(f"Error in balance_chemical_equation: {e}")
        # Return 1:1 stoichiometry as ultimate fallback
        reactant_data = [(1, get_molecular_formula(smiles)) for smiles in reactant_smiles]
        product_data = [(1, get_molecular_formula(smiles)) for smiles in product_smiles]
        return reactant_data, product_data


def balance_chemical_equation_advanced(reactant_smiles, product_smiles, allow_incomplete=True):
    """
    Advanced balancing that can handle incomplete/underdetermined reactions.
    Returns multiple possible solutions for underdetermined systems.
    
    Parameters:
    reactant_smiles (list): A list of SMILES strings for the reactants.
    product_smiles (list): A list of SMILES strings for the products.
    allow_incomplete (bool): Whether to allow balancing of incomplete reactions.
    
    Returns:
    list: A list of possible balanced reactions, each containing reactant and product data.
    """
    if not CHEMPY_AVAILABLE:
        # Fall back to the original method
        try:
            reactant_data, product_data = balance_chemical_equation(reactant_smiles, product_smiles)
            return [{"reactants": reactant_data, "products": product_data, "method": "fallback"}]
        except Exception as e:
            logger.error(f"Fallback balancing failed: {e}")
            return []
    
    try:
        # Convert SMILES to molecular formulas
        reactant_formulas = [smiles_to_formula(smiles) for smiles in reactant_smiles]
        product_formulas = [smiles_to_formula(smiles) for smiles in product_smiles]
        
        if None in reactant_formulas or None in product_formulas:
            raise ValueError("Invalid SMILES strings found")
        
        # Try different balancing approaches
        solutions = []
        
        # Method 1: Standard balancing
        try:
            balanced_r, balanced_p = balance_stoichiometry(
                set(reactant_formulas), 
                set(product_formulas), 
                underdetermined=False
            )
            
            reactant_data = [(balanced_r.get(f, 1), f) for f in reactant_formulas]
            product_data = [(balanced_p.get(f, 1), f) for f in product_formulas]
            
            solutions.append({
                "reactants": reactant_data,
                "products": product_data,
                "method": "standard"
            })
            
        except Exception as e:
            logger.info(f"Standard balancing failed: {e}")
        
        # Method 2: Underdetermined balancing (for incomplete reactions)
        if allow_incomplete:
            try:
                balanced_r, balanced_p = balance_stoichiometry(
                    set(reactant_formulas), 
                    set(product_formulas), 
                    underdetermined=True
                )
                
                # Process symbolic solutions
                reactant_data = []
                for formula in reactant_formulas:
                    coeff = balanced_r.get(formula, 1)
                    if hasattr(coeff, 'free_symbols') and coeff.free_symbols:
                        # For parametric solutions, use a default value
                        coeff = 1
                    elif hasattr(coeff, 'evalf'):
                        coeff = int(max(1, round(float(coeff.evalf()))))
                    else:
                        coeff = max(1, int(coeff))
                    reactant_data.append((coeff, formula))
                
                product_data = []
                for formula in product_formulas:
                    coeff = balanced_p.get(formula, 1)
                    if hasattr(coeff, 'free_symbols') and coeff.free_symbols:
                        coeff = 1
                    elif hasattr(coeff, 'evalf'):
                        coeff = int(max(1, round(float(coeff.evalf()))))
                    else:
                        coeff = max(1, int(coeff))
                    product_data.append((coeff, formula))
                
                solutions.append({
                    "reactants": reactant_data,
                    "products": product_data,
                    "method": "underdetermined"
                })
                
            except Exception as e:
                logger.info(f"Underdetermined balancing failed: {e}")
        
        # If no solutions found, return 1:1 stoichiometry
        if not solutions:
            reactant_data = [(1, get_molecular_formula(smiles)) for smiles in reactant_smiles]
            product_data = [(1, get_molecular_formula(smiles)) for smiles in product_smiles]
            solutions.append({
                "reactants": reactant_data,
                "products": product_data,
                "method": "1:1_fallback"
            })
        
        return solutions
        
    except Exception as e:
        logger.error(f"Advanced balancing failed: {e}")
        # Ultimate fallback
        reactant_data = [(1, get_molecular_formula(smiles)) for smiles in reactant_smiles]
        product_data = [(1, get_molecular_formula(smiles)) for smiles in product_smiles]
        return [{"reactants": reactant_data, "products": product_data, "method": "error_fallback"}]


def display_reaction(reactants, products):
    """
    Format and display the chemical reaction.

    Parameters:
    reactants (list): A list of tuples for reactants with coefficients and molecular formulas.
    products (list): A list of tuples for products with coefficients and molecular formulas.

    Returns:
    str: The formatted chemical reaction as a string.
    """
    def format_component(component):
        try:
            coefficient, molecule = component
            return f"{coefficient} {molecule}" if coefficient != 1 else molecule
        except ValueError:
            raise ValueError(f"Invalid component format: {component}. Expected a tuple of (coefficient, molecule).")

    if not reactants or not products:
        raise ValueError("Both reactants and products need at least one component.")

    try:
        reactants_str = ' + '.join(format_component(r) for r in reactants)
        products_str = ' + '.join(format_component(p) for p in products)
        return f"{reactants_str} → {products_str}"
    except ValueError as e:
        logger.error(e)
        return None


def create_reaction_string(reactants, products):
    """
    Create a reaction string for visualization purposes.

    Parameters:
    reactants (list): A list of SMILES strings for the reactants.
    products (list): A list of SMILES strings for the products.

    Returns:
    str: A string representing the chemical reaction in the format 'reactants>>products'.
    """
    reactants_str = '.'.join(reactants)
    products_str = '.'.join(products)
    return f"{reactants_str}>>{products_str}"


def display_svg(svg):
    """
    Convert SVG to base64 encoded HTML image tag for display.

    Parameters:
    svg (str): The SVG content as a string.

    Returns:
    str: HTML image tag with base64 encoded SVG.
    """
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = f"<img src='data:image/svg+xml;base64,{b64}'/>"
    return html


def compound_state(compound, temp):
    """
    Determine the physical state of a compound at a given temperature.

    Parameters:
    compound (str): The name or identifier of the compound.
    temp (float): The temperature in Kelvin.

    Returns:
    str: The physical state of the compound ('solid', 'liquid', or 'gas').
    """
    try:
        CAS_compound = CAS_from_any(compound)
        if not CAS_compound:
            raise ValueError(f"Could not find CAS number for compound: {compound}")
            
        boiling_p = Tb(CAS_compound)
        melting_p = Tm(CAS_compound)
        
        if boiling_p is None or melting_p is None:
            logger.warning(f"Missing temperature data for {compound} (CAS: {CAS_compound})")
            return 'unknown'

        if float(temp) <= float(melting_p):
            return 'solid'
        elif float(temp) >= float(boiling_p):
            return 'gas'
        else:
            return 'liquid'
    except Exception as e:
        logger.error(f"Error determining state for {compound}: {e}")
        return 'unknown'


def enthalpy(coeff, compound, state):
    """
    Calculate the enthalpy of a compound in a specific state.

    Parameters:
    coeff (float): The stoichiometric coefficient of the compound.
    compound (str): The name or identifier of the compound.
    state (str): The physical state of the compound ('solid', 'liquid', or 'gas').

    Returns:
    float: The enthalpy of the compound, or 0 if data unavailable.
    """
    try:
        Cas_compound = CAS_from_any(compound)
        if not Cas_compound:
            logger.warning(f"Could not find CAS number for compound: {compound}")
            return 0
            
        if state == 'solid': 
            enthalpy_value = Hfs(Cas_compound)
        elif state == 'liquid':
            enthalpy_value = Hfl(Cas_compound)
        else:  # gas
            enthalpy_value = Hfg(Cas_compound)
        
        if enthalpy_value is None:
            logger.warning(f"No enthalpy data available for {compound} in {state} state")
            return 0
        else: 
            return float(coeff) * enthalpy_value
    except Exception as e:
        logger.error(f"Error calculating enthalpy for {compound}: {e}")
        return 0


def entropy(coeff, compound, state):
    """
    Calculate the entropy of a compound in a specific state.

    Parameters:
    coeff (float): The stoichiometric coefficient of the compound.
    compound (str): The name or identifier of the compound.
    state (str): The physical state of the compound ('solid', 'liquid', or 'gas').

    Returns:
    float: The entropy of the compound, or 0 if data unavailable.
    """
    try:
        Cas_compound = CAS_from_any(compound)
        if not Cas_compound:
            logger.warning(f"Could not find CAS number for compound: {compound}")
            return 0
            
        if state == 'solid': 
            entropy_value = S0s(Cas_compound)
        elif state == 'liquid':
            entropy_value = S0l(Cas_compound)
        else:  # gas
            entropy_value = S0g(Cas_compound)
        
        if entropy_value is None:
            logger.warning(f"No entropy data available for {compound} in {state} state")
            return 0
        else: 
            return float(coeff) * entropy_value
    except Exception as e:
        logger.error(f"Error calculating entropy for {compound}: {e}")
        return 0


# Convenience functions for easier use
def balance_reaction_from_names(reactant_names, product_names):
    """
    Balance a chemical equation using compound names instead of SMILES.
    
    Parameters:
    reactant_names (list): List of reactant compound names.
    product_names (list): List of product compound names.
    
    Returns:
    tuple: (reactant_data, product_data, reaction_string)
    """
    try:
        # Convert names to SMILES
        reactant_smiles = []
        for name in reactant_names:
            smiles = get_smiles_from_name(name)
            if "Error" in smiles or "No data" in smiles:
                raise ValueError(f"Could not resolve compound name: {name}")
            reactant_smiles.append(smiles)
        
        product_smiles = []
        for name in product_names:
            smiles = get_smiles_from_name(name)
            if "Error" in smiles or "No data" in smiles:
                raise ValueError(f"Could not resolve compound name: {name}")
            product_smiles.append(smiles)
        
        # Balance the equation
        reactant_data, product_data = balance_chemical_equation(reactant_smiles, product_smiles)
        
        # Create reaction string for display
        reaction_string = display_reaction(reactant_data, product_data)
        
        return reactant_data, product_data, reaction_string
        
    except Exception as e:
        logger.error(f"Error balancing reaction from names: {e}")
        raise


def calculate_reaction_thermodynamics(reactant_data, product_data, reactant_names, product_names, temperature=298.15):
    """
    Calculate the enthalpy and entropy changes for a balanced reaction.
    
    Parameters:
    reactant_data (list): List of (coefficient, formula) tuples for reactants.
    product_data (list): List of (coefficient, formula) tuples for products.
    reactant_names (list): List of reactant compound names.
    product_names (list): List of product compound names.
    temperature (float): Temperature in Kelvin (default: 298.15 K).
    
    Returns:
    dict: Dictionary containing enthalpy change, entropy change, and Gibbs free energy change.
    """
    try:
        # Calculate enthalpy change
        reactant_enthalpy = 0
        for i, (coeff, formula) in enumerate(reactant_data):
            state = compound_state(reactant_names[i], temperature)
            reactant_enthalpy += enthalpy(coeff, reactant_names[i], state)
        
        product_enthalpy = 0
        for i, (coeff, formula) in enumerate(product_data):
            state = compound_state(product_names[i], temperature)
            product_enthalpy += enthalpy(coeff, product_names[i], state)
        
        delta_h = product_enthalpy - reactant_enthalpy
        
        # Calculate entropy change
        reactant_entropy = 0
        for i, (coeff, formula) in enumerate(reactant_data):
            state = compound_state(reactant_names[i], temperature)
            reactant_entropy += entropy(coeff, reactant_names[i], state)
        
        product_entropy = 0
        for i, (coeff, formula) in enumerate(product_data):
            state = compound_state(product_names[i], temperature)
            product_entropy += entropy(coeff, product_names[i], state)
        
        delta_s = product_entropy - reactant_entropy
        
        # Calculate Gibbs free energy change
        delta_g = delta_h - temperature * delta_s
        
        return {
            'delta_h': delta_h,  # J/mol
            'delta_s': delta_s,  # J/(mol·K)
            'delta_g': delta_g,  # J/mol
            'temperature': temperature,
            'spontaneous': delta_g < 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating thermodynamics: {e}")
        return None


def test_chempy_capabilities():
    """
    Test the capabilities of chempy with various reaction types.
    """
    if not CHEMPY_AVAILABLE:
        print("ChemPy not available. Install with: pip install chempy")
        return
    
    print("=== Testing ChemPy Balancing Capabilities ===\n")
    
    # Test 1: Standard reaction (methane combustion)
    print("Test 1: Complete reaction (Methane combustion)")
    try:
        reactants = ["C", "O=O"]  # CH4 + O2
        products = ["O=C=O", "O"]  # CO2 + H2O
        
        solutions = balance_chemical_equation_advanced(reactants, products)
        for i, sol in enumerate(solutions):
            print(f"  Solution {i+1} ({sol['method']}): {sol['reactants']} -> {sol['products']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Underdetermined reaction
    print("\nTest 2: Underdetermined reaction (Carbon + Oxygen)")
    try:
        reactants = ["C", "O=O"]  # C + O2
        products = ["O=C=O", "[C-]#[O+]"]  # CO2 + CO (underdetermined)
        
        solutions = balance_chemical_equation_advanced(reactants, products)
        for i, sol in enumerate(solutions):
            print(f"  Solution {i+1} ({sol['method']}): {sol['reactants']} -> {sol['products']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*50)


# Example usage function
def example_usage():
    """
    Demonstrate the usage of the balance_reaction module with ChemPy.
    """
    print("=== Chemical Reaction Balancer with ChemPy ===\n")
    
    # Test ChemPy capabilities first
    test_chempy_capabilities()
    
    # Example 1: Balance methane combustion using names
    print("Example 1: Methane Combustion (using names)")
    try:
        reactants = ["methane", "oxygen"]
        products = ["carbon dioxide", "water"]
        
        reactant_data, product_data, reaction_string = balance_reaction_from_names(reactants, products)
        print(f"Balanced equation: {reaction_string}")
        
        # Calculate thermodynamics
        thermo = calculate_reaction_thermodynamics(reactant_data, product_data, reactants, products)
        if thermo:
            print(f"ΔH = {thermo['delta_h']:.2f} J/mol")
            print(f"ΔS = {thermo['delta_s']:.2f} J/(mol·K)")
            print(f"ΔG = {thermo['delta_g']:.2f} J/mol")
            print(f"Spontaneous: {thermo['spontaneous']}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    example_usage()