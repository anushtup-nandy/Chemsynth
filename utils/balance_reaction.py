import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import defaultdict
import pulp
import requests
import base64
from chemicals import CAS_from_any, Tb, Tm, Tc, Hfs, Hfl, Hfg, S0s, S0l, S0g
import logging

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
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return smiles
    except requests.exceptions.HTTPError as errh:
        # Handle HTTPError
        return f"HTTP Error: {errh}"
    except requests.exceptions.RequestException as err:
        # Handle other RequestExceptions
        return f"Request Exception: {err}"
    except (KeyError, IndexError):
        # Handle missing data or incorrect JSON format
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


def solve_ilp(A):
    """
    Solve the integer linear programming problem to find stoichiometric coefficients.

    Parameters:
    A (numpy.ndarray): The stoichiometry matrix.

    Returns:
    list: A list of stoichiometric coefficients, or None if no solution found.
    """
    num_vars = A.shape[1]
    prob = pulp.LpProblem("Balancing_Chemical_Equation", pulp.LpMinimize)
    
    # Define variables with lower bound starting from 1
    x_vars = [pulp.LpVariable(f'x{i}', lowBound=1, cat='Integer') for i in range(num_vars)]
    
    # Objective function
    prob += pulp.lpSum(x_vars)
    
    # Constraints
    for i in range(A.shape[0]):
        prob += pulp.lpDot(A[i, :], x_vars) == 0
    
    # Solve the problem
    solver = pulp.PULP_CBC_CMD(msg=False)  # Disable logging from the solver
    prob.solve(solver)
    
    logger.info(f"Status: {pulp.LpStatus[prob.status]}")
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        solution = [int(pulp.value(var)) for var in x_vars]
        logger.info(f"Solution: {solution}")
        
        # Check if solution is not just zeros
        if all(x == 0 for x in solution):
            return None
        
        return solution
    else:
        return None


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


def setup_matrix(elements, compounds):
    """
    Create a stoichiometry matrix for the elements and compounds.

    Parameters:
    elements (list): A list of elements.
    compounds (list): A list of atom counts for the compounds.

    Returns:
    numpy.ndarray: The stoichiometry matrix.
    """
    matrix = []
    # Fixed variable name from 'counts' to 'compound'
    for compound in compounds:
        row = [compound.get(element, 0) for element in elements]
        matrix.append(row)
    
    # Ensure matrix is 2D and transpose it to have elements as rows
    matrix = np.array(matrix).T  # Transpose to get elements as rows, compounds as columns
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)  # Reshape to 2D if it's inadvertently 1D

    return matrix


def balance_chemical_equation(reactant_smiles, product_smiles):
    """
    Balance a chemical equation given reactants and products as SMILES strings.

    Parameters:
    reactant_smiles (list): A list of SMILES strings for the reactants.
    product_smiles (list): A list of SMILES strings for the products.

    Returns:
    tuple: Two lists containing tuples of stoichiometric coefficients and molecular formulas for reactants and products.
    """
    reactant_counts = [count_atoms(smiles) for smiles in reactant_smiles]
    product_counts = [count_atoms(smiles) for smiles in product_smiles]

    reactant_elements = set(sum([list(counts.keys()) for counts in reactant_counts], []))
    product_elements = set(sum([list(counts.keys()) for counts in product_counts], []))

    if reactant_elements != product_elements:
        missing_in_products = reactant_elements - product_elements
        missing_in_reactants = product_elements - reactant_elements
        error_message = "Element mismatch found: "
        if missing_in_products:
            error_message += f"Elements {missing_in_products} are in reactants but not in products. "
        if missing_in_reactants:
            error_message += f"Elements {missing_in_reactants} are in products but not in reactants."
        raise ValueError(error_message)

    elements = sorted(reactant_elements.union(product_elements))
    A_reactants = setup_matrix(elements, reactant_counts)
    A_products = setup_matrix(elements, product_counts)
    A = np.concatenate([A_reactants, -A_products], axis=1)

    integer_coefficients = solve_ilp(A)
    if integer_coefficients is None or not integer_coefficients:
        raise ValueError("Failed to solve the balance equation. The system may be underdetermined or inconsistent.")

    reactant_coeffs = integer_coefficients[:len(reactant_smiles)]
    product_coeffs = integer_coefficients[len(reactant_smiles):]

    reactant_data = [(coeff, get_molecular_formula(smiles)) for coeff, smiles in zip(reactant_coeffs, reactant_smiles)]
    product_data = [(coeff, get_molecular_formula(smiles)) for coeff, smiles in zip(product_coeffs, product_smiles)]

    return reactant_data, product_data


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


# Example usage function
def example_usage():
    """
    Demonstrate the usage of the balance_reaction module.
    """
    print("=== Chemical Reaction Balancer Example ===\n")
    
    # Example 1: Balance methane combustion
    print("Example 1: Methane Combustion")
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