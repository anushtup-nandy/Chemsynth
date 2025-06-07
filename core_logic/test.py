# test_aizynth_cli.py
import os
import logging
from aizynthfinder.aizynthfinder import AiZynthFinder
from rdkit import Chem

# --- Basic Configuration ---

# Configure logging to see AiZynthFinder's progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AiZynthTest")

# --- SCRIPT PARAMETERS: Change these as needed ---

# Target molecule in SMILES format. (Oseltamivir is a good, non-trivial test case)
TARGET_SMILES = "CCC(CC)OC1=CC(=C(C=C1)N)C(=O)N"

# Assumes this script is in the project root and the config is in 'data/config.yml'
# This creates a robust, absolute path to the config file.
CONFIG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yml'))

def run_standalone_aizynth_test():
    """
    Initializes AiZynthFinder, runs a search for the target molecule,
    and prints the results to the terminal.
    """
    logger.info(f"--- Starting AiZynthFinder Standalone Test ---")
    logger.info(f"Target SMILES: {TARGET_SMILES}")
    
    # 1. Verify that the config file exists
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.error(f"FATAL: Configuration file not found at: {CONFIG_FILE_PATH}")
        logger.error("Please ensure the script is run from the project root and the path is correct.")
        return
    
    logger.info(f"Using configuration file: {CONFIG_FILE_PATH}")

    # 2. Initialize AiZynthFinder
    try:
        finder = AiZynthFinder(configfile=CONFIG_FILE_PATH)
        logger.info("AiZynthFinder initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize AiZynthFinder: {e}", exc_info=True)
        return

    # 3. Set target and select policies/stock
    try:
        finder.target_smiles = TARGET_SMILES
        
        # CRITICAL: These names must EXACTLY match the keys in your 'data/config.yml'
        # Example names are used here. Replace with your actual names.
        finder.stock.select("zinc")
        finder.expansion_policy.select(["uspto", "ringbreaker"])
        finder.filter_policy.select("uspto")
        
        logger.info(f"Successfully selected stock: {finder.stock.selection}")
        logger.info(f"Successfully selected expansion policies: {finder.expansion_policy.selection}")
        logger.info(f"Successfully selected filter policy: {finder.filter_policy.selection}")
        
    except Exception as e:
        logger.error(f"Failed to select policies or stock. This is often due to a name mismatch with your config.yml file.", exc_info=True)
        return

    # 4. Run the search
    try:
        logger.info("Preparing search tree...")
        finder.prepare_tree()
        
        logger.info("Setting time limit and starting search (max 120s)...")
        # finder.analysis.max_time = 120  # seconds
        finder.tree_search()
        
        logger.info("Search complete. Building routes...")
        finder.build_routes()
        logger.info("Route building complete.")
        
    except Exception as e:
        logger.error(f"An error occurred during the tree search: {e}", exc_info=True)
        return

    # 5. Display the results
    logger.info(f"\n--- Analysis Finished ---")
    logger.info(f"Found {len(finder.routes)} routes.")

    if not finder.routes:
        logger.warning("No routes were found within the time limit.")
        return

    # Sort routes by score (best first)
    sorted_routes = sorted(finder.routes, key=lambda r: r.score, reverse=True)
    
    for i, route in enumerate(sorted_routes[:5]):  # Print top 5 routes
        print("\n" + "="*80)
        print(f"ROUTE {i+1} | Score: {route.score:.4f} | Number of Steps: {len(route.reaction_tree.reactions())}")
        print("="*80)
        
        # Reactions are returned in retrosynthetic order, so we reverse for forward synthesis
        forward_steps = reversed(list(route.reaction_tree.reactions()))

        for step_num, reaction in enumerate(forward_steps):
            # Extract SMILES for reactants and product
            product_smi = Chem.MolToSmiles(reaction.mol)
            # Reactants are in a nested list, flatten it
            reactants_smi = " . ".join([Chem.MolToSmiles(mol) for group in reaction.reactants for mol in group])
            
            # Get template score as a proxy for yield/confidence
            template_score = reaction.metadata.get("template_score", "N/A")
            
            print(f"  Step {step_num + 1}:")
            print(f"    Reaction: {reactants_smi} >> {product_smi}")
            print(f"    Confidence (Template Score): {template_score:.3f}")
            
    print("\n" + "="*80)
    logger.info("--- Standalone Test Finished ---")


if __name__ == "__main__":
    run_standalone_aizynth_test()