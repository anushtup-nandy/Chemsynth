# utils/reaction_utils.py
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rxnmapper import RXNMapper

# --- Constants ---
# Create a directory to store generated images
STATIC_DIR = Path(__file__).parent.parent / "static"
REACTION_IMG_DIR = STATIC_DIR / "reaction_images"
REACTION_IMG_DIR.mkdir(exist_ok=True)

# --- Caching ---
_rxn_mapper = None

def get_rxn_mapper():
    """Initializes the RXNMapper model, with in-memory caching."""
    global _rxn_mapper
    if _rxn_mapper is None:
        try:
            print("Initializing RXNMapper... (this may take a moment)")
            _rxn_mapper = RXNMapper()
            print("RXNMapper initialized.")
        except Exception as e:
            print(f"Failed to initialize RXNMapper: {e}")
    return _rxn_mapper

def map_reaction(reaction_smiles: str) -> str:
    """
    Generates atom mapping for a reaction SMILES.

    Args:
        reaction_smiles: A reaction SMILES string (e.g., "A.B>>C").

    Returns:
        The atom-mapped reaction SMILES, or the original SMILES on failure.
    """
    mapper = get_rxn_mapper()
    if not mapper:
        return reaction_smiles
    try:
        results = mapper.get_attention_guided_atom_maps([reaction_smiles])
        mapped_smiles = results[0]['mapped_rxn']
        return mapped_smiles
    except Exception as e:
        print(f"Error during reaction mapping for '{reaction_smiles}': {e}")
        return reaction_smiles

def generate_reaction_image(mapped_reaction_smiles: str, step_id: str) -> str | None:
    """
    Generates a PNG image of the reaction and saves it.

    Args:
        mapped_reaction_smiles: The atom-mapped reaction SMILES.
        step_id: A unique identifier for the step (e.g., "route_a_step_1").

    Returns:
        The web-accessible path to the generated image, or None on failure.
    """
    try:
        rxn = Chem.rdChemReactions.ReactionFromSmarts(mapped_reaction_smiles, useSmiles=True)
        if not rxn:
            return None

        # Drawing options
        d2d = rdMolDraw2D.MolDraw2DSVG(450, 150)
        dopts = d2d.drawOptions()
        dopts.useBWAtomPalette()
        dopts.atomHighlightsAreCircles = True
        
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()

        filename = f"rxn_{step_id}.svg"
        filepath = REACTION_IMG_DIR / filename
        
        with open(filepath, "w") as f:
            f.write(svg)
        
        # Return the path that the frontend can use
        return f"/static/reaction_images/{filename}"

    except Exception as e:
        print(f"Error generating reaction image: {e}")
        return None