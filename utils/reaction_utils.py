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
            print("Initializing RXNMapper... (this may take a moment on first run)")
            _rxn_mapper = RXNMapper()
            print("RXNMapper initialized.")
        except Exception as e:
            print(f"Failed to initialize RXNMapper: {e}. Reaction images may not have atom mapping.")
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

def generate_reaction_image(reaction_smiles: str, step_id: str) -> str | None:
    """
    Generates an SVG image of the reaction, saves it, and returns its web path.
    This function now includes the mapping step.

    Args:
        reaction_smiles: The UNMAPPED reaction SMILES.
        step_id: A unique identifier for the step (e.g., "route_a_step_1").

    Returns:
        The web-accessible path to the generated image, or None on failure.
    """
    try:
        # First, get the atom-mapped reaction SMILES
        mapped_reaction_smiles = map_reaction(reaction_smiles)

        rxn = Chem.rdChemReactions.ReactionFromSmarts(mapped_reaction_smiles, useSmiles=True)
        if not rxn:
            # Fallback for complex reactions RDKit can't draw from SMARTS
            print(f"Warning: Could not create reaction object for {step_id}. Skipping image.")
            return None

        # Drawing options for a clean SVG
        d2d = rdMolDraw2D.MolDraw2DSVG(500, 180) # Increased size slightly
        dopts = d2d.drawOptions()
        # Use a color-blind friendly palette for atom highlighting
        dopts.setHighlightColour((0.2, 0.6, 0.8, 0.8)) # A nice blue
        dopts.atomHighlightsAreCircles = True
        dopts.fillHighlights = True
        
        # Highlight the atoms based on the mapping
        highlight_atoms = []
        for i in range(rxn.GetNumReactantTemplates()):
            mol = rxn.GetReactantTemplate(i)
            highlight_atoms.append([at.GetIdx() for at in mol.GetAtoms() if at.GetAtomMapNum() > 0])
        
        d2d.DrawReaction(rxn, highlightByReactant=True, highlightColorsReactants=[(0.2, 0.6, 0.8)])
        
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()

        filename = f"rxn_{step_id}.svg"
        filepath = REACTION_IMG_DIR / filename
        
        with open(filepath, "w") as f:
            f.write(svg)
        
        # Return the path that the frontend can use
        return f"/static/reaction_images/{filename}"

    except Exception as e:
        print(f"Error generating reaction image for step {step_id}: {e}")
        return None