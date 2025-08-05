## Chemsynth AI Documentation:

---
#### `scripts.js`
- "Global Parameters" Panel (Top-Left):
    - Code: created a div for global settings. The "Target Amount" and "Default Yield" inputs are placed here.
    - Result: This gives you a clear area to control overarching assumptions for the entire synthesis.
- "Per-Step Yield Overrides (%)" Panel (Top-Right):
    - Code: This is the most significant new feature. The JavaScript now iterates through the route.steps array. For each step, it generates a div containing:
    - A <label> with the step number and title (e.g., Step 1: Terminal Alkane Dehydrogenation).
    An <input type="number"> field specifically for that step's yield.
    - Result: This creates the list you see, allowing you to override the AI-predicted yield for any step you choose. The original predicted yield is shown as a placeholder for reference.