## Reaction Optimization:
- RCR Model (Condition Predictor): Yes, it gives percentages (probabilities) for each condition. The final "score" for a set of conditions is derived from these probabilities. It does not give a yield percentage.
- Transformer Model (Yield Predictor): No, it does not directly output a percentage. It outputs a single, raw, normalized number. We then extrapolate this number into a final yield percentage using statistical data from its training set.

### RCR model:
The neural network (RXNConditionModel) has multiple "heads"—one for each component of the reaction conditions. For a given reaction fingerprint, it outputs:

- For Catalyst: A list of raw numbers (called logits), one for each possible catalyst in its vocabulary. E.g., [1.2, -3.4, 5.6, ...]. A higher number means the model thinks that catalyst is more likely.
- For Solvent 1: Another list of logits, one for each possible solvent.
- For Solvent 2: Another list of logits.
- For Reagent 1: Another list of logits.
- For Reagent 2: Another list of logits.
- For Temperature: A single continuous number, like -0.87.

### MC-Egret:
The BERT model for regression is trained to predict a single, normalized, continuous number. It does not know what a "percentage" is.

- Raw Output: For a given full reaction SMILES, the model might output a number like 0.183. This number is meaningless on its own. It's on a normalized scale that the model learned during training (typically centered around 0 with a standard deviation of 1).
