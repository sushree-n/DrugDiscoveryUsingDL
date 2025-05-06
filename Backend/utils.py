def build_prompt(smiles, prediction):
    """
    Builds a more detailed and realistic LLM prompt for generating a contextual scientific report.
    """
    logP, logD, logS = prediction['logP'], prediction['logD'], prediction['logS']
    
    prompt = f"""You are a senior scientist in drug discovery and pharmacokinetics with expertise in both early-phase candidate profiling and evaluation of legacy molecules.

You are asked to analyze the following molecule for its suitability as a drug candidate, using both computational property predictions and contextual medicinal chemistry knowledge.

**Molecular Input**
- SMILES: {smiles}

**Predicted Properties**
- logP (partition coefficient): {logP}
- logD (distribution coefficient at physiological pH 7.4): {logD}
- logS (aqueous solubility in log mol/L): {logS}

**Instructions**
1. **Bioavailability & Permeability**
   - Interpret the predicted logP and logD values.
   - Assess membrane permeability potential and pKa/ionization behavior at physiological pH.
   - Discuss potential challenges for oral absorption.

2. **Solubility & Formulation**
   - Evaluate logS in the context of acceptable solubility ranges for oral drugs.
   - Suggest formulation strategies (e.g., salt forms, co-crystals) if solubility is limited.

3. **Drug-Likeness & Rule-Based Profiling**
   - Check compliance with Lipinski’s Rule of Five and Veber’s rule.
   - Discuss the relevance and limitations of these rules, especially in the context of known, approved drugs.

4. **Contextual Evaluation**
   - If the molecule resembles or is known to be a clinically approved drug (e.g., aspirin), consider historical usage, known formulation approaches, and clinical viability.
   - Clearly distinguish between theoretical limitations and real-world usage.

**Final Recommendation**
State whether the molecule is a viable lead candidate. Answer with "Yes" or "No" followed by a nuanced justification. If the molecule is known to be approved, explain why it still succeeded despite potential computational red flags.

Use a scientific and formal tone suitable for pharmaceutical R&D and include all relevant caveats or assumptions. Avoid repetition.
"""
    return prompt
