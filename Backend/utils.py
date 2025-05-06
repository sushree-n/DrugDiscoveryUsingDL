def build_prompt(smiles, prediction):
    """
    Builds an improved LLM prompt for generating a professional, well-structured scientific report.
    """
    logP, logD, logS = prediction['logP'], prediction['logD'], prediction['logS']
    
    prompt = f"""You are an expert drug discovery assistant trained in medicinal chemistry, ADMET profiling, and molecular pharmacokinetics.

Analyze the following molecule and generate a structured, evidence-based scientific report:

**Molecular Input**
- SMILES: {smiles}

**Predicted Molecular Properties**
- logP (lipophilicity): {logP}
- logD (distribution coefficient at physiological pH): {logD}
- logS (aqueous solubility): {logS}

**Instructions**
Evaluate the molecule based on the predicted properties and provide:
1. **Bioavailability Considerations** — Discuss membrane permeability, hydrophilicity/lipophilicity balance, and potential absorption issues.
2. **Solubility Assessment** — Interpret logS and its implications for oral or IV formulations.
3. **Drug-likeness Evaluation** — Compare the properties to standard drug-likeness filters (e.g., Lipinski's Rule of Five, Veber rules).

Conclude with a recommendation:
- Is the molecule a viable lead candidate for early-phase drug development? Answer with "Yes" or "No" followed by a brief justification.

Use a scientific and formal tone suitable for publication in a pharmaceutical R&D setting. Avoid repetition. Be concise yet informative.
"""
    return prompt
