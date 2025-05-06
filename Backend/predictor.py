import torch
from rdkit import Chem
from rdkit.Chem import Crippen
from vocabulary import make_variable_one
from transformer_model import getInput_mask

def predict_properties(smiles, models, max_len=44):
    """
    Predict logP (RDKit), logD, and logS using RDKit and Transformer models.

    Args:
        smiles (str): Input SMILES string
        models (dict): Dictionary with keys 'logD', 'logS' (Transformer models), and optionally 'logP'
        max_len (int): Maximum SMILES length for padding

    Returns:
        dict: Predictions for each property rounded to 4 decimals
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = {}

    # logP from RDKit
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        logp_val = Crippen.MolLogP(mol)
        predictions['logP'] = round(logp_val, 4)
    except Exception as e:
        predictions['logP'] = f"Error: {str(e)}"

    # logD and logS from Transformer models
    for task in ['logD', 'logS']:
        if task in models:
            model, vocab = models[task]
            try:
                encoded = make_variable_one(smiles, vocab, max_len=max_len)
                x = torch.tensor([encoded], dtype=torch.long).to(device)
                mask = getInput_mask(x).to(device)

                with torch.no_grad():
                    output = model(x, mask).squeeze().item()
                predictions[task] = round(output, 4)

            except Exception as e:
                predictions[task] = f"Error: {str(e)}"

    return predictions
