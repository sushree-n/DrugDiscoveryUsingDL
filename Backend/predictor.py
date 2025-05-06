import torch
from vocabulary import make_variable_one
from transformer_model import getInput_mask 


def predict_properties(smiles, models, max_len=44):
    """
    Predict logP, logD, and logS using Transformer models.

    Args:
        smiles (str): Input SMILES string
        models (dict): Dictionary with keys 'logP', 'logD', 'logS', values as (model, vocab)
        max_len (int): Maximum SMILES length for padding

    Returns:
        dict: Predictions for each property rounded to 4 decimals
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = {}

    for task, (model, vocab) in models.items():
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
