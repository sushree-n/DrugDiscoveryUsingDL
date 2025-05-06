import torch
import joblib
import pickle
from transformer_model import modelTransformer_smi
from vocabulary import load_vocab, make_variable_one
from transformer_model import getInput_mask


def load_transformer_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {}
    for task in ['logP', 'logD', 'logS']:
        vocab = load_vocab(f"model/vocab_{task}.pkl")
        args = {
            'dropout': 0.1,
            'num_layer': 2,
            'num_heads': 1 if task == 'logP' else 2,
            'hidden_dim': 256,
            'output_dim': 128,
            'n_output': 1,
            'vocab': vocab
        }
        model = modelTransformer_smi(args).to(device)
        model.load_state_dict(torch.load(f"model/transformer_model_{task}.pt", map_location=device))
        model.eval()
        models[task] = (model, vocab)
    return models
