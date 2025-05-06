import re
import pickle
from rdkit import Chem

def tokenizer(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return []
    smile = Chem.MolToSmiles(mol, canonical=True)
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = regezz.findall(smile)
    return tokens

def load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def make_variable_one(smiles, vocab, max_len=44):
    tokens = tokenizer(smiles)
    vec = [vocab.get(tok, vocab['Unknown']) for tok in tokens]
    vec.append(vocab.get('<end>', 0))
    if len(vec) < max_len:
        vec += [0] * (max_len - len(vec))
    return vec[:max_len]
