import re
from rdkit import Chem

def tokenizer(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return []
    smile = Chem.MolToSmiles(mol, canonical=True)
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    return re.findall(pattern, smile)

def encode_smiles(smile, vocab, max_len=44):
    tokens = tokenizer(smile)
    vec = [vocab.get(tok, vocab['Unknown']) for tok in tokens]
    vec.append(vocab.get('<end>', 0))
    vec += [0] * (max_len - len(vec))
    return vec[:max_len]
