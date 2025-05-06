import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from torch_geometric.data import Data
from model_architecture import modelGcn  # Importing your GNN class

# --- Helper Functions ---

def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetDegree(),
        int(atom.GetHybridization()),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ], dtype=np.float32)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except:
        raise ValueError(f"Could not Kekulize SMILES: {smiles}")
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[], []]
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges[0] += [start, end]
        edges[1] += [end, start]
    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long)
    )

def compute_global_descriptors(smiles, exclude_mollogp=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
    ]
    if not exclude_mollogp:
        descriptors.append(Descriptors.MolLogP(mol))  # index 6
    descriptors += [
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumAromaticRings(mol)
    ]
    return descriptors

# --- Load Models ---

def load_gnn_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    delaney_model = modelGcn(num_features=7, output_dim=1, global_dim=10).to(device)
    delaney_model.load_state_dict(torch.load('model/Delaney_gcn_model.pth', map_location=device))
    delaney_model.eval()

    logd74_model = modelGcn(num_features=7, output_dim=1, global_dim=10).to(device)
    logd74_model.load_state_dict(torch.load('model/logD74_gcn_model.pth', map_location=device))
    logd74_model.eval()

    sampl_model = modelGcn(num_features=7, output_dim=1, global_dim=9).to(device)
    sampl_model.load_state_dict(torch.load('model/SAMPL_gcn_model.pth', map_location=device))
    sampl_model.eval()

    return {
        "logS": delaney_model,
        "logD": logd74_model,
        "logP": sampl_model
    }

# --- Predict Properties ---

def predict_properties(smiles, models):
    import joblib
    import torch
    import numpy as np
    from torch_geometric.data import Batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] SMILES input: {smiles}")

    # Load scalers
    delaney_scaler = joblib.load("model/Delaney_descriptor_scaler.pkl")
    logd_scaler = joblib.load("model/logD74_descriptor_scaler.pkl")
    sampl_scaler = joblib.load("model/SAMPL_descriptor_scaler.pkl")

    # Convert SMILES to graph
    mol_graph = mol_to_graph(smiles)
    mol_graph = mol_graph.to(device)
    mol_graph.batch = torch.zeros(mol_graph.num_nodes, dtype=torch.long).to(device)

    # Global descriptors
    full_descriptors = compute_global_descriptors(smiles, exclude_mollogp=False)
    descriptors_sampl = compute_global_descriptors(smiles, exclude_mollogp=True)

    print(f"[DEBUG] Full Descriptors (with logP): {full_descriptors}")
    print(f"[DEBUG] Descriptors for SAMPL (no logP): {descriptors_sampl}")

    if np.isnan(full_descriptors).any() or np.isnan(descriptors_sampl).any():
        raise ValueError("Descriptor contains NaNs – invalid SMILES or RDKit error.")

    # Normalize descriptors
    scaled_delaney = delaney_scaler.transform([full_descriptors])[0]
    scaled_logd = logd_scaler.transform([full_descriptors])[0]
    scaled_sampl = sampl_scaler.transform([descriptors_sampl])[0]

    print(f"[DEBUG] Scaled Delaney Descriptors: {scaled_delaney}")
    print(f"[DEBUG] Scaled LogD Descriptors  : {scaled_logd}")
    print(f"[DEBUG] Scaled SAMPL Descriptors : {scaled_sampl}")

    desc_10_delaney = torch.tensor(scaled_delaney, dtype=torch.float32).unsqueeze(0).to(device)
    desc_10_logd = torch.tensor(scaled_logd, dtype=torch.float32).unsqueeze(0).to(device)
    desc_9_sampl = torch.tensor(scaled_sampl, dtype=torch.float32).unsqueeze(0).to(device)

    # Retrieve models and set eval mode
    logS_model = models["logS"]
    logD_model = models["logD"]
    logP_model = models["logP"]

    logS_model.eval()
    logD_model.eval()
    logP_model.eval()

    # Predict
    with torch.no_grad():
        logS_pred = logS_model(mol_graph.x, mol_graph.edge_index, mol_graph.batch, desc_10_delaney)
        logD_pred = logD_model(mol_graph.x, mol_graph.edge_index, mol_graph.batch, desc_10_logd)
        logP_pred = logP_model(mol_graph.x, mol_graph.edge_index, mol_graph.batch, desc_9_sampl)

    print(f"[RESULT] Raw Predictions — logS: {logS_pred.item()}, logD: {logD_pred.item()}, logP: {logP_pred.item()}")

    return {
        "logP": round(logP_pred.item(), 4),
        "logD": round(logD_pred.item(), 4),
        "logS": round(logS_pred.item(), 4)
    }
