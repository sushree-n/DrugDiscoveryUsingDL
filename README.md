# Drug Discovery Using Deep Learning

A Multimodal Fused Deep Learning Framework for Predicting Molecular Properties  
**Sushree Nadiminty, Tejas D Jadhav, Deekshitha Prabhakar**  
**Supervised by Dr. Alina Vereshchaka | University at Buffalo**

---

## 🌟 Overview

This project presents a **multimodal deep learning approach** for early-stage drug discovery by predicting key molecular properties—**logP**, **logD**, and **logS**—which are crucial indicators of drug-likeness, permeability, solubility, and bioavailability.  
We fuse diverse molecular perspectives including **2D topology**, **chemical language**, **substructure patterns**, and **3D spatial descriptors** to build a highly accurate and interpretable prediction system.

---

## 🧠 Key Features

- 🧬 **Graph Isomorphism Networks (GIN)** for topological learning from 2D molecular graphs  
- 🧠 **BiGRU with Attention** for substructure-level fingerprint embeddings (ECFP)  
- 🔤 **Transformer Encoder** for SMILES sequence modeling  
- 🔺 **MLP** for 3D molecular descriptors (E3FP)  
- 🔗 **Late Fusion Models** (Tri-LASSO, Tetra-SGD) to combine multimodal embeddings  
- 📈 Performance benchmarks on **Delaney**, **logD74**, and **SAMPL** datasets  
- 🧾 Integrated with **LLM (DeepSeek V3)** for scientific report generation  
- 🌐 Deployed using **Flask (backend)** and **React + Tailwind (frontend)**

---

## 🔬 Methodology

Each molecular representation is processed using a separate neural network:

| Modality | Model | Purpose |
|----------|-------|---------|
| SMILES | Transformer | Captures sequence-level chemical information |
| ECFP | BiGRU + Attention | Learns local substructure patterns |
| Graph | GINConv | Encodes atomic and bond topology |
| E3FP | MLP | Learns spatial molecular features |

All learned features are **fused and passed to a downstream regressor** to predict logP, logD, and logS.

---

## 📊 Results

Fusion models consistently outperform single-modality baselines across all datasets:

| Model | Avg R² | Avg MAE | Avg Pearson |
|-------|--------|----------|--------------|
| Transformer | 0.575 | 0.529 | 0.724 |
| BiGRU | 0.610 | 0.502 | 0.846 |
| GIN | 0.498 | 0.463 | 0.900 |
| **Fusion** | **0.668** | **0.403** | **0.908** |

---

## 🤖 LLM Integration

We use **DeepSeek V3** to generate **readable, interpretable scientific reports** from model predictions:

- **Prompt Engineering** guides the LLM using structured prompts with scientific format
- Outputs explain predicted values, biological relevance, and next steps for drug screening
- Example: Reports summarizing DTIs, ADMET relevance, and drug-likeness metrics

---

## 🧪 Try It Yourself!

### 🔗 Live Demo

```
https://drugdiscoveryusingdl-1.onrender.com
```

### 🧫 Sample SMILES

You can paste the following valid SMILES into the web app:

```
CC(=O)OC1=CC=CC=C1C(=O)O        (Aspirin)
CCCCCCCCCCCCCCCC(=O)O           (Stearic Acid)
COC1=CC=CC=C1OC                 (Anisole)
CC(C)NCC(COC1=CC=CC=C1O)O       (Propranolol)
CC(=O)NC1=CC=C(C=C1)O           (Paracetamol)
```

---

## 🛠️ Tech Stack

- **Backend**: Flask, PyTorch, PyTorch Geometric, RDKit, NumPy, DeepSeek API  
- **Frontend**: React, TypeScript, TailwindCSS  
- **Deployment**: Render, GitHub  
- **Models**: Custom Transformer, BiGRU, GINConv, MLP  
- **Fusion**: Tri-LASSO, Tetra-SGD

---

## 🧑‍🔬 Authors

- **Sushree Nadiminty** 
- **Tejas D Jadhav**
- **Deekshitha Prabhakar** 
- **Mentor** – Dr. Alina Vereshchaka, University at Buffalo

---

## 📚 References

1. Multimodal fused deep learning for drug property prediction  
2. Parsing clinical success rates  
3. Toward Unified AI Drug Discovery with Multimodal Knowledge  
4. Graph Convolutional Policy Network for Goal-Directed Molecule Generation


