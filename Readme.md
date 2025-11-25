<div align="center">

# 🧬 GEN_RESIST

### Antimicrobial Resistance Prediction Using Graph Attention Networks

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"> <img src="https://img.shields.io/badge/Graph_Neural_Networks-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="GNN"> <img src="https://img.shields.io/badge/Bioinformatics-2E7D32?style=for-the-badge&logo=dna&logoColor=white" alt="Bio">

**🚀 Fast • 🎯 Accurate • 🧬 AI-Powered**

</div>

---

## 🌟 Overview

<table>
<tr>
<td width="50%">

**GEN_RESIST** is an AI-powered application that predicts antimicrobial resistance (AMR) in bacteria from whole-genome sequences. Traditional lab testing takes 48-72 hours, but our Graph Attention Network model delivers predictions in seconds.

</td>
<td width="50%">

### Key Highlights
- 🧬 Analyzes bacterial genome FASTA files
- 🎯 Predicts resistance to 30 antibiotics
- 📊 Provides confidence scores
- 🧪 Detects known resistance genes
- ⚡ Results in seconds

</td>
</tr>
</table>

---

## 🎯 The Problem

> **Antimicrobial resistance kills 1.27 million people annually** and is one of the top global health threats.

Traditional AMR testing:
- ⏰ Takes 2-3 days for results
- 💰 Expensive lab procedures
- 🔬 Requires specialized equipment
- 📉 Delays critical treatment decisions

---

## 💡 Our Solution

GEN_RESIST uses cutting-edge **Graph Attention Networks (GAT)** to predict resistance directly from genomic data:

✅ **Instant Predictions** - Upload genome, get results in seconds  
✅ **High Accuracy** - Trained on validated AMR datasets  
✅ **Multi-Drug Analysis** - 30 antibiotics simultaneously  
✅ **Gene Detection** - Identifies resistance mechanisms  
✅ **User-Friendly** - Simple web interface, no bioinformatics expertise needed  

---

## 🔬 How It Works

<div align="center">

### The Pipeline



</div>

### 1️⃣ K-mer Extraction

The genome sequence is broken into overlapping k-mers (short DNA sequences of length k=11). We extract canonical k-mers using reverse complements to reduce redundancy.

**Example:**


### 2️⃣ Graph Construction

K-mers become nodes in a graph. Edges connect k-mers that appear close together (within a sliding window) in the genome, forming a **co-occurrence graph** that captures local sequence patterns.

**Graph Properties:**
- Nodes: Top 5000 most frequent k-mers
- Edges: Co-occurrence within 10-position window
- Bidirectional: Each edge goes both ways

### 3️⃣ Feature Engineering

We compute three types of features:

**Genome Features:**
- **GC Content** - Percentage of G and C nucleotides (0-1)
- **Normalized Length** - Genome length / 5,000,000 (capped at 1.0)
- **N Fraction** - Proportion of ambiguous nucleotides

**Gene Features:**
- Binary presence vector for 15 known resistance genes from CARD database
- Simple hash-based detection on first 200kb of sequence

**Graph Features:**
- K-mer node embeddings learned during training
- Attention-weighted neighbor aggregation

### 4️⃣ Graph Attention Network

The GAT model processes the graph through multiple layers:

**Architecture:**


**Key Mechanisms:**
- **Multi-head Attention**: Learns different k-mer relationship patterns
- **Graph Pooling**: Aggregates node features to genome-level representation
- **Feature Fusion**: Combines learned graph features with domain knowledge (genes, GC content)
- **Dropout (0.3)**: Prevents overfitting

### 5️⃣ Prediction & Ensemble

**Base Model Output:**
- Raw logits for 30 antibiotics
- Apply sigmoid to get probabilities (0-1)

**Ensemble Boost:**
- If resistance genes detected, boost corresponding antibiotic probabilities
- Example: blaCTX-M-15 detected → Ampicillin, Ceftriaxone probabilities set to max(current, 0.85)

**Final Prediction:**
- Probability > 0.5 → "Resistant"
- Probability ≤ 0.5 → "Susceptible"



---


### Interpretation Guide

| Probability | Confidence | Interpretation |
|------------|------------|----------------|
| 0.9 - 1.0 | Very High | Strong resistance signal |
| 0.7 - 0.9 | High | Likely resistant |
| 0.5 - 0.7 | Moderate | Weak resistance signal |
| 0.3 - 0.5 | Moderate | Weak susceptibility signal |
| 0.1 - 0.3 | High | Likely susceptible |
| 0.0 - 0.1 | Very High | Strong susceptibility signal |

**Detected Genes** indicate known resistance mechanisms found in the genome, which provide additional evidence for the predictions.

---

## 🎯 Supported Antibiotics (30 Total)

<div align="center">

| Class | Antibiotics |
|-------|------------|
| **β-lactams** | Ampicillin, Amoxicillin, Piperacillin, Ceftriaxone, Cefotaxime, Ceftazidime, Cefepime |
| **Carbapenems** | Meropenem, Imipenem, Ertapenem |
| **Fluoroquinolones** | Ciprofloxacin, Levofloxacin, Nalidixic acid |
| **Tetracyclines** | Tetracycline, Doxycycline |
| **Aminoglycosides** | Gentamicin, Streptomycin, Kanamycin, Tobramycin |
| **Macrolides** | Azithromycin, Erythromycin |
| **Glycopeptides** | Vancomycin |
| **Polymyxins** | Colistin |
| **Others** | Chloramphenicol, Rifampicin, Trimethoprim, Sulfamethoxazole |

</div>

---

## 🧬 Resistance Gene Database (15 Genes)

We detect the following high-impact resistance genes from the CARD database:

**β-lactamases (Extended-spectrum & Carbapenemases):**
- blaCTX-M-15 (ESBL - cephalosporin resistance)
- blaTEM-1 (penicillinase)
- blaNDM-1 (carbapenemase - broad spectrum)
- blaKPC (carbapenemase)

**Quinolone Resistance:**
- qnrS1 (plasmid-mediated quinolone resistance)
- qnrB
- gyrA (chromosomal mutations)

**Tetracycline Efflux:**
- tetA (efflux pump)
- tetM (ribosomal protection)

**Colistin Resistance:**
- mcr-1 (mobile colistin resistance)
- mcr-2

**Glycopeptide Resistance:**
- vanA (vancomycin resistance)

**Others:**
- ermB (macrolide resistance)
- aac(6')-Ib (aminoglycoside modification)

---

## 🏗️ System Architecture

<div align="center">

### Component Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 🌐 **Frontend** | HTML/CSS/JavaScript | User interface for file upload and visualization |
| ⚙️ **Backend** | FastAPI + Uvicorn | REST API handling requests and preprocessing |
| 🧠 **ML Model** | PyTorch + PyTorch Geometric | GAT neural network for resistance prediction |
| 📊 **Preprocessing** | Biopython + NumPy | Genome parsing, k-mer extraction, graph building |
| 🐳 **Deployment** | Docker + Render | Containerized cloud hosting |
| 📁 **Storage** | Local JSON/PT files | Model weights and configuration |

</div>



