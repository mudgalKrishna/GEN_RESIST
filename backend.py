"""
AMR Prediction - FastAPI Backend
Run with: uvicorn script_name:app --reload
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from Bio import SeqIO
import tempfile
import logging

# ==========================================
# LOGGING & CONFIG
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="AMR Prediction API",
    description="Antimicrobial Resistance Prediction using Graph Attention Networks",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

logger.info(f"🖥️  Device: {device}")

# ==========================================
# MODELS & CONFIG
# ==========================================

class PredictionRequest(BaseModel):
    file_name: str

class PredictionResponse(BaseModel):
    file_name: str
    predictions: Dict[str, str]
    probabilities: Dict[str, float]
    detected_genes: List[str]
    genome_stats: Dict[str, float]

def load_config():
    """Load model configuration"""
    script_dir = Path(__file__).parent
    model_path = script_dir / MODEL_DIR
    
    with open(model_path / "kmer_vocab.json", "r") as f:
        kmer_vocab = json.load(f)
    
    with open(model_path / "antibiotics.json", "r") as f:
        antibiotics = json.load(f)
    
    with open(model_path / "card_genes.json", "r") as f:
        card_genes = json.load(f)["genes"]
    
    return kmer_vocab, antibiotics, card_genes, model_path

kmer_vocab, antibiotics, card_genes, model_path = load_config()
K = kmer_vocab["k"]
kmer2idx = kmer_vocab["kmer2idx"]

logger.info(f"✅ Configuration loaded: k={K}, antibiotics={len(antibiotics)}, genes={len(card_genes)}")

# ==========================================
# MODEL DEFINITION
# ==========================================

class MemoryEfficientGAT(nn.Module):
    def __init__(self, num_kmers, kmer_emb_dim=64, gat_hidden=128, gat_heads=4):
        super().__init__()
        self.kmer_emb = nn.Embedding(num_kmers, kmer_emb_dim)
        self.gat1 = GATConv(kmer_emb_dim, gat_hidden // gat_heads, heads=gat_heads, dropout=0.3)
        self.gat2 = GATConv(gat_hidden, gat_hidden, heads=1, concat=False, dropout=0.3)
        
        self.card_mlp = nn.Sequential(
            nn.Linear(15, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU()
        )
        
        self.genome_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        
        self.final_mlp = nn.Sequential(
            nn.Linear(128 * 2 + 32 + 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 30)
        )
    
    def forward(self, x, edge_index, batch, card, genome_feat):
        x = self.kmer_emb(x).squeeze(1)
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x_graph = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        combined = torch.cat([x_graph, self.card_mlp(card), self.genome_mlp(genome_feat)], dim=1)
        return self.final_mlp(combined)

# Load model
model = MemoryEfficientGAT(num_kmers=len(kmer2idx)).to(device)
model.load_state_dict(torch.load(model_path / "best_model.pt", map_location=device))
model.eval()
logger.info("✅ Model loaded")

# ==========================================
# GENE → ANTIBIOTIC MAPPING
# ==========================================

GENE_TO_ANTIBIOTICS = {
    'blaCTX-M-15': ['Ampicillin', 'Amoxicillin', 'Ceftriaxone', 'Cefotaxime', 'Ceftazidime', 'Cefepime'],
    'blaTEM-1': ['Ampicillin', 'Amoxicillin', 'Piperacillin'],
    'blaNDM-1': ['Meropenem', 'Imipenem', 'Ertapenem'],
    'tetA': ['Tetracycline', 'Doxycycline'],
    'qnrS1': ['Ciprofloxacin', 'Levofloxacin', 'Nalidixic_acid'],
    'mcr-1': ['Colistin'],
}

# ==========================================
# PREPROCESSING
# ==========================================

def extract_kmers_with_rc(seq, k):
    kmers = []
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if set(kmer) <= {"A", "C", "G", "T"}:
            rc = ''.join(complement.get(b, 'N') for b in reversed(kmer))
            kmers.append(min(kmer, rc))
    return kmers

def build_cooccurrence_graph(node_ids, window=10):
    edges = set()
    for i in range(len(node_ids)):
        for j in range(i + 1, min(i + window, len(node_ids))):
            edges.add((i, j))
            edges.add((j, i))
    return np.array(list(edges)).T if edges else np.array([[], []], dtype=np.int64)

def compute_genome_features(seq):
    if not seq:
        return np.zeros(3, dtype=np.float32)
    return np.array([
        (seq.count('G') + seq.count('C')) / len(seq),
        min(len(seq) / 5000000, 1.0),
        seq.count('N') / len(seq) if len(seq) > 0 else 0
    ], dtype=np.float32)

def detect_card_genes_simple(seq, card_genes_list, k=15, min_hits=3):
    """Real CARD gene detection via k-mer exact matching"""
    seq = seq.upper()
    card_vec = np.zeros(len(card_genes_list), dtype=np.float32)
    
    seq_kmers = set(extract_kmers_with_rc(seq, k))
    
    for i, gene in enumerate(card_genes_list):
        gene = gene.upper()
        if len(gene) < k:
            continue
        
        gene_kmers = set(extract_kmers_with_rc(gene, k))
        
        shared = len(seq_kmers.intersection(gene_kmers))
        
        if shared >= min_hits:
            coverage = shared / max(len(gene_kmers), 1)
            card_vec[i] = min(coverage, 1.0)
    
    return card_vec
def preprocess_genome(genome_path):
    seq = "".join(str(rec.seq).upper().replace("N", "") for rec in SeqIO.parse(genome_path, "fasta"))
    if not seq:
        raise ValueError("Empty genome file")
    
    kmers = extract_kmers_with_rc(seq, K)
    node_kmers = [k for k in kmers if k in kmer2idx][:5000]
    if not node_kmers:
        raise ValueError("No valid k-mers found")
    
    node_ids = [kmer2idx[k] for k in node_kmers]
    
    return Data(
        x=torch.tensor(np.array(node_ids).reshape(-1, 1), dtype=torch.long),
        edge_index=torch.tensor(build_cooccurrence_graph(node_ids), dtype=torch.long),
        card=torch.tensor(detect_card_genes_simple(seq, card_genes), dtype=torch.float32).unsqueeze(0),
        genome_feat=torch.tensor(compute_genome_features(seq), dtype=torch.float32).unsqueeze(0)
    ), detect_card_genes_simple(seq, card_genes)

# ==========================================
# PREDICTION
# ==========================================

def predict_amr(genome_path, use_ensemble=True):
    try:
        data, card_vec = preprocess_genome(genome_path)
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")
    
    data = data.to(device)
    
    with torch.no_grad():
        output = model(data.x, data.edge_index, torch.zeros(data.x.shape[0], dtype=torch.long, device=device), data.card, data.genome_feat)
    
    probs = torch.sigmoid(output).cpu().numpy()[0]
    detected_genes = [card_genes[i] for i, v in enumerate(card_vec) if v > 0]
    
    if use_ensemble and detected_genes:
        for gene in detected_genes:
            if gene in GENE_TO_ANTIBIOTICS:
                for ab in GENE_TO_ANTIBIOTICS[gene]:
                    if ab in antibiotics:
                        probs[antibiotics.index(ab)] = max(probs[antibiotics.index(ab)], 0.85)
    
    return {
        "predictions": {ab: "Resistant" if probs[i] > 0.5 else "Susceptible" for i, ab in enumerate(antibiotics)},
        "probabilities": {ab: float(probs[i]) for i, ab in enumerate(antibiotics)},
        "detected_genes": detected_genes,
        "genome_stats": {
            "gc_content": float(data.genome_feat[0, 0]),
            "length_normalized": float(data.genome_feat[0, 1])
        }
    }

# ==========================================
# ROUTES
# ==========================================

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "AMR Prediction API",
        "status": "running",
        "device": str(device),
        "model": "MemoryEfficientGAT",
        "antibiotics": len(antibiotics),
        "genes": len(card_genes)
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(file: UploadFile = File(...)):
    """
    Upload a FASTA genome file and get AMR predictions
    """
    if not file.filename.lower().endswith(('.fasta', '.fna', '.fa')):
        raise HTTPException(status_code=400, detail="File must be FASTA format (.fasta, .fna, .fa)")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        result = predict_amr(tmp_path, use_ensemble=True)
        
        return PredictionResponse(
            file_name=file.filename,
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            detected_genes=result["detected_genes"],
            genome_stats=result["genome_stats"]
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.get("/antibiotics", tags=["Info"])
async def get_antibiotics():
    """Get list of all antibiotics in the model"""
    return {"count": len(antibiotics), "antibiotics": antibiotics}

@app.get("/genes", tags=["Info"])
async def get_genes():
    """Get list of all CARD genes in the model"""
    return {"count": len(card_genes), "genes": card_genes}

@app.get("/config", tags=["Info"])
async def get_config():
    """Get model configuration"""
    return {
        "k_mer_size": K,
        "num_kmers": len(kmer2idx),
        "num_antibiotics": len(antibiotics),
        "num_genes": len(card_genes),
        "device": str(device),
        "model_class": "MemoryEfficientGAT"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
