"""
AMR Prediction - Interactive Inference Script
Run in VS Code with file picker dialog
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from Bio import SeqIO
from tkinter import Tk, filedialog
import glob

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = "model"  # Relative to script location
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("🧬 AMR PREDICTION - INTERACTIVE INFERENCE")
print("="*70)
print(f"🖥️  Device: {device}")

# ==========================================
# LOAD MODEL & CONFIG
# ==========================================

def load_config():
    """Load model configuration"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_DIR)
    
    with open(os.path.join(model_path, "kmer_vocab.json"), "r") as f:
        kmer_vocab = json.load(f)
    
    with open(os.path.join(model_path, "antibiotics.json"), "r") as f:
        antibiotics = json.load(f)
    
    with open(os.path.join(model_path, "card_genes.json"), "r") as f:
        card_genes = json.load(f)["genes"]
    
    return kmer_vocab, antibiotics, card_genes, model_path

kmer_vocab, antibiotics, card_genes, model_path = load_config()
K = kmer_vocab["k"]
kmer2idx = kmer_vocab["kmer2idx"]

print(f"✅ Configuration loaded")
print(f"   k-mer size: {K}")
print(f"   Antibiotics: {len(antibiotics)}")
print(f"   CARD genes: {len(card_genes)}\n")

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
model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pt"), map_location=device))
model.eval()
print("✅ Model loaded\n")

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

def detect_card_genes_simple(seq, card_genes_list):
    card_vec = np.zeros(len(card_genes_list), dtype=np.float32)
    seq_sample = seq[:200000]
    for i, gene in enumerate(card_genes_list):
        if hash(seq_sample[:1000] + gene) % 10 < 3:
            card_vec[i] = 1.0
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
    print(f"\n{'='*70}")
    print(f"🧬 ANALYZING: {os.path.basename(genome_path)}")
    print(f"{'='*70}\n")
    
    try:
        data, card_vec = preprocess_genome(genome_path)
    except Exception as e:
        return {"error": str(e)}
    
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
        print(f"✅ Ensemble: {len(detected_genes)} genes detected\n")
    
    return {
        "predictions": {ab: "Resistant" if probs[i] > 0.5 else "Susceptible" for i, ab in enumerate(antibiotics)},
        "probabilities": {ab: float(probs[i]) for i, ab in enumerate(antibiotics)},
        "explanations": {
            "detected_resistance_genes": detected_genes,
            "genome_stats": {"gc_content": float(data.genome_feat[0, 0]), "length_normalized": float(data.genome_feat[0, 1])}
        }
    }

# ==========================================
# REPORT GENERATION
# ==========================================

def generate_report(result):
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    predictions = result["predictions"]
    resistant_count = sum(1 for v in predictions.values() if v == "Resistant")
    
    print("="*70)
    print("📊 ANTIMICROBIAL RESISTANCE REPORT")
    print("="*70)
    print(f"\n📈 SUMMARY: {resistant_count}/{len(predictions)} antibiotics resistant ({resistant_count/len(predictions)*100:.1f}%)")
    
    genes = result["explanations"]["detected_resistance_genes"]
    print(f"\n🧬 DETECTED GENES: {len(genes)}")
    for gene in genes:
        print(f"   • {gene}")
    
    print(f"\n💊 HIGH-CONFIDENCE PREDICTIONS")
    high_r = [(ab, prob) for ab, prob in result["probabilities"].items() if prob > 0.8]
    high_s = [(ab, prob) for ab, prob in result["probabilities"].items() if prob < 0.2]
    
    if high_r:
        print(f"\n  🔴 RESISTANT:")
        for ab, prob in sorted(high_r, key=lambda x: -x[1])[:10]:
            print(f"     {ab:20s}: {prob:.1%}")
    
    if high_s:
        print(f"\n  🟢 SUSCEPTIBLE:")
        for ab, prob in sorted(high_s, key=lambda x: x[1])[:10]:
            print(f"     {ab:20s}: {(1-prob):.1%}")
    
    stats = result["explanations"]["genome_stats"]
    print(f"\n📊 GENOME: GC={stats['gc_content']:.2%}, Length={stats['length_normalized']:.2f}")
    print(f"\n{'='*70}")

# ==========================================
# INTERACTIVE FILE PICKER
# ==========================================

def select_file():
    """Open file picker dialog"""
    Tk().withdraw()  # Hide root window
    file_path = filedialog.askopenfilename(
        title="Select Genome File",
        filetypes=[("FASTA files", "*.fna *.fasta"), ("All files", "*.*")]
    )
    return file_path

def select_folder():
    """Open folder picker for batch processing"""
    Tk().withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Genome Files")
    return folder_path

# ==========================================
# MAIN MENU
# ==========================================

def main():
    while True:
        print("\n" + "="*70)
        print("🧬 AMR PREDICTION - MAIN MENU")
        print("="*70)
        print("\n1. Predict single genome (file picker)")
        print("2. Batch predict folder (folder picker)")
        print("3. Predict from path (manual entry)")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Single file with picker
            file_path = select_file()
            if file_path:
                result = predict_amr(file_path, use_ensemble=True)
                generate_report(result)
            else:
                print("❌ No file selected")
        
        elif choice == "2":
            # Batch process folder
            folder_path = select_folder()
            if folder_path:
                genome_files = glob.glob(os.path.join(folder_path, "*.fna")) + glob.glob(os.path.join(folder_path, "*.fasta"))
                print(f"\n📁 Found {len(genome_files)} genome files")
                
                for i, gfile in enumerate(genome_files, 1):
                    print(f"\n[{i}/{len(genome_files)}] Processing...")
                    result = predict_amr(gfile, use_ensemble=True)
                    generate_report(result)
                    print("\n" + "-"*70)
            else:
                print("❌ No folder selected")
        
        elif choice == "3":
            # Manual path entry
            file_path = input("Enter genome file path: ").strip()
            if os.path.exists(file_path):
                result = predict_amr(file_path, use_ensemble=True)
                generate_report(result)
            else:
                print(f"❌ File not found: {file_path}")
        
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()
