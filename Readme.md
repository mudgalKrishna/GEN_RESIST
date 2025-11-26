# 🧬 GEN_RESIST

> Predicting antimicrobial resistance from bacterial genomes using Graph Attention Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)


## What is this?

GEN_RESIST predicts whether bacteria are resistant to antibiotics by analyzing their genome sequences. Instead of waiting 2-3 days for lab culture results, you upload a FASTA file and get predictions in seconds.

The model uses Graph Attention Networks (GAT) to analyze k-mer patterns in bacterial genomes and predict resistance across 30 different antibiotics.

## Why graphs?

Traditional methods either:
- Search for known resistance genes (misses novel variants)
- Count k-mer frequencies (loses sequence context)

We build a **graph where k-mers are nodes** and edges connect k-mers that appear near each other. This preserves local sequence structure and lets the attention mechanism learn which patterns matter for resistance.

## How it works

1. **Parse the genome** - Extract DNA sequence from FASTA
2. **Generate k-mers** - Sliding window of length 11, canonical form (min of forward/reverse complement)
3. **Build graph** - Nodes = k-mers, edges = co-occurrence within 10-position window
4. **Add features** - GC content, genome length, presence of known resistance genes
5. **Run model** - 2-layer GAT processes the graph
6. **Get predictions** - Probabilities for 30 antibiotics, threshold at 0.5


## What you get

**Predictions:**
- Resistant or Susceptible for 30 antibiotics
- Confidence score (probability) for each

**Extra info:**
- Detected resistance genes (from CARD database)
- Genome stats (GC content, length)


## Antibiotics covered

**β-lactams:** Ampicillin, Amoxicillin, Piperacillin, Ceftriaxone, Cefotaxime, Ceftazidime, Cefepime

**Carbapenems:** Meropenem, Imipenem, Ertapenem

**Fluoroquinolones:** Ciprofloxacin, Levofloxacin, Nalidixic acid

**Tetracyclines:** Tetracycline, Doxycycline

**Aminoglycosides:** Gentamicin, Streptomycin, Kanamycin, Tobramycin

**Others:** Azithromycin, Erythromycin, Vancomycin, Colistin, Chloramphenicol, Rifampicin, Trimethoprim, Sulfamethoxazole

## Resistance genes

We check for 15 high-impact genes:
- **β-lactamases:** blaCTX-M-15, blaTEM-1, blaNDM-1, blaKPC
- **Quinolone:** qnrS1, qnrB, gyrA
- **Tetracycline:** tetA, tetM
- **Colistin:** mcr-1, mcr-2
- **Others:** vanA, ermB, aac(6')-Ib


## Tech stack

- **ML:** PyTorch, PyTorch Geometric
- **Backend:** FastAPI
- **Data:** Biopython, NumPy
- **Frontend:** Vanilla HTML/CSS/JS
- **Deploy:** Docker, Render


## Use cases

- **Clinical:** Fast guidance for empirical antibiotic therapy
- **Surveillance:** Track resistance trends in populations
- **Research:** Study genotype-phenotype relationships
- **Education:** Demonstrate ML in genomics







