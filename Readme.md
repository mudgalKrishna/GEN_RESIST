🧬 GEN_RESIST
Antimicrobial Resistance Prediction Using Graph Attention Networks

[](LICENSEn](https://img.shields.io/badge/python-3.8+-.100ast, model-driven web application for predicting antimicrobial resistance from bacterial whole-genome sequences

📋 Table of Contents
Overview

Features

Architecture

Getting Started

API Documentation

Deployment

Technology Stack

Contributing

License

🔬 Overview
GEN_RESIST addresses the critical global health challenge of antimicrobial resistance (AMR) by leveraging state-of-the-art Graph Attention Networks to predict bacterial resistance patterns from whole-genome sequencing data.

The Challenge: Traditional AMR testing can take days, delaying critical treatment decisions.

Our Solution: Upload a bacterial genome FASTA file and receive instant predictions across 30 antibiotics with confidence scores and detected resistance genes.

✨ Features
🔹 Fast Predictions – Upload genome files and get results in seconds
🔹 Comprehensive Coverage – Predictions for 30 different antibiotics
🔹 Confidence Scoring – Probability scores for each prediction
🔹 Gene Detection – Identifies known resistance genes from CARD database
🔹 Interactive Dashboard – Clean, intuitive web interface
🔹 REST API – Easy integration with existing workflows
🔹 Production Ready – Fully containerized and cloud-deployable

🏗️ Architecture
GEN_RESIST follows a modern microservices architecture with clear separation of concerns:

Frontend

Technology: HTML5, CSS3, JavaScript

Hosting: GitHub Pages

Function: User interface for file upload and result visualization

Backend

Technology: FastAPI + Uvicorn

Hosting: Render (Docker container)

Function: REST API serving prediction endpoints

ML Model

Architecture: Graph Attention Network (GAT)

Input: K-mer co-occurrence graphs with genome features

Output: Resistance probabilities for 30 antibiotics

Workflow

text
User uploads FASTA → Frontend sends to API → Backend preprocesses genome
→ Extracts k-mers → Builds graph → GAT model inference → Returns predictions
🚀 Getting Started
Prerequisites

Python 3.8 or higher

pip package manager

Git

Local Installation

Clone the repository:

bash
git clone https://github.com/yourusername/GEN_RESIST.git
cd GEN_RESIST
Create and activate virtual environment:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Running Locally

Start the FastAPI backend:

bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
The API will be available at http://localhost:8000

Open the frontend by serving the HTML files or opening index.html in your browser.

📡 API Documentation
Base URL

Local: http://localhost:8000

Production: https://your-service.onrender.com

Endpoints

GET /

Health check and service information

Response:

json
{
  "service": "AMR Prediction API",
  "status": "running",
  "device": "cuda",
  "model": "MemoryEfficientGAT",
  "antibiotics": 30,
  "genes": 15
}
POST /predict

Upload genome and receive AMR predictions

Request:

Method: POST

Content-Type: multipart/form-data

Body: file (FASTA format)

Response:

json
{
  "file_name": "genome.fasta",
  "predictions": {
    "Ampicillin": "Resistant",
    "Ciprofloxacin": "Susceptible"
  },
  "probabilities": {
    "Ampicillin": 0.91,
    "Ciprofloxacin": 0.13
  },
  "detected_genes": ["blaCTX-M-15", "qnrS1"],
  "genome_stats": {
    "gc_content": 0.51,
    "length_normalized": 0.87
  }
}
GET /antibiotics

List all antibiotics covered by the model

Response:

json
{
  "count": 30,
  "antibiotics": ["Ampicillin", "Ciprofloxacin", "..."]
}
GET /genes

List all resistance genes in database

Response:

json
{
  "count": 15,
  "genes": ["blaCTX-M-15", "blaTEM-1", "..."]
}
GET /config

Model configuration details

Response:

json
{
  "k_mer_size": 11,
  "num_kmers": 50000,
  "num_antibiotics": 30,
  "num_genes": 15,
  "device": "cuda",
  "model_class": "MemoryEfficientGAT"
}
🐳 Deployment
Docker

Build the image:

bash
docker build -t gen-resist-api .
Run the container:

bash
docker run -p 8000:8000 gen-resist-api
Render

Push your repository to GitHub

Create a new Web Service on Render

Connect your GitHub repository

Render automatically detects render.yaml and builds from Dockerfile

Your API will be deployed at https://your-service.onrender.com

Frontend (GitHub Pages)

Push frontend files to a GitHub repository

Enable GitHub Pages in repository settings

Update the API base URL in your JavaScript to point to your Render backend

🛠️ Technology Stack
Machine Learning

PyTorch

PyTorch Geometric

Graph Attention Networks

Backend

FastAPI

Uvicorn

Pydantic

Data Processing

Biopython

NumPy

Python

Frontend

HTML5

CSS3

JavaScript (Vanilla)

DevOps

Docker

Render

GitHub Pages

GitHub Actions (optional CI/CD)

