# Confidence-Aware Plant Disease Diagnosis

An end-to-end deep learning and vector retrieval system for robust, confidence-aware diagnosis of plant leaf diseases using ResNet-50 visual representations and FAISS similarity search.

---

## Overview

Accurate diagnosis of plant diseases is critical for agricultural sustainability and food security. Traditional deep classifiers often produce overconfident false predictions when presented with out-of-distribution or ambiguous crop samples.

This project implements a **Retrieval-Augmented Diagnosis Pipeline**:
1. **Feature Representation**: Extracts 2048-dimensional dense semantic embeddings using a ResNet-50 backbone.
2. **High-Speed Vector Database**: Indexes reference disease embeddings with FAISS (Facebook AI Similarity Search).
3. **Nearest-Neighbor Retrieval**: Queries test leaf images against the reference gallery to retrieve top-$k$ nearest matches.
4. **Confidence-Aware Decision**: Aggregates nearest-neighbor label consensus and distance thresholds to assign diagnosis confidence scores and flag uncertain cases for human inspection.

---

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # Dataset inspection, distribution analysis, and class mapping
│   ├── 02_embedding_generation.ipynb   # ResNet-50 backbone setup and single-sample feature extraction
│   ├── 03_embedding_database.ipynb     # Batch extraction and embedding gallery creation with checkpoints
│   ├── 04_faiss_index.ipynb            # Building and evaluating the FAISS vector index
│   └── 05_retrieval_pipeline.ipynb     # Query inference, similarity search, and confidence-aware diagnosis
├── streamlit_app/                      # Interactive web application for leaf diagnosis
├── paper/                              # Research paper drafts and project documentation
├── models/                             # Pretrained weights and checkpoint storage
├── embeddings/                         # Generated embedding arrays (.npy)
├── faiss_index/                        # Serialized FAISS index files (.index)
├── results/                            # Evaluation plots, confusion matrices, and figures
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- CUDA-enabled GPU (optional, CPU execution supported)

### 2. Environment Setup

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/Sujalpandey239/project-4th.git
cd project-4th

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pipeline Execution

Execute the notebooks sequentially in `notebooks/`:

1. **`01_dataset_exploration.ipynb`**: Verify dataset path and inspect class distributions.
2. **`02_embedding_generation.ipynb`**: Test ResNet-50 feature extraction on sample leaf images.
3. **`03_embedding_database.ipynb`**: Compute 2048-dimensional embeddings for all reference training images and save to `embeddings/`.
4. **`04_faiss_index.ipynb`**: Build the index (`faiss.IndexFlatL2` or inner product) and save to `faiss_index/`.
5. **`05_retrieval_pipeline.ipynb`**: Run similarity search on query images, compute distance distributions, and evaluate diagnosis confidence.

---

## Key Features

- **Dense Deep Representations**: Utilizes pretrained ResNet-50 to capture fine-grained leaf textures, lesions, and coloration patterns.
- **Sub-millisecond Retrieval**: Scalable nearest-neighbor vector search powered by FAISS.
- **Uncertainty & Confidence Estimation**: Provides confidence metrics based on nearest-neighbor distance consistency, avoiding arbitrary overconfident errors on novel or corrupted leaf images.

---

## Author

- **Sujal Pandey**
  - GitHub: [@Sujalpandey239](https://github.com/Sujalpandey239)
