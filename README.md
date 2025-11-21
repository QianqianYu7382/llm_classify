# News Headline Classification  
### Method A (N-gram LM) vs Method B (SentenceTransformer)

This repository contains the code used to run two news classification models:
- **Method A:** Class-conditional N-gram Language Model  
- **Method B:** SentenceTransformer embeddings + Logistic Regression  

All experiments in the report can be reproduced by running the scripts in this repo.

---

## 1. Requirements

### **Python version**
Python 3.9+

### **Install dependencies**
```bash
pip install numpy pandas scikit-learn sentence-transformers tqdm
