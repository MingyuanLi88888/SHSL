# SHSL
This is the code for the Paper:Self-supervised hypergraph structure learning
## Abstract  

Traditional Hypergraph Neural Networks (HGNNs) often assume that hypergraph structures are perfectly constructed, yet real-world hypergraphs are typically corrupted by noise, missing data, or irrelevant information, limiting the effectiveness of hypergraph learning. To address this challenge, we propose **SHSL**, a novel **Self-supervised Hypergraph Structure Learning** framework that jointly explores and optimizes hypergraph structures without external labels. SHSL consists of two key components: a **self-organizing initialization module** that constructs latent hypergraph representations, and a **differentiable optimization module** that refines hypergraphs through gradient-based learning. These modules collaboratively capture high-order dependencies to enhance hypergraph representations. Furthermore, SHSL introduces a **dual learning mechanism** to simultaneously guide structure exploration and optimization within a unified framework.  

Experiments on six public datasets demonstrate that SHSL outperforms state-of-the-art baselines, achieving **Accuracy improvements of 1.36%-32.37% and 2.23%-27.54%** on hypergraph exploration and optimization tasks, and **1.19%-8.4%** on non-hypergraph datasets. **Robustness evaluations** further validate SHSL’s effectiveness under noisy and incomplete scenarios, highlighting its practical applicability.  

# Framework diagram of SHSL
![Fig2](https://github.com/user-attachments/assets/23f97ab0-dff9-439c-a841-459b1b4927fe)
# Contrastive views for different tasks
![Fig2](https://github.com/user-attachments/assets/81758415-8a77-4e24-b0dd-224d2fda0d89)
# Example  

```bash
python main.py
```
# Contact
GitHub Issues: Please open an issue for bugs and questions.

# Citation  

If you find this work helpful, please consider citing our paper:  
