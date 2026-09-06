# Behavioral Signatures of Cognition: Toward Explainable Keystroke Intelligence (TypeXAI)

**Authors**: **Gokul Ram K***, Rohit N, and Premanand V  
**Affiliation**: School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India  
**Contact**: gokul.ram.kannan210905@gmail.com | [LinkedIn Profile](https://www.linkedin.com/in/gokul-ram-k-277a6a308/)

---

## Executive Summary

Keystroke dynamics provide a non-invasive, continuous behavioral biometric modality for user identification and access control. However, traditional machine learning models operate as "black boxes", offering minimal interpretability. **TypeXAI** resolves this limitation by coupling a high-performance **XGBoost** tree-ensemble core with a multi-level **Explainable AI (XAI)** framework based on SHAP (SHapley Additive exPlanations) and t-SNE manifold projections.

---

## Key Methodological Contributions

1. **Feature Engineering & Representation**:
   - 31 continuous timing features per record: Hold (`H`), Down-Down (`DD`), and Up-Down (`UD`) latencies across consecutive key events.
   - Evaluated on the industry-standard CMU Keystroke Dynamics Benchmark (51 typists, 20,400 samples).

2. **Discriminative Modeling Layer**:
   - Compared **Multilayer Perceptron (MLP)**, **Random Forest**, and **XGBoost**.
   - Hyperparameter-tuned XGBoost (200 estimators, max depth = 6, learning rate = 0.1, subsample = 0.8) selected as production model.

3. **Multi-Level Explainability Pipeline**:
   - **Global Attributions**: Identified key transition latencies (`UD.n.l`, `UD.Shift.r.o`, `H.Shift.r`) governing overall cognitive signatures.
   - **Local Attributions**: Individual decision pathways traced via SHAP Waterfall plots.
   - **Manifold Visualization**: 2D t-SNE projections confirming distinct user-level cognitive-motor clustering.

---

## Experimental Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (TypeXAI Core)** | **94.93%** | **95.04%** | **94.93%** | **94.92%** |
| Random Forest | 94.02% | 94.16% | 94.02% | 93.98% |
| Multilayer Perceptron (MLP) | 92.08% | 92.24% | 92.08% | 92.05% |

### 5-Fold Stratified Cross-Validation Summary
- **Mean Accuracy**: **94.54% ± 0.0039**
- **Mean Precision**: **94.63% ± 0.0038**
- **Mean Recall**: **94.54% ± 0.0039**
- **Mean F1-Score**: **94.55% ± 0.0039**

---

## Citation (BibTeX)

```bibtex
@article{gokulram2026typexai,
  title={Behavioral Signatures of Cognition: Toward Explainable Keystroke Intelligence},
  author={Gokul Ram K, Rohit N, and Premanand V},
  journal={School of Computer Science and Engineering, Vellore Institute of Technology, Chennai},
  year={2026}
}
```
