# Ab Initio Discovery of Metastable Memory Motifs Driving Hysteresis in the Drosophila Connectome

This repository contains the code and data necessary to reproduce the findings and figures presented in our paper:
1.  **The specific circuit data** extracted from the FlyWire connectome ($N=2220$ neurons).
2.  **A unified Jupyter Notebook** (`Figure.ipynb`) to reproduce all figures (Figures 1-5) in the manuscript.
3.  **Simulation logic** based on the Brian2 simulator.



# Data Source & Reproducibility Note

The connectivity data used in this study (`skeleton2220_data.pkl`) represents the sensorimotor circuit governing feeding and escape behaviors.

*   **Source:** [FlyWire](https://flywire.ai/) Whole-Brain Connectome.
*   **Materialization Version:** **v783**.
*   **Synapse Prediction Model:** The connectivity matrix was constructed using the specific version labeled:
    > *"Connections Predicted With Buhmann Et. Al. [Original Version Used Prior To July 2025]"*

### Robustness Verification
While our primary analysis relies on the specific Buhmann et al. prediction version mentioned above, we have verified that constructing the connectivity matrix using the **latest v783 release** yields qualitatively identical results.
