# OELLM Dense English Scaling Laws 🇪🇺🤖📈

This project contains the **data and analysis scripts** used to study scaling laws for dense English-language transformer models.

The corresponding collection of checkpoints is available in the 🤗 [Hugging Face repository](https://huggingface.co/openeurollm/dense_english_scaling_laws).

The main objective of this work is to investigate how **hyperparameters and model performance scale** with model size, dataset size, and compute, with a particular focus on learning rate and batch size.

## Data

The main released dataset is:
<span style="color:#cc87e0">data/loss_all_exps_post_annealing.csv</span>
<!-- `data/loss_all_exps_post_annealing.csv` -->

It contains validation losses measured **after learning-rate annealing** for the scaling experiments.

The other datasets in `data/` are intermediate files generated and used during the analysis pipeline.

## Scripts

The `scripts/` directory contains the notebooks and utilities used for data processing, hyperparameter analysis, scaling-law fitting, and uncertainty estimation.

Key analyses include:

* **Optimal hyperparameter estimation**
* **Learning-rate and batch-size scaling laws**
* **Compute-optimal scaling**

The main notebooks and scripts are:

```text
scripts/
├── bootstrap_estimation_chinchilla.py          Bootstrap Chinchilla fits
├── bootstrap_estimation_skaling.py             Bootstrap scaling-law fits
├── scaling_laws_bsz.ipynb                      ⭐️ Batch-size scaling laws
├── scaling_laws_joint_lr_bsz.ipynb             ⭐️ Joint LR / batch-size scaling
├── scaling_laws_lr.ipynb                       ⭐️ Learning-rate scaling laws
├── scaling_laws_compute_optimal.ipynb          ⭐️ Compute-optimal loss scaling laws
├── scaling_laws_loss.ipynb                     ⭐️ Parametric loss scalling laws
├── smooth.ipynb                                Smoothing loss measurements, optimal hyperparameter estimation
├── settings.py                                 Plotting settings and style.
├── utils_bootstrap.py
├── utils_plotting.py
├── utils_quadratic_1d.py
├── utils_quadratic_2d.py
├── utils_scaling_fit.py
└── utils.py
```

The remaining Python files contain shared utilities used throughout the analysis.
