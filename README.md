# Interpretable Multimodal AI for Early Prediction of Prostate Adenocarcinoma Aggressiveness

Reference implementation accompanying the manuscript:

> **An Interpretable Artificial Intelligence Framework for Early Prediction of Prostate Adenocarcinoma Aggressiveness**

This repository provides the complete, runnable pipeline described in the paper's Methodology section:
modality-specific encoders (imaging radiomics, clinical, molecular), cross-modal self-attention fusion,
an interpretable sigmoid prediction head, and a multi-level explainability suite (attribution weights,
attention maps, SHAP).

## ⚠️ Important note on reproducibility scope

This repo implements the **architecture, training procedure, and explainability pipeline exactly as
specified in the manuscript** (Section III, Algorithm 1), with fixed seeds and full configs so that
*given the private clinical cohort*, results are reproducible end-to-end.

It does **not** ship:
- The original patient cohort (multiparametric MRI, ultrasound, clinical records, molecular panels) —
  this is real/de-identified patient data governed by the ethics approval in the manuscript and cannot
  be redistributed here. `scripts/prepare_data.py` defines the exact expected schema and preprocessing
  steps so the authors (or a reviewer with IRB-approved access) can point the pipeline at the real cohort.
- Pretrained weights from a real training run — none currently exist. `checkpoints/` is left empty with
  a `.gitkeep`; running `src/train.py` on real data will populate it. Do not present any weights file
  found elsewhere as originating from this study unless it was produced by this exact pipeline.

Everything else requested for reproducibility — source code, preprocessing/data-prep pipeline,
hyperparameter configuration, random seeds, and inference scripts — is included and functional against
synthetic data out of the box (`scripts/make_synthetic_demo_data.py`), so reviewers can verify the
pipeline runs correctly without needing access to real patient data.

## Repository structure

```
├── configs/config.yaml            # all hyperparameters, paths, seed
├── src/
│   ├── seed.py                    # deterministic seeding (Python/NumPy/PyTorch)
│   ├── data/
│   │   ├── preprocessing.py       # z-score norm, radiomics extraction, MI feature selection
│   │   └── dataset.py             # PyTorch Dataset/DataLoader for the 3 modalities
│   ├── models/
│   │   ├── encoders.py            # f_img, f_clin, f_mol modality encoders
│   │   ├── attention_fusion.py    # cross-modal self-attention (Q,K,V)
│   │   └── framework.py           # full model + composite loss (BCE + sparsity + attn reg)
│   ├── explainability/
│   │   ├── shap_explain.py        # SHAP value computation per feature
│   │   └── attention_maps.py      # modality-level attention visualization
│   ├── train.py                   # training loop, 5 independent runs, checkpointing
│   ├── evaluate.py                # accuracy/AUC/F1/sensitivity/specificity + 95% CI
│   └── infer.py                   # single-patient / batch inference script
├── scripts/
│   ├── prepare_data.py            # expected raw-data schema -> processed tensors
│   ├── make_synthetic_demo_data.py# generates synthetic data so the pipeline is runnable end-to-end
│   └── run_pipeline.sh            # one-command: prep -> train -> evaluate -> explain
├── docs/REPRODUCIBILITY.md        # step-by-step reviewer checklist
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Quickstart (synthetic demo — no real data required)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/make_synthetic_demo_data.py         # creates data/synthetic/*
python src/train.py --config configs/config.yaml    # trains on synthetic data, 5 seeded runs
python src/evaluate.py --config configs/config.yaml  # reports mean ± std metrics with 95% CI
python src/infer.py --config configs/config.yaml --patient_id 0001
```

## Running on the real cohort

1. Place raw data according to the schema in `scripts/prepare_data.py`.
2. `python scripts/prepare_data.py --raw_dir <path> --out_dir data/processed`
3. Point `configs/config.yaml: data.processed_dir` at `data/processed`.
4. Run the same `train.py` / `evaluate.py` / `infer.py` commands above.

## Citation

See `CITATION.cff`. Please cite the manuscript if you use this code.

## License

MIT — see `LICENSE`. Clinical use requires independent validation and regulatory clearance; this
code is provided for research reproducibility only, not for clinical deployment.
