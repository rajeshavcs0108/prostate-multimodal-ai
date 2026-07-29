# Reproducibility Checklist

This maps directly to the reviewer's request: "publicly release the complete source code, pretrained
model weights, preprocessing scripts, dataset preparation pipeline, hyperparameter configuration files,
random seeds, and inference scripts."

| Requested item | Status | Location |
|---|---|---|
| Complete source code | ✅ Provided | `src/` |
| Preprocessing scripts | ✅ Provided | `src/data/preprocessing.py` |
| Dataset preparation pipeline | ✅ Provided (schema + transforms) | `scripts/prepare_data.py` |
| Hyperparameter configuration files | ✅ Provided | `configs/config.yaml` |
| Random seeds | ✅ Fixed, logged | `src/seed.py`, `RNG_SEED=42` in config |
| Inference scripts | ✅ Provided | `src/infer.py` |
| Pretrained model weights | ⚠️ Not yet available | see below |
| Raw patient data | ❌ Not redistributable | governed by ethics approval; schema only |

## Why weights aren't included yet

No checkpoint from a real training run on the patient cohort exists at the time of writing. Publishing
a placeholder or synthetic-data checkpoint labeled as if it came from the clinical study would misrepresent
the paper's reported results — so none is included. Once the authors train on the real cohort using
`src/train.py` with the committed config and seed, the resulting `checkpoints/best_model.pt` should be
uploaded to this repo (or archived on Zenodo with a DOI) and this table updated.

## Recommended release process (GitHub + Zenodo)

1. `git init && git add . && git commit -m "Initial reproducibility release"`
2. Create the GitHub repo (e.g. `gh repo create prostate-multimodal-ai --public --source=. --push`,
   or via github.com if you don't have the `gh` CLI).
3. Tag a release once real weights are trained: `git tag v1.0.0 && git push --tags`.
4. Connect the GitHub repo to Zenodo (Zenodo → GitHub → toggle repo on) so each tagged release gets an
   archival DOI — this is the citable, permanent artifact reviewers and readers expect.
5. Add the DOI badge and repo URL to the manuscript's Data/Code Availability statement.

## Determinism notes

- All seeds (`Python random`, `NumPy`, `PyTorch`, CUDA) are fixed via `src/seed.py`, `RNG_SEED=42`.
- `torch.use_deterministic_algorithms(True)` and fixed `cudnn.benchmark=False` are set in `train.py`.
- Five independent runs (paper: "five independent runs of the experiment") are seeded as `42, 43, 44,
  45, 46` and results are reported as mean ± std, matching the manuscript's evaluation protocol.
- Train/val/test split ratios and stratification are fixed in `configs/config.yaml`.
