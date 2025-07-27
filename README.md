# Biases and Hallucinations Experiments

This repository contains experiments for measuring biases and hallucinations in language models.

## Repository Structure

- **`biases/`** – Notebooks and data for various bias scenarios.
  - `ApplicationScoring/` – Experiments with synthetic applications used for bias in evaluation setting `data/`.
  - `HolisticBias/` – Experiment for Likelihood bias in Holistic bias dataset.
  - `PosBias/` – Experiments for positional bias with the multiple-choice TruthfulQA dataset.
  - `utils.py` – Helper utilities shared by the notebooks.
- **`hallucinations/`** – Code and Notebooks for evaluating factual accuracy with TruthfulQA.
  - `submit_truthqa_batches.py` – Creates and submits batch requests to the OpenAI API.
  - `poll_truthqa.py` – Polls batch jobs and merges evaluations.
  - `utils.py` – Same helper functions as above.
  - `data/` – gzipped JSONL files with model answers and evaluations.
  - Jupyter notebooks (`thruthful_generation_code.ipynb`, `truthfullEval.ipynb`) for running models and visualizing metrics.
- **`visu/`** – Precomputed result visualizations .

## Getting Started

1. Install dependencies from requirements.txt
2. Set the environment variable `OPENAI_API_KEY` before running any of the OpenAI batch scripts.
3. Open the notebooks in `biases/` and `hallucinations/` to reproduce or extend the experiments.

## Results and Visualizations

Outputs are stored under the respective `data/` directories. Visualizations in `visu/` show aggregated metrics and comparisons between models or prompt settings.