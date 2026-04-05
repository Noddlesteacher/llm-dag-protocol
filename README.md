# LLM-DAG Protocol

Structured elicitation protocol for reproducible DAG construction in causal inference.

Companion code for: "DAG Construction as a Reproducibility Problem: Structured Elicitation Reduces Analyst Variability in Confounder Selection"

## Files

- `pilot.py` — Main experiment: runs the 4-stage protocol (supports Claude, GPT, Gemini)
- `prompts.py` — Exact prompts used in all four stages
- `run_unstructured_baseline.py` — Unstructured baseline (single-prompt, no protocol)
- `run_gemini_robustness.py` — Cross-model robustness check with Gemini
- `build_cohort.py` — Constructs the MIMIC-IV analysis cohort
- `generate_figure.py` — Generates Figure 1 (forest plot)
- `literature_dags.py` — Extracts adjustment sets from published studies

## Results

- `pilot_results/llm_cache/` — Raw LLM outputs from 20 structured Claude runs + 5 negative controls
- `pilot_results/unstructured/` — 20 unstructured Claude baseline runs
- `pilot_results/gemini/` — 5 structured Gemini runs
- `pilot_results/*.json` — Summary statistics (ATEs, DAG metrics, comparisons)

## Usage

```bash
# Structured protocol (20 runs)
export ANTHROPIC_API_KEY="your-key"
python pilot.py --api anthropic

# Unstructured baseline
python run_unstructured_baseline.py

# Cross-model check with Gemini
export GOOGLE_API_KEY="your-key"
python run_gemini_robustness.py
```

## Data

MIMIC-IV access requires PhysioNet credentials: https://physionet.org
