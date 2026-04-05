#!/usr/bin/env python3
"""
Paper 3: LLM-Assisted Causal DAG Construction — Pilot Experiment

Pilot plan:
  1. Run LLM protocol 5 times (independent sessions)
  2. Compare DAGs: edge agreement, confounder set overlap
  3. Run AIPW under each DAG's confounder set
  4. Compare ATE estimates across DAGs
  5. Run negative control (scrambled prompt) for comparison

Usage:
  python pilot.py --api anthropic   # use Claude
  python pilot.py --api openai      # use GPT
  python pilot.py --skip-llm        # skip LLM calls, use cached results
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

PILOT_RUNS = 20
NEGATIVE_CONTROL_RUNS = 5
OUTPUT_DIR = Path(__file__).parent / "pilot_results"
CACHE_DIR = OUTPUT_DIR / "llm_cache"


# ============================================================
# PHASE 1: LLM DAG GENERATION
# ============================================================

def run_llm_protocol(run_id: int, api: str = "anthropic",
                     negative_control: bool = False) -> dict:
    """Run the full 4-stage protocol once and return structured results."""
    from prompts import (
        stage1_classify_variables, stage1_negative_control,
        stage2_edge_elicitation, stage3_critique_dag,
        stage4_adjustment_set, parse_edge_list, parse_adjustment_set
    )

    cache_prefix = "neg" if negative_control else "run"
    cache_file = CACHE_DIR / f"{cache_prefix}_{run_id}.json"

    if cache_file.exists():
        print(f"  Loading cached result: {cache_file.name}")
        with open(cache_file) as f:
            return json.load(f)

    result = {"run_id": run_id, "negative_control": negative_control,
              "api": api, "stages": {}}

    def call_llm(prompt: str) -> str:
        """Call LLM API and return response text."""
        if api == "anthropic":
            return _call_anthropic(prompt)
        elif api == "openai":
            return _call_openai(prompt)
        elif api == "gemini":
            return _call_gemini(prompt)
        else:
            raise ValueError(f"Unknown API: {api}")

    # Stage 1
    print(f"  Stage 1: Variable classification...")
    if negative_control:
        s1_prompt = stage1_negative_control(seed=run_id)
    else:
        s1_prompt = stage1_classify_variables()
    s1_output = call_llm(s1_prompt)
    result["stages"]["stage1"] = s1_output

    if negative_control:
        # For negative control, only run Stage 1
        result["adjustment_set"] = []
        result["edges"] = []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        return result

    # Stage 2
    print(f"  Stage 2: Edge elicitation...")
    s2_prompt = stage2_edge_elicitation(s1_output)
    s2_output = call_llm(s2_prompt)
    result["stages"]["stage2"] = s2_output

    # Stage 3
    print(f"  Stage 3: DAG critique...")
    edges_raw = parse_edge_list(s2_output)
    edge_str = "\n".join(f"{s} -> {t}" for s, t in edges_raw)
    s3_prompt = stage3_critique_dag(edge_str)
    s3_output = call_llm(s3_prompt)
    result["stages"]["stage3"] = s3_output

    # Stage 4
    print(f"  Stage 4: Adjustment set extraction...")
    revised_edges = parse_edge_list(s3_output)
    if not revised_edges:
        revised_edges = edges_raw  # fallback if critique didn't change
    revised_str = "\n".join(f"{s} -> {t}" for s, t in revised_edges)
    s4_prompt = stage4_adjustment_set(revised_str)
    s4_output = call_llm(s4_prompt)
    result["stages"]["stage4"] = s4_output

    # Parse final outputs
    result["edges"] = [(s, t) for s, t in parse_edge_list(s3_output)]
    result["adjustment_set"] = parse_adjustment_set(s4_output)

    # Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def _call_anthropic(prompt: str) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.7,  # >0 for variability measurement
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai")
        sys.exit(1)

    client = OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str) -> str:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: pip install google-generativeai")
        sys.exit(1)

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=16384,
            temperature=0.7,
        )
    )
    return response.text


# ============================================================
# PHASE 2: DAG COMPARISON METRICS
# ============================================================

def structural_hamming_distance(edges1: list, edges2: list,
                                all_vars: list) -> int:
    """Compute SHD between two DAGs represented as edge lists."""
    set1 = set((s, t) for s, t in edges1)
    set2 = set((s, t) for s, t in edges2)
    # Missing edges + extra edges + reversed edges
    missing = set2 - set1
    extra = set1 - set2
    return len(missing) + len(extra)


def jaccard_index(set1: list, set2: list) -> float:
    """Jaccard similarity between two adjustment sets."""
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2:
        return 1.0
    intersection = s1 & s2
    union = s1 | s2
    return len(intersection) / len(union) if union else 0.0


def edge_agreement_rate(all_runs: list[dict]) -> dict:
    """For each edge, compute fraction of runs that include it."""
    from collections import Counter
    edge_counts = Counter()
    n = len(all_runs)
    for run in all_runs:
        for edge in run["edges"]:
            edge_counts[tuple(edge)] += 1
    return {edge: count / n for edge, count in edge_counts.items()}


def compute_dag_metrics(all_runs: list[dict], expert_set: list[str]) -> dict:
    """Compute all DAG comparison metrics across runs."""
    n = len(all_runs)

    # Pairwise SHD
    shd_pairs = []
    for i, j in combinations(range(n), 2):
        shd = structural_hamming_distance(
            all_runs[i]["edges"], all_runs[j]["edges"], [])
        shd_pairs.append({"run_i": i, "run_j": j, "SHD": shd})

    # Pairwise Jaccard on adjustment sets
    jaccard_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard_index(
            all_runs[i]["adjustment_set"], all_runs[j]["adjustment_set"])
        jaccard_pairs.append({"run_i": i, "run_j": j, "jaccard": jac})

    # Jaccard with expert
    expert_jaccard = []
    for i, run in enumerate(all_runs):
        jac = jaccard_index(run["adjustment_set"], expert_set)
        expert_jaccard.append({"run": i, "jaccard_vs_expert": jac})

    # Edge agreement
    edge_agree = edge_agreement_rate(all_runs)

    # Summary stats
    import numpy as np
    shd_values = [p["SHD"] for p in shd_pairs]
    jac_values = [p["jaccard"] for p in jaccard_pairs]
    expert_jac_values = [p["jaccard_vs_expert"] for p in expert_jaccard]

    return {
        "pairwise_shd": shd_pairs,
        "pairwise_jaccard": jaccard_pairs,
        "expert_jaccard": expert_jaccard,
        "edge_agreement": {str(k): v for k, v in edge_agree.items()},
        "summary": {
            "mean_pairwise_SHD": float(np.mean(shd_values)) if shd_values else None,
            "std_pairwise_SHD": float(np.std(shd_values)) if shd_values else None,
            "mean_pairwise_jaccard": float(np.mean(jac_values)) if jac_values else None,
            "mean_expert_jaccard": float(np.mean(expert_jac_values)) if expert_jac_values else None,
            "n_runs": n,
        }
    }


# ============================================================
# PHASE 3: AIPW UNDER EACH DAG
# ============================================================

def run_aipw_for_dag(confounder_set: list[str], data_path: str,
                     dag_label: str) -> dict:
    """Run 5-fold cross-fitted AIPW with a given confounder set.

    Reuses the Paper 1 AIPW logic but with a flexible confounder set.
    """
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    # Load data
    df = pd.read_csv(data_path)

    # Ensure all confounders exist in data
    available = [c for c in confounder_set if c in df.columns]
    missing = [c for c in confounder_set if c not in df.columns]
    if missing:
        print(f"    Warning: {len(missing)} variables not in data: {missing[:5]}...")

    if not available:
        return {"dag_label": dag_label, "ate": None, "se": None,
                "ci_lower": None, "ci_upper": None,
                "n_confounders": 0, "error": "No valid confounders"}

    X = df[available].values
    A = df["treatment"].values
    Y = df["outcome"].values
    n = len(Y)

    # 5-fold cross-fitted AIPW
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pseudo_outcomes = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        A_tr, A_te = A[train_idx], A[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]

        # Propensity score model
        ps_model = LogisticRegression(max_iter=1000, C=1.0)
        ps_model.fit(X_tr, A_tr)
        e_hat = ps_model.predict_proba(X_te)[:, 1]
        e_hat = np.clip(e_hat, 0.01, 0.99)

        # Outcome models
        mu1_model = LogisticRegression(max_iter=1000, C=1.0)
        mu0_model = LogisticRegression(max_iter=1000, C=1.0)

        treated = A_tr == 1
        control = A_tr == 0

        if treated.sum() > 0 and control.sum() > 0:
            mu1_model.fit(X_tr[treated], Y_tr[treated])
            mu0_model.fit(X_tr[control], Y_tr[control])
            mu1_hat = mu1_model.predict_proba(X_te)[:, 1]
            mu0_hat = mu0_model.predict_proba(X_te)[:, 1]
        else:
            mu1_hat = np.full(len(X_te), Y_tr.mean())
            mu0_hat = np.full(len(X_te), Y_tr.mean())

        # AIPW pseudo-outcomes
        phi1 = mu1_hat + A_te * (Y_te - mu1_hat) / e_hat
        phi0 = mu0_hat + (1 - A_te) * (Y_te - mu0_hat) / (1 - e_hat)
        pseudo_outcomes[test_idx] = phi1 - phi0

    ate = pseudo_outcomes.mean()
    se = pseudo_outcomes.std() / np.sqrt(n)

    return {
        "dag_label": dag_label,
        "ate": float(ate),
        "se": float(se),
        "ci_lower": float(ate - 1.96 * se),
        "ci_upper": float(ate + 1.96 * se),
        "n_confounders": len(available),
        "confounders_used": available,
    }


# ============================================================
# PHASE 4: EXPERT DAG (Paper 1 benchmark)
# ============================================================

EXPERT_ADJUSTMENT_SET = [
    "age_at_admission", "sex",
    "race_white", "race_black", "race_hispanic", "race_asian", "race_other",
    "language_english", "marital_status_married",
    "period_2011_2013", "period_2014_2016", "period_2017_2019", "period_2020_2022",
    "chf", "arrhythmia", "valvular", "pulm_circ", "pvd", "hypertension",
    "paralysis", "neuro_other", "copd", "diabetes_uncomplicated",
    "diabetes_complicated", "hypo_thyroid", "renal_failure", "liver_disease",
    "peptic_ulcer", "hiv", "lymphoma", "metastatic_cancer", "solid_tumor",
    "rheumatoid", "coagulopathy", "obesity", "weight_loss",
    "fluid_electrolyte", "blood_loss_anemia", "deficiency_anemia",
    "alcohol_abuse", "drug_abuse", "psychosis", "depression",
]


# ============================================================
# MAIN PILOT PIPELINE
# ============================================================

def run_pilot(api: str = "anthropic", skip_llm: bool = False,
              data_path: str = None):
    """Execute the full pilot experiment."""
    import numpy as np

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PAPER 3 PILOT EXPERIMENT")
    print("=" * 60)

    # ---- Step 1: Run LLM protocol ----
    if not skip_llm:
        print(f"\n--- Step 1: Running LLM protocol ({PILOT_RUNS} runs) ---")
        for i in range(PILOT_RUNS):
            print(f"\nRun {i+1}/{PILOT_RUNS}:")
            run_llm_protocol(i, api=api, negative_control=False)
            time.sleep(2)  # rate limit courtesy

        print(f"\n--- Running negative control ({NEGATIVE_CONTROL_RUNS} runs) ---")
        for i in range(NEGATIVE_CONTROL_RUNS):
            print(f"\nNegative control {i+1}/{NEGATIVE_CONTROL_RUNS}:")
            run_llm_protocol(i, api=api, negative_control=True)
            time.sleep(1)

    # ---- Step 2: Load all results ----
    print("\n--- Step 2: Loading results ---")
    all_runs = []
    for i in range(PILOT_RUNS):
        cache_file = CACHE_DIR / f"run_{i}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                all_runs.append(json.load(f))
    print(f"  Loaded {len(all_runs)} LLM runs")

    neg_runs = []
    for i in range(NEGATIVE_CONTROL_RUNS):
        cache_file = CACHE_DIR / f"neg_{i}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                neg_runs.append(json.load(f))
    print(f"  Loaded {len(neg_runs)} negative control runs")

    if not all_runs:
        print("  ERROR: No LLM results found. Run without --skip-llm first.")
        return

    # ---- Step 3: DAG comparison metrics ----
    print("\n--- Step 3: DAG comparison metrics ---")
    metrics = compute_dag_metrics(all_runs, EXPERT_ADJUSTMENT_SET)

    print(f"\n  LLM Internal Consistency ({len(all_runs)} runs):")
    print(f"    Mean pairwise SHD:     {metrics['summary']['mean_pairwise_SHD']:.1f}")
    print(f"    Std pairwise SHD:      {metrics['summary']['std_pairwise_SHD']:.1f}")
    print(f"    Mean pairwise Jaccard: {metrics['summary']['mean_pairwise_jaccard']:.3f}")
    print(f"    Mean expert Jaccard:   {metrics['summary']['mean_expert_jaccard']:.3f}")

    # Save metrics
    with open(OUTPUT_DIR / "dag_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # ---- Step 4: AIPW under each DAG ----
    if data_path and os.path.exists(data_path):
        print(f"\n--- Step 4: AIPW under each DAG ---")

        ate_results = []

        # Expert DAG
        print("  Running AIPW with expert DAG...")
        expert_result = run_aipw_for_dag(
            EXPERT_ADJUSTMENT_SET, data_path, "Expert (Paper 1)")
        ate_results.append(expert_result)
        print(f"    ATE = {expert_result['ate']:.4f} "
              f"({expert_result['n_confounders']} confounders)")

        # LLM DAGs
        for i, run in enumerate(all_runs):
            print(f"  Running AIPW with LLM DAG (run {i})...")
            r = run_aipw_for_dag(
                run["adjustment_set"], data_path, f"LLM run {i}")
            ate_results.append(r)
            if r["ate"] is not None:
                print(f"    ATE = {r['ate']:.4f} "
                      f"({r['n_confounders']} confounders)")

        # Summary
        llm_ates = [r["ate"] for r in ate_results
                    if r["dag_label"].startswith("LLM") and r["ate"] is not None]

        if llm_ates:
            llm_ates = np.array(llm_ates)
            ate_mean = llm_ates.mean()
            ate_std = llm_ates.std()
            ate_range = llm_ates.max() - llm_ates.min()
            cv = ate_std / abs(ate_mean) if abs(ate_mean) > 1e-8 else float('inf')

            print(f"\n  ATE Summary (LLM runs):")
            print(f"    Mean ATE:  {ate_mean:.4f}")
            print(f"    Std ATE:   {ate_std:.4f}")
            print(f"    Range:     {ate_range:.4f}")
            print(f"    CV:        {cv:.3f}")
            print(f"    Expert ATE: {expert_result['ate']:.4f}")

        # Save
        with open(OUTPUT_DIR / "ate_results.json", 'w') as f:
            json.dump(ate_results, f, indent=2)
    else:
        print(f"\n--- Step 4: SKIPPED (no data_path provided) ---")
        print(f"  To run AIPW, provide --data-path to MIMIC-IV cohort CSV")

    # ---- Step 5: Print pilot summary ----
    print("\n" + "=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)

    # Adjustment set sizes
    set_sizes = [len(r["adjustment_set"]) for r in all_runs]
    print(f"\n  Adjustment set sizes: {set_sizes}")
    print(f"  Expert set size: {len(EXPERT_ADJUSTMENT_SET)}")

    # Overlap between runs
    if len(all_runs) >= 2:
        common = set(all_runs[0]["adjustment_set"])
        for run in all_runs[1:]:
            common &= set(run["adjustment_set"])
        any_set = set()
        for run in all_runs:
            any_set |= set(run["adjustment_set"])
        print(f"  Variables in ALL runs: {len(common)} / {len(any_set)} total")
        print(f"    Always included: {sorted(common)[:10]}...")

    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print("\nDone.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper 3 Pilot Experiment")
    parser.add_argument("--api", choices=["anthropic", "openai", "gemini"],
                        default="anthropic",
                        help="LLM API to use (default: anthropic)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM calls, use cached results")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to MIMIC-IV cohort CSV for AIPW")
    args = parser.parse_args()

    run_pilot(api=args.api, skip_llm=args.skip_llm,
              data_path=args.data_path)
