#!/usr/bin/env python3
"""
Unstructured baseline: ask LLM to produce adjustment set in a SINGLE prompt
(no 4-stage protocol). This tests whether the protocol itself reduces
variability, or whether the LLM is inherently consistent.

20 runs with Claude (Anthropic API).

Usage:
  export ANTHROPIC_API_KEY="your-key"
  python run_unstructured_baseline.py
"""
import json
import os
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "pilot_results" / "unstructured"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same context as the structured protocol
RESEARCH_CONTEXT = """
STUDY CONTEXT:
- Database: MIMIC-IV (Beth Israel Deaconess Medical Center, Boston)
- Population: Adult patients (age >= 18) admitted to the hospital
- Treatment: Insurance type (Medicaid vs. Private insurance)
- Outcome: In-hospital mortality (binary: died during hospitalization or not)
- Study design: Retrospective observational cohort study
- Goal: Estimate the causal effect of Medicaid (vs. private insurance) on
  in-hospital mortality, using doubly robust methods (AIPW)

KEY BACKGROUND:
- Medicaid patients tend to be younger, lower-income, and have higher
  comorbidity burden than privately insured patients
- Insurance type is determined before hospital admission
- All covariates listed are measured at or before admission (pre-treatment)
"""

CANDIDATE_VARIABLES = [
    "age_at_admission", "sex", "race_white", "race_black", "race_hispanic",
    "race_asian", "race_other", "language_english", "marital_status_married",
    "period_2011_2013", "period_2014_2016", "period_2017_2019", "period_2020_2022",
    "chf", "arrhythmia", "valvular", "pulm_circ", "pvd", "hypertension",
    "paralysis", "neuro_other", "copd", "diabetes_uncomplicated",
    "diabetes_complicated", "hypo_thyroid", "renal_failure", "liver_disease",
    "peptic_ulcer", "hiv", "lymphoma", "metastatic_cancer", "solid_tumor",
    "rheumatoid", "coagulopathy", "obesity", "weight_loss", "fluid_electrolyte",
    "blood_loss_anemia", "deficiency_anemia", "alcohol_abuse", "drug_abuse",
    "psychosis", "depression",
]

VARIABLE_LIST_STR = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(CANDIDATE_VARIABLES))

UNSTRUCTURED_PROMPT = f"""You are an epidemiologist analyzing MIMIC-IV data.

{RESEARCH_CONTEXT}

Here are the available covariates:
{VARIABLE_LIST_STR}

Which of these variables should be included in the adjustment set to estimate
the causal effect of insurance type on in-hospital mortality? Just list the
variable names you would adjust for, separated by commas. Only include variables
that should be in the adjustment set; exclude mediators, colliders, and irrelevant
variables.

ADJUSTMENT SET:
"""


def call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def parse_adjustment_set(text: str) -> list:
    """Try to extract variable names from free-form LLM response."""
    found = []
    text_lower = text.lower()
    for var in CANDIDATE_VARIABLES:
        if var.lower() in text_lower:
            found.append(var)
    return found


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    n_runs = 20
    print(f"Running {n_runs} UNSTRUCTURED baseline runs (single prompt, no protocol)...")
    print("=" * 60)

    results = []
    for i in range(n_runs):
        print(f"\n--- Unstructured Run {i+1}/{n_runs} ---")
        cache_file = OUTPUT_DIR / f"unstructured_{i}.json"

        if cache_file.exists():
            print(f"  Loading cached result")
            with open(cache_file) as f:
                res = json.load(f)
            results.append(res)
            adj = res.get("adjustment_set", [])
            print(f"  Adjustment set size: {len(adj)}")
            continue

        try:
            response = call_anthropic(UNSTRUCTURED_PROMPT)
            adj_set = parse_adjustment_set(response)

            res = {
                "run_id": i,
                "type": "unstructured",
                "raw_response": response,
                "adjustment_set": adj_set,
                "adjustment_set_size": len(adj_set),
            }
            results.append(res)

            with open(cache_file, 'w') as f:
                json.dump(res, f, indent=2)

            print(f"  Adjustment set size: {len(adj_set)}")
            time.sleep(1)  # rate limit

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    sizes = [r["adjustment_set_size"] for r in results]
    print(f"\n{'='*60}")
    print(f"DONE. {len(results)} runs completed.")
    print(f"Adjustment set sizes: {sizes}")
    print(f"Mean size: {sum(sizes)/len(sizes):.1f}")
    print(f"Range: {min(sizes)}-{max(sizes)}")

    # Compute variability metrics
    from collections import Counter
    all_vars = {}
    for r in results:
        for v in r["adjustment_set"]:
            all_vars[v] = all_vars.get(v, 0) + 1

    n_total = len(results)
    consensus = [v for v, c in all_vars.items() if c == n_total]
    contested = [v for v, c in all_vars.items() if 0 < c < n_total]

    print(f"\nConsensus variables (in all runs): {len(consensus)}")
    print(f"Contested variables: {len(contested)}")
    for v in sorted(contested, key=lambda x: all_vars[x], reverse=True):
        print(f"  {v}: {all_vars[v]}/{n_total} runs")

    # Save summary
    summary = {
        "n_runs": len(results),
        "type": "unstructured_baseline",
        "adjustment_set_sizes": sizes,
        "mean_size": sum(sizes) / len(sizes),
        "consensus_variables": consensus,
        "contested_variables": {v: all_vars[v] for v in contested},
    }
    with open(OUTPUT_DIR / "unstructured_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
