#!/usr/bin/env python3
"""
Cross-model robustness check: run the DAG protocol 5 times with Gemini.
Saves results separately from the main Claude runs.

Usage:
  export GOOGLE_API_KEY="your-key-here"
  python run_gemini_robustness.py
"""
import os
import sys
import json
from pathlib import Path

# Override to only 5 runs
os.environ.setdefault("PILOT_RUNS_OVERRIDE", "5")

OUTPUT_DIR = Path(__file__).parent / "pilot_results" / "gemini"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import from pilot.py
sys.path.insert(0, str(Path(__file__).parent))
from pilot import run_llm_protocol, CACHE_DIR

# Override cache dir for gemini
import pilot
original_cache = pilot.CACHE_DIR
pilot.CACHE_DIR = OUTPUT_DIR / "llm_cache"
pilot.CACHE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Set GOOGLE_API_KEY environment variable")
        print("  export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)

    n_runs = 5
    print(f"Running {n_runs} Gemini robustness runs...")
    print("=" * 60)

    results = []
    for i in range(n_runs):
        print(f"\n--- Gemini Run {i+1}/{n_runs} ---")
        try:
            res = run_llm_protocol(run_id=100+i, api="gemini", negative_control=False)
            results.append(res)
            adj_set = res.get("adjustment_set", [])
            print(f"  Adjustment set size: {len(adj_set)}")
            print(f"  Variables: {adj_set[:5]}{'...' if len(adj_set) > 5 else ''}")
        except Exception as e:
            print(f"  ERROR in run {i+1}: {e}")
            continue

    # Save summary
    summary = {
        "n_runs": len(results),
        "model": "gemini-2.0-flash",
        "adjustment_sets": [r.get("adjustment_set", []) for r in results],
        "adjustment_set_sizes": [len(r.get("adjustment_set", [])) for r in results],
    }

    summary_file = OUTPUT_DIR / "gemini_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE. {len(results)} runs completed.")
    print(f"Adjustment set sizes: {summary['adjustment_set_sizes']}")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
