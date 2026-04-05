"""
Generate Figure 1: Forest plot of ATE estimates across DAG sources.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load data
with open("/Users/noodles/Desktop/my data/utd document/paper3/pilot_results/ate_results.json") as f:
    ate_data = json.load(f)
with open("/Users/noodles/Desktop/my data/utd document/paper3/pilot_results/ate_comparison.json") as f:
    ate_comp = json.load(f)

expert = ate_data[0]
llm_runs = ate_data[1:]
lit_dags = {r["dag_label"]: r for r in ate_comp
            if any(k in r["dag_label"] for k in ["Lyon", "LaPar", "Hasan"])}

# Classify runs
cluster_a, cluster_b = [], []
for run in llm_runs:
    if run["ate"] is None:
        continue
    (cluster_a if run["n_confounders"] >= 29 else cluster_b).append(run)
cluster_a.sort(key=lambda x: x["ate"])
cluster_b.sort(key=lambda x: x["ate"])

# Build rows: list of (label, ate, ci_lo, ci_hi, color, marker, msize, is_header)
rows = []

def add_header(text):
    rows.append((text, None, None, None, 'black', None, None, True))

def add_row(label, r, color, marker='o', msize=7):
    rows.append((label, r["ate"]*100, r["ci_lower"]*100, r["ci_upper"]*100, color, marker, msize, False))

# Expert
add_header("Expert DAG")
add_row(f'Fu 2025 ({expert["n_confounders"]} covariates)', expert, '#2ca02c', 's', 10)

# Literature
add_header("Published studies")
for key in ["Lyon et al. 2011", "LaPar et al. 2010", "Hasan et al. 2010"]:
    r = lit_dags[key]
    short = key.split(" et al.")[0] + f' ({r["n_confounders"]} covariates)'
    add_row(short, r, '#9467bd', 'D', 8)

# Interp A
add_header("LLM Interpretation A (comorbidities = confounders)")
for run in cluster_a:
    n = run["dag_label"].replace("LLM run ", "")
    add_row(f'Run {n} ({run["n_confounders"]} covariates)', run, '#1f77b4', 'o', 6)

# Interp B
add_header("LLM Interpretation B (comorbidities = mediators)")
for run in cluster_b:
    n = run["dag_label"].replace("LLM run ", "")
    add_row(f'Run {n} ({run["n_confounders"]} covariates)', run, '#ff7f0e', 'o', 6)

# Plot
fig, ax = plt.subplots(figsize=(8, 12))

y = 0
yticks, ylabels = [], []

for label, ate, ci_lo, ci_hi, color, marker, msize, is_header in rows:
    if is_header:
        y += 0.6  # extra gap before header
        ax.text(-0.02, y, label, fontsize=9.5, fontweight='bold', color=color,
                transform=ax.get_yaxis_transform(), ha='right', va='center')
        y += 0.5  # gap after header text
    else:
        ax.plot(ate, y, marker, color=color, markersize=msize, zorder=5)
        ax.plot([ci_lo, ci_hi], [y, y], '-', color=color, linewidth=1.5, zorder=4)
        yticks.append(y)
        ylabels.append(label)
        y += 1

# Null line
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

# Shaded bands
a_ates = [r["ate"]*100 for r in cluster_a]
b_ates = [r["ate"]*100 for r in cluster_b]
ax.axvspan(min(a_ates)-0.003, max(a_ates)+0.003, alpha=0.06, color='#1f77b4', zorder=0)
ax.axvspan(min(b_ates)-0.003, max(b_ates)+0.003, alpha=0.06, color='#ff7f0e', zorder=0)

# Format
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=7.5)
ax.set_xlabel('AIPW-Estimated ATE (percentage points)', fontsize=11)
ax.set_title('Causal Effect Estimates Across DAG Sources\n(Insurance → In-Hospital Mortality, MIMIC-IV, n = 123,498)',
             fontsize=12, fontweight='bold', pad=12)
ax.invert_yaxis()
ax.set_xlim(-0.05, 0.70)
ax.grid(axis='x', alpha=0.25, linewidth=0.5)
ax.tick_params(axis='y', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=9, label='Expert DAG'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#9467bd', markersize=8, label='Published studies'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=7, label='LLM Interp. A'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=7, label='LLM Interp. B'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95, edgecolor='gray')

plt.tight_layout()
plt.subplots_adjust(left=0.32)

base = "/Users/noodles/Desktop/my data/utd document/paper3/manuscript/figure1_forest"
plt.savefig(f"{base}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{base}.pdf", bbox_inches='tight')
print("Done: figure1_forest.png and .pdf saved.")
