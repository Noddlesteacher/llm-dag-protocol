"""
Paper 3: LLM-Assisted Causal DAG Construction
Structured Prompting Protocol — 4 Stages

This module contains the exact prompts used in the experiment.
Each stage is a function that returns the prompt string.
"""

# ============================================================
# VARIABLES (from Paper 1: Insurance -> ICU Mortality in MIMIC-IV)
# ============================================================

TREATMENT = "insurance_type"  # Medicaid=1, Private=0
OUTCOME = "in_hospital_mortality"

# 44 candidate covariates (same as Paper 1 primary adjustment set)
CANDIDATE_VARIABLES = [
    # Demographics
    "age_at_admission",
    "sex",
    "race_white",
    "race_black",
    "race_hispanic",
    "race_asian",
    "race_other",
    "language_english",        # proxy for access barriers
    "marital_status_married",  # proxy for social support

    # Calendar period
    "period_2011_2013",
    "period_2014_2016",
    "period_2017_2019",
    "period_2020_2022",

    # Elixhauser comorbidities (31)
    "chf",
    "arrhythmia",
    "valvular",
    "pulm_circ",
    "pvd",
    "hypertension",
    "paralysis",
    "neuro_other",
    "copd",
    "diabetes_uncomplicated",
    "diabetes_complicated",
    "hypo_thyroid",
    "renal_failure",
    "liver_disease",
    "peptic_ulcer",
    "hiv",
    "lymphoma",
    "metastatic_cancer",
    "solid_tumor",
    "rheumatoid",
    "coagulopathy",
    "obesity",
    "weight_loss",
    "fluid_electrolyte",
    "blood_loss_anemia",
    "deficiency_anemia",
    "alcohol_abuse",
    "drug_abuse",
    "psychosis",
    "depression",
    "copd",  # duplicate removed below
]

# De-duplicate
CANDIDATE_VARIABLES = list(dict.fromkeys(CANDIDATE_VARIABLES))

VARIABLE_LIST_STR = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(CANDIDATE_VARIABLES))

# ============================================================
# RESEARCH CONTEXT (provided to LLM)
# ============================================================

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
- The question is which of these pre-admission variables should be adjusted
  for (as confounders) vs. excluded (if they are mediators, colliders, or
  instruments)
"""

# ============================================================
# STAGE 1: Variable Classification
# ============================================================

def stage1_classify_variables():
    """Prompt LLM to classify each variable's causal role."""
    return f"""You are an expert in causal inference and epidemiology.

{RESEARCH_CONTEXT}

CANDIDATE VARIABLES:
{VARIABLE_LIST_STR}

TASK:
For each variable listed above, classify its causal role with respect to
the treatment (insurance_type) and outcome (in_hospital_mortality).

For each variable, provide:
1. Classification: one of [confounder / mediator / collider / instrument /
   precision_variable / irrelevant]
2. Brief justification (1-2 sentences citing the causal reasoning)

Definitions:
- Confounder: causes both treatment and outcome (or is a proxy for such a cause)
- Mediator: lies on the causal pathway from treatment to outcome
- Collider: caused by both treatment and outcome (conditioning on it introduces bias)
- Instrument: affects treatment but not outcome except through treatment
- Precision variable: affects outcome only (not treatment); safe to include
  but not required for confounding control
- Irrelevant: no meaningful connection to either treatment or outcome

FORMAT your response as a structured list:
VARIABLE: [name]
ROLE: [classification]
REASON: [justification]
---
(repeat for each variable)
"""


# ============================================================
# STAGE 2: Edge Elicitation (Pairwise)
# ============================================================

def stage2_edge_elicitation(stage1_output: str):
    """Prompt LLM to specify directed edges among confounders."""
    return f"""You are an expert in causal inference and epidemiology.

{RESEARCH_CONTEXT}

In a previous step, you classified the causal roles of candidate variables.
Here is your classification:

{stage1_output}

TASK:
Now construct a causal DAG. For each variable you classified as a
CONFOUNDER, specify:
1. Does it have a direct causal edge TO the treatment (insurance_type)? (yes/no)
2. Does it have a direct causal edge TO the outcome (in_hospital_mortality)? (yes/no)
3. Does it have direct causal edges to/from OTHER confounders? If so, list them.

Also identify any variables that are causes of confounders (i.e., ancestors
in the DAG) that might be relevant.

FORMAT your response as:
VARIABLE: [name]
  -> insurance_type: [yes/no, brief reason]
  -> in_hospital_mortality: [yes/no, brief reason]
  -> other edges: [list of "variable -> variable" with brief reasons]
---
(repeat for each confounder)

After listing all edges, provide the COMPLETE edge list as a summary:
EDGE LIST:
[variable1] -> [variable2]
[variable3] -> [variable4]
...
"""


# ============================================================
# STAGE 3: DAG Critique and Revision
# ============================================================

def stage3_critique_dag(edge_list: str):
    """Prompt LLM to critique its own DAG for common errors."""
    return f"""You are an expert in causal inference and epidemiology.

{RESEARCH_CONTEXT}

You previously constructed the following causal DAG (edge list):

{edge_list}

TASK:
Critically review this DAG for common errors. Specifically check:

1. CYCLES: Are there any directed cycles? (A DAG must be acyclic)
2. MISSING CONFOUNDERS: Are there important common causes of treatment
   and outcome that are missing from the graph?
3. COLLIDER BIAS: Would conditioning on any included variable open a
   collider path that introduces bias?
4. MEDIATOR BIAS: Are any variables on the causal pathway from treatment
   to outcome incorrectly included as confounders? (This would block
   part of the causal effect)
5. UNNECESSARY EDGES: Are any edges unsupported by domain knowledge?
6. MISSING EDGES: Are there plausible direct effects that are omitted?

For each issue found:
- Describe the problem
- Propose a fix (add edge, remove edge, reclassify variable)

Then provide the REVISED EDGE LIST incorporating all fixes.

REVISED EDGE LIST:
[variable1] -> [variable2]
...
"""


# ============================================================
# STAGE 4: Adjustment Set Extraction
# ============================================================

def stage4_adjustment_set(revised_edge_list: str):
    """Prompt LLM to derive the minimal sufficient adjustment set."""
    return f"""You are an expert in causal inference and epidemiology.

{RESEARCH_CONTEXT}

Here is the finalized causal DAG (edge list):

{revised_edge_list}

TASK:
Based on this DAG, determine the adjustment set for estimating the
TOTAL causal effect of insurance_type on in_hospital_mortality.

Apply the backdoor criterion:
1. Identify all backdoor paths from insurance_type to in_hospital_mortality
2. Determine the minimal sufficient adjustment set that blocks all
   backdoor paths without opening collider paths
3. Also determine the full sufficient adjustment set (all valid confounders)

FORMAT:
BACKDOOR PATHS:
1. insurance_type <- [var] -> ... -> in_hospital_mortality
2. ...

MINIMAL SUFFICIENT ADJUSTMENT SET:
[list of variables]

FULL SUFFICIENT ADJUSTMENT SET:
[list of variables]

VARIABLES TO EXCLUDE (and why):
[list with reasons — e.g., mediators, colliders, instruments]
"""


# ============================================================
# NEGATIVE CONTROL: Scrambled prompt (no domain context)
# ============================================================

import random
import string

def _scramble_name(name: str, seed: int = 42) -> str:
    """Replace variable name with random letters, preserving length."""
    rng = random.Random(seed + hash(name))
    return "var_" + "".join(rng.choices(string.ascii_lowercase, k=5))

def stage1_negative_control(seed: int = 42):
    """Same structure as stage1 but with scrambled variable names and no context."""
    scrambled = [_scramble_name(v, seed) for v in CANDIDATE_VARIABLES]
    var_list = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(scrambled))

    return f"""You are an expert in causal inference.

STUDY CONTEXT:
- Treatment: var_treatment (binary)
- Outcome: var_outcome (binary)
- Study design: Observational cohort study
- Goal: Estimate the causal effect of var_treatment on var_outcome

CANDIDATE VARIABLES:
{var_list}

TASK:
For each variable listed above, classify its causal role with respect to
var_treatment and var_outcome.

For each variable, provide:
1. Classification: one of [confounder / mediator / collider / instrument /
   precision_variable / irrelevant]
2. Brief justification

FORMAT your response as:
VARIABLE: [name]
ROLE: [classification]
REASON: [justification]
---
"""


# ============================================================
# UTILITY: Parse LLM output into structured DAG
# ============================================================

def parse_edge_list(llm_output: str) -> list[tuple[str, str]]:
    """Extract edges from LLM text output.

    Looks for lines matching: variable1 -> variable2
    Returns list of (source, target) tuples.
    """
    import re
    edges = []
    pattern = r'(\w+)\s*->\s*(\w+)'
    for line in llm_output.split('\n'):
        match = re.search(pattern, line.strip())
        if match:
            src, tgt = match.group(1).strip(), match.group(2).strip()
            edges.append((src, tgt))
    return edges


def parse_adjustment_set(llm_output: str) -> list[str]:
    """Extract the full sufficient adjustment set from Stage 4 output.

    Handles multiple LLM output formats:
      - [var1, var2, var3]           (bracket list)
      - - var1                       (dash list)
      - - **var1**                   (bold markdown)
      - 1. var1                      (numbered list)
      - MINIMAL SUFFICIENT as fallback if FULL not found
    """
    import re

    # Strategy 1: Find FULL SUFFICIENT section; fallback to MINIMAL SUFFICIENT
    text = llm_output
    section_start = None
    for marker in ['FULL SUFFICIENT ADJUSTMENT SET', 'MINIMAL SUFFICIENT ADJUSTMENT SET']:
        idx = text.upper().find(marker)
        if idx != -1:
            section_start = idx + len(marker)
            break

    if section_start is None:
        # Last resort: scan entire output for known variable names
        found = []
        for cv in CANDIDATE_VARIABLES:
            if cv in text:
                found.append(cv)
        return found

    section_text = text[section_start:]

    # Handle "Same as minimal set" / "Same as above" / "All variables in the minimal set"
    same_match = re.search(r'[Ss]ame as (minimal|above)|[Aa]ll variables in the minimal', section_text[:300])
    if same_match:
        # Fall back to MINIMAL SUFFICIENT section
        min_idx = text.upper().find('MINIMAL SUFFICIENT ADJUSTMENT SET')
        if min_idx != -1:
            min_text = text[min_idx + len('MINIMAL SUFFICIENT ADJUSTMENT SET'):]
            # Cut at next section
            for stop in ['FULL SUFFICIENT', 'VARIABLES TO EXCLUDE', '##']:
                si = min_text.upper().find(stop.upper())
                if si != -1:
                    min_text = min_text[:si]
            section_text = min_text

    # Cut off at next section header
    for stop in ['VARIABLES TO EXCLUDE', 'BACKDOOR', '##', 'Note:']:
        stop_idx = section_text.upper().find(stop.upper())
        if stop_idx != -1:
            section_text = section_text[:stop_idx]

    # Strategy 2: Check for bracket list format [var1, var2, ...]
    bracket_match = re.search(r'\[([^\]]+)\]', section_text)
    if bracket_match:
        items = [s.strip().strip('*').strip() for s in bracket_match.group(1).split(',')]
        variables = []
        for item in items:
            if item in CANDIDATE_VARIABLES:
                variables.append(item)
            else:
                for cv in CANDIDATE_VARIABLES:
                    if cv in item or item in cv:
                        variables.append(cv)
                        break
        if variables:
            return list(dict.fromkeys(variables))  # dedupe

    # Strategy 3: Line-by-line parsing (dash, bullet, numbered lists)
    variables = []
    for line in section_text.split('\n'):
        cleaned = line.strip()
        if not cleaned:
            continue
        # Remove markdown bold, bullets, numbers
        cleaned = re.sub(r'^\s*[-\*\d\.\)]+\s*', '', cleaned)
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        cleaned = cleaned.strip().rstrip(':')

        # Check for exact match
        if cleaned in CANDIDATE_VARIABLES:
            variables.append(cleaned)
            continue

        # Check if line contains a known variable name
        for cv in CANDIDATE_VARIABLES:
            if cv in cleaned:
                variables.append(cv)
                # Don't break — line might contain explanation after var name

    return list(dict.fromkeys(variables))  # dedupe


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1 PROMPT (first 500 chars):")
    print("=" * 60)
    print(stage1_classify_variables()[:500])
    print("\n...")
    print(f"\nTotal candidate variables: {len(CANDIDATE_VARIABLES)}")
    print(f"Negative control scrambled names (first 5):")
    for v in CANDIDATE_VARIABLES[:5]:
        print(f"  {v} -> {_scramble_name(v)}")
