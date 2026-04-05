"""
Literature-derived adjustment sets for insurance → mortality studies.

These represent real-world researcher variability: different teams studying
the same causal question chose different confounder sets.

NOTE: Mapped to MIMIC-IV variables available in our cohort.
Variables in the original papers but unavailable in MIMIC-IV (e.g., APACHE,
hospital characteristics) are noted but excluded.
"""

# Our expert DAG (Paper 1: Fu & Dou 2025)
# 43 variables: demographics + calendar + 30 Elixhauser comorbidities
EXPERT_DAG = {
    "label": "Expert (Fu & Dou 2025)",
    "source": "Companion Paper 1, DAG-informed AIPW",
    "adjustment_set": [
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
    ],
}

# Lyon et al. (2011) AJRCCM 184:809-815
# "The effect of insurance status on mortality in critically ill patients"
# ICU-specific, Project IMPACT database, ~175k patients
# Adjusted for: age, sex, race, APACHE III, ICU type, hospital characteristics,
# admission source, primary diagnosis, mechanical ventilation, DNR status
# MAPPED TO MIMIC-IV: age, sex, race (APACHE/hospital chars unavailable)
LYON_2011 = {
    "label": "Lyon et al. 2011",
    "source": "AJRCCM, Project IMPACT, ICU-specific",
    "adjustment_set": [
        "age_at_admission", "sex",
        "race_white", "race_black", "race_hispanic", "race_asian", "race_other",
        # APACHE III → not in MIMIC admissions, but comorbidities are partial proxy
        # They did NOT adjust for individual comorbidities (used APACHE instead)
        # ICU type, hospital characteristics → not available
        # admission source → excluded (potential mediator in our DAG)
    ],
    "unavailable_in_mimic": [
        "APACHE_III_score", "ICU_type", "hospital_teaching_status",
        "hospital_bed_size", "hospital_region", "admission_source",
        "primary_diagnosis_category", "mechanical_ventilation", "DNR_status"
    ],
}

# LaPar et al. (2010) Ann Surg 252:544-551
# "Primary payer status affects mortality for major surgical operations"
# NIS database, ~894k surgical patients
# Adjusted for: age, sex, race, Elixhauser comorbidity index,
# procedure type, hospital volume/teaching/size/region, admission type, year
# MAPPED TO MIMIC-IV: age, sex, race, Elixhauser comorbidities, calendar period
LAPAR_2010 = {
    "label": "LaPar et al. 2010",
    "source": "Ann Surg, NIS, surgical patients",
    "adjustment_set": [
        "age_at_admission", "sex",
        "race_white", "race_black", "race_hispanic", "race_asian", "race_other",
        # Elixhauser comorbidity INDEX (aggregate score, not individual indicators)
        # We include all individual Elixhauser as closest equivalent
        "chf", "arrhythmia", "valvular", "pulm_circ", "pvd", "hypertension",
        "paralysis", "neuro_other", "copd", "diabetes_uncomplicated",
        "diabetes_complicated", "hypo_thyroid", "renal_failure", "liver_disease",
        "peptic_ulcer", "hiv", "lymphoma", "metastatic_cancer", "solid_tumor",
        "rheumatoid", "coagulopathy", "obesity", "weight_loss",
        "fluid_electrolyte", "blood_loss_anemia", "deficiency_anemia",
        "alcohol_abuse", "drug_abuse", "psychosis", "depression",
        # Calendar period (they adjusted for year)
        "period_2011_2013", "period_2014_2016", "period_2017_2019", "period_2020_2022",
    ],
    "unavailable_in_mimic": [
        "procedure_type", "hospital_volume", "hospital_teaching_status",
        "hospital_bed_size", "hospital_region", "admission_type_elective"
    ],
}

# Hasan et al. (2010) J Hosp Med 5:452-459
# "Insurance status and hospital care for MI, stroke, and pneumonia"
# NIS database, ~1.6M hospitalizations
# Adjusted for: age, sex, race, Charlson comorbidity index,
# hospital characteristics, weekend admission
# MAPPED TO MIMIC-IV: age, sex, race, comorbidities (Charlson subset)
# Charlson includes fewer conditions than Elixhauser — we map the overlap
HASAN_2010 = {
    "label": "Hasan et al. 2010",
    "source": "J Hosp Med, NIS, MI/stroke/pneumonia",
    "adjustment_set": [
        "age_at_admission", "sex",
        "race_white", "race_black", "race_hispanic", "race_asian", "race_other",
        # Charlson comorbidity index — maps to these Elixhauser conditions:
        "chf", "copd", "diabetes_uncomplicated", "diabetes_complicated",
        "renal_failure", "liver_disease", "metastatic_cancer", "solid_tumor",
        "hiv", "pvd", "paralysis", "rheumatoid",
        # Charlson does NOT include: arrhythmia, hypertension, obesity,
        # depression, psychosis, drug/alcohol abuse, etc.
    ],
    "unavailable_in_mimic": [
        "hospital_bed_size", "hospital_teaching_status", "hospital_region",
        "hospital_urban_rural", "weekend_admission", "disease_severity_markers"
    ],
}

# All literature DAGs
LITERATURE_DAGS = [EXPERT_DAG, LYON_2011, LAPAR_2010, HASAN_2010]


def summarize():
    """Print summary of literature-derived adjustment sets."""
    print("Literature-derived adjustment sets for insurance → mortality")
    print("=" * 60)
    for dag in LITERATURE_DAGS:
        n = len(dag["adjustment_set"])
        print(f"\n{dag['label']} ({dag['source']})")
        print(f"  Variables: {n}")
        print(f"  Set: {sorted(dag['adjustment_set'])[:8]}...")
        if "unavailable_in_mimic" in dag:
            print(f"  Unavailable: {dag['unavailable_in_mimic'][:4]}...")

    # Pairwise Jaccard
    print("\n\nPairwise Jaccard similarity:")
    for i in range(len(LITERATURE_DAGS)):
        for j in range(i + 1, len(LITERATURE_DAGS)):
            s1 = set(LITERATURE_DAGS[i]["adjustment_set"])
            s2 = set(LITERATURE_DAGS[j]["adjustment_set"])
            jac = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
            print(f"  {LITERATURE_DAGS[i]['label'][:20]:20s} vs "
                  f"{LITERATURE_DAGS[j]['label'][:20]:20s}: {jac:.3f}")


if __name__ == "__main__":
    summarize()
