#!/usr/bin/env python3
"""
Build Paper 3 cohort CSV from Paper 1 raw MIMIC-IV data.

Runs the same cohort construction logic as Paper 1 (analysis_v2.py)
and saves a clean CSV with standardized column names for Paper 3.

Input:  Raw MIMIC-IV files in Paper 1 directory
Output: paper3_cohort.csv in Paper 3 directory
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PAPER1_DIR = "/Users/noodles/Desktop/my data/utd document/with Dou paper1"
OUTPUT_PATH = "/Users/noodles/Desktop/my data/utd document/paper3/paper3_cohort.csv"

print("=" * 60)
print("Building Paper 3 cohort from MIMIC-IV")
print("=" * 60)

# ---- Load raw data ----
print("\nLoading raw data...")
admissions = pd.read_csv(f"{PAPER1_DIR}/admissions.csv.gz")
patients = pd.read_csv(f"{PAPER1_DIR}/patients.csv.gz")
diagnoses = pd.read_csv(f"{PAPER1_DIR}/diagnoses_icd.csv.gz")
print(f"  Admissions: {len(admissions):,}")

# ---- Merge & filter ----
df = admissions.merge(
    patients[['subject_id', 'anchor_age', 'anchor_year', 'anchor_year_group', 'gender']],
    on='subject_id', how='left'
)
df['admittime'] = pd.to_datetime(df['admittime'])
df['dischtime'] = pd.to_datetime(df['dischtime'])
df['admit_year'] = df['admittime'].dt.year
df['age_at_admission'] = df['anchor_age'] + (df['admit_year'] - df['anchor_year'])

df = df[df['age_at_admission'] >= 18].copy()
df = df.sort_values('admittime').groupby('subject_id', as_index=False).first()
df = df[df['insurance'].isin(['Medicaid', 'Private'])].copy()

# Calendar period
period_order = ['2008 - 2010', '2011 - 2013', '2014 - 2016', '2017 - 2019', '2020 - 2022']
for pg in period_order[1:]:
    safe = pg.replace(' ', '').replace('-', '_')
    df[f'period_{safe}'] = (df['anchor_year_group'] == pg).astype(int)

# Language & marital
df['language_english'] = (df['language'].str.upper() == 'ENGLISH').astype(int)
df['marital_status_married'] = (df['marital_status'] == 'MARRIED').astype(int)

# Drop missing
key_vars = ['age_at_admission', 'gender', 'race', 'insurance',
            'hospital_expire_flag', 'language', 'marital_status',
            'admission_type', 'anchor_year_group']
df = df.dropna(subset=key_vars).copy()

# Treatment & outcome
df['treatment'] = (df['insurance'] == 'Medicaid').astype(int)
df['outcome'] = df['hospital_expire_flag'].astype(int)

print(f"  Cohort after filters: {len(df):,}")

# ---- Elixhauser comorbidities ----
print("Mapping Elixhauser comorbidities...")

ELIX_ICD10 = {
    'chf': ['I50'], 'arrhythmia': ['I47', 'I48', 'I49'],
    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15'],
    'diabetes_uncomplicated': ['E100', 'E101', 'E109', 'E110', 'E111', 'E119'],
    'diabetes_complicated': ['E102', 'E103', 'E104', 'E105', 'E106', 'E107',
                             'E112', 'E113', 'E114', 'E115', 'E116', 'E117'],
    'renal_failure': ['N17', 'N18', 'N19'],
    'liver_disease': ['K70', 'K711', 'K713', 'K714', 'K715', 'K717',
                      'K72', 'K73', 'K74', 'K760', 'K762', 'K763',
                      'K764', 'K765', 'K766', 'K767'],
    'copd': ['J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47',
             'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67'],
    'coagulopathy': ['D65', 'D66', 'D67', 'D68', 'D691', 'D693', 'D694', 'D695', 'D696'],
    'metastatic_cancer': ['C77', 'C78', 'C79', 'C80'],
    'solid_tumor': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C70', 'C71', 'C72',
                    'C73', 'C74', 'C75'],
    'obesity': ['E66'], 'depression': ['F32', 'F33'],
    'psychosis': ['F20', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29'],
    'fluid_electrolyte': ['E222', 'E86', 'E87'],
    'blood_loss_anemia': ['D500'],
    'deficiency_anemia': ['D508', 'D509', 'D51', 'D52', 'D53'],
    'alcohol_abuse': ['F10', 'E52', 'G621', 'I426', 'K292', 'K700', 'K703',
                      'K709', 'T51', 'Z502', 'Z714', 'Z721'],
    'drug_abuse': ['F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18', 'F19'],
    'hypo_thyroid': ['E00', 'E01', 'E02', 'E03'],
    'pulm_circ': ['I26', 'I27', 'I280', 'I288', 'I289'],
    'pvd': ['I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792',
            'K551', 'K558', 'K559', 'Z958', 'Z959'],
    'paralysis': ['G041', 'G114', 'G801', 'G802', 'G81', 'G82',
                  'G830', 'G831', 'G832', 'G833', 'G834', 'G839'],
    'neuro_other': ['G10', 'G11', 'G12', 'G13', 'G20', 'G21', 'G22',
                    'G254', 'G255', 'G312', 'G318', 'G319', 'G32', 'G35', 'G36', 'G37'],
    'rheumatoid': ['L940', 'L941', 'L943', 'M05', 'M06', 'M08', 'M120', 'M123',
                   'M30', 'M31', 'M32', 'M33', 'M34', 'M35', 'M45', 'M461', 'M468', 'M469'],
    'peptic_ulcer': ['K257', 'K259', 'K267', 'K269', 'K277', 'K279', 'K287', 'K289'],
    'hiv': ['B20', 'B21', 'B22', 'B24'],
    'lymphoma': ['C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C96', 'C900', 'C902'],
    'valvular': ['A520', 'I05', 'I06', 'I07', 'I08', 'I091', 'I098',
                 'I34', 'I35', 'I36', 'I37', 'I38', 'I39',
                 'Q230', 'Q231', 'Q232', 'Q233', 'Z952', 'Z953', 'Z954'],
    'weight_loss': ['E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'R634', 'R64'],
}

ELIX_ICD9 = {
    'chf': ['4254', '4255', '4257', '4258', '4259', '428'],
    'arrhythmia': ['4260', '4267', '4269', '427', '7850', 'V450', 'V533'],
    'hypertension': ['401', '402', '403', '404', '405'],
    'diabetes_uncomplicated': ['2500', '2501', '2502', '2503'],
    'diabetes_complicated': ['2504', '2505', '2506', '2507', '2508', '2509'],
    'renal_failure': ['585', '586', 'V420', 'V451', 'V56'],
    'liver_disease': ['07022', '07023', '07032', '07033', '07044', '07054',
                      '4560', '4561', '4562', '570', '571', '5722', '5723', '5724', '5728', 'V427'],
    'copd': ['490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505'],
    'coagulopathy': ['286', '2871', '2873', '2874', '2875'],
    'metastatic_cancer': ['196', '197', '198', '199'],
    'solid_tumor': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149',
                    '150', '151', '152', '153', '154', '155', '156', '157', '158', '159',
                    '160', '161', '162', '163', '164', '165', '170', '171', '172', '174',
                    '175', '176', '179', '180', '181', '182', '183', '184', '185', '186',
                    '187', '188', '189', '190', '191', '192', '193', '194', '195'],
    'obesity': ['2780'], 'depression': ['2962', '2963', '2965', '3004', '309', '311'],
    'psychosis': ['2938', '295', '2970', '2971', '2983'],
    'fluid_electrolyte': ['2536', '276'],
    'blood_loss_anemia': ['2800'],
    'deficiency_anemia': ['2801', '2808', '2809', '281'],
    'alcohol_abuse': ['2652', '2911', '2912', '2913', '2915', '2918', '2919',
                      '3030', '3039', '3050', 'V113'],
    'drug_abuse': ['292', '304', '3052', '3053', '3054', '3055', '3056', '3057', '3058', '3059'],
    'hypo_thyroid': ['2409', '243', '244', '2461', '2468'],
    'pulm_circ': ['4150', '4151', '416', '4170', '4178', '4179'],
    'pvd': ['0930', '4373', '440', '441', '4431', '4432', '4438', '4439',
            '4471', '5571', '5579', 'V434'],
    'paralysis': ['3341', '342', '343', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449'],
    'neuro_other': ['3310', '3311', '3312', '3314', '3315', '3319', '332', '333', '334', '335',
                    '336', '340', '341', '345', '3481', '3483', '7803', '7843'],
    'rheumatoid': ['446', '7010', '7100', '7101', '7102', '7103', '7104', '7108', '7109',
                   '7112', '714', '7193', '720', '725'],
    'peptic_ulcer': ['5317', '5319', '5327', '5329', '5337', '5339', '5347', '5349'],
    'hiv': ['042', '043', '044'],
    'lymphoma': ['200', '201', '202', '2030', '2386'],
    'valvular': ['0932', '394', '395', '396', '397', '424', '7463', '7464', '7465', '7466', 'V422', 'V433'],
    'weight_loss': ['260', '261', '262', '263', '7832', '7994'],
}

COMORBIDITY_COLS = list(ELIX_ICD10.keys())

cohort_hadm = set(df['hadm_id'])
diag_cohort = diagnoses[diagnoses['hadm_id'].isin(cohort_hadm)].copy()
diag_10 = diag_cohort[diag_cohort['icd_version'] == 10]
diag_9 = diag_cohort[diag_cohort['icd_version'] == 9]

for condition in COMORBIDITY_COLS:
    matched_hadm = set()
    if condition in ELIX_ICD10:
        pat10 = '|'.join([f'^{p}' for p in ELIX_ICD10[condition]])
        m10 = diag_10[diag_10['icd_code'].str.match(pat10, na=False)]['hadm_id'].unique()
        matched_hadm.update(m10)
    # Map ICD9 keys (Paper 1 used shortnames like diabetes_uncx)
    icd9_key = condition
    if condition == 'diabetes_uncomplicated':
        icd9_key = 'diabetes_uncomplicated'
    elif condition == 'diabetes_complicated':
        icd9_key = 'diabetes_complicated'
    if icd9_key in ELIX_ICD9:
        pat9 = '|'.join([f'^{p}' for p in ELIX_ICD9[icd9_key]])
        m9 = diag_9[diag_9['icd_code'].str.match(pat9, na=False)]['hadm_id'].unique()
        matched_hadm.update(m9)
    df[condition] = df['hadm_id'].isin(matched_hadm).astype(int)

print(f"  Comorbidities mapped: {len(COMORBIDITY_COLS)}")

# ---- Feature engineering ----
df['sex'] = (df['gender'] == 'F').astype(int)  # 1=female

def simplify_race(r):
    r = str(r).upper()
    if 'WHITE' in r: return 'White'
    elif 'BLACK' in r or 'AFRICAN' in r: return 'Black'
    elif 'HISPANIC' in r or 'LATINO' in r: return 'Hispanic'
    elif 'ASIAN' in r: return 'Asian'
    else: return 'Other'

df['race_simple'] = df['race'].apply(simplify_race)
df['race_white'] = (df['race_simple'] == 'White').astype(int)
df['race_black'] = (df['race_simple'] == 'Black').astype(int)
df['race_hispanic'] = (df['race_simple'] == 'Hispanic').astype(int)
df['race_asian'] = (df['race_simple'] == 'Asian').astype(int)
df['race_other'] = (df['race_simple'] == 'Other').astype(int)

# ---- Select and rename columns to match Paper 3 prompts.py ----
# Paper 3 uses these exact names in CANDIDATE_VARIABLES
period_cols = ['period_2011_2013', 'period_2014_2016', 'period_2017_2019', 'period_2020_2022']

output_cols = (
    ['treatment', 'outcome', 'age_at_admission', 'sex',
     'race_white', 'race_black', 'race_hispanic', 'race_asian', 'race_other',
     'language_english', 'marital_status_married']
    + period_cols
    + COMORBIDITY_COLS
)

# Verify all columns exist
missing = [c for c in output_cols if c not in df.columns]
if missing:
    print(f"  WARNING: Missing columns: {missing}")

cohort = df[output_cols].copy()

# Save
cohort.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"  Shape: {cohort.shape}")
print(f"  Treatment: {cohort['treatment'].sum():,} Medicaid / {(1-cohort['treatment']).sum():,.0f} Private")
print(f"  Mortality: {cohort['outcome'].mean()*100:.2f}%")
print(f"  Columns: {list(cohort.columns)}")
