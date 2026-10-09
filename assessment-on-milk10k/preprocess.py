import pandas as pd
import numpy as np

# ----------------------------
# Inputs
# ----------------------------
train_path = "../milk10k/supplements/training_input.csv"   # your MILK10k paired metadata
meta_path  = "../milk10k/metadata.csv"         # contains diagnosis_1 (Benign/Malignant/Indeterminate)

SEED = 100 #42
N_LESIONS_TOTAL = 500
N_PER_DIAG = 250  # 250 malignant + 250 benign

# ----------------------------
# Load
# ----------------------------
df_train = pd.read_csv(train_path)
df_meta  = pd.read_csv(meta_path)

# Merge diagnosis_1 into training table
df = df_train.merge(df_meta[["isic_id", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4"]], on="isic_id", how="left")

# Keep only lesions that are true pairs (clinical + dermoscopic)
paired_counts = df.groupby("lesion_id")["image_type"].nunique()
paired_lesions = paired_counts[paired_counts == 2].index
df_paired = df[df["lesion_id"].isin(paired_lesions)].copy()

# Lesion-level table (one row per lesion)
lesion_tbl = (
    df_paired.groupby("lesion_id", as_index=False)
    .agg(
        diagnosis_1=("diagnosis_1", "first"),
        skin_tone_class=("skin_tone_class", "first"),
    )
)

# Use only Benign/Malignant
lesion_use = lesion_tbl[lesion_tbl["diagnosis_1"].isin(["Benign", "Malignant"])].copy()

# Recode tone 0 into tone 1 (so 0 is counted as 1)
lesion_use["skin_tone_class"] = lesion_use["skin_tone_class"].replace({0: 1})

# Exclude tone 0 (almost nonexistent in this dataset), and use tones 1–5
tones = sorted([t for t in lesion_use["skin_tone_class"].dropna().unique() if t != 0])

# Availability per stratum (diagnosis x tone)
avail = (
    lesion_use.groupby(["diagnosis_1", "skin_tone_class"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=tones, fill_value=0)
)

# Target totals
diag_targets = {"Benign": N_PER_DIAG, "Malignant": N_PER_DIAG}
tone_target_total = {t: N_LESIONS_TOTAL // len(tones) for t in tones}  #

def allocate_stratified(avail_df, tones, diag_targets, tone_target_total, seed=SEED):
    """
    Allocate lesion counts per (diagnosis, tone) to meet:
      - exact diagnosis totals 
      - tone totals as close to uniform as possible, subject to availability
    """
    diags = list(diag_targets.keys())
    alloc = pd.DataFrame(0, index=diags, columns=tones, dtype=int)

    def remaining_capacity(d, t):
        return int(avail_df.loc[d, t]) - int(alloc.loc[d, t])

    def diag_total(d): return int(alloc.loc[d].sum())
    def tone_total(t): return int(alloc[t].sum())

    # Initial equal split per diagnosis across tones
    for d in diags:
        base = diag_targets[d] // len(tones)
        rem  = diag_targets[d] % len(tones)
        for i, t in enumerate(tones):
            want = base + (1 if i < rem else 0)
            alloc.loc[d, t] = min(want, int(avail_df.loc[d, t]))

    # Fill diagnosis deficits, preferring tones that are under target overall
    for d in diags:
        deficit = diag_targets[d] - diag_total(d)
        while deficit > 0:
            candidates = [t for t in tones if remaining_capacity(d, t) > 0]
            if not candidates:
                break
            # Score by how under-target the tone is overall, then by capacity
            best = None
            for t in candidates:
                under = tone_target_total[t] - tone_total(t)
                cap = remaining_capacity(d, t)
                score = (under, cap)
                if best is None or score > best[0]:
                    best = (score, t)
            t = best[1]
            alloc.loc[d, t] += 1
            deficit -= 1

    # Optional swapping to reduce tone imbalance (when feasible)
    max_iters = 200000
    for _ in range(max_iters):
        dev = {t: tone_total(t) - tone_target_total[t] for t in tones}
        over = max(tones, key=lambda t: dev[t])
        under = min(tones, key=lambda t: dev[t])
        if dev[over] <= 0 or dev[under] >= 0:
            break
        moved = False
        for d in diags:
            if alloc.loc[d, over] > 0 and remaining_capacity(d, under) > 0:
                alloc.loc[d, over] -= 1
                alloc.loc[d, under] += 1
                moved = True
                break
        if not moved:
            break

    return alloc

alloc = allocate_stratified(avail, tones, diag_targets, tone_target_total, seed=SEED)

# Sample lesions per stratum according to allocation
rng = np.random.default_rng(SEED)
selected_lesions = []

lesion_use2 = lesion_use[lesion_use["skin_tone_class"].isin(tones)].copy()

for d in ["Benign", "Malignant"]:
    for t in tones:
        k = int(alloc.loc[d, t])
        pool = lesion_use2[
            (lesion_use2["diagnosis_1"] == d) &
            (lesion_use2["skin_tone_class"] == t)
        ]["lesion_id"].values
        if k > len(pool):
            raise ValueError(f"Not enough lesions for {d}, tone {t}: need {k}, have {len(pool)}")
        chosen = rng.choice(pool, size=k, replace=False)
        selected_lesions.extend(chosen.tolist())

selected_lesions = list(set(selected_lesions))
assert len(selected_lesions) == 500

# Build final outputs at image-row level (2 rows per lesion)
df_full_paired = df_paired.copy()

# In the output, replace skin tone 0 with 1
df_full_paired["skin_tone_class"] = df_full_paired["skin_tone_class"].replace({0: 1})

df_500  = df_full_paired[df_full_paired["lesion_id"].isin(selected_lesions)].copy()
df_rest = df_full_paired[~df_full_paired["lesion_id"].isin(selected_lesions)].copy()

# Sanity checks
assert df_500.groupby("lesion_id").size().eq(2).all()
#assert df_500.drop_duplicates("lesion_id")["diagnosis_1"].value_counts().to_dict() == {"Malignant": 250, "Benign": 250}

# Save
df_500.to_csv("milk10k_500.csv", index=False)
df_rest.to_csv("milk10k_Rest.csv", index=False)

print("Saved milk10k_500.csv and milk10k_Rest.csv")
print("Skin tone distribution (lesion-level):")
print(df_500.drop_duplicates("lesion_id")["skin_tone_class"].value_counts().sort_index())
