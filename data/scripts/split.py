"""
split.py — Filter labels_raw.csv, re-split 70/15/15 stratified by source,
copy images into train/val/test folders, write per-split CSVs,
and append the "After Cleaning" section to dataset_report.md.

Run after combine.py.
"""

import shutil
import csv
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent.parent
COMBINED = BASE / "combined"
ALL_IMAGES = COMBINED / "all_images"
RAW_CSV = COMBINED / "labels_raw.csv"
CLEAN_CSV = COMBINED / "labels_clean.csv"
DROPPED_CSV = COMBINED / "dropped_labels.csv"
REPORT = COMBINED / "dataset_report.md"

SPLIT_DIRS = {
    "train": COMBINED / "train",
    "val":   COMBINED / "val",
    "test":  COMBINED / "test",
}

SPLIT_CSVS = {
    "train": COMBINED / "train_labels.csv",
    "val":   COMBINED / "val_labels.csv",
    "test":  COMBINED / "test_labels.csv",
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


def main():
    if not RAW_CSV.exists():
        print(f"ERROR: {RAW_CSV} not found. Run combine.py first.")
        return

    for d in SPLIT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV, dtype=str)
    print(f"  Total rows loaded: {len(df)}")

    # ── 1. Separate dropped entries ───────────────────────────────────
    mask_drop = df["quality_flag"].isin(["illegible", "needs_review"])
    df_dropped = df[mask_drop].copy()
    df_dropped["drop_reason"] = df_dropped["quality_flag"]
    df_clean = df[~mask_drop].copy()

    print(f"\nDropped (illegible + needs_review): {len(df_dropped)}")
    print(f"  illegible:    {(df_dropped['drop_reason'] == 'illegible').sum()}")
    print(f"  needs_review: {(df_dropped['drop_reason'] == 'needs_review').sum()}")
    print(f"Remaining (ok): {len(df_clean)}")

    # Write dropped CSV
    dropped_cols = list(df.columns) + ["drop_reason"]
    df_dropped[dropped_cols].to_csv(DROPPED_CSV, index=False, encoding="utf-8")
    print(f"dropped_labels.csv written: {len(df_dropped)} rows")

    # Write full clean CSV (unsplit reference)
    df_clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"labels_clean.csv written: {len(df_clean)} rows")

    # ── 2. Stratified re-split ────────────────────────────────────────
    # First pass: split into train (70%) and temp (30%), stratified by source
    df_train, df_temp = train_test_split(
        df_clean,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df_clean["source"],
        random_state=RANDOM_SEED,
    )

    # Second pass: split temp evenly into val (15%) and test (15%)
    val_fraction = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # 0.5
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_fraction),
        stratify=df_temp["source"],
        random_state=RANDOM_SEED,
    )

    splits = {"train": df_train, "val": df_val, "test": df_test}

    print(f"\nRe-split results (stratified by source):")
    for name, sdf in splits.items():
        counts = sdf["source"].value_counts().to_dict()
        print(f"  {name}: {len(sdf)} total | {counts}")

    # ── 3. Copy images and write per-split CSVs ───────────────────────
    for split_name, sdf in splits.items():
        dest_dir = SPLIT_DIRS[split_name]
        print(f"\nCopying {len(sdf)} images to {split_name}/...")
        missing = []
        for filename in sdf["filename"]:
            src = ALL_IMAGES / filename
            dst = dest_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"  WARNING: {filename} not found in all_images/")
                missing.append(filename)
        if missing:
            print(f"  {len(missing)} files missing from {split_name}")

        sdf.to_csv(SPLIT_CSVS[split_name], index=False, encoding="utf-8")
        print(f"  {SPLIT_CSVS[split_name].name} written: {len(sdf)} rows")

    # ── 4. Append "After Cleaning" section to dataset_report.md ──────
    lines = [
        "\n---\n\n",
        "## After Cleaning\n\n",
        "### What Was Dropped\n\n",
        f"| Reason | Count |\n|---|---|\n",
    ]
    drop_counts = df_dropped["drop_reason"].value_counts()
    for reason, count in drop_counts.items():
        lines.append(f"| {reason} | {count} |\n")
    lines.append(f"| **Total dropped** | **{len(df_dropped)}** |\n\n")
    lines.append(f"Dropped entries saved to `dropped_labels.csv` for reference.\n\n")

    lines.append("### Final Dataset After Filtering\n\n")
    lines.append(f"- **Total usable images (ok)**: {len(df_clean)}\n")
    lines.append(f"- **Unique labels**: {df_clean['label'].nunique()}\n")
    src_counts = df_clean["source"].value_counts().to_dict()
    for src, cnt in src_counts.items():
        lines.append(f"- **{src} images kept**: {cnt}\n")
    dosage_count = (df_clean["has_dosage_flag"] == "True").sum()
    lines.append(f"- **has_dosage_flag = True** (kept, for later review): {dosage_count}\n\n")

    lines.append("### New Split Distribution (70 / 15 / 15)\n\n")
    lines.append(f"| Split | Total | ds1 | ds2 |\n|---|---|---|---|\n")
    for split_name, sdf in splits.items():
        counts = sdf["source"].value_counts().to_dict()
        ds1_n = counts.get("ds1", 0)
        ds2_n = counts.get("ds2", 0)
        lines.append(f"| {split_name} | {len(sdf)} | {ds1_n} | {ds2_n} |\n")
    lines.append("\n")

    lines.append("### Generic Name Coverage (Clean Dataset)\n\n")
    gn_counts = df_clean["generic_name_source"].value_counts().to_dict()
    lines.append(f"| generic_name_source | Count |\n|---|---|\n")
    for src, cnt in gn_counts.items():
        lines.append(f"| {src} | {cnt} |\n")
    lines.append("\n")

    with open(REPORT, "a", encoding="utf-8") as f:
        f.writelines(lines)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    for split_name, sdf in splits.items():
        print(f"  {split_name}: {len(sdf)} images")
    print(f"\nDataset report updated: {REPORT}")


if __name__ == "__main__":
    main()
