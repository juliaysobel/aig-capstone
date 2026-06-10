"""
reinstate.py — Incrementally re-include needs_review entries into the split dataset.
Reads dropped_labels.csv, takes the needs_review rows (labels that are readable
letter strings but were flagged for not matching DS1's medicine set), assigns them
70/15/15, copies images from all_images/, and updates all CSVs and dataset_report.md.

The illegible rows (label contains '?') stay in dropped_labels.csv and are never moved.
Existing split images and CSV rows are untouched — only new entries are added.

Run from the project root after combine.py and split.py have already run.
"""

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent.parent
COMBINED = BASE / "combined"
ALL_IMAGES = COMBINED / "all_images"
DROPPED_CSV = COMBINED / "dropped_labels.csv"
CLEAN_CSV = COMBINED / "labels_clean.csv"
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

VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


def main():
    for path in (DROPPED_CSV, CLEAN_CSV, REPORT):
        if not path.exists():
            print(f"ERROR: {path} not found. Run combine.py and split.py first.")
            return

    print(f"Loading {DROPPED_CSV}...")
    df_dropped = pd.read_csv(DROPPED_CSV, dtype=str)
    print(f"  Total dropped rows: {len(df_dropped)}")

    df_reinstate = df_dropped[df_dropped["quality_flag"] == "needs_review"].copy()
    df_keep_dropped = df_dropped[df_dropped["quality_flag"] != "needs_review"].copy()

    print(f"\nTo reinstate (needs_review): {len(df_reinstate)}")
    print(f"To keep dropped (illegible):  {len(df_keep_dropped)}")

    if len(df_reinstate) == 0:
        print("Nothing to reinstate.")
        return

    # Drop the drop_reason column — not present in split CSVs or labels_clean.csv
    reinstate_cols = [c for c in df_reinstate.columns if c != "drop_reason"]
    df_reinstate = df_reinstate[reinstate_cols]

    # ── 1. Split 70 / 15 / 15 ────────────────────────────────────────
    df_train, df_temp = train_test_split(
        df_reinstate,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
    )
    val_fraction = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_fraction),
        random_state=RANDOM_SEED,
    )

    new_splits = {"train": df_train, "val": df_val, "test": df_test}

    print(f"\nNew entries per split:")
    for name, sdf in new_splits.items():
        print(f"  {name}: {len(sdf)}")

    # ── 2. Copy images and append to per-split CSVs ───────────────────
    for split_name, sdf in new_splits.items():
        dest_dir = SPLIT_DIRS[split_name]
        dest_dir.mkdir(parents=True, exist_ok=True)

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

        existing = pd.read_csv(SPLIT_CSVS[split_name], dtype=str)
        already_present = set(existing["filename"])
        sdf_new = sdf[~sdf["filename"].isin(already_present)]
        updated = pd.concat([existing, sdf_new], ignore_index=True)
        updated.to_csv(SPLIT_CSVS[split_name], index=False, encoding="utf-8")
        print(f"  {SPLIT_CSVS[split_name].name}: {len(existing)} -> {len(updated)} rows ({len(sdf_new)} added)")

    # ── 3. Append to labels_clean.csv ────────────────────────────────
    df_clean = pd.read_csv(CLEAN_CSV, dtype=str)
    already_in_clean = set(df_clean["filename"])
    df_reinstate_new = df_reinstate[~df_reinstate["filename"].isin(already_in_clean)]
    df_clean_updated = pd.concat([df_clean, df_reinstate_new], ignore_index=True)
    df_clean_updated.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
    print(f"\nlabels_clean.csv: {len(df_clean)} -> {len(df_clean_updated)} rows ({len(df_reinstate_new)} added)")

    # ── 4. Overwrite dropped_labels.csv (illegible only) ─────────────
    df_keep_dropped.to_csv(DROPPED_CSV, index=False, encoding="utf-8")
    print(f"dropped_labels.csv: updated to {len(df_keep_dropped)} rows")

    # ── 5. Replace the After Cleaning section in dataset_report.md ───
    with open(REPORT, "r", encoding="utf-8") as f:
        report_text = f.read()

    marker = "\n---\n\n## After Cleaning"
    before = report_text.split(marker)[0] if marker in report_text else report_text

    total_clean = len(df_clean_updated)
    src_counts = df_clean_updated["source"].value_counts().to_dict()
    dosage_count = (df_clean_updated["has_dosage_flag"] == "True").sum()
    gn_counts = df_clean_updated["generic_name_source"].value_counts().to_dict()
    keep_counts = df_keep_dropped["drop_reason"].value_counts() if len(df_keep_dropped) > 0 else {}

    final_splits = {
        name: pd.read_csv(SPLIT_CSVS[name], dtype=str)
        for name in ("train", "val", "test")
    }

    lines = [
        "\n---\n\n",
        "## After Cleaning\n\n",
        "### What Was Dropped\n\n",
        "| Reason | Count |\n|---|---|\n",
    ]
    for reason, count in keep_counts.items():
        lines.append(f"| {reason} | {count} |\n")
    lines.append(f"| **Total dropped** | **{len(df_keep_dropped)}** |\n\n")
    lines.append("Dropped entries saved to `dropped_labels.csv` for reference.\n\n")

    lines.append("### Final Dataset After Filtering\n\n")
    lines.append(f"- **Total usable images**: {total_clean}\n")
    lines.append(f"- **Unique labels**: {df_clean_updated['label'].nunique()}\n")
    for src, cnt in src_counts.items():
        lines.append(f"- **{src} images kept**: {cnt}\n")
    lines.append(f"- **has_dosage_flag = True** (kept, for later review): {dosage_count}\n\n")

    lines.append("### Split Distribution (70 / 15 / 15)\n\n")
    lines.append("| Split | Total | ds1 | ds2 |\n|---|---|---|---|\n")
    for split_name, sdf in final_splits.items():
        counts = sdf["source"].value_counts().to_dict()
        ds1_n = counts.get("ds1", 0)
        ds2_n = counts.get("ds2", 0)
        lines.append(f"| {split_name} | {len(sdf)} | {ds1_n} | {ds2_n} |\n")
    lines.append("\n")

    lines.append("### Generic Name Coverage (Clean Dataset)\n\n")
    lines.append("| generic_name_source | Count |\n|---|---|\n")
    for src, cnt in gn_counts.items():
        lines.append(f"| {src} | {cnt} |\n")
    lines.append("\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(before)
        f.writelines(lines)

    print(f"Dataset report updated: {REPORT}")

    print("\n" + "=" * 60)
    print("REINSTATE COMPLETE")
    print("=" * 60)
    for split_name, sdf in final_splits.items():
        print(f"  {split_name}: {len(sdf)} images")
    print(f"  Total clean: {total_clean}")
    print(f"  Still dropped: {len(df_keep_dropped)}")


if __name__ == "__main__":
    main()
