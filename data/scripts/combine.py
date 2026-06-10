"""
combine.py — Rename images with unique prefixes, copy to combined/all_images/,
and build labels_raw.csv covering all 10,258 images with the unified schema.

Run after audit.py, before split.py.
"""

import re
import csv
import shutil
from pathlib import Path

BASE = Path(__file__).parent.parent
DS1_BASE = BASE / "archive" / "Doctor’s Handwritten Prescription BD dataset"
DS2_BASE = BASE / "RxHandBD-ML" / "RxHandBD-ML"
ALL_IMAGES = BASE / "combined" / "all_images"
OUTPUT_CSV = BASE / "combined" / "labels_raw.csv"

DS1_SPLITS = {
    "train": (DS1_BASE / "Training"   / "training_labels.csv",    DS1_BASE / "Training"   / "training_words",   "ds1_train_"),
    "val":   (DS1_BASE / "Validation" / "validation_labels.csv",  DS1_BASE / "Validation" / "validation_words", "ds1_val_"),
    "test":  (DS1_BASE / "Testing"    / "testing_labels.csv",     DS1_BASE / "Testing"    / "testing_words",    "ds1_test_"),
}

DS2_SPLITS = {
    "train": (DS2_BASE / "Train_Label.csv", DS2_BASE / "Train_Set", "ds2_train_"),
    "test":  (DS2_BASE / "Test_Label.csv",  DS2_BASE / "Test_Set",  "ds2_test_"),
}

UNIFIED_COLUMNS = [
    "filename",
    "original_filename",
    "label",
    "generic_name",
    "generic_name_source",
    "source",
    "original_split",
    "has_dosage_flag",
    "quality_flag",
]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_ds1_map(splits):
    """Returns {medicine_name_lower: generic_name} from all DS1 CSVs."""
    mapping = {}
    for split, (csv_path, _, _) in splits.items():
        for row in read_csv(csv_path):
            name = row["MEDICINE_NAME"].strip()
            generic = row["GENERIC_NAME"].strip()
            mapping[name.lower()] = generic
    return mapping


def compute_quality_flag(label, ds1_medicine_set):
    if "?" in label:
        return "illegible"
    if label == label.lower() and " " not in label and label.lower() not in ds1_medicine_set:
        return "needs_review"
    return "ok"


def has_dosage(label):
    return bool(re.search(r"\d", label))


def process_ds1(splits, ds1_map, rows_out, copied, skipped):
    ds1_medicine_set = set(ds1_map.keys())

    for split_name, (csv_path, img_folder, prefix) in splits.items():
        rows = read_csv(csv_path)
        print(f"  DS1 {split_name}: {len(rows)} rows from {img_folder.name}/")

        for row in rows:
            original_filename = row["IMAGE"].strip()
            label = row["MEDICINE_NAME"].strip()
            generic_name = row["GENERIC_NAME"].strip()

            new_filename = prefix + original_filename
            src = img_folder / original_filename
            dst = ALL_IMAGES / new_filename

            if not src.exists():
                print(f"    WARNING: image not found: {src}")
                skipped.append(new_filename)
                continue

            shutil.copy2(src, dst)
            copied.append(new_filename)

            rows_out.append({
                "filename": new_filename,
                "original_filename": original_filename,
                "label": label,
                "generic_name": generic_name,
                "generic_name_source": "source",
                "source": "ds1",
                "original_split": split_name,
                "has_dosage_flag": has_dosage(label),
                "quality_flag": compute_quality_flag(label, ds1_medicine_set),
            })


def process_ds2(splits, ds1_map, rows_out, copied, skipped):
    ds1_medicine_set = set(ds1_map.keys())

    for split_name, (csv_path, img_folder, prefix) in splits.items():
        rows = read_csv(csv_path)
        print(f"  DS2 {split_name}: {len(rows)} rows from {img_folder.name}/")

        for row in rows:
            original_filename = row["Images"].strip()
            label = row["Text"].strip()

            label_lower = label.lower()
            if label_lower in ds1_map:
                generic_name = ds1_map[label_lower]
                generic_name_source = "auto"
            else:
                generic_name = ""
                generic_name_source = "unknown"

            new_filename = prefix + original_filename
            src = img_folder / original_filename
            dst = ALL_IMAGES / new_filename

            if not src.exists():
                print(f"    WARNING: image not found: {src}")
                skipped.append(new_filename)
                continue

            shutil.copy2(src, dst)
            copied.append(new_filename)

            rows_out.append({
                "filename": new_filename,
                "original_filename": original_filename,
                "label": label,
                "generic_name": generic_name,
                "generic_name_source": generic_name_source,
                "source": "ds2",
                "original_split": split_name,
                "has_dosage_flag": has_dosage(label),
                "quality_flag": compute_quality_flag(label, ds1_medicine_set),
            })


def main():
    ALL_IMAGES.mkdir(parents=True, exist_ok=True)

    print("Building DS1 medicine -> generic map...")
    ds1_map = build_ds1_map(DS1_SPLITS)
    print(f"  {len(ds1_map)} unique medicines in DS1 map\n")

    rows_out = []
    copied = []
    skipped = []

    print("Processing Dataset 1 (PNG)...")
    process_ds1(DS1_SPLITS, ds1_map, rows_out, copied, skipped)

    print("\nProcessing Dataset 2 (JPG)...")
    process_ds2(DS2_SPLITS, ds1_map, rows_out, copied, skipped)

    print(f"\nWriting labels_raw.csv ({len(rows_out)} rows)...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    print("=" * 60)
    print("COMBINE COMPLETE")
    print("=" * 60)
    print(f"Images copied to all_images/: {len(copied)}")
    if skipped:
        print(f"Images skipped (not found): {len(skipped)}")
        for s in skipped:
            print(f"  {s}")
    print(f"labels_raw.csv rows: {len(rows_out)}")
    print(f"Output: {OUTPUT_CSV}")

    # Quick sanity check — count files in all_images/
    actual = len(list(ALL_IMAGES.iterdir()))
    print(f"Files now in all_images/: {actual}")
    if actual != len(copied):
        print(f"WARNING: mismatch — expected {len(copied)}, found {actual}")


if __name__ == "__main__":
    main()
