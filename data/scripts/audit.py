"""
audit.py — Read-only analysis of both datasets.
Writes the "Before Cleaning" section of combined/dataset_report.md.
Run this first before combine.py or split.py.
"""

import re
import csv
from pathlib import Path
from collections import Counter

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Image property checks will be skipped.")

BASE = Path(__file__).parent.parent
DS1_BASE = BASE / "archive" / "Doctor’s Handwritten Prescription BD dataset"
DS2_BASE = BASE / "RxHandBD-ML" / "RxHandBD-ML"
OUTPUT = BASE / "combined"

DS1_SPLITS = {
    "train": (DS1_BASE / "Training" / "training_labels.csv",   DS1_BASE / "Training"   / "training_words"),
    "val":   (DS1_BASE / "Validation" / "validation_labels.csv", DS1_BASE / "Validation" / "validation_words"),
    "test":  (DS1_BASE / "Testing" / "testing_labels.csv",    DS1_BASE / "Testing"    / "testing_words"),
}

DS2_SPLITS = {
    "train": (DS2_BASE / "Train_Label.csv", DS2_BASE / "Train_Set"),
    "test":  (DS2_BASE / "Test_Label.csv",  DS2_BASE / "Test_Set"),
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_quality_flag(label, ds1_medicine_set):
    if "?" in label:
        return "illegible"
    if label == label.lower() and " " not in label and label.lower() not in ds1_medicine_set:
        return "needs_review"
    return "ok"


def has_dosage(label):
    return bool(re.search(r"\d", label))


def sample_image_props(folder, extensions, sample_size=15):
    if not PIL_AVAILABLE:
        return []
    images = []
    for ext in extensions:
        images.extend(folder.glob(f"*.{ext}"))
    props = []
    for img_path in images[:sample_size]:
        try:
            with Image.open(img_path) as img:
                props.append({"size": img.size, "mode": img.mode})
        except Exception:
            pass
    return props


def format_props_summary(props):
    if not props:
        return "  - Image properties: Pillow not available, skipped\n"
    sizes = [p["size"] for p in props]
    modes = list({p["mode"] for p in props})
    unique_sizes = list({s for s in sizes})
    return (
        f"  - Sample dimensions (w×h): {unique_sizes[:5]}\n"
        f"  - Color modes in sample: {modes}\n"
    )


def main():
    OUTPUT.mkdir(exist_ok=True)

    # Build DS1 medicine → generic map (used for auto-matching DS2 later)
    ds1_medicine_map = {}
    for split, (csv_path, _) in DS1_SPLITS.items():
        for row in read_csv(csv_path):
            name = row["MEDICINE_NAME"].strip()
            generic = row["GENERIC_NAME"].strip()
            ds1_medicine_map[name.lower()] = generic
    ds1_medicine_set = set(ds1_medicine_map.keys())

    lines = [
        "# Dataset Report\n\n",
        "---\n\n",
        "## Before Cleaning\n\n",
    ]

    # ── DS1 ──────────────────────────────────────────────────────────
    lines.append("### Dataset 1 — Doctor's Handwritten Prescription BD dataset\n\n")
    lines.append(f"- **Path**: `archive/Doctor's Handwritten Prescription BD dataset/`\n")
    lines.append("- **Image format**: PNG\n")
    lines.append("- **Original label columns**: `IMAGE`, `MEDICINE_NAME`, `GENERIC_NAME`\n")
    lines.append("- **Unified label column mapping**: `MEDICINE_NAME` → `label`; `GENERIC_NAME` → `generic_name`\n\n")

    ds1_total = 0
    ds1_quality = Counter()
    ds1_dosage = 0
    ds1_vocab = set()

    for split, (csv_path, img_folder) in DS1_SPLITS.items():
        rows = read_csv(csv_path)
        img_count = len(list(img_folder.glob("*.png")))
        lines.append(f"**{split.capitalize()} split**: {img_count} images, {len(rows)} label rows\n")
        ds1_total += img_count
        for row in rows:
            label = row["MEDICINE_NAME"].strip()
            ds1_vocab.add(label)
            ds1_quality[compute_quality_flag(label, ds1_medicine_set)] += 1
            if has_dosage(label):
                ds1_dosage += 1
        props = sample_image_props(img_folder, ["png"])
        lines.append(format_props_summary(props))

    lines.append(f"\n**DS1 total images**: {ds1_total}  \n")
    lines.append(f"**DS1 unique labels**: {len(ds1_vocab)}  \n")
    lines.append(f"**DS1 quality flags**: {dict(ds1_quality)}  \n")
    lines.append(f"**DS1 has-dosage entries**: {ds1_dosage}  \n\n")

    # ── DS2 ──────────────────────────────────────────────────────────
    lines.append("### Dataset 2 — RxHandBD-ML\n\n")
    lines.append("- **Path**: `RxHandBD-ML/RxHandBD-ML/`\n")
    lines.append("- **Image format**: JPG\n")
    lines.append("- **Original label columns**: `Images`, `Text`\n")
    lines.append("- **Unified label column mapping**: `Text` → `label`; no `generic_name` in source\n\n")

    ds2_total = 0
    ds2_quality = Counter()
    ds2_dosage = 0
    ds2_vocab = set()
    ds2_auto = 0
    ds2_unknown = 0

    for split, (csv_path, img_folder) in DS2_SPLITS.items():
        rows = read_csv(csv_path)
        img_count = len(list(img_folder.glob("*.jpg")))
        lines.append(f"**{split.capitalize()} split**: {img_count} images, {len(rows)} label rows\n")
        ds2_total += img_count
        for row in rows:
            label = row["Text"].strip()
            ds2_vocab.add(label)
            ds2_quality[compute_quality_flag(label, ds1_medicine_set)] += 1
            if has_dosage(label):
                ds2_dosage += 1
            if label.lower() in ds1_medicine_set:
                ds2_auto += 1
            else:
                ds2_unknown += 1
        props = sample_image_props(img_folder, ["jpg"])
        lines.append(format_props_summary(props))

    lines.append(f"\n**DS2 total images**: {ds2_total}  \n")
    lines.append(f"**DS2 unique labels**: {len(ds2_vocab)}  \n")
    lines.append(f"**DS2 quality flags**: {dict(ds2_quality)}  \n")
    lines.append(f"**DS2 has-dosage entries**: {ds2_dosage}  \n")
    lines.append(f"**DS2 generic_name auto-matchable** (label found in DS1 map): {ds2_auto}  \n")
    lines.append(f"**DS2 generic_name unknown** (no DS1 match, will be blank): {ds2_unknown}  \n\n")

    # ── Combined totals ───────────────────────────────────────────────
    lines.append("### Combined Totals (Before Cleaning)\n\n")
    grand_total = ds1_total + ds2_total
    combined_quality = ds1_quality + ds2_quality
    lines.append(f"| Metric | Count |\n|---|---|\n")
    lines.append(f"| Total images | {grand_total} |\n")
    lines.append(f"| quality_flag = ok | {combined_quality['ok']} |\n")
    lines.append(f"| quality_flag = illegible | {combined_quality['illegible']} |\n")
    lines.append(f"| quality_flag = needs_review | {combined_quality['needs_review']} |\n")
    lines.append(f"| has_dosage_flag = True | {ds1_dosage + ds2_dosage} |\n\n")

    report_path = OUTPUT / "dataset_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print(f"DS1: {ds1_total} images | {len(ds1_vocab)} unique labels")
    print(f"DS2: {ds2_total} images | {len(ds2_vocab)} unique labels")
    print(f"Grand total: {grand_total} images")
    print(f"\nQuality flags (combined):")
    for flag, count in combined_quality.items():
        print(f"  {flag}: {count}")
    print(f"\nHas-dosage entries: {ds1_dosage + ds2_dosage}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
