# Dataset Report

---

## Before Cleaning

### Dataset 1 — Doctor's Handwritten Prescription BD dataset

- **Path**: `archive/Doctor's Handwritten Prescription BD dataset/`
- **Image format**: PNG
- **Original label columns**: `IMAGE`, `MEDICINE_NAME`, `GENERIC_NAME`
- **Unified label column mapping**: `MEDICINE_NAME` → `label`; `GENERIC_NAME` → `generic_name`

**Train split**: 3120 images, 3120 label rows
  - Sample dimensions (w×h): [(79, 30), (170, 68), (234, 93), (82, 24), (87, 29)]
  - Color modes in sample: ['L', 'RGBA', 'RGB']
**Val split**: 780 images, 780 label rows
  - Sample dimensions (w×h): [(182, 46), (263, 52), (257, 50), (283, 143), (185, 59)]
  - Color modes in sample: ['RGBA', 'RGB']
**Test split**: 780 images, 780 label rows
  - Sample dimensions (w×h): [(212, 82), (208, 103), (133, 58), (227, 84), (222, 71)]
  - Color modes in sample: ['L', 'RGBA', 'RGB']

**DS1 total images**: 4680  
**DS1 unique labels**: 78  
**DS1 quality flags**: {'ok': 4680}  
**DS1 has-dosage entries**: 0  

### Dataset 2 — RxHandBD-ML

- **Path**: `RxHandBD-ML/RxHandBD-ML/`
- **Image format**: JPG
- **Original label columns**: `Images`, `Text`
- **Unified label column mapping**: `Text` → `label`; no `generic_name` in source

**Train split**: 4463 images, 4463 label rows
  - Sample dimensions (w×h): [(512, 512)]
  - Color modes in sample: ['RGB']
**Test split**: 1115 images, 1115 label rows
  - Sample dimensions (w×h): [(512, 512)]
  - Color modes in sample: ['RGB']

**DS2 total images**: 5578  
**DS2 unique labels**: 1769  
**DS2 quality flags**: {'ok': 4408, 'needs_review': 1166, 'illegible': 4}  
**DS2 has-dosage entries**: 200  
**DS2 generic_name auto-matchable** (label found in DS1 map): 435  
**DS2 generic_name unknown** (no DS1 match, will be blank): 5143  

### Combined Totals (Before Cleaning)

| Metric | Count |
|---|---|
| Total images | 10258 |
| quality_flag = ok | 9088 |
| quality_flag = illegible | 4 |
| quality_flag = needs_review | 1166 |
| has_dosage_flag = True | 200 |


---

## After Cleaning

### What Was Dropped

| Reason | Count |
|---|---|
| needs_review | 1166 |
| illegible | 4 |
| **Total dropped** | **1170** |

Dropped entries saved to `dropped_labels.csv` for reference.

### Final Dataset After Filtering

- **Total usable images (ok)**: 9088
- **Unique labels**: 1414
- **ds1 images kept**: 4680
- **ds2 images kept**: 4408
- **has_dosage_flag = True** (kept, for later review): 188

### New Split Distribution (70 / 15 / 15)

| Split | Total | ds1 | ds2 |
|---|---|---|---|
| train | 6361 | 3276 | 3085 |
| val | 1363 | 702 | 661 |
| test | 1364 | 702 | 662 |

### Generic Name Coverage (Clean Dataset)

| generic_name_source | Count |
|---|---|
| source | 4680 |
| unknown | 3973 |
| auto | 435 |

