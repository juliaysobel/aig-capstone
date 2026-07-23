"""
Apply fuzzy matching to Donut outputs at various thresholds and evaluate.
"""
import pandas as pd
import evaluate
from pathlib import Path
from server.fuzzy_match import DrugMatcher

RAW_RESULTS_FNAME = "test_predictions_no_fuzzy.csv"
FM_DICT_FPATH = Path(__file__).parent.parent / "server" / "drug_dictionary.csv"
FM_THRESHOLDS = [99, 95, 90, 80, 70, 60]

donut_output = pd.read_csv(RAW_RESULTS_FNAME)
num_samples = donut_output.shape[0]
results_df = pd.DataFrame(donut_output, copy=True)

for threshold in FM_THRESHOLDS:
    print(f"Fuzzy Matching Threshold: {threshold}")
    # apply fuzzy to predictions
    fuzzy_matcher = DrugMatcher(FM_DICT_FPATH, threshold)
    fm_preds = pd.Series()
    for idx, row in donut_output.iterrows():
        print(f"\rProcessing {idx+1} of {num_samples}", end="")
        fm_result = fuzzy_matcher.match(row["prediction"])
        if fm_result["matched_label"]:
            fm_preds.loc[idx] = fm_result["matched_label"]
        else:
            fm_preds.loc[idx] = fm_result["raw_ocr_text"]
    print("")

    # compare fuzzed predictions to reference
    cer_metric = evaluate.load("cer")
    cer_score = cer_metric.compute(predictions=fm_preds, references=donut_output["reference"])
    print(f"Character Error Rate (CER): {cer_score:.4f}")

    # save fuzzed predictions
    results_df["fm_prediction"] = fm_preds
    results_df.to_csv(f"test_predictions_fuzzy_matching_{threshold:.0f}.csv")