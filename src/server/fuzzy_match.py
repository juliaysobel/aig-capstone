import csv
from pathlib import Path
from typing import Optional, Union

from rapidfuzz import fuzz, process, utils

# rapidfuzz ratio score (0-100) below which a candidate is treated as no match.
# Uses fuzz.ratio (plain Levenshtein similarity) rather than WRatio: WRatio's
# partial-match component scores short substrings of a long query too highly
# (e.g. "Actrafid" vs "Act" scores 90 under WRatio but only 55 under ratio),
# which produces confident-looking wrong matches for short brand names.
# Matching runs through utils.default_process (lowercase + whitespace/punctuation
# normalization) so case/spacing differences like "D-cap" vs "D-Cap" aren't
# scored as if they were typos.
DEFAULT_THRESHOLD = 80.0


class DrugMatcher:
    def __init__(self, dictionary_path: Union[str, Path], threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._rows: dict[str, dict] = {}
        self._labels: list[str] = []

        with open(Path(dictionary_path), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = row["label"]
                if label and label not in self._rows:
                    self._rows[label] = row
                    self._labels.append(label)

    def match(self, raw_ocr_text: str) -> dict:
        result = {
            "raw_ocr_text": raw_ocr_text,
            "matched_label": None,
            "generic_name": None,
            "match_confidence": None,
            "source": None,
            "drug_class": None,
            "common_form": None,
        }

        query = (raw_ocr_text or "").strip()
        if not query or not self._labels:
            return result

        best = process.extractOne(
            query, self._labels, scorer=fuzz.ratio, processor=utils.default_process
        )
        if best is None:
            return result

        matched_label, score, _ = best
        if score < self.threshold:
            return result

        row = self._rows[matched_label]
        result.update({
            "matched_label": matched_label,
            "generic_name": row["generic_name"] or None,
            "match_confidence": round(score / 100, 2),
            "source": row["source"] or None,
            "drug_class": row["drug_class"] or None,
            "common_form": row["common_form"] or None,
        })
        return result
