from __future__ import annotations

import json
from typing import Dict, Any, Tuple


def _safe_load_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        s, e = txt.find('{'), txt.rfind('}')
        if s != -1 and e != -1 and e > s:
            txt = txt[s:e+1]
        return json.loads(txt)
    except Exception:
        return None


def _compare_dicts(gt: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for k, v in gt.items():
        pv = pred.get(k, None)
        if isinstance(v, dict) and isinstance(pv, dict):
            tpi, fpi, fni = _compare_dicts(v, pv)
            tp += tpi; fp += fpi; fn += fni
        else:
            if pv is None:
                fn += 1
            elif str(pv) == str(v):
                tp += 1
            else:
                fp += 1
    for k in pred.keys():
        if k not in gt:
            fp += 1
    return tp, fp, fn


def calculate_metrics(ground_truth_path: str, predicted_path: str) -> Dict[str, float]:
    gt = _safe_load_json(ground_truth_path) or {}
    pred = _safe_load_json(predicted_path) or {}
    tp, fp, fn = _compare_dicts(gt, pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


