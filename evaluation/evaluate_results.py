import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

RESULT_ROOT = os.path.join(os.path.dirname(__file__), "result")
GROUND_TRUTH_DIR = os.path.join(RESULT_ROOT, "ground truth")

# 方法文件夹与其预测文件命名约定
METHODS = {
    "qwen3": {
        "dir": os.path.join(RESULT_ROOT, "qwen3"),
        "filename": "{doc_id}_Overall.txt",
        "subdir": "{doc_id}",
    },
    "local llm finetuned": {
        "dir": os.path.join(RESULT_ROOT, "local llm finetuned"),
        "filename": "Embedding_{doc_id}_Overall.txt",
        "subdir": "{doc_id}",
    },
    "local llm unfinetuned": {
        "dir": os.path.join(RESULT_ROOT, "local llm unfinetuned"),
        "filename": "Embedding_{doc_id}_Overall.txt",
        "subdir": "{doc_id}",
    },
    # 特殊：ground truth 作为基准，不从文件读取预测，而是直接使用GT本身
    "ground truth": {
        "dir": GROUND_TRUTH_DIR,
        "filename": "{doc_id}_annotated.json",
        "subdir": "",
    },
}


def read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


def try_parse_json_fragment(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    # 直接尝试整体解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 尝试截取第一个 { 到最后一个 } 之间的子串
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = s[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            # 尝试去掉可能多余的右花括号
            fragment = fragment.rstrip("}")
            try:
                return json.loads(fragment + "}")
            except json.JSONDecodeError:
                return None
    return None


def parse_prediction_file(path: str, prefer_best: bool) -> Optional[Dict[str, Any]]:
    """
    支持以下情况：
    - 文件整体为一个JSON对象
    - 每行一个JSON对象（行间包含空行或截断行），选取“信息量最大”的一条（在 prefer_best=True 时）
    """
    text = read_text(path)
    if text is None:
        return None
    text_stripped = text.strip()
    # 先尝试整体解析
    whole = try_parse_json_fragment(text_stripped)
    if whole is not None and not prefer_best:
        return whole
    # 行级解析
    candidates: List[Dict[str, Any]] = []
    for line in text.splitlines():
        obj = try_parse_json_fragment(line)
        if isinstance(obj, dict):
            candidates.append(obj)
    if not candidates:
        # 如果没有候选，但整体有、则返回整体
        return whole
    if not prefer_best:
        return candidates[0]
    # 否则基于“信息量评分”选择最佳
    best = max(candidates, key=lambda o: count_non_null_leaves(normalize_prediction(o)))
    return best


def extract_structure_keys(obj: Dict[str, Any], src_obj: Dict[str, Any]) -> Set[str]:
    """
    从原始对象 src_obj 中提取存在的结构键。
    """
    struct_keys = set()
    
    # 1. Root Keys
    if "reaction_summary" in obj:
        struct_keys.add("has_reaction_summary")
    if "reaction_type" in src_obj:
        struct_keys.add("has_reaction_type")
    
    # 2. Lists Presence
    if "reactants" in src_obj and isinstance(src_obj["reactants"], list):
        struct_keys.add("has_reactants_list")
        # Check internal keys of list items
        for r in src_obj["reactants"]:
            if isinstance(r, dict):
                if "name" in r:
                    struct_keys.add("reactant_has_name")
                if "role" in r:
                    struct_keys.add("reactant_has_role")
    
    if "products" in src_obj and isinstance(src_obj["products"], list):
        struct_keys.add("has_products_list")
        for p in src_obj["products"]:
            if isinstance(p, dict):
                if "name" in p:
                    struct_keys.add("product_has_name")
    
    if "conditions" in src_obj and isinstance(src_obj["conditions"], list):
        struct_keys.add("has_conditions_list")
        for c in src_obj["conditions"]:
            if isinstance(c, dict):
                if "type" in c:
                    struct_keys.add("condition_has_type")
                if "value" in c:
                    struct_keys.add("condition_has_value")

    # 3. Dicts Presence
    reactor = src_obj.get("reactor", obj.get("reactor"))
    if isinstance(reactor, dict):
        struct_keys.add("has_reactor_dict")
        if "type" in reactor:
            struct_keys.add("reactor_has_type")
        if "inner_diameter" in reactor:
            struct_keys.add("reactor_has_inner_diameter")

    metrics = src_obj.get("metrics", obj.get("metrics"))
    if isinstance(metrics, dict):
        struct_keys.add("has_metrics_dict")
        if "conversion" in metrics:
            struct_keys.add("metrics_has_conversion")
        if "yield" in metrics:
            struct_keys.add("metrics_has_yield")
        if "selectivity" in metrics:
            struct_keys.add("metrics_has_selectivity")
        if "unit" in metrics:
            struct_keys.add("metrics_has_unit")
            
    return struct_keys


def normalize_prediction(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    将预测对象归一化为与GT相同的字段布局：
    """
    # 预测有时包在 reaction_summary 里，有时是混合结构（部分在根下，部分在 summary 里）
    # 策略：将 reaction_summary 的内容（如果存在）合并到根对象中作为一个统一的源
    summary = obj.get("reaction_summary")
    if isinstance(summary, dict):
        # 优先使用 summary 中的字段，补充以根对象中的字段
        src = {**obj, **summary}
    else:
        src = obj

    result: Dict[str, Any] = {
        "reaction_type": _norm_str(src.get("reaction_type")),
        "reactants": [],
        "products": [],
        "conditions": [],
        "reactor": {"type": None, "inner_diameter": None},
        "metrics": {"conversion": None, "yield": None, "selectivity": None, "unit": None},
        "_struct_keys": extract_structure_keys(obj, src)
    }
    
    # reactants: 可能是字符串列表或对象列表
    reactants = src.get("reactants")
    if isinstance(reactants, list):
        for r in reactants:
            if isinstance(r, str):
                result["reactants"].append({"name": _norm_str(r), "role": None})
            elif isinstance(r, dict):
                result["reactants"].append(
                    {"name": _norm_str(r.get("name")), "role": _norm_str(r.get("role"))}
                )
    # products: 可能是字符串或对象
    products = src.get("products")
    if isinstance(products, list):
        for p in products:
            if isinstance(p, str):
                result["products"].append({"name": _norm_str(p)})
            elif isinstance(p, dict):
                result["products"].append({"name": _norm_str(p.get("name"))})
    # conditions: 统一为 {type, value}
    conditions = src.get("conditions")
    if isinstance(conditions, list):
        for c in conditions:
            if isinstance(c, dict):
                ctype = _norm_str(c.get("type"))
                cval = _norm_val(c.get("value"))
                if ctype:
                    result["conditions"].append({"type": ctype, "value": cval})
    # reactor: 预测可能在 reaction_summary 或根级
    reactor_src = src.get("reactor", obj.get("reactor", {}))
    if isinstance(reactor_src, dict):
        result["reactor"]["type"] = _norm_str(reactor_src.get("type"))
        result["reactor"]["inner_diameter"] = _norm_str(reactor_src.get("inner_diameter"))
    # metrics: 预测可能在 reaction_summary 或根级
    metrics_src = src.get("metrics", obj.get("metrics", {}))
    if isinstance(metrics_src, dict):
        result["metrics"]["conversion"] = _norm_val(metrics_src.get("conversion"))
        result["metrics"]["yield"] = _norm_val(metrics_src.get("yield"))
        result["metrics"]["selectivity"] = _norm_val(metrics_src.get("selectivity"))
        result["metrics"]["unit"] = _norm_str(metrics_src.get("unit"))
    return result


def normalize_ground_truth(obj: Dict[str, Any]) -> Dict[str, Any]:
    # 兼容两种结构：顶层字段或包在 reaction_summary 内
    src = obj.get("reaction_summary", obj)
    
    result: Dict[str, Any] = {
        "reaction_type": _norm_str(src.get("reaction_type")),
        "reactants": [],
        "products": [],
        "conditions": [],
        "reactor": {"type": None, "inner_diameter": None},
        "metrics": {"conversion": None, "yield": None, "selectivity": None, "unit": None},
        "_struct_keys": extract_structure_keys(obj, src)
    }
    
    reactants = src.get("reactants")
    if isinstance(reactants, list):
        for r in reactants:
            if isinstance(r, dict):
                result["reactants"].append(
                    {"name": _norm_str(r.get("name")), "role": _norm_str(r.get("role"))}
                )
            elif isinstance(r, str):
                result["reactants"].append({"name": _norm_str(r), "role": None})
    products = src.get("products")
    if isinstance(products, list):
        for p in products:
            if isinstance(p, dict):
                result["products"].append({"name": _norm_str(p.get("name"))})
            elif isinstance(p, str):
                result["products"].append({"name": _norm_str(p)})
    conditions = src.get("conditions")
    if isinstance(conditions, list):
        for c in conditions:
            if isinstance(c, dict):
                ctype = _norm_str(c.get("type"))
                cval = _norm_val(c.get("value"))
                if ctype:
                    result["conditions"].append({"type": ctype, "value": cval})
    reactor = src.get("reactor", obj.get("reactor", {}))
    if isinstance(reactor, dict):
        result["reactor"]["type"] = _norm_str(reactor.get("type"))
        result["reactor"]["inner_diameter"] = _norm_str(reactor.get("inner_diameter"))
    metrics = src.get("metrics", obj.get("metrics", {}))
    if isinstance(metrics, dict):
        result["metrics"]["conversion"] = _norm_val(metrics.get("conversion"))
        result["metrics"]["yield"] = _norm_val(metrics.get("yield"))
        result["metrics"]["selectivity"] = _norm_val(metrics.get("selectivity"))
        result["metrics"]["unit"] = _norm_str(metrics.get("unit"))
    return result


def count_non_null_leaves(norm: Dict[str, Any]) -> int:
    cnt = 0
    if norm.get("reaction_type"):
        cnt += 1
    for r in norm.get("reactants", []):
        if r.get("name"):
            cnt += 1
        if r.get("role"):
            cnt += 1
    for p in norm.get("products", []):
        if p.get("name"):
            cnt += 1
    for c in norm.get("conditions", []):
        if c.get("type") and c.get("value") is not None:
            cnt += 1
    reactor = norm.get("reactor", {})
    if reactor.get("type"):
        cnt += 1
    if reactor.get("inner_diameter"):
        cnt += 1
    metrics = norm.get("metrics", {})
    for k in ["conversion", "yield", "selectivity", "unit"]:
        if metrics.get(k) is not None:
            cnt += 1
    return cnt


def flatten_items(norm: Dict[str, Any]) -> Set[Tuple]:
    """
    转为可比较的“项”集合，用于微平均：
    - 标量：("reaction_type", v)，("reactor.type", v) 等
    - 条件：("condition", type, value)
    - 反应物：("reactant.name", name) 和 ("reactant.role", name, role)
    - 产物：("product.name", name)
    - 结构：("struct", key_name)
    """
    items: Set[Tuple] = set()
    
    # 0. 结构项
    struct_keys = norm.get("_struct_keys", set())
    for k in struct_keys:
        items.add(("struct", k))
        
    # 1. 值项
    if norm.get("reaction_type"):
        items.add(("reaction_type", norm["reaction_type"]))
    reactor = norm.get("reactor", {})
    if reactor.get("type"):
        items.add(("reactor.type", reactor["type"]))
    if reactor.get("inner_diameter"):
        items.add(("reactor.inner_diameter", reactor["inner_diameter"]))
    metrics = norm.get("metrics", {})
    for k in ["conversion", "yield", "selectivity", "unit"]:
        v = metrics.get(k)
        if v is not None:
            items.add((f"metrics.{k}", v))
    for c in norm.get("conditions", []):
        ctype = c.get("type")
        cval = c.get("value")
        if ctype and cval is not None:
            items.add(("condition", ctype, cval))
    for r in norm.get("reactants", []):
        name = r.get("name")
        role = r.get("role")
        if name:
            items.add(("reactant.name", name))
            if role:
                items.add(("reactant.role", name, role))
    for p in norm.get("products", []):
        name = p.get("name")
        if name:
            items.add(("product.name", name))
    return items


def compute_counts(gt_items: Set[Tuple], pred_items: Set[Tuple]) -> Tuple[int, int, int]:
    # 由于引入了模糊匹配，不能简单用集合交集
    # TP: 对于每个 pred_item，如果它能匹配上任意一个尚未被匹配的 gt_item，则算 TP
    
    tp = 0
    # 先把 gt_items 转为列表以便标记是否已使用
    gt_list = list(gt_items)
    gt_matched = [False] * len(gt_list)
    
    # 剩余的 pred_items 即为 FP
    fp_count = 0
    
    for p_item in pred_items:
        # p_item: (key, val) or (key, val1, val2) ...
        matched_idx = -1
        
        for i, g_item in enumerate(gt_list):
            if gt_matched[i]:
                continue
                
            # 键必须完全一致（除了 struct 这种特殊键？这里假设 key 必须一致）
            # item 结构通常是 (key, val, ...)
            if len(p_item) != len(g_item):
                continue
            if p_item[0] != g_item[0]: # Key mismatch
                continue
                
            # 比较剩余的值部分
            is_match = True
            for k in range(1, len(p_item)):
                if not is_value_match(g_item[k], p_item[k]):
                    is_match = False
                    break
            
            if is_match:
                matched_idx = i
                break
        
        if matched_idx != -1:
            tp += 1
            gt_matched[matched_idx] = True
        else:
            fp_count += 1
            
    fn = len(gt_list) - sum(gt_matched)
    return tp, fp_count, fn


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "support": float(tp + fn),
    }


def _norm_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).strip().lower()
    # 统一多个空白为单个空格
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def _extract_number_if_possible(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)  # 统一转为float
    s = str(v).strip()
    # 提取第一个可能的浮点数/整数
    m = re.search(r"-?\d+(\.\d+)?", s)
    if m:
        num_str = m.group(0)
        try:
            return float(num_str)  # 统一转为float
        except ValueError:
            return s.lower()
    return s.lower()

def _norm_val(v: Any) -> Any:
    # 对数值尝试抽取数值，其他统一小写字符串
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)  # 统一转为float
    return _extract_number_if_possible(v)

def is_value_match(gt_val: Any, pred_val: Any) -> bool:
    """
    判断两个值是否匹配（支持数值容差和字符串宽松匹配）
    """
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
        
    # 1. 数值比较 (float)
    if isinstance(gt_val, float) and isinstance(pred_val, float):
        return abs(gt_val - pred_val) < 1e-4
    
    # 2. 字符串比较
    s_gt = str(gt_val).lower().strip()
    s_pred = str(pred_val).lower().strip()
    
    if s_gt == s_pred:
        return True
        
    # 3. 宽松匹配：括号别名与包含关系
    # 如果一个是全名+缩写形式 "Full Name (Abbr)"，尝试拆解
    def get_variants(s: str) -> Set[str]:
        parts = {s}
        # 提取括号内容 "A (B)" -> A, B
        m = re.match(r"^(.*?)\s*\((.*?)\)$", s)
        if m:
            main_part = m.group(1).strip()
            paren_part = m.group(2).strip()
            parts.add(main_part)
            parts.add(paren_part)
        return {p for p in parts if p}
    
    gt_vars = get_variants(s_gt)
    pred_vars = get_variants(s_pred)
    
    # 只要集合有交集就算对
    if gt_vars & pred_vars:
        return True
        
    # 4. 单向包含（仅针对较长字符串，避免误判）
    # 比如 "TFMB" vs "Trifluoromethoxybenzene" -> 如果我们有一个已知缩写表最好
    # 这里简单地：如果一个是另一个的子串且长度差不离谱？不，这太危险。
    # 仅针对特定情况：如果一个是全大写（缩写），出现在另一个中？
    # 暂时只使用上面的括号拆解逻辑，已经能解决 "trifluoromethoxybenzene (TFMB)" vs "TFMB" 的问题
    
    return False

def compute_counts(gt_items: Set[Tuple], pred_items: Set[Tuple]) -> Tuple[int, int, int]:
    # 由于引入了模糊匹配，不能简单用集合交集
    # TP: 对于每个 pred_item，如果它能匹配上任意一个尚未被匹配的 gt_item，则算 TP
    
    tp = 0
    # 先把 gt_items 转为列表以便标记是否已使用
    gt_list = list(gt_items)
    gt_matched = [False] * len(gt_list)
    
    # 剩余的 pred_items 即为 FP
    fp_count = 0
    
    for p_item in pred_items:
        # p_item: (key, val) or (key, val1, val2) ...
        matched_idx = -1
        
        for i, g_item in enumerate(gt_list):
            if gt_matched[i]:
                continue
                
            # 键必须完全一致（除了 struct 这种特殊键？这里假设 key 必须一致）
            # item 结构通常是 (key, val, ...)
            if len(p_item) != len(g_item):
                continue
            if p_item[0] != g_item[0]: # Key mismatch
                continue
                
            # 比较剩余的值部分
            is_match = True
            for k in range(1, len(p_item)):
                if not is_value_match(g_item[k], p_item[k]):
                    is_match = False
                    break
            
            if is_match:
                matched_idx = i
                break
        
        if matched_idx != -1:
            tp += 1
            gt_matched[matched_idx] = True
        else:
            fp_count += 1
            
    fn = len(gt_list) - sum(gt_matched)
    return tp, fp_count, fn
def list_document_ids() -> List[str]:
    ids: List[str] = []
    for fname in os.listdir(GROUND_TRUTH_DIR):
        if fname.endswith("_annotated.json"):
            doc_id = fname.split("_annotated.json")[0]
            ids.append(doc_id)
    ids.sort(key=lambda x: int(x) if x.isdigit() else x)
    return ids


def load_ground_truth(doc_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(GROUND_TRUTH_DIR, f"{doc_id}_annotated.json")
    raw = read_json(path)
    if raw is None:
        return None
    return normalize_ground_truth(raw)


def load_prediction(method: str, doc_id: str) -> Optional[Dict[str, Any]]:
    # ground truth 直接返回GT（用于在汇总中显示对齐后的满分基线）
    if method == "ground truth":
        return load_ground_truth(doc_id)
    spec = METHODS[method]
    subdir = spec["subdir"].format(doc_id=doc_id)
    fname = spec["filename"].format(doc_id=doc_id)
    path = os.path.join(spec["dir"], subdir, fname) if subdir else os.path.join(spec["dir"], fname)
    prefer_best = method in {"local llm finetuned", "local llm unfinetuned"}
    raw = parse_prediction_file(path, prefer_best=prefer_best)
    if raw is None:
        return None
    return normalize_prediction(raw)


def evaluate_method(method: str, doc_ids: List[str]) -> Dict[str, float]:
    total_tp = total_fp = total_fn = 0
    for doc_id in doc_ids:
        gt = load_ground_truth(doc_id)
        if gt is None:
            continue
        pred = load_prediction(method, doc_id)
        if pred is None:
            # 预测缺失等同于全FN
            tp, fp, fn = 0, 0, len(flatten_items(gt))
        else:
            gt_items = flatten_items(gt)
            pred_items = flatten_items(pred)
            tp, fp, fn = compute_counts(gt_items, pred_items)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return compute_metrics(total_tp, total_fp, total_fn)


def evaluate_method_per_doc(method: str, doc_ids: List[str]) -> List[Dict[str, Any]]:
    per_doc: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        gt = load_ground_truth(doc_id)
        if gt is None:
            continue
        pred = load_prediction(method, doc_id)
        if pred is None:
            tp, fp, fn = 0, 0, len(flatten_items(gt))
        else:
            gt_items = flatten_items(gt)
            pred_items = flatten_items(pred)
            tp, fp, fn = compute_counts(gt_items, pred_items)
        m = compute_metrics(tp, fp, fn)
        per_doc.append({"method": method, "doc_id": doc_id, **m})
    return per_doc


def main():
    doc_ids = list_document_ids()
    if not doc_ids:
        print("No ground truth found.")
        return
    rows: List[Dict[str, Any]] = []
    per_doc_rows: List[Dict[str, Any]] = []
    for method in METHODS.keys():
        metrics = evaluate_method(method, doc_ids)
        row = {"method": method, **metrics}
        rows.append(row)
        per_doc_rows.extend(evaluate_method_per_doc(method, doc_ids))
    # 打印简表
    print(f"Documents evaluated: {len(doc_ids)} -> {', '.join(doc_ids)}")
    print("Method, Precision, Recall, F1, Accuracy, TP, FP, FN, Support")
    for r in rows:
        print(
            f"{r['method']}, "
            f"{r['precision']:.4f}, {r['recall']:.4f}, {r['f1']:.4f}, {r['accuracy']:.4f}, "
            f"{int(r['tp'])}, {int(r['fp'])}, {int(r['fn'])}, {int(r['support'])}"
        )
    # 写出CSV
    out_csv = os.path.join(RESULT_ROOT, "extraction_metrics_summary.csv")
    try:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("method,precision,recall,f1,accuracy,tp,fp,fn,support\n")
            for r in rows:
                f.write(
                    f"{r['method']},{r['precision']:.6f},{r['recall']:.6f},{r['f1']:.6f},{r['accuracy']:.6f},"
                    f"{int(r['tp'])},{int(r['fp'])},{int(r['fn'])},{int(r['support'])}\n"
                )
        print(f"Saved summary to: {out_csv}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")
    # 写出逐文档CSV
    per_doc_csv = os.path.join(RESULT_ROOT, "extraction_metrics_per_doc.csv")
    try:
        with open(per_doc_csv, "w", encoding="utf-8") as f:
            f.write("method,doc_id,precision,recall,f1,accuracy,tp,fp,fn,support\n")
            for r in per_doc_rows:
                f.write(
                    f"{r['method']},{r['doc_id']},{r['precision']:.6f},{r['recall']:.6f},{r['f1']:.6f},{r['accuracy']:.6f},"
                    f"{int(r['tp'])},{int(r['fp'])},{int(r['fn'])},{int(r['support'])}\n"
                )
        print(f"Saved per-doc details to: {per_doc_csv}")
    except Exception as e:
        print(f"Failed to write per-doc CSV: {e}")


if __name__ == "__main__":
    main()
