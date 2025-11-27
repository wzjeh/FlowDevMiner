import os
import sys
import pandas as pd

# 导入 evaluate_results 中的函数
sys.path.append(os.path.dirname(__file__))
from evaluate_results import (
    load_ground_truth, 
    load_prediction, 
    flatten_items, 
    compute_counts,
    list_document_ids
)

def check_reaction_type_counts(method, doc_ids):
    print(f"\n--- Checking {method} ---")
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    
    for doc_id in doc_ids:
        gt = load_ground_truth(doc_id)
        pred = load_prediction(method, doc_id)
        
        # 仅提取 reaction_type 相关的 items
        gt_items = {x for x in flatten_items(gt) if x[0] == 'reaction_type'}
        if pred:
            pred_items = {x for x in flatten_items(pred) if x[0] == 'reaction_type'}
        else:
            pred_items = set()
            
        # 计算计数 (evaluate_results 中已更新为模糊匹配，但此处 reaction_type 主要是字符串匹配)
        # 为了简单复现逻辑，直接调用 updated compute_counts
        tp, fp, fn = compute_counts(gt_items, pred_items)
        
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        
        # 打印每篇文档的具体情况
        gt_val = list(gt_items)[0][1] if gt_items else "None"
        pred_val = list(pred_items)[0][1] if pred_items else "None"
        match_mark = "✅" if tp > 0 else "❌"
        print(f"Doc {doc_id:<2}: GT='{gt_val}' | Pred='{pred_val}' -> {match_mark}")

    print(f"Total: TP={tp_tot}, FP={fp_tot}, FN={fn_tot}")
    precision = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0
    print(f"Precision = {tp_tot}/{tp_tot+fp_tot} = {precision:.4f}")

def main():
    doc_ids = list_document_ids()
    print(f"Total Docs: {len(doc_ids)}")
    
    # 检查 Qwen3
    check_reaction_type_counts("qwen3", doc_ids)
    # 检查 Local Finetuned
    check_reaction_type_counts("local llm finetuned", doc_ids)

if __name__ == "__main__":
    main()








