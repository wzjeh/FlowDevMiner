import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Any

# 强制添加当前目录到 sys.path，确保能导入同目录模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from evaluate_results import (
        METHODS, 
        list_document_ids, 
        load_ground_truth, 
        load_prediction, 
        flatten_items, 
        compute_counts,
        compute_metrics
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

print("Successfully imported evaluate_results", flush=True)

# 定义字段所属的能力类别
CATEGORY_MAP = {
    "struct": "1. Instruction Following (JSON Skeleton)",
    "reaction_type": "2. Reaction Type",
    "reactant.name": "3. Reactants",
    "reactant.role": "3. Reactants",
    "product.name": "4. Products",
    "condition": "5. Conditions",
    "reactor.type": "6. Reactor",
    "reactor.inner_diameter": "6. Reactor",
    "metrics.conversion": "7. Metrics",
    "metrics.yield": "7. Metrics",
    "metrics.selectivity": "7. Metrics",
    "metrics.unit": "7. Metrics"
}

def categorize_item(item: Tuple) -> str:
    """根据 tuple 的第一个元素（key）返回其所属的类别"""
    key = item[0]
    return CATEGORY_MAP.get(key, "8. Others")

def analyze_method_detailed(method: str, doc_ids: List[str]) -> pd.DataFrame:
    """
    计算某方法在不同类别下的 TP/FP/FN 统计
    """
    # 存储累积计数：counts[category] = {'tp': 0, 'fp': 0, 'fn': 0}
    counts = {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in set(CATEGORY_MAP.values())}
    
    for doc_id in doc_ids:
        gt = load_ground_truth(doc_id)
        if gt is None:
            continue
        pred = load_prediction(method, doc_id)
        
        if pred is None:
            # 全 FN
            gt_items = flatten_items(gt)
            for item in gt_items:
                cat = categorize_item(item)
                counts[cat]['fn'] += 1
        else:
            gt_items = flatten_items(gt)
            pred_items = flatten_items(pred)
            
            # 按类别拆分集合
            gt_by_cat = {cat: set() for cat in counts}
            pred_by_cat = {cat: set() for cat in counts}
            
            for item in gt_items:
                gt_by_cat[categorize_item(item)].add(item)
            for item in pred_items:
                pred_by_cat[categorize_item(item)].add(item)
                
            for cat in counts:
                tp, fp, fn = compute_counts(gt_by_cat[cat], pred_by_cat[cat])
                counts[cat]['tp'] += tp
                counts[cat]['fp'] += fp
                counts[cat]['fn'] += fn

    # 转换为 DataFrame
    data = []
    for cat, metrics in counts.items():
        m = compute_metrics(metrics['tp'], metrics['fp'], metrics['fn'])
        data.append({
            "Method": method,
            "Category": cat,
            "Precision": m['precision'],
            "Recall": m['recall'],
            "F1": m['f1'],
            "Support": metrics['tp'] + metrics['fn']
        })
    
    return pd.DataFrame(data)

def generate_diagnosis_report(df: pd.DataFrame):
    """
    生成三方对比报告: Qwen3 vs Local Finetuned vs Local Unfinetuned
    """
    print(f"\n{'='*30} 深度诊断报告: 三方横向对比 {'='*30}")
    
    # 获取各方法的数据
    qwen_df = df[df['Method'] == "qwen3"].set_index('Category')
    ft_df = df[df['Method'] == "local llm finetuned"].set_index('Category')
    unft_df = df[df['Method'] == "local llm unfinetuned"].set_index('Category')
    
    cats = sorted(list(set(qwen_df.index) | set(ft_df.index) | set(unft_df.index)))
    
    print(f"\n{'Category':<40} | {'Qwen3':<8} | {'Local FT':<8} | {'Local UnFT':<10} | {'FT vs UnFT':<10} | {'FT vs Qwen':<10}")
    print("-" * 110)
    
    improvements = []
    regressions = []
    
    for cat in cats:
        q_f1 = qwen_df.loc[cat, 'F1'] if cat in qwen_df.index else 0.0
        ft_f1 = ft_df.loc[cat, 'F1'] if cat in ft_df.index else 0.0
        unft_f1 = unft_df.loc[cat, 'F1'] if cat in unft_df.index else 0.0
        
        diff_ft_unft = ft_f1 - unft_f1
        diff_ft_qwen = ft_f1 - q_f1
        
        print(f"{cat:<40} | {q_f1:.4f}   | {ft_f1:.4f}   | {unft_f1:.4f}     | {diff_ft_unft:+.2f}       | {diff_ft_qwen:+.2f}")
        
        if diff_ft_unft > 0.05:
            improvements.append(f"- {cat}: 微调提升了 {diff_ft_unft*100:.1f}%")
        elif diff_ft_unft < -0.05:
            regressions.append(f"- {cat}: 微调导致倒退了 {abs(diff_ft_unft)*100:.1f}%")

    print("\n>>> 微调效果分析 (Impact of Finetuning):")
    if improvements:
        print("✅ 显著提升的领域:")
        for i in improvements:
            print(i)
    else:
        print("⚠️ 微调未带来显著提升。")
        
    if regressions:
        print("\n❌ 出现倒退的领域 (需检查过拟合/遗忘):")
        for r in regressions:
            print(r)
            
    print("\n>>> 距离商用模型差距 (Gap to Qwen3):")
    gaps = []
    for cat in cats:
        q_f1 = qwen_df.loc[cat, 'F1'] if cat in qwen_df.index else 0.0
        ft_f1 = ft_df.loc[cat, 'F1'] if cat in ft_df.index else 0.0
        if q_f1 > ft_f1 + 0.1:
            gaps.append(f"- {cat}: 落后 {abs(ft_f1 - q_f1)*100:.1f}%")
    
    if not gaps:
        print("🎉 Local Finetuned 模型已基本追平 Qwen3！")
    else:
        print(f"在以下 {len(gaps)} 个领域仍有较大差距:")
        for g in gaps:
            print(g)

def main():
    print("Start analyzing...", flush=True)
    doc_ids = list_document_ids()
    if not doc_ids:
        print("No ground truth found.")
        return
        
    print(f"Analyzing {len(doc_ids)} documents: {doc_ids}", flush=True)
    
    all_results = []
    # 筛选要对比的方法
    methods_to_run = [m for m in METHODS.keys() if m != 'ground truth']
    
    for method in methods_to_run:
        print(f"Processing {method}...", flush=True)
        df = analyze_method_detailed(method, doc_ids)
        all_results.append(df)
    
    final_df = pd.concat(all_results, ignore_index=True)
    
    # 1. 保存详细 CSV
    out_csv = os.path.join(os.path.dirname(__file__), "result", "detailed_metrics_by_category.csv")
    final_df.sort_values(by=['Category', 'F1'], ascending=[True, False]).to_csv(out_csv, index=False, float_format='%.4f')
    print(f"\nDetailed metrics saved to: {out_csv}")
    
    # 2. 打印对比表格（Pivot Table）
    pivot = final_df.pivot(index='Category', columns='Method', values='F1')
    print("\n" + "="*60)
    print("F1 Score Comparison Matrix")
    print("="*60)
    print(pivot.round(4))
    
    # 3. 自动诊断 (三方对比)
    generate_diagnosis_report(final_df)

if __name__ == "__main__":
    main()
