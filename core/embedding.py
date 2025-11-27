from __future__ import annotations

import os
import re
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _read_paragraphs(txt_path: str) -> List[str]:
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    segs: List[str] = []
    cur: List[str] = []
    for line in lines:
        if line.strip():
            cur.append(line.strip())
        else:
            if cur:
                segs.append(' '.join(cur))
                cur = []
    if cur:
        segs.append(' '.join(cur))
    return segs


def _write_paragraphs(paragraphs: List[str], out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(p + '\n\n')


def clean_chunks(chunks: List[str]) -> List[str]:
    """
    在 Embedding 之前过滤掉无效的文本块。
    规则：
    - 过短块（<10词）
    - 期刊/页眉（Journal of ... (YEAR) VOL:PAGE–PAGE、DOI、版权）
    - 图表/方案题注（Fig/Figure/Scheme/Table N）
    - 板块噪声（Keywords/Graphical Abstract/Supplementary Information/Acknowledgements/Declarations/References）
    - 纯数字（可能为页码）
    """
    cleaned: List[str] = []
    journal_pattern = re.compile(r"Journal of [A-Za-z\s]+ \(\d{4}\) \d+:\d+–\d+", re.IGNORECASE)
    caption_pattern = re.compile(r"^(Fig|Figure|Scheme|Table)\s*\.?\s*\d+", re.IGNORECASE)
    section_pattern = re.compile(
        r"^(Keywords|Graphical Abstract|Supplementary Information|Acknowledgements|Declarations|References)\b",
        re.IGNORECASE,
    )
    doi_pattern = re.compile(r"(^https?://|^doi:|^10\.\d{4,9}/)", re.IGNORECASE)
    copyright_pattern = re.compile(r"©|All rights reserved", re.IGNORECASE)

    for chunk in chunks:
        text = (chunk or "").strip()
        if not text:
            continue
        # 1) 长度阈值（按词数）
        if len(text.split()) < 10:
            continue
        # 2) 纯数字
        if text.isdigit():
            continue
        # 3) 期刊/页眉/DOI/版权
        if journal_pattern.search(text) or doi_pattern.search(text) or copyright_pattern.search(text):
            continue
        # 4) 图表/方案题注
        if caption_pattern.search(text):
            continue
        # 5) 板块噪声
        if section_pattern.search(text):
            continue
        cleaned.append(text)
    return cleaned


def run_embedding_selection(txt_path: str, top_n: int = 10) -> str:
    """读取txt段落，按与参考关键词相似度排序，选Top-N写出到 Embedding_<base>.txt"""
    paragraphs = _read_paragraphs(txt_path)
    if not paragraphs:
        # 空输入则直接复制为Embedding文件
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
        _write_paragraphs([], out_path)
        return out_path

    # 先进行规则清洗
    paragraphs = clean_chunks(paragraphs)
    if not paragraphs:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
        _write_paragraphs([], out_path)
        return out_path

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 多查询向量（条件/装置/结果），覆盖单位和关键结果词
    queries = {
        "conditions": (
            "flow chemistry reaction conditions parameters flow rate residence time RT "
            "temperature °C K pressure bar BPR back pressure concentration mL/h mL/min µL/min uL/min"
        ),
        "equipment": (
            "reactor setup coil tubular microreactor microchannel packed bed packed tubular "
            "ID i.d. inner diameter mm μm"
        ),
        "outcomes": (
            "optimal yield conversion selectivity productivity mg/h g/h percent % product distribution"
        ),
    }

    # 编码段落与查询
    para_vecs = model.encode(paragraphs)
    ref_vecs = {k: model.encode(v) for k, v in queries.items()}

    # 计算三组相似度，取最大值作为基准分
    import numpy as np
    sims_list = []
    for key in ("conditions", "equipment", "outcomes"):
        v = np.asarray(ref_vecs[key]).reshape(1, -1)
        sims_k = cosine_similarity(np.asarray(para_vecs), v).reshape(-1)
        sims_list.append(sims_k)
    sims_stack = np.vstack(sims_list)  # (3, N)
    base_scores = sims_stack.max(axis=0)  # (N,)

    # 规则加权：命中数值/单位/结果词等给予小幅加权（封顶）
    percent_re = re.compile(r"\b\d{1,3}\s?%\b")
    units_re = re.compile(
        r"(?:\bmL\s*/\s*(?:h|min)\b|\bµL\s*/\s*min\b|\buL\s*/\s*min\b|\bmg\s*/\s*h\b|\bbar\b|\bMPa\b|\b°C\b|\bK\b|\b°F\b)",
        re.IGNORECASE,
    )
    outcomes_re = re.compile(r"\b(yield|conversion|selectivity|productivity)\b", re.IGNORECASE)
    rt_re = re.compile(r"\b(residence time|RT)\b", re.IGNORECASE)
    flow_re = re.compile(r"\bflow rate\b", re.IGNORECASE)
    bpr_re = re.compile(r"\bBPR\b", re.IGNORECASE)

    bonuses = np.zeros_like(base_scores)
    unit_boost = 0.0
    try:
        unit_boost = float(os.getenv("FCPD_EMB_UNIT_BOOST", "0.12"))
    except Exception:
        unit_boost = 0.12
    for i, text in enumerate(paragraphs):
        bonus = 0.0
        if percent_re.search(text):
            bonus += 0.12
        if units_re.search(text):
            bonus += unit_boost
        if outcomes_re.search(text):
            bonus += 0.08
        if rt_re.search(text):
            bonus += 0.06
        if flow_re.search(text):
            bonus += 0.06
        if bpr_re.search(text):
            bonus += 0.04
        if bonus > 0.25:
            bonus = 0.25
        bonuses[i] = bonus

    # 段落长度加权：长段落优先级略高（加分权重适度提高）
    lengths = np.array([len(t) for t in paragraphs], dtype=float)
    if lengths.size and lengths.max() > lengths.min():
        lengths_norm = (lengths - lengths.min()) / (lengths.max() - lengths.min())
    else:
        lengths_norm = np.zeros_like(lengths)
    length_bonus = 0.12 * lengths_norm

    # 对短且缺少定量/单位/结果词的段落进行轻微惩罚，避免短句占前
    short_penalty = np.zeros_like(base_scores)
    for i, text in enumerate(paragraphs):
        if len(text) < 160:
            has_signal = bool(percent_re.search(text) or units_re.search(text) or outcomes_re.search(text) or rt_re.search(text) or flow_re.search(text) or bpr_re.search(text))
            if not has_signal:
                short_penalty[i] = 0.06

    final_scores = base_scores + bonuses + length_bonus - short_penalty
    # 选Top-N
    idx_sorted = np.argsort(-final_scores)[: max(top_n, 1)]

    # 可选：合并相邻段落以增加信息量，避免短句丢信息（通过环境变量控制）
    expand = os.getenv('FCPD_EMB_EXPAND', '0') == '1'
    min_chars = int(os.getenv('FCPD_EMB_MIN_CHARS', '300'))
    max_chars = int(os.getenv('FCPD_EMB_MAX_CHARS', '1200'))
    selected_blocks = []
    used = set()
    for i in idx_sorted:
        if i in used:
            continue
        base_text = paragraphs[i]
        combined = base_text
        block_min_idx = i
        if expand and len(base_text) < min_chars:
            # 优先尝试向后合并，再尝试向前合并
            if i + 1 < len(paragraphs) and (i + 1) not in used:
                cand = combined + " " + paragraphs[i + 1]
                if len(cand) <= max_chars:
                    combined = cand
                    used.add(i + 1)
                    block_min_idx = min(block_min_idx, i + 1)
            if len(combined) < min_chars and i - 1 >= 0 and (i - 1) not in used:
                cand = paragraphs[i - 1] + " " + combined
                if len(cand) <= max_chars:
                    combined = cand
                    used.add(i - 1)
                    block_min_idx = min(block_min_idx, i - 1)
        selected_blocks.append((block_min_idx, combined))
        used.add(i)

    # 写出前是否按原文顺序输出（仅影响文件中顺序，不影响候选选择）
    if os.getenv('FCPD_EMB_KEEP_ORDER', '0') == '1':
        selected_blocks.sort(key=lambda x: x[0])
    selected = [text for _, text in selected_blocks]

    base = os.path.splitext(os.path.basename(txt_path))[0]
    out_path = os.path.join(os.path.dirname(txt_path), f"Embedding_{base}.txt")
    _write_paragraphs(selected, out_path)
    return out_path


