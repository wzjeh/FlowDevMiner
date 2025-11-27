from __future__ import annotations

import os
import re
from typing import List

import fitz  # PyMuPDF


def _normalize_ligatures(s: str) -> str:
    return (
        s.replace("ﬁ", "fi").replace("ﬂ", "fl")
         .replace("’", "'").replace("“", '"').replace("”", '"')
    )


def _cleanup_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for ln in lines:
        t = _normalize_ligatures(ln)
        # 连字符断行修复：在清洗阶段主要处理 "-\n" 情况，这里按行内残留连字符做基本修复
        t = re.sub(r"\b-\s+", "", t)
        # 多空白压缩
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            cleaned.append(t)
    return cleaned


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    blocks级抽取 + 轻量清洗：
    - 跳过页眉/页脚（按页面高度前后5%）
    - 去页码纯数字行
    - 处理连字/引号/连字符
    - References之后可选截断（若命中）
    - 块内换行转空格；块与块之间以空行分隔
    """
    doc = fitz.open(pdf_path)
    try:
        out_blocks: List[str] = []
        for page in doc:
            ph = page.rect.height
            top_cut = ph * 0.05
            bot_cut = ph * 0.95
            blocks = page.get_text("blocks") or []
            # 按y坐标排序
            blocks.sort(key=lambda b: (b[1], b[0]))
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4] if len(b) > 4 else ""
                if not text or not text.strip():
                    continue
                # 跳过页眉/页脚区域
                if y0 < top_cut or y1 > bot_cut:
                    # 但避免误删正文：若包含典型正文关键词则保留
                    low = text.lower()
                    if not any(k in low for k in ["introduction", "abstract", "experiment", "method", "result", "discussion", "conclusion", "flow", "reactor"]):
                        continue
                # 行清洗
                raw_lines = text.splitlines()
                # 去纯页码
                raw_lines = [ln for ln in raw_lines if not re.match(r"^\s*\d+\s*$", ln)]
                lines = _cleanup_lines(raw_lines)
                if not lines:
                    continue
                block_text = " ".join(lines)
                out_blocks.append(block_text)

        # 合并并在References处截断（可选，若误杀可去除该段）
        full_text = "\n\n".join(out_blocks)
        # References截断：更宽松，允许可选冒号与空白行
        m = re.search(r"\n\s*references\s*:?.*?(\n|$)", full_text, flags=re.IGNORECASE)
        if m:
            full_text = full_text[:m.start()].rstrip()
        return full_text
    finally:
        doc.close()


def write_text(text: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)


def split_paragraphs(text: str, min_len: int = 60) -> List[str]:
    segments: List[str] = []
    current: List[str] = []
    for line in (text or '').splitlines():
        if line.strip():
            current.append(line.strip())
        else:
            if current:
                seg = ' '.join(current).strip()
                if len(seg) >= min_len:
                    segments.append(seg)
                current = []
    if current:
        seg = ' '.join(current).strip()
        if len(seg) >= min_len:
            segments.append(seg)
    # 去重（保留顺序）
    seen = set()
    uniq: List[str] = []
    for s in segments:
        key = s[:200]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


