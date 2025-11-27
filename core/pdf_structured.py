from __future__ import annotations

import os
from typing import Dict, List

from .text_utils import extract_text_from_pdf


def _split_paragraphs(text: str) -> List[str]:
    paragraphs: List[str] = []
    cur: List[str] = []
    for line in (text or '').splitlines():
        if line.strip():
            cur.append(line.strip())
        else:
            if cur:
                paragraphs.append(' '.join(cur))
                cur = []
    if cur:
        paragraphs.append(' '.join(cur))
    return paragraphs


def _write_paragraphs(paragraphs: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in paragraphs:
            f.write(p + '\n\n')


def process_pdf_structured(pdf_path: str, output_root: str) -> Dict[str, str]:
    """
    轻量结构化解析（兼容旧Notebook的“other”文件输出）。
    - 读取PDF为纯文本
    - 简单按空行切分段落
    - 写出 <base>_other.txt
    返回: {'other': path}
    """
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(output_root, base)
    out_path_other = os.path.join(out_dir, f"{base}_other.txt")

    text = extract_text_from_pdf(pdf_path)
    paragraphs = _split_paragraphs(text)
    _write_paragraphs(paragraphs, out_path_other)

    return {"other": out_path_other}


