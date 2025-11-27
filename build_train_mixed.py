#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

import pandas as pd

try:
	from datasets import load_dataset  # type: ignore
	_HAS_HF = True
except Exception:
	_HAS_HF = False


ROOT = Path(__file__).resolve().parent

# Source files
PATH_PARA2JSON = ROOT / "finetune/jsonl/para2json.mix_7_3.jsonl"
PATH_CLEAN_TXT2JSON = ROOT / "finetune/jsonl/clean_txt2json.jsonl"
PATH_JSON2JSON = ROOT / "finetune/jsonl/json2json.jsonl"

# Output
OUTPUT_JSONL = ROOT / "train_mixed.jsonl"

# New, simplified prompts
SYSTEM_PROMPT_PARA_TXT = "You are a specialized assistant for extracting chemical reaction data into JSON format."
USER_SUFFIX_PARA_TXT = "Extract the reaction JSON."

SYSTEM_PROMPT_JSON2JSON = "You are a specialized assistant for extracting chemical reaction data into JSON format."
USER_SUFFIX_JSON2JSON = "Merge into a complete reaction JSON."


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def to_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_context_from_user_content(user_content: str) -> str:
	"""
	From a user content that usually starts with:
	  'Context\\n<paragraphs>\\n\\nTask\\n...'
	or similar, remove Few-shot/examples/long instruction.
	Keep ONLY the context text after 'Context' and before 'Task'/policies/etc.
	"""
	# Normalize newlines
	text = user_content.replace("\r\n", "\n")

	# Find where context starts: "Context" (case-insensitive) + newline
	# If not found, fallback to whole content
	m = re.search(r"(?i)\bcontext\s*:\s*|\bcontext\s*\n", text)
	if m:
		context_start = m.end()
	else:
		# Sometimes exactly "Context\n"
		m2 = re.search(r"(?i)^context\s*\n", text)
		if m2:
			context_start = m2.end()
		else:
			# If we cannot find "Context", just use the original text
			context_start = 0

	fragment = text[context_start:].lstrip()

	# Cut at first "Task" or typical Few-shot/policy sections if present
	cut_patterns = [
		r"\n\s*Task\b",
		r"\n\s*STRICT\s+EXTRACTION\s+POLICY\b",
		r"\n\s*Few[-\s]*Shot\b",
		r"\n\s*Examples?:\b",
		r"\n\s*Formatting\s+Instructions?\b",
		r"\n\s*Output\s+Format\b",
		r"\n\s*Instructions?\b",
	]
	cut_positions: List[int] = []
	for pat in cut_patterns:
		mc = re.search(pat, fragment, flags=re.IGNORECASE)
		if mc:
			cut_positions.append(mc.start())

	if cut_positions:
		stop = min(cut_positions)
		fragment = fragment[:stop].rstrip()
	else:
		fragment = fragment.strip()

	return fragment


def simplify_record_to_chat(
	rec: Dict[str, Any],
	system_prompt: str,
	user_suffix_instruction: str,
	context_extractor=extract_context_from_user_content,
) -> Dict[str, Any]:
	"""
	Transform a single JSONL record with messages into a simplified llama 3.1 chat format:
	- system: provided system_prompt
	- user: "Context:\n{context}\n{user_suffix_instruction}"
	- assistant: keep original assistant content if present
	"""
	messages = rec.get("messages") or []
	assistant_contents: List[str] = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
	assistant_content = assistant_contents[0] if assistant_contents else ""

	user_contents: List[str] = [m.get("content", "") for m in messages if m.get("role") == "user"]
	user_content_raw = user_contents[0] if user_contents else ""

	context_text = context_extractor(user_content_raw)
	user_new = f"Context:\n{context_text}\n{user_suffix_instruction}"

	return {
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_new},
			{"role": "assistant", "content": assistant_content},
		]
	}


def process_para2json(path: Path) -> List[Dict[str, Any]]:
	processed: List[Dict[str, Any]] = []
	for rec in read_jsonl(path):
		processed.append(
			simplify_record_to_chat(
				rec,
				system_prompt=SYSTEM_PROMPT_PARA_TXT,
				user_suffix_instruction=USER_SUFFIX_PARA_TXT,
			)
		)
	return processed


def process_clean_txt2json(path: Path) -> List[Dict[str, Any]]:
	processed: List[Dict[str, Any]] = []
	for rec in read_jsonl(path):
		processed.append(
			simplify_record_to_chat(
				rec,
				system_prompt=SYSTEM_PROMPT_PARA_TXT,
				user_suffix_instruction=USER_SUFFIX_PARA_TXT,
			)
		)
	return processed


def process_json2json(path: Path) -> List[Dict[str, Any]]:
	def context_passthrough(user_content: str) -> str:
		# For json2json, keep the whole content after "Context" without trimming multiple JSON blocks
		text = user_content.replace("\r\n", "\n")
		m = re.search(r"(?i)\bcontext\s*:\s*|\bcontext\s*\n", text)
		if m:
			return text[m.end():].strip()
		m2 = re.search(r"(?i)^context\s*\n", text)
		if m2:
			return text[m2.end():].strip()
		return text.strip()

	processed: List[Dict[str, Any]] = []
	for rec in read_jsonl(path):
		processed.append(
			simplify_record_to_chat(
				rec,
				system_prompt=SYSTEM_PROMPT_JSON2JSON,
				user_suffix_instruction=USER_SUFFIX_JSON2JSON,
				context_extractor=context_passthrough,
			)
		)
	return processed


def load_alpaca_cleaned_sample(n: int = 500, seed: int = 42) -> List[Dict[str, Any]]:
	if not _HAS_HF:
		raise RuntimeError("datasets 未安装，无法下载 yahma/alpaca-cleaned。请先: pip install datasets")

	ds = load_dataset("yahma/alpaca-cleaned")
	# default split 'train'
	df = pd.DataFrame(ds["train"])
	# Random sample n rows
	if len(df) > n:
		df = df.sample(n=n, random_state=seed)
	else:
		df = df.sample(frac=1.0, random_state=seed)

	rows: List[Dict[str, Any]] = []
	for _, row in df.iterrows():
		instruction = str(row.get("instruction") or "").strip()
		inp = str(row.get("input") or "").strip()
		output = str(row.get("output") or "").strip()

		if inp:
			user_content = f"{instruction}\n{inp}"
		else:
			user_content = instruction

		rows.append({
			"messages": [
				{"role": "system", "content": "You are a helpful assistant."},
				{"role": "user", "content": user_content},
				{"role": "assistant", "content": output},
			]
		})
	return rows


def to_dataframe(rows: List[Dict[str, Any]], source: str) -> pd.DataFrame:
	# Keep a temporary 'source' column for ratio reporting. Will be dropped before saving.
	return pd.DataFrame({"messages": rows, "source": source})


def main() -> None:
	random.seed(42)

	# 1) Process local datasets with simplified prompts
	para_rows = process_para2json(PATH_PARA2JSON)
	clean_rows = process_clean_txt2json(PATH_CLEAN_TXT2JSON)
	json2json_rows = process_json2json(PATH_JSON2JSON)

	df_para = to_dataframe(para_rows, source="para2json")
	df_clean = to_dataframe(clean_rows, source="clean_txt2json")
	df_json2json = to_dataframe(json2json_rows, source="json2json")

	# 2) Download + sample alpaca-cleaned 500
	alpaca_rows = load_alpaca_cleaned_sample(n=500, seed=42)
	df_alpaca = to_dataframe(alpaca_rows, source="alpaca_cleaned")

	# 3) Enforce composition ratios: A(para+clean)=70%, B(json2json)=10%, C(alpaca)=20%
	df_A = pd.concat([df_para, df_clean], ignore_index=True)
	df_B = df_json2json
	df_C = df_alpaca  # Already sampled to 500

	# Determine target total based on fixed C=20%
	num_C = len(df_C)
	if num_C == 0:
		raise RuntimeError("通用数据（Alpaca）为 0，无法按照 20% 构建总量。")
	target_total = int(round(num_C / 0.20))
	target_B = int(round(0.10 * target_total))
	target_A = target_total - num_C - target_B
	if target_A <= 0 or target_B <= 0:
		raise RuntimeError(f"计算得到的目标规模异常: total={target_total}, A={target_A}, B={target_B}, C={num_C}")

	def sample_df_to_size(df: pd.DataFrame, size: int, seed: int, allow_oversample: bool) -> pd.DataFrame:
		n = len(df)
		if n == 0:
			raise RuntimeError("尝试从空数据集中采样。")
		if n >= size:
			return df.sample(n=size, random_state=seed)
		# 如果不允许过采样，则返回全部数据（后续会整体打乱）
		if not allow_oversample:
			return df.sample(frac=1.0, random_state=seed)
		# 允许过采样则用有放回采样补齐
		return df.sample(n=size, replace=True, random_state=seed)

	# A 类允许轻度过采样来贴近 70%
	df_A_sel = sample_df_to_size(df_A, target_A, seed=42, allow_oversample=True)
	# B 类（json2json）不进行过采样，不足则全量使用
	df_B_sel = sample_df_to_size(df_B, target_B, seed=42, allow_oversample=False)
	df_C_sel = df_C  # fixed

	# Concat, shuffle
	df_all = pd.concat([df_A_sel, df_B_sel, df_C_sel], ignore_index=True)
	df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

	# Save JSONL with only 'messages' (HuggingFace/Unsloth style)
	rows_to_save = df_all["messages"].tolist()
	OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
	to_jsonl(OUTPUT_JSONL, rows_to_save)

	# Print ratios
	print("Saved:", str(OUTPUT_JSONL))
	total = len(df_all)
	counts = df_all["source"].value_counts().to_dict()
	print("Total samples:", total)
	# 打印目标与实际（用于 json2json 是否被上限裁剪的确认）
	try:
		print(f"Target sizes => A:{target_A}, B:{target_B}, C:{num_C}")
		print(f"Actual sizes => A:{len(df_A_sel)}, B:{len(df_B_sel)}, C:{len(df_C_sel)}")
		if len(df_B_sel) < target_B:
			print("Note: json2json 数量不足目标，占比将低于 10%，已使用全部可用样本。")
	except Exception:
		pass
	for k in ["para2json", "clean_txt2json", "json2json", "alpaca_cleaned"]:
		v = counts.get(k, 0)
		ratio = (v / total * 100.0) if total > 0 else 0.0
		print(f"{k}: {v} ({ratio:.2f}%)")
	# System prompt distribution and quick sanity check
	try:
		from collections import Counter
		sys_list: List[str] = []
		for rec in df_all["messages"].tolist():
			msgs = rec.get("messages") or []
			if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
				sys_list.append(str(msgs[0].get("content") or "").strip())
		sys_counts = Counter(sys_list)
		print("System prompt distribution:")
		for s, c in sys_counts.most_common():
			pct = (c / total * 100.0) if total > 0 else 0.0
			print(f"- {repr(s)}: {c} ({pct:.2f}%)")
		# Show a few generic (Alpaca) samples
		print("Sample generic (Alpaca-like) entries:")
		shown = 0
		for rec in df_all["messages"].tolist():
			msgs = rec.get("messages") or []
			if not msgs:
				continue
			sysc = str(msgs[0].get("content") or "").strip()
			if sysc == "You are a helpful assistant.":
				print(json.dumps(rec, ensure_ascii=False)[:300])
				shown += 1
				if shown >= 3:
					break
	except Exception as e:
		print("System prompt distribution check skipped due to error:", e)


if __name__ == "__main__":
	main()


