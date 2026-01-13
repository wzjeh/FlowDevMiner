from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd
from gpt4all import GPT4All


class LocalPipeline:
    def __init__(self, model_name: str | None = None, model_path: str = 'models/', *, filter_model: str | None = None, abstract_model: str | None = None, summarize_model: str | None = None, overall_model: str | None = None, impact_model: str | None = None, finetuned_trigger_name: str | None = None) -> None:
        abs_model_path = os.path.abspath(model_path)
        # 单模型或分阶段模型装配
        self.model_path = abs_model_path
        self.model_single = None
        if model_name:
            self.model_single = GPT4All(model_name, model_path=abs_model_path, allow_download=False)
        self.model_filter_name = filter_model
        self.model_abstract_name = abstract_model
        self.model_summarize_name = summarize_model
        # 新增：用于“多段JSON汇总为1个最佳JSON”的专用模型
        self.model_overall_name = overall_model
        # 新增：Impact 阶段专用模型（例如原始 meta-llama-3.1-8b）
        self.model_impact_name = impact_model
        self.finetuned_trigger_name = finetuned_trigger_name or 'My_Finetuned_Model'
        
        # 调试信息：打印模型加载情况
        if os.getenv("FCPD_DEBUG", "0") == "1":
            print(f"  [Debug] LocalPipeline Init: Summarize='{self.model_summarize_name}', Overall='{self.model_overall_name}'")
            print(f"  [Debug] Finetuned Trigger: '{self.finetuned_trigger_name}'")

    def _create_prompt(self, system: str, user: str, context: str = "") -> str:
        """创建普通格式的 prompt"""
        parts = []
        if system:
            parts.append(system)
        if context:
            parts.append(f"Context\n{context}")
        parts.append(f"Task\n{user}")
        return "\n\n".join(parts)
    def _create_llama31_chat_prompt(self, system: str, user: str, context: str = "") -> str:
        """创建 Llama 3.1 chat template 格式的 prompt（用于微调模型）"""
        user_content = ""
        if context:
            user_content += f"Context\n{context}\n\n"
        user_content += f"Task\n{user}"
        
        # Llama 3.1 chat template 格式
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def _is_finetuned_model(self, model_name: str) -> bool:
        """判断是否是微调模型"""
        # 1. 环境变量强制覆盖
        if os.getenv('FCPD_FORCE_FINETUNED') == '1':
            return True
        if not model_name:
            return False
        # 2. 名称匹配
        is_ft = self.finetuned_trigger_name.lower() in model_name.lower()
        # if os.getenv("FCPD_DEBUG", "0") == "1":
        #     print(f"  [Debug] Is Finetuned? Model='{model_name}' vs Trigger='{self.finetuned_trigger_name}' -> {is_ft}")
        return is_ft

    def _get_chat_prompt(self, system: str, user: str, context: str = "", stage: str = "filter") -> str:
        """根据模型类型选择 prompt 格式"""
        # 判断当前阶段的模型是否是微调模型
        if stage == 'summarize' and self.model_summarize_name:
            if self._is_finetuned_model(self.model_summarize_name):
                return self._create_llama31_chat_prompt(system, user, context)
        if stage == 'overall' and self.model_overall_name:
            if self._is_finetuned_model(self.model_overall_name):
                return self._create_llama31_chat_prompt(system, user, context)
        return self._create_prompt(system, user, context)

    

    def _clip(self, text: str, max_chars: int) -> str:
        if not text:
            return text
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _safe_generate(self, model: GPT4All, prompt: str, *, max_tokens: int, temp: float = 0.0) -> str:
        try:
            out = (model.generate(prompt=prompt, max_tokens=max_tokens, temp=temp) or '').strip()
            # 若模型返回上下文超限类错误文本，而非抛异常，则主动降载重试
            low = out.lower()
            if ("context window" in low and "exceed" in low) or ("prompt size" in low and "context window" in low):
                clipped = self._clip(prompt, 2000)
                out2 = (model.generate(prompt=clipped, max_tokens=max_tokens - 100, temp=temp) or '').strip()
                low2 = out2.lower()
                if ("context window" in low2 and "exceed" in low2) or ("prompt size" in low2 and "context window" in low2):
                    clipped = self._clip(prompt, 1500)
                    return (model.generate(prompt=clipped, max_tokens=max_tokens - 200, temp=temp) or '').strip()
                return out2
            return out
        except Exception:
            # 二次降载：减少prompt与max_tokens后重试
            clipped = self._clip(prompt, 2000)
            try:
                return (model.generate(prompt=clipped, max_tokens=max_tokens - 100, temp=temp) or '').strip()
            except Exception:
                # 最终兜底
                clipped = self._clip(prompt, 1500)
                return (model.generate(prompt=clipped, max_tokens=max_tokens - 200, temp=temp) or '').strip()

    def _pack_adjacent_rows(self, df: pd.DataFrame, column: str, *, min_chars: int, max_chars: int) -> pd.DataFrame:
        """将相邻行按顺序合并到[min_chars, max_chars]范围，避免短句信息不足，控制总长避免溢出。"""
        if column not in df.columns:
            return df
        packed_rows: list[str] = []
        cur_parts: list[str] = []
        cur_len = 0
        for _, row in df.iterrows():
            s = str(row.get(column, "") or "")
            if not s.strip():
                continue
            if cur_len == 0:
                # 起始直接加入
                cur_parts = [s]
                cur_len = len(s)
                # 单段超长：直接输出为一个块（不切分）
                if cur_len >= max_chars:
                    packed_rows.append(s[:max_chars])
                    cur_parts, cur_len = [], 0
                continue
            # 尝试合并
            candidate_len = cur_len + 1 + len(s)
            if candidate_len <= max_chars:
                cur_parts.append(s)
                cur_len = candidate_len
                continue
            # 达到上限：若当前块不足下限，仍然先输出以保证顺序；否则输出当前块
            packed_rows.append(" ".join(cur_parts))
            # 开启新的块
            cur_parts = [s]
            cur_len = len(s)
            if cur_len >= max_chars:
                packed_rows.append(s[:max_chars])
                cur_parts, cur_len = [], 0
        if cur_len > 0 and cur_parts:
            # 末尾块
            if cur_len < min_chars and packed_rows:
                # 若末尾不足下限，尽量与上一块合并（不超过上限）
                prev = packed_rows.pop()
                merged = (prev + " " + " ".join(cur_parts)).strip()
                if len(merged) <= max_chars:
                    packed_rows.append(merged)
                else:
                    packed_rows.append(prev)
                    packed_rows.append(" ".join(cur_parts))
            else:
                packed_rows.append(" ".join(cur_parts))
        return pd.DataFrame(packed_rows, columns=[column])

    def save_df_to_text(self, df: pd.DataFrame, file_path: str, content_column: str = 'content') -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(str(row.get(content_column, "")) + "\n\n")

    def _smart_pack_rows(self, df: pd.DataFrame, column: str, *, max_chars: int = 1500, window: int = 1) -> pd.DataFrame:
        """
        智能相邻打包：当某行包含高收益/高转化信号，但缺少条件单位时，与相邻±window行拼接，长度不超过 max_chars。
        触发启发式：
          - 文本含百分号或关键词 (yield|conversion|selectivity)
          - 且不含温度/时间/压力/总流量单位（允许后续 Summarize 更容易就近捕获条件）
        """
        if column not in df.columns:
            return df
        import re
        texts = [str(x or "") for x in df[column].tolist()]
        n = len(texts)
        if n == 0:
            return df
        percent_re = re.compile(r"\b\d{1,3}\s?%\b")
        outcomes_re = re.compile(r"\b(yield|conversion|selectivity)\b", re.IGNORECASE)
        # 条件单位：温度/时间/压力/总流量
        units_re = re.compile(
            r"(?:\b(?:°C|K|°F)\b|\b(?:\d+\.?\d*)\s*(?:min|s|h)\b|\b(?:bar|MPa)\b|\bmL\s*/\s*(?:min|h)\b|\bµL\s*/\s*min\b|\buL\s*/\s*min\b)",
            re.IGNORECASE,
        )
        used = [False] * n
        packed: list[str] = []
        for i in range(n):
            if used[i]:
                continue
            base = texts[i]
            t_low = base.lower()
            has_result = bool(percent_re.search(base) or outcomes_re.search(t_low))
            lacks_units = not bool(units_re.search(base))
            if has_result and lacks_units:
                # 触发智能拼接：与相邻±window拼接（优先后，再前），限制长度
                combined = base
                used[i] = True
                # 向后
                for off in range(1, window + 1):
                    j = i + off
                    if j < n and not used[j] and len(combined) + 1 + len(texts[j]) <= max_chars:
                        combined = combined + " " + texts[j]
                        used[j] = True
                # 向前
                for off in range(1, window + 1):
                    j = i - off
                    if j >= 0 and not used[j] and len(texts[j]) + 1 + len(combined) <= max_chars:
                        combined = texts[j] + " " + combined
                        used[j] = True
                packed.append(combined)
            else:
                used[i] = True
                packed.append(base)
        return pd.DataFrame(packed, columns=[column])

    def filter_content(self, df: pd.DataFrame) -> pd.DataFrame:
        user_question = (
            "Does this paragraph describe the authors' OWN experimental work, results, or conclusions in THIS SPECIFIC study? "
            "Answer 'Yes' ONLY if it details new experiments, specific conditions (temp/time/yield), or results performed by the authors.\n"
            "Answer 'No' if it is Introduction, Background, Literature Review (citing previous work/references), general theory, or future plans.\n"
            "Strictly answer 'Yes' or 'No'."
        )
        system_prompt = (
            "You are an expert assistant for scientific literature mining. "
            "Keep experimental sections, results, abstracts, and conclusions. Discard background/intro."
        )
        results = []
        for _, row in df.iterrows():
            content = self._clip(str(row['content']), 1200)
            # 扩展关键词：增加章节标题类强信号，直接保留
            kw_strong = [
                "abstract", "conclusion", "experimental", "experimental section",
                "materials and methods", "methods", "procedure", "general procedure",
                "typical procedure", "optimization", "best conditions", "results and discussion"
            ]
            # kw_technical 已被移除，不再用于跳过 LLM 检查
            content_lower = content.lower()
            # 1) 标题窗口命中：前 N 字符内包含强信号标题则直接保留
            try:
                title_window = int(os.getenv("FCPD_FILTER_TITLE_WINDOW", "80"))
            except Exception:
                title_window = 80
            prefix = content_lower[:max(0, title_window)]
            if any(k in prefix for k in kw_strong):
                results.append('Yes')
                continue
            # 2) 技术关键词命中 -> 强制交给 LLM 判断，以减少误报
            # 恢复部分极高置信度的技术指标直通，以避免 LLM 过度过滤
            # 规则：必须包含明确的数值结果（%）或关键流程参数（residence time）
            if "residence time" in content_lower:
                results.append('Yes')
                continue
            if "%" in content and ("yield" in content_lower or "conversion" in content_lower):
                # 简单的正则检查数值
                import re
                if re.search(r"\d+\s*%", content):
                    results.append('Yes')
                    continue
            
            # --- NEW: Enhanced Filter for Reactor & Conditions ---
            # 捕获反应器特定信息 (inner diameter, microreactor, etc.)
            if any(k in content_lower for k in ["inner diameter", "internal diameter", "microreactor", "micro-reactor", "flow rate", "residence time", "back pressure"]):
                results.append('Yes')
                continue
            
            # 捕获带有单位的实验条件 (temp, pressure, flow)
            # e.g. "100 °C", "5 bar", "1.0 mL/min"
            # 这是一个强信号，表明包含具体实验参数
            import re
            if re.search(r"(?:\d+\s*(?:°C|°F|K|bar|MPa|psi|mL/min|µL/min|ul/min))", content, re.IGNORECASE):
                results.append('Yes')
                continue
            # -----------------------------------------------------
            
            # 3) 其它情况交给 LLM 判断
            prompt = self._create_prompt(system_prompt, user_question, content)
            model = self._get_stage_model('filter')
            resp = self._safe_generate(model, prompt, max_tokens=5, temp=0.0)
            results.append('Yes' if resp.strip().lower().startswith('yes') else 'No')
        out = df.copy()
        out['classification'] = results
        return out[out['classification'] == 'Yes'].copy()

    def abstract_text(self, df: pd.DataFrame) -> pd.DataFrame:
        user_prompt = (
            "Summarize the paragraph focusing on flow-chemistry process development. "
            "Highlight: reaction type, key reactants/solvent/catalyst, products, reactor details (type/ID), "
            "critical conditions (flow rate(s), residence time, temperature, pressure), and outcomes (conversion/yield/selectivity). "
            "Be concise, faithful to text, no speculation."
        )
        system_prompt = (
            "You are an expert assistant for scientific literature mining. Return concise, faithful summaries."
        )
        abstracts = []
        for _, row in df.iterrows():
            content = self._clip(str(row['content']), 2000)
            prompt = self._create_prompt(system_prompt, user_prompt, content)
            model = self._get_stage_model('abstract')
            text = self._safe_generate(model, prompt, max_tokens=280, temp=0.0)
            abstracts.append(text.strip() or content[:400])
        out = df.copy()
        out['abstract'] = abstracts
        return out

    def _sanitize_json_text(self, text: str) -> str:
        import re
        s = text or ""
        # 1) 去除 markdown 围栏
        s = re.sub(r"```json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"```\s*", "", s)
        # 2) 仅保留最大完整 { ... } 块
        brace_count = 0
        start = -1
        candidates = []
        for i, ch in enumerate(s):
            if ch == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif ch == '}':
                if brace_count > 0:
                    brace_count -= 1
                    if brace_count == 0 and start != -1:
                        candidates.append(s[start:i+1])
                        start = -1
        if candidates:
            s = max(candidates, key=len)
        # 3) 去注释与尾逗号
        s = re.sub(r"//.*?(?=\n|$)", "", s)
        s = re.sub(r"/\*[\s\S]*?\*/", "", s)
        s = re.sub(r",\s*(\}|\])", r"\1", s)
        # 4) 数值化（将 "82" → 82，仅限纯数字的值）
        s = re.sub(r':\s*"(-?\d+\.?\d*)"\s*([,\}])', r': \1\2', s)
        # 5) 将未加引号的省略号（... 或 …）标准化为字符串，避免 JSON 解析失败
        s = re.sub(r':\s*(\.\.\.|…)\s*([,\}\]])', r': "..." \2', s)
        return s.strip()

    def _keep_best_only(self, json_text: str) -> str:
        import json
        try:
            obj = json.loads(json_text)
        except Exception:
            return json_text
        
        # 1. 结构标准化：将 reaction_summary 内嵌的 products/metrics/reactants/conditions 移到顶层
        rs = obj.get("reaction_summary", {}) or {}
        if isinstance(rs, dict):
            if "products" in rs and isinstance(rs["products"], list):
                obj.setdefault("products", []).extend(rs["products"])
                rs.pop("products", None)
            if "metrics" in rs and isinstance(rs["metrics"], dict):
                if not obj.get("metrics"):
                    obj["metrics"] = rs["metrics"]
                rs.pop("metrics", None)
            if "reactants" in rs and isinstance(rs["reactants"], list):
                obj.setdefault("reactants", []).extend(rs["reactants"])
                rs.pop("reactants", None)
            if "conditions" in rs and isinstance(rs["conditions"], list):
                obj.setdefault("conditions", []).extend(rs["conditions"])
                rs.pop("conditions", None)

        # 2. 确保列表字段类型正确
        for k in ["products", "reactants", "conditions"]:
            if not isinstance(obj.get(k), list):
                obj[k] = []
        
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json_text
            # return json_text
    def _normalize_reaction_type(self, value: str | None) -> str | None:
        if not value or not isinstance(value, str):
            return None
        v = value.strip().lower()
        if not v:
            return None
        synonyms = {
            "hydrogenation": {"hydrogenation", "h2 addition"},
            "nitration": {"nitration"},
            "oxidation": {"oxidation", "oxidative"},
            "reduction": {"reduction", "reductive"},
            "photoreduction": {"photoreduction", "photo-reduction"},
            "esterification": {"esterification"},
            "amidation": {"amidation"},
            "halogenation": {"halogenation", "chlorination", "bromination", "fluorination", "iodination"},
            "alkylation": {"alkylation"},
            "acylation": {"acylation", "friedel-crafts acylation"},
            "isomerization": {"isomerization"},
            "dehydration": {"dehydration"},
            "dehydrogenation": {"dehydrogenation"},
            "polymerization": {"polymerization"},
            "coupling": {"coupling", "suzuki coupling", "heck coupling", "sonogashira coupling"},
            "hydrolysis": {"hydrolysis"},
            "photocatalysis": {"photocatalysis", "photocatalytic"},
        }
        non_types = {
            "conversion", "selectivity", "yield", "production", "regime", "microfluidic regime",
            "batch synthesis", "gas-liquid system", "gas-liquid catalytic reaction", "bayesian optimization",
            "synthesis", "process", "purge"
        }
        if v in non_types:
            return None
        for norm, alts in synonyms.items():
            if v in alts:
                return norm
        for norm, alts in synonyms.items():
            for a in alts:
                if a in v:
                    return norm
        strict = os.getenv("FCPD_RXN_TAXONOMY_STRICT", "1") == "1"
        return None if strict else v
    def _clean_chem_name(self, name: str | None) -> str | None:
        if not name or not isinstance(name, str):
            return None
        n = name.strip()
        if not n: return None
        # 1. 基础清洗
        # 去除首尾标点
        n = n.strip(".,;:")
        # 2. 常见格式错误修正 (可按需扩展)
        # e.g. "4-chloro nitrobenzene" -> "4-chloronitrobenzene" (仅作示例，此处逻辑较激进，需谨慎)
        # 修正字典 (用户可在此添加特定映射)
        corrections = {
            "nltrobenzene": "nitrobenzene",
            # "nitrosobenzene": "nitrobenzene", # 警告：这是不同物质，仅当确信是误识时才启用
        }
        if n.lower() in corrections:
            return corrections[n.lower()]
        
        # 3. 括号编号去除：如 "Compound (1)" -> "Compound"
        import re
        # 去除末尾的 "(数字/字母)" 引用
        n = re.sub(r'\s*\(\s*\d+[a-z]?\s*\)$', '', n)
        # 去除末尾的纯数字编号 "Product 3" -> "Product"
        n = re.sub(r'\s+\d+$', '', n)
        
        return n if len(n) > 1 else None

    def _ensure_schema_and_order(self, obj_in: dict) -> dict:
        import copy
        obj = copy.deepcopy(obj_in) if isinstance(obj_in, dict) else {}
        rs = obj.get("reaction_summary") or {}
        if not isinstance(rs, dict):
            rs = {}
        rxn_type = rs.get("reaction_type") or obj.get("reaction_type")
        rxn_type = self._normalize_reaction_type(rxn_type if isinstance(rxn_type, str) else None)
        reactants = obj.get("reactants")
        if not isinstance(reactants, list):
            reactants = rs.get("reactants") if isinstance(rs.get("reactants"), list) else []
        products = obj.get("products")
        if not isinstance(products, list):
            products = rs.get("products") if isinstance(rs.get("products"), list) else []
        
        # 自动修正：根据 role 字段调整归属
        final_reactants = []
        final_products = list(products) # 复制一份
        
        for r in reactants:
            if not isinstance(r, dict): 
                continue
            # 清洗名称
            r["name"] = self._clean_chem_name(r.get("name"))
            if not r["name"]: continue
            
            role = str(r.get("role", "")).lower()
            name = r.get("name")
            if "product" in role and "reactant" not in role:
                # 误入 reactants 的 product，移动过去
                # 检查是否已存在
                if not any(p.get("name") == name for p in final_products):
                    final_products.append({"name": name, "yield_optimal": None, "unit": "%"})
            else:
                final_reactants.append(r)
        
        # 清洗 products 名称
        for p in final_products:
            if isinstance(p, dict):
                p["name"] = self._clean_chem_name(p.get("name"))
        # 过滤掉清洗后名字为空的
        final_products = [p for p in final_products if p.get("name") or p.get("yield_optimal")]

        reactants = final_reactants
        products = final_products

        conditions = obj.get("conditions")
        if not isinstance(conditions, list):
            conditions = rs.get("conditions") if isinstance(rs.get("conditions"), list) else []
        reactor = obj.get("reactor")
        if not isinstance(reactor, dict):
            reactor = rs.get("reactor") if isinstance(rs.get("reactor"), dict) else {}
        metrics = obj.get("metrics")
        if not isinstance(metrics, dict):
            metrics = rs.get("metrics") if isinstance(rs.get("metrics"), dict) else {}
        target_types = ["temperature", "residence_time", "flow_rate_total", "pressure"]
        def _find_cond_value(t, conds):
            for c in conds if isinstance(conds, list) else []:
                if isinstance(c, dict) and c.get("type") == t:
                    return c.get("value")
            return None
        fixed_conds = []
        for t in target_types:
            v = _find_cond_value(t, conditions)
            v = v if isinstance(v, str) and v.strip() else None
            fixed_conds.append({"type": t, "value": v})
        conv = metrics.get("conversion") if isinstance(metrics, dict) else None
        yld = metrics.get("yield") if isinstance(metrics, dict) else None
        sel = metrics.get("selectivity") if isinstance(metrics, dict) else None
        fixed_metrics = {
            "conversion": conv if isinstance(conv, (int, float)) else None,
            "yield": yld if isinstance(yld, (int, float)) else None,
            "selectivity": sel if isinstance(sel, (int, float)) else None,
            "unit": "%"
        }
        fixed_reactor = {
            "type": reactor.get("type") if isinstance(reactor, dict) else None,
            "inner_diameter": reactor.get("inner_diameter") if isinstance(reactor, dict) else None
        }
        out = {}
        out["reaction_summary"] = {"reaction_type": rxn_type}
        out["reactants"] = [r for r in reactants if isinstance(r, dict)]
        out["products"] = [p for p in products if isinstance(p, dict)]
        out["conditions"] = fixed_conds
        out["reactor"] = fixed_reactor
        out["metrics"] = fixed_metrics
        return out

    def summarize_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        # 路径分支：微调模型使用精简 Prompt；未微调保持原有严格策略
        use_finetuned = bool(self.model_summarize_name) and self._is_finetuned_model(self.model_summarize_name)
        if use_finetuned:
            system_prompt = "You output ONLY valid JSON. No explanations."
            user_prompt = (
                "Only use THIS paragraph; do not infer across other paragraphs.\n"
                "NEGATIVE CONSTRAINTS: If the paragraph describes 'previous work', 'literature review', or 'background theory', ONLY extract data if it is explicitly compared with or part of the current study's experimental results. Otherwise, return a JSON with all nulls. Do NOT extract data cited from other papers unless relevant to the current study's optimization.\n\n"
                "SPECIAL ATTENTION (Gases & Solvents):\n"
                "- Explicitly extract gaseous reactants (e.g., Hydrogen/H2, Oxygen/O2, CO) if flow rates or partial pressures are mentioned, even if not in a 'reagents' list.\n"
                "- Include simple molecules (e.g., ethylene, benzene, toluene) as reactants/products if they are the main reaction components.\n\n"
                # "For conditions, prefer values with units (°C/K/°F; s/min/h; bar/MPa; mL/min). Ignore counts like '7 experiments'.\n"
                # "If a result (yield/conversion) is mentioned, associate the nearest temperature/time in the same or preceding sentence.\n\n"
                "Schema (keys in this exact order):\n"
                "1) reaction_summary.reaction_type\n"
                "2) reactants[{name,role}] (include gases like H2/O2 if used)\n"
                "3) products[{name,yield_optimal,unit}]\n"
                "4) conditions[{type,value}]  (types: temperature,residence_time,flow_rate_total,pressure)\n"
                "5) reactor{type,inner_diameter}\n"
                "6) metrics{conversion,yield,selectivity,unit}\n\n"
                "Reaction type taxonomy (choose ONE label, otherwise null): hydrogenation, nitration, oxidation, reduction, photoreduction, esterification, amidation, halogenation, alkylation, acylation, isomerization, dehydration, dehydrogenation, polymerization, coupling, hydrolysis, photocatalysis.\n"
                "Do NOT output non-types such as conversion, batch synthesis, microfluidic regime, gas-liquid system, Bayesian optimization, synthesis, process, purge.\n\n"
                "Rules:\n"
                "- Extract values exactly as they appear; if not present, use null.\n"
                "- SPECIAL ATTENTION: For reactor 'inner_diameter', ONLY extract if a specific number with unit (mm/microns) is explicitly stated. DO NOT assume 0.5 mm or 'coil' unless written.\n"
                "- Prefer full chemical names with abbreviations (e.g. \"3-methyl-2-nitrobenzoic acid (MNA)\") if available in text.\n"
                "- No guessing or inventions; use null/empty arrays when unknown.\n"
                "- Output ONLY valid JSON without extra text.\n"
            )
        else:
            # 3. 回退到类似于 Baseline (Summerized.py) 的启发式提取策略
            # 允许模型根据上下文推断最佳条件，而不是死板地仅提取显式声明
            system_prompt = (
                "You are an expert assistant for scientific literature mining. "
                "Your goal is to extract the SINGLE BEST reaction condition set from the text."
            )
            user_prompt = (
                "Analyze the provided paragraph regarding flow-chemistry process development.\n"
                "Task:\n"
                "1. Identify the MAIN reaction described (Authors' OWN work only).\n"
                "2. Extract the OPTIMAL/BEST condition set mentioned (highest yield/conversion).\n"
                "   - If no explicit 'best' is labeled, infer it based on the highest reported metrics.\n"
                "   - Look for temperature, residence time, flow rates, and reactor details associated with that result.\n"
                "3. Extract all reactants and products involved in this main reaction.\n\n"
                "NEGATIVE CONSTRAINTS: If the paragraph describes 'previous work', 'literature review', or 'background theory', ONLY extract data if it is explicitly compared with or part of the current study's experimental results. Otherwise, return a JSON with all nulls. Do NOT extract data cited from other papers unless relevant to the current study's optimization.\n\n"
                # "For conditions, prefer values with units (°C/K/°F; s/min/h; bar/MPa; mL/min). Ignore counts like '7 experiments'.\n"
                "Guidance:\n"
                "- For conditions, prefer values with units (°C/K/°F; s/min/h; bar/MPa; mL/min). Ignore counts like '7 experiments'.\n"
                "- If a result (yield/conversion) is mentioned, associate the nearest temperature/time in the same or preceding sentence.\n"
                "Schema:\n"
                "{ \"reaction_summary\": {\n"
                "  \"reaction_type\": <string|null>,\n"
                "  \"reactants\": [ {\"name\": <string>, \"role\": \"reactant\"|\"catalyst\"|\"solvent\"} ],\n"
                "  \"products\": [ {\"name\": <string>, \"yield_optimal\": <number|null>, \"unit\": \"%\"} ],\n"
                "  \"conditions\": [\n"
                "    {\"type\": \"temperature\", \"value\": <string>},\n"
                "    {\"type\": \"residence_time\", \"value\": <string>},\n"
                "    {\"type\": \"flow_rate_total\", \"value\": <string>},\n"
                "    {\"type\": \"pressure\", \"value\": <string>}\n"
                "  ],\n"
                "  \"reactor\": {\"type\": <string>, \"inner_diameter\": <string>},\n"
                "  \"metrics\": {\"conversion\": <number|null>, \"yield\": <number|null>, \"selectivity\": <number|null>, \"unit\": \"%\"}\n"
                "} }\n"
                "Rules:\n"
                "- Keep original units (e.g. '100 °C', '30 min').\n"
                "- Map flow rates to 'flow_rate_total' if unspecified.\n"
                "- Extract full chemical names where possible.\n"
            )
        summarized = []
        used_col = 'content' if 'content' in df.columns else ('abstract' if 'abstract' in df.columns else 'content')
        total_cnt = 0
        parsed_ok_cnt = 0
        ellipsis_cnt = 0
        signal0_cnt = 0
        for _, row in df.iterrows():
            # 优先使用过滤后的原文段落（content）；若无则使用 abstract
            content = str(row.get(used_col, row.get('content', row.get('abstract', ''))))
            content = self._clip(content, 2200)
            
            # 使用新的 _get_chat_prompt 方法，自动选择格式
            base_prompt = self._get_chat_prompt(system_prompt, user_prompt, content, stage='summarize')
            
            model = self._get_stage_model('summarize')
            # 根据模型类型调整 max_tokens：微调模型给足 2048，未微调模型限制在 800 以免挤占 Input Context
            gen_max_tokens = 2048 if use_finetuned else 800
            raw = self._safe_generate(model, base_prompt, max_tokens=gen_max_tokens, temp=0.0)
            
            # 去围栏并抽取花括号块
            import re as _re
            raw = _re.sub(r"```json\s*", "", raw, flags=_re.IGNORECASE)
            raw = _re.sub(r"```\s*", "", raw)
            start, end = raw.find('{'), raw.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end+1]
            txt = self._sanitize_json_text(raw)
            txt = self._keep_best_only(txt)
            # 段落后处理验证：强类型与原文对齐
            def _post_validate_json(json_text: str, paragraph: str) -> str:
                import json as _json, re as _re
                try:
                    obj = _json.loads(json_text)
                except Exception:
                    return json_text
                para = paragraph or ""
                para_low = para.lower()
                # 1) 产品名称强类型：数字/编号作名 → null
                prods = obj.get("products", []) or []
                fixed_prods = []
                for p in prods if isinstance(prods, list) else []:
                    if not isinstance(p, dict):
                        continue
                    name = p.get("name")
                    # 纯数字或编号样式不作为合法名称
                    if isinstance(name, (int, float)):
                        p["name"] = None
                    elif isinstance(name, str):
                        n = name.strip()
                        if _re.fullmatch(r"\(?\d+\)?\.?", n):
                            p["name"] = None
                    fixed_prods.append(p)
                obj["products"] = fixed_prods
                # 2) 数值原文对齐：metrics 与 yield_optimal 必须在段落中出现
                def _num_in_text(val: float) -> bool:
                    # 构造若干匹配形式：99.5、99.5%（允许空格）
                    s = f"{val}".rstrip('0').rstrip('.') if isinstance(val, float) else f"{val}"
                    patterns = [
                        rf"\b{s}\b",
                        rf"\b{s}\s*%\b",
                    ]
                    for pat in patterns:
                        if _re.search(pat, para):
                            return True
                    return False
                met = obj.get("metrics") or {}
                if not isinstance(met, dict):
                    met = {}
                for k in ["conversion", "yield", "selectivity"]:
                    v = met.get(k)
                    if isinstance(v, (int, float)):
                        if not _num_in_text(v):
                            met[k] = None
                obj["metrics"] = met
                # yield_optimal 校验
                prods2 = obj.get("products", []) or []
                for p in prods2:
                    y = p.get("yield_optimal")
                    if isinstance(y, (int, float)):
                        if not _num_in_text(y):
                            p["yield_optimal"] = None
                obj["products"] = prods2
                # 3) 条件值出现性：放宽校验
                # 原因：LLM 可能会将 "25 °C" 格式化为 "25°C"，或者将 "room temperature" 提取为 "25 °C"。
                # 过于严格的原文匹配会导致大量有效信息丢失。
                conds = obj.get("conditions", []) or []
                fixed_conds = []
                # 条件值单位/短语校验：允许单位或常见短语（room temperature/RT），否则置 None
                _unit_re = _re.compile(
                    r"(?:\b(?:°C|K|°F)\b|\b\d+\.?\d*\s*(?:min|s|h)\b|\b(?:bar|MPa)\b|\bmL\s*/\s*(?:min|h)\b|\bµL\s*/\s*min\b|\buL\s*/\s*min\b)",
                    _re.IGNORECASE,
                )
                for c in conds if isinstance(conds, list) else []:
                    if not isinstance(c, dict):
                        continue
                    val = c.get("value")
                    if isinstance(val, str):
                        v = val.strip()
                        if not v:
                            c["value"] = None
                        else:
                            low = v.lower()
                            if ("room temperature" in low) or (low == "rt"):
                                c["value"] = v
                            elif _unit_re.search(v):
                                c["value"] = v
                            else:
                                c["value"] = None
                    else:
                        c["value"] = None
                    fixed_conds.append(c)
                obj["conditions"] = fixed_conds
                
                # 5) Reactor 原文对齐校验: 避免幻觉 (如 0.5 mm)
                # 必须在原文中找到对应的数值 (0.5) 或者完整短语
                rea = obj.get("reactor") or {}
                if isinstance(rea, dict):
                    dia = rea.get("inner_diameter")
                    if isinstance(dia, (int, float)):
                        if not _num_in_text(dia):
                            rea["inner_diameter"] = None
                    elif isinstance(dia, str) and dia:
                        # 尝试提取数字部分进行校验
                        import re
                        m = re.search(r"(\d+\.?\d*)", dia)
                        if m:
                            num_val = m.group(1)
                            # 如果原文里连这个数字都没出现，那肯定是幻觉
                            if num_val not in para:
                                rea["inner_diameter"] = None
                        else:
                            # 纯文本描述，如果原文没出现该短语，也置空
                            if dia not in para:
                                rea["inner_diameter"] = None
                    obj["reactor"] = rea

                # 6) 反应类型归一化与最终 schema/顺序
                obj = self._ensure_schema_and_order(obj)
                try:
                    return _json.dumps(obj, ensure_ascii=False)
                except Exception:
                    return json_text
            # 严格后处理开关：设置环境变量 FCPD_POST_STRICT=0 可跳过严格校验，以便评测放宽
            if os.getenv("FCPD_POST_STRICT", "1") != "0":
                txt = _post_validate_json(txt, content)
            # 骨架 JSON（用于解析失败或异常时兜底）
            def _skeleton_json() -> str:
                import json as _json
                return _json.dumps({
                    "reaction_summary": {"reaction_type": None},
                    "reactants": [],
                    "products": [],
                    "conditions": [
                        {"type": "temperature", "value": None},
                        {"type": "residence_time", "value": None},
                        {"type": "flow_rate_total", "value": None},
                        {"type": "pressure", "value": None}
                    ],
                    "reactor": {"type": None, "inner_diameter": None},
                    "metrics": {"conversion": None, "yield": None, "selectivity": None, "unit": "%"}
                }, ensure_ascii=False)
            # 统计与放宽过滤：仅“信号=0”时丢弃；'...' 仅计数不直接丢弃；解析失败才丢弃
            total_cnt += 1
            if '...' in txt or '…' in txt or '...' in raw or '…' in raw:
                ellipsis_cnt += 1
            try:
                import json as _json
                obj = _json.loads(txt)
                parsed_ok_cnt += 1
            except Exception:
                summarized.append(_skeleton_json())  # 解析失败输出骨架
                continue
            # 统计有效信号
            signals = 0
            # reactants
            for r in obj.get('reactants', []) or []:
                if isinstance(r, dict) and r.get('name'):
                    signals += 1
                    break
            # products
            for p in obj.get('products', []) or []:
                if isinstance(p, dict) and (p.get('name') or isinstance(p.get('yield_optimal'), (int, float))):
                    signals += 1
                    break
            # conditions
            for c in obj.get('conditions', []) or []:
                if isinstance(c, dict) and c.get('value'):
                    signals += 1
                    break
            # reactor
            reactor = obj.get('reactor') or {}
            if isinstance(reactor, dict) and (reactor.get('type') or reactor.get('inner_diameter')):
                signals += 1
            # metrics
            met = obj.get('metrics') or {}
            if isinstance(met, dict) and any(isinstance(met.get(k), (int, float)) for k in ['conversion','yield','selectivity']):
                signals += 1
            if signals == 0:
                signal0_cnt += 1
            try:
                import json as _json
                summarized.append(_json.dumps(obj, ensure_ascii=False))
            except Exception:
                summarized.append(txt if signals >= 0 else _skeleton_json())
        out = df.copy()
        out['summarized'] = summarized
        # 诊断可观测性（仅在调试模式输出）
        if os.getenv("FCPD_DEBUG", "0") == "1":
            try:
                print(f"  🔎 Summarize统计 -> 段落来源列: {used_col}, 段落总数: {total_cnt}, 解析成功: {parsed_ok_cnt}, 含省略号: {ellipsis_cnt}, 信号=0条数: {signal0_cnt}")
            except Exception:
                pass
        return out

    def summarize_document_overall(self, df_abstract: pd.DataFrame) -> str:
        """将多个段落级结构化JSON汇总为单一最佳JSON；若无JSON则回落到基于文本的汇总。"""
        import re, json
        # 预算与分批参数（避免上下文溢出）
        max_candidates = int(os.getenv('FCPD_OVERALL_MAX_CAND', '24'))
        char_budget = int(os.getenv('FCPD_OVERALL_CHAR_BUDGET', '8000'))
        chunk_size = int(os.getenv('FCPD_OVERALL_CHUNK_SIZE', '12'))
        overall_max_tokens = int(os.getenv('FCPD_OVERALL_MAX_TOKENS', '450'))

        def _select_under_budget(items: list[str]) -> list[str]:
            selected = []
            total = 0
            for s in items:
                if len(selected) >= max_candidates:
                    break
                add_len = len(s) + (2 if selected else 0)
                if (total + add_len) > char_budget:
                    break
                selected.append(s)
                total += add_len
            return selected

        def _merge_once(cands: list[str], system_prompt: str, use_ft_overall: bool, user_prompt_str: str) -> str:
            nonlocal overall_max_tokens
            if not cands:
                return ""
            # 按预算裁剪候选
            cands = _select_under_budget(cands)
            combined_local = "\n\n".join(cands)
            # 构建提示词
            user_prompt = user_prompt_str
            prompt_local = self._get_chat_prompt(system_prompt, user_prompt, combined_local, stage='overall')
            raw_local = self._safe_generate(self._get_stage_model('overall'), prompt_local, max_tokens=overall_max_tokens, temp=0.0)
            raw_local = re.sub(r"```json\s*", "", raw_local, flags=re.IGNORECASE)
            raw_local = re.sub(r"```\s*", "", raw_local)
            s0, e0 = raw_local.find("{"), raw_local.rfind("}")
            if s0 != -1 and e0 != -1 and e0 > s0:
                raw_local = raw_local[s0:e0+1]
            return self._keep_best_only(self._sanitize_json_text(raw_local))

        # 优先使用段落级JSON（列名为 summarized）
        if 'summarized' in df_abstract.columns:
            # 仅收集包含花括号的候选 JSON 行，逐条拼接到 Context
            raw_items = df_abstract['summarized'].fillna("").tolist()
            json_items = []
            # 预收集候选中的条件值，用于“唯一一致值”补全
            cond_pool = {"temperature": set(), "residence_time": set(), "flow_rate_total": set(), "pressure": set()}
            def _valid_cond_value(v: str) -> bool:
                try:
                    import re as _re
                    if not isinstance(v, str) or not v.strip():
                        return False
                    low = v.strip().lower()
                    if ("room temperature" in low) or (low == "rt"):
                        return True
                    unit_re = _re.compile(r"(?:\b(?:°C|K|°F)\b|\b\d+\.?\d*\s*(?:min|s|h)\b|\b(?:bar|MPa)\b|\bmL\s*/\s*(?:min|h)\b|\bµL\s*/\s*min\b|\buL\s*/\s*min\b)", _re.IGNORECASE)
                    return bool(unit_re.search(v))
                except Exception:
                    return False
            def _collect_cond_values(txt: str) -> None:
                import json as _json
                try:
                    obj = _json.loads(txt)
                except Exception:
                    return
                conds = obj.get("conditions") or obj.get("reaction_summary", {}).get("conditions") or []
                if isinstance(conds, list):
                    for c in conds:
                        if isinstance(c, dict):
                            t = c.get("type")
                            v = c.get("value")
                            if t in cond_pool and _valid_cond_value(v):
                                cond_pool[t].add(v.strip())
            for t in raw_items:
                s = str(t).strip()
                if not s:
                    continue
                if "{" in s and "}" in s:
                    json_items.append(s)
                    _collect_cond_values(s)
            for t in raw_items:
                s = str(t).strip()
                if not s:
                    continue
                if "{" in s and "}" in s:
                    json_items.append(s)
            # 为防止超长，限制候选条数与字符预算
            combined_items = _select_under_budget(json_items)
            combined = "\n\n".join(combined_items)
            if not json_items:
                # 若 summarized 存在但均为空/无效，触发回落
                col = 'abstract' if 'abstract' in df_abstract.columns else 'content'
                texts = [t for t in df_abstract[col].fillna("").tolist() if t.strip()]
                combined = "\n\n".join(texts)
                if len(combined) > char_budget:
                    combined = combined[:char_budget]
                system_prompt = "You output ONLY valid JSON. No explanations."
                user_prompt = (
                    "Extract ONE JSON summarizing the optimal condition set from the document text. "
                    "Use null/empty arrays for unknowns. Do not invent chemicals or numbers not present in text. "
                    "If conflicting info exists, choose null. Keep metrics numeric when present. "
                    "Schema keys must be exactly: reaction_summary.reaction_type, reactants[{name,role}], "
                    "products[{name,yield_optimal,unit}], conditions[{type,value}], reactor{type,inner_diameter}, "
                    "metrics{conversion,yield,selectivity,unit}."
                )
            else:
                # 对微调模型使用更简洁且明确的合并提示词
                use_ft_overall = False
                try:
                    if self.model_overall_name and self._is_finetuned_model(self.model_overall_name):
                        use_ft_overall = True
                    elif self.model_summarize_name and self._is_finetuned_model(self.model_summarize_name):
                        use_ft_overall = True
                except Exception:
                    use_ft_overall = False
                system_prompt = "You output ONLY valid JSON. No explanations."
                if use_ft_overall:
                    user_prompt_template = (
                        "You are given multiple JSON candidates (one per paragraph). "
                        "Merge them into ONE JSON following this schema strictly (keys in this exact order):\n"
                        "1) reaction_summary.reaction_type\n"
                        "2) reactants[{name,role}]\n"
                        "3) products[{name,yield_optimal,unit}]\n"
                        "4) conditions[{type,value}]  (types: temperature,residence_time,flow_rate_total,pressure)\n"
                        "5) reactor{type,inner_diameter}\n"
                        "6) metrics{conversion,yield,selectivity,unit}\n\n"
                        "CRITICAL FILTERING RULES (Focus on Main Reaction):\n"
                        "- Identify the MAIN Target Reaction of this specific study/paper. \n"
                        "- EXCLUDE chemicals and data clearly labeled as \"previous work\", \"reported by others\", \"literature precedent\", or \"background\".\n"
                        "- Only aggregate reactants/products belonging to the current study's experiments.\n\n"
                        "MERGING RULES:\n"
                        "- Reactants & Products: Collect ALL unique substances involved in the MAIN reaction (reactants, solvents, catalysts). Merge duplicates by name.\n"
                        "- Metrics: Find the candidate with the HIGHEST yield (or conversion).\n"
                        "- Conditions: You MUST prioritize extracting the condition set (temp, time, etc.) FROM THE SAME paragraph/candidate that provided the highest yield. \n"
                        "- IF and ONLY IF the best yield paragraph lacks specific conditions (e.g., temperature/time is null), you MAY infer them from other paragraphs describing the SAME reaction setup or \"general procedure\".\n"
                        "- Reactor: choose the most informative object describing the main reactor.\n"
                        "- Do NOT invent new data. Output valid JSON only.\n\n"
                        "Use ONLY the candidates provided in Context.\n"
                    )
                    # 若候选过多，执行分批两段式合并
                    items_for_merge = combined_items
                    if len(json_items) > max_candidates:
                        partials = []
                        for i in range(0, len(json_items), chunk_size):
                            chunk = json_items[i:i+chunk_size]
                            partial = _merge_once(chunk, system_prompt, True, user_prompt_template)
                            if partial:
                                partials.append(partial)
                        # 第二阶段：合并分批结果
                        if partials:
                            items_for_merge = partials
                    # 一次合并（对预算内的候选或分批中间结果）
                    combined = "\n\n".join(_select_under_budget(items_for_merge))
                    user_prompt = user_prompt_template
                else:
                    user_prompt = (
                        "From the following JSON candidates, output ONE JSON that keeps the optimal condition set "
                        "(highest yield/conversion). Use null/empty arrays for unknowns. "
                        "Do NOT introduce chemicals/conditions/metrics not present in candidates. "
                        "If conflicting info exists, choose null. Metrics must be numeric when present. "
                        "Schema keys must be exactly: reaction_summary.reaction_type, reactants[{name,role}], "
                        "products[{name,yield_optimal,unit}], conditions[{type,value}], reactor{type,inner_diameter}, "
                        "metrics{conversion,yield,selectivity,unit}."
                    )
        else:
            # 回落：基于抽象/原文进行一次整体抽取
            col = 'abstract' if 'abstract' in df_abstract.columns else 'content'
            texts = [t for t in df_abstract[col].fillna("").tolist() if t.strip()]
            combined = "\n\n".join(texts)
            if len(combined) > char_budget:
                combined = combined[:char_budget]
            system_prompt = "You output ONLY valid JSON. No explanations."
            user_prompt = (
                "Extract ONE JSON summarizing the optimal condition set from the document text. "
                "Use null/empty arrays for unknowns. Do not invent chemicals or numbers not present in text. "
                "If conflicting info exists, choose null. Keep metrics numeric when present. "
                "Schema keys must be exactly: reaction_summary.reaction_type, reactants[{name,role}], "
                "products[{name,yield_optimal,unit}], conditions[{type,value}], reactor{type,inner_diameter}, "
                "metrics{conversion,yield,selectivity,unit}."
            )
        # 使用 overall 阶段模型与对应的 chat template（若为微调模型）
        prompt = self._get_chat_prompt(system_prompt, user_prompt, combined, stage='overall')
        raw = self._safe_generate(self._get_stage_model('overall'), prompt, max_tokens=overall_max_tokens, temp=0.0)
        # 去围栏并抽取最大花括号块
        raw = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"```\s*", "", raw)
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            raw = raw[s:e+1]
        cleaned = self._sanitize_json_text(raw)
        cleaned = self._keep_best_only(cleaned)
        
        # ---------------------------------------------------------------------
        # (已移除) 强力后处理：硬编码合并
        # 原因：为了响应“只合并主反应”的需求，我们现在完全依赖 Prompt 进行智能过滤，
        # 不再无脑合并所有候选，以免引入背景介绍中的噪音。
        # ---------------------------------------------------------------------
        
        try:
            final_obj = json.loads(cleaned)
        except:
            final_obj = {}

        # 自动修正：根据 role 字段调整归属
        try:
            obj = json.loads(cleaned)
            obj = self._ensure_schema_and_order(obj)
            cleaned = json.dumps(obj, ensure_ascii=False)
        except Exception:
            pass
        def _looks_invalid(text: str) -> bool:
            if not text or '...' in text or '…' in text:
                return True
            try:
                json.loads(text)
                return False
            except Exception:
                return True
        if _looks_invalid(cleaned):
            # 一次更严格重试：强调仅基于候选/文本且必须可解析
            strict_user = user_prompt + " STRICT MODE: Use only fields present in candidates/text; ensure valid, parseable JSON; no markdown or code fences."
            prompt2 = self._get_chat_prompt(system_prompt, strict_user, combined, stage='overall')
            raw2 = self._safe_generate(self._get_stage_model('overall'), prompt2, max_tokens=overall_max_tokens, temp=0.0)
            raw2 = re.sub(r"```json\s*", "", raw2, flags=re.IGNORECASE)
            raw2 = re.sub(r"```\s*", "", raw2)
            s2, e2 = raw2.find("{"), raw2.rfind("}")
            if s2 != -1 and e2 != -1 and e2 > s2:
                raw2 = raw2[s2:e2+1]
            cleaned2 = self._keep_best_only(self._sanitize_json_text(raw2))
            try:
                obj2 = json.loads(cleaned2)
                obj2 = self._ensure_schema_and_order(obj2)
                return json.dumps(obj2, ensure_ascii=False)
            except Exception:
                return raw2
        try:
            objf = json.loads(cleaned)
            objf = self._ensure_schema_and_order(objf)
            
            # 轻量“角色回收”：若某物质在候选中高频作为溶剂且在当前 products 无产率，则移回 reactants
            try:
                # 1. 统计候选中各物质作为 solvent 的次数
                solvent_counts = {}
                for item in json_items:
                    try:
                        c_obj = json.loads(item)
                        c_reactants = c_obj.get("reactants") or c_obj.get("reaction_summary", {}).get("reactants") or []
                        if isinstance(c_reactants, list):
                            for r in c_reactants:
                                if isinstance(r, dict):
                                    name = self._clean_chem_name(r.get("name"))
                                    role = str(r.get("role", "")).lower()
                                    if name and "solvent" in role:
                                        solvent_counts[name] = solvent_counts.get(name, 0) + 1
                    except Exception:
                        continue
                
                # 2. 检查 products 并迁移
                final_prods = objf.get("products") or []
                final_reacts = objf.get("reactants") or []
                new_prods = []
                dirty = False
                
                for p in final_prods:
                    if not isinstance(p, dict):
                        continue
                    p_name = self._clean_chem_name(p.get("name"))
                    p_yield = p.get("yield_optimal")
                    
                    # 触发条件：名字匹配溶剂高频词（>=2次）且无产率
                    if p_name and p_yield is None and solvent_counts.get(p_name, 0) >= 2:
                        # 迁移到 reactants
                        # 查重：reactants 里是否已存在
                        exists = any(self._clean_chem_name(r.get("name")) == p_name for r in final_reacts)
                        if not exists:
                            final_reacts.append({"name": p_name, "role": "solvent"})
                            dirty = True
                    else:
                        new_prods.append(p)
                
                if dirty:
                    objf["products"] = new_prods
                    objf["reactants"] = final_reacts
            except Exception:
                pass

            # 轻量条件补全：当最佳收益JSON缺某条件，且候选唯一一致时进行补全
            if os.getenv('FCPD_OVERALL_COND_FILL', '1') == '1':
                try:
                    # cond_pool 仅在 summarized 分支中构建
                    if 'cond_pool' in locals():
                        existing = objf.get("conditions") or []
                        # 构建类型到索引映射
                        idx_map = {}
                        for idx, c in enumerate(existing if isinstance(existing, list) else []):
                            if isinstance(c, dict) and c.get("type") in ("temperature","residence_time","flow_rate_total","pressure"):
                                idx_map[c.get("type")] = idx
                        for t in ("temperature","residence_time","flow_rate_total","pressure"):
                            pool = cond_pool.get(t, set())
                            if isinstance(pool, set) and len(pool) == 1:
                                val = list(pool)[0]
                                # 如果当前缺失则补全
                                if t in idx_map:
                                    if not existing[idx_map[t]].get("value"):
                                        existing[idx_map[t]]["value"] = val
                                else:
                                    if isinstance(existing, list):
                                        existing.append({"type": t, "value": val})
                        objf["conditions"] = existing
                except Exception:
                    pass
            return json.dumps(objf, ensure_ascii=False)
        except Exception:
            return raw

    def _extract_influence_candidates(self, df: pd.DataFrame) -> list:
        import re, os
        content_col = 'abstract' if 'abstract' in df.columns else 'content'
        candidates = []
        topK = int(os.getenv('FCPD_IMPACT_TOPK', '12'))
        
        causal_verbs = [
            'increase', 'decrease', 'improve', 'enhance', 'reduce', 'affect',
            'influence', 'impact', 'promote', 'inhibit', 'facilitate', 'optimize',
            'control', 'determine', 'depend', 'vary', 'change', 'modulate'
        ]
        metric_keywords = [
            'conversion', 'yield', 'selectivity', 'purity', 'efficiency',
            'product distribution', 'heat transfer', 'mixing'
        ]
        
        for idx, row in df.iterrows():
            paragraph = str(row[content_col])
            if not paragraph or len(paragraph.strip()) < 30:
                continue
            score = 0
            para_lower = paragraph.lower()
            
            # Results/Discussion sections
            is_results = any(sec in para_lower for sec in ['result', 'discussion', 'finding', 'observation'])
            if is_results:
                has_metric = any(kw in para_lower for kw in metric_keywords)
                if has_metric:
                    score += 25
                else:
                    score += 10
            
            # 因果动词
            causal_count = sum(1 for verb in causal_verbs if verb in para_lower)
            if causal_count > 0:
                has_metric = any(m in para_lower for m in metric_keywords)
                if has_metric:
                    score += 8 * min(causal_count, 3)
                else:
                    score += 3 * min(causal_count, 2)
            
            # 定量数据
            quant_matches = re.findall(r'\b\d+\.?\d*\s*(?:%|K|°C|°F|mL|L|min|h|MPa|bar|M|mol|g)\b', paragraph)
            if len(quant_matches) >= 2:
                score += 10
            
            if score >= 3:  # 进一步放宽阈值，确保有足够候选
                candidates.append({'index': idx, 'text': paragraph, 'score': score})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:topK]

    def _to_markdown_impact(self, items: list) -> str:
        if not items:
            return "| Factor | Metric | Direction |\n|--------|--------|-----------|\n| None | - | - |"
        lines = [
            "| Factor | Metric | Direction |",
            "|--------|--------|-----------|",
        ]
        for it in items:
            factor = (it.get('factor','-') or '-').replace('|','\\|')
            metric = (it.get('metric','-') or '-').replace('|','\\|')
            direction = (it.get('direction','') or '').lower()
            if direction not in ['increase','decrease','unchanged']:
                direction = '-'
            lines.append(f"| {factor} | {metric} | {direction} |")
        return "\n".join(lines)

    def extract_influence_factors_with_llm(self, df: pd.DataFrame) -> str:
        cands = self._extract_influence_candidates(df)
        if os.getenv("FCPD_DEBUG", "0") == "1":
            print(f"  🐛 Impact候选段落数: {len(cands)}")
        if not cands:
            return self._to_markdown_impact([])
        joined = "\n\n".join(c['text'][:800] for c in cands[:min(len(cands), 15)])
        if os.getenv("FCPD_DEBUG", "0") == "1":
            print(f"  🐛 拼接文本长度: {len(joined)} 字符")

        # Few-shot 简化但完整的案例
        system_prompt = "Extract cause-effect relationships from chemical paragraphs."
        user_prompt = (
            "Example paragraph:\n"
            "Residence time is an important parameter. A longer residence time will result in "
            "a higher conversion of TFMB. The product selectivity nearly remains unchanged.\n\n"
            "Example output:\n"
            "residence_time | conversion of TFMB | increase\n"
            "residence_time | product selectivity | unchanged\n\n"
            "Extract Factor-Metric-Direction from the paragraphs below.\n"
            "Format: Factor | Metric | Direction (one per line, no table headers)\n"
            "Direction: increase, decrease, or unchanged\n\n"
            "Paragraphs:\n"
        )
        prompt = self._create_prompt(system_prompt, user_prompt, joined)
        # 影响因素：改为使用 impact 专用模型（若已配置），否则回退到 summarize 阶段模型
        model = self._get_stage_model('impact')
        raw = self._safe_generate(model, prompt, max_tokens=700, temp=0.3)
        if os.getenv("FCPD_DEBUG", "0") == "1":
            print(f"  🐛 LLM原始输出前500字符: {raw[:500]}")

        # 解析为三列（兼容2列和3列格式）
        items = []
        for line in raw.splitlines():
            if '|' not in line:
                continue
            # 跳过表头行
            low = line.lower()
            if 'factor' in low and 'metric' in low:
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 2:  # 改为至少2列即可
                continue
            # 跳过空行
            if not parts[0] or not parts[1] or parts[0] == '-':
                continue
            
            # 兼容2列和3列格式
            if len(parts) == 2:
                # 两列格式：Factor | Metric（从Metric中提取Direction）
                factor = parts[0]
                metric_with_dir = parts[1].lower()
                # 尝试从metric字段提取方向
                if 'increase' in metric_with_dir or 'higher' in metric_with_dir:
                    direction = 'increase'
                    metric = parts[1].split()[0]  # 取第一个词作为metric
                elif 'decrease' in metric_with_dir or 'lower' in metric_with_dir:
                    direction = 'decrease'
                    metric = parts[1].split()[0]
                elif 'unchange' in metric_with_dir:
                    direction = 'unchanged'
                    metric = parts[1].split()[0]
                else:
                    direction = '-'
                    metric = parts[1]
            else:
                # 三列格式：Factor | Metric | Direction
                factor = parts[0]
                metric = parts[1]
                direction = parts[2].lower()
                if 'increase' in direction or 'higher' in direction or 'improve' in direction or 'enhance' in direction:
                    direction = 'increase'
                elif 'decrease' in direction or 'lower' in direction or 'reduce' in direction or 'inhibit' in direction:
                    direction = 'decrease'
                elif 'unchange' in direction or 'unchanged' in direction or 'no effect' in direction:
                    direction = 'unchanged'
                else:
                    direction = ''
            
            items.append({'factor': factor, 'metric': metric, 'direction': direction})
        return self._to_markdown_impact(items)

    def process_text_file_comprehensive(self, file_path: str, mode: str = 'comprehensive') -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        segs, cur = [], []
        for line in lines:
            if line.strip():
                cur.append(line.strip())
            else:
                if cur:
                    segs.append(' '.join(cur))
                    cur = []
        if cur:
            segs.append(' '.join(cur))
        df = pd.DataFrame(segs, columns=['content'])

        outputs: Dict[str, Any] = {}
        if mode in ['filter', 'comprehensive']:
            df_filtered = self.filter_content(df)
            filter_file = file_path.replace('.txt', '_Filtered.txt')
            self.save_df_to_text(df_filtered, filter_file)
            outputs['filter'] = filter_file
        if mode in ['abstract', 'comprehensive']:
            # 优化：若不需要 Impact 或 Impact 源非 abstract，且已有 filtered，则跳过 abstract 节省时间
            skip_abstract = False
            if mode == 'comprehensive':
                run_impact = os.getenv('FCPD_RUN_IMPACT', '1') == '1'
                impact_src = os.getenv('FCPD_IMPACT_SOURCE', 'summarized')
                # 如果 Impact 关闭，或者 Impact 开启但源不是 abstract，则不需要跑 Abstract
                if (not run_impact) or (impact_src != 'abstract'):
                    skip_abstract = True
                    if os.getenv("FCPD_DEBUG", "0") == "1":
                        print("  ⏩ 跳过 Abstract 步骤 (优化)")

            if not skip_abstract:
                df_abstract = self.abstract_text(df_filtered if 'df_filtered' in locals() else df)
                abstract_file = file_path.replace('.txt', '_Abstract.txt')
                self.save_df_to_text(df_abstract, abstract_file, 'abstract')
                outputs['abstract'] = abstract_file
        if mode in ['summarize', 'comprehensive']:
            # 优先使用过滤后的原文段落（content）进行结构化抽取；若无则退到 abstract，再退到原始 df
            df_input = (df_filtered if 'df_filtered' in locals() else (df_abstract if 'df_abstract' in locals() else df))
            # 抽取前打包（可开关）：将相邻段落按顺序合并到字符预算范围，最大化LLM上下文利用
            try:
                # 智能相邻打包（默认开启）：仅对疑似“有结果但缺条件”的行，与相邻±1行拼接（长度≤1500）
                if os.getenv('FCPD_SUM_SMART_PACK', '1') == '1':
                    smart_max = int(os.getenv('FCPD_SUM_SMART_MAX_CHARS', '1500'))
                    smart_win = int(os.getenv('FCPD_SUM_SMART_WINDOW', '1'))
                    use_col = 'content' if 'content' in df_input.columns else ('abstract' if 'abstract' in df_input.columns else None)
                    if use_col:
                        df_input = self._smart_pack_rows(df_input, use_col, max_chars=smart_max, window=smart_win)
                if os.getenv('FCPD_SUM_PACK', '0') == '1':
                    min_chars = int(os.getenv('FCPD_SUM_PACK_MIN_CHARS', '2000'))
                    max_chars = int(os.getenv('FCPD_SUM_PACK_MAX_CHARS', '10000'))
                    use_col = 'content' if 'content' in df_input.columns else ('abstract' if 'abstract' in df_input.columns else None)
                    if use_col:
                        df_input = self._pack_adjacent_rows(df_input, use_col, min_chars=min_chars, max_chars=max_chars)
            except Exception:
                pass
            df_summarized = self.summarize_parameters(df_input)
            summarize_file = file_path.replace('.txt', '_Summarized.txt')
            self.save_df_to_text(df_summarized, summarize_file, 'summarized')
            outputs['summarized'] = summarize_file
            # Overall（基于抽象优先）
            # 优先使用段落级JSON结果；若无则退回到抽象，再退回原始输入
            overall_input = df_summarized if 'df_summarized' in locals() else (df_abstract if 'df_abstract' in locals() else df_input)
            overall_json = self.summarize_document_overall(overall_input)
            overall_file = file_path.replace('.txt', '_Overall.txt')
            with open(overall_file, 'w', encoding='utf-8') as f:
                f.write(overall_json)
            outputs['summarized_overall'] = overall_file

            # 影响因素（简化三列）
            # 影响因素（简化三列）
            try:
                if os.getenv('FCPD_RUN_IMPACT', '1') == '1':
                    src = os.getenv('FCPD_IMPACT_SOURCE', 'summarized')  # summarized | abstract | filtered | content
                    if src == 'abstract' and 'df_abstract' in locals():
                        impact_input = df_abstract
                    elif src == 'filtered' and 'df_filtered' in locals():
                        impact_input = df_filtered
                    elif src == 'summarized' and 'df_summarized' in locals():
                        impact_input = df_summarized
                    else:
                        impact_input = overall_input  # 默认与当前一致（df_summarized优先）

                    impact_md = self.extract_influence_factors_with_llm(impact_input)
                    impact_file = file_path.replace('.txt', '_Impact_Analysis.txt')
                    with open(impact_file, 'w', encoding='utf-8') as f:
                        f.write("# Influence Factor Summary\n\n")
                        f.write(impact_md)
                    outputs['impact_analysis'] = impact_file
            except Exception:
                pass
        return outputs

    def _get_stage_model(self, stage: str) -> GPT4All:
        """按阶段返回对应模型；若未配置分阶段，则回退到单模型。"""
        if self.model_single is not None:
            return self.model_single
        name = None
        if stage == 'filter':
            name = self.model_filter_name
        elif stage == 'abstract':
            name = self.model_abstract_name
        elif stage == 'summarize':
            name = self.model_summarize_name
        elif stage == 'overall':
            name = self.model_overall_name
        elif stage == 'impact':
            name = self.model_impact_name
        if not name:
            # 兜底：使用摘要模型或任一可用
            name = self.model_abstract_name or self.model_summarize_name or self.model_overall_name or self.model_filter_name
        # 统一：尝试将上下文窗口提升到 4096，避免默认 2048 导致的 Context Full 报错
        # 这对未微调的 Llama2/3 模型通常是安全的且必要的
        kwargs = dict(model_path=self.model_path, allow_download=False)
        kwargs['n_ctx'] = 4096
        return GPT4All(name, **kwargs)