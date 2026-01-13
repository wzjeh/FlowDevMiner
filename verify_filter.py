
import pandas as pd
import re

# Mocking the LocalPipeline class partially to test filter_content logic
class MockLocalPipeline:
    def _create_prompt(self, s, u, c): return ""
    def _get_stage_model(self, s): return None
    def _safe_generate(self, m, p, max_tokens, temp): return "No" # Default to No to prove whitelist works

    def filter_content(self, df: pd.DataFrame) -> pd.DataFrame:
        user_question = "..."
        system_prompt = "..."
        results = []
        for _, row in df.iterrows():
            content = str(row['content']) # _clip omitted
            kw_strong = [
                "abstract", "conclusion", "experimental", "experimental section",
                "materials and methods", "methods", "procedure", "general procedure",
                "typical procedure", "optimization", "best conditions", "results and discussion"
            ]
            content_lower = content.lower()
            
            # 1) Title Window (simplified)
            prefix = content_lower[:80]
            if any(k in prefix for k in kw_strong):
                results.append('Yes')
                continue

            # 2) Tech keywords
            if "residence time" in content_lower:
                results.append('Yes')
                continue
            
            if "%" in content and ("yield" in content_lower or "conversion" in content_lower):
                 if re.search(r"\d+\s*%", content):
                    results.append('Yes')
                    continue

            # --- NEW: Enhanced Filter for Reactor & Conditions ---
            # 捕获反应器特定信息 (inner diameter, microreactor, etc.)
            if any(k in content_lower for k in ["inner diameter", "internal diameter", "microreactor", "micro-reactor", "flow rate", "residence time", "back pressure"]):
                results.append('Yes')
                continue
            
            # 捕获带有单位的实验条件 (temp, pressure, flow)
            if re.search(r"(?:\d+\s*(?:°C|°F|K|bar|MPa|psi|mL/min|µL/min|ul/min))", content, re.IGNORECASE):
                results.append('Yes')
                continue
            # -----------------------------------------------------

            # 3) LLM fallback (mocked to No)
            results.append('No')
            
        out = df.copy()
        out['classification'] = results
        # Print debug info
        for i, res in enumerate(results):
            print(f"DEBUG: Row {i} classified as {res}")
        return out[out['classification'] == 'Yes'].copy()

def test_filter():
    data = [
        "The reaction was performed in a Teflon coil reactor with 0.8 mm internal diameter.",
        "A microreactor was employed for the synthesis.",
        "The flow rate was maintained at 1.5 mL/min using a syringe pump.",
        "The system was pressurized to 5 bar using a BPR.",
        "This is a general introduction about flow chemistry history. (Should be No)",
        "We discuss the future of the field. (Should be No)"
    ]
    df = pd.DataFrame({'content': data})
    pipeline = MockLocalPipeline()
    out = pipeline.filter_content(df)
    
    print("Results:")
    for _, row in out.iterrows():
        print(f"[Yes] {row['content']}")
        
    print(f"\nTotal kept: {len(out)} / {len(data)}")
    
    # Assertions
    expected_yes_count = 4
    if len(out) == expected_yes_count:
        print("✅ Verification Passed!")
    else:
        print(f"❌ Verification Failed! Expected {expected_yes_count}, got {len(out)}")

if __name__ == "__main__":
    test_filter()
