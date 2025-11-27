#!/usr/bin/env python3
"""
测试 Overall 总结功能的改进
"""
import re

# 测试原来的坏输出
bad_json = """{
    "reaction_summary": {
        "reaction_type": "nitration",
        "conditions": [
            {"type": "temperature", "value": "273 K"},
            {"type": "residence_time", "value": "..."}, // Not specified in the abstracts
            {"type": "pressure", "value": "..."} // Not specified in the abstracts
        ],
        "reactor": {
            "type": "microchannel reactor",
            "inner_diameter": "..."}, // Not specified in the abstracts
    }
} 

Note: The residence time, pressure and inner diameter of reactor are not mentioned.
Best regards, [Your Name]  ###

The final answer is: { "reaction_summary": {"""

print("=" * 70)
print("🧪 测试 JSON 清洗功能")
print("=" * 70)

def sanitize_json_text(text: str) -> str:
    """新的清洗函数"""
    s = text or ""
    
    # 1. 找到第一个 { 和最后一个 }，只保留这之间的内容
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        s = s[first_brace:last_brace+1]
    
    # 2. 去除 // 行内注释
    s = re.sub(r"//.*?(?=\n|$)", "", s)
    
    # 3. 去除 /* ... */ 块注释
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    
    # 4. 修复未加引号的键
    s = re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*', r'\1"\2": ', s)
    
    # 5. 删除对象/数组中的尾随逗号
    s = re.sub(r",\s*(\}|\])", r"\1", s)
    
    # 6. 替换 ... 占位符为 null
    s = re.sub(r':\s*"\.\.\."\s*([,\}])', r': null\1', s)
    s = re.sub(r':\s*\.\.\.(\s*[,\}])', r': null\1', s)
    
    # 7. 修复可能的格式问题：确保数字不带引号
    s = re.sub(r':\s*"(\d+\.?\d*)"\s*([,\}])', r': \1\2', s)
    
    # 8. 去除多余的空白和换行
    s = s.strip()
    
    return s

print("\n【原始输出】(有问题):")
print("-" * 70)
print(bad_json[:200] + "...")

print("\n【清洗后】:")
print("-" * 70)
cleaned = sanitize_json_text(bad_json)
print(cleaned)

print("\n【验证 JSON 格式】:")
print("-" * 70)
import json
try:
    parsed = json.loads(cleaned)
    print("✅ JSON 格式正确！")
    print("\n解析后的数据:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"❌ JSON 格式错误: {e}")

print("\n" + "=" * 70)
print("📝 改进总结")
print("=" * 70)
print("""
新的改进：
1. ✅ 自动移除 "Note:", "Best regards" 等非 JSON 文本
2. ✅ 去除 // 和 /* */ 注释
3. ✅ 将 "..." 占位符替换为 null
4. ✅ 修复未加引号的键
5. ✅ 删除尾随逗号
6. ✅ 只保留最外层 { } 之间的内容

新的 Prompt 改进：
1. ✅ 明确禁止输出注释和额外文本
2. ✅ 要求使用 null 而不是 "..."
3. ✅ 增加 max_tokens 从 500 到 800
4. ✅ 增加输入文本长度从 8000 到 12000
5. ✅ 使用 "START JSON OUTPUT NOW:" 引导直接输出
""")


