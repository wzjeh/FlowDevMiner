import os
import sys
sys.path.append(os.path.dirname(__file__))
from evaluate_results import list_document_ids, METHODS, GROUND_TRUTH_DIR

print(f"GT Dir: {GROUND_TRUTH_DIR}")
print(f"Exists? {os.path.exists(GROUND_TRUTH_DIR)}")
if os.path.exists(GROUND_TRUTH_DIR):
    print(f"Files: {os.listdir(GROUND_TRUTH_DIR)}")

doc_ids = list_document_ids()
print(f"Doc IDs found: {len(doc_ids)}")
print(f"Sample: {doc_ids[:5]}")

print("Methods:")
for k, v in METHODS.items():
    print(f"  {k}: {v['dir']} (Exists? {os.path.exists(v['dir'])})")








