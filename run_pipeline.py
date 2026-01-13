
import os
import glob
import shutil
import yaml
from core.text_utils import extract_text_from_pdf, write_text
from core.embedding import run_embedding_selection
from core.local_pipeline import LocalPipeline

# 1. Environment Configuration
os.environ['FCPD_TOP_N'] = '20' # Notebook used 15 or 20, let's use 20 as in notebook comments
os.environ['FCPD_EMB_KEEP_ORDER'] = '1'
os.environ['FCPD_EMB_EXPAND'] = '1'
os.environ['FCPD_EMB_MIN_CHARS'] = '300'
os.environ['FCPD_EMB_MAX_CHARS'] = '1000'
os.environ['FCPD_SUM_PACK'] = '1'
os.environ['FCPD_SUM_PACK_MIN_CHARS'] = '500'
os.environ['FCPD_SUM_PACK_MAX_CHARS'] = '1500'
os.environ['FCPD_FILTER_TITLE_WINDOW'] = '100'
os.environ['FCPD_RUN_IMPACT'] = '0'
os.environ['FCPD_OVERALL_MAX_TOKENS'] = '4096'
os.environ['FCPD_FORCE_FINETUNED'] = '1'
os.environ['FCPD_DEBUG'] = '1'

# 2. Config & Engine
with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

lc = cfg.get('local_model', {})
engine = LocalPipeline(
    model_name=None,
    model_path=lc.get('path', '/Users/zhaowenyuan/Projects/FCPDExtractor/models'),
    filter_model=lc.get('filter'),
    abstract_model=lc.get('abstract'),
    summarize_model=lc.get('summarize'),
    overall_model=lc.get('overall'),
    impact_model=lc.get('impact'),
    finetuned_trigger_name=lc.get('finetuned_trigger_name', 'My_Finetuned_Model'),
)

# 3. Process Files
input_dir = 'evaluation/raw papers'
output_base = 'evaluation/result/local llm finetuned'
os.makedirs(output_base, exist_ok=True)

pdf_files = sorted(glob.glob(os.path.join(input_dir, '1.pdf')))
print(f"📄 Found {len(pdf_files)} PDF files.")

for i, pdf_path in enumerate(pdf_files, 1):
    print(f"Processing {i}/{len(pdf_files)}: {pdf_path}")
    
    # Prep Output Dir
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    specific_out_dir = os.path.join(output_base, base_name)
    os.makedirs(specific_out_dir, exist_ok=True)
    
    # 1. PDF -> TXT
    txt_path = os.path.join(specific_out_dir, f"{base_name}.txt")
    if not os.path.exists(txt_path):
        text = extract_text_from_pdf(pdf_path)
        write_text(text, txt_path)
    
    # 2. Embedding
    try:
        if os.path.getsize(txt_path) < 100:
             print("Text too short, skipping.")
             continue
        emb_path = run_embedding_selection(txt_path, top_n=int(os.getenv('FCPD_TOP_N')))
        # Copy embedding file to output (ignore if same file)
        dst_emb = os.path.join(specific_out_dir, os.path.basename(emb_path))
        if os.path.abspath(emb_path) != os.path.abspath(dst_emb):
            shutil.copy2(emb_path, dst_emb)
        
        # 3. Pipeline Run
        res = engine.process_text_file_comprehensive(emb_path, mode='comprehensive')
        
        # Save results
        for k, v in res.items():
            if v and os.path.exists(v):
                shutil.copy2(v, os.path.join(specific_out_dir, os.path.basename(v)))
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()

print("✅ Pipeline execution complete.")
