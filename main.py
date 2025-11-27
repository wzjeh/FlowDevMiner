import argparse
import glob
import os
import sys
import yaml
import shutil
from typing import List

from core.text_utils import extract_text_from_pdf, write_text
from core.embedding import run_embedding_selection
from core.models.gemini_llm import GeminiLLM
from core.models.qwen_llm import QwenLLM
from core.processor import UnifiedTextProcessor
from core.local_pipeline import LocalPipeline
from evaluation.metrics import calculate_metrics


def iter_inputs(input_dir: str, limit: int | None) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, '*.*')))
    files = [f for f in files if f.lower().endswith(('.txt', '.pdf'))]
    return files[:limit] if limit else files


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def ensure_txt(fp: str, output_dir: str) -> str:
    if fp.lower().endswith('.txt'):
        return fp
    base = os.path.splitext(os.path.basename(fp))[0]
    out_dir = os.path.join(output_dir, base)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"{base}.txt")
    text = extract_text_from_pdf(fp)
    write_text(text, out_txt)
    return out_txt


def set_default_env_vars(args) -> None:
    """Set default environment variables to match optimized notebook settings."""
    defaults = {
        'FCPD_TOP_N': '15',
        'FCPD_EMB_KEEP_ORDER': '1',
        'FCPD_EMB_EXPAND': '1',
        'FCPD_EMB_MIN_CHARS': '300',
        'FCPD_EMB_MAX_CHARS': '1000',
        'FCPD_SUM_PACK': '1',
        'FCPD_SUM_PACK_MIN_CHARS': '500',
        'FCPD_SUM_PACK_MAX_CHARS': '1500',
        'FCPD_FILTER_TITLE_WINDOW': '100',
        'FCPD_RUN_IMPACT': '0', # Default off for speed
        'FCPD_IMPACT_SOURCE': 'abstract',
        'FCPD_OVERALL_MAX_CAND': '20',
        'FCPD_OVERALL_CHAR_BUDGET': '7000',
        'FCPD_OVERALL_CHUNK_SIZE': '10',
        'FCPD_OVERALL_MAX_TOKENS': '4096', # Optimized for finetuned
        'FCPD_OVERALL_COND_FILL': '1',
        'FCPD_EMB_UNIT_BOOST': '0.12',
        'FCPD_SUM_SMART_PACK': '1'
    }
    
    for k, v in defaults.items():
        if k not in os.environ:
            os.environ[k] = v
            
    # Force finetuned if requested
    if args.force_finetuned:
        os.environ['FCPD_FORCE_FINETUNED'] = '1'


def main() -> None:
    parser = argparse.ArgumentParser(description="FCPDExtractor - New Architecture CLI")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input_dir', type=str, help='Input directory (.txt or .pdf)')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--engine', type=str, choices=['qwen','gemini','local'])
    parser.add_argument('--mode', type=str, default='comprehensive', choices=['filter','abstract','summarize','comprehensive','evaluate'])
    parser.add_argument('--force-finetuned', action='store_true', help='Force usage of finetuned model logic')
    args = parser.parse_args()

    # 1. Load config and set environment variables
    cfg = load_config(args.config)
    set_default_env_vars(args)
    
    if os.getenv('FCPD_DEBUG') == '1' or args.force_finetuned:
        print("🔧 Environment Config:")
        print(f"  FCPD_FORCE_FINETUNED: {os.getenv('FCPD_FORCE_FINETUNED', '0')}")
        print(f"  FCPD_TOP_N: {os.getenv('FCPD_TOP_N')}")

    paths = cfg.get('paths', {})
    input_dir = args.input_dir or paths.get('papers_dir', 'data/papers')
    output_dir = args.output_dir or paths.get('output_dir', 'data')
    os.makedirs(output_dir, exist_ok=True)

    engine_choice = args.engine or cfg.get('engine', 'qwen')
    if engine_choice == 'local':
        local_cfg = cfg.get('local_model', {})
        engine = LocalPipeline(
            model_name=None,  # Use staged config
            model_path=local_cfg.get('path', 'models/'),
            filter_model=local_cfg.get('filter'),
            abstract_model=local_cfg.get('abstract'),
            summarize_model=local_cfg.get('summarize'),
            overall_model=local_cfg.get('overall'),
            impact_model=local_cfg.get('impact'),
            finetuned_trigger_name=local_cfg.get('finetuned_trigger_name', 'My_Finetuned_Model'),
        )
    elif engine_choice == 'qwen':
        qcfg = cfg.get('qwen_api', {})
        llm = QwenLLM(api_key_env_var=qcfg.get('api_key_env_var', 'QWEN_API_KEY'), 
                      model_name=qcfg.get('model_name', 'qwen-plus'))
        engine = UnifiedTextProcessor(llm)
    else:
        # Gemini
        gcfg = cfg.get('gemini_api', {})
        llm = GeminiLLM(api_key_env_var=gcfg.get('api_key_env_var', 'GOOGLE_API_KEY'), 
                        model_name=gcfg.get('model_name', 'gemini-1.5-flash'))
        engine = UnifiedTextProcessor(llm)

    if args.mode == 'evaluate':
        gt_dir = paths.get('ground_truth_dir', 'data/ground_truth')
        # Iterate over *_Overall.txt in output dir and compare with ground_truth
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith('_Overall.txt'):
                    base = f.replace('_Overall.txt', '')
                    pred = os.path.join(root, f)
                    gt_json = os.path.join(gt_dir, f'{base}.json')
                    if os.path.exists(gt_json):
                        m = calculate_metrics(gt_json, pred)
                        print(f"{base}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        return

    inputs = iter_inputs(input_dir, args.limit)
    if not inputs:
        print('No input files found (.txt/.pdf).')
        sys.exit(1)

    for fp in inputs:
        fp_txt = ensure_txt(fp, output_dir)
        # Embedding selection (Pre-filtering)
        # Note: process_text_file_comprehensive logic handles its own flow, 
        # but passing the embedding filtered text file ensures we focus on relevant parts.
        emb_txt = run_embedding_selection(fp_txt, top_n=int(os.getenv('FCPD_TOP_N', '10')))

        # Use embedding result if available, otherwise full text
        use_txt = emb_txt if os.path.exists(emb_txt) else fp_txt

        res = engine.process_text_file_comprehensive(use_txt, mode=args.mode)
        for k, v in res.items():
            print(f"✓ {k}: {v}")

        # Archive to engine specific directory
        base_name = os.path.splitext(os.path.basename(fp_txt))[0]
        subdir = 'gemini' if engine_choice == 'gemini' else ('qwen' if engine_choice == 'qwen' else 'local llm')
        dest_dir = os.path.join(output_dir, subdir, base_name)
        os.makedirs(dest_dir, exist_ok=True)

        # Source text
        try:
            shutil.copy2(fp_txt, os.path.join(dest_dir, os.path.basename(fp_txt)))
        except Exception:
            pass

        # Results
        for v in res.values():
            try:
                if v and os.path.exists(v):
                    shutil.copy2(v, os.path.join(dest_dir, os.path.basename(v)))
            except Exception:
                pass

        # Embedding file
        try:
            if emb_txt and os.path.exists(emb_txt):
                shutil.copy2(emb_txt, os.path.join(dest_dir, os.path.basename(emb_txt)))
        except Exception:
            pass

    # Clean up legacy output folder if exists
    legacy_output = os.path.join(output_dir, 'output')
    if os.path.isdir(legacy_output):
        try:
            shutil.rmtree(legacy_output)
            print('🧹 Cleaned legacy data/output')
        except Exception as e:
            print(f'⚠️ Failed to clean data/output: {e}')

    print('✅ Done')


if __name__ == '__main__':
    main()
