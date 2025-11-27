import json
import os
import glob
import re
import csv
import math

def parse_value_unit(value_str):
    if not value_str or not isinstance(value_str, str):
        return None, None
    # Pre-clean
    value_str = value_str.lower().strip()
    value_str = value_str.replace("approx.", "").replace("ca.", "")
    value_str = value_str.replace("−", "-") # Replace minus sign
    
    # Try to handle addition: "9.22+4.94 ml/min"
    # Match pattern: num + num unit
    # Be careful with scientific notation which might use +
    # Simple check: if + is surrounded by digits or spaces
    if '+' in value_str:
        # Check if it looks like an addition of two numbers
        add_match = re.match(r"([\d\.]+)\s*\+\s*([\d\.]+)\s*(.*)", value_str)
        if add_match:
            try:
                val1 = float(add_match.group(1))
                val2 = float(add_match.group(2))
                unit = add_match.group(3)
                return val1 + val2, unit.strip()
            except:
                pass

    # Standard match
    # Capture everything after the number as unit
    match = re.match(r"([\d\.]+)\s*(.*)", value_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2).strip()
        # Clean unit
        unit = unit.replace(" ", "")
        return val, unit
    return None, None

def normalize_diameter(value_str):
    val, unit = parse_value_unit(value_str)
    if val is None:
        return None
    
    # Handle known junk or variants
    if unit in ['mm', 'millimeters']:
        return val
    elif unit in ['um', 'µm', 'micrometers', 'microns', 'μm', 'ﾂｵm', 'ﾎｼm']:
        return val / 1000.0
    elif unit in ['m', 'meters']:
        return val * 1000.0
    elif unit in ['cm', 'centimeters']:
        return val * 10.0
    elif unit in ['in', 'inch', 'inches']:
        return val * 25.4
    else:
        # print(f"Unknown diameter unit: {unit}")
        return None

def normalize_flow_rate(value_str):
    val, unit = parse_value_unit(value_str)
    if val is None:
        return None
    
    # Normalize to mL/min
    if unit in ['ml/min', 'milliliters/minute', 'mlmin-1', 'ml/min-1']:
        return val
    elif unit in ['ml/h', 'ml/hr', 'milliliters/hour', 'mlh-1']:
        return val / 60.0
    elif unit in ['ul/min', 'µl/min', 'microliters/minute', 'μl/min', 'ﾂｵl/min', 'ﾎｼl/min', 'ulmin-1']:
        return val / 1000.0
    elif unit in ['l/min', 'liters/minute', 'lmin-1']:
        return val * 1000.0
    elif unit in ['l/h', 'l/hr', 'lh-1']:
        return val * 1000.0 / 60.0
    else:
        # print(f"Unknown flow rate unit: {unit}")
        return None

def normalize_pressure(value_str):
    val, unit = parse_value_unit(value_str)
    if val is None:
        return None
    
    # Normalize to bar
    if unit in ['bar']:
        return val
    elif unit in ['mpa']:
        return val * 10.0
    elif unit in ['kpa']:
        return val * 0.01
    elif unit in ['psi']:
        return val * 0.0689476
    elif unit in ['atm']:
        return val * 1.01325
    elif unit in ['torr', 'mmhg']:
        return val * 0.00133322
    else:
        # print(f"Unknown pressure unit: {unit}")
        return None

def categorize_reaction(reaction_type_raw):
    if not reaction_type_raw:
        return "Others"
    r = reaction_type_raw.lower()
    
    if "nitration" in r:
        return "Nitration"
    elif "hydrogenation" in r or "hydrogenolysis" in r or "hydrogenaton" in r or "hydrodechlorination" in r:
        return "Hydrogenation"
    elif "oxidation" in r:
        return "Oxidation"
    elif "polymerization" in r:
        return "Polymerization"
    elif "coupling" in r:
        return "Cross-coupling"
    elif "halogenation" in r or "chlorination" in r or "fluorination" in r or "bromination" in r:
        return "Halogenation"
    else:
        return "Others"

def categorize_reactor(reactor_type_raw):
    if not reactor_type_raw:
        return "Others"
    r = reactor_type_raw.lower()
    
    # 2. Packed Bed/Monolith
    # Prioritize 'packed' to catch 'packed tubular' into this group
    if "packed" in r or "monolith" in r:
        return "Packed Bed/Monolith"
    
    # 1. Capillary/Coil Reactor
    if "capillary" in r or "coil" in r or "tubing" in r or "loop" in r:
        return "Capillary/Coil Reactor"
    
    # 4. Microchannel/Chip Reactor
    # 'microreactor', 'microchannel' are very common, need to ensure they don't swallow others if intended
    if "microreactor" in r or "microchannel" in r or "chip" in r or "y-shaped" in r or "heart" in r or "membrane" in r or "mixer" in r:
        return "Microchannel/Chip Reactor"
        
    # 3. Tubular Reactor
    # Check this after 'packed' to ensure 'packed tubular' goes to group 2
    if "tubular" in r or "mesoscale" in r or "open channel" in r or "flow reactor" in r:
        return "Tubular Reactor"
        
    return "Others"

def main():
    # Use multiple directories
    ground_truth_dir = 'finetune/ground_truth'
    qwen_dir = 'finetune/qwen 直接提取所有，可用于abstract2impact微调'
    test_dir = 'evaluation/result/ground truth' # New test set directory
    
    # Use glob to find files
    files_gt = glob.glob(os.path.join(ground_truth_dir, '*_annotated.json'))
    files_qwen = glob.glob(os.path.join(qwen_dir, '*_reaction.json'))
    files_test = glob.glob(os.path.join(test_dir, '*_annotated.json'))
    
    # Combine all files, but track source for scatter plot
    # Create list of tuples: (file_path, is_test_set)
    all_files = []
    for f in files_gt:
        all_files.append((f, False))
    for f in files_qwen:
        all_files.append((f, False))
    for f in files_test:
        all_files.append((f, True)) # Mark as test set
    
    sankey_data = {} # (reaction_category, reactor_category) -> count
    scatter_data = [] # list of dicts

    print(f"Found {len(all_files)} total JSON files ({len(files_gt)} GT, {len(files_qwen)} Qwen, {len(files_test)} Test).")

    for fpath, is_test in all_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            reaction_type_raw = data.get('reaction_type', 'Unknown')
            reactor = data.get('reactor', {})
            reactor_type_raw = reactor.get('type', 'Unknown')
            
            reaction_cat = categorize_reaction(reaction_type_raw)
            reactor_cat = categorize_reactor(reactor_type_raw)
            
            # For Sankey (include all data, even test set as per user request "统计进去桑葚图数据")
            if reaction_cat and reactor_cat:
                key = (reaction_cat, reactor_cat)
                sankey_data[key] = sankey_data.get(key, 0) + 1
            
            # For Scatter
            inner_diameter_str = reactor.get('inner_diameter')
            inner_diameter_mm = normalize_diameter(inner_diameter_str)
            
            conditions = data.get('conditions', [])
            flow_rate_ml_min = None
            pressure_bar = None
            
            for cond in conditions:
                c_type = cond.get('type')
                c_val = cond.get('value')
                if c_type == 'flow_rate_total':
                    flow_rate_ml_min = normalize_flow_rate(c_val)
                elif c_type == 'pressure':
                    pressure_bar = normalize_pressure(c_val)
            
            if inner_diameter_mm is not None and flow_rate_ml_min is not None:
                 scatter_data.append({
                     'reaction_type': reaction_cat,
                     'reactor_type': reactor_cat,
                     'inner_diameter_mm': inner_diameter_mm,
                     'flow_rate_ml_min': flow_rate_ml_min,
                     'pressure_bar': pressure_bar if pressure_bar is not None else 0,
                     'dataset': 'test' if is_test else 'train' # New column for dataset type
                 })
            else:
                missing = []
                if inner_diameter_mm is None:
                    missing.append(f"inner_diameter({inner_diameter_str})")
                if flow_rate_ml_min is None:
                    raw_flow = None
                    for cond in conditions:
                        if cond.get('type') == 'flow_rate_total':
                            raw_flow = cond.get('value')
                            break
                    missing.append(f"flow_rate({raw_flow})")
                print(f"Skipping {os.path.basename(fpath)}: Missing {', '.join(missing)}")


        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Output Sankey CSV
    with open('evaluation/sankey_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source (Reaction Type)', 'Target (Reactor Type)', 'Value'])
        for (src, tgt), count in sankey_data.items():
            writer.writerow([src, tgt, count])
    
    # Output Scatter CSV
    with open('evaluation/scatter_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Reaction Type', 'Reactor Inner Diameter (mm)', 'Flow Rate (mL/min)', 'Pressure (bar)', 'Dataset'])
        for item in scatter_data:
            writer.writerow([
                item['reaction_type'],
                item['inner_diameter_mm'],
                item['flow_rate_ml_min'],
                item['pressure_bar'],
                item['dataset']
            ])

    print("Data generation complete.")
    print(f"Sankey data written to evaluation/sankey_data.csv with {len(sankey_data)} flows.")
    print(f"Scatter data written to evaluation/scatter_data.csv with {len(scatter_data)} points.")

if __name__ == "__main__":
    main()
