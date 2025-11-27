import json
import os
import glob
import csv

def categorize_reaction(reaction_type_raw):
    if not reaction_type_raw:
        return "Others"
    r = reaction_type_raw.lower()
    
    if "hydrogenation" in r or "hydrogenolysis" in r or "hydrogenaton" in r or "hydrodechlorination" in r:
        return "Hydrogenation"
    return "Others"

def get_reaction_info(fpath):
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check Reaction Type
        reaction_type_raw = data.get('reaction_type', '')
        if categorize_reaction(reaction_type_raw) != "Hydrogenation":
            return None

        # Extract Reactants and Catalysts
        reactants_list = data.get('reactants', [])
        substrates = []
        catalysts = []
        
        for item in reactants_list:
            role = item.get('role', '').lower()
            name = item.get('name', '').strip()
            if not name:
                continue
            
            if role == 'reactant':
                substrates.append(name)
            elif role == 'catalyst':
                catalysts.append(name)
        
        # Extract Temperature
        conditions = data.get('conditions', [])
        temperature = None
        for cond in conditions:
            if cond.get('type') == 'temperature':
                val = cond.get('value')
                if val:
                    temperature = val.strip()
                break
        
        # Validation: All three must exist
        if not substrates:
            # print(f"Skipping {os.path.basename(fpath)}: No substrates found.")
            return None
        if not catalysts:
            # print(f"Skipping {os.path.basename(fpath)}: No catalysts found.")
            return None
        if not temperature:
            # print(f"Skipping {os.path.basename(fpath)}: No temperature found.")
            return None

        # Sort lists for consistent deduping
        substrates.sort()
        catalysts.sort()

        return {
            "substrates": "; ".join(substrates),
            "catalysts": "; ".join(catalysts),
            "temperature": temperature
        }

    except Exception as e:
        print(f"Error reading {fpath}: {e}")
        return None

def main():
    ground_truth_dir = 'finetune/ground_truth'
    qwen_dir = 'finetune/qwen 直接提取所有，可用于abstract2impact微调'
    test_dir = 'evaluation/result/ground truth'
    
    files_gt = glob.glob(os.path.join(ground_truth_dir, '*_annotated.json'))
    files_qwen = glob.glob(os.path.join(qwen_dir, '*_reaction.json'))
    files_test = glob.glob(os.path.join(test_dir, '*_annotated.json'))
    
    all_files = files_gt + files_qwen + files_test
    print(f"Scanning {len(all_files)} files...")

    unique_entries = set()
    rows = []

    for fpath in all_files:
        info = get_reaction_info(fpath)
        if info:
            # Create a tuple for deduping
            entry_key = (info['substrates'], info['catalysts'], info['temperature'])
            
            if entry_key not in unique_entries:
                unique_entries.add(entry_key)
                rows.append(info)
    
    output_file = 'evaluation/hydrogenation_data.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["substrates", "catalysts", "temperature"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Extraction complete. Found {len(rows)} unique valid hydrogenation entries.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()



