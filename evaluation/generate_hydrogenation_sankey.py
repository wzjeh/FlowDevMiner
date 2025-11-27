import csv
import re

def parse_temperature_to_celsius(temp_str):
    if not temp_str:
        return None
    
    # Lowercase and clean
    t = temp_str.lower().strip()
    
    # Handle "room temperature"
    if "room" in t or "ambient" in t or "rt" in t:
        return 25.0
    
    # Extract number
    match = re.match(r"([\d\.]+)", t)
    if not match:
        return None
        
    val = float(match.group(1))
    
    # Check unit
    if "k" in t:
        return val - 273.15
    # Default assume Celsius if C or no unit (or weird encoding like ﾂｰC)
    return val

def get_catalyst_metal(cat_str):
    if not cat_str:
        return "Others"
    
    # Common hydrogenation metals to look for
    # Prioritize finding the symbol in the string
    # Using regex to find symbol boundaries to avoid partial matches (e.g. "Cu" in "Curium" - unlikely but safe)
    # Also handle things like "Pd/C", "Pd-Au", etc. We prioritize the first one found or specific priority list?
    # User said "按照首个金属分类", implying finding the first recognizable metal.
    
    # List of common catalysis metals
    metals = ["Ru", "Pd", "Pt", "Cu", "Ni", "Rh", "Au", "Fe", "Co", "Ag", "Zn"]
    
    # We scan the string and find the earliest occurrence of any metal symbol
    found_metal = "Others"
    min_index = len(cat_str) + 1
    
    for metal in metals:
        # Case insensitive check? Metal symbols are usually capitalized in chemical formulas, 
        # but user text might vary. Let's do case-insensitive search but be careful.
        # Actually, "Ni" matches "Nitro". So strict case for symbols might be safer if the text is good.
        # But text like "palladium" is also possible.
        
        # Strategy: Search for Symbol (case sensitive) OR Full Name (case insensitive)
        metal_full = {
            "Ru": "ruthenium",
            "Pd": "palladium",
            "Pt": "platinum",
            "Cu": "copper",
            "Ni": "nickel",
            "Rh": "rhodium",
            "Au": "gold",
            "Fe": "iron",
            "Co": "cobalt",
            "Ag": "silver",
            "Zn": "zinc"
        }
        
        # Find index of symbol
        # Regex to match word boundary or start/end
        # e.g. "Pd" in "Pd/C" -> match. "Cu" in "Cu-Zn" -> match.
        idx = -1
        # Try finding full name first (case insensitive)
        idx_name = cat_str.lower().find(metal_full[metal])
        
        # Try finding symbol (case sensitive usually better for chem formulas, but let's try strict first)
        # We assume standard capitalization for symbols in JSON
        idx_symbol = cat_str.find(metal)
        
        # If symbol is found, verify it's not part of another word (like 'Ni' in 'Nitrogen')
        # Simple heuristic: check next char is not lowercase letter
        if idx_symbol != -1:
            if idx_symbol + len(metal) < len(cat_str):
                next_char = cat_str[idx_symbol + len(metal)]
                if next_char.islower():
                    idx_symbol = -1 # Probably part of a word
        
        current_idx = -1
        if idx_name != -1 and idx_symbol != -1:
            current_idx = min(idx_name, idx_symbol)
        elif idx_name != -1:
            current_idx = idx_name
        elif idx_symbol != -1:
            current_idx = idx_symbol
            
        if current_idx != -1 and current_idx < min_index:
            min_index = current_idx
            found_metal = metal
            
    return found_metal

def categorize_temp(temp_c):
    if temp_c is None:
        return "Unknown"
    
    if temp_c <= 50:
        return "0-50°C"
    elif temp_c <= 100:
        return "50-100°C"
    else:
        return ">100°C"

def main():
    input_file = 'evaluation/hydrogenation_data.csv'
    output_file = 'evaluation/hydrogenation_sankey_data.csv'
    
    # Structure:
    # Source -> Target -> Value
    # 1. Hydrogenation (Total) -> Catalyst Metal
    # 2. Catalyst Metal -> Temperature Range
    
    # We need to count flows.
    # Flow 1: "Hydrogenation" -> Metal
    # Flow 2: Metal -> Temp Range
    
    # But for a multi-level Sankey, we usually list all links in one file.
    # Source, Target, Value
    
    data_rows = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                catalyst = row['catalysts']
                temp_str = row['temperature']
                
                metal = get_catalyst_metal(catalyst)
                temp_c = parse_temperature_to_celsius(temp_str)
                temp_cat = categorize_temp(temp_c)
                
                if temp_cat != "Unknown":
                    data_rows.append({
                        "metal": metal,
                        "temp_cat": temp_cat
                    })
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Count frequencies
    # Level 1: Hydrogenation -> Metal
    metal_counts = {}
    # Level 2: Metal -> Temp Range
    metal_temp_counts = {} # Key: (metal, temp_cat)
    
    total_count = len(data_rows)
    
    for item in data_rows:
        m = item['metal']
        t = item['temp_cat']
        
        metal_counts[m] = metal_counts.get(m, 0) + 1
        
        key = (m, t)
        metal_temp_counts[key] = metal_temp_counts.get(key, 0) + 1
        
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Value'])
        
        # Level 1 flows
        # Note: If we just say "Hydrogenation" -> Metal, it's a simple fan out.
        # User said "第一列是hydrogenation总的是100%", so yes, one source node.
        for metal, count in metal_counts.items():
            writer.writerow(['Hydrogenation', metal, count])
            
        # Level 2 flows
        for (metal, t_cat), count in metal_temp_counts.items():
            writer.writerow([metal, t_cat, count])
            
    print(f"Sankey data generated with {total_count} valid entries.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()



