import pandas as pd
import json

# Path to your dataset (update if needed)
df = pd.read_csv(r"../../thesis-project/data/raw/locust_dataset.csv")

# Clean and standardize
# Use uppercase for country, title case for region
if 'COUNTRYNAME' in df.columns and 'REGION' in df.columns:
    df['COUNTRYNAME'] = df['COUNTRYNAME'].astype(str).str.strip().str.upper()
    df['REGION'] = df['REGION'].astype(str).str.strip().str.title()

    # Build mapping
    mapping = {}
    for country, group in df.groupby('COUNTRYNAME'):
        regions = sorted(group['REGION'].dropna().unique())
        if regions:
            mapping[country] = regions

    # Output as JS object (for direct copy-paste)
    js_object = "const countryRegionMap = " + json.dumps(mapping, indent=2, ensure_ascii=False) + ";"
    with open("country_region_map.js", "w", encoding="utf-8") as f:
        f.write(js_object)
    print("country_region_map.js has been created. Copy its contents into your predict.html!")
else:
    print("COUNTRYNAME and/or REGION columns not found in the CSV.")
