# Create data frame with flood data
"""
Kristen Epperly
"""

import pandas as pd

# FEMA dataset link with column definitions:
# https://www.fema.gov/openfema-data-page/individual-assistance-multiple-loss-flood-properties-v1

# Specify the path to the dataset
file_path = '/content/drive/MyDrive/IndividualAssistanceMultipleLossFloodProperties.csv'
Flood = pd.read_csv(file_path)

# Standardize column names (optional, but good practice)
Flood.columns = [c.strip() for c in Flood.columns]

# Drop id, latitude, and longitude
# Flood = Flood.drop(columns=['id', 'latitude', 'longitude'], errors='ignore')
Flood = Flood.rename(columns={"damagedZipCode": "zip"})


# Preview the dataframe
print(f"Loaded {len(Flood)} records.")
Flood.head()

# US Elevation Data with ZIP Code

# Specify the path to the dataset
file_path = '/content/drive/MyDrive/us_elevation_with_zip.csv'
Elevation = pd.read_csv(file_path)

# Standardize column names (optional, but good practice)
Elevation.columns = [c.strip() for c in Elevation.columns]

# Drop id, latitude, and longitude
# Elevation = Elevation.drop(columns=['latitude', 'longitude'], errors='ignore')

# Preview the dataframe
print(f"Loaded {len(Elevation)} records.")
Elevation.head()

# Combine the Flood Data and Elevation Data on ZIP Code

Combined = pd.merge(Flood, Elevation, on=["zip"])
Combined.to_csv("combined_flood.csv", index=False)
Combined.head()

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Calculated proximity score for each zip code we found the distance to the nearest zip code that had recorded flood damage
# or losses. Zip codes closer to a flooded area got a higher proximity score.
# Combined into a weighted risk score each factor was multiplied by a weight reflecting its importance and then added together.
# Elevation and proximity got the highest weights (25% each) since they are the strongest predictors of flood risk,
# followed by flood damage (20%), water level (15%), number of losses (10%), and flood frequency (5%).
# Converted to 1-5 scale the raw risk score was a decimal between 0 and 1. We used pd.cut to divide it into 5 equal buckets
# and label them 1 through 5, giving you the final intuitive risk rating.

combined_flood = Combined

# ── Step 1: Normalize each factor at the row level ────────────────────────
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

combined_flood["elev_score"]        = 1 - normalize(combined_flood["elevation"])
combined_flood["damage_score"]      = normalize(combined_flood["floodDamage"])
combined_flood["loss_score"]        = normalize(combined_flood["numberOfLosses"])
combined_flood["water_level_score"] = normalize(combined_flood["waterLevel"])

# ── Step 2: Fill NaN scores with 0 ────────────────────────────────────────
combined_flood["elev_score"]        = combined_flood["elev_score"].fillna(0)
combined_flood["damage_score"]      = combined_flood["damage_score"].fillna(0)
combined_flood["loss_score"]        = combined_flood["loss_score"].fillna(0)
combined_flood["water_level_score"] = combined_flood["water_level_score"].fillna(0)

# ── Step 3: Proximity score ────────────────────────────────────────────────
zip_coords = Elevation.groupby("zip").agg(
    lat=("latitude", "mean"),
    lon=("longitude", "mean")
).reset_index()

combined_flood = combined_flood.merge(zip_coords, on="zip", how="left")

# Identify flooded zip coordinates
flooded_zips = combined_flood[(combined_flood["numberOfLosses"] > 0) | (combined_flood["floodDamage"] > 0)]
flooded_coords = np.radians(flooded_zips[["lat", "lon"]].values)
all_coords = np.radians(combined_flood[["lat", "lon"]].values)

tree = cKDTree(flooded_coords)
distances, _ = tree.query(all_coords, k=1)
combined_flood["dist_to_flood_km"] = distances * 6371
combined_flood["proximity_score"]  = 1 - normalize(combined_flood["dist_to_flood_km"])
combined_flood["proximity_score"]  = combined_flood["proximity_score"].fillna(0)

# ── Step 4: Weighted risk score ────────────────────────────────────────────
combined_flood["raw_risk"] = (
    combined_flood["elev_score"]        * 0.25 +
    combined_flood["proximity_score"]   * 0.25 +
    combined_flood["damage_score"]      * 0.20 +
    combined_flood["water_level_score"] * 0.15 +
    combined_flood["loss_score"]        * 0.10
)

# ── Step 5: Convert to 1-5 risk scale ─────────────────────────────────────
combined_flood["risk_score"] = pd.cut(
    combined_flood["raw_risk"],
    bins=5,
    labels=[1, 2, 3, 4, 5],
    duplicates="drop"
).astype(int)

# ── Step 6: Save ───────────────────────────────────────────────────────────
combined_flood.to_csv("flood_with_risk_scores.csv", index=False)
print(combined_flood[["zip", "elevation", "floodDamage", "dist_to_flood_km", "risk_score"]].head(20))
print(f"\nRisk score distribution:\n{combined_flood['risk_score'].value_counts().sort_index()}")

combined_flood.head()

