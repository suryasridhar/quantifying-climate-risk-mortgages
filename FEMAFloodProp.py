# Create data frame with flood data
"""
Kristen Epperly
"""

import pandas as pd

# Import FEMA Individual Assistance Multiple Loss Flood Properties - v1
floodProp = pd.read_csv(r'C:\Users\krist\OneDrive\Documents\Master FIM\FIM 500&601\FIM 601 Project\Data Sources\IndividualAssistanceMultipleLossFloodProperties.csv')
print(floodProp.columns.tolist())
print(floodProp.head())

# Standardize column names 
floodProp.columns = [c.strip() for c in floodProp.columns]

# Drop id to make dataset smaller
if 'id' in floodProp.columns:
    floodProp = floodProp.drop(columns=['id'])

# Print the first few rows of data
print(f"Loaded {len(floodProp)} records.")
floodProp.head()
