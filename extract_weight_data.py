import xml.etree.ElementTree as ET
import pandas as pd

# Load and parse the XML file
tree = ET.parse('export.xml')
root = tree.getroot()

# Initialize a list to store weight records
weight_records = []

# Loop through the XML to find weight data
for record in root.findall(".//Record[@type='HKQuantityTypeIdentifierBodyMass']"):
    # Use a consistent date format (ISO 8601)
    date = pd.to_datetime(record.attrib['startDate']).strftime('%Y-%m-%d')
    weight = float(record.attrib['value'])
    weight_records.append({'Date': date, 'Weight': weight})

# Convert the list to a DataFrame
weight_df = pd.DataFrame(weight_records)

# Ensure date column is in the correct datetime format for pandas
weight_df['Date'] = pd.to_datetime(weight_df['Date'], format='%Y-%m-%d')

# Save to CSV in a consistent date format
weight_df.to_csv('weight_data.csv', index=False)
