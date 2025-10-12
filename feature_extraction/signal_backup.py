import csv

# Define the input and output file paths
input_file = "signalp_prediction_results.txt"  # Replace with your actual file name
output_file = "signalp6_features.csv"

# Open the SignalP output file
with open(input_file, "r") as file:
    lines = file.readlines()

# Prepare a list to store the extracted data
extracted_data = []

# Process each line
for line in lines:
    # Skip comments and header lines
    if line.startswith("#") or not line.strip():
        continue
    
    # Split the line into columns by tab
    columns = line.strip().split("\t")
    
    # Extract the ID and OTHER columns
    seq_id = columns[0]
    other_value = columns[2]
    
    # Add to the extracted data
    extracted_data.append({"ID": seq_id, "OTHER": other_value})

# Write the extracted data to a CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["ID", "OTHER"])
    writer.writeheader()
    writer.writerows(extracted_data)

print(f"Data successfully written to {output_file}")
