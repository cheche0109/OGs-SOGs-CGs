import re
import pandas as pd

def parse_pepstats(file_path):
    # Regular expressions for extracting accession numbers and properties
    accession_regex = r"PEPSTATS of (\S+) from"
    properties_regex = r"(Tiny|Small|Aliphatic|Aromatic|Non-polar|Polar|Charged|Basic|Acidic)\s+\(.*?\)\s+\d+\s+([\d.]+)"
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into blocks for each entry
    entries = re.split(r"(PEPSTATS of \S+ from \d+ to \d)", content)

    # Group header with its corresponding entry text
    grouped_entries = [(entries[i], entries[i+1]) for i in range(1, len(entries) - 1, 2)]

    parsed_data = []

    for header, body in grouped_entries:
        # Extract accession number
        accession_match = re.search(accession_regex, header)
        #print(accession_match)
        if not accession_match:
            print("Warning: Could not find accession number for an entry. Skipping.")
            continue
        accession = accession_match.group(1)
        #print(accession)

        # Extract property values (Mole%)
        properties = re.findall(properties_regex, body)
        if not properties:
            print(f"Warning: No properties found for accession {accession}. Skipping.")
            continue

        # Build a dictionary for this entry
        property_data = {prop[0]: float(prop[1]) for prop in properties}
        property_data["Accession"] = accession

        # Append to the parsed data list
        parsed_data.append(property_data)
        #print(parsed_data)
    
    return parsed_data

def generate_csv(data, output_path):
    # Convert parsed data to a DataFrame
    df = pd.DataFrame(data)
    # Ensure all properties are included as columns even if missing in some entries
    df = df.fillna(0)
    # Reorder columns for consistency
    columns = ["Accession", "Tiny", "Small", "Aliphatic", "Aromatic", 
               "Non-polar", "Polar", "Charged", "Basic", "Acidic"]
    df = df.reindex(columns=columns)
    # Save to a CSV file
    df.to_csv(output_path, index=False)
    print(f"CSV file generated at {output_path}")


# Main script
if __name__ == "__main__":
    input_file = "pepstats_output.txt"
    output_file = "pepstats.csv"
    data = parse_pepstats(input_file)
    generate_csv(data, output_file)
