import pandas as pd
import re
import argparse

# Function to load and process the files
def process_files(txt_file_path, csv_file_path, label_csv_file_path):
    # Load the text file with gate numbers
    with open(txt_file_path, 'r') as file:
        trojan_gates = set(file.read().splitlines()[2:-1])  # Extract gate numbers from the text file

    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Clean up the column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Function to identify gate numbers from a row
    def extract_gate_numbers(row):
        gate_numbers = re.findall(r'g\d+', str(row))  # Match pattern like g40, g50, etc.
        return gate_numbers

    # Apply extraction function to all columns and create a new column for gate numbers
    df['gate_numbers'] = df.apply(lambda row: extract_gate_numbers(row), axis=1)

    # Create a set from the extracted gate numbers for efficient lookup
    trojan_gate_set = set(trojan_gates)

    # Add the label column based on whether the gate number is in the Trojan gate list
    df['label'] = df['gate_numbers'].apply(lambda x: 'Trojan' if any(gate in trojan_gate_set for gate in x) else 'Not_Trojan')

    # Save the updated DataFrame to a new CSV file
    output_csv_path = label_csv_file_path
    df.to_csv(output_csv_path, index=False)
    print(f"Training file saved to: {output_csv_path}")

# Main function to handle user input for file paths
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process CSV and text files to label rows as Trojan or Not_Trojan.")
    parser.add_argument("txt_file", help="Path to the text file containing Trojan gate numbers.")
    parser.add_argument("csv_file", help="Path to the CSV file to process.")
    parser.add_argument("label_csv_file", help="Path to the label CSV file to process.")

    # Parse arguments
    args = parser.parse_args()

    # Process the files
    process_files(args.txt_file, args.csv_file, args.label_csv_file)

if __name__ == "__main__":
    main()
