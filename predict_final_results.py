import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

# ---------------------------
# Function: Predict and Save Results as txt files
# ---------------------------

def predict_and_save_results(test_files, model, le_gate_type):
    drop_cols = ['output', 'inputs', 'gate_numbers', 'design_id', 'gate_name']
    
    for file in test_files:
        df = pd.read_csv(file)
        design_id = os.path.basename(file).split('.')[0]  # Extract design ID from filename

        # Encode 'gate_type' in test data
        df['gate_type'] = le_gate_type.transform(df['gate_type'])

        # Store gate_name with prediction results
        gate_name_to_prediction = {}

        # Drop non-numeric or ID-based columns (excluding 'gate_name')
        df_w_gate_name = df.copy()
        df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Ensure numeric and drop NaNs
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Predict and store results
        y_probs = model.predict_proba(df)[:, 1]  # Get probabilities for the positive class (Trojan)
        y_pred = (y_probs > 0.5).astype(int)

        # Create a dictionary of gate_name to predicted result
        for gate_name, prediction in zip(df_w_gate_name['gate_name'], y_pred):
            gate_name_to_prediction[gate_name] = 'Trojan' if prediction == 1 else 'Not_Trojan'

        # Create the output filename based on the design ID
        output_filename = f"predict/{design_id}_predict.txt"
        
        with open(output_filename, 'w') as f:
            trojan_gates = [gate_name for gate_name, pred in gate_name_to_prediction.items() if pred == 'Trojan']
            if len(trojan_gates) == 0:
                f.write("NO_TROJAN")
            else:
                # Write the header
                f.write("TROJANED\n")
                f.write("TROJAN_GATES\n")
                
                # Write the list of Trojan gates (e.g., g40, g41, etc.)
                for gate_name in trojan_gates:
                    f.write(f"{gate_name}\n")
                
                # Write the footer
                f.write("END_TROJAN_GATES")

        print(f"Saved Trojan gates for {design_id} as {output_filename}")

# ---------------------------
# Example Usage
# ---------------------------

if __name__ == "__main__":
    # Load the trained model
    model = joblib.load('xgb_model.pkl')
    print("\nModel loaded from xgb_model.pkl")

    # Load LabelEncoder for 'gate_type' encoding
    le_gate_type = joblib.load('le_gate_type.pkl')  # Ensure you have saved this encoder earlier
    print("LabelEncoder loaded from le_gate_type.pkl\n")

    # Choose your test files here
    test_files = [f"training_data/design{i}.csv" for i in range(0, 30)]  # Example for 30 files

    # Predict and save the results for each test file
    predict_and_save_results(test_files, model, le_gate_type)
