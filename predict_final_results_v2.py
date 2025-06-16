import re
import pandas as pd
import os
import joblib
from collections import defaultdict, deque
import matplotlib.pyplot as plt
# design numbers and their corresponding Verilog file paths
design2v = {
    'design0': 'release(20250520)/release/design0.v',
    'design1': 'release(20250520)/release/design1.v',
    'design2': 'release(20250520)/release/design2.v',
    'design3': 'release(20250520)/release/design3.v',
    'design4': 'release(20250520)/release/design4.v',
    'design5': 'release(20250520)/release/design5.v',
    'design6': 'release(20250520)/release/design6.v',
    'design7': 'release(20250520)/release/design7.v',
    'design8': 'release(20250520)/release/design8.v',
    'design9': 'release(20250520)/release/design9.v',
    'design10': 'release(20250522)/release2/trojan_design/design10.v',
    'design11': 'release(20250522)/release2/trojan_design/design11.v',
    'design12': 'release(20250522)/release2/trojan_design/design12.v',
    'design13': 'release(20250522)/release2/trojan_design/design13.v',
    'design14': 'release(20250522)/release2/trojan_design/design14.v',
    'design15': 'release(20250522)/release2/trojan_design/design15.v',
    'design16': 'release(20250522)/release2/trojan_design/design16.v',
    'design17': 'release(20250522)/release2/trojan_design/design17.v',
    'design18': 'release(20250522)/release2/trojan_design/design18.v',
    'design19': 'release(20250522)/release2/trojan_design/design19.v',
    'design20': 'release(20250522)/release2/trojan_free/design20.v',
    'design21': 'release(20250522)/release2/trojan_free/design21.v',
    'design22': 'release(20250522)/release2/trojan_free/design22.v',
    'design23': 'release(20250522)/release2/trojan_free/design23.v',
    'design24': 'release(20250522)/release2/trojan_free/design24.v',
    'design25': 'release(20250522)/release2/trojan_free/design25.v',
    'design26': 'release(20250522)/release2/trojan_free/design26.v',
    'design27': 'release(20250522)/release2/trojan_free/design27.v',
    'design28': 'release(20250522)/release2/trojan_free/design28.v',
    'design29': 'release(20250522)/release2/trojan_free/design29.v'
}

# ---------------------------
# Function: Predict and Save Results as txt files
# ---------------------------

def extract_signals(line):
    # support input/output/wire declarations
    line = line.replace(';', '').strip()
    range_match = re.search(r'\[(\d+):(\d+)\]', line)
    if range_match:
        msb, lsb = map(int, range_match.groups())
        signals = line[range_match.end():].split(',')
        result = []
        for sig in signals:
            base = sig.strip()
            for i in range(lsb, msb + 1):
                result.append(f"{base}[{i}]")
        return result
    else:
        return [sig.strip() for sig in re.split(r',\s*', line.split()[-1]) if sig.strip()]

def parse_verilog_with_gate_type(lines):
    gates = []
    gate_inputs = {}
    gate_outputs = {}

    primary_inputs, primary_outputs = set(), set()
    dff_inputs, dff_outputs = set(), set()

    gate_name_to_type = {}  # Mapping from gate name to its type

    for line in lines:
        line = line.strip().rstrip(';')

        if line.startswith('input'):
            primary_inputs.update(extract_signals(line))
        elif line.startswith('output'):
            primary_outputs.update(extract_signals(line))

        m = re.match(r'(\w+)\s+(g\d+)\((.*)\)', line)
        if not m:
            continue
        gate_type, gate_name, arg_string = m.groups()
        gate_name_to_type[gate_name] = gate_type  # Mapping gate name to type

        if '.' in arg_string:  # Named port style (dff)
            args = dict(re.findall(r'\.(\w+)\(([^)]+)\)', arg_string))
            output = args.get('Q', '')
            inputs = [v for k, v in args.items() if k != 'Q']
            if 'D' in args: dff_inputs.add(args['D'])
            if 'Q' in args: dff_outputs.add(args['Q'])
        else:
            parts = [x.strip() for x in arg_string.split(',')]
            output, inputs = parts[0], parts[1:]

        gates.append((gate_name, gate_type, output, inputs))
        gate_inputs[gate_name] = inputs
        gate_outputs[gate_name] = output

    return gates, gate_inputs, gate_outputs, primary_inputs, primary_outputs, dff_inputs, dff_outputs, gate_name_to_type

def build_gate_graphs(gate_inputs, gate_outputs, gate_name_to_type):
    fanin = defaultdict(list)
    fanout = defaultdict(list)
    # For each gate, find the gates feeding into it (fanin)
    for gate, ins in gate_inputs.items():
        if gate_name_to_type.get(gate) == 'dff':
            continue  # Skip DFFs for fanin/fanout analysis
        # The output of this gate
        out = gate_outputs[gate]
        # Find the gates that feed into this gate's output (fanin)
        for other_gate, other_inputs in gate_inputs.items():
            if out in other_inputs:
                fanout[gate].append(other_gate)
    # For each gate, find the gates it drives (fanout)
    for gate, out in gate_outputs.items():
        if gate_name_to_type.get(gate) == 'dff':
            continue  # Skip DFFs for fanin/fanout analysis
        # The inputs of this gate
        ins = gate_inputs[gate]
        # Find the gates that are driven by this gate's output (fanout)
        for other_gate, other_out in gate_outputs.items():
            if other_out in ins:
                fanin[gate].append(other_gate)
    return fanin, fanout

def neighbor_check_for_candidates(gate_name, gate_fanin_graph, gate_fanout_graph, gate_name_probs_dict, prob1):
    current_prob = gate_name_probs_dict.get(gate_name, 0)  # Default to 0 if not in the dictionary
    if current_prob <= prob1:
        return set()  # Return an empty set if the probability is not greater than prob1
    if gate_name in gate_fanin_graph:
        for fanin_gate in gate_fanin_graph[gate_name]:
            if gate_name_probs_dict.get(fanin_gate, 0) > prob1:
                return True
    if gate_name in gate_fanout_graph:
        for fanout_gate in gate_fanout_graph[gate_name]:
            if gate_name_probs_dict.get(fanout_gate, 0) > prob1:
                return True
    return False


def determine_trojan_gate(gate_name, gate_type, gate_name_probs_dict, gate_fanin_graph, gate_fanout_graph, prob1, prob2):
    if gate_name_probs_dict.get(gate_name, 0) > prob2: # if the gate's probability is greater than prob2, it is definitely a Trojan
        return 'Trojan'
    elif neighbor_check_for_candidates(gate_name, gate_fanin_graph, gate_fanout_graph, gate_name_probs_dict, prob1):
        return 'Trojan'
    else:
        return 'Not_Trojan'
    # return 'Trojan' if (gate_name_probs_dict[gate_name] > prob1) else 'Not_Trojan'

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
        
        # Load the Verilog file
        with open(design2v[design_id], 'r') as f:
            verilog_lines = f.readlines()

        # Parse the Verilog file
        gates, gate_inputs, gate_outputs, primary_inputs, primary_outputs, dff_inputs, dff_outputs, gate_name_to_type = parse_verilog_with_gate_type(verilog_lines)

        # Build fanin and fanout graphs
        gate_fanin_graph, gate_fanout_graph = build_gate_graphs(gate_inputs, gate_outputs, gate_name_to_type)

        # y_pred = (y_probs > 0.5).astype(int) # Thresholding at 0.75 for Trojan detection

        gate_name_probs_dict = dict(zip(df_w_gate_name['gate_name'], y_probs))

        # plot the distribution of probabilities (PDF)
        # plt.figure(figsize=(10, 6))
        # plt.hist(y_probs, bins=50, color='blue', alpha=0.7)
        # plt.title(f'Probability Distribution for {design_id}')
        # plt.xlabel('Probability of Trojan')
        # plt.ylabel('Frequency')
        # plt.grid()
        # plt.savefig(f'predict_prob/{design_id}_probability_distribution.png')

        # plot the distribution of probabilities (CDF)
        # plt.figure(figsize=(10, 6))
        # plt.hist(y_probs, bins=50, color='blue', alpha=0.7, cumulative=True, density=True)
        # plt.title(f'Probability CDF for {design_id}')
        # plt.xlabel('Probability of Trojan')
        # plt.ylabel('Cumulative Frequency')
        # plt.grid()
        # plt.savefig(f'predict_prob/{design_id}_probability_cdf.png')


        
        # Set the probabilities for determining Trojan gates 
        prob1 = 0.5  # Threshold for Trojan gate candidates
        prob2 = 0.75  # Threshold for definite Trojan gates
        perctent99 = pd.Series(y_probs).quantile(0.99)  # Set prob2 as the 80th percentile of the probabilities
        perctent90 = pd.Series(y_probs).quantile(0.90)  # Set prob1 as the 90th percentile of the probabilities
        if perctent99 < prob2 and perctent99 > 0.45:
            prob2 = perctent99
            prob1 = pd.Series(y_probs).quantile(0.8)  # Set prob1 as the median of the probabilities


        # Create a dictionary of gate_name to predicted result
        for gate_name, gate_type in zip(df_w_gate_name['gate_name'], df_w_gate_name['gate_type']):
            gate_name_to_prediction[gate_name] = determine_trojan_gate(gate_name, gate_type, gate_name_probs_dict, gate_fanin_graph, gate_fanout_graph, prob1, prob2)

        # Create the output filename based on the design ID
        output_filename = f"predict/{design_id}_predict.txt"
        
        with open(output_filename, 'w') as f:
            trojan_gates = [gate_name for gate_name, pred in gate_name_to_prediction.items() if pred == 'Trojan']
            if len(trojan_gates) <= 15: # Threshold for No Trojan design
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
