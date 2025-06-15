from sklearn.metrics import f1_score

def read_trojan_gates(file_path):
    """Reads a file and extracts the list of trojan gates between TROJAN_GATES and END_TROJAN_GATES."""
    with open(file_path, 'r') as file:
        content = file.readlines()
    
    trojan_start = False
    trojan_gates = set()
    
    for line in content:
        if line.strip() == "TROJAN_GATES":
            trojan_start = True
        elif line.strip() == "END_TROJAN_GATES":
            break
        elif line.strip() == "NO_TROJAN":
            break
        elif trojan_start:
            trojan_gates.add(line.strip())
    
    return trojan_gates

def calculate_f1_score(golden_gates, predicted_gates):
    """Calculate the F1 score between the golden and predicted trojan gates."""
    # Convert gates into binary values: 1 if gate is in set, else 0
    y_true = [1 if gate in golden_gates else 0 for gate in golden_gates.union(predicted_gates)]
    y_pred = [1 if gate in predicted_gates else 0 for gate in golden_gates.union(predicted_gates)]
    
    # Compute the F1 score
    return f1_score(y_true, y_pred)

def calculate_score(golden_file, predicted_file):
    """Calculates the score based on the comparison of the golden and predicted files."""
    golden_gates = read_trojan_gates(golden_file)
    predicted_gates = read_trojan_gates(predicted_file)
    
    # Check if both files contain TROJAN or NOTROJAN
    if golden_gates and not predicted_gates:
        return 0
    elif not golden_gates and predicted_gates:
        return 0
    elif not golden_gates and not predicted_gates:
        return 1
    else:
        # Both contain TROJAN, calculate F1 score
        return calculate_f1_score(golden_gates, predicted_gates)

total_score = 0
# Loop through the designs and calculate the score for each
training_cases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 24, 28, 29]
test_cases = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27]

for i in training_cases:
    # Assuming the files are named as per the example
    golden_file_path = f'reference/reference/result{i}.txt'
    predicted_file_path = f'predict/design{i}_predict.txt'

    score = calculate_score(golden_file_path, predicted_file_path)
    print(f"The score for design {i} is: {score}")
print("\n=======================================================")

for i in test_cases:
    # Assuming the files are named as per the example
    golden_file_path = f'reference/reference/result{i}.txt'
    predicted_file_path = f'predict/design{i}_predict.txt'

    score = calculate_score(golden_file_path, predicted_file_path)
    print(f"The score for design {i} is: {score}")
    total_score += score

print("\n=======================================================")
print(f"Total score across all designs: {total_score}")