import re
import csv
import argparse
from collections import defaultdict, deque

def is_bus(signal):
    return '[' in signal and ']' in signal

def is_const(signal):
    return "1'b" in signal or signal.strip() in ("1'b0", "1'b1")

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

def build_graphs(gate_inputs, gate_outputs):
    fanin = defaultdict(list)
    fanout = defaultdict(list)
    for gate, ins in gate_inputs.items():
        out = gate_outputs[gate]
        for i in ins:
            fanin[out].append(i)
            fanout[i].append(out)
    return fanin, fanout

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

def bfs(source_set, graph):
    visited = {}
    queue = deque((s, 0) for s in source_set)
    while queue:
        node, level = queue.popleft()
        if node in visited:
            continue
        visited[node] = level
        for neigh in graph.get(node, []):
            queue.append((neigh, level + 1))
    return visited

# Recursive function to count gate types in a cone (fanin or fanout)
def count_gate_types_in_cone(gate, graph, gate_name_to_type, visited):
    # If the gate is not in the graph, return an empty count
    if gate not in graph:
        return defaultdict(int)
    # If the gate has already been visited, return the cached counts
    if gate in visited:
        return visited[gate]
    # Initialize the counts for this gate
    counts = defaultdict(int)
    # Get the gate type and update the count
    gate_type = gate_name_to_type.get(gate)
    if gate_type:
        counts[gate_type] += 1
    # Recursively count the gate types for all neighbors (fanin or fanout)
    for neighbor in graph.get(gate, []):
        neighbor_counts = count_gate_types_in_cone(neighbor, graph, gate_name_to_type, visited)
        for key, value in neighbor_counts.items():
            counts[key] += value
    visited[gate] = counts
    return counts

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Parse Verilog file and output gate features to CSV")
    parser.add_argument('input_file', help="Input Verilog netlist file")
    parser.add_argument('output_file', help="Output CSV file path")
    args = parser.parse_args()

    # Load the Verilog file
    with open(args.input_file, 'r') as file:
        verilog_lines = file.readlines()

    # Parse the Verilog file
    gates, gate_inputs, gate_outputs, primary_inputs, primary_outputs, dff_inputs, dff_outputs, gate_name_to_type = parse_verilog_with_gate_type(verilog_lines)

    # Build fanin and fanout graphs
    fanin_graph, fanout_graph = build_graphs(gate_inputs, gate_outputs)
    gate_fanin_graph, gate_fanout_graph = build_gate_graphs(gate_inputs, gate_outputs, gate_name_to_type)

    #　build pre-level, post-level dictionaries
    pre_level = defaultdict(lambda: defaultdict(int))
    post_level = defaultdict(lambda: defaultdict(int))

    for this_gate, input_gates in gate_fanin_graph.items():
        for input_gate in input_gates:
            if gate_name_to_type[input_gate] in pre_level[this_gate]:
                pre_level[this_gate][gate_name_to_type[input_gate]] += 1
            else:
                pre_level[this_gate][gate_name_to_type[input_gate]] = 1
    for this_gate, output_gates in gate_fanout_graph.items():
        for output_gate in output_gates:
            if gate_name_to_type[output_gate] in post_level[this_gate]:
                post_level[this_gate][gate_name_to_type[output_gate]] += 1
            else:
                post_level[this_gate][gate_name_to_type[output_gate]] = 1

    # Perform BFS to find levels of fanin and fanout
    lvl_from_input = bfs(primary_inputs, fanout_graph)
    lvl_to_output = bfs(primary_outputs, fanin_graph)
    lvl_to_output_ff = bfs(dff_outputs, fanin_graph)
    lvl_from_input_ff = bfs(dff_inputs, fanout_graph)

    # Count the gate types in both fanin and fanout cones for each gate
    gate_type_count_fanin = defaultdict(lambda: defaultdict(int))
    gate_type_count_fanout = defaultdict(lambda: defaultdict(int))

    visited_fanin = {}
    visited_fanout = {}
    for gate_name, gate_type, out, ins in gates:
        # Count gate types in the fanin cone
        fanin_counts = count_gate_types_in_cone(gate_name, gate_fanin_graph, gate_name_to_type, visited_fanin)
        for gate_type, count in fanin_counts.items():
            gate_type_count_fanin[gate_name][gate_type] = count

        # Count gate types in the fanout cone
        fanout_counts = count_gate_types_in_cone(gate_name, gate_fanout_graph, gate_name_to_type, visited_fanout)
        for gate_type, count in fanout_counts.items():
            gate_type_count_fanout[gate_name][gate_type] = count

    # Prepare the headers for the CSV file, including the original 12 features + 18 new features
    csv_headers = [
        'gate_name', 'gate_type', 'output', 'inputs',
        'input_count', 'is_sequential', 'is_constant_input', 'has_bus_signal',
        'level_to_input', 'level_to_output', 'level_to_output_ff', 'level_to_input_ff',
        'and_gates_fanin', 'or_gates_fanin', 'nand_gates_fanin', 'nor_gates_fanin', 
        'not_gates_fanin', 'buf_gates_fanin', 'xor_gates_fanin', 'xnor_gates_fanin', 'dff_gates_fanin',
        'and_gates_fanout', 'or_gates_fanout', 'nand_gates_fanout', 'nor_gates_fanout', 
        'not_gates_fanout', 'buf_gates_fanout', 'xor_gates_fanout', 'xnor_gates_fanout', 'dff_gates_fanout',
        'and_gates_prelevel', 'or_gates_prelevel', 'nand_gates_prelevel', 'nor_gates_prelevel', 
        'not_gates_prelevel', 'buf_gates_prelevel', 'xor_gates_prelevel', 'xnor_gates_prelevel', 'dff_gates_prelevel',
        'and_gates_postlevel', 'or_gates_postlevel', 'nand_gates_postlevel', 'nor_gates_postlevel', 
        'not_gates_postlevel', 'buf_gates_postlevel', 'xor_gates_postlevel', 'xnor_gates_postlevel', 'dff_gates_postlevel'
    ]

    # Write the CSV file
    with open(args.output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for gate_name, gate_type, out, ins in gates:
            row = {
                'gate_name': gate_name,
                'gate_type': gate_type,
                'output': out,
                'inputs': ', '.join(ins),
                'input_count': len(ins),
                'is_sequential': int(gate_type.lower() == 'dff'),
                'is_constant_input': int(any(is_const(i) for i in ins)),
                'has_bus_signal': int(is_bus(out) or any(is_bus(i) for i in ins)),
                'level_to_input': lvl_from_input.get(out, -1),
                'level_to_output': lvl_to_output.get(out, -1),
                'level_to_output_ff': lvl_to_output_ff.get(out, -1),
                'level_to_input_ff': lvl_from_input_ff.get(out, -1),
                'and_gates_fanin': gate_type_count_fanin[gate_name].get('and', 0),
                'or_gates_fanin': gate_type_count_fanin[gate_name].get('or', 0),
                'nand_gates_fanin': gate_type_count_fanin[gate_name].get('nand', 0),
                'nor_gates_fanin': gate_type_count_fanin[gate_name].get('nor', 0),
                'not_gates_fanin': gate_type_count_fanin[gate_name].get('not', 0),
                'buf_gates_fanin': gate_type_count_fanin[gate_name].get('buf', 0),
                'xor_gates_fanin': gate_type_count_fanin[gate_name].get('xor', 0),
                'xnor_gates_fanin': gate_type_count_fanin[gate_name].get('xnor', 0),
                'dff_gates_fanin': gate_type_count_fanin[gate_name].get('dff', 0),
                'and_gates_fanout': gate_type_count_fanout[gate_name].get('and', 0),
                'or_gates_fanout': gate_type_count_fanout[gate_name].get('or', 0),
                'nand_gates_fanout': gate_type_count_fanout[gate_name].get('nand', 0),
                'nor_gates_fanout': gate_type_count_fanout[gate_name].get('nor', 0),
                'not_gates_fanout': gate_type_count_fanout[gate_name].get('not', 0),
                'buf_gates_fanout': gate_type_count_fanout[gate_name].get('buf', 0),
                'xor_gates_fanout': gate_type_count_fanout[gate_name].get('xor', 0),
                'xnor_gates_fanout': gate_type_count_fanout[gate_name].get('xnor', 0),
                'dff_gates_fanout': gate_type_count_fanout[gate_name].get('dff', 0),
                'and_gates_prelevel': gate_type_count_fanin[gate_name].get('and', 0),
                'or_gates_prelevel': pre_level[gate_name].get('or', 0),
                'nand_gates_prelevel': pre_level[gate_name].get('nand', 0),
                'nor_gates_prelevel': pre_level[gate_name].get('nor', 0),
                'not_gates_prelevel': pre_level[gate_name].get('not', 0),
                'buf_gates_prelevel': pre_level[gate_name].get('buf', 0),
                'xor_gates_prelevel': pre_level[gate_name].get('xor', 0),
                'xnor_gates_prelevel': pre_level[gate_name].get('xnor', 0),
                'dff_gates_prelevel': pre_level[gate_name].get('dff', 0),
                'and_gates_postlevel': post_level[gate_name].get('and', 0),
                'or_gates_postlevel': post_level[gate_name].get('or', 0),
                'nand_gates_postlevel': post_level[gate_name].get('nand', 0),
                'nor_gates_postlevel': post_level[gate_name].get('nor', 0),
                'not_gates_postlevel': post_level[gate_name].get('not', 0),
                'buf_gates_postlevel': post_level[gate_name].get('buf', 0),
                'xor_gates_postlevel': post_level[gate_name].get('xor', 0),
                'xnor_gates_postlevel': post_level[gate_name].get('xnor', 0),
                'dff_gates_postlevel': post_level[gate_name].get('dff', 0)
            }
            writer.writerow(row)
    print(f"Processed file saved to: {args.output_file}")

if __name__ == '__main__':
    main()
