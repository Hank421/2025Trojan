import re
import csv
import argparse
from collections import defaultdict, deque

def is_bus(signal):
    return '[' in signal and ']' in signal

def is_const(signal):
    return "1'b" in signal or signal.strip() in ("1'b0", "1'b1")

def extract_signals(line):
    # 支援 input/output/wire 宣告解析
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

def parse_verilog(lines):
    gates = []
    gate_inputs = {}
    gate_outputs = {}
    wire_drivers = defaultdict(list)

    primary_inputs, primary_outputs = set(), set()
    dff_inputs, dff_outputs = set(), set()

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
        wire_drivers[output].append(gate_name)

    return gates, gate_inputs, gate_outputs, wire_drivers, primary_inputs, primary_outputs, dff_inputs, dff_outputs

def build_graphs(gate_inputs, gate_outputs):
    fanin = defaultdict(list)
    fanout = defaultdict(list)
    for gate, ins in gate_inputs.items():
        out = gate_outputs[gate]
        for i in ins:
            fanin[out].append(i)
            fanout[i].append(out)
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

def write_csv(output_file, gates, gate_inputs, gate_outputs, lvl_from_input, lvl_to_output, lvl_to_output_ff, lvl_from_input_ff):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'gate_name', 'gate_type', 'output', 'inputs',
            'input_count', 'is_sequential', 'is_constant_input', 'has_bus_signal',
            'level_to_input', 'level_to_output', 'level_to_output_ff', 'level_to_input_ff'
        ])
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
                'level_to_input_ff': lvl_from_input_ff.get(out, -1)
            }
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Verilog netlist file')
    parser.add_argument('output_file', help='CSV output path')
    args = parser.parse_args()

    with open(args.input_file) as f:
        lines = f.readlines()

    gates, gate_inputs, gate_outputs, wire_drivers, primary_inputs, primary_outputs, dff_inputs, dff_outputs = parse_verilog(lines)

    fanin_graph, fanout_graph = build_graphs(gate_inputs, gate_outputs)

    lvl_from_input = bfs(primary_inputs, fanout_graph)
    lvl_to_output = bfs(primary_outputs, fanin_graph)
    lvl_to_output_ff = bfs(dff_outputs, fanin_graph)
    lvl_from_input_ff = bfs(dff_inputs, fanout_graph)

    write_csv(args.output_file, gates, gate_inputs, gate_outputs, lvl_from_input, lvl_to_output, lvl_to_output_ff, lvl_from_input_ff)

if __name__ == '__main__':
    main()
