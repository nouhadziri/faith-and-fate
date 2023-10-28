import argparse
import json
import glob
from tqdm import tqdm

import networkx as nx
from build_graph import build_scratchpad_v2_graph
from generate_graph_from_scratchpad import extract_numbers, create_graph
import re



def compute_pattern_occur(pattern_regex, train_pads, pad_idxes):
    num_hit, node_pad_idxes = 0, []
    for i, train_pad in enumerate(train_pads):
        if i not in pad_idxes:
            continue

        train_pad = train_pad.replace('\n', ' ')
        all_match = re.findall(pattern_regex, train_pad)
        is_hit = len(all_match) > 0

        if is_hit:
            num_hit += 1
            node_pad_idxes.append(i)
    return num_hit, node_pad_idxes


def compute_pattern_occurrence(predict_graph, gold_graph, x, y, train_pads):
    def _count_pattern(nodes):
        if not nodes:
            return

        next_nodes = set()
        for node in nodes:
            if "label" in gold_graph.nodes[node]:
                continue

            parents = list(gold_graph.predecessors(node))
            # parent node hasn't got label
            if any('label' not in gold_graph.nodes[p] for p in parents):
                next_nodes.add(node)
                continue

            # correct path
            parents_correct = all(gold_graph.nodes[p]['label'] == 'correct' for p in parents)
            node_correct = node in predict_graph.nodes() and int(gold_graph.nodes[node]['value']) == predict_graph.nodes[node]['value']
            if parents_correct and node_correct:
                gold_graph.nodes[node]['label'] = 'correct'
            else:
                gold_graph.nodes[node]['label'] = 'wrong'

            parent_pad_idxes = [gold_graph.nodes[p]['pad_idxes'] for p in parents]
            intersect_pad_idxes = set(parent_pad_idxes[0]).intersection(*parent_pad_idxes)

            if not intersect_pad_idxes or node not in predict_graph.nodes():
                num_hit, node_pad_idxes = 0, []
            else:
                pattern_regex = predict_graph.nodes[node]['regex']
                num_hit, node_pad_idxes = compute_pattern_occur(pattern_regex, train_pads, intersect_pad_idxes)
            gold_graph.nodes[node]['num_hit'] = num_hit
            gold_graph.nodes[node]['pad_idxes'] = node_pad_idxes
            next_nodes.update(list(gold_graph.successors(node)))
        _count_pattern(next_nodes)

    all_children, base_nodes = [], set()
    for i in range(len(str(x))):
        for j in range(len(str(y))):
            gold_graph.nodes[f'x{i}']['label'] = 'correct'
            gold_graph.nodes[f'y{j}']['label'] = 'correct'
            children = list(gold_graph.successors(f'x{i}')) + list(gold_graph.successors(f'y{j}'))
            all_children.extend(children)
            base_nodes.update({f'x{i}', f'y{j}'})
            gold_graph.nodes[f'x{i}']['pad_idxes'] = list(range(len(train_pads)))
            gold_graph.nodes[f'y{j}']['pad_idxes'] = list(range(len(train_pads)))
    _count_pattern(set(all_children))

    acc_stats, pattern_stats = {}, {}
    for node in gold_graph.nodes:
        if node in base_nodes:
            continue
        longest_path = 1
        for base_node in list(base_nodes):
            try:
                path = max(nx.all_simple_paths(gold_graph, source=base_node, target=node), key=lambda x: len(x))
                longest_path = max(len(path), longest_path)
            except:
                continue

        if longest_path not in acc_stats.keys():
            acc_stats[longest_path] = {'correct': 0, 'wrong': 0}
            pattern_stats[longest_path] = {'correct': 0, 'wrong': 0}
        acc_stats[longest_path][gold_graph.nodes[node]['label']] += 1
        pattern_stats[longest_path][gold_graph.nodes[node]['label']] += gold_graph.nodes[node]['num_hit']

    return acc_stats, pattern_stats


def scratchpad_graph_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratchpad_folder", type=str, help="Path to the scratchpad folder")
    parser.add_argument("--train_data_path", type=str, help="Path to the training data")

    args = parser.parse_args()

    train_data = [json.loads(l.strip()) for l in open(args.train_data_path, 'r').readlines()]
    train_scatchpads = [x["completion"] for x in train_data]
    print(len(train_scatchpads))

    node_types = ['correct', 'wrong']

    overall_acc_stats, overall_pattern_stats = {}, {}
    for file in glob.glob(f'{args.scratchpad_folder}/*.json*'):
        name = '_'.join(file.split('/')[-1].split('_')[:3])
        print(file.split('/')[-1])
        if file.endswith(".json"):
            data = json.load(open(file))
        else:
            with open(file, "r") as f:
                data = []
                for line in f.readlines():
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except:
                            continue

        scratchpad_parsing_error = 0
        for item in tqdm(data):
            x, y = extract_numbers(item["question"])
            generated_answer = item["GPT3 answer"]
            if type(generated_answer) == list:
                generated_answer = generated_answer[0]
            try:
                graph_from_scrathcpad = create_graph(x, y, generated_answer)
                graph_from_input = build_scratchpad_v2_graph(x, y)
            except:
                scratchpad_parsing_error += 1
                continue

            acc_stats, pattern_stats = compute_pattern_occurrence(graph_from_scrathcpad, graph_from_input, x, y, train_scatchpads)
            for key, values in acc_stats.items():
                if key not in overall_acc_stats:
                    overall_acc_stats[key] = {'correct': 0, 'wrong': 0}
                for node_type in node_types:
                    overall_acc_stats[key][node_type] += values[node_type]

            for key, values in pattern_stats.items():
                if key not in overall_pattern_stats:
                    overall_pattern_stats[key] = {'correct': 0, 'wrong': 0}
                for node_type in node_types:
                    overall_pattern_stats[key][node_type] += values[node_type]

        print(f'parsing scratchpad: {len(data) - scratchpad_parsing_error} succeed, {scratchpad_parsing_error} fail')

    print(json.dumps(overall_acc_stats))
    print(json.dumps(overall_pattern_stats))

    print("***************************")

    percent_acc_stats = {}
    for key, values in overall_acc_stats.items():
        percent_acc_stats[key] = {}
        total_nodes = sum([values[t] for t in node_types])
        for t in node_types:
            percent_acc_stats[key][t] = values[t] / total_nodes
    print(json.dumps(percent_acc_stats))

    avg_pattern_stats = {}
    for key, values in overall_pattern_stats.items():
        avg_pattern_stats[key] = {}
        for t in node_types:
            if overall_acc_stats[key][t]:
                avg_pattern_stats[key][t] = values[t] / overall_acc_stats[key][t]
    print(json.dumps(avg_pattern_stats))


if __name__ == "__main__":
    scratchpad_graph_analysis()
