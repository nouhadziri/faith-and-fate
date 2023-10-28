import argparse
import json
import glob
import traceback
from statistics import mean
from tqdm import tqdm

import networkx as nx
from generate_graph_from_scratchpad import extract_input, extract_output, create_graph, HallucinatedNodeException
import re
from collections import defaultdict
import numpy as np


regex_final_output = re.compile("Reconstructing all together, output=\[([-, 0-9])+\]")


def compute_accuracy(gold_answer, generated_answer):
    parsed_answer = extract_output(generated_answer)
    return int(gold_answer == parsed_answer)


def compute_node_type(predict_graph, gold_graph, x):

    # FIXME: change to new task
    def _assign_label(nodes):
        if not nodes:
            return

        next_nodes = set()
        for node in nodes:
            if "label" in predict_graph.nodes[node]:
                continue

            parents = list(predict_graph.predecessors(node))

            # parent node hasn't got label
            if any('label' not in predict_graph.nodes[p] for p in parents):
                next_nodes.add(node)
                continue

            # correct path
            parents_correct = all(predict_graph.nodes[p]['label'] == 'correct' for p in parents)
            node_correct = node in gold_graph.nodes and predict_graph.nodes[node]['value'] == gold_graph.nodes[node]['value']
            if parents_correct and node_correct:
                predict_graph.nodes[node]['label'] = 'correct'
            elif not parents_correct and node_correct:
                predict_graph.nodes[node]['label'] = 'type3'

            # check which type of error
            else:
                operation = predict_graph.edges[(parents[0], node)]['operation']
                parent_values = [predict_graph.nodes[p]['value'] for p in parents]

                if operation == 'max1_zero':
                    assert len(parents) == 1
                    correct_value = max(parent_values[0], 0)
                elif operation == 'max2_zero':
                    assert len(parents) == 2
                    correct_value = max(max(parent_values[0], parent_values[1]), 0)
                elif operation == 'sum':
                    assert len(parents) == 2
                    correct_value = sum(parent_values)
                elif operation == 'equals':
                    assert len(parents) == 2
                    correct_value = (parent_values[0] == parent_values[1])
                elif operation == 'and':
                    assert len(parents) == 2
                    correct_value = (parent_values[0] and parent_values[1])
                elif operation == 'not':
                    assert len(parents) == 1
                    correct_value = (not parent_values[0])
                elif operation == 'mapping_1_2':
                    assert len(parents) == 1
                    correct_value = 1 if parent_values[0] else 2
                elif operation == 'concat':
                    correct_value = ", ".join([str(p) for p in parent_values])
                else:
                    print(f'unrecognized operation: {operation}')
                    correct_value = -1

                if predict_graph.nodes[node]['value'] == correct_value:
                    predict_graph.nodes[node]['label'] = 'type2'
                else:
                    predict_graph.nodes[node]['label'] = 'type1'

            next_nodes.update(list(predict_graph.successors(node)))
            _assign_label(next_nodes)

    # start off recursion by assigning the input as correct, and calling _assign_label with their children
    all_children, base_nodes = [], set()
    for i in range(len(x)):
        input_node_names_i = f"input[{i}]"
        predict_graph.nodes[input_node_names_i]['label'] = 'correct'
        children = list(predict_graph.successors(input_node_names_i))
        all_children.extend(children)
        base_nodes.update({input_node_names_i})

    input_node_names_i = 'can_use_next_item_node[0]'
    predict_graph.nodes[input_node_names_i]['label'] = 'correct'
    children = list(predict_graph.successors(input_node_names_i))
    all_children.extend(children)
    base_nodes.update({input_node_names_i})

    _assign_label(set(all_children))

    stats = {}
    for node in predict_graph.nodes:
        longest_path = 1
        for base_node in list(base_nodes):
            try:
                path = max(nx.all_simple_paths(gold_graph, source=base_node, target=node), key=lambda x: len(x))
                longest_path = max(len(path), longest_path)
            except:
                continue

        for base_node in list(base_nodes):
            try:
                path = max(nx.all_simple_paths(predict_graph, source=base_node, target=node), key=lambda x: len(x))
                longest_path = max(len(path), longest_path)
            except:
                continue

        if longest_path not in stats.keys():
            stats[longest_path] = {'correct': 0, 'type1': 0, 'type2': 0, 'type3': 0, 'accuracy': []}

        if 'label' in predict_graph.nodes[node]:
            stats[longest_path][predict_graph.nodes[node]['label']] += 1
        else:
            if node in gold_graph.nodes and predict_graph.nodes[node]['value'] == gold_graph.nodes[node]['value']:
                label = 'type3'
            else:
                label = 'type1'
            stats[longest_path][label] += 1

    return stats


def compute_width(predict_graph, input_list):
    base_nodes = [f"input[{i}]" for i in range(len(input_list))]

    widths = {}
    for node in base_nodes:
        widths[node] = 0

    for node in base_nodes:
        for c, successors in nx.bfs_successors(predict_graph, source=node):
            for succ in successors:
                widths[succ] = max(widths.get(succ, -1), widths[c] + 1)

    return max(widths.values())


def compute_depth_graph(gold_graph, x):
    examples_with_depth = {}
    all_children, base_nodes = [], set()
    for i in range(len(str(x))):
        base_nodes.update({f"input[{i}]"})

    number_of_nodes = len(gold_graph.nodes)
    longest_path = -float('inf')
    for node in gold_graph.nodes:
        for base_node in list(base_nodes):
            try:
                path = max(nx.all_simple_paths(gold_graph, source=base_node, target=node), key=lambda x: len(x))
                longest_path = max(len(path), longest_path)
                examples_with_depth[node] = longest_path
            except:
                continue
    return longest_path, number_of_nodes


def scratchpad_graph_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratchpad_folder", type=str, help="Path to the scratchpad folder")
    args = parser.parse_args()
    node_types = ['correct', 'type1', 'type2', 'type3']

    node_stats_total, depth_dist = {}, {}
    accuracy_depth_total = defaultdict(list)
    for file in glob.glob(f'{args.scratchpad_folder}/*.json*'):
        print(file.split('/')[-1])
        name = '_'.join(file.split('/')[-1].split('_')[:3])
        depth_dist[name] = []
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

        overall_stats = {}
        accuracy_depth = defaultdict(list)
        scratchpad_parsing_error = 0
        for item in tqdm(data):
            input_list = extract_input(item["question"])
            gold_answer = item["gold answer"]
            generated_answer = item["GPT3 answer"]

            correct_output = extract_output(gold_answer)
            graph_from_input = create_graph(input_list, gold_answer, print_graph=False)
            longest_path, number_of_nodes = compute_depth_graph(graph_from_input, input_list)

            depth_dist[name].append(longest_path)
            try:
                graph_from_scrathcpad = create_graph(input_list, generated_answer, print_graph=False)
            except HallucinatedNodeException:
                scratchpad_parsing_error += 1
                continue
            except AssertionError as e:
                print('-' * 100)
                print(traceback.print_exc())
                print(generated_answer)
                print('-' * 100)
                scratchpad_parsing_error += 1
                continue

            stats = compute_node_type(graph_from_scrathcpad, graph_from_input, input_list)

            width = compute_width(graph_from_input, input_list)
            average_parallelism = number_of_nodes / longest_path
            accuracy = compute_accuracy(correct_output, generated_answer)
            accuracy_depth[average_parallelism].append(accuracy)
            accuracy_depth_total[average_parallelism].append(accuracy)
            for key, values in stats.items():
                if key not in overall_stats:
                    overall_stats[key] = {'correct': 0, 'type1': 0, 'type2': 0, 'type3': 0, 'accuracy': []}
                for node_type in overall_stats[key].keys():
                    overall_stats[key][node_type] += values[node_type]

        print(f'parsing scratchpad: {len(data) - scratchpad_parsing_error} succeed, {scratchpad_parsing_error} fail')
        percent_stats = {}
        for key, values in overall_stats.items():
            percent_stats[key] = {}
            total_nodes = sum([values[t] for t in node_types])
            for t in node_types:
                percent_stats[key][t] = values[t] / total_nodes
        print(json.dumps(percent_stats))

        for layer, values in overall_stats.items():
            if layer not in node_stats_total.keys():
                node_stats_total[layer] = {'correct': 0, 'type1': 0, 'type2': 0, 'type3': 0}
            for t in node_types:
                node_stats_total[layer][t] += values[t]

        for depth, accuracies in accuracy_depth.items():
            print(f"Parallelism {depth} has accuracy: {np.mean(accuracies)}")

    print("*******")
    import collections
    od = collections.OrderedDict(sorted(accuracy_depth_total.items()))
    for depth, accuracies in od.items():
        print(f"Parallelism {depth} has accuracy: {np.mean(accuracies)}")

    percent_stats_overall = {}
    for key, values in node_stats_total.items():
        percent_stats_overall[key] = {}
        total_nodes = sum([values[t] for t in node_types])
        for t in node_types:
            percent_stats_overall[key][t] = values[t] / total_nodes
    print(json.dumps(percent_stats_overall))

    depth_sum1, depth_sum2 = {}, {}
    for name, depths in depth_dist.items():
        if not depths:
            continue
        depth_sum1[name] = (np.percentile(depths, 50), np.percentile(depths, 25), np.percentile(depths, 75))
        depth_sum2[name] = (mean(depths), mean(depths) - np.std(depths), mean(depths) + np.std(depths))
    print(json.dumps(depth_sum2))


if __name__ == "__main__":
    scratchpad_graph_analysis()
