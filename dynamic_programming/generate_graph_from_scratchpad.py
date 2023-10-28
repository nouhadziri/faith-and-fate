import argparse
import ast
import json
import logging
import networkx as nx
import re
from graphviz import Source
from networkx.drawing.nx_pydot import to_pydot


class HallucinatedNodeException(Exception):
    pass


logger = logging.getLogger("generate_graph_from_scratchpad_v2")

regex_dp_N_1 = re.compile("dp\[([0-9]+)\] = max\(input\[([0-9]+)\], 0\) = max\(([-0-9]+), 0\) = ([-0-9]+)")
regex_dp_N_2 = re.compile(
    "dp\[([0-9]+)\] = max\(input\[([0-9]+)\], input\[([0-9]+)\], 0\) = max\(([-0-9]+), ([-0-9]+), 0\) = ([-0-9]+)")
regex_dp_i = re.compile(
    "dp\[([0-9]+)\] = max\(dp\[([0-9]+)\], input\[([0-9]+)\] \+ dp\[([0-9]+)\], 0\) = max\(([-0-9]+), ([-0-9]+) \+ ([-0-9]+), 0\) = ([-0-9]+)")
filler_text = re.compile(
    "Finally, we reconstruct the lexicographically smallest subsequence that fulfills the task objective by selecting numbers as follows. We store the result on a list named \"output\".\n\nLet can_use_next_item = True.")

regex_reconstruction_dp_i_combo = re.compile(
    "Since dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \+ dp\[([0-9]+)\] \(([-0-9]+) (?:!=|==) ([-0-9]+) \+ ([-0-9]+)\) (?:or|and) can_use_next_item == (False|True), we store output\[([0-9]+)\] = (2|1). We update can_use_next_item = (True|False).")
regex_reconstruction_dp_N_2_combo = re.compile(
    "Since dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \(([-0-9]+) (?:!=|==) ([-0-9]+)\) (?:or|and) can_use_next_item == (False|True), we store output\[([0-9]+)\] = (2|1). We update can_use_next_item = (True|False).")  # ok
regex_reconstruction_dp_N_1_combo = re.compile(
    "Since dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \(([-0-9]+) (?:!=|==) ([-0-9]+)\) (?:or|and) can_use_next_item == (False|True), we store output\[([0-9]+)\] = (2|1).")
regex_final_output = re.compile("Reconstructing all together, output=\[([-, 0-9]+)\]")

regex_input = re.compile("Let\'s solve input = \[([-, 0-9]+)\]")


def extract_input(question: str):
    assert question.startswith("Let\'s solve input = ")
    question = question[len("Let\'s solve input = "):].split('\n')[0].strip(' ').strip('.')
    return ast.literal_eval(question)


def extract_output(gold_answer: str):
    regex_match = regex_final_output.search(gold_answer)
    numbers = regex_match.groups()[0].split(',')
    return [int(e) for e in numbers]


def do_dp_N_1_assignment(scratchpad_gold, input_node_names, dp_node_names, graph):
    accum_idx = 0
    regex_match = regex_dp_N_1.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, False
    min_idx_1, max_idx_1 = regex_match.span()
    accum_idx += max_idx_1

    idx_dp_i, idx_input_i, input_i, dp_i = regex_match.groups()
    idx_dp_i, idx_input_i = int(idx_dp_i), int(idx_input_i)
    dp_i = int(dp_i)

    graph.add_node(dp_node_names[idx_dp_i], type="max1_zero", value=dp_i,
                   regex=f"dp\[([0-9]+)\] = max\(input\[([0-9]+)\], 0\) = max\({dp_i}, 0\) = {dp_i}")
    graph.add_edge(input_node_names[idx_input_i], dp_node_names[idx_dp_i], operation="max1_zero")
    return graph, accum_idx, True


def do_dp_N_2_assignment(scratchpad_gold, accum_idx, input_node_names, dp_node_names, graph):
    regex_match = regex_dp_N_2.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, False
    min_idx_2, max_idx_2 = regex_match.span()

    idx_dp_N_2, idx_input_N_2, idx_input_N_1, input_N_2, input_N_1, result = regex_match.groups()
    idx_dp_N_2, idx_input_N_2, idx_input_N_1 = int(idx_dp_N_2), int(idx_input_N_2), int(idx_input_N_1)
    result = int(result)

    graph.add_node(dp_node_names[idx_dp_N_2], type="max2_zero", value=result,
                   regex=f"dp\[([0-9]+)\] = max\(input\[([0-9]+)\], input\[([0-9]+)\], 0\) = max\({input_N_2}, {input_N_1}, 0\) = {result}")
    graph.add_edge(input_node_names[idx_input_N_1], dp_node_names[idx_dp_N_2], operation="max2_zero")
    graph.add_edge(input_node_names[idx_input_N_2], dp_node_names[idx_dp_N_2], operation="max2_zero")

    assert scratchpad_gold[accum_idx:accum_idx + min_idx_2].isspace()
    accum_idx += max_idx_2
    return graph, accum_idx, True


def do_dp_i_assignment(scratchpad_gold, accum_idx, input_node_names, dp_node_names, graph):
    regex_match = regex_dp_i.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, False
    min_idx_3, max_idx_3 = regex_match.span()
    assert scratchpad_gold[accum_idx:accum_idx + min_idx_3].isspace()
    accum_idx += max_idx_3

    idx_dp_i, idx_dp_i_plus_1, idx_input_i, idx_dp_i_plus_2, \
    dp_i_plus_1, input_i, dp_i_plus_2, result = regex_match.groups()
    result = int(result)

    idx_dp_i, idx_dp_i_plus_1, idx_input_i, idx_dp_i_plus_2 = \
        int(idx_dp_i), int(idx_dp_i_plus_1), int(idx_input_i), int(idx_dp_i_plus_2)

    node_name_sum = f"{input_node_names[idx_input_i]} + {dp_node_names[idx_dp_i_plus_2]}"
    graph.add_node(node_name_sum, type="sum", value=int(input_i) + int(dp_i_plus_2),
                   regex='input\[([0-9]+)\] \+ dp\[([0-9]+)\]')
    graph.add_edge(input_node_names[idx_input_i], node_name_sum, operation="sum")
    graph.add_edge(dp_node_names[idx_dp_i_plus_2], node_name_sum, operation="sum")

    node_name = dp_node_names[idx_dp_i]
    graph.add_node(node_name, type="max2_zero", value=result,
                   regex=f"dp\[([0-9]+)\] = max\(dp\[([0-9]+)\], input\[([0-9]+)\] \+ dp\[([0-9]+)\], 0\) = max\({dp_i_plus_1}, {input_i} \+ {dp_i_plus_2}, 0\) = {result}")
    graph.add_edge(dp_node_names[idx_dp_i_plus_1], node_name, operation="max2_zero")
    graph.add_edge(node_name_sum, node_name, operation="max2_zero")

    return graph, accum_idx, True


def do_reconstruct_dp_i(scratchpad_gold, accum_idx, idx_can_use_next_item_node_i,
                        input_node_names, dp_node_names, can_use_next_item_node_names, graph):
    regex_match = regex_reconstruction_dp_i_combo.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, idx_can_use_next_item_node_i, None, False

    min_idx_3, max_idx_3 = regex_match.span()
    assert scratchpad_gold[accum_idx:accum_idx + min_idx_3].isspace(), scratchpad_gold[accum_idx:accum_idx + min_idx_3]
    accum_idx += max_idx_3

    idx_dp_i, idx_input_i, idx_dp_i_plus_2, dp_i, input_i, dp_i_plus_2, can_use_next_item_i, idx_output_i, output_i, can_use_next_item_i_plus_1 = regex_match.groups()
    idx_dp_i, idx_input_i, idx_dp_i_plus_2, idx_output_i = \
        int(idx_dp_i), int(idx_input_i), int(idx_dp_i_plus_2), int(idx_output_i)

    dp_i = int(dp_i)
    input_i = int(input_i)
    dp_i_plus_2 = int(dp_i_plus_2)
    output_i = int(output_i)
    can_use_next_item_i_plus_1 = bool(can_use_next_item_i_plus_1)

    # the model hallucinated a node name that does not exist. We cannot recover from this parsing mistake
    if idx_dp_i_plus_2 >= len(dp_node_names) or idx_input_i >= len(input_node_names):
        raise HallucinatedNodeException()

    summed = f"{input_node_names[idx_input_i]} + {dp_node_names[idx_dp_i_plus_2]}"
    node_name_equals = f"{dp_node_names[idx_dp_i]} == {summed}"
    graph.add_node(node_name_equals, type="equals",
                   value=(dp_i == input_i + dp_i_plus_2),
                   regex=f'dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \+ dp\[([0-9]+)\] \({dp_i} (?:!=|==) {input_i} \+ {dp_i_plus_2}\)')  # FIXME?: scratchpad doesn't sum this
    graph.add_edge(dp_node_names[idx_dp_i], node_name_equals, operation="equals")
    graph.add_edge(summed, node_name_equals, operation="equals")

    node_name_will_use_number_for_solution_node = f"({node_name_equals}) and {can_use_next_item_node_names[idx_can_use_next_item_node_i]}"
    graph.add_node(node_name_will_use_number_for_solution_node, type="and",
                   value=(dp_i == input_i + dp_i_plus_2) and can_use_next_item_i,
                   regex=f'dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \+ dp\[([0-9]+)\] \({dp_i} (?:!=|==) {input_i} \+ {dp_i_plus_2}\) (?:or|and) can_use_next_item == {can_use_next_item_i}')
    graph.add_edge(node_name_equals, node_name_will_use_number_for_solution_node, operation="and")
    graph.add_edge(can_use_next_item_node_names[idx_can_use_next_item_node_i],
                   node_name_will_use_number_for_solution_node, operation="and")

    node_name_output_i = f"output[{idx_output_i}]"
    graph.add_node(node_name_output_i, type="mapping_1_2", value=output_i,
                   regex=f'we store output\[([0-9]+)\] = {output_i}')
    graph.add_edge(node_name_will_use_number_for_solution_node, node_name_output_i, operation="mapping_1_2")

    idx_can_use_next_item_node_i += 1
    node_name_can_use_next_item_i_plus_1 = can_use_next_item_node_names[idx_can_use_next_item_node_i]
    graph.add_node(node_name_can_use_next_item_i_plus_1, type="not", value=can_use_next_item_i_plus_1,
                   regex=f'We update can_use_next_item = {can_use_next_item_i_plus_1}')
    graph.add_edge(node_name_will_use_number_for_solution_node,
                   node_name_can_use_next_item_i_plus_1, operation="not")

    return graph, accum_idx, idx_can_use_next_item_node_i, node_name_output_i, True


def do_reconstruct_dp_N_2(scratchpad_gold, accum_idx, idx_can_use_next_item_node_i,
                          input_node_names, dp_node_names, can_use_next_item_node_names, graph):

    regex_match = regex_reconstruction_dp_N_2_combo.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, idx_can_use_next_item_node_i, None, False

    min_idx_3, max_idx_3 = regex_match.span()
    assert scratchpad_gold[accum_idx:accum_idx + min_idx_3].isspace(), scratchpad_gold[accum_idx:accum_idx + min_idx_3]
    accum_idx += max_idx_3

    idx_dp_n_2, idx_input_n_2, dp_n_2, input_n_2, can_use_next_item_n_2, idx_output_n_2, output_n_2, can_use_next_item_n_1 = regex_match.groups()
    idx_dp_n_2, idx_input_n_2, idx_output_n_2 = int(idx_dp_n_2), int(idx_input_n_2), int(idx_output_n_2)

    dp_n_2 = int(dp_n_2)
    input_n_2 = int(input_n_2)
    can_use_next_item_n_2 = bool(can_use_next_item_n_2)
    can_use_next_item_n_1 = bool(can_use_next_item_n_1)
    output_n_2 = int(output_n_2)

    node_name_equals = f"{dp_node_names[idx_dp_n_2]} == {input_node_names[idx_input_n_2]}"
    graph.add_node(node_name_equals, type="equals", value=(dp_n_2 == input_n_2),
                   regex=f'dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \({dp_n_2} (?:!=|==) {input_n_2}\)')
    graph.add_edge(dp_node_names[idx_dp_n_2], node_name_equals, operation="equals")
    graph.add_edge(input_node_names[idx_input_n_2], node_name_equals, operation="equals")

    node_name_will_use_number_for_solution_node = \
        f"({node_name_equals}) and {can_use_next_item_node_names[idx_can_use_next_item_node_i]}"
    graph.add_node(node_name_will_use_number_for_solution_node, type="and",
                   value=(dp_n_2 == input_n_2) and can_use_next_item_n_2,
                   regex=f'dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \({dp_n_2} (?:!=|==) {input_n_2}\) (?:or|and) can_use_next_item == {can_use_next_item_n_2}')
    graph.add_edge(node_name_equals, node_name_will_use_number_for_solution_node, operation="and")
    graph.add_edge(can_use_next_item_node_names[idx_can_use_next_item_node_i],
                   node_name_will_use_number_for_solution_node, operation="and")

    node_name_output_i = f"output[{idx_output_n_2}]"
    graph.add_node(node_name_output_i, type="mapping_1_2", value=output_n_2,
                   regex=f'we store output\[([0-9]+)\] = {output_n_2}')
    graph.add_edge(node_name_will_use_number_for_solution_node, node_name_output_i, operation="mapping_1_2")

    idx_can_use_next_item_node_i += 1
    node_name_can_use_next_item_i_plus_1 = can_use_next_item_node_names[idx_can_use_next_item_node_i]
    graph.add_node(node_name_can_use_next_item_i_plus_1, type="not", value=can_use_next_item_n_1,
                   regex=f'We update can_use_next_item = {can_use_next_item_n_1}')
    graph.add_edge(node_name_will_use_number_for_solution_node,
                   node_name_can_use_next_item_i_plus_1, operation="not")

    return graph, accum_idx, idx_can_use_next_item_node_i, node_name_output_i, True


def do_reconstruct_dp_N_1(scratchpad_gold, accum_idx, idx_can_use_next_item_node_i,
                          input_node_names, dp_node_names, can_use_next_item_node_names, graph):

    regex_match = regex_reconstruction_dp_N_1_combo.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, idx_can_use_next_item_node_i, None, False

    min_idx_3, max_idx_3 = regex_match.span()
    assert scratchpad_gold[accum_idx:accum_idx + min_idx_3].isspace(), scratchpad_gold[accum_idx:accum_idx + min_idx_3]
    accum_idx += max_idx_3

    idx_dp_n_1, idx_input_n_1, dp_n_1, input_n_1, can_use_next_item_n_1, idx_output_n_1, output_n_1 = \
        regex_match.groups()
    idx_dp_n_1, idx_input_n_1, idx_output_n_1 = int(idx_dp_n_1), int(idx_input_n_1), int(idx_output_n_1)

    dp_n_1 = int(dp_n_1)
    input_n_1 = int(input_n_1)
    can_use_next_item_n_1 = bool(can_use_next_item_n_1)
    output_n_1 = int(output_n_1)

    node_name_equals = f"{dp_node_names[idx_dp_n_1]} == {input_node_names[idx_input_n_1]}"
    graph.add_node(node_name_equals, type="equals", value=(dp_n_1 == input_n_1),
                   regex=f'dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \({dp_n_1} (?:!=|==) {input_n_1}\)')
    graph.add_edge(dp_node_names[idx_dp_n_1], node_name_equals, operation="equals")
    graph.add_edge(input_node_names[idx_input_n_1], node_name_equals, operation="equals")

    node_name_will_use_number_for_solution_node = f"({node_name_equals}) and {can_use_next_item_node_names[idx_can_use_next_item_node_i]}"
    graph.add_node(node_name_will_use_number_for_solution_node, type="and",
                   value=(dp_n_1 == input_n_1) and can_use_next_item_n_1,
                   regex=f'dp\[([0-9]+)\] (?:!=|==) input\[([0-9]+)\] \({dp_n_1} (?:!=|==) {input_n_1}\) (?:or|and) can_use_next_item == {can_use_next_item_n_1}')
    graph.add_edge(node_name_equals, node_name_will_use_number_for_solution_node, operation="and")
    graph.add_edge(can_use_next_item_node_names[idx_can_use_next_item_node_i], node_name_will_use_number_for_solution_node, operation="and")

    node_name_output_i = f"output[{idx_output_n_1}]"
    graph.add_node(node_name_output_i, type="mapping_1_2", value=output_n_1,
                   regex=f'we store output\[([0-9]+)\] = {output_n_1}')
    graph.add_edge(node_name_will_use_number_for_solution_node, node_name_output_i, operation="mapping_1_2")

    return graph, accum_idx, idx_can_use_next_item_node_i, node_name_output_i, True


def do_get_final_answer(scratchpad_gold, accum_idx, all_output_node_names, graph):
    regex_match = regex_final_output.search(scratchpad_gold[accum_idx:])
    assert regex_match

    min_idx_3, max_idx_3 = regex_match.span()
    assert scratchpad_gold[accum_idx:accum_idx + min_idx_3].isspace()
    accum_idx += max_idx_3

    final_output = regex_match.groups()

    final_output_node_name = ", ".join(all_output_node_names)
    final_output_str = ", ".join(final_output)
    graph.add_node(final_output_node_name, type="concat", value=final_output_str,
                   regex=f'Reconstructing all together, output=\[{final_output_str}\]')
    for output_node_name in all_output_node_names:
        graph.add_edge(output_node_name, final_output_node_name, operation="concat")

    return graph, accum_idx


def do_parse_filler(scratchpad_gold, accum_idx, input_node_names, dp_node_names, graph):
    regex_match = filler_text.search(scratchpad_gold[accum_idx:])
    if regex_match is None:
        return graph, accum_idx, False

    min_idx_3, max_idx_3 = regex_match.span()
    assert scratchpad_gold[accum_idx:accum_idx + min_idx_3].isspace(), scratchpad_gold[accum_idx:accum_idx + min_idx_3]
    accum_idx += max_idx_3
    return graph, accum_idx, True


def create_graph(x: list, scratchpad_gold: str, print_graph=False):
    """
    this should parse the scratchpad and generate a Graphviz graph.

    x is the input.

    Consider not using graphviz for speed.
    """

    N = len(x)
    graph = nx.DiGraph()

    # add input nodes
    input_node_names = []
    for i, value in enumerate(x):
        graph.add_node(f"input[{i}]", type="input", value=value, regex=str(value))
        input_node_names.append(f"input[{i}]")

    dp_node_names = [f'dp[{i}]' for i in range(N)]

    can_use_next_item_node_names = [f'can_use_next_item_node[{i}]' for i in range(2 * N)]
    graph, accum_idx, _ = do_dp_N_1_assignment(scratchpad_gold, input_node_names, dp_node_names, graph)
    graph, accum_idx, _ = do_dp_N_2_assignment(scratchpad_gold, accum_idx, input_node_names, dp_node_names, graph)

    successful_parse = True
    while successful_parse:
        graph, accum_idx, successful_parse = do_dp_i_assignment(
            scratchpad_gold, accum_idx, input_node_names, dp_node_names, graph)

    graph, accum_idx, _ =\
        do_parse_filler(scratchpad_gold, accum_idx, input_node_names, dp_node_names, graph)

    current_can_use_next_item_node_idx = 0
    graph.add_node(can_use_next_item_node_names[current_can_use_next_item_node_idx], type="input",
                   value=True, regex='Let can_use_next_item = True')  # FIXME, not really input

    all_output_node_names = []

    successful_parse = True
    while successful_parse:
        graph, accum_idx, current_can_use_next_item_node_idx, new_output_node_name, successful_parse = \
            do_reconstruct_dp_i(
                scratchpad_gold, accum_idx, current_can_use_next_item_node_idx, input_node_names, dp_node_names,
                can_use_next_item_node_names, graph
            )
        if new_output_node_name:
            all_output_node_names.append(new_output_node_name)

    # few-shot GPT4 may perform more than one step of N-2/N-1 reconstruction and we need to cover that case
    successful_parse = True
    while successful_parse:
        graph, accum_idx, current_can_use_next_item_node_idx, new_output_node_name, successful_parse = \
            do_reconstruct_dp_N_2(
                scratchpad_gold, accum_idx, current_can_use_next_item_node_idx, input_node_names, dp_node_names,
                can_use_next_item_node_names, graph
            )
        if new_output_node_name:
            all_output_node_names.append(new_output_node_name)

    successful_parse = True
    while successful_parse:
        graph, accum_idx, current_can_use_next_item_node_idx, new_output_node_name, successful_parse = \
            do_reconstruct_dp_N_1(
                scratchpad_gold, accum_idx, current_can_use_next_item_node_idx, input_node_names, dp_node_names,
                can_use_next_item_node_names, graph
            )
        if new_output_node_name:
            all_output_node_names.append(new_output_node_name)

    graph, accum_idx = do_get_final_answer(
        scratchpad_gold, accum_idx, all_output_node_names, graph
    )

    if print_graph:
        for node in graph.nodes:
            graph.nodes[node]['label'] = f"{node} (val={graph.nodes[node]['value']})"

        dot = to_pydot(graph).to_string()
        src = Source(dot)
        src.view()

    return graph


def get_prefix(node_name):
    return "-".join(node_name.split("-")[:-1])
