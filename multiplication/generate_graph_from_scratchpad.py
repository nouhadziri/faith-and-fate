import argparse
import json
import logging
import re

import networkx as nx
from build_graph import build_scratchpad_graph
from generate_scratchpads import generate_prompt

logger = logging.getLogger("generate_graph_from_scratchpad_v2")


def extract_numbers(question: str):
    # Define the pattern to match the string
    pattern = r"What is (\d+) times (\d+)\?"
    # Use the re.search() function to find the pattern in the string
    match = re.search(pattern, question)
    # If there is a match, extract the numbers
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return [x, y]
    else:
        print("No match found.")


def create_graph(x: int, y: int, generated_answer: str):
    graph = nx.DiGraph()
    xs = add_input_node(x, "x", graph)
    ys = add_input_node(y, "y", graph)

    lines = generated_answer.split("\n")

    prev_carry_node = None
    prev_pp_node_id = None
    pp = ""
    pp_nodes = []
    x_i, y_i = 0, 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^\d+\.\s+Multiply.+", line):
            m = re.search(
                r"Multiply (\d+) by the digit in the [\w\-]+ place of \d+, which is (\d)\."
                r"( Add the carryover from the previous step to account for this\.)? "
                r"This gives \(?\d+ x \d(\)? \+ (\d))? = (\d+). "
                r"Write down the result (\d+) and carry over the (\d) to the next step\.",
                line,
            )
            m2 = re.search(
                r"Multiply (\d+) by the digit in the [\w\-]+ place of \d+, which is (\d)\."
                r"( Add the carryover from the previous step to account for this\.)? "
                r"This gives \(?\d+ x \d(\)? \+ (\d))? = (\d+). "
                r"Write down the result (\d+)\.",
                line,
            )

            if not m and not m2:
                continue

            matched = m or m2
            ydigit_index = 1
            xdigit_index = 2
            prd_plus_carry_index = 6
            residual_index = 7

            y_digit = int(matched.group(ydigit_index))
            y_node = graph.nodes[ys[y_i]]
            x_digit = int(matched.group(xdigit_index))
            if x_i >= len(xs):
                for j in range(x_i - 1, -1, -1):
                    if str(x)[::-1][j] == str(x_digit):
                        x_i = j
                        break

            x_node = graph.nodes[xs[x_i]]
            idx, idy = xs[x_i], ys[y_i]
            x_i += 1

            xy_rank = x_node["rank"] + y_node["rank"]
            xy_prd_v = x_digit * y_digit

            if f"{idx}{idy}-prd" in graph.nodes:
                continue

            xy_prd = f"{idx}{idy}-prd"
            graph.add_node(xy_prd, type="product", value=xy_prd_v, rank=xy_rank)
            graph.add_edge(idx, xy_prd, operation="multiply")
            graph.add_edge(idy, xy_prd, operation="multiply")

            prev_carryover = matched.group(4)
            if prev_carryover:
                xy_prd_plus_carry = int(matched.group(prd_plus_carry_index))
                graph.add_node(f"{idx}{idy}-prd+", type="residual", value=xy_prd_plus_carry, rank=xy_rank)
                graph.add_edge(xy_prd, f"{idx}{idy}-prd+", operation="add")
                graph.add_edge(prev_carry_node, f"{idx}{idy}-prd+", operation="add")
                xy_prd = f"{idx}{idy}-prd+"

            residual = int(matched.group(residual_index))

            xy_prd_res = xy_prd
            if m:
                carryover_index = 8
                carryover = int(matched.group(carryover_index))
                if carryover > 0:
                    xy_prd_carry_v, xy_prd_res_v = carryover, residual
                    graph.add_node(f"{idx}{idy}-prd_carry", type="carry", value=xy_prd_carry_v, rank=xy_rank + 1)
                    graph.add_edge(xy_prd, f"{idx}{idy}-prd_carry", operation="mod10")
                    graph.add_node(f"{idx}{idy}-prd_res", type="residual", value=xy_prd_res_v, rank=xy_rank)
                    graph.add_edge(xy_prd, f"{idx}{idy}-prd_res", operation="mod10")
                    prev_carry_node, xy_prd_res = f"{idx}{idy}-prd_carry", f"{idx}{idy}-prd_res"

            pp = f"{residual}{pp}"
            new_partial_product = int(pp)
            graph.add_node(f"{idx}{idy}-output", type="output", value=new_partial_product, rank=y_node["rank"])
            graph.add_edge(xy_prd_res, f"{idx}{idy}-output", operation="concat")
            xy_output = f"{idx}{idy}-output"

            if prev_pp_node_id is not None:
                graph.add_edge(prev_pp_node_id, xy_output, operation="concat")
            prev_pp_node_id = xy_output
        elif re.match(r"^\d+\.\s+The partial product for this step.+", line):
            m = re.search(
                r"The partial product for this step is [A-Z]\s*=\s*(\d+),?( which is the concatenation of the digits we found in each step)?\.?",
                line,
            )
            adjusted_pp = int(m.group(1)) * 10 ** y_i
            output_node = f"{get_prefix(prev_pp_node_id)}-output+"
            graph.add_node(
                output_node,
                type="output",
                value=adjusted_pp,
                rank=graph.nodes[prev_pp_node_id]["rank"],
            )
            graph.add_edge(prev_pp_node_id, output_node, operation="exp")
            pp_nodes.append(output_node)
            x_i = 0
            y_i += 1
        elif line.startswith("Now, let's sum the "):
            m = re.search(r"The final answer is ((\d+) x 10*( \+ )?)+ = ((\d+)( \+ )?)+ = (\d+)\.?$", line)
            predicted_answer = int(m.groups()[-1])
            graph.add_node("final_output", type="output", value=predicted_answer, rank=0)
            for node in pp_nodes:
                graph.add_edge(node, "final_output", operation="add")

    return graph


def add_input_node(number, name, graph):
    number = str(number)[::-1]
    nodes = []
    for i, n in enumerate(number):
        graph.add_node(f"{name}{i}", type="input", value=int(n), rank=i)
        nodes.append(f"{name}{i}")
    return nodes


def get_prefix(node_name):
    return "-".join(node_name.split("-")[:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scratchpad_file", type=str, default=None, help="Path to the scratchpad file")

    args = parser.parse_args()

    if args.scratchpad_file.endswith(".json"):
        data = json.load(open(args.scratchpad_file))
    else:
        with open(args.scratchpad_file, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]

    for item in data:
        x, y = extract_numbers(item["question"])
        generated_answer = item["GPT3 answer"]

        if isinstance(generated_answer, (list, tuple)):
            generated_answer = generated_answer[0]
            gold_answer = "Let's perform the multiplication step by step:\n\n" + generate_prompt(x, y)[0][:-4]
        graph_from_scrathcpad = create_graph(x, y, generated_answer)
        gold_graph_scrathcpad = create_graph(x, y, gold_answer)
        graph_from_input = build_scratchpad_graph(x, y)
        print(f"gold graph nodes: {gold_graph_scrathcpad.nodes()}")
        print(f"gold graph edges: {gold_graph_scrathcpad.edges()}")
        print("****************")

        print(f"generated graph nodes: {graph_from_scrathcpad.nodes()}")
        print(f"generated graph edges: {graph_from_scrathcpad.edges()}")
        print("****************")

        print(f"graph v2 nodes: {graph_from_input.nodes()}")
        print(f"graph v2 edges: {graph_from_input.edges()}")
        break


if __name__ == "__main__":
    main()