import networkx as nx


def get_digit(x):
    return len(str(x))


def get_base_operation(operate, x=None, y=None):
    """
    :param operate: mul, add, mod 10
    :param x: one digit for mul, add; two digit for mod 10
    :param y: one digit for mul, add; None for mod 10
    :return: result for mul, add; carry and residual for mod 10
    """
    if operate == 'mul':
        assert get_digit(x) == 1 and get_digit(y) == 1, f'mul {x} and {y}'
        return x * y
    elif operate == 'add':
        assert get_digit(x) == 1 and get_digit(y) == 1, f'add {x} and {y}'
        return x + y
    elif operate == 'mod10':
        assert get_digit(x) == 2 and y is None, f'{x} mod 10'
        return x // 10, x % 10
    else:
        return NotImplementedError


def add_input_node(number, name, graph):
    number = str(number)[::-1]
    nodes = []
    for i, n in enumerate(number):
        graph.add_node(f'{name}{i}', type='input', value=int(n), rank=i)
        nodes.append(f'{name}{i}')
    return nodes


def get_output(graph, output_dict):
    output = 0
    for rank, node_name in output_dict.items():
        output += 10 ** rank * graph.nodes[node_name]['value']
    return output


def get_prefix(node_name):
    return '-'.join(node_name.split('-')[:-1])


def build_graph(number_x, number_y):
    graph = nx.DiGraph()
    xs = add_input_node(number_x, 'x', graph)
    ys = add_input_node(number_y, 'y', graph)

    outputs_to_sum = None
    for y in ys:
        outputs, carry = {}, None
        for x in xs:
            xy_prd_v = get_base_operation('mul', graph.nodes[x]['value'], graph.nodes[y]['value'])
            xy_rank = graph.nodes[x]['rank'] + graph.nodes[y]['rank']
            graph.add_node(f'{x}{y}-prd', type='product', value=xy_prd_v, rank=xy_rank)
            graph.add_edge(x, f'{x}{y}-prd', operation='multiply')
            graph.add_edge(y, f'{x}{y}-prd', operation='multiply')

            xy_prd_carry, xy_prd_res = None, None
            if get_digit(xy_prd_v) == 1:
                xy_prd_res = f'{x}{y}-prd'
            else:
                assert get_digit(xy_prd_v) == 2, f'{graph.nodes[x]["value"]} * {graph.nodes[y]["value"]} = {xy_prd_v}'
                xy_prd_carry_v, xy_prd_res_v = get_base_operation('mod10', xy_prd_v)
                graph.add_node(f'{x}{y}-prd_carry', type='carry', value=xy_prd_carry_v, rank=xy_rank + 1)
                graph.add_edge(f'{x}{y}-prd', f'{x}{y}-prd_carry', operation='mod10')
                graph.add_node(f'{x}{y}-prd_res', type='residual', value=xy_prd_res_v, rank=xy_rank)
                graph.add_edge(f'{x}{y}-prd', f'{x}{y}-prd_res', operation='mod10')
                xy_prd_carry, xy_prd_res = f'{x}{y}-prd_carry', f'{x}{y}-prd_res'

            if carry is not None:
                xy_prd_res_sum_v = get_base_operation('add', graph.nodes[xy_prd_res]['value'], graph.nodes[carry]['value'])
                graph.add_node(f'{x}{y}-prd_res+', type='residual', value=xy_prd_res_sum_v, rank=xy_rank)
                graph.add_edge(xy_prd_res, f'{x}{y}-prd_res+', operation='add')
                graph.add_edge(carry, f'{x}{y}-prd_res+', operation='add')

                if get_digit(xy_prd_res_sum_v) == 1:
                    xy_prd_res = f'{x}{y}-prd_res+'
                else:
                    assert get_digit(xy_prd_res_sum_v) == 2, f'{graph.nodes[xy_prd_res]["value"]} + {graph.nodes[carry]["value"]} = {xy_prd_res_sum_v}'
                    xy_prd_res_carry_v, xy_prd_resn_v = get_base_operation('mod10', xy_prd_res_sum_v)
                    graph.add_node(f'{x}{y}-prd_res_carry', type='carry', value=xy_prd_res_carry_v, rank=xy_rank + 1)
                    graph.add_edge(f'{x}{y}-prd_res+', f'{x}{y}-prd_res_carry', operation='mod10')
                    graph.add_node(f'{x}{y}-prd_res^', type='residual', value=xy_prd_resn_v, rank=xy_rank)
                    graph.add_edge(f'{x}{y}-prd_res+', f'{x}{y}-prd_res^', operation='mod10')
                    xy_prd_res_carry, xy_prd_res = f'{x}{y}-prd_res_carry', f'{x}{y}-prd_res^'

                    if xy_prd_carry is None:
                        xy_prd_carry = f'{x}{y}-prd_res_carry'
                    else:
                        xy_prd_carry_sum_v = get_base_operation('add', graph.nodes[xy_prd_carry]['value'], graph.nodes[xy_prd_res_carry]['value'])
                        graph.add_node(f'{x}{y}-prd_carry+', type='carry', value=xy_prd_carry_sum_v, rank=xy_rank + 1)
                        graph.add_edge(xy_prd_carry, f'{x}{y}-prd_carry+', operation='add')
                        graph.add_edge(xy_prd_res_carry, f'{x}{y}-prd_carry+', operation='add')
                        xy_prd_carry = f'{x}{y}-prd_carry+'
                        assert get_digit(xy_prd_carry_sum_v) == 1, f'{graph.nodes[xy_prd_carry]["value"]} + {graph.nodes[xy_prd_res_carry]["value"]} = {xy_prd_carry_sum_v}'

            carry = xy_prd_carry
            outputs[graph.nodes[xy_prd_res]["rank"]] = xy_prd_res

        if carry is not None:
            outputs[graph.nodes[carry]["rank"]] = carry

        if outputs_to_sum is None:
            outputs_to_sum = outputs
        else:
            previous_nodes = [(rank, node_name) for rank, node_name in outputs_to_sum.items()]
            previous_nodes = sorted(previous_nodes, key=lambda rn: rn[0])

            output_carry = None
            for rank, node_name in previous_nodes:
                output_res = node_name
                ori_new_carry, ori_new_res = None, None

                if rank in outputs.keys():
                    ori_new_sum_v = get_base_operation('add', graph.nodes[output_res]['value'], graph.nodes[outputs[rank]]['value'])
                    ori_prefix, new_prefix = get_prefix(output_res), get_prefix(outputs[rank])
                    graph.add_node(f'{ori_prefix}-{new_prefix}-sum', type='residual', value=ori_new_sum_v, rank=rank)
                    graph.add_edge(output_res, f'{ori_prefix}-{new_prefix}-sum', operation='add')
                    graph.add_edge(outputs[rank], f'{ori_prefix}-{new_prefix}-sum', operation='add')

                    if get_digit(ori_new_sum_v) == 1:
                        ori_new_res = f'{ori_prefix}-{new_prefix}-sum'
                    else:
                        assert get_digit(ori_new_sum_v) == 2, f'{graph.nodes[output_res]["value"]} + {graph.nodes[outputs[rank]]["value"]} = {ori_new_sum_v}'
                        ori_new_carry_v, ori_new_res_v = get_base_operation('mod10', ori_new_sum_v)
                        graph.add_node(f'{ori_prefix}-{new_prefix}-carry', type='carry', value=ori_new_carry_v, rank=rank + 1)
                        graph.add_edge(f'{ori_prefix}-{new_prefix}-sum', f'{ori_prefix}-{new_prefix}-carry', operation='mod10')
                        graph.add_node(f'{ori_prefix}-{new_prefix}-res', type='residual', value=ori_new_res_v, rank=rank)
                        graph.add_edge(f'{ori_prefix}-{new_prefix}-sum', f'{ori_prefix}-{new_prefix}-res', operation='mod10')
                        ori_new_carry, ori_new_res = f'{ori_prefix}-{new_prefix}-carry', f'{ori_prefix}-{new_prefix}-res'

                    if output_carry is not None:
                        ori_new_res_sum_v = get_base_operation('add', graph.nodes[ori_new_res]['value'], graph.nodes[output_carry]['value'])
                        graph.add_node(f'{ori_prefix}-{new_prefix}-res+', type='residual', value=ori_new_res_sum_v, rank=rank)
                        graph.add_edge(ori_new_res, f'{ori_prefix}-{new_prefix}-res+', operation='add')
                        graph.add_edge(output_carry, f'{ori_prefix}-{new_prefix}-res+', operation='add')

                        if get_digit(ori_new_res_sum_v) == 1:
                            ori_new_res = f'{ori_prefix}-{new_prefix}-res+'
                        else:
                            assert get_digit(ori_new_res_sum_v) == 2, f'{graph.nodes[ori_new_res]["value"]} + {graph.nodes[output_carry]["value"]} = {ori_new_res_sum_v}'
                            ori_new_res_carry_v, ori_new_resn_v = get_base_operation('mod10', ori_new_res_sum_v)
                            graph.add_node(f'{ori_prefix}-{new_prefix}-res_carry', type='carry', value=ori_new_res_carry_v, rank=rank + 1)
                            graph.add_edge(f'{ori_prefix}-{new_prefix}-res+', f'{ori_prefix}-{new_prefix}-res_carry', operation='mod10')
                            graph.add_node(f'{ori_prefix}-{new_prefix}-res^', type='residual', value=ori_new_resn_v, rank=rank)
                            graph.add_edge(f'{ori_prefix}-{new_prefix}-res+', f'{ori_prefix}-{new_prefix}-res^', operation='mod10')
                            ori_new_res_carry, ori_new_res = f'{ori_prefix}-{new_prefix}-res_carry', f'{ori_prefix}-{new_prefix}-res^'

                            if ori_new_carry is None:
                                ori_new_carry = f'{ori_prefix}-{new_prefix}-res_carry'
                            else:
                                ori_new_carry_sum_v = get_base_operation('add', graph.nodes[ori_new_carry]['value'], graph.nodes[ori_new_res_carry]['value'])
                                graph.add_node(f'{ori_prefix}-{new_prefix}-carry+', type='carry', value=ori_new_carry_sum_v, rank=rank + 1)
                                graph.add_edge(ori_new_carry, f'{ori_prefix}-{new_prefix}-carry+', operation='add')
                                graph.add_edge(ori_new_res_carry, f'{ori_prefix}-{new_prefix}-carry+', operation='add')
                                ori_new_carry = f'{ori_prefix}-{new_prefix}-carry+'
                                assert get_digit(ori_new_carry_sum_v) == 1, f'{graph.nodes[ori_new_carry]["value"]} + {graph.nodes[ori_new_res_carry]["value"]} = {ori_new_carry_sum_v}'

                else:
                    ori_new_res = output_res

                output_carry = ori_new_carry
                outputs_to_sum[graph.nodes[ori_new_res]["rank"]] = ori_new_res
                assert graph.nodes[ori_new_res]["rank"] == rank

            previous_ranks = [x[0] for x in previous_nodes]
            new_nodes = [(rank, node_name) for rank, node_name in outputs.items() if rank not in previous_ranks]
            new_nodes = sorted(new_nodes, key=lambda rn: rn[0])

            for rank, node_name in new_nodes:
                output_res = node_name
                new_carry, new_res = None, None

                if output_carry is not None:
                    new_prefix = get_prefix(output_res)
                    new_res_sum_v = get_base_operation('add', graph.nodes[output_res]['value'], graph.nodes[output_carry]['value'])
                    graph.add_node(f'{new_prefix}-res+', type='residual', value=new_res_sum_v, rank=rank)
                    graph.add_edge(output_res, f'{new_prefix}-res+', operation='add')
                    graph.add_edge(output_carry, f'{new_prefix}-res+', operation='add')

                    if get_digit(new_res_sum_v) == 1:
                        new_res = f'{new_prefix}-res+'
                    else:
                        assert get_digit(new_res_sum_v) == 2, f'{graph.nodes[output_res]["value"]} + {graph.nodes[output_carry]["value"]} = {new_res_sum_v}'
                        new_res_carry_v, new_resn_v = get_base_operation('mod10', new_res_sum_v)
                        graph.add_node(f'{new_prefix}-res_carry', type='carry', value=new_res_carry_v, rank=rank + 1)
                        graph.add_edge(f'{new_prefix}-res+', f'{new_prefix}-res_carry', operation='mod10')
                        graph.add_node(f'{new_prefix}-res^', type='residual', value=new_resn_v, rank=rank)
                        graph.add_edge(f'{new_prefix}-res+', f'{new_prefix}-res^', operation='mod10')
                        new_res_carry, new_res = f'{new_prefix}-res_carry', f'{new_prefix}-res^'
                        new_carry = new_res_carry
                else:
                    new_res = output_res

                output_carry = new_carry
                outputs_to_sum[graph.nodes[new_res]["rank"]] = new_res
                assert graph.nodes[new_res]["rank"] == rank

            if output_carry is not None:
                outputs_to_sum[graph.nodes[output_carry]["rank"]] = output_carry

    return graph


def build_scratchpad_graph(number_x, number_y):
    graph = nx.DiGraph()
    xs = add_input_node(number_x, 'x', graph)
    ys = add_input_node(number_y, 'y', graph)

    partial_products = []
    for y in ys:
        partial_product, carry = None, None
        for i, x in enumerate(xs):
            xy_prd_v = get_base_operation('mul', graph.nodes[x]['value'], graph.nodes[y]['value'])
            xy_rank = graph.nodes[x]['rank'] + graph.nodes[y]['rank']
            graph.add_node(f'{x}{y}-prd', type='product', value=xy_prd_v, rank=xy_rank)
            graph.add_edge(x, f'{x}{y}-prd', operation='multiply')
            graph.add_edge(y, f'{x}{y}-prd', operation='multiply')

            xy_prd = f'{x}{y}-prd'
            xy_prd_carry, xy_prd_res = None, None

            if carry is not None:
                xy_prd_sum_v = graph.nodes[xy_prd]['value'] + graph.nodes[carry]['value']
                graph.add_node(f'{x}{y}-prd+', type='residual', value=xy_prd_sum_v, rank=xy_rank)
                graph.add_edge(xy_prd, f'{x}{y}-prd+', operation='add')
                graph.add_edge(carry, f'{x}{y}-prd+', operation='add')
                xy_prd = f'{x}{y}-prd+'

            if i < len(xs) - 1:
                if get_digit(graph.nodes[xy_prd]["value"]) == 1:
                    xy_prd_res = xy_prd
                else:
                    assert get_digit(graph.nodes[xy_prd]["value"]) == 2
                    xy_prd_carry_v, xy_prd_res_v = get_base_operation('mod10', graph.nodes[xy_prd]["value"])
                    graph.add_node(f'{x}{y}-prd_carry', type='carry', value=xy_prd_carry_v, rank=xy_rank + 1)
                    graph.add_edge(xy_prd, f'{x}{y}-prd_carry', operation='mod10')
                    graph.add_node(f'{x}{y}-prd_res', type='residual', value=xy_prd_res_v, rank=xy_rank)
                    graph.add_edge(xy_prd, f'{x}{y}-prd_res', operation='mod10')
                    xy_prd_carry, xy_prd_res = f'{x}{y}-prd_carry', f'{x}{y}-prd_res'
            else:
                xy_prd_res = xy_prd
            carry = xy_prd_carry
            output_res_v = graph.nodes[xy_prd_res]['value'] * (10 ** graph.nodes[x]['rank'])
            graph.add_node(f'{x}{y}-output', type='output', value=output_res_v, rank=graph.nodes[y]['rank'])
            graph.add_edge(xy_prd_res, f'{x}{y}-output', operation='exp')
            graph.add_edge(x, f'{x}{y}-output', operation='exp')
            xy_output = f'{x}{y}-output'

            if partial_product is None:
                partial_product = xy_output
            else:
                partial_product_v = graph.nodes[partial_product]['value'] + graph.nodes[xy_output]['value']
                assert graph.nodes[partial_product]['rank'] == graph.nodes[xy_output]['rank'] == graph.nodes[y]['rank'], 'rank mismatch'
                ori_prefix, new_prefix = get_prefix(partial_product), get_prefix(xy_output)
                graph.add_node(f'{ori_prefix}{new_prefix}-prd', type='output', value=partial_product_v, rank=graph.nodes[y]['rank'])
                graph.add_edge(partial_product, f'{ori_prefix}{new_prefix}-prd', operation='add')
                graph.add_edge(xy_output, f'{ori_prefix}{new_prefix}-prd', operation='add')
                partial_product = f'{ori_prefix}{new_prefix}-prd'

        partial_products.append(partial_product)

    final_product_v = sum([graph.nodes[node]['value'] * (10 ** graph.nodes[node]['rank']) for node in partial_products])
    graph.add_node('final_output', type='output', value=final_product_v, rank=0)
    for node in partial_products:
        graph.add_edge(node, 'final_output', operation='exp_add')

    return graph


def build_scratchpad_v2_graph(number_x, number_y):
    graph = nx.DiGraph()
    xs = add_input_node(number_x, 'x', graph)
    ys = add_input_node(number_y, 'y', graph)

    partial_products = []
    for j, y in enumerate(ys):
        partial_product, carry = None, None
        for i, x in enumerate(xs):
            # multiple x and y
            xy_prd_v = get_base_operation('mul', graph.nodes[x]['value'], graph.nodes[y]['value'])
            xy_rank = graph.nodes[x]['rank'] + graph.nodes[y]['rank']
            graph.add_node(f'{x}{y}-prd', type='product', value=xy_prd_v, rank=xy_rank)
            graph.add_edge(x, f'{x}{y}-prd', operation='multiply')
            graph.add_edge(y, f'{x}{y}-prd', operation='multiply')

            xy_prd = f'{x}{y}-prd'
            xy_prd_carry, xy_prd_res = None, None

            if carry is not None:
                xy_prd_sum_v = graph.nodes[xy_prd]['value'] + graph.nodes[carry]['value']
                graph.add_node(f'{x}{y}-prd+', type='residual', value=xy_prd_sum_v, rank=xy_rank)
                graph.add_edge(xy_prd, f'{x}{y}-prd+', operation='add')
                graph.add_edge(carry, f'{x}{y}-prd+', operation='add')
                xy_prd = f'{x}{y}-prd+'

            if i < len(xs) - 1:
                if get_digit(graph.nodes[xy_prd]["value"]) == 1:
                    xy_prd_res = xy_prd
                else:
                    assert get_digit(graph.nodes[xy_prd]["value"]) == 2
                    xy_prd_carry_v, xy_prd_res_v = get_base_operation('mod10', graph.nodes[xy_prd]["value"])
                    graph.add_node(f'{x}{y}-prd_carry', type='carry', value=xy_prd_carry_v, rank=xy_rank + 1)
                    graph.add_edge(xy_prd, f'{x}{y}-prd_carry', operation='mod10')
                    graph.add_node(f'{x}{y}-prd_res', type='residual', value=xy_prd_res_v, rank=xy_rank)
                    graph.add_edge(xy_prd, f'{x}{y}-prd_res', operation='mod10')
                    xy_prd_carry, xy_prd_res = f'{x}{y}-prd_carry', f'{x}{y}-prd_res'
            else:
                xy_prd_res = xy_prd

            carry = xy_prd_carry
            partial_product_v = graph.nodes[partial_product]['value'] if partial_product else 0 + graph.nodes[xy_prd_res]['value'] * (
                        10 ** graph.nodes[x]['rank'])
            graph.add_node(f'{x}{y}-output', type='output', value=partial_product_v, rank=graph.nodes[y]['rank'])
            graph.add_edge(xy_prd_res, f'{x}{y}-output', operation='concat')
            xy_output = f'{x}{y}-output'

            if partial_product:
                assert graph.nodes[partial_product]['rank'] == graph.nodes[xy_output]['rank'] == graph.nodes[y]['rank'], 'rank mismatch'
                ori_prefix, new_prefix = get_prefix(partial_product), get_prefix(xy_output)
                graph.add_edge(partial_product, xy_output, operation='concat')
            partial_product = xy_output

        adjusted_pp = graph.nodes[partial_product]['value'] * 10 ** j
        output_node = f"{get_prefix(partial_product)}-output+"
        graph.add_node(
            output_node,
            type="output",
            value=adjusted_pp,
            rank=graph.nodes[partial_product]["rank"],
        )
        graph.add_edge(partial_product, output_node, operation="exp")
        partial_products.append(output_node)

    final_product_v = sum([graph.nodes[node]['value'] for node in partial_products])
    graph.add_node('final_output', type='output', value=final_product_v, rank=0)
    for node in partial_products:
        graph.add_edge(node, 'final_output', operation='add')

    return graph

