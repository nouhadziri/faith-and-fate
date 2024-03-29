import argparse
import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd
import json


def compute_entropy(output_class_values):
    output_class_counter = Counter(output_class_values)
    total = sum(output_class_counter.values())
    assert len(output_class_values) == total
    entropy = sum([- (c / total) * np.log2(c / total) for c in output_class_counter.values()])
    return entropy

# https://www.geeksforgeeks.org/maximum-sum-such-that-no-two-elements-are-adjacent/
def findMaxSumGeeksForGeeks(arr):
    N = len(arr)

    dp = [[0 for i in range(2)] for j in range(N)]
    if N == 1:
        return arr[0]

    dp[0][0] = 0
    dp[0][1] = arr[0]

    for i in range(1, N):
        dp[i][1] = dp[i - 1][0] + arr[i]
        dp[i][0] = max(dp[i - 1][1], dp[i - 1][0])

    return max(dp[N - 1][0], dp[N - 1][1])


def find_max_sum_nonadjacent(arr):
    """
    When there are many results, choose the one that appears first lexicographically,
    where 1=picking the number and 2=not picking it.

    dp[i][0] = maximum subsequence of arr[i:] where we do not use arr[i]
    dp[i][1] = maximum subsequence of arr[i:] where we do use arr[i]

    dp[i][0] = max(dp[i+1][0], dp[i+1][1])
    dp[i][1] = arr[i] + dp[i+1][0]
    """
    N = len(arr)

    dp = [[0 for _ in range(2)] for _ in range(N)]
    dp[N-1][0] = 0
    dp[N-1][1] = arr[N-1]
    for i in range(N-2, -1, -1):
        dp[i][1] = dp[i + 1][0] + arr[i]
        dp[i][0] = max(dp[i + 1][0], dp[i + 1][1])

    max_sum = max(dp[0][0], dp[0][1])

    result = []
    remaining_sum = max_sum
    can_access_next_item = True
    for i in range(N):
        if dp[i][1] == remaining_sum and can_access_next_item:
            result.append(1)
            remaining_sum -= arr[i]
            can_access_next_item = False
        elif dp[i][0] == remaining_sum:
            result.append(2)
            can_access_next_item = True
        else:
            assert False

    return result, max_sum


def get_feature_ids_to_analyze(input_nodes):
    # features are = each single feature, each summand with the target value, each pair of summands plus target value
    features_to_analyze = []
    features_to_analyze = [[feature_i] for feature_i in range(len(input_nodes))]
    features_to_analyze += [[i, i+1]
                            for i in range(len(input_nodes) - 1)]
    features_to_analyze += [[i, i+1, i+2]
                            for i in range(len(input_nodes) - 2)]
    return features_to_analyze


def sample_entries(num_numbers, min_value, max_value, num_samples=1000000):
    result = set()
    while len(result) < num_samples:
        input = [random.randint(min_value, max_value) for _ in range(num_numbers)]

        output_sequence, my_max_sum = find_max_sum_nonadjacent(input)
        # assert expected_max_sum == my_max_sum

        result.add((tuple(input), tuple(output_sequence)))

    result = list(result)
    result = [(list(input), list(output_seq)) for input, output_seq in result]

    return result


def all_entries(num_numbers, min_value, max_value):
    all_inputs = itertools.product(list(range(min_value, max_value + 1)), repeat=num_numbers)

    result = []
    for input in all_inputs:
        input = list(input)
        output_sequence, my_max_sum = find_max_sum_nonadjacent(input)
        # assert expected_max_sum == my_max_sum

        result.append((input, output_sequence))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_numbers', type=int, required=True)
    parser.add_argument('--min_value', type=int, required=True)
    parser.add_argument('--max_value', type=int, required=True)
    parser.add_argument('--num_samples', type=int, default=100000)
    args = parser.parse_args()

    entries = sample_entries(args.num_numbers, args.min_value, args.max_value, num_samples=args.num_samples)
    json.dump(entries, open(f'all_entries_numnumbers_{args.num_numbers}_minval_{args.min_value}_maxval_{args.max_value}.json', 'w'))
    exit(0)

    input_nodes = [f'arr_{i}' for i in range(args.num_numbers)]
    features_to_analyze = get_feature_ids_to_analyze(input_nodes)

    output_node_names = [f'output_{i}' for i in range(args.num_numbers)]
    all_node_names_concat = input_nodes + output_node_names

    print('WE WILL ANALYZE THE FOLLOWING FEATURES:')
    for features_i in features_to_analyze:
        print([all_node_names_concat[e] for e in features_i])

    df_list = []
    for node_name in output_node_names:
        # find all data point values: find all input node values
        data_point_values = []
        for input, result in entries:
            print(input, result)
            tmp = [str(e) for e in input + result]
            data_point_values.append((tmp))

        # compute H(T)  [last feature is the class]
        entropy = compute_entropy([d[-1] for d in data_point_values])

        # compute H(T|feature_i)
        for features_i in features_to_analyze:
            possible_values_features_i = set([tuple(d[i] for i in features_i) for d in data_point_values])

            conditional_entropy = 0
            for values_i in possible_values_features_i:
                support = [d[-1] for d in data_point_values if all(d[i] == j for i, j in zip(features_i, values_i))]
                entropy_value_i = compute_entropy(support)

                support_size = len(support)
                conditional_entropy += support_size / len(data_point_values) * entropy_value_i

            print(f'node={node_name}\n'
                  f'features={" ".join(all_node_names_concat[feature_i] for feature_i in features_i)}\n'
                  f'\tentropy={entropy:.2f}\n'
                  f'\tinformation_gain={(entropy - conditional_entropy):.2f}')
            df_list.append(
                {
                    'input_node': " ".join(all_node_names_concat[feature_i] for feature_i in features_i),
                    'node_name': node_name,
                    'information_gain': f'{(entropy - conditional_entropy):.2f}',
                    'entropy': entropy,
                    'conditional_entropy': conditional_entropy,
                    'output_node': next(iter(i for i, e in enumerate(output_node_names) if e == node_name), None)
                }
            )

    pd.DataFrame(df_list).sort_values(by='information_gain').to_csv(
        f'max_sum_nonadjacent_ordered_information_gain_by_input_node_{args.num_numbers}_numbers_minval_{args.min_value}_maxval{args.max_value}_maxvalue_allsamples.csv')