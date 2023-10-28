import argparse
import json
import os
import random

random.seed(0)

from testing_task_for_info_gain_patterns import sample_entries, all_entries

TASK_INSTRUCTION = """Given a sequence of integers, find a subsequence with the highest sum, such that no two numbers in the subsequence are adjacent in the original sequence.

To indicate the selected numbers, print an array with "1" for chosen numbers and "2" for unchosen ones. For instance, [1, 2, 2, 2, 2] implies selecting only the first number. If multiple solutions exist, select the lexicographically smallest."""

#  used as prefix before putting the scratchpads
GENERAL_DP_INSTRUCTION = "We will solve any task instance by using dynamic programming. We define dp[i] as the maximum sum of a subsequence that does not include adjacent elements, when considering only the elements of the input from the i-th position onwards."

# one-shot example for non-scratchpad version
IO_EXAMPLE = """input = [10, 10, 7, 5, 6]
output = [1, 2, 1, 2, 1]"""


def generate_scratchpad_completion(arr):
    N = len(arr)

    dp = [0 for _ in range(N)]
    dp[N - 1] = max(arr[N - 1], 0)
    dp[N - 2] = max(max(arr[N - 1], arr[N - 2]), 0)
    for i in range(N - 3, -1, -1):
        dp[i] = max(max(dp[i + 1], arr[i] + dp[i + 2]), 0)
    max_sum = dp[0]

    # reconstruct the answer with a fixed-size graph
    reconstruction_lines = []
    result = []
    can_access_next_item = True
    for i in range(N - 2):
        if dp[i] == arr[i] + dp[i + 2] and can_access_next_item:
            result.append(1)
            can_access_next_item = False
            reconstruction_lines.append(
                f'Since dp[{i}] == input[{i}] + dp[{i + 2}] ({dp[i]} == {arr[i]} + {dp[i + 2]}) and can_use_next_item == True, we store output[{i}] = 1. We update can_use_next_item = False.')
        else:
            result.append(2)
            can_access_next_item = True
            reconstruction_lines.append(
                f'Since dp[{i}] != input[{i}] + dp[{i + 2}] ({dp[i]} != {arr[i]} + {dp[i + 2]}) or can_use_next_item == False, we store output[{i}] = 2. We update can_use_next_item = True.')

    if dp[N - 2] == arr[N - 2] and can_access_next_item:
        result.append(1)
        can_access_next_item = False
        reconstruction_lines.append(
            f'Since dp[{N - 2}] == input[{N - 2}] ({dp[N - 2]} == {arr[N - 2]}) and can_use_next_item == True, we store output[{N - 2}] = 1. We update can_use_next_item = False.')
    else:
        result.append(2)
        can_access_next_item = True
        reconstruction_lines.append(
            f'Since dp[{N - 2}] != input[{N - 2}] ({dp[N - 2]} != {arr[N - 2]}) or can_use_next_item == False, we store output[{N - 2}] = 2. We update can_use_next_item = True.')

    if dp[N - 1] == arr[N - 1] and can_access_next_item:
        result.append(1)
        reconstruction_lines.append(
            f'Since dp[{N - 1}] == input[{N - 1}] ({dp[N - 1]} == {arr[N - 1]}) and can_use_next_item == True, we store output[{N - 1}] = 1.')
    else:
        result.append(2)
        reconstruction_lines.append(
            f'Since dp[{N - 1}] != input[{N - 1}] ({dp[N - 1]} != {arr[N - 1]}) or can_use_next_item == False, we store output[{N - 1}] = 2.')

    lines = []
    for i in range(N - 3, -1, -1):
        lines.append(
            f'dp[{i}] = max(dp[{i + 1}], input[{i}] + dp[{i + 2}], 0) = max({dp[i + 1]}, {arr[i]} + {dp[i + 2]}, 0) = {dp[i]}')

    lines_str = '\n'.join(lines)
    reconstruction_lines_str = '\n'.join(reconstruction_lines)

    return f"""dp[{N - 1}] = max(input[{N - 1}], 0) = max({arr[N - 1]}, 0) = {dp[N - 1]}
dp[{N - 2}] = max(input[{N - 2}], input[{N - 1}], 0) = max({arr[N - 2]}, {arr[N - 1]}, 0) = {dp[N - 2]}
{lines_str}

Finally, we reconstruct the lexicographically smallest subsequence that fulfills the task objective by selecting numbers as follows. We store the result on a list named "output".

Let can_use_next_item = True.
{reconstruction_lines_str}

Reconstructing all together, output={result}.
""", result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_numbers', type=int, required=True)
    parser.add_argument('--min_value', type=int, default=-5)  # keep as is
    parser.add_argument('--max_value', type=int, default=5)  # keep as is
    parser.add_argument('--num_samples', type=int, default=10000000)
    parser.add_argument('--output_dir', type=str, default='tmp')
    parser.add_argument('--scratchpad', action='store_true')

    args = parser.parse_args()

    print('-' * 100)
    print('For testing few-shot scratchpad versions, use the following prefix before any prompt completion, and then use the fields prompt_scratchpad and completion_scratchpad:\n\n')
    print(TASK_INSTRUCTION + "\n\n\n" + GENERAL_DP_INSTRUCTION + "\n\n")
    print('-' * 100)
    print('For testing non-scratchpad versions, use the following prefix before any prompt completion, and then use the fields prompt_no_scratchpad and completion_no_scratchpad:\n\n')
    print(TASK_INSTRUCTION)

    total_combinations = (args.max_value - args.min_value + 1) ** args.num_numbers

    if total_combinations > args.num_samples:
        entries = sample_entries(args.num_numbers, args.min_value, args.max_value, num_samples=args.num_samples)
        suffix_sampled = f'sampled_{args.num_samples}'
    else:
        entries = all_entries(args.num_numbers, args.min_value, args.max_value)
        suffix_sampled = 'all'

    scratchpad_dir = 'scratchpad' if args.scratchpad else 'no_scratchpad'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, scratchpad_dir), exist_ok=True)

    output_filename = f'data_{scratchpad_dir}_n_{args.num_numbers}_minval_{args.min_value}_maxval_{args.max_value}_{suffix_sampled}.jsonl'
    with open(os.path.join(args.output_dir, scratchpad_dir, output_filename), 'w') as f:
        for input_list, output_list in entries:
            if args.scratchpad:
                scratchpad, output_list_from_scratchpad = generate_scratchpad_completion(input_list)
                scratchpad = scratchpad.strip()
                assert output_list == output_list_from_scratchpad
                fields = {
                    'prompt': f"Let's solve input = {input_list}." + "\n\n###\n\n",
                    'completion': ' ' + scratchpad + ' ###',
                }
            else:
                fields = {
                    'prompt': f'input = {input_list}' + "\n\n###\n\n",
                    'completion': ' ' + f'output = {output_list}' + ' ###'
                }
            f.write(json.dumps(fields) + '\n')
