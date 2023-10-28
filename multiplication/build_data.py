import json
import argparse
from tqdm import tqdm
import math
import itertools
import random
from random import randrange
from pathlib import Path
import csv


random.seed(0)


def all_n_digit(num_digit):
    return list(range(int(math.pow(10, num_digit - 1)), int(math.pow(10, num_digit))))


def random_n_digit(num_digit):
    return randrange(int(math.pow(10, num_digit - 1)), int(math.pow(10, num_digit)))


def cartesian(a_num_digit, b_num_digit):
    a_numbers, b_numbers = all_n_digit(a_num_digit), all_n_digit(b_num_digit)
    inputs = [e for e in itertools.product(a_numbers, b_numbers)]
    return inputs

def sample(a_num_digit, b_num_digit, max_sequence):
    inputs = []
    while len(inputs) < max_sequence:
        a, b = random_n_digit(a_num_digit), random_n_digit(b_num_digit)
        if (a, b) not in inputs:
            inputs.append((a, b))
    return inputs


def count_tokens_per_example(gpt2_tokenizer, example):
    tokenizer = gpt2_tokenizer.from_pretrained("gpt2")
    number_of_tokens = len(tokenizer(example)['input_ids'])
    return number_of_tokens


def construct_dataset(num_digit, max_sequence):
    digits = list(range(1, num_digit + 1))
    datasets = {}
    for a_num_digit in digits:
        for b_num_digit in tqdm(digits[:a_num_digit]):
            name = f'{a_num_digit}_by_{b_num_digit}'
            num_combination = math.pow(10, a_num_digit + b_num_digit)
            if num_combination < max_sequence:
                inputs = cartesian(a_num_digit, b_num_digit)
            else:
                inputs = sample(a_num_digit, b_num_digit, max_sequence)
            datasets[name] = inputs
    return datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_digit', type=int, default=4, help='maximum number of digits')
    parser.add_argument(
        '--max_sequence', type=int, default=3000, help='maximum number of inputs per combination')
    parser.add_argument(
        "--output_dir", default='math_data', type=str, help="output directory")
    parser.add_argument(
        '--format', type=str, default='jsonl', help='the format of the output_file')

    parser.add_argument("--count_tokens", action="store_true", help="whether to count the nb of tokens")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    digits = list(range(1, args.num_digit + 1))
    nb_examples = 0
    tot_examples = 0
    for a_num_digit in digits:
        for b_num_digit in tqdm(digits[:a_num_digit]):
            openai_fine_tune = []
            name = f'{a_num_digit}_by_{b_num_digit}'
            num_combination = math.pow(10, a_num_digit + b_num_digit)
            if args.format == "jsonl":
                output_file = output_dir / f'{name}_digit_fine_tune.jsonl'
                with open(output_file, "w") as f:
                    if num_combination < args.max_sequence:
                        inputs = cartesian(a_num_digit, b_num_digit)
                    else:
                        inputs = sample(a_num_digit, b_num_digit, args.max_sequence)
                    for numbers in tqdm(inputs, desc=name):
                        target = numbers[0] * numbers[1]
                        p1 = f'What is {numbers[0]} times {numbers[1]}?\n\n###\n\n'
                        openai_fine_tune.append({"prompt": p1, "completion": " " + str(target) + "\n"})
                        tot_examples += 1
                        nb_examples += 1

                    if openai_fine_tune:
                        for o in openai_fine_tune:
                            f.write(json.dumps(o) + "\n")

            elif args.format == "tsv":
                output_file = output_dir / f'{name}_digit_fine_tune.tsv'
                with open(output_file, "w") as f:
                    tsv_writer = csv.writer(f, delimiter="\t")
                    tsv_writer.writerow(("prompt", "completion"))
                    inputs = cartesian(a_num_digit, b_num_digit)
                    for numbers in tqdm(inputs, desc=name):
                        target = numbers[0] * numbers[1]
                        p1 = f'What is {numbers[0]} times {numbers[1]}?'
                        tsv_writer.writerow((p1, target))


    print("****** Done *******")
    print(f"The final number of examples in the data:{tot_examples}")


if __name__ == "__main__":
    main()
