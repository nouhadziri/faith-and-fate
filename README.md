# Faith and Fate: Limits of Transformers on Compositionality

This repository hosts the code and data for our paper [Faith and Fate: Limits of Transformers on Compositionality](https://arxiv.org/pdf/2305.18654.pdf) which will be presented at NeurIPS (Spotlight)! 
Please check out the [updated paper](https://arxiv.org/pdf/2305.18654.pdf), grokking results included!

## Quick Links
  - [Data](#data)
  - [Requirements](#requirements)
  - [Multiplication](#multiplication)
  - [Dynamic Programming](#dynamic-programming)
  - [Einstein Puzzle](#einstein-puzzle)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [License](#license)

## Requirements
To install dependencies, first clone this repo and then, install dependencies in `requirements.txt` (python >= `3.8`, the code is tested with python `3.8`):

```bash
git clone git@github.com:nouhadziri/faith-and-fate.git
pip install -r requirements.txt
```


## Data
In this study, we address three tasks: multiplication, 
dynamic programming, and the Einstein puzzle. 
Dataset samples are available in the ``data`` directory. The password is "faith-and-fate." For a more exhaustive data generation, use the provided code for each task and customize your parameters.
## Multiplication

### Generate data without a scratchpad
```bash
python mutliplication/build_data.py --num_digit <max number of digits> \ 
  --max_sequence  <max number of inputs per combination> \
  --output_dir <path to output directory>
```

### Generate scratchpad data
```bash
python mutliplication/generate_scratchpads.py  --num_digit <max number of digits> \  --number_prompts <max number of instances> \
  --output_path <path to output directory>
```

### Build a graph
To build a graph over the multiplication scratchpad, simply run:

```bash
python multiplication/generate_graph_from_scratchpad.py <path to the generated scratchpad file> 
```

### Error analysis
To analyse the types of errors that Transformers make at every step in the compuatational graph, run:
```bash
python multiplication/graph_error_analysis.py --scratchpad_folder <path to the generated scratchpad file>
```

### Analyzing the patterns 
To explore whether models' correct predictions on unseen test data are due to learning the underlying algorithm or, instead, explainable by exposure to similar training examples, run:
```bash
python multiplication/graph_pattern_analysis.py --scratchpad_folder <path to the generated scratchpad> \
    --train_data_path <path to the training data>
```

## Dynamic Programming

### Generate data with and without a scratchpad
```bash
python dynamic_programming/generate_training_data.py --output_dir <path to the output dir>
  --scratchpad
```
If``--scratchpad``is enabled, the code will generate a scratchpad for each example, otherwise not. 

### Build a graph
To build a graph over the scratchpad, simply run:

```bash
python dynamic_programming/generate_graph_from_scratchpad.py 
```

### Error analysis

```bash
python dynamic_programming/graph_error_analysis.py --scratchpad_folder <path to the files folder>
```

### Analyzing the patterns 

```bash
python dynamic_programming/graph_pattern_analysis.py --scratchpad_folder <path to the generated scratchpad> \
    --train_data_path <path to the training data>
```

## Einstein Puzzle

### Generate the puzzle data
Generate the puzzle without a scratchpad:
```bash
python puzzle/generate.py 
```

### Generate the puzzle scratchpad data
To generate the scratchpad for each puzzle, specify the file name and the size of the puzzle since it could take long time to generate the path for 5x5 puzzles.
```bash
python logic_puzzle/graph/reasoning_path.py --input_data <inputdata location> --ground_truth <ground truth location> --size 5x
```

### Build the graph and analyze errors
```bash
python logic_puzzle/graph/pattern_analysis.py --cot_dir <path to the generated scratchpad> --output_dir <path to output dir>
```


## Bugs or questions?

If you have any questions (:question:) related to the code, or encounter any problems (:hammer_and_wrench:), or want to report a bug (:bug:), feel free to open an issue.

## Citation

If you want to cite our papers, please use:

```bibtex
@article{dziri2023faith,
  title={Faith and Fate: Limits of Transformers on Compositionality},
  author={Dziri, Nouha and Lu, Ximing and Sclar, Melanie and Li, Xiang Lorraine and Jian, Liwei and Lin, Bill Yuchen and West, Peter and Bhagavatula, Chandra and Bras, Ronan Le and Hwang, Jena D and others},
  journal={arXiv preprint arXiv:2305.18654},
  year={2023}
}
```


## License

This work is licensed under the MIT license. See [LICENSE](LICENSE) for details.
