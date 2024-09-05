# A First Look at Conventional Commits Classification

<div align="center">

[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-green)](https://huggingface.co/datasets/0x404/ccs_dataset)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green)](https://huggingface.co/0x404/ccs-code-llama-7b)
[![Figshare](https://img.shields.io/badge/Figshare-005C47)](https://doi.org/10.6084/m9.figshare.26507083)

</div>


[Conventional Commits🪄](https://www.conventionalcommits.org/en/v1.0.0/), as a specification for adding both human and machine-readable meaning to commit messages, is increasingly gaining popularity among open-source projects and developers. We conducts a preliminary study of CCS, encompassing its application status and the challenges developers encounter when using it. We observe a growing popularity of CCS, yet developers do misclassify commits into incorrect CCS types, attributable to the absence of a clear and distinct definition list for each type. We have developed a more precise and less overlapping definition list to address this, grounded in industry practices and literature review. To assist developers in classifying conventional commits, we propose an approach for automated conventional commit classification.

This repository contains all the data and code we used in the study.

## Reproduction

### Using Hugging Face (Recommended)

We have uploaded the dataset and the model's parameters to the Hugging Face hub, making it very easy to replicate our results. First, ensure you have installed the necessary environment:

```shell
pip3 install transformers datasets scikit-learn
```

Then, you can use `transformers` and `datasets` to load our model and dataset, and test it on the test set:

```python
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

test_dataset = load_dataset("0x404/ccs_dataset", split="test")
pipe = pipeline("text-generation", model="0x404/ccs-code-llama-7b", device_map="auto")

generated_data = pipe(test_dataset["input_prompt"], max_new_tokens=10, pad_token_id=pipe.tokenizer.eos_token_id)
generated_label = [s[0]["generated_text"].split()[-1] for s in generated_data]

accuracy = accuracy_score(test_dataset["annotated_type"], generated_label)
f1 = f1_score(test_dataset["annotated_type"], generated_label, average="macro")

print("Accuracy:", accuracy)
print("F1 Score (Macro):", f1)
```

### Use the Code from Scratch

The dataset is available on Hugging Face, making it very easy to access:

```python
from datasets import load_dataset
ccs_dataset = load_dataset("0x404/ccs_dataset")
```

Additionally, in the `Dataset` directory of this repository, we provide two datasets: one containing 88,704 commits in the Conventional Commits format mined from 116 repositories, and another dataset of 2,000 commits that were sampled and manually annotated from them. The manually annotated dataset is utilized for model training, validation, and testing. For detailed information about the datasets, please refer to the README.md in the `Dataset` directory.

To run our code from scratch, you need to install the necessary environments. We provide the version information of all dependent environments in `requirements.txt`, which can be installed with:

```shell
pip3 install -r requirements.txt
```

- To replicate the experimental results of ChatGPT4, access the code in the `RQ3/ChatGPT` directory. It includes a `test.csv` file which is our test dataset. Insert your OpenAI Key (with access to ChatGPT4) in the code, then execute it.
- For replicating BERT experimental results, navigate to the `RQ3/BERT` directory and use the command `python3 main.py --train` for training, and `python3 main.py --test <checkpointpath>` for testing, where `<checkpointpath>` is the location of the saved parameter checkpoint.
- To replicate the experiments for Llama2 and CodeLlama, find the code in the `RQ3/Llama2` and `RQ3/CodeLlama` directories respectively. Training is executed with `python3 train.py`, and testing with `python3 infer.py --checkpoint <checkpointpath>`, where `<checkpointpath>` is the checkpoint's location, by default in the `llama-output` directory within the current directory.

*Note: To replicate Llama2 and CodeLlama, approximately two 24GB memory GPUs are required. If the appropriate hardware is not available, for ease of replication, we provide the trained LORA parameters in the `CheckPoints` directory. If you wish to directly use the pre-trained CodeLlama parameters, execute the following command:

```shell
cd RQ3/CodeLlama
python3 infer.py --checkpoint ../../CheckPoints/CodeLlama-checkpoints
```

## How to Use

Our model basically accepts two inputs: the commit message and the corresponding git diff. To classify your desired commit, you must follow a specific prompt format. We provide a code snippet as follows:

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="0x404/ccs-code-llama-7b", device_map="auto")
tokenizer = pipe.tokenizer


def prepare_prompt(commit_message: str, git_diff: str, context_window: int = 1024):
    prompt_head = "<s>[INST] <<SYS>>\nYou are a commit classifier based on commit message and code diff.Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, fix, test, ci, build, and chore. The definitions of each category are as follows:\n**feat**: Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.\n**fix**: Code changes aim to fix bugs and faults within the codebase.\n**perf**: Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.\n**style**: Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.\n**refactor**: Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as ``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.\n**docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.\n**test**: Code changes that modify test files, including the addition or updating of tests.\n**ci**: Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD scripts, e.g., ``.travis.yml'' and ``.github/workflows''.\n**build**: Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating dependencies, configuring build configurations, and adding scripts.\n**chore**: Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.\n<</SYS>>\n\n"
    prompt_head = tokenizer.encode(prompt_head, add_special_tokens=False)
    prompt_message = tokenizer.encode(
        f"- given commit message:\n{commit_message}\n",
        max_length=64,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_diff = tokenizer.encode(
        f"- given commit diff: \n{git_diff}\n",
        max_length=context_window - len(prompt_head) - len(prompt_message) - 6,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_end = tokenizer.encode(" [/INST]", add_special_tokens=False)
    return tokenizer.decode(prompt_head + prompt_message + prompt_diff + prompt_end)


def classify_commit(commit_message: str, git_diff: str, context_window: int = 1024):
    prompt = prepare_prompt(commit_message, git_diff, context_window)
    result = pipe(prompt, max_new_tokens=10, pad_token_id=pipe.tokenizer.eos_token_id)
    label = result[0]["generated_text"].split()[-1]
    return label

```

Here, you can use the `classify_commit` function to classify your commit by inputting the commit's message and git diff. The `context_window` controls the size of the entire prompt, set to 1024 by default but adjustable to a larger value like 2048 to include more git diff in one prompt. Here is an example of its usage:

```python
import requests
from github import Github

def fetch_message_and_diff(repo_name, commit_sha):
    g = Github()
    try:
        repo = g.get_repo(repo_name)
        commit = repo.get_commit(commit_sha)
        if commit.parents:
            parent_sha = commit.parents[0].sha
            diff_url = repo.compare(parent_sha, commit_sha).diff_url
            return commit.commit.message, requests.get(diff_url).text
        else:
            raise ValueError("No parent found for this commit, unable to retrieve diff.")
    except Exception as e:
        raise RuntimeError(f"Error retrieving commit information: {e}")

message, diff = fetch_message_and_diff("pytorch/pytorch", "9856bc50a251ac054debfdbbb5ed29fc4f6aeb39")
print(classify_commit(message, diff))
```

In this setup, we've defined a function `fetch_message_and_diff` that fetches the commit message and diff for any specified SHA from a GitHub repository, enabling our model to classify the commit accordingly.


## Performance of specific CCS types

This table is the full table provided in our RQ3, including precision, recall, and f1 score for each of the ten specific CCS types, with the highest score highlighted in **bold**.

| Metrics            | BERT       | ChatGPT4   | Llama2     | Our Approach |
|:-------------------|:-----------:|:-----------:|:-----------:|:------------:|
| build_precision    | 0.6304     | **0.8286** | 0.6905     | 0.7442      |
| build_recall       | 0.725      | 0.725      | 0.725      | **0.8**     |
| build_f1           | 0.6744     | **0.7733** | 0.7073     | 0.7711      |
| ci_precision       | **0.8718** | 0.8571     | 0.8409     | 0.8605      |
| ci_recall          | 0.85       | 0.9        | **0.925**  | 0.925       |
| ci_f1              | 0.8608     | 0.878      | 0.881      | **0.8916**  |
| docs_precision     | 0.8095     | **0.8974** | 0.7451     | 0.8372      |
| docs_recall        | 0.85       | 0.875      | **0.95**   | 0.9         |
| docs_f1            | 0.8293     | **0.8861** | 0.8352     | 0.8675      |
| perf_precision     | 0.3939     | **0.9545** | 0.875      | 0.8378      |
| perf_recall        | 0.65       | 0.525      | 0.7        | **0.775**   |
| perf_f1            | 0.4906     | 0.6774     | 0.7778     | **0.8052**  |
| chore_precision    | 0.3846     | 0.6957     | 0.6129     | **0.7391**  |
| chore_recall       | **0.5**    | 0.4        | 0.475      | 0.425       |
| chore_f1           | 0.4348     | 0.5079     | 0.5352     | **0.5397**  |
| test_precision     | 0.6923     | 0.9        | 0.8889     | **0.9459**  |
| test_recall        | 0.675      | 0.675      | 0.8        | **0.875**   |
| test_f1            | 0.6835     | 0.7714     | 0.8421     | **0.9091**  |
| fix_precision      | 0.4167     | 0.4808     | **0.6829** | 0.6667      |
| fix_recall         | 0.25       | 0.625      | **0.7**    | 0.7         |
| fix_f1             | 0.3125     | 0.5435     | **0.6914** | 0.6829      |
| refactor_precision | 0.2414     | 0.4545     | **0.5814** | 0.5085      |
| refactor_recall    | 0.175      | 0.625      | 0.625      | **0.75**    |
| refactor_f1        | 0.2029     | 0.5263     | 0.6024     | **0.6061**  |
| style_precision    | 0.5333     | **0.8966** | 0.8049     | 0.7805      |
| style_recall       | 0.2        | 0.65       | **0.825**  | 0.8         |
| style_f1           | 0.2909     | 0.7536     | **0.8148** | 0.7901      |
| feat_precision     | 0.4583     | 0.5205     | 0.8205     | **0.875**   |
| feat_recall        | 0.55       | **0.95**   | 0.8        | 0.7         |
| feat_f1            | 0.5        | 0.6726     | **0.8101** | 0.7778      |
| macro_precision    | 0.5432     | 0.7486     | 0.7543     | **0.7795**  |
| macro_recall       | 0.5425     | 0.695      | 0.7525     | **0.765**   |
| macro_f1           | 0.528      | 0.699      | 0.7497     | **0.7641**  |
| accuracy           | 0.5425     | 0.695      | 0.7525     | **0.765**   |


## File Structure
```
.
├── CheckPoints: Contains checkpoints of the parameters of our fine-tuned models
├── Dataset: Datasets built and used in our research
├── README.md: Description of this repository
├── requirements.txt: Environmental dependencies
├── RQ1: Data and code used in RQ1
├── RQ2: Analysis of developer challenges in RQ2
└── RQ3: Code used for training models in RQ3
```

## Cite Us

If you use this repository in your research, please cite us using the following BibTeX entry:

```bibtex
@inproceedings{zeng2025conventional,
  title={A First Look at Conventional Commits Classification},
  author={Zeng, Qunhong and Zhang, Yuxia and Qiu, Zhiqing and Liu, Hui},
  booktitle={Proceedings of the IEEE/ACM 47th International Conference on Software Engineering},
  year={2025}
}
```
