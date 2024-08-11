import torch
import datasets
import pandas
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import json

DATASET_PATH = "../../Dataset/annotated_dataset.csv"
LLAMA_MODEL = "meta-llama/Llama-2-7b-hf"
ENABLE_PROFILER = False
OUTPUT_DIR = "./llama-output"
PROMPT_HEAD = (
    f"<s>[INST] <<SYS>>\n"
    f"You are a commit classifier based on commit message and code diff."
    f"Please classify the given commit into one of the ten categories: docs, perf, style, refactor, feat, fix, test, ci, build, and chore. The definitions of each category are as follows:\n"
    f"**feat**: Code changes aim to introduce new features to the codebase, encompassing both internal and user-oriented features.\n"
    f"**fix**: Code changes aim to fix bugs and faults within the codebase.\n"
    f"**perf**: Code changes aim to improve performance, such as enhancing execution speed or reducing memory consumption.\n"
    f"**style**: Code changes aim to improve readability without affecting the meaning of the code. This type encompasses aspects like variable naming, indentation, and addressing linting or code analysis warnings.\n"
    f"**refactor**: Code changes aim to restructure the program without changing its behavior, aiming to improve maintainability. To avoid confusion and overlap, we propose the constraint that this category does not include changes classified as ``perf'' or ``style''. Examples include enhancing modularity, refining exception handling, improving scalability, conducting code cleanup, and removing deprecated code.\n"
    f"**docs**: Code changes that modify documentation or text, such as correcting typos, modifying comments, or updating documentation.\n"
    f"**test**: Code changes that modify test files, including the addition or updating of tests.\n"
    f"**ci**: Code changes to CI (Continuous Integration) configuration files and scripts, such as configuring or updating CI/CD scripts, e.g., ``.travis.yml'' and ``.github/workflows''.\n"
    f"**build**: Code changes affecting the build system (e.g., Maven, Gradle, Cargo). Change examples include updating dependencies, configuring build configurations, and adding scripts.\n"
    f"**chore**: Code changes for other miscellaneous tasks that do not neatly fit into any of the above categories.\n"
    f"<</SYS>>\n\n"
)
PROMPT_COMMIT_MESSAGE = f"- given commit message:\n{{message}}\n"
PROMPT_COMMIT_DIFF = f"- given commit diff: \n{{diff}}\n"
TOKENIZER = LlamaTokenizer.from_pretrained(LLAMA_MODEL)
TOKENIZER.pad_token = TOKENIZER.eos_token
TOKENIZER.padding_side = "left"


def preprocess_dataset(dataset: datasets.Dataset):
    def apply_prompt_template(sample):
        return {
            "prompt_commit_message": PROMPT_COMMIT_MESSAGE.format(
                message=sample["masked_commit_message"]
            ),
            "prompt_commit_diff": PROMPT_COMMIT_DIFF.format(diff=sample["git_diff"]),
            "response": f"[/INST] {sample['annotated_type']} </s>",
        }

    def tokenize_add_label(sample):
        prompt_head = TOKENIZER.encode(
            PROMPT_HEAD,
            add_special_tokens=False,
        )
        message = TOKENIZER.encode(
            sample["prompt_commit_message"],
            max_length=64,
            truncation=True,
            add_special_tokens=False,
        )
        response = TOKENIZER.encode(
            sample["response"], max_length=20, truncation=True, add_special_tokens=False
        )
        diff = TOKENIZER.encode(
            sample["prompt_commit_diff"],
            max_length=1023 - len(prompt_head) - len(message) - len(response),
            truncation=True,
            add_special_tokens=False,
        )

        end = TOKENIZER.encode(" [/INST]", add_special_tokens=False)

        sample = {
            "text": TOKENIZER.decode(prompt_head + message + diff + end),
        }

        return sample

    dataset = dataset.map(apply_prompt_template)
    dataset = dataset.map(tokenize_add_label)
    return dataset


def make_dataset():
    df = pandas.read_csv(DATASET_PATH)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["annotated_type"], random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, stratify=temp_df["annotated_type"], random_state=42
    )
    test_df.to_csv("test.csv", index=False)
    test_dataset = datasets.Dataset.from_pandas(test_df)
    test_dataset = preprocess_dataset(test_dataset)
    return test_dataset


def gen(checkpoint):
    model = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, checkpoint)

    model.eval()

    dataset = make_dataset()

    labels, preds = [], []
    for data in tqdm(dataset):
        model_input = TOKENIZER(data["text"], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = TOKENIZER.decode(
                model.generate(
                    **model_input,
                    max_new_tokens=32,
                    pad_token_id=TOKENIZER.eos_token_id,
                )[0],
                skip_special_tokens=True,
            )
            labels.append(data["annotated_type"])
            preds.append(output.split()[-1])
            print(output)

    print(labels)
    print(preds)
    print(sum(1 for x, y in zip(labels, preds) if x == y))
    df = pandas.read_csv("test.csv")
    df["generated_type"] = preds
    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    paser = ArgumentParser()
    paser.add_argument("--checkpoint", type=str)
    args = paser.parse_args()
    gen(args.checkpoint)
