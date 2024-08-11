import torch
import datasets
import pandas
from sklearn.model_selection import train_test_split
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import TrainerCallback
from contextlib import nullcontext


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
TOKENIZER.padding_side = "right"


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

        max_length = 1024 - len(prompt_head) - len(diff) - len(response) - len(message)

        if max_length > 0:
            pad = TOKENIZER.encode(
                TOKENIZER.eos_token,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        else:
            pad = []

        sample = {
            "input_ids": prompt_head + message + diff + response + pad,
            "attention_mask": [1] * 1024,
            "labels": [-100] * len(prompt_head + message + diff)
            + response
            + [-100] * len(pad),
        }

        return sample

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset


def make_dataset():
    df = pandas.read_csv(DATASET_PATH)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["annotated_type"], random_state=42
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=2 / 3, stratify=temp_df["annotated_type"], random_state=42
    )

    train_dataset = datasets.Dataset.from_pandas(train_df)

    train_dataset = preprocess_dataset(train_dataset)

    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset


def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def train():
    model = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )
    model, lora_config = create_peft_config(model)

    if ENABLE_PROFILER:
        wait, warmup, active, repeat = 1, 1, 2, 1
        total_steps = (wait + warmup + active) * (1 + repeat)
        schedule = torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        )
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{OUTPUT_DIR}/logs/tensorboard"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

        class ProfilerCallback(TrainerCallback):
            def __init__(self, profiler):
                self.profiler = profiler

            def on_step_end(self, *args, **kwargs):
                self.profiler.step()

        profiler_callback = ProfilerCallback(profiler)
    else:
        profiler = nullcontext()

    config = {
        "lora_config": lora_config,
        "learning_rate": 1e-4,
        "num_train_epochs": 5,
        "gradient_accumulation_steps": 1,
        "per_device_train_batch_size": 10,
        "gradient_checkpointing": False,
    }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        bf16=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch_fused",
        max_steps=total_steps if ENABLE_PROFILER else -1,
        **{k: v for k, v in config.items() if k != "lora_config"},
    )

    train_dataset = make_dataset()

    with profiler:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(TOKENIZER),
            callbacks=[profiler_callback] if ENABLE_PROFILER else [],
        )

        trainer.train()

    model.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()
