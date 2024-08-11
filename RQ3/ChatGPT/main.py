from openai import OpenAI
import pandas
import tiktoken
import random
from tqdm import tqdm
from time import sleep
import numpy

client = OpenAI(api_key="your_api_key")
model_engine = "gpt-4-turbo-preview"

PROMPT_HEAD = (
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
    f"Please avoid any explanations and only provide the category label."
)
PROMPT_COMMIT_MESSAGE = f"- given commit message:\n{{message}}\n"
PROMPT_COMMIT_DIFF = f"- given commit diff: \n{{diff}}\n"
ENCODING = tiktoken.get_encoding("cl100k_base")
PROMPT_HEAD_TOKEN_LEN = len(ENCODING.encode(PROMPT_HEAD))


def prepare_prompt(row):
    message = PROMPT_COMMIT_MESSAGE.format(message=row["masked_commit_message"])
    message_encoding = ENCODING.encode(message)[:64]
    message = ENCODING.decode(message_encoding)

    diff = PROMPT_COMMIT_DIFF.format(diff=row["git_diff"])
    diff_encoding = ENCODING.encode(diff)[:1024-PROMPT_HEAD_TOKEN_LEN-64]
    diff = ENCODING.decode(diff_encoding)

    return PROMPT_HEAD + message + diff

def gpt_api(prompt):
    completion = client.chat.completions.create(
                    model=model_engine,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
    return completion.choices[0].message.content


if __name__ == "__main__":    
    df = pandas.read_csv("test.csv")
    if "gpt_type" not in df.columns:
        df["gpt_type"] = ""

    total_right = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if not pandas.isna(row["gpt_type"]):
            continue
        prompt = prepare_prompt(row)
        label = gpt_api(prompt)
        df.at[index, "gpt_type"] = label
        if label == row["annotated_type"]:
            total_right += 1
        print(f"GPT: {label} Answer: {row['annotated_type']} right_num = {total_right}")
        df.to_csv("test.csv", index=False)
        sleep(3)