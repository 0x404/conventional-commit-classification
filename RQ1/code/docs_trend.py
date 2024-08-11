import os
from typing import Callable
from datetime import datetime
from dataclasses import dataclass
import pygit2
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter


@dataclass
class Commit:
    sha: str
    time: datetime


@dataclass
class Repository:
    name: str
    path: str


def search_keyword_in_all_files(
    repo: pygit2.Repository, commit_sha: str, keyword: str, valid_file_func: Callable
):
    commit = repo[commit_sha]
    search_result = False

    def search_in_tree(tree):
        nonlocal search_result

        if search_result:
            return

        for entry in tree:
            if search_result:
                return

            if entry.type == pygit2.GIT_OBJ_BLOB and valid_file_func(entry.name):
                blob = repo[entry.id]
                if keyword in blob.data.decode("utf-8", errors="ignore"):
                    search_result = True
                    return

            elif entry.type == pygit2.GIT_OBJ_TREE:
                sub_tree = repo[entry.id]
                search_in_tree(sub_tree)

    search_in_tree(commit.tree)
    return search_result


def _default_branch(repo: pygit2.Repository) -> str:
    branches = repo.listall_branches()
    for branch in ("main", "master"):
        if branch in branches:
            return branch
    if len(branches) == 1:
        return branches[0]
    raise ValueError(branches)


def _list_repo_commits(repo: pygit2.Repository) -> list[Commit]:
    branch = _default_branch(repo)
    last_commit = repo.branches[branch].peel()

    commits_sha = []
    for commit in repo.walk(last_commit.id, pygit2.GIT_SORT_TIME):
        commits_sha.append(
            Commit(commit.hex, datetime.fromtimestamp(commit.commit_time))
        )

    commits_sha = sorted(commits_sha, key=lambda commit: commit.time)
    return commits_sha


def _binary_find_first_keyword_occur_time(repo_path: str, keyword: str):
    repo = pygit2.Repository(repo_path)
    commits = _list_repo_commits(repo)

    def valid_func(filename: str) -> bool:
        return True

    l, r = 0, len(commits) - 1
    while l <= r:
        mid = (l + r) // 2
        if search_keyword_in_all_files(repo, commits[mid].sha, keyword, valid_func):
            r = mid - 1
        else:
            l = mid + 1
    if l >= len(commits):
        return None

    if search_keyword_in_all_files(repo, commits[l].sha, keyword, valid_func):
        return commits[l].time
    return None


def find_first_keyword_occur_time(repo_path: str, keyword: str):
    repo = pygit2.Repository(repo_path)
    commits = _list_repo_commits(repo)

    def valid_func(filename: str) -> bool:
        return filename.endswith(".md")

    for commit in tqdm(commits):
        if search_keyword_in_all_files(repo, commit.sha, keyword, valid_func):
            return commit.time
    return None


def list_116_repo() -> list[Repository]:
    root = "/root/public/repo116/"
    repos = [Repository(name, os.path.join(root, name)) for name in os.listdir(root)]
    assert len(repos) == 116
    return repos


def draw_bar_pic(x: list[datetime]):
    plt.style.use("seaborn-v0_8-bright")
    x = sorted(x)
    print(x)
    y = list(range(1, len(x) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color="seagreen", linestyle="-")
    plt.title("Conventional Commit Usage Over Time", fontsize=14)
    plt.xlabel("Period", fontsize=12)
    plt.ylabel("#Repos Start Applying Conventional Commits", fontsize=12)
    plt.savefig("docs_tren.png")


def draw_line_pic(x, y, ratio, factor):
    # plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        ratio,
        color="#BE3536",
        linewidth=2,
        marker="x",
        markersize=6,
        label=f"{factor} * Ratio of the projects",
    )
    plt.plot(
        x,
        y,
        color="#428646",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Number of the projects",
    )
    plt.xlabel("Year")
    plt.ylabel("Number")
    plt.legend()
    plt.savefig("docs_tren.pdf")


def main():
    repos = list_116_repo()
    keyword = "conventionalcommits.org"
    results = []
    for repo in tqdm(repos):
        time = _binary_find_first_keyword_occur_time(repo.path, keyword)
        if time is not None:
            results.append(time)

    assert len(results) == 116

    results = [t.year for t in results]
    accumulate = Counter(results)

    x, y = [], []
    for key in sorted(accumulate.keys()):
        x.append(int(key))
        y.append(accumulate[key])

    s = 0
    for index, cy in enumerate(y):
        y[index] += s
        s += cy

    factor = 1000
    ratio = [factor * cy / 3058 for cy in y]

    draw_line_pic(x, y, ratio, factor)


if __name__ == "__main__":
    main()
