import os
from typing import Iterator
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

import pygit2
import matplotlib.pyplot as plt
from conventional_pre_commit.format import is_conventional
from tqdm import tqdm


@dataclass
class Repository:
    name: str
    path: str


@dataclass
class Commit:
    sha: str
    message: str
    time: datetime

    @property
    def is_conventional(self):
        return is_conventional(self.message)


def _default_branch(repo: pygit2.Repository) -> str:
    branches = repo.listall_branches()
    for branch in ("main", "master"):
        if branch in branches:
            return branch
    if len(branches) == 1:
        return branches[0]
    raise ValueError(branches)


def get_all_commits(repo: pygit2.Repository) -> Iterator[Commit]:
    branch_name = _default_branch(repo)
    branch = repo.lookup_branch(branch_name)
    last_commit = branch.peel(pygit2.Commit)

    for commit in repo.walk(last_commit.id, pygit2.GIT_SORT_TIME):
        yield Commit(
            commit.hex,
            commit.message.strip(),
            datetime.fromtimestamp(commit.commit_time),
        )


def count_repo_commits(repo: pygit2.Repository) -> tuple[Counter, Counter]:
    if isinstance(repo, Repository):
        repo = pygit2.Repository(repo.path)

    conventional_counter, total_counter = Counter(), Counter()
    for commit in get_all_commits(repo):
        total_counter[commit.time.year] += 1
        if commit.is_conventional:
            conventional_counter[commit.time.year] += 1
    return conventional_counter, total_counter


def list_116_repo() -> list[Repository]:
    root = "/root/public/repo116/"
    repos = [Repository(name, os.path.join(root, name)) for name in os.listdir(root)]
    assert len(repos) == 116
    return repos


def list_500_repo() -> list[Repository]:
    root = "/root/public/nonconventional500/"
    repos = [Repository(name, os.path.join(root, name)) for name in os.listdir(root)]
    assert len(repos) == 500
    return repos


def count_all_commits(repos: list[Repository]) -> tuple[Counter, Counter]:
    conventional_counter, total_counter = Counter(), Counter()
    for repo in tqdm(repos):
        if isinstance(repo, Repository):
            repo = pygit2.Repository(repo.path)
        _cc, _tc = count_repo_commits(repo)
        conventional_counter += _cc
        total_counter += _tc
    return conventional_counter, total_counter


def draw_bar_pic(data: dict[str, float], savename):
    years = list(sorted(data.keys()))
    years = [year for year in years if year >= 2012 and year != 2024]
    percentages = [100 * data[year] for year in years]
    years = list(map(str, years))

    plt.style.use("seaborn-v0_8-bright")
    plt.figure(figsize=(10, 6))
    plt.bar(years, percentages, color="seagreen")

    plt.title("Conventional Commits Percentage by Year", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)

    plt.savefig(savename)


def main():
    repos_applied = list_116_repo()
    repos_non_applied = list_500_repo()

    conventional_counter_applied, total_counter_applied = count_all_commits(
        repos_applied
    )

    for year in total_counter_applied.keys():
        if year not in conventional_counter_applied:
            conventional_counter_applied[year] = 0

    conventional_counter_non_applied, total_counter_non_applied = count_all_commits(
        repos_non_applied
    )
    for year in total_counter_non_applied:
        if year not in conventional_counter_non_applied:
            conventional_counter_non_applied[year] = 0

    x = sorted(
        set(list(total_counter_applied.keys()) + list(total_counter_non_applied.keys()))
    )
    x = [xx for xx in x if xx >= 2017 and xx <= 2023]
    for xx in x:
        if xx not in conventional_counter_applied:
            conventional_counter_applied[xx] = 0
        if xx not in conventional_counter_non_applied:
            conventional_counter_non_applied[xx] = 0
        if xx not in total_counter_applied:
            total_counter_applied[xx] = 1
        if xx not in total_counter_non_applied:
            total_counter_non_applied[xx] = 1

    y_applied = [
        100 * conventional_counter_applied[xx] / total_counter_applied[xx] for xx in x
    ]
    y_non_applied = [
        100 * conventional_counter_non_applied[xx] / total_counter_non_applied[xx]
        for xx in x
    ]

    plt.plot(
        x,
        y_applied,
        color="#BE3536",
        linewidth=2,
        marker="x",
        markersize=6,
        label=f"CC Ratio (Applied)",
    )
    plt.plot(
        x,
        y_non_applied,
        color="#428646",
        linewidth=2,
        marker="o",
        markersize=6,
        label="CC Ratio (Not Applied)",
    )
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.savefig("commit_tren.pdf")


if __name__ == "__main__":
    main()
