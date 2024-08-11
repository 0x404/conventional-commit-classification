
# Dataset Description

`allcommits.csv`: This dataset includes 88,704 human-crafted conventional commits from 116 state-of-the-art projects that explicitly adopt CCS as their commit message convention.

`annotated_dataset.csv`: This dataset is a stratified sample from `allcommits.csv`, manually annotated based on our proposed definition, and contains 2,000 commits, with each type having 200 commits. This dataset is used for training, validation, and testing of models. It includes the commit message, the masked commit message (message with the type field removed), the commit diff, and the annotated type for each commit.