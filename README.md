# A First Look at Conventional Commits Classification
[Conventional Commits🪄](https://www.conventionalcommits.org/en/v1.0.0/), as a specification for adding both human and machine-readable meaning to commit messages, is increasingly gaining popularity among open-source projects and developers. This study conducts a preliminary study of CCS, encompassing its application status and the challenges developers encounter when using it. We observe a growing popularity of CCS, yet developers do misclassify commits into incorrect CCS types, attributable to the absence of a clear and distinct definition list for each type. We have developed a more precise and less overlapping definition list to address this, grounded in industry practices and literature review. To assist developers in classifying conventional commits, we propose an approach for automated conventional commit classification. Our evaluation demonstrates that our model outperforms a series of baselines as well as ChatGPT4, showcasing promising potential for both industrial and academic applications.

This repository contains all the data and code we used in the study.


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

## Thematic Analysis
The details of thematic analysis in RQ2 can be found in `RQ2` directory.

## Reproduction

### Dataset
In the `Dataset` directory, we provide two datasets: one is 88,704 commits in the Conventional Commits format mined from 116 repositories, and the other is a dataset of 2,000 commits sampled and manually annotated from them. The manually annotated dataset is used for model training, validation, and testing. For detailed information about the datasets, please see the README.md in the `Dataset` directory.

### Environment
Our experiments are based on CUDA 11.8.0 and Ubuntu 22.04.3 LTS, with the main environmental dependencies as follows:
- `torch==2.1.2+cu118`
- `transformers==4.36.2`
- `pytorch-lightning==2.2.0.post0`
- `datasets==2.17.0`
- ...

We provide the version information of all the dependent environments in `requirements.txt`, which can be installed using `pip3 install -r requirements.txt`.

### Replication Method
- To replicate the experimental results of ChatGPT4, the code is located in `RQ3/ChatGPT`, which includes a `test.csv` file that is our test dataset. Fill in your OpenAI Key (with access to ChatGPT4) in the code, then run it.
- To replicate the BERT experimental results, the code is located in the `RQ3/BERT` directory. Use the command `python3 main.py --train` for training and `python3 main.py --test <checkpointpath>` for testing, where `<checkpointpath>` is the location of the saved parameter checkpoint.
- To replicate the experiments of Llama2 and CodeLlama, the code is in `RQ3/Llama2` and `RQ3/CodeLlama` directories, where training uses `python3 train.py` and testing uses `python3 infer.py --checkpoint <checkpointpath>`, with `<checkpointpath>` being the location of the saved parameter checkpoint, defaulting to the `llama-output` directory in the current directory.

*Note that to replicate Llama2 and CodeLlama, approximately two 24GB memory GPUs are needed. If the corresponding hardware conditions are not available, for ease of replication of our results, we provide the trained LORA parameters in the `CheckPoints` directory.
For instance, if you want to use the pre-trained CodeLlama parameters directly, you can do so with the following command:

```shell
cd RQ3/CodeLlama
python3 infer.py --checkpoint ../../CheckPoints/CodeLlama-checkpoints
```

## Cite Us

If you use this repository in your research, please cite us using the following BibTeX entry:

```bibtex
@inproceedings{zeng2025conventional,
  title={A First Look at Conventional Commits Classification},
  author={Zeng, Qunhong and Zhang, Yuxia and Qiu, Zhiqing and Liu, Hui},
  booktitle={Proceedings of the IEEE/ACM 47th International Conference on Software Engineering},
  pages={1--13},
  year={2025}
}
```
