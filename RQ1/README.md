# File Structure Description

- data
    - `data/116_repos_adopted_CCS.csv`: The 116 repositories identified in RQ1, as well as how these repositories adopt CCS.
    - `500_repos_not_aopted_CCS.txt`: The 500 repositories in RQ1 that do not use CCS are used to investigate the trend of CCS adoption from developers’ perspective.
- code
    - contains the code we investigate the trend of CCS adoption.

Since we need to extract information from the commit history of the repo, it is necessary to download the repo locally. If you want to run these scripts, you first need to clone the repo to your local machine and then modify the directory paths in list_116_repo and list_500_repo in the code. As these repos are still being actively updated, the version of the repo you clone may include some commits that were not present during our research. Therefore, the data you obtain might slightly differ from the data in the paper.