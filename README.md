# Assignment3-IntroRL
Group 8 Assignment 3 Introduction to Reinforcement Learning
### Setting up the enviorment
```Bash
conda env create -f environment.yml
```
```Bash
conda activate RL_3_Group8
```
### Running
You can train the 3 strategies and save them as weights; however, the weights used in the experiment results in the paper are included in the repository and can be run directly
```Bash
python train_models.py 
```
Testing of differently trained DQNs + Plots, and excel files creation:
```Bash
python main.py          # runs test_script.py on all 3 strategies + combined bar plot + significance tests
python main.py test     # only the testing part
python main.py compare  # only the comparison part (needs the xlsx files from the test)
```
