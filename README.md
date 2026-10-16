# Assignment3-IntroRL
Group8 Assigment 3
### Setting up the enviorment
```Bash
conda env create -f environment.yml
```
```Bash
conda activate RL_3_Group8
```
### Running
```Bash
python main.py train    # trains the 3 strategies and saves them to weights
python main.py test     # runs test_script.py on all 3 strategies
python main.py compare  # combined bar plot + significance tests
python main.py          # all of the above
```
