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
python train_models.py  # trains the 3 strategies and saves them to weights/
python main.py          # runs test_script.py on all 3 strategies + combined bar plot + significance tests
python main.py test     # only the testing part
python main.py compare  # only the comparison part (needs the xlsx files from the test)
```
