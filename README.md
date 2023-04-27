# Note

## Submissions

| Notebook \# | Name, link | Status |
|---------|-------|----------|
| 02 | [Data Wrangling](/Notebooks/02_data_wrangling.ipynb) | *submitted* |
| 03 | [Exploratory Data Analysis](/Notebooks/03_exploratory_data_analysis.ipynb) | *in progress* |
| 04 | [Preprocessing and Training](/Notebooks/04_preprocessing_and_training.ipynb) | -- |
| 05 | [Modeling](/Notebooks/05_modeling.ipynb) | -- |

## Environment

I have used my base environment for this assignment, instead of the suggest virtual environment, to use more up to date packages. 
I will note any significant changes on the assignments.

```env
[packages]
pandas = "==1.5.3"
jupyter_client = "==7.4.9"
jupyter_core = "==5.3.0"
jupyter_server = "==1.23.4"
jupyterlab = "==3.5.3"
jupyterlab_server = "==2.10.2"
scikit-learn = "==1.2.2"
matplotlib = "==3.7.1"
seaborn = "==0.12.2"
lxml = "==" # not installed, will see if needed
```

----

# DataScienceGuidedCapstone

Hello students!
Welcome to the Data Science Guided Capstone! 

## Getting Started

Start by forking this repository to your personal GitHub account and cloning the fork to your local machine. 

**Note**: If forking and cloning a repo is new to you and/or github is new to you then it is strongly suggested to use [GitHub desktop](https://desktop.github.com/) and follow instructions in the docs [here](https://docs.github.com/en/free-pro-team@latest/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop).

From https://github.com/springboard-curriculum/DataScienceGuidedCapstone press the green "code" dropdown and then press "Open with GitHub Desktop". This will fork the springboard repository into your own github account and then clone that fork to your local machine - it is in here that you will do your work and push your changes back to your fork of the repo in your own github account. 

You will find the notebooks in the Notebooks/ directory. 

You will find instructions on how to complete and submit each step of the Guided Capstone in the course materials. Each subunit will focus on one step of the Capstone, corresponding to a step of the Data Science Method. Find the Jupyter Notebook corresponding to the subunit you are working on, and open it. Follow along as you are guided through the work, and fill in the blanks!

When you are done with the notebook, push the changes to your personal GitHub account.

## Pipenv

The `Pipefile` has all the python dependencies and requirements you should need. So you can use [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/) is you want to create a seperate python enviornment for this project. 

To install pipenv see [here](https://pipenv-fork.readthedocs.io/en/latest/#install-pipenv-today).

To create the env and install the required libraries (once you have pipenv installed) you can just do:
```
pipenv install
```

Then to activate the env and launch jupyter from this env you can do something like the below two commands:
```
pipenv shell
jupyter lab
```
