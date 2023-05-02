# Notes

[Original Contents](#datascienceguidedcapstone)

## Submissions

| Notebook \# | Name, link | Status |
|---------|-------|----------|
| 02 | [Data Wrangling](/Notebooks/02_data_wrangling.ipynb) | *submitted* |
| 03 | [Exploratory Data Analysis](/Notebooks/03_exploratory_data_analysis.ipynb) | *in progress* |
| 04 | [Preprocessing and Training](/Notebooks/04_preprocessing_and_training.ipynb) | -- |
| 05 | [Modeling](/Notebooks/05_modeling.ipynb) | -- |

## Useful Tidbits

 - **03 [Exploratory Data Analysis](#03---eda)**
   - [Standard Deviation Calculation, *ddof*](#standard-deviation-calculation-biased-vs-unbiased)
   - [Principal Component Analysis, *with scikit-learn*](#principal-component-analysis)
 - **04 [Preprocessing and Training](#04---pre-processing-and-training-data)**
   - .
 - **05 [Modeling](#)**
   - .

### 03 - EDA

 * Background
   * [**Tech with Tim** Machine Learning series](https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr)
   * [PCA lecture series on youtube](https://www.youtube.com/watch?v=IbE0tbjy6JQ&list=PLbPhAbAhvjUzeLkPVnv0kc3_9rAfXpGtS&index=8)
   * [Standard Deviation, sample considerations](#standard-deviation-calculation-biased-vs-unbiased)
   * Preprocessing data for machine learning --> **Standardization**
     * [Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler) | [Scaler Selection](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

#### Standard Deviation Calculation: Biased vs Unbiased

Learning: ___

***scikit-learn's** `decomposition.scale()` uses different method, ddof=1, than **pandas** `std()`, ddof=1*<br>
***numpy** `std()` uses ddof=0!*

> `np.decomposition.scale()` uses the biased estimator for standard deviation (ddof=0). This doesn't mean it's bad! 
It simply means it calculates the standard deviation of the sample it was given. 
The `pd.Series.std()` method, on the other hand, defaults to using ddof=1, that is it's normalized by N-1. 
In other words, the `std()` method default is to assume you want your best estimate of the population parameter based on the given sample.

 - [blog post](https://blogs.uoregon.edu/rclub/2015/01/20/biased-and-unbiased-estimates/)
 - [wikipedia](https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation)
 - **Docs:** [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std) | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html) | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)
   
### Principal Component Analysis

From [notebook 3.5.3, Visualizing High Dimension Data](/Notebooks/03_exploratory_data_analysis.ipynb)

> find linear combinations of the original features that are uncorrelated with one another and order them by the amount of variance they explain
The basic steps in this process are:

>  1. scale the data *(important here because our features are heterogenous)*

- preprocessing data with [scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler), more from docs below
- [comparison](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py) of different scalers, transformers, and normalizers
- *refer to [`sklearn.decomposition.scale()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)*
  - **function warning:** A common mistake is to apply it to the entire data before splitting into training and test sets. This will bias the model evaluation because information would have leaked from the test set to the training set. In general, we recommend using StandardScaler within a Pipeline in order to prevent most risks of data leaking, `pipe = make_pipeline(StandardScaler(), LogisticRegression())`	
  - in this case, PCA simply being used for EDA, not preprocessing

>  2. fit the PCA transformation (learn the transformation from the data)

- `sklearn.preprocessing.PCA().fit(*<scaled-data>*)`, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
- **[User Guide](https://scikit-learn.org/stable/modules/decomposition.html#pca)**
  - *plot notes*

- **pandas** 
  - [`qcut()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html?)  
  - `fillna()` used to avoid deleting incomplete row of data
    - `qcut` produced a column with **category** dtype, placeholder "NA" category had to be explicitly added to fill this column.


>  3. apply the transformation to the data to create the derived features
>  4. *optionally: use the derived features to look for patterns in the data and explore the coefficients*


### 04 - Pre-processing and Training Data


## Environment

I used my base environment for this assignment, instead of the suggested virtual environment, to use more up to date packages. 
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

<details><summary>click to expand</summary>

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

</details>