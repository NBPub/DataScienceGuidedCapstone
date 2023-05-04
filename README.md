# Notes

[Python Environment](#environment) | [Original Contents](#datascienceguidedcapstone)

## Submissions

| Notebook \# | Name, link | Status |
|---------|-------|----------|
| 02 | [Data Wrangling](/Notebooks/02_data_wrangling.ipynb) | ✔️ |
| 03 | [Exploratory Data Analysis](/Notebooks/03_exploratory_data_analysis.ipynb) | *submitted* |
| 04 | [Preprocessing and Training](/Notebooks/04_preprocessing_and_training.ipynb) | *in progress* |
| 05 | [Modeling](/Notebooks/05_modeling.ipynb) | *in progress* |

## Useful Tidbits

 - **03 [Exploratory Data Analysis](#03---eda)**
   - [Standard Deviation Calculation, *ddof*](#standard-deviation-calculation-biased-vs-unbiased)
   - [Principal Component Analysis, *with scikit-learn*](#principal-component-analysis-1)
   - [Data Exploration Plots](#more-eda-plots)
 - **04 [Preprocessing and Training](#04---pre-processing-and-training-data)**
   - [General notes](#process-notes)
   - [Principal Component Analysis, *background notes from lecture series*](#principal-component-analysis-2)
 - **05 [Modeling](#)**
   - [General notes](#process-notes-1)

### 03 - EDA

* [Standard Deviation, sample considerations](#standard-deviation-calculation-biased-vs-unbiased)
* Preprocessing data for machine learning --> **Standardization**
  * [Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler) | [Scaler Selection](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)

#### Standard Deviation Calculation: Biased vs Unbiased

Learning: ___

**scikit-learn** `decomposition.scale()` uses different method, ddof=1, than **pandas** `std()`, ddof=1<br>
**numpy** `std()` uses ddof=0!

> `np.decomposition.scale()` uses the biased estimator for standard deviation (ddof=0). This doesn't mean it's bad! 
It simply means it calculates the standard deviation of the sample it was given. 
The `pd.Series.std()` method, on the other hand, defaults to using ddof=1, that is it's normalized by N-1. 
In other words, the `std()` method default is to assume you want your best estimate of the population parameter based on the given sample.

 - [blog post](https://blogs.uoregon.edu/rclub/2015/01/20/biased-and-unbiased-estimates/)
 - [wikipedia](https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation)
 - **Docs:** [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std) | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html) | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)
   
#### Principal Component Analysis 1

From [notebook 3.5.3, Visualizing High Dimension Data](/Notebooks/03_exploratory_data_analysis.ipynb). 

**State Summary** DataFrame used, which contains per-state metrics in each row.

> find linear combinations of the original features that are uncorrelated with one another and order them by the amount of variance they explain
The basic steps in this process are:

>  1. scale the data *(important here because our features are heterogenous)*

`sk.learn.preprocessing.scale(<numeric_DataFrame>)`

- preprocessed data with [scikit-learn](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)
  - **standardize**: center each feature about mean, variance scaled to 1
  - [scikit-learn comparison](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py) of different scalers, transformers, and normalizers
- [`scale()` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)
  - **function warning:** In "production" use, don't apply to entire set to avoid data leaking from training to testing. Use `StandardScalar` within a `Pipeline`. `pipe = make_pipeline(StandardScaler(), LogisticRegression())`	
  - in this case, PCA simply being used for EDA, not preprocessing

>  2. fit the PCA transformation (learn the transformation from the data)

`sklearn.preprocessing.PCA().fit(<scaled-data>)`, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)

- [scikit-learn PCA User Guide](https://scikit-learn.org/stable/modules/decomposition.html#pca), for background information behind this process, see [PCA 2 below](#principal-component-analysis-2)
- **Data Exploration**
  - Cumulative sum of `<PCA_object>.explained_varaince_ratio_` vs PCA component
  - how many features needed to explain *`<>%`* variance of data

![PCA_1](/images/3-5-3-2_PCA.png "Comulative Variance by Dimension") 

>  3. apply the transformation, *dimensionality reduction*, to the data to create the derived features 

`<PCA_object>.transform(<scaled-data>)`, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA.transform). 

`<PCA_object>` *created from step 2,* `<scaled-data>` *from step 1.*

- **Data Exploration**
  - plot above showed 2 principal components (**PC1**, **PC2**) explain ~77% of the variance
  - these components were plotted against each other for each state
  - then, average ticket price for each state added to points (marker size/color)
    - ticket price quartiles calculated to provide a few discrete color choices, see [`qcut()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html?) 
	  - `fillna()` used to avoid deleting incomplete row of data
	  - `qcut` produced a column with **category** dtype, placeholder "NA" category had to be explicitly added to fill this column.

>  4. *optionally: use the derived features to look for patterns in the data and explore the coefficients*

- **Data Exploration**
  - *continued from PC1 vs PC2 plot above*
  - states well outside main cluster (extreme in either of the first two PCA component, **PC1** or **PC2**) explored further
    - `<PCA_object>.components_` method shows contribution of each original feature to PCA component
	- example:
	  - **NH** and **Ver** relatively high in **PC2**
	  - **PC2** had relatively high contribution from resorts per population and per area
	  - both of the small states had relatively large values in those metrics

#### More EDA Plots

![PCA_2](/images/3-5-3-4_PCA1_vs_PCA2.png "First two PCA components plotted against each other") 

![CorrHeatmap](/images/3-5-5-2_Correlation^2_Heatmap.png "Resort Feature correlation heatmap") 

![Feature_v_Price](/images/3-5-5-3_Feature_vs_Price.png "Resort Features vs Ticket Price") 

----


### 04 - Pre-processing and Training Data

See also:

- [**Tech with Tim** Machine Learning series](https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr)



#### Process Notes

 - load cleaned data, extract data for client resort
 - assign 70/30 split to training/testing
   - [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
   - inputs: data withot price, price data only, split parameters
   - returns: slices of the source data. train/test set for data without price `X` and set for price `y`.
 - extract text features from train/test data, `Name`, `state`, `Region`. Save the resort **Names**, drop the others.
   - only keep numeric data 
 - start by using the **mean** as a predictor, establish conservative baseline for future models
   - test against `sklearn.dummy.DummyRegressor` with `strategy=mean`
   - `<training_set_y>.mean()` ~= `<DummyRegressor.fit(<training_set_X>, <traing_set_y>).constant_`
 - **Metrics**
   - **R<sup>2</sup>**
     - notes
   - **Mean Absolute Error**
     - notes
   - **Mean Squared Error**
     - notes
   - **use with scikit-learn**
     - notes
 - **Initial Models**
   - data processing, scaling-py
 - **Pipelines** 
 - **Model Refinement**
 - **Assessment via Cross-Validation**
   - notes
 - Try again with **Random Forest Model**
 - **Model Selection**
   - compare performance between linear model and random forest model
 - **Data Quantiy Assessment**
   - do I need to ask for more data? is it worth the cost?
   - model learning curve

*keeping imports here for now to help find appropriate scikit-learn links, will delete as they are added*
```python
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
```

#### Principal Component Analysis 2

Notes from [PCA lecture series](https://www.youtube.com/watch?v=IbE0tbjy6JQ&list=PLbPhAbAhvjUzeLkPVnv0kc3_9rAfXpGtS&index=8), to provide background to EDA and Pre-Processing steps. 
The playlist also includes a Linear Algebra review from Khan Academy in the beginning.

<details><summary>. . . click to expand . . .</summary>

##### Background, Methods to Reduce Dimensionality

- preserve as much "structure" as possible, *variance*. be discriminative
- use domain knowledge - background knowledge
  - use existing "tricks", feature engineering: [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) for image analysis, [MFCCs](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) for audio analysis
- make assumptions about dimensions
  - **Independence** | **Smoothness** | **Symmetry** |
- Feature Selection vs Extraction
  - Selection: subset of the original dimension, pick the better predictors
  - Extraction: construct new dimension(s) from combinations of all the original dimensions
	
##### PCA Overview

Feature Extraction using linear combinations, define new set of dimensions.

  - **1**, first pick dimension along which the data shows the *greatest variability*
    - projecting to greatest variability helps maintain ***structure***
  - **2**, next dimension must be perpendicular to first, also fitting greatest remaining variability
  - **. . .** continue until original dimensionality
  - **transform** coordinates of data to new dimensions, choosing less than the original dimensionality

##### Gathering Principal Components

- center data, *subtract mean from each attribute*
- calculate covariance matrix, *how do dimensions relate to another?*
  - multiplying random vector by covariance matrix "turns" it towards dimension of greatest variance
  - vector that does not get turned is already pointing in that direction, will just change magnitude, scale
  - ***eigenvectors***, have corresponding *eigenvalues*, **principal components** are those with the greatest eigenvalues
    - [proof video](https://youtu.be/cIE2MDxyf80), maintain unit length when solving for eigenvectors to ensure unique solutions
	- [calculation of variance along each eigenvector](https://youtu.be/tL0wFZ9aJP8)
	
	
##### Projecting to New Dimensions

- already have *centered* data
- compute dot products of centered data with each eigenvector
- how many new dimensions to use? could saw how many are needed to explain `<>%` variance in data
  - [notebook 3.5.3.2, Cumulative Varaince Ratio](/Notebooks/03_exploratory_data_analysis.ipynb)
    - PCA fit to scaled data `PCA().fit(<scaled_data>)`
	- then cumulative sum of `<PCA object>.explained_varaiance_ratio_` was plotted against diemnsion, `.cumsum()` 
	- could also use **scree plot**, ex: K-means

![PCA_1](/images/3-5-3-2_PCA.png "Comulative Variance by Dimension") 

*First 3 dimensions explain %91 of the total variance*

```python
pca.explained_varaince_ratio_ =
array([0.5637151 , 0.2084733 , 0.13852841, 0.05269655, 0.03027124,
       0.00336836, 0.00294704])
```

##### Practical Issues

- covariance sensitive to large values --> normalize each attribute to unit variance so they don't swamp each other
- assumes linear subspace, won't perform well for certain datasets
- **unsupervised**: existing classification will be ignored and may become indistinguishable , only greatest variance matters
- **Linear Discriminant Analysis (LDA)** could be used instead. new dimension(s) seek to 
  - maximize separation between means of projected classes
  - minimize varaiance within each class
  - assumes data is normally distributed and there is existing separation between classes (in the mean)
    - i.e. not always better than PCA with classification
- can be "expensive" for some applications, [**cubic** complexity](https://inst.eecs.berkeley.edu/~cs10/fa15/discussion/04/Algorithmic%20Complexity%20Slides.pdf)
  - ex: web application with lots of redundant data
- not great for capturing outliers, special cases. *fine grained* classes

</details>

### 05 - Modeling

#### Process Notes

 - stuff
 - later

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

<details><summary>. . . click to expand . . .</summary>

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