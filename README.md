# Notes

[Python Environment](#environment) | [Original Contents](#datascienceguidedcapstone)

## Submissions

| Notebook \# | Name, link | Status |
|---------|-------|----------|
| 02 | [Data Wrangling](/Notebooks/02_data_wrangling.ipynb) | ✔️ |
| 03 | [Exploratory Data Analysis](/Notebooks/03_exploratory_data_analysis.ipynb) | ✔️ |
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

<details><summary>. . . click to expand . . .</summary>

**scikit-learn** `decomposition.scale()` uses different method, ddof=1, than **pandas** `std()`, ddof=1<br>
**numpy** `std()` uses ddof=0!

> `np.decomposition.scale()` uses the biased estimator for standard deviation (ddof=0). This doesn't mean it's bad! 
It simply means it calculates the standard deviation of the sample it was given. 
The `pd.Series.std()` method, on the other hand, defaults to using ddof=1, that is it's normalized by N-1. 
In other words, the `std()` method default is to assume you want your best estimate of the population parameter based on the given sample.

 - [blog post](https://blogs.uoregon.edu/rclub/2015/01/20/biased-and-unbiased-estimates/)
 - [wikipedia](https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation)
 - **Docs:** [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std) | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.std.html) | [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html)

</details>
   
#### Principal Component Analysis 1

From [notebook 3.5.3, Visualizing High Dimension Data](/Notebooks/03_exploratory_data_analysis.ipynb). 

<details><summary>. . . click to expand . . .</summary>

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
  - these components were [plotted](#more-eda-plots) against each other for each state
  - then, average ticket price for each state added to points (marker size/color)
    - ticket price quartiles calculated to provide a few discrete color choices, see [`qcut()`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html?) 
	  - `fillna()` used to avoid deleting incomplete row of data
	  - `qcut` produced a column with **category** dtype, placeholder "NA" category had to be explicitly added to fill this column.

>  4. *optionally: use the derived features to look for patterns in the data and explore the coefficients*

- **Data Exploration**
  - *see PC1 vs PC2 plot below*
  - states well outside main cluster (extreme in either of the first two PCA component, **PC1** or **PC2**) explored further
    - `<PCA_object>.components_` method shows contribution of each original feature to PCA component
	- example:
	  - **NH** and **Ver** had high **PC2**, y-axis below
	  - **PC2** had relatively high contribution from resorts per population and per area
	  - both of the small states had large values in those metrics
	  
</details>

#### More EDA Plots

<details><summary>. . . click to expand . . .</summary>

![PCA_2](/images/3-5-3-4_PCA1_vs_PCA2.png "First two PCA components plotted against each other") 

**Ticket Price vs Resort Features: Correlation Heatmap**
 - *some new metrics were added, such as a resort's share of terrain parks from its state's total `terrain_park_state_ratio`
 - see [notebook](/Notebooks/03_exploratory_data_analysis.ipynb) for more notes

![CorrHeatmap](/images/3-5-5-2_r^2_Heatmap.png "Resort Feature correlation heatmap") 

**Ticket Price vs Resort Features: Distributions**
 - see [notebook](/Notebooks/03_exploratory_data_analysis.ipynb) for more notes

![Feature_v_Price](/images/3-5-5-3_Feature_vs_Price.png "Resort Features vs Ticket Price") 

</details>

----


### 04 - Pre-processing and Training Data

See also:

- [**Tech with Tim** Machine Learning series](https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr)
- process notes updated as of getting to **4.8.2** Initial Models - Pipelines



#### Process Notes

**Start**
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
   - **DummyRegressor** | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) | [User Guide - Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
     - make predictions using simple rules, utilized as simple baseline to compare with "real" regressors
     - can also use median, specify quantile, or certain values
	 - can broadcast to array and has `fit()`, `predict()`, and other *scikit-learn* methods avalailable

***See [notes on model metrics](#metrics) below***

**Initial Models**
   - missing values can be filled with scikit-learn, i.e. [imputed](https://en.wikipedia.org/wiki/Imputation_(statistics))
     - learn values to impute from the training split, apply them to the test split, then assess imputation
	 - tried with **median**:
	   - `filled_tr = <training_set_X>.fillna(<training_set_X>.median())`
	   - standardize data with `StandardScaler` so features can be compared, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
	     - `scaler = sklearn.preprocessing.StandardScaler()
		 - `scaler.fit(<filled_tr>)
		 - `scaled_tr = scaler.transform(<filled_tr>)`
		 - [notebook 4.8.1.1.3](/Notebooks/04_preprocessing_and_training.ipynb)
	   - train **Linear Regression** model with training split, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
	     - `sklearn.linear_model.LinearRegression().fit(<scaled_tr>, <training_set_y>)
	   - predict values using model, assess performance with R2, MAE, MSE
	     - `LinearRegression.predict(<scaled_tr>)` --> `sklearn.metrics.r2_score(<training_set_y>, <predicted>) = 0.82`
		   - linear regression model explains 80% of the variance on the training set, score was lower for test set, `0.72`
		   - lower value for the test set suggests overfitting
		 - `sklearn.metrics.mean_absolute_error(<training_set_y>, <predicted>) = 8.54`, and MAE for test set: `9.41`
		   - recall [MAE](#metrics) from mean DummyRegressor was ~19, so this is much better
		 - `sklearn.metrics.mean_squared_error(<training_set_y>, <predicted>) = 111.9`, and MSE for test set: `161.7` 
		   - also better than MSE from mean DummyRegressor
     - repeated exercise with **mean**, reported values are for `<train split>, <test split>`
	   - R2: `0.82, 0.72` | MAE: `8.54, 9.42` | MSE: `112.4, 164.4`	   
	   - basically same results as using median for imputation
     - can define function to perform these steps: impute data, scale features, train model, assess metrics . . .	
	 
**Pipelines** 
   - [User Guide - Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) | [`make_pipeline()` function docs](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline)
   - chain steps of scikit-learn estimators together to create a `Pipeline` object.
     - use to cross-validate a series of steps while setting different parameters
     - intermediate steps must be "transforms", they must implement `fit` and `transform` methods
	 - last step, final estimator, only needs to implement `fit`
   - goodness
   
**Model Refinement**
   - notes
   
**Assessment via Cross-Validation**
   - notes
   
**Random Forest Model example**
   - notes
   
**Model Selection**
   - compare performance between linear model and random forest model
   
**Data Quantiy Assessment**
   - do I need to ask for more data? is it worth the cost?
   - model learning curve

#### Metrics

Example calculations used the DummyRegressor (mean) object described above. 

<details><summary>. . . click to expand . . .</summary>

**[Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) AKA R<sup>2</sup>**
- one minus the ratio of residual variance to original variance $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
- `train = 0,	test = -0.003`, example model was "trained" on the training set
 - measure of goodness of fit of a model. for regression, a statistical measure of how well predictions fit data
   - values outside 0-1 may indicate a different model should be used
 - can be negative if linear function used without an intercept, or non-linear function used
   - may indicate model is worse at predicting than the mean
 - cautious with interpretation, as it always increases as number of variables in model increase
 - interpretable in amount of variance explained, but not in hose close predictions are to true values . . .
	 
**[Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) AKA MAE**
- average of the absolute error $$MAE = \frac{1}{n}\sum_i^n|y_i - \hat{y}|$$
- `train = 17.92,	test = 19.14`
 - easily interpreatble has seem units as quantity being estimated
   - *training set will be off by ~$18 dollars on average, test set by ~$19*
 - scale dependent, cannot compare predicted values that use different scales
	 
**[Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) AKA MSE**
- average of the errors squared $$MSE = \frac{1}{n}\sum_i^n(y_i - \hat{y})^2$$
- `train = 581.4,	test = 614.1`
 - alternative to **MAE**, almost always > 0
 - measure of the quality of an estimator, *decreases as error approaches 0*
 - incorporates *variance* and *bias* of an estimator
 - take square root to get **RMSE**, which like **MAE** has same units as quantity being estimated
	 
**[`sklearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)**, many more available
 - [`.r2_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score)
 - [`.mean_absolute_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error)
 - [`.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error) 
 - order of inputs might matter, can use `<function>?` to see docstring and `<function>??` to see source code from within a Jupyter notebook
   - example in [notebook 4.7.3](/Notebooks/04_preprocessing_and_training.ipynb)
   
</details>

*keeping imports here for now to help find appropriate scikit-learn links, will delete as they are added*
```python
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import , MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
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