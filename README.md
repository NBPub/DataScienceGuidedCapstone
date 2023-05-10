# Notes

[Python Environment](#environment) | [Original Contents](#datascienceguidedcapstone)

## Submissions

| Notebook \# | Name, link | Status |
|---------|-------|----------|
| 02 | [Data Wrangling](/Notebooks/02_data_wrangling.ipynb) | ✔️ |
| 03 | [Exploratory Data Analysis](/Notebooks/03_exploratory_data_analysis.ipynb) | ✔️ |
| 04 | [Preprocessing and Training](/Notebooks/04_preprocessing_and_training.ipynb) | *submitted* |
| 05 | [Modeling](/Notebooks/05_modeling.ipynb) | *in progress* |

## Useful Tidbits

 - **Statistics Review**
   - [Standard Deviation Calculation, *ddof*](#standard-deviation-calculation-biased-vs-unbiased)
   - [Principal Component Analysis, *background notes from lecture series*](#principal-component-analysis-1)
   - [Metrics](#metrics)
     - R<sup>2</sup>, MAE, MSE
 - **03 [Exploratory Data Analysis](#03---eda)**
   - [Principal Component Analysis, *with scikit-learn*](#principal-component-analysis-2)
   - [Data Exploration Plots](#more-eda-plots)
 - **04 [Preprocessing and Training](#04---pre-processing-and-training-data)**
   - [General notes](#process-notes)
     - [Start](#start)
	 - [Initial Models](#initial-models)
	 - [Model Refinement](#model-refinement)
	 - [Cross Validation](#assessment-via-cross-validation--user-guide---cross-validation-evaluating-estimator-performance)
	 - [GridSearch CV - Hyperparameter tuning](#hyperparameter-search-using-gridsearchcv)
	 - [RandomForest Model](#random-forest-model-example)
	 - [Model Selection](#model-selection)
	 - [Data Quantiy Assessment](#data-quantiy-assessment)
 - **05 [Modeling](#05---modeling)**
   - [General notes](#process-notes-1)

### Statistics Review

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

### Metrics



<details><summary>. . . click to expand . . .</summary>

*Example calculations used the DummyRegressor (mean) object described in [04 - Initial Models](#initial-models)*

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

----

### 03 - EDA

* Preprocessing data for machine learning --> **Standardization**
  * [Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler) | [Scaler Selection](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py)
 
#### Principal Component Analysis 2

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

![CorrHeatmap](/images/3-5-5-2_Correlation^2_Heatmap.png "Resort Feature correlation heatmap") 

**Ticket Price vs Resort Features: Distributions**
 - see [notebook](/Notebooks/03_exploratory_data_analysis.ipynb) for more notes

![Feature_v_Price](/images/3-5-5-3_Feature_vs_Price.png "Resort Features vs Ticket Price") 

</details>

----


### 04 - Pre-processing and Training Data

See also:

- [**Tech with Tim** Machine Learning series](https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr)

#### Process Notes

##### Start

<details><summary>. . . click to expand . . .</summary>
 
 - load cleaned data, extract data for client resort
 - assign 70/30 split to training/testing
   - [`sklearn.model_selection.train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
   - inputs: data withot price, price data only, split parameters
   - returns: slices of the source data. train/test set for data without price `X` and set for price `y`.
 - extract text features from train/test data, `Name`, `state`, `Region`. Save the resort **Names**, drop the others.
   - only keep numeric data 
 - start by using the **mean** as a predictor, establish conservative baseline for future models
   - test against `sklearn.dummy.DummyRegressor` with `strategy=mean`
   - `<training_set_y>.mean()` ~= `<DummyRegressor.fit(<training_set_X>, <training_set_y>).constant_`
   - **DummyRegressor** | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) | **[User Guide - Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)**
     - make predictions using simple rules, utilized as simple baseline to compare with "real" regressors
     - can also use median, specify quantile, or certain values
	 - can broadcast to array and has `fit()`, `predict()`, and other *scikit-learn* methods avalailable

</details>	 

***See [notes on model metrics](#metrics) above***

##### Initial Models

<details><summary>. . . click to expand . . .</summary>
  
  - missing values can be filled with scikit-learn, i.e. [imputed](https://en.wikipedia.org/wiki/Imputation_(statistics))
     - learn values to impute from the training split, apply them to the test split, then assess imputation
	 - tried with **median**:
	   - `filled_tr = <training_set_X>.fillna(<training_set_X>.median())`
	   - standardize data with `StandardScaler` so features can be compared, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
	     - `scaler = sklearn.preprocessing.StandardScaler()`, see also [MinMax Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
		 - `scaler.fit(<filled_tr>)`
		 - `scaled_tr = scaler.transform(<filled_tr>)`
		 - [notebook 4.8.1.1.3](/Notebooks/04_preprocessing_and_training.ipynb)
	   - train **Linear Regression** model with training split, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
	     - `sklearn.linear_model.LinearRegression().fit(<scaled_tr>, <training_set_y>)`
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

</details>	 
	 
##### Pipelines
   
[`make_pipeline()`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) | **[User Guide - Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html#combining-estimators)**
   
<details><summary>. . . click to expand . . .</summary>

   - chain steps of scikit-learn estimators together to create a `Pipeline` object.
     - use to cross-validate a series of steps while setting different parameters
       - intermediate steps must be "transforms", they must implement `fit` and `transform` methods
	   - last step, final estimator, only needs to implement `fit`
   - notebook example steps, i.e. steps/parameters passed to `PIPE = make_pipeline(<steps>)`
     - `impute.SimpleImputer(strategy = <>)`, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn-impute-simpleimputer)
	   - can pass different strategies, see above
	   - **[User Guide - Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html#impute)**
     - `preprocessing.StandardScaler()`
	   - scale data for models, [discussion above](#principal-component-analysis-1)
     - `linear_model.LinearRegression(strategy = <>)`
	   - train model, *discussion above*
   - use methods available for defined pipeline to perform its steps with a chosen dataset. *see available methods: `PIP.get_params().keys()`*
     - learning, scaling, and training with the training datasets
	   - `PIPE.fit(<training_set_X, training_set_y)`
	 - make predictions
	   - `PIPE.predict(<training_set_X>)`
	   - `PIPE.predict(<test_set_X>)`
	 - assess performance (R2, MAE, MSE, etc . . .)
	   - pipeline shows same results as manual process tried above!

</details>	 	   
	   
##### Model Refinement

[docs](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection) | **[User Guide - Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection)**

<details><summary>. . . click to expand . . .</summary>
  
  - model performance may have suggested overfitting, and features were blindly selected. can features be selected more smartly and can the model be refined?
   - utilize available **sklearn** [feature selection functions](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)
     - `SelectKBest` used for example, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest)
	   - method of *univariate feature selection* - choose features based on univariate statistical tests
	     - it removes all but the highest `k` scoring features, defaults to keeping `k=10`
		 - score function can be specified, [`feature_selection.f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression) used in example
		   - returns [F-statistic](https://en.wikipedia.org/wiki/F-test) and [p-values](https://en.wikipedia.org/wiki/P-value)
	   - add in between **scaling** and **training** steps in piepline, `SelectKBest(f_regression)`
   - `fit()` and `predict()` with new pipeline
     - performance was worse than first try, indicates limiting to 10 features was worse than using all of them
	 - try again with a different specification, `SelectKBest(f_regression, k=15)`
	   - slightly improved
   - should not keep repeating this step, as the training subset of the data has not been updated, and the model should generalize to new data

</details>	 

##### Assessment via Cross-Validation | [User Guide - Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation)

<details><summary>. . . click to expand . . .</summary>
   
   - partition training set into `n` folds, then train model on `n-1` folds and calculate performance on the unused training fold
   - cycle through `n` times with a different fold used for performance calculation each time 
   - results in `n` models trained  on `n` sets of data with `n` estimates of how the model performs
   - [`model_selection.cross_validate`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
     - `<results> = model_selection.cross_validate(NEWPIPE, <training_set_X>, <training_set_y>, cv=5)`
	   - folds (`n` in the bullets above) specified by `cv=`, defaults to 5-fold splitting strategy
	   - could analyze statistics of returned scores, elucidate uncertainty/variability of model
	     - see notebook: 3, 5, 10, and 15 fold tried
		 - explored more with **GridSearchCV** below
		   - "best" number of parameters determined to be 8, with a score: `0.68 ± 0.05` , see graph below
	   - returns dictionary with **fit_time**, **score_time**, and **test_score** as keys with arrays of length `cv`
	     - type of score can be specified, `scoring=<>`, *ex: 'neg_mean_absolute_error'*

</details>	 

##### Hyperparameter search using GridSearchCV

[docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) | **[User Guide - Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html#)**

<details><summary>. . . click to expand . . .</summary>

**Hyper-parameters** are not directly learnt within estimators, they are passed as arguments to the contructor of estimators. 
GridSearchCV searches over specified parameter values for an estimator.
  
  - for the example, anywhere from 1 to all 32 of the features were scored using GridSearch and the pipeline
    - `PARAMETERS` = `{'best_k_value': [k+1 for k in range(len(features))]}`
	- `<grid> = model_selection.GridSearchCV(PIPE, param_grid=PARAMETERS, n_jobs= -1)`
	  - `n_jobs` specifies processers to use, defaults to 1. all used by specifying `-1`
	  - `cv` can be specified as with `cross_validate()`, defaults to 5-fold splitting strategy
    - resulting object has various available methods, fit to training data `<grid>.fit(<training_set_X>, <training_set_y>)`
	- now has `cv_results` attribute
	  - dictionary with column header keys and numpy arrays for the values, `<grid>.cv_results_['mean_test_score']`
	  - can loaded as `DataFrame`
	    - see information for k=8, `pd.DataFrame(lr_grid_cv.cv_results_).loc[7,:]`
	- access best **k** value with `<grid>.best_params_`
	- access best **estimator** (features of most importance) with `<grid>.best_estimator_`
	  - `<linear_model_coefficients>` = `<grid>.best_estimator_.named_steps.linearregression.coef_`
	  - can represent as series with column names as index
	    - access via `<selected columns>` = `<grid>.best_estimator_.named_steps.selectkbest.get_support()`
		- `pd.Series(<linear_model_coefficients>, index = <dataset>.columns[<selected_columns>])`

</details>	 
	
![GridSearchCV_1](/images/4-9-8_GridSearchCV.png "Determine best k value for LinearRegression model") 
![GridSearchCV_2](/images/4-9-8_ParameterContributions.png "Determine best estimators and their contributions to model") 
   
##### Random Forest Model example

[`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn-ensemble-randomforestregressor), **[User Guide - Forest of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest)**

<details><summary>. . . click to expand . . .</summary>

   - see [references](https://scikit-learn.org/stable/modules/ensemble.html#b1998) to papers in the user guide
   - ***ensemble methods***, two main families:
     - **averaging:** average independent estimators to reduce variance
	   - ex: [decision trees](https://scikit-learn.org/stable/modules/tree.html#tree), RandomForest and Extra-Trees
     - **boosting:** base estimators built sequentially, reduce bias of the combined estimator
   - [parameter tuning](https://scikit-learn.org/stable/modules/ensemble.html#random-forest-parameters)
     - `n_estimators`, trees in the forest. more is better *at the cost of* computation and diminishing returns. default=`100`
	 - `max_features`, sizes of random subsets of features to consider when splitting nodes. lower reduces variance *at the cost of* bias. default = `1.0` = `None` AKA all features
	   - guide suggests trying `"sqrt"` as good starting value
   - RandomForest performance assessed with `cross_validate(RF-PIPE, <training_set_X>, <training_set_y>, cv=5)`
     - *test score* `0.70 ± 0.07`, *test score* for LinearRegression in initial CV discussion, `0.68 ± 0.05`
   - Model refined using **GridSearchCV** to evaluate a few *hyperparameters*, see above for process
     - n_estimators: `[int(n) for n in np.logspace(start=1, stop=3, num=20)]`. 0-1000
	 - with and without **scaling data** `StandardScaler()` or `None`. data shouldn't need to be scaled for decision trees
	 - missing values need to be filled, though! mean and median **imputation** strategies
       - *note!* `fit()` step took 45s
   - **HyperParameter** evaluation: `<grid>.best_params_`
     - `{n_estimators : 69}`
	 - `{imputation : 'median'}`
	 - `{scaling: None}`
   - `<results> = cross_validate(<grid>.best_estimator_, <training_set_X>, <training_set_y>, cv=5)`
     - *best_scores* = `0.71 ± 0.06`
   - feature importance to RandomForest model viewed using `<grid>.best_estimator_.named_steps.randomforestregressor.feature_importances_`
     - add feature names using `<dataset>.columns`

</details>	 

![RandomForest_1](/images/4-10-3_RandomForestModel_feature_importances.png "RandomForest model feature importance") 
   
   
##### Model Selection

<details><summary>. . . click to expand . . .</summary>

   - use the previously created **GridSearchCV** objects' `best_estimator_`'s to cross_validate models for comparison
     - specify `scoring='neg_mean_absolute_error` for interpretable scores
	 - "best" **LinearRegression** `10.5 ± 1.6`
	 - "best" **RandomForest** `9.6 ± 1.3`
   - **RandomForest** model shows lower mean absolute error and less variability than **LinearRegression** model. 
     - Model performance with the testing sets showed similar performance to these cross-validation results.

</details>	 

*metrics provided in `training subset, testing subset` format*

| Section | Model | Description | R<sup>2</sup> | MAE | MSE |
|---------|-------|----------|----------|----------|----------|
| 4.7 | DummyRegressor | Use mean as a predictor, R<sup>2</sup> should be 0 by definition. | `0`,<br> `-.003`| `17.9`,<br> `19.1` | `614`,<br> `581` |
| 4.8 | Linear, all parameters | Initial Model + first pipeline, fill missing with median, scale, linear regression | `0.82`,<br> `0.72` | `8.5`,<br> `9.4` | `112`,<br> `162` | 
| 4.9.1-3 | Linear, 10 parameters| SelectKBest (k=10) with f_regression added to pipeline before training model | `0.77`,<br> `0.63` | `9.5`,<br> `11.2` | `143`,<br> `217` | 
| 4.9.4 | Linear, 15 parameters | SelectKBest (k=15) with f_regression added to pipeline before training model | `0.79`,<br> `0.64` | `9.2`,<br> `10.5` | `127`,<br> `210` | 
| 4.11.1 | Linear, 8 parameters | SelectKBest (k=8) with f_regression added to pipeline before training model | *scoring via `cross_validate()`* | `10.5 ± 1.62`,<br> `11.8` |  | 
| 4.11.2 | RandomForest, best estimators | RandomForest model refined by GridSearch. Model made in section 4.10 | *scoring via `cross_validate()`* | `9.6 ± 1.35`,<br> `9.5` |   | 
   
##### Data Quantiy Assessment

[`sklearn.model_selection.learning_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve) | **[User Guide - Validation curves: plotting scores to evaluate models](https://scikit-learn.org/stable/modules/learning_curve.html)**
   
<details><summary>. . . click to expand . . .</summary>

 - do I need to ask for more data? is the "usefullness" worth the cost?
 - decompose generalization error of an estimator into
   - **bias** average error for different training sets
   - **variance** sensitivity to varying training sets
   - **noise** property of data
 - can be useful to plot influence of single hyperparameter on training and validation scores
   - is estimator *overfitting* or *underfitting* for hyperparameter values? see [`validation_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve)
 - **learning curve** shows how validation and training scores with varying numbers of training samples
   - what is the benefit of adding more training data?
   - does estimator suffer more from **variance** or **bias** error?
   - see user guide for examples
 - training set fractions of 0.2 to 1.0 examined with learning curve
   - curve suggests I have sufficient data (193 rows)
   - model improvement diminishes from 40-60 samples and seems to plateau after 70

</details>	 

![DataQuantity_1](/images/4-12_Data_quantity_assessment.png "Learning Curve - model improvement with training set size") 

----

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