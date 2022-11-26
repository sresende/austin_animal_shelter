# Austin Animal Shelter
## Problem Statement

We are interested in building a classification model to predict the outcome for animals at the Austin Animal Shelter. We are looking to identify the model with the highest accuracy of predictions of outcome. These outcome predictions will be used to help the shelter allocate resources; by using the model to predict for different animals what their outcome will be, the shelter can predict how much money should be allocated to potential transfers, euthanasia, etc. 


## Description of Data
---
The data used in this project gives information on various features of an animal from the Austin Animal Shelter and its type of outcome. The dataset includes the target variable "OutcomeType" and 9 other features. The dataset has 26,729 rows. The data was source from the Kaggle Shelter Animal Outcomes competition.

## Data Dictionary
---

|Feature|Type|Dataset|Description|
|---|---|---|---|
|AnimalID|object|shelter_data| A alphanumerical ID unique to each animal| 
|Name|object|shelter_data| Name of the animal| 
|DateTime|datetime64|shelter_data| The date and time of the outcome| 
|OutcomeType|object|shelter_data| Outcome of the animal, either Adoption, Transfer, Return_to_owner, Euthanasia, or Died, later coded as 0-4| 
|OutcomeSubtype|object|shelter_data| Futher information specifying the outcome | 
|AnimalType|object|shelter_data|Type of animal, either Cat or Dog, later made into dummy variables| 
|SexuponOutcome|object|shelter_data| Animal sex and if it has been nuetered, later made into dummy variables| 
|AgeuponOutcome|object|shelter_data| Age of animal at the time of outcome| 
|Breed|object|shelter_data| Animal breed| 
|Color|object|shelter_data| Animal color| 
|AgeInYears|float|shelter_data| Age of animal at the time of outcome converted to years| 
|Pedigree|integer|shelter_data| Pedigree of the animal (0 if mixed, 1 if purebred)| 
|Solid_Color|integer|shelter_data| Whether the animal was one color or more than one (0 if more than one, 1 if solid)| 
|DayofWeek|integer|shelter_data| The day of week of the animal's outcome| 
|Month|integer|shelter_data| The month of the animal's outcome| 



## Data Cleaning 
---
First, we looked at the null values in our dataset. We decided not to use the name column or the OutcomeSubtype columns because they had many null values and also had so many categories that converting them to dummy variables would cause problems with the dimensionality of our data. We dropped the 18 null values in AgeuponOutcome because they were a very small percentage of our dataset. We then created a continuous numeric column that put all the AgeuponOutcome information, which included day, week, and year measurements, in terms of years. There were far too many breeds to create dummy variables, so we captured some of the influence of breed by making a column for pedigree, where animals that are mixes get a 0 and animals that are purebred or not a mix get a 1. There were also far too many color types to create dummy variables for so we decided to capture some of the influence of color by creating a column to indicate whethere the animal was a solid color (1) or a mix of colors(0). We changed the DateTime column to be the data type DateTime. We thought that different outcomes (such as adoption) might have a relation to the day of the week or month, so we created columns for the day of the week and the month that the outcome occured in. 


## Data Analysis
### Exploratory Data Analysis
----

A few operations were performed as an intial exploration of the data to look for trends and potential areas of interest. 

First, we looked a bar chart of the frequency of each outcome. Adoption and Transfer were the most frequent, while Died was the least frequent outcome. Next, we looked at the distribution of Age in Years of the animals at the time of the outcome. The histogram had a strong right skew and the majority of animals were between 0 and 2.5 years old at the time of outcome. Next, we looked at the boxplots of the Age in Years of the animal grouped by the Outcome Type. The distributions of Age for Return to Owner and Euthanasia were larger than the distribution for Adoption, Died, and Transfered. The mean value for Adoption was lower than the mean values for Euthanasia and Transfer. 

# ![](https://git.generalassemb.ly/sresende/project-5/blob/master/images/barplot.png) 
# ![](https://git.generalassemb.ly/sresende/project-5/blob/master/images/distribution.png) 
# ![](https://git.generalassemb.ly/sresende/project-5/blob/master/images/boxplot.png) 


### Classification Modeling 
----

We selected 'AnimalType', 'SexuponOutcome', 'AgeInYears', 'Pedigree', 'Solid_Color', 'DayofWeek', and 'Month' as our X features to predict outcome. We created dummy variables for each of our categorical variables from our X features. With the dummy variables, the baseline AnimalType was Cats, the baseline SexuponOutcome was Intact Female, the baseline DayofWeek was 0 (Monday), and the baseline Month was 1. We then performed a train_test_split on our data, stratified on y. 

Our baseline accuracy, if we predicted the majority class every time, was 0.40.

#### Logistic Regression
For our Logistic Regression model, we first ran a basic model without changing any parameters. This model had an accuracy score of 0.54 on the train data and 0.53 on the validation data. We then used GridSeach to test different hyperparameters to see if we could build a better model, checking hwether penalty should be l1 or l2 and whether C should be 0.001, 0.01, 0.1, 1, 10, 100, or 1000. The GridSearch returns the combination hyperparameters that built the best performing model. We then fit this model on our X_train data. This model had an accuracy score of 0.63 on the train data and 0.63 on the validation data. We could get the features importances for this model and the interpretation was that for the outcome as Adopted, the fact of being Spayed female contribute 2.9 times for dogs relative to cats. 

# ![](https://git.generalassemb.ly/sresende/project-5/blob/master/images/featuresImportance.png) 


#### KNN

For our K Nearest Neighbors Classifier, we built a pipeline with StandardScaler and KNeighborsClassifier so we could gridsearch for the best hyperparameters. StandardScaler was included in this pipeline because it is necessary to scale data before using KNN models. We used GridSeach to check whether StandardScaler with_mean should be True or False and whether with_std should be True or False. We also GridSearched over KNeighborsClassifier to check whether n_neighbors should be 3, 5, 7, or 10, whether weights should be uniform or distance, and whether p should be 1 or 2. The GridSearch returns the combination hyperparameters that built the best performing model. The KNN best parameters were: 'knn__p: 1, 'knn__n_neighbors': 10, 'knn__weights': 'uniform',  'ss__with_mean': False, and 'ss__with_std': False.
 
We then fit this model on our X_train data. This model had an accuracy score of 0.67 on the train data and 0.61 on the validation data. 


#### Random Forests
For our Random Forests Decision Trees Classifier, we  used GridSeach to check whether whether n_estimators should be 100, 150 or 200 and whether max depth of branches should be None, 2, 3, 5, or 7. The GridSearch returns the combination hyperparameters that built the best performing model. The Random Forest best parameters were:  'rf__max_depth': 7, 'rf__n_estimators': 150. We then fit this model on our X_train data. This model had an accuracy score of 0.64 on the train data and 0.62 on the validation data. 



## Conclusions and Recommendations 
----

Our best performing model, Logistic Regression, performed with an accuracy roughly 20% higher than the baseline accuracy. However, it still only had a 63% accuracy on the validation data. We would recommend that this model continue to be built on to best predict the outcomes of animals in the shelter. Feature engineering with the breed column specifically could help improve the accuracy. The current approach of classifiying each animal by whether they had the word "mix" or not in their breed is a rough approximation of the nuance of the impact that information on breed can have in determining animal outcomes. Further investigation into different approaches into feature engineering involving breed could have a large impact on model performance. 

Additional next steps for this model would be to test the LINE assumptions to see if inference is possible using the coefficients of the Logistic Regression model. Building a model that successfully meets the LINE assumptions could be very valuable to the shelter because it would give insight into which features have the most impact on animal outcome.

The current model is a useful tool for the shelter to consider, given that it performs better than the baseline accuracy, but for long term use we recommend additional time invested in the steps outlined above. 
