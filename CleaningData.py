import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#To be used to explore dataset: also to be used in each model:

rng = np.random.RandomState(28)

#Original dataset contains 1190 instances
heart = pd.read_csv('Datasets/heart_statlog_cleveland_hungary_final.csv')
print("Total dataframe distribution(before dropping duplicates)", heart['target'].value_counts())

#after using the .drop_duplicates() method, drops down to 918 rows
heart = heart.drop_duplicates()
print("Total dataframe distribution", heart['target'].value_counts())


trainProportionPercent = .70
validateProportionPercent =.20
testProportionPercent = .10

# Split dataset into training/validation data and testing data
trainAndValidate, testDatasetPercent = train_test_split(heart, test_size=testProportionPercent, random_state=rng)

# Split training/validation data into training data and validation data
trainDatasetPercent, validateDatasetPercent = train_test_split(trainAndValidate, train_size=trainProportionPercent/(trainProportionPercent+validateProportionPercent), random_state=rng)
#trainDatasetPercent: Dataset with 75% of data
#validateDatasetPercent: Dataset with 10% of data
#testDatasetPercent: Dataset with 15% of data

#Info on each Dataframe:
print("Training dataset distribution of target value:",trainDatasetPercent['target'].value_counts())
print("Validation dataset distribution of target value:",validateDatasetPercent['target'].value_counts())
print("Test dataset distribution of target value:",testDatasetPercent['target'].value_counts())

#Preparing undersampling on training dataset
y = trainDatasetPercent['target']
X = trainDatasetPercent.drop('target', axis=1)

#Replacement is set to false by default, ensuring no duplicates in the new dataframe
undersampling = RandomUnderSampler(random_state=rng)

#Mergin the results back into a Dataframe
X_under, y_under = undersampling.fit_resample(X,y)
train_resampled = pd.DataFrame(X_under, columns= X.columns)
train_resampled['target'] = y_under

#Result: DataFrame with an equal amount of instances with/without heart disease:
print(train_resampled)

#Info on the resampled DataFrame:
target_total = train_resampled['target'].value_counts()
print("Training dataset distribution after undersampling:",target_total)

#Dataframe contains no missing values in any of its features:
missing_features = train_resampled.isnull().sum()
print(missing_features)

#Each features datatype: helps to know what type it is to calculate entropy:
print(train_resampled.dtypes)

#TODO: Determine weak features: gini impurity or entropy (if time allows)



