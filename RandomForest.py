#Random Forest: A collection of many decision trees
#Instances from the original dataset are randomly selected using bootstrapping, or random sampling with replacement. 
#Each bootstrap sample is used to fit a classification tree.
#In a random forest model, each classification tree is grown using a random sample of features.
#In our case, the random forest will take features like age, sex, cholesterol, etc. randomly out of the random samples

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, RandomizedSearchCV

#INITIALIZE PHASE: Creating random forest and splitting dataset into training/validation/testing phase
#-----------------------------------------------------------------------------------------
rng = np.random.RandomState(28)
trainProportionPercent = .70
validateProportionPercent =.20
testProportionPercent = .10

Heart = pd.read_csv('Datasets/heart_statlog_cleveland_hungary_final.csv')
Heart = Heart.drop_duplicates()

# Split dataset into training/validation data and testing data
trainAndValidate, testDatasetPercent = train_test_split(Heart, test_size=testProportionPercent, random_state=rng)

# Split training/validation data into training data and validation data
trainDatasetPercent, validateDatasetPercent = train_test_split(trainAndValidate, train_size=trainProportionPercent/(trainProportionPercent+validateProportionPercent), random_state=rng)

# CHANGE X AND Y TO FIT INTO THE TESTING DATASET, NOT THE WHOLE DATASET
y = trainDatasetPercent['target']
X = trainDatasetPercent.drop('target', axis=1)

#Undersampling on training dataset:
undersampling = RandomUnderSampler(random_state=rng)

#Merging the results back into a Dataframe
X_under, y_under = undersampling.fit_resample(X,y)
train_resampled = pd.DataFrame(X_under, columns= X.columns)
train_resampled['target'] = y_under

RandomHeart = RandomForestClassifier(n_estimators=100, max_features= 'sqrt', max_depth= 3, criterion='gini', bootstrap=True)
RandomHeart.fit(X_under,y_under)
#-----------------------------------------------------------------------------------------

#VALIDATION PHASE
#-----------------------------------------------------------------------------------------
#INITIALIZING THE VALIDATION DATASET:
y_validate = validateDatasetPercent['target']
X_validate = validateDatasetPercent.drop('target', axis = 1)

prediction = RandomHeart.predict(X_validate)
accuracy = accuracy_score(y_validate, prediction)
print('Validation Accuracy:', accuracy)

cm = confusion_matrix(y_validate, prediction)
# Display the confusion matrix of the Validation Test:
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RandomHeart.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Validation Dataset')
plt.show()

#Outputing the precision, recall, and f1-score of the model. Support is the number of instances in each class (0 = no presence, 1= presence)
report = classification_report(y_validate, prediction)
print('Classification Report for Valiation Set:\n', report)

#Calculating ROC curve for Validation/Original Random Forest Model:
y_val_prob = RandomHeart.predict_proba(X_validate)[:, 1]  # Probability for the positive class
fpr_val, tpr_val, threshold_val = roc_curve(y_validate, y_val_prob)
roc_auc_val = auc(fpr_val, tpr_val)
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
#Hypertuning phase: Randomly determining the best parameters for the random forest model in hopes of better accuracy:
param_dist = {
    'n_estimators': [50, 100, 200, 250, 300, 400, 500],                  
    'max_features': ['sqrt', 'log2', 1, 2, 3],              
    'max_depth': [1, 2, 3, 4, 5, 6],                  
    'criterion': ['gini'],                
    'bootstrap': [True],
    'min_samples_split': [2, 5, 10, 20]
}

random_search = RandomizedSearchCV(
    estimator=RandomHeart,
    param_distributions=param_dist,                   
    cv=10,                         # 10-fold cross-validation
    random_state=rng,              # Random state for reproducibility
    verbose=1                      # Print progress information
)

random_search.fit(X_under, y_under)
print("Best Hyperparameters:", random_search.best_params_)
print("Best Validation Accuracy (from cross-validation):", random_search.best_score_)

best_hyperparameters = random_search.best_params_

#Rebuilding Random Forest with new found optimal hyperparameters:

optimized_RandomForest = RandomForestClassifier(
    n_estimators=best_hyperparameters['n_estimators'],
    max_features=best_hyperparameters['max_features'],
    max_depth=best_hyperparameters['max_depth'],
    criterion=best_hyperparameters['criterion'],
    bootstrap=best_hyperparameters['bootstrap'],
    min_samples_split=best_hyperparameters['min_samples_split'],
    random_state=rng
)


optimized_RandomForest.fit(X_under, y_under)

#-----------------------------------------------------------------------------------------


#TESTING PHASE: 
#-----------------------------------------------------------------------------------------
#INITIALIZING THE TESTING DATASET:
y_test = testDatasetPercent['target']
X_test = testDatasetPercent.drop('target', axis = 1)

test_prediction = optimized_RandomForest.predict(X_test)
test_accuracy = accuracy_score(y_test, test_prediction)
print('Test Accuracy:', test_accuracy)


# Generate confusion matrix
cm_test = confusion_matrix(y_test, test_prediction)

report = classification_report(y_test, test_prediction)
print('Classification Report for Test Set:\n', report)

# Creating ROC curve for Testing/optimized dataset 
y_test_prob = optimized_RandomForest.predict_proba(X_test)[:, 1]  # Probability for the positive class
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# Display the confusion matrix of the Validation Test:
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=optimized_RandomForest.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Testing Dataset')
plt.show()
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
#Displaying the ROC curve of both Random Forest models:
plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Validation ROC curve (area = {roc_auc_val:.2f})')
plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Test ROC curve (area = {roc_auc_test:.2f})')

# Plot the 45-degree line representing random guessing
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Axis Labels and Title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



#-----------------------------------------------------------------------------------------








