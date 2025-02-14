import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

#Differences from Stacking.py: Removed DT from the stacking model and instead userd it as the meta-model
#Just want to see if theres any difference 

# Setting up Data frame and initial models
rng = np.random.RandomState(30)
heart = pd.read_csv("Datasets/heart_statlog_cleveland_hungary_final.csv")

#after using the .drop_duplicates() method, drops down to 918 rows
heart = heart.drop_duplicates()

#Change percentages freely to split dataset into training/validation/testing:
trainProportionPercent = .70
validateProportionPercent = .20
testProportionPercent = .10

# Split dataset into training/validation data and testing data
trainAndValidate, testDatasetPercent = train_test_split(heart, test_size=testProportionPercent, random_state=rng)

# Split training/validation data into training data and validation data
trainDatasetPercent, validateDatasetPercent = train_test_split(trainAndValidate, train_size=trainProportionPercent/(trainProportionPercent+validateProportionPercent), random_state=rng)

# CHANGE X AND Y TO FIT INTO THE TESTING DATASET, NOT THE WHOLE DATASET
y = trainDatasetPercent['target']
X = trainDatasetPercent.drop('target', axis=1)

#undersampling on training dataset: make the number of presence/no presence features equal:
undersampling = RandomUnderSampler(random_state=rng)

#Merging the results back into a Dataframe
X_under, y_under = undersampling.fit_resample(X, y)
train_resampled = pd.DataFrame(X_under, columns=X.columns)
train_resampled['target'] = y_under


# Set up Models to be added to the stacking method with the best hyperparam we have found
ETModel = ExtraTreesClassifier(n_jobs=-1, n_estimators=100, min_weight_fraction_leaf=0.0, min_samples_split=4, min_samples_leaf=2, max_leaf_nodes=3, max_features=None, max_depth=2, criterion='gini', random_state=rng)
AdaModel = AdaBoostClassifier(n_estimators=100, learning_rate=0.7, algorithm='SAMME', random_state=rng)
RFModel = RandomForestClassifier(n_estimators=250, min_samples_split=20, max_features=3, max_depth=5, criterion='gini', bootstrap=True, random_state=rng)


ETfit = ETModel.fit(X_under, y_under)
Adafit = AdaModel.fit(X_under, y_under)
RFfit = RFModel.fit(X_under, y_under)

#TESTING THE MODEL:

#INITIALIZING THE VALIDATION DATASET:
#y_validate contains the target feature:
y_validate = validateDatasetPercent['target']

#X_validate contains the features, excluding the target feature:
X_validate = validateDatasetPercent.drop('target', axis=1)

# paramaterized entry for the stacking method for us to easily add more models
estimators = [('ET', make_pipeline(StandardScaler(), ETModel)), ('Ada', make_pipeline(StandardScaler(), AdaModel)),
             ('RF', make_pipeline(StandardScaler(), RFModel))]

# Stacking model initialization and accuracy before hyperparam tuning
StackingModel = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=2, max_features='log2', max_depth=7, criterion='gini', random_state=rng))
StackingModel.fit(X_under, y_under)
StackPrediction = StackingModel.predict(X_validate)
StackAccuracy = accuracy_score(y_validate, StackPrediction)
print('Stacking Validation Accuracy:', StackAccuracy)


# Displaying initial Confusion matrix
Stackingcm = confusion_matrix(y_validate, StackPrediction)

disp = ConfusionMatrixDisplay(confusion_matrix=Stackingcm, display_labels=StackingModel.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Validation Dataset')
plt.show()

report = classification_report(y_validate, StackPrediction)
print('Classification Report for Stacking Valiation Set:\n', report)

#Calculating ROC curve for Validation/Original ExtraTrees Model:
y_val_prob = ETModel.predict_proba(X_validate)[:, 1]  # Probability for the positive class
fpr_val, tpr_val, threshold_val = roc_curve(y_validate, y_val_prob)
roc_auc_val_ET = auc(fpr_val, tpr_val)

#Calculating ROC curve for Validation/Original AdaBoost Model:
y_val_prob = AdaModel.predict_proba(X_validate)[:, 1]  # Probability for the positive class
fpr_val, tpr_val, threshold_val = roc_curve(y_validate, y_val_prob)
roc_auc_val_Ada = auc(fpr_val, tpr_val)

#Calculating ROC curve for Validation/Original Stacking Model:
y_val_prob = StackingModel.predict_proba(X_validate)[:, 1]  # Probability for the positive class
fpr_val, tpr_val, threshold_val = roc_curve(y_validate, y_val_prob)
roc_auc_val_Stacking = auc(fpr_val, tpr_val)

#Hypertuning phase: Randomly determining the best parameters for the random forest model in hopes of better accuracy:
param_dist = {
#    'final_estimator': ["LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier"],
    'cv': [5, 10, 15, 20],
    'stack_method': ["auto", "bagging"],
    'n_jobs': [None, 1, 2, 4, 8, -1],
    'passthrough': [True, False],
   # 'verbose': [0, 2, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=StackingModel,
    param_distributions=param_dist,
    cv=10,                         # 10-fold cross-validation
    random_state=rng,              # Random state for reproducibility
    verbose=1                      # Print progress information
)

random_search.fit(X_under, y_under)
print("Best Hyperparameters:", random_search.best_params_)
print("Best Validation Accuracy (from cross-validation):", random_search.best_score_)

best_hyperparameters = random_search.best_params_

#Rebuilding Stacking with new found optimal hyperparameters:

optimized_Stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=2, max_features='log2', max_depth=7, criterion='gini', random_state=rng),
    cv=best_hyperparameters['cv'],
    stack_method=best_hyperparameters['stack_method'],
    n_jobs=best_hyperparameters['n_jobs'],
    passthrough=best_hyperparameters['passthrough'],
    verbose=0,
)


optimized_Stacking.fit(X_under, y_under)

#-----------------------------------------------------------------------------------------
#TESTING PHASE:  SAME AS OTHER FILES WITH STACKING AS THE MODEL
#-----------------------------------------------------------------------------------------
#INITIALIZING THE TESTING DATASET:
y_test = testDatasetPercent['target']
X_test = testDatasetPercent.drop('target', axis=1)

test_prediction = optimized_Stacking.predict(X_test)
test_accuracy = accuracy_score(y_test, test_prediction)
print('Optimized Stacking Test Accuracy:', test_accuracy)


# Generate confusion matrix
cm_test = confusion_matrix(y_test, test_prediction)

report = classification_report(y_test, test_prediction)
print('Classification Report for Optimized Stacking Test Set:\n', report)

# Creating ROC curve for Testing/optimized dataset
y_test_prob = optimized_Stacking.predict_proba(X_test)[:, 1]  # Probability for the positive class
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# Display the confusion matrix of the Validation Test:
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=optimized_Stacking.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Stacking Confusion Matrix of Testing Dataset')
plt.show()
#-----------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
#Displaying the ROC curve of both Random Forest models:
plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Stacking Validation ROC curve (area = {roc_auc_val_Stacking:.2f})')
plt.plot(fpr_test, tpr_test, color='green', lw=2, label=f'Stacking Test ROC curve (area = {roc_auc_test:.2f})')

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