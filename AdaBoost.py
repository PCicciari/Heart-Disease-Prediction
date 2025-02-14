import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

rng = np.random.RandomState(28)

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
X_under, y_under = undersampling.fit_resample(X,y)
train_resampled = pd.DataFrame(X_under, columns= X.columns)
train_resampled['target'] = y_under

AdaModel = AdaBoostClassifier(n_estimators=100, random_state=rng)

BasicModel = AdaModel.fit(X_under, y_under)

Y_pred = BasicModel.predict(X_under)

print("Basic Recall: ", metrics.recall_score(y_under, Y_pred))
print("Basic Precision: ", metrics.precision_score(y_under, Y_pred))
print("Basic Accuracy: ", metrics.accuracy_score(y_under, Y_pred))
print("Basic F1 Score: ", metrics.f1_score(y_under, Y_pred))

print("------------------------------------------------------------------")
print("------------------------------------------------------------------")
print("------------------------------------------------------------------")


#TESTING THE MODEL:

#INITIALIZING THE VALIDATION DATASET:
#y_validate contains the target feature:
y_validate = validateDatasetPercent['target']

#X_validate contains the features, excluding the target feature:
X_validate = validateDatasetPercent.drop('target', axis=1)

prediction = BasicModel.predict(X_validate)
accuracy = accuracy_score(y_validate, prediction)
print('Validation Accuracy:', accuracy)


# Generate confusion matrix
cm = confusion_matrix(y_validate, prediction)

# Display the confusion matrix of the Validation Test:
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=AdaModel.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Validation Dataset')
plt.show()

#Outputing the precision, recall, and f1-score of the model. Support is the number of instances in each class (0 = no presence, 1= presence)
report = classification_report(y_validate, prediction)
print('Classification Report for Valiation Set:\n', report)

#Calculating ROC curve for Validation/Original model
y_val_prob = BasicModel.predict_proba(X_validate)[:, 1]  # Probability for the positive class
fpr_val, tpr_val, threshold_val = roc_curve(y_validate, y_val_prob)
roc_auc_val = auc(fpr_val, tpr_val)





param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.1, 0.4, 0.5, 0.7, 0.1],
    'algorithm': ["SAMME", "SAMME.R"],
}

random_search = RandomizedSearchCV(
    estimator=BasicModel,
    param_distributions=param_dist,
    cv=10,                         # 10-fold cross-validation
    random_state=rng,              # Random state for reproducibility
    verbose=1                      # Print progress information
)

random_search.fit(X_under, y_under)


print("Best Hyperparameters:", random_search.best_params_)
print("Best Validation Accuracy (from cross-validation):", random_search.best_score_)

best_hyperparameters = random_search.best_params_


optimized_AdaBoost = AdaBoostClassifier(
    n_estimators=best_hyperparameters['n_estimators'],
    learning_rate=best_hyperparameters['learning_rate'],
    algorithm=best_hyperparameters['algorithm'],
    random_state=rng)


optimized_AdaBoost.fit(X_under, y_under)

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------



#INITIALIZING THE TESTING DATASET:
y_test = testDatasetPercent['target']
X_test = testDatasetPercent.drop('target', axis=1)

test_prediction = optimized_AdaBoost.predict(X_test)
test_accuracy = accuracy_score(y_test, test_prediction)
print('Test Accuracy:', test_accuracy)


# Generate confusion matrix
cm_test = confusion_matrix(y_test, test_prediction)

report = classification_report(y_test, test_prediction)
print('Classification Report for Test Set:\n', report)

# Creating ROC curve for Testing/optimized dataset
y_test_prob = optimized_AdaBoost.predict_proba(X_test)[:, 1]  # Probability for the positive class
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# Display the confusion matrix of the Validation Test:
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=optimized_AdaBoost.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix of Testing Dataset')
plt.show()
#------------------------------------------------------------------------------------------------------------


# Plot ROC Curves
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


