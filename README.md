# machine-learning
This repository contains Python code and its associated dataset for predicting heart disease using Decision Tree and various ensemble machine learning models.<br/>

## DecisionTree.py
DecisionTree.py can be prone to overfitting; testing accuracy may be lower than validation accuracy.<br/>
This program trains a single Decision Tree classifier with parameters such as:<br/><br/>
    max_depth=3: Maximum depth of the decision tree<br/>
    min_samples_split=5: Minimum number of samples required to split a node<br/>
    min_samples_leaf=1: Minimum number of samples required at a leaf node<br/>
    random_state=rng: Ensures reproducible results across runs<br/><br/>
The dataset is split into:<br/>
    70% training data<br/>
    20% validation data<br/>
    10% test data<br/>
After initial training and validation (including confusion matrix evaluation), the model is tuned using RandomizedSearchCV to find optimal hyperparameters before final testing.<br/>
  
## RandomForest.py
Uses the Random Forest ensemble method — multiple decision trees aggregated to improve accuracy.<br/>
Key parameters include:<br/>
    n_estimators=100: Number of trees in the forest<br/>
    max_features='sqrt': Number of features considered for each split<br/>
    criterion='gini': Split quality metric<br/>
    bootstrap=True: Sampling with replacement for each tree<br/>
The same train/validate/test process is followed. In current testing, Random Forest achieved ~98% prediction accuracy on both validation and test sets.<br/> 

## ExtraTrees.py
Implements the Extremely Randomized Trees (ExtraTrees) ensemble method. Similar to Random Forest, but splits are chosen more randomly, often improving variance reduction and training speed. Evaluated using the same data split and metrics.<br/>

## AdaBoost.py
Applies the AdaBoost ensemble boosting method, sequentially training weak learners (Decision Trees) and adjusting weights to focus on misclassified examples. Tuned using RandomizedSearchCV for maximum performance. In testing, AdaBoost achieved the highest overall accuracy and ROC-AUC score among all models.<br/>

## Stacking.py & StackingDT.py
Implements stacking ensemble learning — combining predictions from multiple base models (e.g., Decision Tree, Random Forest, Extra Trees, AdaBoost) to feed into a meta-model for final prediction. The StackingDT.py version uses Decision Trees as base estimators.<br/>

## CleaningData.py
Handles all dataset preparation steps:<br/>
    • Removal of duplicate rows<br/>
    • Splitting into training, validation, and test sets<br/>
    • Balancing classes via RandomUnderSampler to address class imbalance<br/>

## Summary of Results
Ensemble methods (AdaBoost, Random Forest, Extra Trees) consistently outperformed single Decision Trees.<br/>
Class balancing significantly improved recall, which is crucial in medical prediction.<br/>
AdaBoost (tuned) achieved the best results with a percision of 97.9%, recall of 90%, and Accuracy of 92.3% and a good ROC-AUC score on the test dataset.<br/>
