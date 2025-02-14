# machine-learning
This repository conatins python code and its associated dataset needed to generate a decision tree model and other ensemble models.<br/>
In order for program to work, you may need to change the file path of the "pd.read_csv" line on each program.<br/> 
## DecisionTree.py
***DecisionTree.py is currently prone to overfitting; Testing accuracy is lower than validation accuracy.***<br/>
DecisionTree.py is a program that utilizes only one decision tree. It first creates a basic model with some parameters:<br/><br/>
    max_depth=3: The total depth of the decision tree<br/>
    min_samples_split=5: The minimum number of samples per node split<br/>
    min_samples_leaf=1: The minimum number of samples per leaf/termination node<br/>
    random_state = rng: Used to reproduce results through program reruns<br/><br/>
After the decision tree model is made, the model is then fit into a training set, split beforehand from the original dataset<br/>
The spliiting of the original dataset is:<br/>
    trainProportionPercent = .75: training dataset is 75% of total data<br/>
    validateProportionPercent =.10: validation dataset is 10% of total data<br/>
    testProportionPercent = .15: test dataset is 15% of total data<br/>
The model is then validated with the validation dataset, with its respective confusion matrix.<br/>
After determining the validation accuracy of the decision model, we hypertune the decision tree's parameters with a process known as Randon Search<br/>
Random Search "searches" for the most optimal decision tree based on the parameters you can change on creation. This includes the ones stated above.<br/>
After finding the most optimal parameters the decision tree is rebuilt, and tested with the test set.<br/>

  
## RandomForest.py
RandomForest.py is a program that utilizes the Random Forest bagging ensemble technique, which is the use of multiple decision trees to predict an outcome<br/>
In this program a randome forest is created with the followjng properties/parameters:<br/>
    n_estimators=100: determines how many decision trees are in the "forest"<br/>
    max_features='sqrt': determines how many maximum features are used in each decision tree<br/>
    criterion='gini': determines the method of information gain<br/>
    bootstrap=True: also known as bootstrap aggregation, determines if decision trees can retrieve instances of a dataset WITH replacement<br/>
The same procedure applies here as in the normal DecisionTree.py method: test the model, evaluate its performance on the validation set, hypertune its parameters as necessary, and test the new model with the new parameters. In the current build of RandomForest.py, no parameters were changed; the validation accuracy and testing accuracy are both similar: each gain a prediction score of approx. 98%. 


In progress: creating multiple ensemble models to evaluaate each respective model's accuracy. 
Possible ensemble methods: ~~Random Forest~~, AdaBosst/GradientBoost, ExtraTrees etc.
