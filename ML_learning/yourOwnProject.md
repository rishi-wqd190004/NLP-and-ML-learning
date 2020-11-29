# RMSE and MAE
When to use when and what is called as RMSE and MAE

Well if you know l1 and l2 transformation then you can skip the below read. Also in general its more like to just calculate the distance between two vectors.

## RMSE
Root Mean Square Error
In short you can remember like its prefereed for regression tasks majorly.
Now just to be bit funny, as it has the word root in means thats more like a person with most access (can be related with linux root). Hence it gets offended if you have more outliers. Won't bore you how it works and what are all the formulas and all. For more info of RMSE see this link. (https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)

## MAE
Also called as l1 norm or Manhattan norm. For more comparison here check this out (https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d). In short it just averages of the errors made by the classifier irrespective of the magnitude of the error.

## Data snooping
When model overfits and more precisely just mugs up the patterns it will not do good in testing data. More like the model becomes too optimistic.

## Basic ideas like keypoints
# 1. Verify that the data is not biased before you stream it into your modelling.

## Pearson's correlation
Also called as standard correlation coefficient. It shows the linear correlation between X and Y. Has a value of +1 and -1 with
    - total positive linear correlation --> +1
    - no linear correlation --> 0
    - total negative linear correlation --> -1
In general,
A correlation coefficient of 1 means that for every positive increase in one variable, there is a positive increase of a fixed proportion in the other. For example, shoe sizes go up in (almost) perfect correlation with foot length.
A correlation coefficient of -1 means that for every positive increase in one variable, there is a negative decrease of a fixed proportion in the other. For example, the amount of gas in a tank decreases in (almost) perfect correlation with speed.
Zero means that for every increase, there isn’t a positive or negative increase. The two just aren’t related.
For further refer to (https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/)

## Using Imputer
Create an imputer instance, specifying what you wanna replace missing vales with median of the attribute. Use sklearn.impute.SimpleImputer and see the documents for more. It works for filling the missing values.

## LabelEncoder
Majorly taken from sklearn.preprocessing.LabelEncoder. Changing the textual or categorical data into numerical data. For more checkout (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html). More importantly it contain only values between 0 and n_classes-1.

## DecisionTreeRegressor
To be fair with you all, DT is like a police personal or an investigator who looks for narrowing down answers and mainly work in building the order of those questions and then dig in to predict you are guilty or not....just an example.
A DecisionTreeRegressor model can be summarized with the below main points:
    - Decison tree are predictive model that use a set of binary rules to calculate a target value.
    - Each individual tree is merely a simple model that has branches, nodes and leaves.
Some simple terms:
    - Root node:
      - Represent the complete data
    - Splitting:
      - Dividing a node into two or more nodes or sub-nodes
    - Decision Node:
      - When a sub-node splits into further sub-nodes
    - Leaf Node:
      - Nodes that don't split
    - Pruning:
      - When remove sub-nodes is called pruning. Also opposite of splitting
    - Branch:
      - A part of the tree; another sub section of tree
    - Parent node:
      - A node which are divided into sub-nodes
    - Child node:
      - The nodes which are divided and genereated from the parent node
So in a DT, ** during training ** the model is fitted with any historical data that is relevant to the problem and the true value we want the model to predict. The model learns any relation between data and the target variable.
After the training is done, the tree:
    - produces the questions
    - also produces the order of these questions to be asked
Also for DecisionTreeRegressor() mainly uses RMSE to decide to split a node in two or more sub-nodes.

## sklearn.model_selection.cross_val_score
Randomly splits the data into 10 distinct subsets called folds and then evaluates the DT model 10 times. (see the code at line 240)
It estimate the performance of your model and also measure of how precise the estimate is i.e. its standard deviation.
For further searching of which scoring type one wants to use see (https://scikit-learn.org/stable/modules/model_evaluation.html)

## RandomForestRegressor
Training many DT on random subsets of the features and then averaging out their preedictions.

## Support Vector Machine (SVM)
    - Both used in classification and regression problems
    - Main aim of SVM is to differentiate the dataset into various classes and to find maximum marginal hyperplane, and its done:
      - First SVM, creates hyperplanes iteratively and then segregates the classes in best ways
      - Then choose the hyperplane which segregates the classes correctly
SVM is smart and uses kernel for various types of calculation, in other words it takes low dimensional input data space and then outputs high dimensional space.
    -## Linear:
        -svm.SVC(kernel='linear', C=1.0)
    - ## Radial Bias function
      - svm.SVC(kernel = 'rbf', gamma =‘auto’,C = C)

## GridSearchCV
Fine tuning the model by just telling which hyperparameter you want to experiment with and what values to try out.

## RandomizedSearchCV
Works the same way the GridSearchCV works but instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration. Benefits are:
    -## Instead of few values of hyperparameter, RandomizedSearchCV searches for many different values
    -## You are the one who have more control on which hyperparameter search should be allocated by just setting the number of iterations

## Ensemble Methods
We all know well that group of good people would impact more goodness as compared to a single individual, hence ensemble methods are just the same of grouping all your good methods together. Best example if, Random Forests perform well as compared to and individual Decision Tree.

## What is 90% or 95% confidence interval?
Basically, confidence interval for an unknown parameter is based on sampling the distribution of a corresponding estimator. Other words, it proposes a range of plausible values for an unknown parameter(for example, mean). Majorly it defines values which you are certain of. For more info (https://askinglot.com/what-is-a-90-confidence-interval)
