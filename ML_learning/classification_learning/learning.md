## Precision:
As simple as the word means precision i.e. precise. Precision is accuracy of the positive prediction. See the formula
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{TP}{TP&space;&plus;&space;FP}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\frac{TP}{TP&space;&plus;&space;FP}" title="\frac{TP}{TP + FP}" /></a> 

## Recall:
This is also called as TPR or **senstivity**.
This is the ratio of positive instances that are correctly detected by the classifier. The simple formula is:
\frac{TP}{TP + FN}

## F1 Score:
Combine both the precision and recall and it creates a matrix. In simple terms it compares two classifiers (if that what you wanna do). It follows simple approach i.e. harmonic mean of precision and recall. See the formula:
F1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}}

## ROC Curve:
Its True Positive Rate vs False Positive Rate. 
Where True Positive Rate is same as *Recall*. 
And *Flase Positive Rate* are the negative instances that are incorrectly classified as positive.
*In other words, 1 - True Negative Rate*. Also TNR is called as **Specificity**