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
