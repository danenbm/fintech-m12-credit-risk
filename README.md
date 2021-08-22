# Credit Risk Classification for Peer-to-peer Lending

## Overview of the Analysis

The purpose of this analysis is to create and evaluate two machine learning models that predict whether a given loan to a borrower will result in a healthy loan or a high-risk loan scenario. 

The machine learning models are based on historical borrower data from a peer-to-peer lending services company.  The input feature data include loan size, loan interest rate, the borrower's income, the borrower's debt-to-income ratio, the number of accounts the borrower has, the number of long-lasting negative indications (derogatory remarks) on the borrower's credit report, and the borrower's total debt.  For each row of input feature data, there is also a known historical outcome.  The "loan status" outcome indicates whether the loan was ultimately a healthy loan or became a high-risk loan.  This target outcome is exactly what the models will attempt to predict.

The historical dataset we are using is imbalanced.  When we look at the value counts of the loan status outcomes, we see there were 75036 healthy loans in the data, and 2500 high-risk loans - so only about 3% of the data is for high-risk loan scenarios.  Therefore, we will analyze one model using the original dataset, and a second model that included resampling.  In the case of resampling, we will randomly oversample the minority class (high-risk loans) before fitting the model with the data.

The stages of development for the first machine learning model (based on the original data) were:
* The data was split 75%/25% into training data and testing data.
* An sklearn LogisticRegression() model was chosen, and the model was fit using the training feature data and and training target outcome data.
* Next, the model was used to make target predictions for the test feature data.
* Finally, the model was evaluated by calculating a balanced accuracy score, a confusion matrix, and a classification report.

These steps were repeated for the second machine learning model, again using a LogisticRegression() model.  However, this time, instead of fitting the model with the original training data, the imblearn RandomOverSampler() was used to randomly oversample the minority class, resulting in a balanced dataset containing equal amounts of healthy and high-risk loans.

The second machine learning model was used to make target predictions on the same test feature data as used before, and the same evaluation steps were performed for the second model.

## Results
* Machine Learning Model 1:
  * Balanced Accuracy for Model 1 is about 95.2%, meaning that 95.2% of the time it predicts the correct outcome.  However, accuracy alone is misleading because the data used in Model 1 is imbalanced.  See Summary below for further discussion on this topic.
  * Precision: when the model chooses the `0` label, it is about 100% likely to be correct. When the model chooses the `1` label, it is about 85% likely to be correct.
  * Recall: when the loan is actually a `0` (healthy loan), there is about a 99% chance the model will predict it.  When the loan is actually a `1` (high-risk loan), there is about a 91% chance the model will predict it.
* Machine Learning Model 2:
  * Balanced Accuracy for Model 2 is 99.4%, which is slightly lower than with the original model.  However, this time it is based on a balanced dataset.
  * Precision: when the model chooses the `0` label, it is about 100% likely to be correct. When the model chooses the `1` label, it is about 84% likely to be correct.
  * Recall: for either label, if the label is actually `0` or actually `1`, there is about a 99% chance the model will correctly predict it.

## Summary

The Balanced Accuracy of Model 1 (which is based on the original data), is slightly higher than that of Model 2 (the model based on the resampled data).  As mentioned above, Accuracy alone is misleading because the original data is imbalanced.  Due to this imbalance, Model 1 may have an overall high accuracy by simply being heavily biased towards choosing the `0` label, since choosing `0` is likely to be correct in most cases.  We need to examine Precision and Recall to get a better idea of how the two models compare.

Looking at Precision for both Model 1 and Model 2, they are about the same.  If the model chooses the `1` label it is most likely to be correct, but with both models there is some loss of Precision when the `0` label is chosen.

Looking at Recall for both Model 1 and Model 2, when the loan is actually a `0` (healthy loan), the Recall is 99% for either model.  However, for Model 2, oversampling of the `1` (high risk loan) label data resulted in a notable improvement of the `1` label recall, from 91% Recall with the original data up to to 99% Recall with the oversampled dataset.  

In the case of flagging a high-risk loan, we likely want to optimize for high Recall (even at the expense of higher Precision), because we want to make sure we detect high-risk loans, even if that means we have more false positives.

In our case, Precision is comparable between the two models, but Model 2 has much better Recall for the high-risk loan target outcome.  We can always deal with false positives by manually looking deeper into the borrower's information to see if the false positive was a misclassification.

Therefore, in conclusion, this analysis recommends using Model 2 for predicting whether a given loan to a borrower will result in a healthy loan or a high-risk loan scenario.

