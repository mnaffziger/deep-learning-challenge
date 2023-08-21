# deep-learning-challenge
Module 21 Challenge: Utilizing neural networks with TensorFlow and Keras on Google Colab

## Overview
Overview of the analysis: Explain the purpose of this analysis.
For this analysis, the nonprofit group, Alphabet Soup, wanted a tool to aid in their selecton process of financially successful applicants.  The models report on within this report utilized a dataset containing over 34,000 organizations that had received funding from Alphabet Soup over the years.  This dataset contained the following informaiton:
*   Identification: EIN and Name
*   Application type: Alphabet Soup's internal application classification
*   Affiliation: Industrial sector
*   Classification: Government based classification
*   Use Case: Applicants claimed usage of the funding
*   Organization: Applicants organization type
*   Status: Funding provided is active or not
*   Income amout: Applicants stated income
*   Special Considerations: Applicant's notes of interest in selection process
*   Ask Amount: Funding amount requested
*   Is Successful: Alphabet Soup's internal auditing of applicant's usage of funding

All of the features, excluding EIN, of the data set were utilized in training of the neural netowrk model.  The 'Is Successful' feature was utilized as the funding target since Alphabet Soup's internal selection process were not disclosed at the time of the model's creation.

---
## Results
Results: Using bulleted lists and images to support your answers, address the following questions:
*   Initial attempts lead to a training accuracy no greater that 73%.
*   Additional data from the original dataset was required to puch the accuracy greater than 75%. 
*   Models' training accuracy with the 'NAMES' included were between 78% - 80%.
*   There was a trend towards overfitting as the number of epochs increased
---
### Data Preprocessing

#### **Model Initialization**
Every model utilized the 'Is Successful' feature as its target.  As per the client's request, this predictive model was to aid in the selection of potentially successful applicants. Due to the complexity of the features, some of the categorical features were bined before encoding.  For all models the following features were binned with the corresponding cut off values:
*   Application Type:  Types less than 500 application were classified as 'other'
*   Classifiacation: Organization classes that accounted for less than 1000 applicants were reclassified as 'other'
*   Names:  If the applicant applied for less than 5 funding requests, they were classified as 'other'

Two models were utilized for training purposes.  The major difference between the two models is the utilization of the 'Names' feature.  
*   Model 1 did not contain the 'Names' column in the cleaned and encoded dataset.  
*   Model 2 utilized the 'Names' feature.  Due to the variability of applicants, this feature was binned as descripbed above.

Considering the compexity of the features and the number of instances for each dataset, Keras tuner search was utilized with the hyperband subclass to find optimal hyperparameters.  Hyperband was utilized due to its ability to quickly determine optimal parameters by early stopping and trying several combination of parameters.  After the tuner completed, the best set of hyperparameters, evaluated based on validation accuracy, was selected and used to train the same model out to 100 epochs.  After 100 epochs, the best epoch was determined once the model started to overfit the data.  THe model was retrained again, and stopped when the best epoch was achieved.

The only feature not utilized in any of the models was the 'EIN'.  Initial attempts to use this feature were halted due to the memory resources allotted from Google Colab.  Even when this feature was significantly binned, the model exceeded the allowed memory upon encoding.

#### **Model Building**
The Keras Tuner hyperband search was set to choose the activation function, number of hidden layers, and number of neurons for each layer.  The last layer had only 1 neuron with a sigmoid activation function to act as the final step in the successful / non successful classification.
##### *Activation Function Choice*
Activation functions: relu, tanh, sigmoid were chooses banes on their utility on classification models
##### *Hidden Layers*
The tuner search ultiized between 1 - 6 hidden layers.  Due to time and computation restraints, possible hidden layers were capped at 6.
##### *Neurons per Layer*
Due to the overall shape of the datasets for  - number of features in the encoded dataset (between 41 and 450 depending on the binning cut off)
How many neurons, layers, and activation functions did you select for your neural network model, and why?

Were you able to achieve the target model performance?

What steps did you take in your attempts to increase model performance?

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


Further work:
What variable(s) are the target(s) for your model?