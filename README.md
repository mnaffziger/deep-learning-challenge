# deep-learning-challenge
Module 21 Challenge: Utilizing neural networks with TensorFlow and Keras on Google Colab

---
## Abstract
*   Initial attempts lead to a training accuracy no greater that 73%.
*   Additional data pulled from the original dataset was required to increase the accuracy greater than 75%. 
*   Training accuracy from models including the 'NAMES' category included were between 78% - 80%.
*   There was a trend towards overfitting as the number of epochs increased
---
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
## Data Preprocessing

### **Model Initialization**
Every model utilized the 'Is Successful' feature as its target.  As per the client's request, this predictive model was to aid in the selection of potentially successful applicants. Due to the complexity of the features, some of the categorical features were bined before encoding.  For all models the following features were binned with the corresponding cut off values:
*   Application Type:  Types less than 500 application were classified as 'other'
*   Classifiacation: Organization classes that accounted for less than 1000 applicants were reclassified as 'other'
*   Names:  If the applicant applied for less than 5 funding requests, they were classified as 'other'

Two models were utilized for training purposes.  The major difference between the two models is the utilization of the 'Names' feature.  
*   Model 1 did not contain the 'Names' column in the cleaned and encoded dataset.  
*   Model 2 utilized the 'Names' feature.  Due to the variability of applicants, this feature was binned as descripbed above.

Considering the compexity of the features and the number of instances for each dataset, Keras tuner search was utilized with the hyperband subclass to find optimal hyperparameters.  Hyperband was utilized due to its ability to quickly determine optimal parameters by early stopping and trying several combination of parameters.  After the tuner completed, the best set of hyperparameters, evaluated based on validation accuracy, was selected and used to train the same model out to 100 epochs.  After 100 epochs, the best epoch was determined once the model started to overfit the data.  THe model was retrained again, and stopped when the best epoch was achieved.

The only feature not utilized in any of the models was the 'EIN'.  Initial attempts to use this feature were halted due to the memory resources allotted from Google Colab.  Even when this feature was significantly binned, the model exceeded the allowed memory upon encoding.

### **Model Building**
The Keras Tuner hyperband search was set to choose the activation function, number of hidden layers, and number of neurons for each layer.  The last layer had only 1 neuron with a sigmoid activation function to act as the final step in the successful / non successful classification.
##### *Activation Function Choice*
Activation functions: relu, tanh, sigmoid were chooses banes on their utility on classification models
##### *Neurons per Layer*
Due to the overall shape of the datasets for either model, the number of features in the encoded dataset varried between 41 and 450 depending on the binning cut off.  For this reason, the hyperband tuner was allowed to choose between 1 and the total number of features to determin how many neuron each layer contained.  This created a natural bias in the parameter count, since there was a greater variety of neurons in Model 2.  Hence the range of availible neuron to choose was an indirect test of the instance:feature ratio.  Since the datasets used for each model did not change the number of instances, the instance:feature ratio for the models was 850:1 (model 1) and 76:1 (Model 2).
##### *Hidden Layers*
The tuner search ultilized between 1 - 6 hidden layers.  Due to time and computation restraints, allowed hidden layers were capped at 6.  After the best hyperparameters were chosen, the model summary for the models counted 5,347 parameters (Model 1) and 175,617 parameters (Model 2)
---
## Results
#### Model 1: Dataset without 'Names' category
This model contained the least about of information contained within the dataset.  With the total number of features no more than 45, taking into face the binning cut off and encoding process, this model utilized around 5,347 parameters to provide a instance:feature ratio of 850:1.  This model could only achieve an accuracy of 73% with a loss around 0.531.  Despite several attempts at changing the cut off values for the binned features, activation function, and total possible neurons/layer, a training accuracy of 73% became the ceiling for this model. From the hyperband tuning process, the activation functions that typically resulted in the top 5 performing models (based on accuracy) were relu or tanh.
From the loss and accuracy curves from this model, the models quickly started overfitting- typically after 10 epochs. Additionally, eventhough the training data provided hyperparameters that could achieve ~74%, the validation/test data set fluctuated between 72.5% and 73.1%.
# INSERT MODEL 1 LOSS FIGURE

#### Model 2: Dataset with 'Names' category
The second model was roughly 10 times larger in feature count.  This routinly provided a model with around 175,000 parameters to configure.  As mentioned in the Model Building section, this created a bias in the number of available neurons a layer could have.  Since the tuner search was allowed to choose a number between 1 and ~440 it is more likely a layer would have around 200 neurons on average.  In contrast, Model 1's neuron count choice was bound between 1 and ~45, so more layers will have around 20 neurons per layer.  
The additional data, neurons, and resulting model parameters easily provided a training accuracy over 80%, with a loss no higher than 0.41.  Model 2 also started to overfit, however, overfitting did not occur until higher epochs, compared to model 1. Typically, when Model 2 was re-initiated, the best epoch was determined to be between 20-30.  Like Model 1, however, the validation/test accuracy fluctuated between ~79%.  The validation accuracy did not significantly improve with the number of epochs.
# INSERT MODEL 2 LOSS FIGURE

#### Model 3: Investigate instance:feature ratio
Due to the considerably large difference in model parameters between Model 1 and Model 2, a third model was studied.  Model 3 was built with the following parameters:
*   Input features: same number as Model 2, *i.e.* utilized the same dataset that included the 'Names' category with the same binning cut off value
*   Number of hidden layers: 6.  From every Keras hyperband parameter search, the best hyperparameters consisted of six hidden layers
*   Output layer: Contained 1 neuron with the sigmoid activation function
*   Neurons/layer: 100.  Since there were around 445 input features, 100 was viewed as a dimention reduction layer. 
*   Total parameters: 85,201 (based on 447 input features).  Model 3 consits of ~16x more parameters than Model 1, however it is around half the number of parameter as model 2.
Overall, Model 3 performed very similiar to Model 2.  Both Model 2 and 3 achieced at least 79% validation/test accuracy, with a loss no higher than 0.41.  This included a similar overfitting trend around epoch #30.  Interestingly, the validation accuracy kept fluctuating around 79% and did not significantly increase with as the epochs increased. This result suggests a couple of considerations:
*   Limiting the hidden layers to 6 does not necessarily require a large number of neurons per layer
*   Dimention reduction, as the number of neurons per layer is an area of interest to increase accuracy
# INSERT MODEL 3 LOSS FIGURE

#### All Models: Underrepresentative dataset
Considering all of the features (once encoded) all of the models utilized, it is possible that the dataset does not represent a balanced number of instances for the model to consider.  From the loss curve for each of the models, the validation loss starts with underfit values, when compared to the training data loss.  As the epochs progress, the loss improves for a couple epochs, before overfitting takes over, but there is consitant gap between the training and validation curve.  This result suggests the training data could be too small relative to the validation dataset.
# INSERT RETRAINED FIGURE MODEL 2
---
## Summary
In summary, the target accuracy of 75% was achieved when the dataset included the 'Names' category from the originally provided data from Alphabet Soup.  With the increased data size, a validation accuracy of ~79% was achieved with a validation loss around 0.44, when the training was stopped early before overfitting occured.  Based on the trainind plots for loss and accuracy, it appears that effeciency and accuracy/loss can be increased by fine tuning the total number of model parameters.  Lastly, before the model is signed-off on for the clients, the dataset original dataset should be evaluated in detail to address possible unrepresentative models.