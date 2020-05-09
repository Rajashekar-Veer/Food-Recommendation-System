## The Analyser

The overview of the project is to create a clustered of ingredients and train a classifier to predict the cuisine type of a new food. 
The dataset used is Yummly.com.

------
Packages Used
---
- json
- pandas
- numpy
- sklearn
- nltk

## How did you turn your text into features and why?
In the analyser implementation, I have used TfidfVectorizer to convert the ingredients into features.
TdidfVectorizer gives the scoring and ranking a words relevance in all document's. Machine learning modelling 
cannot work with raw text directly therefore we convert the text into vectors of numbers. 

## What classifiers/clustering methods did you choose and why?
I have used Random Forest Classifier for modelling. I worked on other models like super vector machine, logistic regression and found random forest classifier provided the best accuracy for the given dataset.

## What N did you choose and why?
I have used default n_estimators that is 100, i found best results with this n_estimator.

## dataRead()
This method takes in one argument that is inputFile name. In this method we read the data file using pandas read_json funtion which stores the data as a dataframe.
This method returns the dataframe. The dataframe has a shape of (39774,3).

## dataProcessing()
This method takes in two arguments that is dataframe and the list of input ingredients for which we need to predict cuisine.
In this method we pre-process the data for the modelling convenience.
•	Converted the ingredients column of the data frame into a data list and appended the input ingredients to the 0th index of the data list.
•	Converted the data list to lower case.
•	Removed any numbers in the ingredient data.
•	Applied nltk word tokenizer on the data list.
This function returns the complete data list of ingredients.

## vectorization():
This method takes in 1 argument which is the data list of ingredients. The ingredients list is vectorized using TfidfVectorizer. The 0th row of the vectorized matrix is the input ingredient data for which we need to predict, the remaining matrix is the actual data on which we apply our classifier model.
We split the 0th row and the remaining matrix data and return it.

## randomForestModel():
I have Random forest classifier algorithm for modelling. Random forest classifier is a set of decision trees from a randomly selected data subset from the training set. The model aggregates different decision trees and finds the best class of the test data.
The arguments for this dataset is the vectorized matrix, cuisine list of the ingredients and the input vector for which we need to predict.
The following steps are followed in the method:
-	We find encode the cusine data into using labelEncoder from processing library of sklearn package for better modelling result.
-	Split our dataset into train and test using train_test_split from sklearn package.
-	Create a random forest model with default values.
-	Fit the train data.
-	Predict the test data based on the model
-	Accuracy is found to 73% using this basic model
-	Finally we predict the cuisine on the user input ingredients.

## closestRecipe():

This method is used to find the closest cuisine type.
In this method we find the cosine similarity score between the user input ingredient matrix and the actual data input matrix. We store the scores into the data frame. From the dataframe we pick the top 10 scores and print the corresponding cuisine type.

## Tests:
I have tested the code with many input ingredients.
input = ['paprika', 'banana','rice krispies','plain flour', 'ground pepper', 'salt', 'tomatoes']

The output is in the below format.
```
Model Accuracy : 72.75898768122015
Cuisine:  ['indian']

Closest 10 Recipes
          id    Scores
28497   8498  0.415908
18138   9944  0.405546
3180   49233  0.352461
32001  13474  0.352267
37987  30333  0.348387
13011  30881  0.333593
20750  18184  0.332843
22917  47160  0.324110
19220  40583  0.318538
12759  44122  0.303510
```
- In the above output we display the accuracy of the model.
- Predict the cuisine type for the input passed.
- Display the closest 10 recipes



