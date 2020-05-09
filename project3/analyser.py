import json
import sys
import nltk
import pandas as pd
import re
import string
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

inputFile = "yummly.json"
# input = ['paprika', 'banana','rice krispies','plain flour', 'ground pepper', 'salt', 'tomatoes']
# input= ['coriander powder','ground turmeric','red pepper flakes','japanese eggplants','plums','grated parmesan cheese','fresh parsley','tomatoes with juice']
# input = ['paprika','banana','rice krispies']

#Reading the data
def dataRead(inputFile):
    dataFrame = pd.read_json(inputFile)
    return dataFrame

#Method to preprocess data and tokenize data
def dataProcessing(dataFrame, input):
    exclude = set(string.punctuation)
    stop_words = set(stopwords.words("english"))
    ingredientList = []
    dataList = []
    for i in dataFrame['ingredients']:
        i = " ".join(i)
        dataList.append(i)
    input = " ".join(input)
    dataList.insert(0, input)

    for data in dataList:
        data = data.lower()
        data = re.sub(r"(\d)", "", data)
        token = nltk.word_tokenize(data)
        mytokens = " ".join([word for word in token if word not in exclude and word not in stop_words])
        ingredientList.append(mytokens)
    return ingredientList

# Vectorizing the data
def vectorization(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vector = vectorizer.fit_transform(data)
    inputMatrix = vector[0]
    dataMatrix = vector[1:]
    print("Vectorization Completed")
    return inputMatrix, dataMatrix

#Random Forest Model for prediction
def randomForestModel(inputMatrix, dataMatrix, dataFrame):
    LabelEncoder = preprocessing.LabelEncoder()
    LabelEncoder.fit(dataFrame['cuisine'])
    X = dataMatrix
    Y = LabelEncoder.transform(dataFrame['cuisine'])

    x_train, x_test, y_train, y_test = train_test_split(dataMatrix, Y, test_size=0.3)
    model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Model Accuracy :",accuracy_score(y_test, y_pred) * 100)

    inputPredict = model.predict(inputMatrix)
    cuisine = LabelEncoder.inverse_transform(inputPredict)
    print("Cuisine: ",cuisine)

#Method to print the closest 10 Recipes
def closestRecipe(inputMatrix, dataMatrix, dataFrame):
    scores = cosine_similarity(inputMatrix,dataMatrix).transpose()
    dataFrame['Scores'] = scores
    closeRecipe = dataFrame[['id','Scores']].nlargest(10, ['Scores'])
    print("Closest 10 Recipes \n",closeRecipe)

if __name__ == '__main__':
    input = []
    params = sys.argv
    print(params)
    for i in range(len(params)):
        if params[i] == '--ingredient':
            input.append(params[i+1])
    print(input)
    dataFrame = dataRead(inputFile)
    data = dataProcessing(dataFrame, input)
    inputMatrix, dataMatrix = vectorization(data)
    randomForestModel(inputMatrix, dataMatrix, dataFrame)
    closestRecipe(inputMatrix, dataMatrix, dataFrame)
