# Analyzer

Recipe analyzer and predictor

## Summary

This is a recipe predictor based on the ingredients given from the user. The program takes data from yummly.json (Provided by Yummly.com) and processes it into a training dataset for a Doc2Vec model. The model is fed the training data and is able to check all recipes in yummly.json for similarites in the given user ingredients. The output generated includes the most similar cuisine type found among the most similar cuisines, and the top 5 most similar cuisines to the user given ingredients.

## Processing the text

Doc2Vec is a simple and straightforward library. At first, I was struggling using BERT, as it was not so clear, but Doc2Vec is practically plug and play. The data needs to be tokenized and tagged. Naturally I used the recipe ID's for tagging respective recipes. Each ingredients list is pretty much in base form except for the fact that some ingredients contain more than 1 word in them. To account for this, I just join all of the ingredients with a " " and then split based on another " ". This separates out all of the words individually. No further processing of ingredients text was necessary, and each ingredients list was tagged with its own recipe ID to create a tagged_data data structure.

## Classifiers/Clustering Methods

Doc2Vec makes cluserting unnecessary. All I really needed to do after producing the tagged data was feed it to the Doc2Vec model. Over 20 epochs, the data was used to train the model, while decresing learning rate and applying no decay to it as well. The model generated used a distributed memory algorithm. This algorithm preserves word order in the ingredients list which helps with accuracy; ingredients can sometimes contain more than one word, so keeping the order helps ensure that prediction is more accurate on user given ingredients with more than one word.
