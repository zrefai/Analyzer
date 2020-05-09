# Analyzer

Recipe analyzer and predictor

## Setup

Run these commands for proper setup

```
pipenv install
pipenv shell
```

## Running the program

You can run the program regularly with

```
python src/main.py --ingredient cinnamon --ingredient apple --ingredient salt --ingredient sugar
```

The --ingredient tag needs to be followed by an ingredient.

NOTE: NO NEED TO QUOTE INGREDIENTS THAT CONTAIN MORE THAN ONE WORD

### Other forms of running the program

Automatically, the program does not generate a new model of Doc2Vec. The command above skips the process of generating a new model and just queries the previously generated model during development. To run the entire program, and generate a new model on the spot, run this:

```
python src/main.py --ingredient cinnamon --ingredient apple --ingredient salt --ingredient sugar --model 1
```

The --model tag is automatically set to 0. Setting it to 1 will tell the program to generate a new model.

## Summary

This is a recipe predictor based on the ingredients given from the user. The program takes data from yummly.json (Provided by Yummly.com) and processes it into a training dataset for a Doc2Vec model. The model is fed the training data and is able to check all recipes in yummly.json for similarites in the given user ingredients. The output generated includes the most similar cuisine type found among the most similar cuisines, and the top 5 most similar cuisines to the user given ingredients.

## Processing the text

Doc2Vec is a simple and straightforward library. At first, I was struggling using BERT, as it was not so clear, but Doc2Vec is practically plug and play. The data needs to be tokenized and tagged. Naturally I used the recipe ID's for tagging respective recipes. Each ingredients list is pretty much in base form except for the fact that some ingredients contain more than 1 word in them. To account for this, I just join all of the ingredients with a " " and then split based on another " ". This separates out all of the words individually. No further processing of ingredients text was necessary, and each ingredients list was tagged with its own recipe ID to create a tagged_data data structure.

## Classifiers/Clustering Methods

Doc2Vec makes cluserting unnecessary. All I really needed to do after producing the tagged data was feed it to the Doc2Vec model. Over 20 epochs, the data was used to train the model, while decresing learning rate and applying no decay to it as well. The model generated used a distributed memory algorithm. This algorithm preserves word order in the ingredients list which helps with accuracy; ingredients can sometimes contain more than one word, so keeping the order helps ensure that prediction is more accurate on user given ingredients with more than one word.

## N Predictions

The program goes through the top 10 closest recipes in similiarity to the ingredients given by the user. Out of the 10 recipes, the program will count each time a specific cuisine type has come up in predictions. The maximum occuring cuisine type and its occurances is used and divided by 10 to achieve the predicted cuisine type during output. For example, if cuisine type 'italian' appeared 5 times among the 10 cuisine types (and it is also the maximum) then the output would be 'Cuisine: italian (.5).' If more than one cuisine type happen to have the max count among all other cuisine types, then those n number of cuisine types will be printed in the output. For example, if 'italian' and 'mexiacan' both appeared 2 times and 2 happens to be the maximum cuisine type count among the rest of the cuisine types, then the output will be 'Cuisine: italian (.2), mexican (.2)'. The 10 predictions helps the program get a prediction on cuisine type, while the top 5 predictions of the 10 are printed in similarity order. The number in paranthesis next to each recipe or cuisine type is a similarity score given by the model.

## Functions

### data_load()

data_load takes the yummly.json file and load it into a pandas data frame. The data frame is returned

### data_process(data_frame)

data_process takes the data frame produced by data_load a separate out portions of it for tagging. The recipe ids and recipe ingredients are put into different arrays. These arrays are then used to create a TaggedDocument data structure that is needed for Doc2Vec. Each ingredients list in a recipe is tagged with its own recipe id. The tagged document data structure is returned.

### build_model(tagged_data)

buil_model takes the tagged data produced by data_process and use it to train a Doc2Vec model. The model is created and uses the distributed memory algorithm for training. At each epoch, over 20 epochs, the data is used to train the model. The model is generated and then saved. The model name is returned.

### build_prediction(model_file, ingredients_list)

build_prediction will take the model generated and the ingredients list, structure the ingredients list given by the user, feed it to the model for questioning, and the model will output a set of 10 predictions that closely relate to the ingredients list. The ingredients list is structured by tokenizing each ingredient into words in the order they were recieved, which is then given to the model. The model uses model.infer_vector to create a list of 10 vectors that closely relate to the user given ingredients list. A recipe id and its similarity score to the ingredients list is given in the output.

## output(predictions, data_frame)

output will take the predictions and the dataframe generated from data_load to create a formatted output. The cuisine types of each predicted recipe is used to genereate a prediction on the type of cuisine the user given ingredients follow. The data from predictions and the dataframe output a predicted cuisine type, the closest 5 recipes, and their similarity scores.

## Tests

-  test_data_load: tests if data was loaded into data frame properply
-  test_data_process: tests if the data was processes correctly as a tagged document data structure
-  test_build_model: tests if the model was generated and if the file exists after saving
-  test_build_predictions: tests if the predictions was generated properly
