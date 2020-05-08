import json
import numpy
import operator
import pandas
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def data_load():

    # Function helps us load data from the json file into an array

    file_path = 'Data/yummly.json'

    # Open the json file
    j_f = open(file_path)
    # Load the file
    data = json.load(j_f)

    df = pandas.DataFrame(data)

    return df


def data_process(data_frame):
    # Separate out ids and ingredients from each other
    recipe_ids = data_frame['id']
    ingredients = data_frame['ingredients']
    ingredients = ingredients.apply(' '.join)

    # Create a tagged documents data structure for doc2vec
    tagged_recipes = [TaggedDocument(words=word_tokenize(ingredients.lower()),
                                     tags=[str(recipe_ids[i])])
                      for i, ingredients in enumerate(ingredients)]

    return tagged_recipes


def build_model(data):
    # Create variables for model generation
    model = Doc2Vec(vector_size=300, window=10, alpha=0.025,
                    min_alpha=0.025, min_count=5, dm=1, workers=11)
    # Create voabulary for model
    model.build_vocab(data)

    # Loop through and train model
    for epoch in range(20):
        print("Epoch iteration: " + str(epoch))
        model.train(data, total_examples=model.corpus_count, epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("Model/NLP.model")

    return "Model/NLP.model"


def build_predictions(model_file, ingredients_list):
    # Load in generated file
    model = Doc2Vec.load(model_file)

    # Create new array from ingredients list, we do this do remove any spaces
    #    between ingredients with two words
    ingredients_str = " ".join(ingredients_list)
    user_ingredients = ingredients_str.split(" ")

    predictions = model.docvecs.most_similar(
        positive=[model.infer_vector(user_ingredients)], topn=10)

    return predictions


def output(predictions, data_frame):
    # Separate ids out from data_frame
    recipe_ids = data_frame['id']

    # List of predicted ids
    predicted_ids = [int(_id[0]) for _id in predictions]

    # Contains list of counted cuisine types
    cuisine_dictionary = {}
    # Loop through data_frame and gather cuisine types
    for _id in predicted_ids:
        for index, recipe in data_frame.iterrows():
            if (recipe['id'] == _id):
                if (recipe['cuisine'] not in cuisine_dictionary.keys()):
                    cuisine_dictionary[recipe['cuisine']] = 0

                cuisine_dictionary[recipe['cuisine']] += 1

    # Find cuisine type percentage among all cuisine types found
    total_count = 10
    max_cuisine_type = ""
    max_cuisine_count = 0
    # For cuisine types with same count as max_cuisine_type
    similar_cuisine = []
    for cuisine in cuisine_dictionary:
        n = max(max_cuisine_count, cuisine_dictionary[cuisine])
        if n != max_cuisine_count:
            max_cuisine_count = n
            max_cuisine_type = cuisine
            similar_cuisine = []
        elif cuisine_dictionary[cuisine] == max_cuisine_count:
            similar_cuisine.append(cuisine)

    if len(similar_cuisine) != 0:
        # Output
        print("Cuisine: " + max_cuisine_type + " (" +
              str(round(float(max_cuisine_count/total_count), 2)) + ")", end='')

        for cuisine_type in similar_cuisine:
            print(", " + cuisine_type + " (" +
                  str(round(float(cuisine_dictionary[cuisine_type]/total_count), 2)) + ")", end='')
        print("")
    else:
        # Output
        print("Cuisine: " + max_cuisine_type + " (" +
              str(round(float(max_cuisine_count/total_count), 2)) + ")")

    print("Closest 5 Recipes: ", end='')

    # Predictions list for printing output
    predictions_list = [
        (str(_id[0]) + " (" + str(round(_id[1], 2)) + ")") for _id in predictions[:5]]

    print(', '.join(predictions_list))
