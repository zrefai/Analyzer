import argparse
import functions
import time


def main(args):

    # Load in data from file
    df = functions.data_load()

    MODEL = 0
    # Check if user wanted
    if args.model is not None:
        MODEL = int(args.model[0])

    # If you want to generate a new model
    if MODEL:
        start = time.time()
        # Process data from data_frame
        tagged_recipes = functions.data_process(df)
        end = time.time()

        print("Data process execution: " +
              str(round((end - start), 2)) + " seconds")
        print("Model training and generation started for 20 epochs")

        start = time.time()
        # Build the model for predictions of recipes
        model = functions.build_model(tagged_recipes)
        end = time.time()

        print("Model build execution: " +
              str(round((end-start), 2)) + " seconds")
        print("")

    # Convert list of list to flat list
    ingredients_list = [
        ingredient for sublist in args.ingredient for ingredient in sublist]

    print(ingredients_list)
    # Generate predictions
    predictions = functions.build_predictions(
        'Models/NLP.model', ingredients_list)

    # Build output
    functions.output(predictions, df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ingredient", action='append', nargs='+', type=str, required=True,
                        help="Ingredients to find recipes with")
    parser.add_argument("--model", action='append', type=str, required=False,
                        help="Whether to generate new model")
    args = parser.parse_args()

    main(args)
