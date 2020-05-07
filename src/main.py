import argparse
import functions
import time


def main(args):

    # Load in data from file
    start = time.time()
    df = functions.data_load()
    end = time.time()

    print("Data load execution: " + str(end - start))

    # Process data into format acceptable by BERT
    start = time.time()
    padded_data, masked_data = functions.data_process(df)
    end = time.time()

    print("Data process execution: " + str(end - start))

    # Now run through BERT model
    start = time.time()
    features = functions.model_run(padded_data, masked_data)
    end = time.time()

    print("Model run execution: " + str(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ingredient", action='append', nargs='+', type=str, required=True,
                        help="Ingredients to find recipes with")
    args = parser.parse_args()

    main(args)
