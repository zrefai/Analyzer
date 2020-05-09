from src import functions
import unittest
import sys
import os
import json
import pandas


class Test_Fetch(unittest.TestCase):

    def test_data_load(self):

        df1 = functions.data_load()

        if df1.empty:
            assert False
        else:
            assert True

    def test_data_process(self):

        df = functions.data_load()
        tagged_data = functions.data_process(df)
        assert tagged_data[0] == (['romaine', 'lettuce', 'black', 'olives', 'grape', 'tomatoes', 'garlic', 'pepper',
                                   'purple', 'onion', 'seasoning', 'garbanzo', 'beans', 'feta', 'cheese', 'crumbles'], ['10259'])

    def test_build_model(self):
        df = functions.data_load()
        tagged_data = functions.data_process(df)
        NLP_model = functions.build_model(tagged_data)

        assert os.path.exists(NLP_model)

    def test_build_predictions(self):
        df = functions.data_load()

        predictions = functions.build_predictions(
            'Models/NLP.model', ['banana'])

        assert predictions
        assert len(predictions) == 10
