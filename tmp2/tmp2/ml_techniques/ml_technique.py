import numpy as np
import pandas as pd
import pickle
import os

pd.options.mode.chained_assignment = None

class MLTechnique:
    """
    Represents a Machine Technique and states that
    are maintained by the instance

    Instance Members:
    - datafile: file path that points to the Excel File
        that contains the data to be used for ML.
        The only file type supported is xlsx and the data
        must be in data buckets - one sheet per bucket -
        labelled Set<number>
    - model: sklearn model
    - features: list of training params to be used for cross validation
    - pickle_file: to store this instance as a pickle object in the system
        for faster I/O
    - actual: list of actual prediction result corresponding to the datafile
    - predicted: list of prediction result corresponding to the datafile
    """

    def __init__(self, name, model, features, nlp=None, data_frame=None, data_file='', ml_code='', pickle_file=''):
        print("Initialise " + name)
        # print("Pickle file path: " + pickle_file)
        self.name = name
        self.data_file = data_file
        self.model = model
        self.features = features
        self.pickle_file = pickle_file
        self.actual = []
        self.predicted = []
        self.ml_code = ml_code
        self.data_frame = data_frame
        self.nlp = nlp
        # Initial test size is 0, and will be updated after
        # the prediction() function is called

    def cross_validate(self, cat_col, cat_map, buckets=5, forced=False):
        """
        Predict using categories used in cat_sheet and mapping
        defined in cat_map
        :param cat_col: Name of column that contains the categories
        :type cat_col: str
        :param cat_map: Dictionary mapping category to integer values
        :type cat_map: dict
        :param buckets: Number of buckets used
        :type buckets: int
        :param forced: Forced refreshing cross-validation
        :type forced: boolean
        :return: Dictionary containing an array of 'true' values and
            an array of 'predict'-ed values
        :rtype: dict
        """

        # Cross validation is done before, return stored results
        if not forced:
            if len(self.actual) > 0 and len(self.predicted) > 0:
                print("Return saved cross-validation")
                # Return a dictionary with cross validation results
                return {
                    'total_size': len(self.predicted),
                    'true_array': self.actual,
                    'predict_array': self.predicted,
                    'categories': list(cat_map.keys())
                }

        if self.data_file == '':
            raise AssertionError

        # Parsing data from excel file
        file = pd.ExcelFile(self.data_file)
        # inputs_ as an array of bucket inputs
        inputs_ = []
        # expects_ as an array of bucket expected NPS scores
        expects_ = []
        # Populate inputs_ and expects_
        for i in range(0, buckets):
            data_frame = file.parse('Set' + str(i + 1), index_col='jobid')
            print("Data frame columns: " + str(data_frame.columns))
            # Set training params
            print("Features selected (Cross-val): " + str(self.features))

            if 'bag_vector' in self.features:
                def string_to_list(row):
                    row['bag_vector'] = [int(s) for s in row['bag_vector'][1:-1].split(', ')]
                    return row[['bag_vector']]
                
                # converting from string to list and flattening
                slice_df = data_frame[self.features]
                slice_df['bag_vector'] = slice_df['bag_vector'].astype(list)
                slice_df[['bag_vector']] = slice_df.apply(string_to_list, axis=1)

                input_var = list(slice_df.as_matrix())

                for j in range(len(input_var)):
                    input_var[j] = [y for x in input_var[j] for y in (x if isinstance(x, list) else (x,))]
            else:
                input_var = data_frame[self.features].as_matrix()

            # nps_rating is the target variable
            expect = data_frame[cat_col].map(cat_map)
            expect = expect.as_matrix()
            # Add to the current list of domain and range
            inputs_.append(input_var)
            expects_.append(expect)

        # -- Cross Validation over buckets --
        test_expect = []
        test_prediction = []

        for i in range(0, buckets):
            # Training over all but one buckets
            train_input = inputs_[i]
            train_expect = expects_[i]
            for j in range(1, buckets - 1):
                index = (i + j) % buckets
                train_input = np.concatenate([train_input, inputs_[index]])
                train_expect = np.concatenate([train_expect, expects_[index]])

            # Train the model
            print("Train Model: " + self.name + " " + str(i))
            self.model.fit(train_input, np.ravel(train_expect))
            # Prediction on remaining bucket
            j = (i + buckets - 1) % buckets
            # Get the corresponding input set
            test_input = inputs_[j]
            # Flatten to make the array of n-arrays to be an array of n elements
            test_expect = np.concatenate([test_expect, expects_[j].flatten()])
            # predict the chosen set of input and put the result in test_prediction
            predictions = self.model.predict(test_input)
            test_prediction = np.concatenate([test_prediction, predictions])

        # Saved actual and predicted array
        self.actual = test_expect
        self.predicted = test_prediction
        self.update_or_create_pickle()

        file.close()

        # Return a dictionary with cross validation results
        return {
            'total_size': len(test_prediction),
            'true_array': test_expect,
            'predict_array': test_prediction,
            'categories': list(cat_map.keys())
        }

    def cross_validation_forced(self, cat_col, cat_map, buckets=5):
        return self.cross_validate(cat_col=cat_col, cat_map=cat_map, buckets=buckets, forced=True)

    def get_name(self):
        return self.name

    def train(self, cat_col, cat_map, forced=False):
        """
        Train the instance using categories used in cat_sheet and mapping
        defined in cat_map
        :param cat_col: Name of column that contains the categories
        :type cat_col: str
        :param cat_map: Dictionary mapping category to integer values
        :type cat_map: dict
        :param features: User-defined training params, if is None then self.params is used
        :type features: str[]
        """

        pred_ml = self.read_instance_pickle(self.pickle_file)
        if pred_ml is not None and not forced:
            print("Return saved Prediction ML technique: " + self.name)
            return

        print("Features: " + str(self.features))
        # Parsing data from excel file
        # file = pd.ExcelFile(self.data_file)
        # data_frame = file.parse('Balance', index_col='jobid')
        # file.close()
        if self.data_file is None:
            raise AssertionError

        data_frame = self.data_frame
        print("Training size (rows, columns): " + str(data_frame.shape))
        # converting from string to list and flattening
        if 'bag_vector' in self.features:
            def string_to_list(row):
                row['bag_vector'] = [int(s) for s in row['bag_vector'][1:-1].split(', ')]
                return row[['bag_vector']]
            
            slice_df = data_frame[self.features]
            # slice_df['bag_vector'] = slice_df['bag_vector'].astype(list)
            # slice_df[['bag_vector']] = slice_df.apply(string_to_list, axis=1)
        
            input_ = list(slice_df.as_matrix())

            for i in range(len(input_)):
                input_[i] = [y for x in input_[i] for y in (x if isinstance(x, list) else (x,))]
        else:
            input_ = data_frame[self.features].as_matrix()

        print("No of features (Training): " + str(len(input_[0])))
        print("Training " + self.name + " ...")
        expect_ = (data_frame[cat_col].map(cat_map)).as_matrix()
        self.model.fit(input_, expect_)
        print("Training " + self.name + " COMPLETE.")
        self.update_or_create_pickle()
        return

    def predict_single(self, file_path):
        """
        Predict using categories used in cat_sheet and mapping
        defined in cat_map
        :param file_path: file path that points to the Excel File
        that contains the data to be used for ML
        :type file_path: str
        :return: Predicted value based on the current training
        :rtype: int
        """
        file = pd.ExcelFile(file_path)
        data_frame = file.parse('Balance', index_col='jobid')
        file.close()

        if 'bag_vector' in self.features:
            def string_to_list(row):
                row['bag_vector'] = [int(s) for s in row['bag_vector'][1:-1].split(', ')]
                return row[['bag_vector']]
            
            # converting from string to list and flattening
            slice_df = data_frame[self.features]
            slice_df['bag_vector'] = slice_df['bag_vector'].astype(list)
            slice_df[['bag_vector']] = slice_df.apply(string_to_list, axis=1)
        
            input_ = list(slice_df.as_matrix())

            for i in range(len(input_)):
                input_[i] = [y for x in input_[i] for y in (x if isinstance(x, list) else (x,))]
        else:
            input_ = data_frame[self.features].as_matrix()

        result = self.model.predict(input_)
        return result


    # DUP
    def update_or_create_pickle(self):
        if self.pickle_file:
            with open(self.pickle_file, "wb") as file:
                pickle.dump(self, file)

    # DUP
    def read_instance_pickle(self, path):
        if os.path.isfile(path):
            with open(path, "rb") as file:
                try:
                    ml = pickle.load(file)
                    # print("Load saved instance.")
                    return ml
                except (OSError, IOError):
                    pass
        return None