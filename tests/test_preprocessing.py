import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

# Suppress TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

class TestPreprocessing(unittest.TestCase):

    def test_scale_age(self):
        from preprocessing import scale_age
        # Test case where age is greater than 89
        self.assertEqual(scale_age(90), -210)
        # Test case where age is less than or equal to 89
        self.assertEqual(scale_age(50), 50)

    @patch('preprocessing.wfdb.rdsamp')
    def test_load_raw_data(self, mock_rdsamp):
        from preprocessing import load_raw_data
        # Mocking wfdb.rdsamp to return dummy ECG data
        mock_rdsamp.return_value = (np.random.rand(1000, 12), {})  # 1000 timesteps, 12 leads

        # Create a mock DataFrame for testing the raw data loading function
        df = pd.DataFrame({
            'filename_lr': ['00001_lr', '00002_lr'],
            'filename_hr': ['00001_hr', '00002_hr']
        })
        path = 'path_to_test_data/'  # Test data path

        # Simulate loading data with sampling rate of 500 Hz
        result_500 = load_raw_data(df, sampling_rate=500, path=path)
        
        # Assertions
        self.assertIsInstance(result_500, np.ndarray)
        self.assertEqual(result_500.shape, (2, 1000, 12))  # Ensure shape is (num_records, timesteps, 12 leads)

    @patch('preprocessing.wfdb.rdsamp')
    @patch('preprocessing.pd.read_csv')
    def test_preprocess_data(self, mock_read_csv, mock_rdsamp):
        from preprocessing import preprocess_data
        # Mocking pd.read_csv to return dummy DataFrames
        # First call returns ptbxl_database.csv, second call returns scp_statements.csv
        mock_read_csv.side_effect = [
            pd.DataFrame({
                'ecg_id': [1, 2, 3, 4, 5],
                'scp_codes': [
                    "{'NORM': 1}",
                    "{'MI': 1}",
                    "{'STTC': 1}",
                    "{'CD': 1}",
                    "{'HYP': 1}"
                ],
                'age': [60, 70, 80, 85, 90],
                'sex': [1, 0, 1, 0, 1],  # Alternating Male and Female
                'strat_fold': [1, 2, 3, 4, 5],
                'filename_lr': ['00001_lr', '00002_lr', '00003_lr', '00004_lr', '00005_lr'],
                'filename_hr': ['00001_hr', '00002_hr', '00003_hr', '00004_hr', '00005_hr']
            }),
            pd.DataFrame({
                'diagnostic_class': ['NORM', 'MI', 'STTC', 'CD', 'HYP']
            }, index=['NORM', 'MI', 'STTC', 'CD', 'HYP'])  # Correct indexing
        ]

        # Mocking wfdb.rdsamp to return dummy ECG data
        mock_rdsamp.return_value = (np.random.rand(1000, 12), {})  # 1000 timesteps, 12 leads

        # Simulate paths to dataset and scp_statements.csv for testing
        data_path = 'path_to_test_data/ptbxl_database.csv'
        scp_statements_path = 'path_to_test_data/scp_statements.csv'
        output_path = 'output_test_data.csv'
        sampling_rate = 500
        path = 'path_to_test_data/'

        # Preprocess the data
        X_ecg, X_features, data, Y = preprocess_data(
            data_path, scp_statements_path, output_path, sampling_rate, path
        )

        # Assertions to verify the shapes and contents
        self.assertIsInstance(X_ecg, np.ndarray)
        self.assertIsInstance(X_features, pd.DataFrame)
        self.assertEqual(X_ecg.shape, (5, 1000, 12))  # Ensure shape is (num_records, timesteps, 12 leads)
        self.assertTrue('age' in X_features.columns)  # 'age' column exists
        self.assertTrue('sex_Male' in X_features.columns)  # 'sex_Male' column exists
        self.assertTrue('sex_Female' in X_features.columns)  # 'sex_Female' column exists

        # Verify that 'sex_Male' and 'sex_Female' are correctly encoded
        expected_sex_male = pd.Series([1, 0, 1, 0, 1], name='sex_Male')
        expected_sex_female = pd.Series([0, 1, 0, 1, 0], name='sex_Female')
        pd.testing.assert_series_equal(X_features['sex_Male'], expected_sex_male)
        pd.testing.assert_series_equal(X_features['sex_Female'], expected_sex_female)

        # Additional checks for scaled 'age'
        self.assertTrue(X_features['age'].min() >= 0 and X_features['age'].max() <= 1)

        # Check that Y has the correct columns
        expected_Y_columns = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        for col in expected_Y_columns:
            self.assertIn(col, Y.columns)
        
        # Check sample values in Y
        self.assertEqual(Y.loc['NORM', 'NORM'], 1)
        self.assertEqual(Y.loc['MI', 'MI'], 1)
        self.assertEqual(Y.loc['STTC', 'STTC'], 1)
        self.assertEqual(Y.loc['CD', 'CD'], 1)
        self.assertEqual(Y.loc['HYP', 'HYP'], 1)
        
        # Ensure that other columns are 0 where appropriate
        for idx, row in Y.iterrows():
            for col in expected_Y_columns:
                if row[col] != 1 and col in ['NORM', 'MI', 'STTC', 'CD', 'HYP']:
                    self.assertEqual(row[col], 0)

if __name__ == '__main__':
    unittest.main()
