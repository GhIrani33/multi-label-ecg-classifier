import numpy as np
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import tensorflow as tf

path = 'dataset/ptb-xl/1.0.3/'
ptbxl_database_path = path + 'ptbxl_database.csv'
scp_statements_path = path + 'scp_statements.csv'
output_path = 'preprocessed_dataset.csv'

def scale_age(age):
    if age > 89:
        age -= 300
    return age

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 500:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def preprocess_data(data_path, scp_statements_path, output_path, sampling_rate, path):
    data = pd.read_csv(data_path, index_col='ecg_id')
    data.scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))

    data['age'] = data['age'].apply(scale_age)
    scaler = MinMaxScaler()
    data['age'] = scaler.fit_transform(data[['age']])

    data['sex'] = data['sex'].map({0: 'Female', 1: 'Male'})
    sex_encoded = pd.get_dummies(data['sex'], prefix='sex')
    data = pd.concat([data, sex_encoded], axis=1)
    data = data.drop('sex', axis=1)

    boolean_columns = ['sex_Male', 'sex_Female']
    for col in boolean_columns:
        data[col] = data[col].astype(int)

    X_ecg = load_raw_data(data, sampling_rate, path)

    agg_df = pd.read_csv(scp_statements_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def map_diagnostic_superclass(scp_codes, agg_df):
        superclass_mapping = {'NORM': 0, 'MI': 0, 'STTC': 0, 'CD': 0, 'HYP': 0}
        for code in scp_codes.keys():
            if code in agg_df.index:
                superclass = agg_df.loc[code].diagnostic_class
                if superclass in superclass_mapping:
                    superclass_mapping[superclass] = 1
        return superclass_mapping

    diagnostic_labels = data['scp_codes'].apply(lambda x: map_diagnostic_superclass(x, agg_df))
    diagnostic_df = pd.DataFrame(diagnostic_labels.tolist(), index=data.index)

    data = pd.concat([data, diagnostic_df], axis=1)

    selected_columns = ['age'] + boolean_columns
    X_features = data[selected_columns]
    Y = diagnostic_df

    data_preprocessed = pd.concat([X_features, Y], axis=1)
    data_preprocessed.to_csv(output_path, index=True)

    return X_ecg, X_features, data, Y



