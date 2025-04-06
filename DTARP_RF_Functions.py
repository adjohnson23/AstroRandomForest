import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

# Trains a random forest using the RandomForestClassifier from the scikit-learn library.
# PARAMETERS
# training_df: Training set data for the RF
# testing_df: Testing set data for the RF
# feature_list: A list of feature names for the random forest
# rf_save_path: The file path to save the random forest into.
# num_trees: Number of trees to grow. The default is 100.
# seed: What seed to generate a random forest from. The default is 42.
def train_rf(training_df, testing_df, feature_list, rf_save_path,
             num_trees=100, seed=42):
    if training_df is None:
        print("The training set doesn't exist!")
        return
    
    # Constrain RF training to given feature list
    rf_train = training_df[feature_list]
    rf_test = testing_df[feature_list]

    # Replace instances of infinity with NaN
    rf_train = rf_train.mask(np.isinf(rf_train), np.nan)
    rf_test = rf_test.mask(np.isinf(rf_train), np.nan)

    # Drop any rows with NaN in them
    rf_train = rf_train.dropna()
    rf_test = rf_test.dropna()

    # Split the X and Y 
    x_train = rf_train.drop(columns=['Class'])
    y_train = rf_train['Class']
    x_test = rf_test.drop(columns=['Class'])
    y_test = rf_test['Class']
    # print(x_train.head(5))
    # print(y_train.head(5))
    # print(f"Training dataset max values:")
    # for col in rf_train.columns:
    #     print(f"{col}: {rf_train[col].max()}")
    
    # print(f"Test dataset max values:")
    # for col in rf_train.columns:
    #     print(f"{col}: {rf_train[col].max()}")

    # Set up a random forest classifier
    # n_estimators: How many trees are grown?
    # random_state: Seed for tree growth
    # The random forest will only be set up if the save path doesn't exist already (don't accidentally overwrite existing rfs)
    if not os.path.exists(rf_save_path):
        rf_classifier = RandomForestClassifier(n_estimators=num_trees, random_state=seed, verbose=1, bootstrap=True, oob_score=True)
        print(f"Generated a random forest with {num_trees} trees and seed {seed}")

        # Train the classifier
        rf_classifier.fit(x_train, y_train)

        # Make predictions
        y_pred = rf_classifier.predict(x_test)

        # Create the directory to store rf results
        os.mkdir(rf_save_path)

        # TODO
        # Save the random forest
        # The random forest is compressed to save disk space
        rf_trees_path = os.path.join(rf_save_path, "random_forest.joblib")
        with open(rf_trees_path, 'wb') as f:
            joblib.dump(rf_classifier, f, compress=3)
        
        # Save metrics used in a txt file
        rf_data_file = os.path.join(rf_save_path, "analysis_data.txt")
        with open(rf_data_file, 'w') as f:
            f.write("\nRANDOM FOREST ANALYSIS SHEET\n")
            f.write("List of features used\n---------------------------------\n")
            for feature in feature_list:
                f.write(f"{feature}\n")

            f.write("\nRANDOM FOREST PARAMETERS\n---------------------------------\n")
            f.write(f"Number of trees grown: {num_trees}\n")
            f.write(f"Seed used: {seed}\n")
    else:
        print(f"A random forest already exists in the save path {rf_save_path}")

def rf_analysis(rf_save_path: str):
    if not os.path.exists(rf_save_path):
        print(f"There is no directory to {rf_save_path}!")
        return
    rf_tree_path = os.path.join(rf_save_path, "random_forest.joblib")
    if not os.path.exists(rf_tree_path):
        print(f"There is no random forest in the path {rf_tree_path}!")
        return
    rf = joblib.load(os.path.join(rf_save_path, "random_forest.joblib"))
    

training_path = "Random_Forest/RFtraining.csv"
testing_path = "Random_Forest/RFtesting.csv"

training_df = pd.read_csv(training_path)
testing_df = pd.read_csv(testing_path)

# Add some extra columns to the dataframes
# Training
training_df['IQR.improv'] = training_df['IQR.resid'] / training_df['IQR.lc']
training_df['Redchisq.improv'] = training_df['Redchisq.resid'] / training_df['Redchisq.lc']
training_df['P_norm.improv'] = training_df['Prob_norm.resid'] / training_df['P_norm.lc']
training_df['P_autocor.improv'] = training_df['Prob_autocor.resid'] / training_df['P_autocor.lc']
training_df['P_trend.improv'] = training_df['Prob_trend.resid'] / training_df['P_trend.lc']

# Testing
testing_df['IQR.improv'] = testing_df['IQR.resid'] / testing_df['IQR.lc']
testing_df['Redchisq.improv'] = testing_df['Redchisq.resid'] / testing_df['Redchisq.lc']
testing_df['P_norm.improv'] = testing_df['Prob_norm.resid'] / testing_df['P_norm.lc']
testing_df['P_autocor.improv'] = testing_df['Prob_autocor.resid'] / testing_df['P_autocor.lc']
testing_df['P_trend.improv'] = testing_df['Prob_trend.resid'] / testing_df['P_trend.lc']

feature_list = ['tic_Radius', 'tic_eTmag', 'TCF_power', 'snr.transit', 
                                  'TCF_mad', 'TCF_depthSNR', 'TCF_harmonic', 'sm.axis', 
                                  'Redchisq.improv', 'Redchisq.lc', 'outer_range.lc', 
                                  'even.odd.p_value', 'Median_sd.diff', 
                                  'P_autocor.lc', 'Prob_autocor.resid', 
                                  'P_norm.improv', 'P_norm.lc', 'P_trend.improv', 
                                  'P_trend.lc', 'Prob_trend.resid', 'trans.p_value', 'TCF_period', 'Class']
print(f"Creating random forest of feature list {feature_list}")
train_rf(training_df, testing_df, feature_list, "Random_Forest/first_fs_rf_python/", num_trees=100)