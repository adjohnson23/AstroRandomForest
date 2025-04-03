import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Trains a random forest using the RandomForestClassifier from the scikit-learn library.
# PARAMETERS
# training_df: Training set data for the RF
# testing_df: Testing set data for the RF
def train_rf(training_df, testing_df, feature_list, num_trees=100, seed=42):
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
    print(x_train.head(5))
    print(y_train.head(5))
    # print(f"Training dataset max values:")
    # for col in rf_train.columns:
    #     print(f"{col}: {rf_train[col].max()}")
    
    # print(f"Test dataset max values:")
    # for col in rf_train.columns:
    #     print(f"{col}: {rf_train[col].max()}")

    # Set up a random forest classifier
    # n_estimators: How many trees are grown?
    # random_state: Seed for tree growth
    rf_classifer = RandomForestClassifier(n_estimators=num_trees, random_state=seed, verbose=1)
    print(f"Generated a random forest with {num_trees} trees and seed {seed}")

    # Train the classifier
    rf_classifer.fit(x_train, y_train)

    # Make predictions
    y_pred = rf_classifer.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"This random forest had an accuracy of {accuracy}")

    # TODO
    # Save the random forest
    

    # Get TPR, FPR, TNR, FNR

    # Confusion matrix?

    # Feature Importance

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
train_rf(training_df, testing_df, feature_list, 100)