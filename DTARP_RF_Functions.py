import numpy as np
import pandas as pd


# Trains a random forest using the RandomForestClassifier from the scikit-learn library.
# PARAMETERS
# training_df: Training set data for the RF
# testing_df: Testing set data for the RF
def train_rf(training_df, testing_df, feature_list):
    if training_df is None:
        print("The training set doesn't exist!")
        return
    
    # Constrain RF training to given feature list
    rf_train = training_df[feature_list]
    rf_test = testing_df[feature_list]
    print(rf_train.head(5))

    # Set up a random forest classifier
    


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
train_rf(training_df, testing_df, feature_list)