import pandas as pd
import DTARP_RF_Functions as drf_func

# simulate_rf_combinations
# Simulate a set of random forests based on provided arguments.
# Parameters
# feature_list: A list of features to simulate random forests on.
# rf_trees: A list of the number of trees to simulate.
# rf_criterions: A list of the criterions to simulate. Restricted to "gini", "log_loss", and "entropy"
# rf_seeds: A list of seeds to simulate.
def simulate_rf_combinations(feature_list, training_df, testing_df, rf_trees=[10000], rf_criterions=["gini"], rf_seeds=[42]):
    for seed in rf_seeds:
        for num_trees in rf_trees:
            for criterion in rf_criterions:
                print(f"Growing random forest with {num_trees} trees and criterion {criterion} with seed {seed}")
                rf_save_path = f"Random_Forest/rf_trees{num_trees}_{criterion}_seed{seed}/"
                drf_func.train_rf(training_df, testing_df, feature_list, rf_save_path, num_trees=num_trees, criterion=criterion, seed=seed)
                drf_func.rf_analysis(rf_save_path)
    print("DONE")

# TODO: Tinker with feature set
# Possibly look more into feature clusters, or use the ones I already made
# Use this to implement a feature list combination generator

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

# Create simulation sets
rf_trees = [10000, 25000, 50000]
rf_criterions = ["gini", "log_loss", "entropy"]
rf_seeds = [42, 10, 25]

simulate_rf_combinations(feature_list, training_df, testing_df, rf_trees, rf_criterions, rf_seeds)