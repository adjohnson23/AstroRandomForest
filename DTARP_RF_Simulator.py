import pandas as pd
import DTARP_RF_Functions as drf_func
import random

# simulate_rf_combinations
# Simulate a set of random forests based on provided arguments.
# Parameters
# feature_list: A list of features to simulate random forests on.
# rf_trees: A list of the number of trees to simulate.
# rf_criterions: A list of the criterions to simulate. Restricted to "gini", "log_loss", and "entropy"
# rf_seeds: A list of seeds to simulate.
# fs_num: An identifier for a feature set being passed in. By default, this is zero.
def simulate_rf_combinations(feature_list, training_df, testing_df, rf_trees=[10000], rf_criterions=["gini"], rf_seeds=[42], fs_num=0):
    for seed in rf_seeds:
        for num_trees in rf_trees:
            for criterion in rf_criterions:
                print(f"Growing random forest with {num_trees} trees and criterion {criterion} with seed {seed}")
                rf_save_path = f"Random_Forest/rf_trees{num_trees}_{criterion}_seed{seed}_{fs_num}/"
                drf_func.train_rf(training_df, testing_df, feature_list, rf_save_path, num_trees=num_trees, criterion=criterion, seed=seed)
                drf_func.rf_analysis(rf_save_path)
    print("DONE")

# TODO: Tinker with feature set
# Possibly look more into feature clusters, or use the ones I already made
# Use this to implement a feature list combination generator

def simulate_feature_sets(num_feature_sets: int):
    # For now, start with base features and pick features from the feature groups
    # This is a mostly random picker. Given a number, generate that many random feature sets. 
    # For each group, 50/50 pick one random variable from that group. DO NOT include duplicate variables already in the feature set.
    # This strategy is susceptible to duplicate feature sets.
    
    # Build the base feature set and feature groups
    feature_groups = generate_feature_clusters()
    base_feature_set = build_base_feature_set()
    print(f"The base feature set is {base_feature_set}")

    feature_sets = []

    for i in range(num_feature_sets):
        fs = base_feature_set.copy()
        for fgroup in feature_groups:
            # Remove any entries already in the feature set
            fgroup = list(set(fgroup) - set(base_feature_set))
            
            if len(fgroup) > 0:
                pickVariable = random.randint(0, 1)
                # A feature will be chosen
                if pickVariable == 1:
                    # Choose a random feature
                    feature = fgroup[random.randint(0, len(fgroup) - 1)]
                    fs.append(feature)
        feature_sets.append(fs.copy())

    return feature_sets

# Builds the base feature set by obtaining all variables that should definitely be included.
def build_base_feature_set():
    feature_path = "feature_list.csv"
    feature_df = pd.read_csv(feature_path)
    feature_df = feature_df[feature_df['Include'] == "YES"]
    return feature_df['Var'].tolist()

# Load the data file and group features that are highly correlated with one another.
# Return a list of the thread clusters, which will form the basis of building a feature list.
def generate_feature_clusters():
    # Remove features that WONT be used
    feature_path = "feature_list.csv"
    feature_df = pd.read_csv(feature_path)
    feature_df = feature_df[feature_df['Include'] != "NO"]

    feature_group_names = feature_df['Group'].unique()
    feature_groups = []
    for group_name in feature_group_names:
        fg = feature_df[feature_df['Group'] == group_name]
        feature_groups.append(fg['Var'].tolist())

    # Generate the feature clusters
    print(f"The following feature groups were formed: {feature_groups}")
    return feature_groups

training_path = "Random_Forest/RFtraining.csv"
testing_path = "Random_Forest/RFtesting.csv"

training_df = pd.read_csv(training_path)
testing_df = pd.read_csv(testing_path)

# # Add some extra columns to the dataframes
# # Training
# training_df['IQR.improv'] = training_df['IQR.resid'] / training_df['IQR.lc']
# training_df['Redchisq.improv'] = training_df['Redchisq.resid'] / training_df['Redchisq.lc']
# training_df['P_norm.improv'] = training_df['Prob_norm.resid'] / training_df['P_norm.lc']
# training_df['P_autocor.improv'] = training_df['Prob_autocor.resid'] / training_df['P_autocor.lc']
# training_df['P_trend.improv'] = training_df['Prob_trend.resid'] / training_df['P_trend.lc']

# # Testing
# testing_df['IQR.improv'] = testing_df['IQR.resid'] / testing_df['IQR.lc']
# testing_df['Redchisq.improv'] = testing_df['Redchisq.resid'] / testing_df['Redchisq.lc']
# testing_df['P_norm.improv'] = testing_df['Prob_norm.resid'] / testing_df['P_norm.lc']
# testing_df['P_autocor.improv'] = testing_df['Prob_autocor.resid'] / testing_df['P_autocor.lc']
# testing_df['P_trend.improv'] = testing_df['Prob_trend.resid'] / testing_df['P_trend.lc']

# feature_list = ['tic_Radius', 'tic_eTmag', 'TCF_power', 'snr.transit', 
#                                   'TCF_mad', 'TCF_depthSNR', 'TCF_harmonic', 'sm.axis', 
#                                   'Redchisq.improv', 'Redchisq.lc', 'outer_range.lc', 
#                                   'even.odd.p_value', 'Median_sd.diff', 
#                                   'P_autocor.lc', 'Prob_autocor.resid', 
#                                   'P_norm.improv', 'P_norm.lc', 'P_trend.improv', 
#                                   'P_trend.lc', 'Prob_trend.resid', 'trans.p_value', 'TCF_period', 'Class']
# print(f"Creating random forest of feature list {feature_list}")

# Create simulation sets
rf_trees = [10000]
rf_criterions = ["log_loss"]
rf_seeds = [42]

feature_sets = simulate_feature_sets(100)
print(f"Created a total of {len(feature_sets)} feature sets.")
for fs in feature_sets:
    print(f"FEATURE SET: {fs}")

fs_num = 0
for feature_list in feature_sets:
    simulate_rf_combinations(feature_list, training_df, testing_df, rf_trees, rf_criterions, rf_seeds, fs_num)
    fs_num += 1