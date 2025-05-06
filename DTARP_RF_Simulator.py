import pandas as pd
import DTARP_RF_Functions as drf_func
import random, time

# Try the year 2 dataset
# Could see what 
# Radius VS period (Y1 & Y2) to verify the planet types are consistent

# TODO: Make a NEW recursive tree program
# BRAINSTORMING: The goal to to build a feature list that provides the best results.
# How do we cut this down?
# Inputs: Features selected
# Outputs: ROC for training + testing. ROC difference could be evaluated, looking for a value as close to 0 at possible to signify little to no overfitting.
# Cites I have read
# General Feature Selection Guide: https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
# sklearn doc on KSelect: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
#
# The Strategy: Filtering. I am somewhat doing this, although manually. The idea is to look at feature importance and use that to build the feature set.
# Instead of keeping the base feature set constant between trees, select the K best features upon evaluation as the base feature set for the next forest.
# Will need to calculate statistics to use to determine this.
# ROC and ROC-diff should be good, but other statistics may be good too
# 1. Start with base feature set
# 2. Add a pseudo-random selection of features
# 2. Grow a forest
# 3. Select the K most important features based on statistics
# 4. Grow another forest with only those features
# 5. Repeat steps 2-5 X number of times specified by the user.
# What needs to be figuring out: The scoring functions with which to choose the K best features
# Some scoring functions to try: chi2, f_classif, mutual_info_classif
# If I can combine results from multiple of the test datasets, that would be good too
# Perhaps I make my own custom scoring functions too

# Adjustable things: K, score functions
def iterative_rf_build(training_df, rf_analysis_folder, rf_trees=[10000], rf_criterions=["gini"], rf_seeds=[42], fs_num=0):
    for seed in rf_seeds:
        for num_trees in rf_trees:
            for criterion in rf_criterions:
                # Begin with base feature set
                feature_groups = generate_feature_clusters()
                base_feature_set = build_base_feature_set()

                # Grow a tree with it to establish a baseline
                rf_save_path = f"Random_Forest/rf_trees{num_trees}_{criterion}_seed{seed}_{fs_num}/"
                drf_func.train_rf(training_df, feature_list, rf_save_path, num_trees=num_trees, criterion=criterion, seed=seed)
                
                # Exploration phase: Two modes perhaps
                # Mode 1: Don't select the same feature again until all features have been selected
                # Mode 2: Repeat selection is fine (Implement this first)

                # Select a random number of features

                # Grow a forest

                # Select K most important features using score mechanisms

                # Grow a forest to inspect the new baseline

                # Repeat

    print("DONE")
    return

# simulate_rf_combinations
# Simulate a set of random forests based on provided arguments.
# Parameters
# feature_list: A list of features to simulate random forests on.
# rf_trees: A list of the number of trees to simulate.
# rf_criterions: A list of the criterions to simulate. Restricted to "gini", "log_loss", and "entropy"
# rf_seeds: A list of seeds to simulate.
# fs_num: An identifier for a feature set being passed in. By default, this is zero.
# plot_fi: Whether to plot the feature importance plot for the forest.
# dtarpsPlus: Whether to save all forests or only forests that surpass DTARPS-1 performance.
def simulate_rf_combinations(feature_list, training_df, rf_analysis_folder, rf_trees=[10000], rf_criterions=["gini"], rf_seeds=[42], fs_num=0, plot_fi=False, dtarpsPlus=False):
    for seed in rf_seeds:
        for num_trees in rf_trees:
            for criterion in rf_criterions:
                print(f"Growing random forest with {num_trees} trees and criterion {criterion} with seed {seed}")
                rf_save_path = f"Random_Forest/rf_trees{num_trees}_{criterion}_seed{seed}_{fs_num}/"
                drf_func.train_rf(training_df, feature_list, rf_save_path, num_trees=num_trees, criterion=criterion, seed=seed)
                drf_func.rf_analysis(rf_save_path, rf_analysis_folder, feature_list, dtarpsPlus=dtarpsPlus, csv_mode=True, csv_file="Random_Forest/forest_analysis_data.csv")
                if plot_fi:
                    drf_func.rf_feature_importance(rf_save_path, rf_analysis_folder, feature_list)
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
    eliminated_feature_set = []
    print(f"The base feature set is {base_feature_set}")

    feature_sets = []

    for i in range(num_feature_sets):
        fs = base_feature_set.copy()
        for fgroup in feature_groups:
            # Remove any entries already in the feature set or eliminated
            fgroup = list(set(fgroup) - set(base_feature_set) - set(eliminated_feature_set))
            
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
    # Also need to add the "Class" variable for the classification label
    base_fs = feature_df['Var'].tolist()
    base_fs.append('Class')
    return base_fs

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

def addExtraColumns():
    training_path = "Random_Forest/RFtraining.csv"
    testing_path = "Random_Forest/test_data/RFtesting.csv"

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

    # Save the datasets as csv files
    training_df.to_csv("Random_Forest/RFtrainingUpdate.csv", index=False)
    testing_df.to_csv("Random_Forest/test_data/RFtestingUpdate.csv", index=False)

def findMissingFeatures(rf_test1_path: str, rf_test2_path: str):
    df1 = pd.read_csv(rf_test1_path)
    df2 = pd.read_csv(rf_test2_path)

    # Figure out what features are missing
    f1 = df1.columns
    f2 = df2.columns
    print(f"Dataframe 1 has {len(f1)} columns, dataframe 2 has {len(f2)} columns")
    missing_columns = [col for col in f1 if col not in f2]
    print(f"There are {len(missing_columns)} missing columns, being: {missing_columns}")

# feature_list = ['tic_Radius', 'tic_eTmag', 'TCF_power', 'snr.transit', 
#                                   'TCF_mad', 'TCF_depthSNR', 'TCF_harmonic', 'sm.axis', 
#                                   'Redchisq.improv', 'Redchisq.lc', 'outer_range.lc', 
#                                   'even.odd.p_value', 'Median_sd.diff', 
#                                   'P_autocor.lc', 'Prob_autocor.resid', 
#                                   'P_norm.improv', 'P_norm.lc', 'P_trend.improv', 
#                                   'P_trend.lc', 'Prob_trend.resid', 'trans.p_value', 'TCF_period', 'Class']
# print(f"Creating random forest of feature list {feature_list}")

# TODO
# Currently, forests are being tested on a limited testing dataset, a subset of the year 1 TESS dataset.
# Now that the year 2 TESS dataset has been retreived, the forests should also be evaluated on the year 2 TESS data.

feature_list = ['TCF_period',
                'TCF_mad',
                'snr.transit',
                'planet_rad_tcf',
                'Folded_AD',
                'even.odd.p_value',
                'Class',
                'TCF_shape',
                'logg',
                'skew.lc',
                'Redchisq.lc',
                'IQR.lc',
                'LOESS_mnsnr',
                'POM.lc',
                'Prob_autocor.resid',
                'P_norm.lc',
                'Prob_trend.resid',
                'quantiles.resid.10',
                'trans.p_value'
                ]

# Create simulation sets
rf_trees = [10000]
rf_criterions = ["log_loss"]
rf_seeds = [42]

# feature_sets = simulate_feature_sets(100)
# print(f"Created a total of {len(feature_sets)} feature sets.")
# for fs in feature_sets:
#     print(f"FEATURE SET: {fs}")

training_path = "Random_Forest/RFtrainingUpdate.csv"
analysis_folder = "Random_Forest/test_data/"
training_df = pd.read_csv(training_path)

# fs_num = 0
# for feature_list in feature_sets:
#     simulate_rf_combinations(feature_list, training_df, analysis_folder, rf_trees, rf_criterions, rf_seeds, fs_num, dtarpsPlus=True)
#     fs_num += 1

# print(f"ALL DONE, it took {time.time}")
simulate_rf_combinations(feature_list, training_df, analysis_folder, rf_trees, rf_criterions, rf_seeds, 20, dtarpsPlus=True)
# training_path = "Random_Forest/RFtrainingUpdate.csv"
# analysis_folder = "Random_Forest/test_data/"
# training_df = pd.read_csv(training_path)
# simulate_rf_combinations(feature_list, training_df, analysis_folder, rf_trees, rf_criterions, rf_seeds, 13, plot_fi=False, dtarpsPlus=True)

# testing_path1 = "Random_Forest/test_data/year2_testing.csv"
# testing_path2 = "Random_Forest/test_data/year2_testing_with_large_zero_class.csv"

# testing_df1 = pd.read_csv(testing_path1)
# testing_df2 = pd.read_csv(testing_path2)

# # Add some extra columns to the dataframes
# # Note, this is in Jupiter radii, so need to multiply by the solar to jupiter radii ratio
# testing_df1['planet_rad_tcf'] = (testing_df1['TCF_depth'] ** 0.5) * testing_df1['tic_Radius'] * 9.73116
# testing_df2['planet_rad_tcf'] = (testing_df2['TCF_depth'] ** 0.5) * testing_df2['tic_Radius'] * 9.73116

# # Save the datasets as csv files
# testing_df1.to_csv("Random_Forest/test_data/RFYear2Testing.csv", index=False)
# testing_df2.to_csv("Random_Forest/test_data/RFYear2ZerosTesting.csv", index=False)