from datetime import datetime
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
from sklearn import set_config

# Trains a random forest using the RandomForestClassifier from the scikit-learn library.
# PARAMETERS
# training_df: Training set data for the RF
# testing_df: Testing set data for the RF
# feature_list: A list of feature names for the random forest
# rf_save_path: The file path to save the random forest into.
# num_trees: Number of trees to grow. The default is 100.
# seed: What seed to generate a random forest from. The default is 42.
def train_rf(training_df, feature_list, rf_save_path,
             num_trees=100, criterion="gini", seed=42):
    if training_df is None:
        print("The training set doesn't exist!")
        return -1
    
    # Constrain RF training to given feature list
    rf_train = training_df[feature_list]

    # Replace instances of infinity with NaN
    rf_train = rf_train.mask(np.isinf(rf_train), np.nan)

    # Drop any rows with NaN in them
    rf_train = rf_train.dropna()

    # Split the X and Y 
    x_train = rf_train.drop(columns=['Class'])
    y_train = rf_train['Class']

    # Set up a random forest classifier
    # n_estimators: How many trees are grown?
    # random_state: Seed for tree growth
    # The random forest will only be set up if the save path doesn't exist already (don't accidentally overwrite existing rfs)
    if not os.path.exists(rf_save_path):
        rf_classifier = RandomForestClassifier(n_estimators=num_trees, random_state=seed, verbose=1, bootstrap=True, oob_score=True, criterion=criterion, n_jobs=10)
        print(f"Generated a random forest with {num_trees} trees and seed {seed}")

        # Train the classifier
        rf_classifier.fit(x_train, y_train)

        # Create the directory to store rf results
        os.mkdir(rf_save_path)

        # Save the random forest
        # The random forest is compressed to save disk space
        rf_trees_path = os.path.join(rf_save_path, "random_forest.joblib")
        with open(rf_trees_path, 'wb') as f:
            joblib.dump(rf_classifier, f, compress=3)
        
        # Save metrics used in a txt file
        rf_data_file = os.path.join(rf_save_path, "rf_parameters_data.txt")
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
        return -1
    return 1

# Writes feature metrics into a specified csv path.
# This function is particularly used when doing the iterative_rf_build algorithm.
# PARAMETERS
# feature_csv_path: The path to read/update/save feature metrics to
# feature_list: The list of features that were selected for the last iteration.
# unified_list: The list of features that were chosen for the next iteration.
# METRICS WRITTEN
# Iterations: The number of forest iterations where this feature has made its appearance.
# Times_Chosen: The number of times this feature was chosen to be part of the new base feature list.
# Times_Discarded: The number of times this feature was discarded during feature selection.
# Persistence_Rate: The rate in which the feature was kept (Times_Chosen / (Times_Chosen + Times_Discarded))
# Rows are features, columns are described above
def write_featurecsv(feature_csv_path: str, feature_list: list, unified_list: list):
    csv_df = pd.DataFrame()
    if os.path.exists(feature_csv_path):
        csv_df = pd.read_csv(feature_csv_path)
    num_rows = len(csv_df)
    print(f"The CSV currently has {num_rows} features documented. There are {len(feature_list)} features in total to document.")

    # Iterate through the feature list
    for feature in feature_list:
        num_rows = len(csv_df)
        # Is this feature new to the csv?
        # METRICS
        # Iterations = Iterations + 1, otherwise 0
        # Times_Chosen = Times_Chosen + 1 if in unified_list, Times_Chosen if not, otherwise 1 or 0 if new
        # Times_Discarded = Times_Discarded + 1 if not in unified list, Times_Discarded if it is, otherwise 0 or 1 if new
        # Persistence_Rate = Update accordingly with other two metrics

        if "Feature Name" not in csv_df.columns or not [f for f in csv_df["Feature Name"] if f == feature]:
            rowNum = num_rows
            write_to_df(csv_df, rowNum, "Feature Name", feature)
            write_to_df(csv_df, rowNum, "Iterations", 1)
            times_chosen = 0
            times_discarded = 0
        else:
            rowNum = csv_df.index[csv_df["Feature Name"] == feature].tolist()[0]
            iterations = csv_df.at[rowNum, "Iterations"]
            iterations += 1
            write_to_df(csv_df, rowNum, "Iterations", iterations)
            times_chosen = csv_df.at[rowNum, "Times Chosen"]
            times_discarded = csv_df.at[rowNum, "Times Discarded"]
        if feature in unified_list:
            times_chosen += 1
        else:
            times_discarded += 1
        write_to_df(csv_df, rowNum, "Times Chosen", times_chosen)
        write_to_df(csv_df, rowNum, "Times Discarded", times_discarded)
        persistence_rate = times_chosen / (times_chosen + times_discarded)
        write_to_df(csv_df, rowNum, "Persistence Rate", persistence_rate)

    # Save the feature csv
    csv_df.to_csv(feature_csv_path, index=False)
    return

# Unify the provided feature lists into one coherent feature list.
# NOTE: This is NOT constrained by K. That is handled by the K function specifically.
def unify_feature_sets(feature_lists, secondary_lists):
    # Unify the results into one feature set.
    # Start by taking all features agreed upon.
    # From there, work down to get features until feature set has K features.
    unified_fs = list(reduce(lambda x, y: set(x) & set(y), feature_lists))
    print(f"Common features between them all: {unified_fs}")

    # Append from common features in the secondary_lists
    # Meant to be a subset of the feature lists that could add more features
    sub_fs = list(reduce(lambda x, y: set(x) & set(y), secondary_lists))

    # Append values until length = K or end of sub_fs
    for i in range(len(sub_fs)):
        if sub_fs[i] not in unified_fs:
            unified_fs.append(sub_fs[i])
    
    return unified_fs

# Algorithm to perform the ANOVA feature selection test and select the K best features from that algorithm (for each dataset).
# K is specified as a parameter.
# Lower down, individual selected feature sets are unified into one coherent feature list to be used.
def select_Kfeatures(rf_analysis_folder: str, feature_csv_path: str, feature_list: list, k: int):
    print("Selecting K features. Currently testing.")
    # Tell sklearn to preserve pandas dataframes so we can preserve the feature names
    set_config(transform_output="pandas")

    # Store resulting selected feature sets
    feature_lists = []

    for tf in os.listdir(rf_analysis_folder):
        testing_df = pd.read_csv(os.path.join(rf_analysis_folder, tf))
        # Constrain dataframe to the feature list
        testing_df = testing_df[feature_list]

        # Replace instances of infinity with NaN
        testing_df = testing_df.mask(np.isinf(testing_df), np.nan)

        # Drop any rows with NaN in them
        testing_df = testing_df.dropna()

        x_test = testing_df.drop(columns=['Class'])
        y_test = testing_df['Class']
        print(f"Loaded test data. Number of entries: {len(x_test)}")

        # Perform the ANOVA feature test.
        # Works best on numerical-input, categorical-output based sets.
        print("Performing ANOVA feature selection test...")
        print(f"Shapes: Training: {x_test.shape}, Testing: {y_test.shape}")
        x_new = SelectKBest(f_classif, k=k).fit_transform(x_test, y_test)
        print(f"Number of features: {len(x_new.columns)}")
        print(f"Features selected: {x_new.columns}")
        feature_lists.append(x_new.columns)

    # Make secondary list consist of other two datasets
    # MANUALLY SET
    # -------------------------------------------------------------------
    y2_lists = []
    y2_lists.append(feature_lists[1])
    y2_lists.append(feature_lists[2])
    unified_fs = unify_feature_sets(feature_lists, y2_lists)
    # -------------------------------------------------------------------
    # Constrain it to k features
    unified_fs = unified_fs[0:max(k, len(unified_fs))]

    print(f"Final feature list: {unified_fs}")

    # When the final unified list is determined, write to the feature importance csv
    write_featurecsv(feature_csv_path, feature_list, unified_fs)
    return unified_fs

# Algorithm to perform the ANOVA feature selection test and select the features on the top percentile from that algorithm (for each dataset).
# The percentile is specified as a parameter.
# Lower down, individual selected feature sets are unified into one coherent feature list to be used.
def select_TopPercentage(rf_analysis_folder: str, feature_csv_path: str, feature_list: list, percent: float):
    print("Selecting features above a percentage. Currently testing.")
    # Tell sklearn to preserve pandas dataframes so we can preserve the feature names
    set_config(transform_output="pandas")

    # Store resulting selected feature sets
    feature_lists = []

    for tf in os.listdir(rf_analysis_folder):
        testing_df = pd.read_csv(os.path.join(rf_analysis_folder, tf))
        # Constrain dataframe to the feature list
        testing_df = testing_df[feature_list]

        # Replace instances of infinity with NaN
        testing_df = testing_df.mask(np.isinf(testing_df), np.nan)

        # Drop any rows with NaN in them
        testing_df = testing_df.dropna()

        x_test = testing_df.drop(columns=['Class'])
        y_test = testing_df['Class']
        print(f"Loaded test data. Number of entries: {len(x_test)}")

        # Perform the ANOVA feature test.
        # Works best on numerical-input, categorical-output based sets.
        print("Performing ANOVA feature selection test...")
        print(f"Shapes: Training: {x_test.shape}, Testing: {y_test.shape}")
        x_new = SelectPercentile(f_classif, percentile=percent).fit_transform(x_test, y_test)
        #x_new = SelectKBest(f_classif, k=k).fit_transform(x_test, y_test)
        print(f"Number of features: {len(x_new.columns)}")
        print(f"Features selected: {x_new.columns}")
        feature_lists.append(x_new.columns)

    # Make secondary list consist of the year 2 datasets
    y2_lists = []
    y2_lists.append(feature_lists[1])
    y2_lists.append(feature_lists[2])
    unified_fs = unify_feature_sets(feature_lists, y2_lists)

    print(f"Final feature list: {unified_fs}")

    # When the final unified list is determined, write to the feature importance csv
    write_featurecsv(feature_csv_path, feature_list, unified_fs)
    return unified_fs

# Helper function to write data to a txt file
# Open the file and write the provided string into it
def write_to_txt(rf_data_path: str, data: str):
    with open(rf_data_path, 'a') as f:
            return f.write(data)

# Helper function to write data to a csv file
# Open the csv file, check if the corresponding row/column exist.
# If not, create them
# Then, write the data into the csv file
def write_to_df(csv_df: pd.DataFrame, rowNum: int, columnName: str, data):
    if columnName not in csv_df.columns:
        csv_df[columnName] = np.nan
    csv_df.at[rowNum, columnName] = data
    return

# Perform random forest analysis. This function will write metrics into txt files (and a csv file if csv_mode is enabled).
def rf_analysis(rf_save_path: str, rf_analysis_folder: str, feature_list: list, csv_mode: bool = False, csv_file: str = "", keepForest: bool = False):
    if not ensure_forest_exists(rf_save_path, rf_analysis_folder):
        print(f"Some files are missing. Aborting...")
        return
    if csv_mode:
        print(f"Analyzing random forest. CSV mode selected")
    else:
        print(f"Analyzing random forest. TXT mode selected")
    
    # Load random forest and test dataset
    rf = joblib.load(os.path.join(rf_save_path, "random_forest.joblib"))

    # Index through the list of test datasets.
    # File index is tracked to save unique result file names.
    file_ind = 0

    csv_df = pd.DataFrame()
    if os.path.exists(csv_file):
        csv_df = pd.read_csv(csv_file)
    row_num = len(csv_df)

    fpr_ds = []
    tpr_ds = []
    thresholds_ds = []
    roc_auc_ds = []
    prec_ds = []
    recall_ds = []

    # Record name of the forest. Only run CSV analysis if this is a new forest.
    forest_name = os.path.dirname(rf_save_path)
    forest_name = os.path.basename(forest_name)
    
    if csv_mode and "Forest Name" in csv_df.columns and forest_name in csv_df["Forest Name"]:
        print(f"Forest {forest_name} has been previously analyzed, skipping analysis.")
        return
    elif csv_mode:
        print(f"Forest {forest_name} is being analyzed.")
        # You will get a deprecation warning here. This is because pd.Dataframe does not like str, it prefers the generic object type.
        # However, there is no easy way to typecast the string here into an object, so this warning is ignored.
        write_to_df(csv_df, row_num, "Forest Name", forest_name)

    for tf in os.listdir(rf_analysis_folder):
        testing_df = pd.read_csv(os.path.join(rf_analysis_folder, tf))
        # Constrain dataframe to the feature list
        testing_df = testing_df[feature_list]

        # Replace instances of infinity with NaN
        testing_df = testing_df.mask(np.isinf(testing_df), np.nan)

        # Drop any rows with NaN in them
        testing_df = testing_df.dropna()

        x_test = testing_df.drop(columns=['Class'])
        y_test = testing_df['Class']
        print(f"Loaded test data. Number of entries: {len(x_test)}")

        # Make predictions
        y_pred_prob = rf.predict_proba(x_test)
        testing_df[f"{forest_name}"] = y_pred_prob

        removeForest = False
        # As the metrics are found, they should be written into a txt file or saved as a csv
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1], pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        prec, recall, _ = precision_recall_curve(y_test, y_pred_prob[:, 1], pos_label=1)

        # Save ROC values if the csv doesn't exist already
        roc_df = pd.DataFrame({
            'TPR': tpr,
            'FPR': fpr,
            'Threshold': thresholds
        })
        if not os.path.exists(os.path.join(rf_save_path, f'roc_thresholds{file_ind}.csv')):
            roc_df.to_csv(os.path.join(rf_save_path, f'roc_thresholds{file_ind}.csv'), index=False)

        # Record timestamp
        time = datetime.now()
        if csv_mode:
            # You will get a deprecation warning here. This is because pd.Dataframe wants this to be cast as a datetime.
            # The way to typecast it doesn't seem straightforward, hence this warning is ignored.
            write_to_df(csv_df, row_num, f"Time", time)
            write_to_df(csv_df, row_num, f"ROC_AUC {tf}", roc_auc)

        # Add values to datastructures
        fpr_ds.append(fpr)
        tpr_ds.append(tpr)
        thresholds_ds.append(thresholds)
        roc_auc_ds.append(roc_auc)
        prec_ds.append(prec)
        recall_ds.append(recall)

        # Search for notable thresholds. 

        thresholdTPR = 0
        thresholdFPR = 0

        thresholdTPR = 0.9283
        thresholdFPR = 0.0037
        # First condition
        firstVals = roc_df[roc_df['TPR'] >= thresholdTPR]
        if csv_mode:
            write_to_df(csv_df, row_num, f"DtTPRf {tf}", firstVals.values[0][0])
            write_to_df(csv_df, row_num, f"DtFPRf {tf}", firstVals.values[0][1])

        # Second condition
        secondVals = roc_df[roc_df['FPR'] <= thresholdFPR]
        if csv_mode:
            write_to_df(csv_df, row_num, f"DtTPRl {tf}", secondVals.values[len(secondVals.values) - 1][0])
            write_to_df(csv_df, row_num, f"DtFPRl {tf}", secondVals.values[len(secondVals.values) - 1][1])
            
        file_ind += 1
    
    # If the forest is not to be kept, remove it at the end
    if not keepForest:
        print("Forest not being stored to save space.")
        os.remove(os.path.join(rf_save_path, "random_forest.joblib"))

    # Save the csv file if csv_mode is enabled
    if csv_mode:
        csv_df.to_csv(csv_file, index=False)

    # Save the feature list as a csv file, this is so the user doesn't need to remember what features were used every time.
    feature_df = pd.DataFrame({
        'Feature name': feature_list
    })
    feature_df.to_csv(os.path.join(rf_save_path, f'feature_list{file_ind}.csv'), index=False)
    
    # Generate a plt plot showing the ROC curve
    # TODO: This plot is squished, I haven't been able to figure out how to fix it! Come back to this in the future.
    # These plots will display data from all analysis files, in addition to the base one.

    plt.figure(figsize=(10,10))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.set_title('ROC curve on random forest')
    ax1.set_xscale('log')
    ax1.set_yscale('linear')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')

    for i in range(len(fpr_ds)):
        roc_display = RocCurveDisplay(fpr=fpr_ds[i], tpr=tpr_ds[i], roc_auc=roc_auc_ds[i], estimator_name=f'Rf {i}', pos_label=1).plot(ax=ax1)
        pr_display = PrecisionRecallDisplay(precision=prec_ds[i], recall=recall_ds[i]).plot(ax=ax2)

    # Show grids for plot
    plt.tight_layout()
    # Uncomment for debugging purposes: comment to prevent program interrupt
    #plt.show()
    plt.savefig(os.path.join(rf_save_path, "ROC_curve.png"))
    plt.cla()
    plt.clf()

# Plot the feature importance plot here.
def rf_feature_importance(rf_save_path: str, rf_analysis_folder: str, feature_list: list):
    if (not ensure_forest_exists(rf_save_path, rf_analysis_folder)):
        print(f"Some required files missing. Aborting...")
        return
    test_path = os.path.join(rf_analysis_folder, "RFtestingUpdate.csv")
    print("Generating feature importance plot")
    
    # Plot feature importance: what variables were the most important?
    # Load random forest and test dataset
    rf = joblib.load(os.path.join(rf_save_path, "random_forest.joblib"))
    testing_df = pd.read_csv(test_path)
    # Constrain dataframe to the feature list
    testing_df = testing_df[feature_list]

    # Replace instances of infinity with NaN
    testing_df = testing_df.mask(np.isinf(testing_df), np.nan)

    # Drop any rows with NaN in them
    testing_df = testing_df.dropna()

    x_test = testing_df.drop(columns=['Class'])
    y_test = testing_df['Class']
    print("Loaded test data")

    # Feature importance is calculated using feature permutation.
    # Sklearn docs: https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
    # TLDR: Measures contribution of each feature to a fitted model's statistical performance on a given dataset
    # Randomly shuffle values of a single feature and observing resulting degradation
    result = permutation_importance(
        rf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=x_test.columns)

    ax3 = plt.subplot()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax3)
    ax3.set_title("Feature importances using permutation on random forest")
    ax3.set_ylabel("Mean accuracy decrease")
    # Uncomment for debugging purposes: comment to prevent program interrupt
    #plt.show()

    plt.savefig(os.path.join(rf_save_path, "feature_importance.png"), bbox_inches="tight")
    plt.cla()
    plt.clf()

# Ensures that all required forest files exist for analysis.
def ensure_forest_exists(rf_save_path: str, rf_analysis_folder: str) -> bool:
    rf_forest_dir = os.path.join(rf_save_path, "random_forest.joblib")
    if not os.path.exists(rf_forest_dir):
        print(f"There is no forest in the path {rf_forest_dir}!")
        return False
    if not os.path.exists(rf_analysis_folder):
        print(f"The analysis folder path {rf_forest_dir} doesn't exist!")
        return False
    if os.listdir(rf_analysis_folder).count == 0:
        print(f"Analysis folder provided, but no analysis files within it.")
        return False
    return True
    