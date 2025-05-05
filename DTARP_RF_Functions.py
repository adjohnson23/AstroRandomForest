import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.inspection import permutation_importance

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
        return
    
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


# TODO: Make a version of rf_analysis which puts all data into a csv file instead

def rf_csv_analysis(rf_save_path: str, csv_path: str, feature_list: list):
    
    return

# RF ANALYSIS FUNCTION
# TODO: Figure out a more elegant/smart way to get a good feature list
# 

def rf_analysis(rf_save_path: str, rf_analysis_folder: str, feature_list: list, dtarpsPlus: bool = False):
    if not ensure_forest_exists(rf_save_path, rf_analysis_folder):
        print(f"Some files are missing. Aborting...")
        return

    print("Analyzing random forest...")
    
    # Load random forest and test dataset
    rf = joblib.load(os.path.join(rf_save_path, "random_forest.joblib"))

    # Indexer + datastructures to hold results
    file_ind = 0
    fpr_ds = []
    tpr_ds = []
    thresholds_ds = []
    roc_auc_ds = []
    prec_ds = []
    recall_ds = []

    for f in os.listdir(rf_analysis_folder):
        testing_df = pd.read_csv(os.path.join(rf_analysis_folder, f))
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

        rf_data_file = os.path.join(rf_save_path, f"analysis_data{file_ind}.txt")
        if (os.path.exists(rf_data_file)):
            # Overwrite the existing analysis file
            print("Analysis file already exists, overwriting...")
            os.remove(rf_data_file)

        removeForest = False
        with open(rf_data_file, 'w') as f:
            print("Writing to analysis file...")
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
                chars_written = f.write(f"ROC_AUC score: {roc_auc}\n")
            
            # Add values to datastructures
            fpr_ds.append(fpr)
            tpr_ds.append(tpr)
            thresholds_ds.append(thresholds)
            roc_auc_ds.append(roc_auc)
            prec_ds.append(prec)
            recall_ds.append(recall)

            # Write notable threshold values into the txt file
            # If the file is the one used for DTARPS, use DTARPS thresholds, otherwise use set thresholds
            # 1. First threshold where TPR > TPR of DTARPS1
            # 2. Last threshold where FPR < FPR of DTARPS1
            # 3+. Any thresholds where both 1 and 2 are met

            if f.name == "RFtesting.csv":
                dtarpsTPR = 0.9283
                dtarpsFPR = 0.0037
                dtarpsThreshold = 0.174
                chars_written = f.write(f"\nNOTABLE THRESHOLDS\nComparing DTARPS performance: TPR {dtarpsTPR}, FPR {dtarpsFPR}, Threshold {dtarpsThreshold}\n")
                # First condition
                firstVals = roc_df[roc_df['TPR'] >= dtarpsTPR]
                chars_written = f.write(f"First threshold that surprasses DTARPS TPR: {firstVals.values[0]}\n")

                # Second condition
                secondVals = roc_df[roc_df['FPR'] <= dtarpsFPR]
                chars_written = f.write(f"Last threshold that surprasses DTARPS FPR: {secondVals.values[len(secondVals.values) - 1]}\n")

                # Finally any thresholds that meet both conditions
                better_thresholds = secondVals[secondVals['TPR'] >= dtarpsTPR]
                better_thresholds = better_thresholds.dropna()
                if better_thresholds.size == 0:
                    chars_written = f.write(f"No thresholds perform decisively better than DTARPS\n")
                    removeForest = True
                else:
                    chars_written = f.write(f"DECISIVE THRESHOLDS\n")
                    better_vals = better_thresholds.values
                    for i in range(len(better_vals)):
                        chars_written = f.write(f"Threshold {i}: {better_vals[i]}\n")
            else:
                thresholdTPR = 0.85
                thresholdFPR = 0.005
                chars_written = f.write(f"\nNOTABLE THRESHOLDS\nComparing general performance: TPR {thresholdTPR}, FPR {thresholdFPR}\n")
                # First condition
                firstVals = roc_df[roc_df['TPR'] >= thresholdTPR]
                chars_written = f.write(f"First threshold that surprasses set TPR: {firstVals.values[0]}\n")

                # Second condition
                secondVals = roc_df[roc_df['FPR'] <= thresholdFPR]
                chars_written = f.write(f"Last threshold that surprasses set FPR: {secondVals.values[len(secondVals.values) - 1]}\n")

                # Finally any thresholds that meet both conditions
                better_thresholds = secondVals[secondVals['TPR'] >= thresholdTPR]
                better_thresholds = better_thresholds.dropna()
                if better_thresholds.size == 0:
                    chars_written = f.write(f"No thresholds perform decisively better than set thresholds\n")
                    removeForest = True
                else:
                    chars_written = f.write(f"DECISIVE THRESHOLDS\n")
                    better_vals = better_thresholds.values
                    for i in range(len(better_vals)):
                        chars_written = f.write(f"Threshold {i}: {better_vals[i]}\n")
        
        # This is the base analysis file to be looking for and whether it did better than DTARPS
        if f.name == "RFtesting.csv":
            # Only remove if forests worse than DTARPS are being filtered out
            if removeForest and dtarpsPlus:
                print("Forest did not perform better than DTARPS: Scrapping")
                for f in os.listdir(rf_save_path):
                    os.remove(os.path.join(rf_save_path, f))
                os.rmdir(rf_save_path)
                return
        file_ind += 1
    
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
    plt.savefig(os.path.join(rf_save_path, "ROC_curve.png"))
    plt.cla()
    plt.clf()

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