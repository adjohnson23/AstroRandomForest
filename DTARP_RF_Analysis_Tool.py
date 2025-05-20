# This file defines functions for an interactive console tool for looking at predicted data on random forests.
# NOTE: This is currently configured to analyze exoplanet data.

import os
import pandas as pd
import matplotlib.pyplot as plt

def init_data_analyzer_tool():
    dataset_path = "placeholder"
    df = []

    print("Welcome to the Forest Planet Data Analyzer Tool (FPDAT)!\nThis tool is to be used after using a forest to predict on a dataset.")

    # Finding a dataset
    while dataset_path == "placeholder" and not os.path.exists(dataset_path):
        dataset_path = input("Enter the path to your dataset to analyze here: ")
        root, extension = os.path.splitext(dataset_path)
        if not (extension == ".csv" or extension == ".xlsx"):
            print(f"Invalid file with extension {extension} provided. Please provide a csv or xlsx file.")
            dataset_path = "placeholder"
        elif os.path.exists(dataset_path):
            print("Dataset found!")
            df = pd.read_csv(dataset_path)
            if "Predicted Class" not in df.columns:
                print("This dataset hasn't been predicted by a random forest yet. Please complete this before running this tool on it.")
                dataset_path = "placeholder"
        else:
            print("The provided dataset does not exist. Make sure you spelled it correctly.")
    
    ui = ""

    # Remove non-feature columns
    x_df = df.drop(columns=['Class', 'Predicted Class'])
    y_df = {
        'Class': df['Class'],
        'Predicted Class': df['Predicted Class']
    }

    # Command interface
    while ui != "q":
        ui = input("Enter a command for analyzing the data: ")
        if ui == "h":
            print("--- HELP: List of commands and their syntax ---")
            print("h: Print out list of commands available ")
        elif ui == "q":
            print("Quitting...")
            exit(1)
        else:
            command = ui.split(" ")

            print(f"Entered command {command}")
            # FEATURE COMMAND: View results from one or all features, or list available features
            if command[0] == "feature":
                if len(command) == 1:
                    list_features(x_df)
                elif command[1] == "-h":
                    print("--- HELP: Feature command ---")
                    print("Syntax: feature <arg1>")
                    print("feature -h : Get help for this command.")
                    print("feature : List available features to analyze.")
                    print("feature -a : Analyze ALL features.")
                    print("feature -f <eligible feature> : Analyze the selected feature.")
                elif command[1] == "-a":
                    print("Analyzing all features. Not yet implemented")
                elif command[1] == "-f":
                    if command[2] in x_df.columns:
                        print(f"Analyzing feature {command[2]}.")
                        analyze_feature(df, command[2])
                    else:
                        print(f"Feature {command[2]} not recognized in dataset. Please enter a valid feature or check available features with the command >feature")
                else:
                    print("Command not recognized. Type >feature -h for help.")
            else:
                print("Command not recognized. Type >h for help.")

def list_features(df: pd.DataFrame):
    print(f"Available features in the dataset:")
    for f in df.columns:
        print(f)
    return

# Analyze a selected feature.
# This function currently plots a scatterplot comparing the feature to the Predicted Class by default.
# It also colors in green dots for positive cases and red dots for negative ones to paint a better picture of the spread.
def analyze_feature(df: pd.DataFrame, f: str):
    # Divide datasets into valid and invalid exoplanets
    # Remove non-feature columns
    df_valid = df[df['Class'] == 1]
    df_invalid = df[df['Class'] == 0]

    # Plot the scatterplot
    plt.figure(figsize=(6,6))

    # Plot classified planets in green and classified nonplanets in red
    plt.scatter(df_valid[f], df_valid['Predicted Class'], label="Exoplanets", s=10)
    plt.scatter(df_invalid[f], df_invalid['Predicted Class'], label="Non-exoplanets", s=10)
    plt.xlabel(f"{f}")
    plt.ylabel("Predicted Class")
    plt.title(f"Analyzing feature {f}")

    # Show the plot
    plt.show()
    return

init_data_analyzer_tool()