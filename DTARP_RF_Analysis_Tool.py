# This file defines functions for an interactive console tool for looking at predicted data on random forests.
# NOTE: This is currently configured to analyze exoplanet data.

import os

def init_data_analyzer_tool():
    dataset_path = "placeholder"

    print("Welcome to the Forest Planet Data Analyzer Tool (FPDAT)!\nThis tool is to be used after using a forest to predict on a dataset.")

    while dataset_path == "placeholder" and not os.path.exists(dataset_path):
        dataset_path = input("Enter the path to your dataset to analyze here: ")
        if os.path.exists(dataset_path):
            print("Dataset found!")
        else:
            print("The provided dataset does not exist. Make sure you spelled it correctly.")
    
    ui = ""

    while ui != "q":
        ui = input("Enter a command for analyzing the data. ")
        if ui == "h":
            print("HELP: List of commands and their syntax")
            print("h: Get help ")