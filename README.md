# AstroRandomForest
Random forests in python to train on exoplanet data!

# Required Libraries
- Pandas: Responsible for reading and writing csv files as dataframes and utilizing those dataframes to train the forests.
- sklearn: Has many functions for implementing machine learning-related algorithms. It is the othe core library required for this program to function.
- matplotlib: A library to create various visual plots from provided data. This library isn't strictly required for this program to function, but it is required if you want to visualize some of the results (e.g. the ROC curves or the feature importance bar plot)
- numpy: A support library for the other three libraries above.

# Required Files
- test_data folder: This folder should be where all testing datasets to be used in analyzing grown random forests are located. 
- At least one training and one testing dataset csv files. These will serve as the basis for growing and analyzing random forests.
- A file to list all features. The three relevant columns are 'Var' (list feature names), 'Include' (Whether it is included in feature grouping. Use "YES" to indicate including it in a base feature set and "NO" to exclude it, and anything else for adding it to feature groups to be randomly selected), and 'Group' (You can designate what groups features are in. The algorithms will only select one feature per group at a time when building feature sets.)