# Machine Learning in C++
![C++](https://img.shields.io/badge/c++14-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Overview
This repository contains implementations of basic machine learning algorithms written in C++ 14, along with examples of their usage on datasets such as:

- [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
- [Iris Species](https://www.kaggle.com/datasets/uciml/iris)
- [Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

Please ensure that all data is preprocessed into numeric form before use (especially for Iris species), that the Eigen library is installed, and that the path in the CMakeLists is correctly set to the Eigen library location.

## Algorithms
The src folder contains the following implementations: 
- Supervised Learning
   - Classification
      1. Logistic Regression
      2. K-Nearest Neighbors (KNN)
      3. Naive Bayes
   - Regression:
      1. Linear Regression
- Unsupervised Learning
   1. Principal Component Analysis (PCA)
   2. K-Means Clustering (K-Mean)
- Data Handling
   1. data_utils (additional tools for training algorithms)
   2. dataframe (DataFrame class which has basic methods to load and preprocess the data.)

Each algorithm is described in its own folder.

Additionally, there are implementations of basic data pipelines, metrics, and examples of usage of these algorithms on the datasets listed above.


## License
The project is licensed under the [MIT License](LICENSE)