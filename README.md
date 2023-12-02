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
The [src](src/) folder contains the following implementations: 
- Supervised Learning
   - Classification
      - [Logistic Regression](src/logistic_reggresion/)
      - [K-Nearest Neighbors](src/knn/) (KNN)
      - [Naive Bayes](src/naive)
   - Regression:
      - [Linear Regression](src/linear_reggresion/)

- Unsupervised Learning
   - [Principal Component Analysis](src/pca/) (PCA)
   - [K-Means Clustering](src/kmean/) (K-Mean)
- Data Handling
   - [data_utils](src/data/) (additional tools for training algorithms)
   - [ dataframe](src/data/) (DataFrame class which has basic methods to load and preprocess the data.)

Each algorithm is described in its own folder.

Additionally, there are implementations of basic data pipelines, metrics, and examples of usage of these algorithms on the datasets listed above.


## License
The project is licensed under the [MIT License](LICENSE)
