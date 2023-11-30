#ifndef DATA_UTILS
#define DATA_UTILS

#include "dataframe.h"
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <Eigen/Dense> 


std::array<DataFrame, 2> split_data(DataFrame dataset, float val_size);

Eigen::MatrixXd vector_to_matrix(std::vector<std::vector<double>> array);

double mean_squared_error(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);

double mean_error(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);

double classification_accuracy(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);

Eigen::MatrixXd binary_threshold(Eigen::MatrixXd matrix);

#endif