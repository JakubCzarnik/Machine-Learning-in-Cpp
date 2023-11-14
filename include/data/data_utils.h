#ifndef DATA_UTILS
#define DATA_UTILS

#include "csv_handler.h"
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <Eigen/Dense> 


std::array<CSV_Handler, 2> split_data(CSV_Handler dataset, float val_size);

Eigen::MatrixXd vector_to_matrix(std::vector<std::vector<double>> array);

Eigen::MatrixXd standarize_cols(Eigen::MatrixXd mat);

double mean_squared_error(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);

double classification_accuracy(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);

#endif