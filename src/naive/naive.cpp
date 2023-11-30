#include "naive.h"

NaiveBayes::NaiveBayes() {
}

NaiveBayes::~NaiveBayes() {
}

void NaiveBayes::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y){
    // find all unique classes
    _classes = y;
    std::sort(_classes.data(), _classes.data() + _classes.size());
    _classes.conservativeResize(std::unique(_classes.data(), _classes.data() + _classes.size()) - _classes.data());

    // init variables
    _mean.resize(_classes.size(), X.cols());
    _var.resize(_classes.size(), X.cols());
    _priors.resize(_classes.size());

    // for each class calc the mean, variance, and prior probability
    for(int i = 0; i < _classes.size(); i++) {
        double c = _classes[i];

        // mask for class c
        Eigen::Array<bool, Eigen::Dynamic, 1> mask = (y.array() == c);
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> mask_matrix = mask.replicate(1, X.cols());

        // calc the mean and variance for class c
        _mean.row(i) = (mask_matrix.select(X, 0).colwise().sum() / mask.count()).matrix();
        _var.row(i) = (mask_matrix.select((X.rowwise() - _mean.row(i)).array().square(), 0).colwise().sum() / mask.count()).matrix();

        // calc the prior probability for class c
        _priors[i] = static_cast<double>(mask.count()) / y.size();
    }
}

Eigen::VectorXd NaiveBayes::predict(const Eigen::MatrixXd& X){
    Eigen::MatrixXd probabilities(X.rows(), _classes.size());

    for(int i = 0; i < _classes.size(); i++){
        // calc the diff and create var matrix
        Eigen::MatrixXd diff = X.rowwise() - _mean.row(i);
        Eigen::MatrixXd diff_square = diff.array().square().matrix();
        Eigen::MatrixXd var_sqrt_replicated = _var.row(i).array().sqrt().replicate(diff.rows(), 1);

        // calc the exponent
        Eigen::MatrixXd division = diff_square.array() / (2 * var_sqrt_replicated.array());
        Eigen::MatrixXd exp_term = (-division.array()).exp();
        // calculate the Gaussian probability density function
        Eigen::VectorXd temp1 = (1.0 / (std::sqrt(2 * M_PI) * var_sqrt_replicated.array())).matrix().rowwise().prod();
        Eigen::VectorXd temp2 = exp_term.rowwise().prod();
        Eigen::VectorXd prob = temp1.array() * temp2.array();

        probabilities.col(i) = prob * _priors[i];
    }

    Eigen::VectorXd predictions(X.rows());
    for(int i = 0; i < X.rows(); i++){
        // find the class with the highest probability for each example
        Eigen::VectorXd::Index max_index;
        probabilities.row(i).maxCoeff(&max_index);
        predictions[i] = _classes[max_index];
    }
    return predictions;
}