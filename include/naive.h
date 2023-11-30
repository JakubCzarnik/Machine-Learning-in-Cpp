#ifndef NAIVE
#define NAIVE

#include <iostream>
#include <algorithm>
#include <Eigen/Dense>

class NaiveBayes {
private:
    Eigen::VectorXd _classes;
    Eigen::MatrixXd _mean;
    Eigen::MatrixXd _var;
    Eigen::VectorXd _priors;

public:
    NaiveBayes();
    ~NaiveBayes();

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X);
};

#endif