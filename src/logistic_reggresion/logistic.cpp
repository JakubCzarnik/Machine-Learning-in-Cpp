#include "logistic.h"
#include <iostream>

LogisticReggresion::LogisticReggresion(double lr): n_features(), weights(), bias(){
    learning_rate = lr;
}


LogisticReggresion::~LogisticReggresion(){
}


void LogisticReggresion::fit(Eigen::MatrixXd X, Eigen::MatrixXd y, int epochs) {
   double loss = 0;

   weights = 0.01 * Eigen::MatrixXd::Random(X.cols(), y.cols());
   bias = Eigen::VectorXd::Zero(y.cols());

   // training
   for(int i = 0; i < epochs; i++){
      Eigen::MatrixXd predictions = X * weights;
      predictions.rowwise() += bias.transpose();

      // apply sigmoid function
      predictions = 1.0 / (1.0 + (-predictions).array().exp());
      Eigen::MatrixXd error = y - predictions;

      // d_loss/d_w
      Eigen::MatrixXd gradients_weights = -1.0 / X.rows() * X.transpose() * error;

      // d_loss/d_b
      Eigen::VectorXd gradients_bias = -1.0 / X.rows() * error.colwise().sum();

      // SGD
      weights -= learning_rate * gradients_weights;
      bias -= learning_rate * gradients_bias;

   }
}



Eigen::MatrixXd LogisticReggresion::predict(Eigen::MatrixXd X){
   Eigen::MatrixXd predictions = X * weights;
   predictions.rowwise() += bias.transpose();
   predictions = 1.0 / (1.0 + (-predictions).array().exp());
   return predictions;
}


