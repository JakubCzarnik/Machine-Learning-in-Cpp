#include "logistic.h"


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

      Eigen::MatrixXd residuals = y - predictions;
      Eigen::MatrixXd squared_residuals = residuals.array().square();
      // d_loss/d_w
      Eigen::MatrixXd gradients_weights = -2.0 / X.rows() * X.transpose() * residuals;
      // d_loss/d_b
      Eigen::VectorXd gradients_bias = -2.0 / X.rows() * residuals.colwise().sum();
      // SGD
      weights -= learning_rate * gradients_weights;
      bias -= learning_rate * gradients_bias;

   }
}


Eigen::MatrixXd LogisticReggresion::predict(Eigen::MatrixXd X){
   Eigen::MatrixXd predictions = X * weights;
   predictions.rowwise() += bias.transpose();
   return predictions;
}


