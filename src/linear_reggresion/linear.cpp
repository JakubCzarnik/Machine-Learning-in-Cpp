#include "linear.h"


LinearReggresion::LinearReggresion(double lr): n_features(), weights(), bias(){
   learning_rate = lr;
}


LinearReggresion::~LinearReggresion(){
}

void LinearReggresion::fit(Eigen::MatrixXd X, Eigen::MatrixXd y, int epochs) {
   double loss = 0;

   weights = 0.01 * Eigen::MatrixXd::Random(X.cols(), y.cols());
   bias = Eigen::VectorXd::Zero(y.cols());

   // training
   for(int i = 0; i < epochs; i++){
      Eigen::MatrixXd predictions = X * weights;
      predictions.rowwise() += bias.transpose();

      Eigen::MatrixXd error = y - predictions;

      // d_loss/d_w
      Eigen::MatrixXd gradients_weights = -2.0 / X.rows() * X.transpose() * error; // mse derivative 
      // d_loss/d_b
      Eigen::VectorXd gradients_bias = -2.0 / X.rows() * error.colwise().sum();
      // SGD
      weights -= learning_rate * gradients_weights;
      bias -= learning_rate * gradients_bias;
   }
}


Eigen::MatrixXd LinearReggresion::predict(Eigen::MatrixXd X){
   Eigen::MatrixXd predictions = X * weights;
   predictions.rowwise() += bias.transpose();
   return predictions;
}


