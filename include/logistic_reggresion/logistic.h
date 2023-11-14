#ifndef LOGISTIC_REGGRESION
#define LOGISTIC_REGGRESION

#include <vector>
#include <Eigen/Dense>
#include <iostream>

class LogisticReggresion{
   int n_features;
   double learning_rate;
   Eigen::MatrixXd weights; 
   Eigen::VectorXd bias;
   
public:
   LogisticReggresion(double lr = 0.01);
   ~LogisticReggresion();

   void fit(Eigen::MatrixXd X, Eigen::MatrixXd y, int epochs=500);
   Eigen::MatrixXd predict(Eigen::MatrixXd X);
};


#endif