#ifndef LINEAR_REGGRESION
#define LINEAR_REGGRESION

#include <vector>
#include <Eigen/Dense>

class LinearReggresion{
   int n_features;
   double learning_rate;
   Eigen::MatrixXd weights; 
   Eigen::VectorXd bias;
   
public:
   LinearReggresion(double lr = 0.01); // using Gradient Descent 
   ~LinearReggresion();

   void fit(Eigen::MatrixXd X, Eigen::MatrixXd y, int epochs=500);
   Eigen::MatrixXd predict(Eigen::MatrixXd X);


};


#endif