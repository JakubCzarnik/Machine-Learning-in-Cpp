#ifndef PCA
#define PCA

#include <vector>
#include <Eigen/Dense>
#include <iostream>

class PCAnalysis{
private:
   Eigen::MatrixXd data;
   Eigen::VectorXd eigenvalues;
   Eigen::MatrixXd eigenvectors;

public:
   PCAnalysis();
   ~PCAnalysis();

   void fit(Eigen::MatrixXd X);
   Eigen::MatrixXd transform(Eigen::MatrixXd X, int n_components);

   Eigen::VectorXd get_eigenvalues();
   Eigen::MatrixXd get_eigenvectors();

};

#endif
