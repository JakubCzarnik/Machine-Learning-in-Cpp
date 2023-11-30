#include "pca.h"

PCAnalysis::PCAnalysis(): data(), eigenvalues(), eigenvectors(){
}

PCAnalysis::~PCAnalysis(){
}



void PCAnalysis::fit(Eigen::MatrixXd X){
   // center the data
   Eigen::VectorXd mean = X.colwise().mean();
   X.rowwise() -= mean.transpose();

   // calculate covariance
   Eigen::MatrixXd covariance = (X.transpose() * X) / double(X.rows() - 1);

   // calculate eigen values and eigen vectiors
   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance);
   eigenvalues = eigen_solver.eigenvalues();
   eigenvectors = eigen_solver.eigenvectors();
}


Eigen::MatrixXd PCAnalysis::transform(Eigen::MatrixXd X, int n_components){
   if(n_components > eigenvectors.cols()){
      std::cout << "Error: Number of components is greater than the number of eigenvectors." << std::endl;
      return Eigen::MatrixXd();  // returns empty matrix
   }
   Eigen::MatrixXd selected_vectors = eigenvectors.rightCols(n_components);

   Eigen::MatrixXd transformed_data = X * selected_vectors;

   return transformed_data;
}

Eigen::VectorXd PCAnalysis::get_eigenvalues(){
   return eigenvalues;
}

Eigen::MatrixXd PCAnalysis::get_eigenvectors(){
   return eigenvectors;
}


