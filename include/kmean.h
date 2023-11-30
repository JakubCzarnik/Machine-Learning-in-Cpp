#ifndef KMEAN
#define KMEAN

#include <vector>
#include <Eigen/Dense>
#include <iostream>

class KMeans{
   int k;
   Eigen::MatrixXd centroids;

   
public:
   KMeans(int k);
   ~KMeans();

   void fit(Eigen::MatrixXd X, int steps=100);
   Eigen::MatrixXd predict(const Eigen::MatrixXd& X);

   Eigen::MatrixXd get_centroids();

};


#endif