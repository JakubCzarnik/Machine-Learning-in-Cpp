#include "kmean.h"


KMeans::KMeans(int k): k(k), centroids() {
}
 
KMeans::~KMeans(){
}

void KMeans::fit(Eigen::MatrixXd X, int steps){
   centroids = Eigen::MatrixXd::Zero(k, X.cols());
   
   for(int i = 0; i < k; i++){
      centroids.row(i) = X.row(rand() % X.rows());
   }

   for(int step = 0; step < steps; step++){
      Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k, X.cols());
      Eigen::VectorXi counts = Eigen::VectorXi::Zero(k);

      for(int i = 0; i < X.rows(); i++){
         int closest_centroid = 0;
         double closest_distance = (X.row(i) - centroids.row(0)).norm();

         for(int j = 1; j < k; j++){
            double distance = (X.row(i) - centroids.row(j)).norm();
            if(distance < closest_distance){
               closest_centroid = j;
               closest_distance = distance;
            }
         }
         new_centroids.row(closest_centroid) += X.row(i);
         counts(closest_centroid)++;
      }
      for(int i = 0; i < k; i++){
         if(counts(i) != 0){
            new_centroids.row(i) /= counts(i);
         }
      }
      centroids = new_centroids;
   }
}

Eigen::MatrixXd KMeans::predict(const Eigen::MatrixXd& X){
   Eigen::VectorXi labels_vector(X.rows());
   Eigen::MatrixXd labels_matrix(X.rows(), 1);

   for(int i = 0; i < X.rows(); i++){

      int closest_centroid = 0;
      double closest_distance = (X.row(i) - centroids.row(0)).norm();

      for(int j = 1; j < k; j++){
         double distance = (X.row(i) - centroids.row(j)).norm();
         if(distance < closest_distance){
            closest_centroid = j;
            closest_distance = distance;
         }
      }

      labels_vector(i) = closest_centroid;
   }

   labels_matrix = labels_vector.cast<double>();

   return labels_matrix;
}

Eigen::MatrixXd KMeans::get_centroids(){
   return centroids;
}
