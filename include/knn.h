#ifndef KNN
#define KNN


#include <unordered_map>
#include <queue>
#include <Eigen/Dense> 

struct DistIndex {
   DistIndex();
   ~DistIndex();

   double dist;
   int index;
   bool operator<(const DistIndex& other) const;
};

 
class KNeighborsClassifier{
   int k;
   Eigen::MatrixXd X_train;
   Eigen::VectorXd y_train;


public:
   KNeighborsClassifier(int k);
   ~KNeighborsClassifier();

   void fit(Eigen::MatrixXd X, Eigen::VectorXd y);
   Eigen::VectorXd predict(Eigen::MatrixXd X);

};


#endif