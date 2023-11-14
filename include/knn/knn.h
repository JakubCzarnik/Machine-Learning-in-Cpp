#ifndef KNN
#define KNN

#include "csv_handler.h"
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
   Eigen::MatrixXd y_train;


public:
   KNeighborsClassifier(int k);
   ~KNeighborsClassifier();

   void fit(Eigen::MatrixXd X, Eigen::MatrixXd y);
   Eigen::MatrixXd predict(Eigen::MatrixXd X);

};


#endif