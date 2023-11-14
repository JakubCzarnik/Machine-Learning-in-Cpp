#include "knn.h"

// DistIndex
DistIndex::DistIndex(){
}


DistIndex::~DistIndex(){
}


bool DistIndex::operator<(const DistIndex& other) const{
   return dist < other.dist;
}

// KNeighborsClassifier


KNeighborsClassifier::KNeighborsClassifier(int k ): k(k){
}


KNeighborsClassifier::~KNeighborsClassifier(){
}


void KNeighborsClassifier::fit(Eigen::MatrixXd X, Eigen::MatrixXd y){
   assert(y.cols()==1); 
   X_train = X;
   y_train = y;
}

Eigen::MatrixXd KNeighborsClassifier::predict(Eigen::MatrixXd X){
   Eigen::MatrixXd y_pred(X.rows(), 1);

   for(int i = 0; i < X.rows(); ++i){ // every test/val row with every train row
      std::priority_queue<DistIndex> pq;
      
      for(int j = 0; j < X_train.rows(); ++j){
         // to reach the minimum heap, we use the trick with opposite values
         double dist = -(X.row(i) - X_train.row(j)).norm(); 
         DistIndex di;
         di.dist = dist;
         di.index = j;
         if(pq.size() < k || dist < pq.top().dist){
            pq.push(di);
            if(pq.size() > k) {
               pq.pop();
            }
         }
      }
      // count "votes"
      std::unordered_map<double, int> counts;
      while(!pq.empty()) {
         counts[y_train(pq.top().index, 0)]++;
         pq.pop();
      }
      // choose the class with the most "votes"
      double max_count = 0;
      double pred_class = -1;
      for(const auto& pair : counts) {
         if(pair.second > max_count) {
            max_count = pair.second;
            pred_class = pair.first;
         }
      }
      y_pred(i, 0) = pred_class;
   }
   return y_pred;
}