#include "kmean.h"
#include "data_utils.h"


int main(){
   // Load data from csv
   DataFrame df = DataFrame();
   df.read_csv("E:/datasets/Iris dataset/IRIS.csv");
   df.head();

   // preprocess the data
   df.drop_col("petal_length");
   df.drop_col("petal_width");
   df.standarize_cols(0, df.get_shape()[1]-1);

   std::array<DataFrame, 2> split = split_data(df, 0.2);
   DataFrame train_ds = split[0];
   DataFrame val_ds = split[1];

   // Show first 5 rows
   df.head(); // should be only 2 independent variables + target variable
   train_ds.head();
   val_ds.head();

   // train data
   std::vector<std::vector<double>> X_train_vec = train_ds.get_slice(0, -1, 0, 2); // all rows, first 2 columns
   std::vector<std::vector<double>> y_train_vec = train_ds.get_slice(0, -1, 2, 3); // all rows, the last one column
   
   Eigen::MatrixXd X_train = vector_to_matrix(X_train_vec);
   Eigen::MatrixXd y_train = vector_to_matrix(y_train_vec);


   // validation data
   std::vector<std::vector<double>> X_val_vec = val_ds.get_slice(0, -1, 0, 2); // all rows, first 2 columns
   std::vector<std::vector<double>> y_val_vec = val_ds.get_slice(0, -1, 2, 3); // all rows, the last one column
   
   Eigen::MatrixXd X_val = vector_to_matrix(X_val_vec);
   Eigen::MatrixXd y_val = vector_to_matrix(y_val_vec);


   // fit the Linear Reggresion model
   int n_clusters = 3;
   int steps = 50;

   KMeans kmean = KMeans(n_clusters);
   kmean.fit(X_train, steps);

   std::cout << "Centroids" << std::endl;
   std::cout << kmean.get_centroids() << std::endl;
  
   // predictions on unseen data
   Eigen::MatrixXd y_pred = kmean.predict(X_val);

   double accuracy = classification_accuracy(y_val, y_pred);
   double me = mean_error(y_val, y_pred);

   // show
   int n_show = 5;
   Eigen::MatrixXd combined(n_show, 2);
   combined << y_val.block(0, 0, n_show, y_val.cols()), y_pred.block(0, 0, n_show, y_pred.cols());

   std::cout << "True" << " " << "Predicted" << std::endl;
   std::cout << combined << std::endl;
   std::cout << "\nAccuracy: " << accuracy << std::endl; 
   std::cout << "Mean error: " << me << std::endl;
   // for classification problem, in this case clustering is not the best option 
   return 0;
}

