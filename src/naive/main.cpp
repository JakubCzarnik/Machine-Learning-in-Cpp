#include "naive.h"
#include "data_utils.h"

int main(){
   // Load data from csv
   DataFrame df = DataFrame();
   df.read_csv("E:/datasets/Red Wine Quality/winequality-red.csv");
   df.head();


   // standarize all columns and split the data
   df.standarize_cols(0, df.get_shape()[1]-1); // without target

   std::array<DataFrame, 2> split = split_data(df, 0.2);
   DataFrame train_ds = split[0];
   DataFrame val_ds = split[1];
   

   // Show first 5 rows
   train_ds.head();
   val_ds.head();


   // preprocess the train data
   std::vector<std::vector<double>> X_train_vec = train_ds.get_slice(0, -1, 0, 11); // all rows, first 11 columns
   std::vector<std::vector<double>> y_train_vec = train_ds.get_slice(0, -1, 11, 12); // all rows, the last one column
   
   Eigen::MatrixXd X_train = vector_to_matrix(X_train_vec);
   Eigen::MatrixXd y_train = vector_to_matrix(y_train_vec);
   Eigen::VectorXd y_train_vector = Eigen::VectorXd::Map(y_train.data(), y_train.size());


   // preprocess the validation data
   std::vector<std::vector<double>> X_val_vec = val_ds.get_slice(0, -1, 0, 11); // all rows, first 11 columns
   std::vector<std::vector<double>> y_val_vec = val_ds.get_slice(0, -1, 11, 12); // all rows, the last one column
   
   Eigen::MatrixXd X_val = vector_to_matrix(X_val_vec);
   Eigen::MatrixXd y_val = vector_to_matrix(y_val_vec);


   // fit the Naive Bayes model
   NaiveBayes nb = NaiveBayes();
   nb.fit(X_train, y_train_vector); 


   // calculate predictions and mean error
   Eigen::VectorXd y_pred_vec = nb.predict(X_val);
   Eigen::MatrixXd y_pred = y_pred_vec;

   double accuracy = classification_accuracy(y_val, y_pred);
   double me = mean_error(y_val, y_pred);


   // show results
   int n_show = 5;
   Eigen::MatrixXd combined(n_show, 2);
   combined << y_val.block(0, 0, n_show, y_val.cols()), y_pred.block(0, 0, n_show, y_pred.cols());

   std::cout << "True" << " " << "Predicted" << std::endl;
   std::cout << combined << std::endl;
   std::cout << "\nAccuracy: " << accuracy << std::endl;
   std::cout << "Mean error: " << me << std::endl;
   
   return 0;
}
