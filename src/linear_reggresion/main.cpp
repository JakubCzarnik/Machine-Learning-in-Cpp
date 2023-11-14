#include "linear.h"
#include "data_utils.h"

int main(){
   // Load data from csv
   CSV_Handler dataset = CSV_Handler();
   dataset.read_csv("E:/Vs Code Scripts/cpp/Machine-Learning-in-Cpp/datasets/Red Wine Quality/winequality-red.csv");


   dataset.head();
   std::array<CSV_Handler, 2> split = split_data(dataset, 0.2);
   CSV_Handler train_ds = split[0];
   CSV_Handler val_ds = split[1];
   // Show first 5 rows
   //train_ds.head();
   val_ds.head();

   // train data
   std::vector<std::vector<double>> X_train_vec = train_ds.get_columns_values(0, 11);
   std::vector<std::vector<double>> y_train_vec = train_ds.get_columns_values(11, 12);
   Eigen::MatrixXd X_train = vector_to_matrix(X_train_vec);
   Eigen::MatrixXd y_train = vector_to_matrix(y_train_vec);
   X_train = standarize_cols(X_train);

   // validation data
   std::vector<std::vector<double>> X_val_vec = val_ds.get_columns_values(0, 11);
   std::vector<std::vector<double>> y_val_vec = val_ds.get_columns_values(11, 12);
   Eigen::MatrixXd X_val = vector_to_matrix(X_val_vec);
   Eigen::MatrixXd y_val = vector_to_matrix(y_val_vec);
   X_val = standarize_cols(X_val);

   // fit the Linear Reggresion model
   LinearReggresion linear = LinearReggresion(0.008);
   linear.fit(X_train, y_train, 300);

   // calculate predictions and loss
   Eigen::MatrixXd y_pred = linear.predict(X_val);

   double mse = mean_squared_error(y_val, y_pred);
   // show results
   int n_show = 5;
   Eigen::MatrixXd combined(n_show, 2);
   combined << y_val.block(0, 0, n_show, y_val.cols()), y_pred.block(0, 0, n_show, y_pred.cols());

   std::cout << "    True" << " " << "Predicted" << std::endl;
   std::cout << combined << std::endl;
   std::cout << "\nMean Squared Error: " << mse << std::endl;
   
   return 0;
}
