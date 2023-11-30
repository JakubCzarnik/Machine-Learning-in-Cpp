#include "logistic.h"
#include "data_utils.h"


int main(){
   // Load data from csv
   DataFrame dataset = DataFrame();
   dataset.read_csv("E:/datasets/Heart Attack/heart.csv");
   

   // preprocess dataset
   dataset.head(5, 14); // show first 5 rows and 14 cols (first 7 and last 7)
   
   dataset.create_dummy("cp"); // creates m-1 dummies to avoid dummy variable trap
   dataset.create_dummy("restecg"); // -;;-
   dataset.create_dummy("slp");
   dataset.create_dummy("caa");
   dataset.create_dummy("thall");
   
   dataset.standarize_cols(0, 1); // age
   dataset.standarize_cols(2, 4); // trtbps, chol
   dataset.standarize_cols(5, 6); // thalachh
   dataset.standarize_cols(7, 8); // oldpeak

   dataset.head(); 
   dataset.unique();


   // split the data
   std::array<DataFrame, 2> split = split_data(dataset, 0.2);
   DataFrame train_ds = split[0];
   DataFrame val_ds = split[1];
   train_ds.head(); 
   val_ds.head();
   

   // preproces the train data
   std::vector<std::vector<double>> y_train_vec = train_ds.drop_col("output");
   std::vector<std::vector<double>> X_train_vec = train_ds.get_data();
   Eigen::MatrixXd X_train = vector_to_matrix(X_train_vec);
   Eigen::MatrixXd y_train = vector_to_matrix(y_train_vec);

 
   // preproces the validation data
   std::vector<std::vector<double>> y_val_vec = val_ds.drop_col("output");
   std::vector<std::vector<double>> X_val_vec = val_ds.get_data();
   Eigen::MatrixXd y_val = vector_to_matrix(y_val_vec);
   Eigen::MatrixXd X_val = vector_to_matrix(X_val_vec);
   

   // fit the Linear Reggresion model
   double learning_rate = 0.008;
   int epochs = 300;
   LogisticReggresion logistic = LogisticReggresion(learning_rate);
   logistic.fit(X_train, y_train, epochs);


   // calculate predictions and metrics
   Eigen::MatrixXd y_pred = logistic.predict(X_val);

   y_pred = binary_threshold(y_pred); // 1 where element > 0.5 else 0
   double accuracy = classification_accuracy(y_val, y_pred);


   // show results
   int n_show = 10;
   Eigen::MatrixXd combined(n_show, 2);
   combined << y_val.block(0, 0, n_show, y_val.cols()), y_pred.block(0, 0, n_show, y_pred.cols());

   std::cout << "True" << " " << "Predicted" << std::endl;
   std::cout << combined << std::endl;
   std::cout << "\nAccuracy: " << accuracy << std::endl;
   return 0;
}
