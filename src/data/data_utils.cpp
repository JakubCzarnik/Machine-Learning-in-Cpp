#include "data_utils.h"

std::array<CSV_Handler, 2> split_data(CSV_Handler dataset, float val_size){
   if(val_size > 1 || val_size < 0){
      std::cout << "Please choose valid val_size: " << val_size << " in split_data method." << std::endl;
      exit(1);
   }
   if(dataset.get_shape()[0] == 0){
      std::cout << "Cannot split empty dataset." << std::endl;
      exit(1);
   }
   std::vector<std::vector<double>> dataset_values = dataset.get_data();

   std::vector<std::vector<double>> train_values;
   std::vector<std::vector<double>> validation_values;
   std::array<CSV_Handler, 2> splited_datasets; // {train, val}
   int train_split_size = static_cast<int>(dataset.get_shape()[0] * (1.0 - val_size));
   std::vector<int> indices(dataset.get_shape()[0]);

   // create and shuffle indices
   std::iota(indices.begin(), indices.end(), 0);
   std::random_device rd;
   std::mt19937 g(rd());
   std::shuffle(indices.begin(), indices.end(), g);
   // split the dataset
   for (int i = 0; i < train_split_size; ++i) {
      train_values.push_back(dataset_values[indices[i]]);
   }
   for (int i = train_split_size; i < indices.size(); ++i) {
      validation_values.push_back(dataset_values[indices[i]]);
   }

   splited_datasets[0].from_array(train_values, dataset.get_columns_names());
   splited_datasets[1].from_array(validation_values, dataset.get_columns_names());
   return splited_datasets;
}


Eigen::MatrixXd vector_to_matrix(std::vector<std::vector<double>> array){
   for (size_t i = 1; i < array.size(); ++i) {
      if (array[i].size() != array[0].size()) {
         std::cerr << "Row " << i << " has " << array[i].size() << " columns, but the first row has " << array[0].size() << " columns.\n";
         exit(1);
      }
   }

   Eigen::MatrixXd mat(array.size(), array[0].size());

   for (int i = 0; i < array.size(); i++) {
      mat.row(i) = Eigen::VectorXd::Map(&array[i][0], array[0].size());
   }
   return mat;
}


Eigen::MatrixXd standarize_cols(Eigen::MatrixXd mat) {
   for (int i = 0; i < mat.cols(); ++i) {
      double mean = mat.col(i).mean();
      double stddev = std::sqrt((mat.col(i).array() - mean).square().sum() / mat.rows());
      mat.col(i) = (mat.col(i).array() - mean) / stddev;
   }
   return mat;
}


double mean_squared_error(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred){
   assert(y_true.rows() == y_pred.rows() && y_true.cols() == y_pred.cols());
   
   Eigen::MatrixXd diff = y_true - y_pred;
   Eigen::MatrixXd diff_squared = diff.array().square();

   return diff_squared.sum() / (diff_squared.rows() * diff_squared.cols());
}


double classification_accuracy(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred){
   assert(y_true.rows() == y_pred.rows() && y_true.cols() == y_pred.cols());

   // make sure that type of values is int
   Eigen::MatrixXi y_true_int = y_true.cast<int>();
   Eigen::MatrixXi y_pred_int = y_pred.cast<int>();

   int correct_predictions = (y_true_int.array() == y_pred_int.array()).count();

   return static_cast<double>(correct_predictions) / y_true.size();
}