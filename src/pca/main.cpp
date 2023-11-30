#include "pca.h"
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
   

   // Show first 5 rows
   train_ds.head();


   // preprocess the train independent features
   std::vector<std::vector<double>> X_train_vec = train_ds.get_slice(0, -1, 0, 11); // all rows, first 11 columns
   Eigen::MatrixXd X_train = vector_to_matrix(X_train_vec);


   // fit the PCA and transform train data
   PCAnalysis pca = PCAnalysis();
   pca.fit(X_train); 
   Eigen::MatrixXd comp =  pca.transform(X_train, 2);
   

   // print the variance (eigenvalues) of the first two components
   Eigen::VectorXd eigenvalues = pca.get_eigenvalues();
   Eigen::MatrixXd eigenvectors = pca.get_eigenvectors();
   double total_variance = eigenvalues.sum();
   Eigen::VectorXd explained_variance_ratio = eigenvalues / total_variance;

   // std::cout << "values" << std::endl;
   // std::cout << eigenvalues << std::endl;
   // std::cout << "vectors" << std::endl;
   // std::cout << eigenvectors << std::endl;

   std::cout << "Percentage of variance of the first component: " 
            << explained_variance_ratio[explained_variance_ratio.size()-1] * 100 << "%" << std::endl;
   std::cout << "Percentage of variance of the second component: " 
            << explained_variance_ratio[explained_variance_ratio.size()-2] * 100 << "%" << std::endl;
   return 0;
}
