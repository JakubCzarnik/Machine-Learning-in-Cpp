#include "dataframe.h"


DataFrame::DataFrame(): data(), column_names(){
}


DataFrame::~DataFrame(){

}



void DataFrame::read_csv(std::string path){
   std::ifstream file(path);

   if (!file.is_open()) {
      std::cout << "Could not open the file: " << path << std::endl;
   }

   std::string line;
   // col names
   std::getline(file, line); 
   std::stringstream ss(line);
   std::string cell;
   while (std::getline(ss, cell, ',')) {
      column_names.push_back(cell);
   }
   // features
   std::vector<std::vector<double>> columns(column_names.size());
   while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string cell;
      int col = 0;
      while (std::getline(ss, cell, ',')) {
         if (col >= column_names.size()) {
            std::cerr << "Row has more columns than the header row." << std::endl;
            exit(1);
         }
         columns[col].push_back(std::stod(cell));
         col++;
      }
   }
   file.close();
   data = columns;
   std::cout << "Successfully read csv." << std::endl;
}


void DataFrame::from_array(std::vector<std::vector<double>> values, std::vector<std::string> col_names){
   for (size_t i = 1; i < values.size(); ++i) {
      if (values[i].size() != values[0].size()) {
         std::cerr << "Row " << i << " has " << values[i].size() << " columns, but the first row has " << values[0].size() << " columns.\n";
         exit(1);
      }
   }
   // rows, col -> cols, rows
   std::vector<std::vector<double>> transposed_values(values[0].size(), std::vector<double>(values.size()));
   for (size_t i = 0; i < values.size(); ++i){
      for (size_t j = 0; j < values[0].size(); ++j){
         transposed_values[j][i] = values[i][j];
      }
   }

   data = transposed_values;
   column_names = col_names;
}


std::array<int, 2> DataFrame::get_shape(){
   std::array<int, 2> shape = {static_cast<int>(data[0].size()), static_cast<int>(data.size())};;
   return shape;
}


void DataFrame::head(int n, int max_cols){
   if(n > data[0].size()){
      n = data[0].size(); 
   }

   // column names
   if(data.size() <= max_cols) {
      for (int i = 0; i < data.size(); i++) {
         std::cout << std::setw(8) << column_names[i] << "; ";
      }
   } else {
      for (int i = 0; i < 4; i++) {
         std::cout << std::setw(8) << column_names[i] << "; ";
      }
      std::cout << " ... ";
      for (int i = data.size() - 4; i < data.size(); i++) {
         std::cout << std::setw(8) << column_names[i] << "; ";
      }
   }
   std::cout << std::endl;
   
   // values
   for (int j = 0; j < n; j++) {
      if(data.size() <= max_cols) {
         for (int i = 0; i < data.size(); i++) {
            std::cout << std::setw(6) << data[i][j] << "   ";
         }
      } else {
         for (int i = 0; i < 4; i++) {
            std::cout << std::setw(6) << data[i][j] << "   ";
         }
         std::cout << " ... ";
         for (int i = data.size() - 4; i < data.size(); i++) {
            std::cout << std::setw(6) << data[i][j] << "   ";
         }
      }
      std::cout << std::endl;
   }
   std::cout << "DataFrame of " <<"shape (" << get_shape()[0] << ", " << get_shape()[1]<<")" << std::endl << std::endl;
}


std::vector<std::vector<double>> DataFrame::drop_col(std::string col_name){
   int col_index = -1;

   for(int i = 0; i < column_names.size(); i++){
      if (column_names[i] == col_name){
         col_index = i;
         break;
      }
   }

   if(col_index == -1){
      std::cout << "Column doesn't exist" << std::endl;
      exit(1);
   }

   std::vector<std::vector<double>> dropped_col = get_slice(0, -1, col_index, col_index + 1);
   data.erase(data.begin() + col_index);
   column_names.erase(column_names.begin() + col_index);

   return dropped_col;
}


void DataFrame::delete_row(int index){
   if(index >= 0 && index < data[0].size()){
      for(auto& column : data){
         column.erase(column.begin() + index);
      }
   }
   else{
      std::cout << "Index: " << index << " is out of data size: " << data[0].size() << std::endl;
   }
}


std::vector<std::vector<double>> DataFrame::get_data(){
   return get_slice(0, -1, 0, -1);
}


std::vector<std::vector<double>> DataFrame::get_slice(int row_st, int row_end, int col_st, int col_end){
   if (col_st < 0) col_st = data.size() + col_st+1;
   if (col_end < 0) col_end = data.size() + col_end+1;
   if (row_st < 0) row_st = data[0].size() + row_st+1;
   if (row_end < 0) row_end = data[0].size() + row_end+1;
   
   std::vector<std::vector<double>> slice;
   for (int i = col_st; i < col_end; i++){
      std::vector<double> column_slice(data[i].begin() + row_st, data[i].begin() + row_end);
      slice.push_back(column_slice);
   }

   // cols, rows -> rows, cols
   std::vector<std::vector<double>> transposed_slice(slice[0].size(), std::vector<double>(slice.size()));
   for (size_t i = 0; i < slice.size(); ++i){
      for (size_t j = 0; j < slice[0].size(); ++j){
         transposed_slice[j][i] = slice[i][j];
      }
   }

   return transposed_slice;
}


std::vector<std::string> DataFrame::get_columns_names(){
   return column_names;
}


void DataFrame::create_dummy(std::string column_name){
   int column_index = std::distance(column_names.begin(), std::find(column_names.begin(), column_names.end(), column_name));

   std::vector<double> column = data[column_index];
   std::set<double> categories(column.begin(), column.end());
   int new_column_index = 1;
   for(auto it = categories.begin(); it != std::prev(categories.end()); ++it){
      double category = *it;
      std::vector<double> dummy_column;
      for(auto& value : column){
         if(value == category)
            dummy_column.push_back(1.0);
         else
            dummy_column.push_back(0.0);
      }
      data.push_back(dummy_column);
      column_names.push_back("col_" + std::to_string(new_column_index) + "_" + column_names[column_index]);
      new_column_index++;
   }
   data.erase(data.begin() + column_index);
   column_names.erase(column_names.begin() + column_index);
}


void DataFrame::unique(){
   std::cout << "Number of unique values per column:\n";
   for (int i = 0; i < data.size(); i++) {
      std::set<double> unique_values(data[i].begin(), data[i].end());
      std::cout << std::setw(8) << column_names[i] << ": " << unique_values.size() << std::endl;
   }
   std::cout << std::endl;
}


void DataFrame::standarize_cols(int col_st, int col_end){
   for(int i = col_st; i < col_end; i++){
      double sum = 0.0, sum_sq = 0.0;
      int n = data[i].size();
      for(int j = 0; j < n; j++){
         sum += data[i][j];
         sum_sq += data[i][j] * data[i][j];
      }
      double mean = sum / n;
      double std_dev = std::sqrt(sum_sq/n - mean*mean);

      for(int j = 0; j < n; j++){
         data[i][j] = (data[i][j] - mean) / std_dev;
      }
   }
}


void DataFrame::clear(){
   data.clear();
   column_names.clear();
}