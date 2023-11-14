#include "csv_handler.h"


CSV_Handler::CSV_Handler(): data(), shape(), column_names(){
}


CSV_Handler::~CSV_Handler(){

}



void CSV_Handler::read_csv(std::string path){
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
   shape[0] = data.size();
   shape[1] = data[0].size();
   std::cout << "Successfully read csv." << std::endl;
}


void CSV_Handler::from_array(std::vector<std::vector<double>> values, std::vector<std::string> col_names){
   for (size_t i = 1; i < values.size(); ++i) {
      if (values[i].size() != values[0].size()) {
         std::cerr << "Row " << i << " has " << values[i].size() << " columns, but the first row has " << values[0].size() << " columns.\n";
         exit(1);
      }
   }

   data = values;
   column_names = col_names;
   
   shape[0] = values.size();
   shape[1] = values[0].size();
}


std::array<int, 2> CSV_Handler::get_shape(){
   return shape;
}


void CSV_Handler::head(int n, int max_cols){
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
   std::cout << "CSV of " <<"shape (" << shape[0] << ", " << shape[1]<<")" << std::endl << std::endl;
}



void CSV_Handler::delete_row(int index){
   if(index >= 0 && index < data.size()){
      data.erase(data.begin()+index);
      shape[0] = shape[0] - 1;
   }
   else{
      std::cout << "Index: " << index << " is out of data size: " << data.size() << std::endl;
   }
}


std::vector<std::vector<double>> CSV_Handler::get_data(){
   return data;
}


std::vector<std::vector<double>> CSV_Handler::get_slice(int col_st, int col_end, int row_st, int row_end){
   if (col_st < 0) col_st = data.size() + col_st;
   if (col_end < 0) col_end = data.size() + col_end;
   if (row_st < 0) row_st = data[0].size() + row_st;
   if (row_end < 0) row_end = data[0].size() + row_end;

   std::vector<std::vector<double>> slice;

   for (int i = col_st; i < col_end; i++){
      std::vector<double> column_slice(data[i].begin() + row_st, data[i].begin() + row_end);
      slice.push_back(column_slice);
   }
   return slice;
}


std::vector<std::string> CSV_Handler::get_columns_names(){
   return column_names;
}


void CSV_Handler::create_dummy(int column_index){
   std::vector<double> column = data[column_index];
   std::set<double> categories(column.begin(), column.end());
   
   for(auto& category : categories){
      std::vector<double> dummy_column;
      for(auto& value : column){
         if(value == category)
            dummy_column.push_back(1.0);
         else
            dummy_column.push_back(0.0);
      }
      data.push_back(dummy_column);
      column_names.push_back(std::to_string(category) + "_" + column_names[column_index]);
   }
   data.erase(data.begin() + column_index);
   column_names.erase(column_names.begin() + column_index);
}


void CSV_Handler::unique(){
   std::cout << "Number of unique values per column:\n";
   for (int i = 0; i < data.size(); i++) {
      std::set<double> unique_values(data[i].begin(), data[i].end());
      std::cout << std::setw(8) << column_names[i] << ": " << unique_values.size() << "\n";
   }
}


void CSV_Handler::clear(){
   data.clear();
   column_names.clear();
   shape.fill(0);
}