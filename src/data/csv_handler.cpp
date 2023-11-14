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
   int rows = 0;
   int cols = -1;
   // col names
   std::getline(file, line); 
   std::stringstream ss(line);
   std::string cell;
   while (std::getline(ss, cell, ',')) {
      column_names.push_back(cell);
   }
   // features
   while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::vector<double> cells;
      std::string cell;
      while (std::getline(ss, cell, ',')) {
         cells.push_back(std::stod(cell));
      }
      if (cols == -1) {
         cols = cells.size();
      }
      else if (cells.size() != cols) {
         std::cerr << "Row " << rows << " has " << cells.size() << " columns, but the first row has " << cols << " columns." << std::endl;
         exit(1);
      }
      data.push_back(cells);
      rows++;
   }
   file.close();
   shape[0] = rows;
   shape[1] = cols;
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


void CSV_Handler::head(int n){
   if(n > data.size()){
      n = data.size(); 
   }
   int features = data[0].size();

   // column names
   for (int i = 0; i < std::min(4, (int)column_names.size()); i++) {
      std::cout << std::setw(8) << column_names[i] << "; ";
   }
   if (column_names.size() > 8) {
      std::cout << " ... ";
      for (int i = column_names.size() - 4; i < column_names.size(); i++) {
         std::cout << std::setw(8) << column_names[i] << "; ";
      }
   } else if (column_names.size() > 4) {
      for (int i = 4; i < column_names.size(); i++) {
         std::cout << std::setw(8) << column_names[i] << "; ";
      }
   }
   std::cout << std::endl;
   // values
   for (int i = 0; i < n; i++) {
      for (int j = 0; j < std::min(4, (int)data[i].size()); j++) {
         std::cout << std::setw(6) << data[i][j] << "   ";
      }
      if (data[i].size() > 8) {
         std::cout << " ... ";
         for (int j = data[i].size() - 4; j < data[i].size(); j++) {
            std::cout << std::setw(6) << data[i][j] << "   ";
         }
      } else if (data[i].size() > 4) {
         for (int j = 4; j < data[i].size(); j++) {
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


std::vector<std::string> CSV_Handler::get_columns_names(){
   return column_names;
}

std::vector<std::vector<double>> CSV_Handler::get_columns_values(int start, int end){
   if(end <= start || end > data.size() || start < 0){
      std::cout << "Invalid start: " << start << " or end: " << end << " argument, for data shape of: (" << 
      shape[0] << ", " << shape[1] << ")" << std::endl;
   }

   std::vector<int> column_idexes(end-start);
   std::iota(column_idexes.begin(), column_idexes.end(), start);

   std::vector<std::vector<double>> columns(data.size() , std::vector<double>(column_idexes.size()));
   for(int i = 0; i < data.size(); i++){
      for(int j = 0; j < column_idexes.size(); j++){
         columns[i][j] = data[i][column_idexes[j]];
      }
   }
   return columns;
} 


void CSV_Handler::clear(){
   data.clear();
   column_names.clear();
   shape.fill(0);
}