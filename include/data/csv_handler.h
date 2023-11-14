#ifndef CSV_HANDLER
#define CSV_HANDLER

#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <iomanip>


class CSV_Handler{
   std::vector<std::string> column_names;
   std::vector<std::vector<double>> data;

   std::array<int, 2> shape;

public:
   CSV_Handler();
   ~CSV_Handler();
   void head(int n=5);
   
   // setters
   void read_csv(std::string path);
   void from_array(std::vector<std::vector<double>> array, std::vector<std::string> col_names);

   void delete_row(int index);

   std::array<int, 2> get_shape();
   std::vector<std::vector<double>> get_data(); // returns values
   std::vector<std::string>get_columns_names(); // returns values
   
   std::vector<std::vector<double>> get_columns_values(int start, int end); // returns colums from {start} id to {end} id excluded 

   void clear();
   friend std::ostream& operator<<(std::ostream& os, const CSV_Handler& handler);
};


#endif