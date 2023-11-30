#ifndef DATAFRAME
#define DATAFRAME

#include <array>
#include <set>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <iomanip>


class DataFrame{
   std::vector<std::string> column_names;
   std::vector<std::vector<double>> data; // stored in (cols, rows) format

public:
   DataFrame(); 
   ~DataFrame();
   void head(int n=5, int max_cols = 8);
   
   // setters
   void read_csv(std::string path);
   void from_array(std::vector<std::vector<double>> array, std::vector<std::string> col_names); // (rows, cols) format                 

   // deleters
   std::vector<std::vector<double>> drop_col(std::string col_name);
   
   void delete_row(int index);
   void clear();

   // getters
   std::array<int, 2> get_shape(); // returns shape (row, col)
   std::vector<std::vector<double>> get_data(); // returns csv values without col_names in (row, cols) format
   std::vector<std::vector<double>> get_slice(int row_st, int row_end, int col_st, int col_end); // returns slice of csv [st:ed, st:ed] in (rows, cols)
   std::vector<std::string>get_columns_names(); // returns column names
   
   // other
   void unique();
   void create_dummy(std::string column_name); // transforms column to dummies
   void standarize_cols(int col_st, int col_end);

   friend std::ostream& operator<<(std::ostream& os, const DataFrame& handler);
};


#endif