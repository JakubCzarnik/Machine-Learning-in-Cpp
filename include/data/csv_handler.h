#ifndef CSV_HANDLER
#define CSV_HANDLER

#include <array>
#include <set>
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

   std::array<int, 2> shape; // (cols, rows)

public:
   CSV_Handler();
   ~CSV_Handler();
   void head(int n=5, int max_cols = 8);
   
   // setters
   void read_csv(std::string path);
   void from_array(std::vector<std::vector<double>> array, std::vector<std::string> col_names);

   // deleters
   void delete_row(int index);
   void clear();

   // getters
   std::array<int, 2> get_shape(); // returns shape (col, row)
   std::vector<std::vector<double>> get_data(); // returns csv values without col_names in (col, rows) form
   std::vector<std::vector<double>> get_slice(int col_st, int col_end, int row_st, int row_end); // returns slice of csv [st:ed, st:ed]
   std::vector<std::string>get_columns_names(); // returns column names
   
   // other
   void unique();
   void create_dummy(int column_index); // transforms column to dummies

   friend std::ostream& operator<<(std::ostream& os, const CSV_Handler& handler);
};


#endif