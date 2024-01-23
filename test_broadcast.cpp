#include <iostream>
#include "tensor.cpp"
using namespace std;
using namespace ts;

int main()
{
     // 测试broadcast
     cout << endl
          << "-------------------------bonus: 测试broadcast-------------------------" << endl
          << endl;
     
     cout << "broadcast可通配int, float等多种数据类型，支持加、减、乘、除等多种运算。" << endl << endl;
     cout << "测试int类型broadcast：" << endl << endl;

     int shape_int_b1 = 4;
     int size_int_b1[] = {2, 4, 2, 1};
     int dtype_int_b1 = dtype_int;
     int data_int_b1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
     ts::Tensor<int> int_b1(shape_int_b1, size_int_b1, dtype_int_b1, data_int_b1);

     cout << "broadcast tensor int_b1: size = {";
     for (int i = 0; i < shape_int_b1-1; i++)
     {
          cout << size_int_b1[i] << ", ";
     }
     cout << size_int_b1[shape_int_b1-1] << "}" << endl << endl << int_b1 << endl << endl;


     int shape_int_b2 = 3;
     int size_int_b2[] = {1, 2, 3};
     int dtype_int_b2 = dtype_int;
     int data_int_b2[] = {100, 200, 300, 400, 500, 600};
     ts::Tensor<int> int_b2(shape_int_b2, size_int_b2, dtype_int_b2, data_int_b2);

     cout << "int_b2 : size = {";
     for (int i = 0; i < shape_int_b2-1; i++)
     {
          cout << size_int_b2[i] << ", ";
     }
     cout << size_int_b2[shape_int_b2-1] << "}" << endl << endl << int_b2 << endl << endl;

     ts::Tensor<int> int_add_res = add(int_b1, int_b2);
     cout << "int_add_res size = {";
     for (int i = 0; i < int_add_res.getShape()-1; i++)
     {
          cout << int_add_res.getSize()[i] << ", ";
     }
     cout << int_add_res.getSize()[int_add_res.getShape()-1] << "}" << endl << endl << int_add_res << endl << endl;

     cout << "测试float类型broadcast：" << endl << endl;

     int shape_b1 = 4;
     int size_b1[] = {2, 4, 2, 1};
     int dtype_b1 = dtype_float;
     float data_b1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
     ts::Tensor<float> b1(shape_b1, size_b1, dtype_b1, data_b1);

     cout << "broadcast tensor float_b1: size = {";
     for (int i = 0; i < shape_b1-1; i++)
     {
          cout << size_b1[i] << ", ";
     }
     cout << size_b1[shape_b1-1] << "}" << endl << endl << b1 << endl << endl;


     int shape_b2 = 3;
     int size_b2[] = {1, 2, 3};
     int dtype_b2 = dtype_float;
     float data_b2[] = {100.1, 200.2, 300.3, 400.4, 500.5, 600.6};
     ts::Tensor<float> b2(shape_b2, size_b2, dtype_b2, data_b2);

     cout << "float_b2 : size = {";
     for (int i = 0; i < shape_b2-1; i++)
     {
          cout << size_b2[i] << ", ";
     }
     cout << size_b2[shape_b2-1] << "}" << endl << endl << b2 << endl << endl;
     

     ts::Tensor<float> add_res = add(b1, b2);
     cout << "add_res size = {";
     for (int i = 0; i < add_res.getShape()-1; i++)
     {
          cout << add_res.getSize()[i] << ", ";
     }
     cout << add_res.getSize()[add_res.getShape()-1] << "}" << endl << endl << add_res << endl << endl;
     

     ts::Tensor<float> sub_res = sub(b1, b2);
     cout << "sub_res size = {";
     for (int i = 0; i < sub_res.getShape()-1; i++)
     {
          cout << sub_res.getSize()[i] << ", ";
     }
     cout << sub_res.getSize()[sub_res.getShape()-1] << "}" << endl << endl << sub_res << endl << endl;
     

     ts::Tensor<float> mul_res = mul(b1, b2);
     cout << "mul_res size = {";
     for (int i = 0; i < mul_res.getShape()-1; i++)
     {
          cout << mul_res.getSize()[i] << ", ";
     }
     cout << mul_res.getSize()[mul_res.getShape()-1] << "}" << endl << endl << mul_res << endl << endl;
     

     ts::Tensor<float> div_res = div(b1, b2);
     cout << "div_res size = {";
     for (int i = 0; i < div_res.getShape()-1; i++)
     {
          cout << div_res.getSize()[i] << ", ";
     }
     cout << div_res.getSize()[div_res.getShape()-1] << "}" << endl << endl << div_res << endl << endl;
     
     return 0;
}
