#include <iostream>
#include "tensor.cpp"
using namespace std;
using namespace ts;

int main() {
    // 示例用法
    int shape = 3;
    int size[] = {2, 3, 4};
    int dtype = dtype_float;
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

    ts::Tensor<float> tensor(shape, size, dtype, data);

    // 示例输出
    cout << "Shape: " << tensor.getShape() << endl;

    cout << "Size: ";
    for (int i = 0; i < shape; ++i) {
        cout << tensor.getSize()[i] << " ";
    }
    cout << endl;

    cout << "Total Size: " << tensor.getTotalSize() << endl;

    cout << "dtype: " << tensor.type_name() << endl;

    cout << "data_ptr: " << tensor.data_ptr() << endl;

    cout << endl << tensor << endl;


    cout << endl << endl << endl << endl;


    // ts::rand<long>();

    int * rand_size = new int[3]{3, 4, 5};
    // ts::Tensor<bool> xxx = ts::rand<bool>(3, rand_size);
    // ts::Tensor<char> xxx = ts::rand<char>(3, rand_size);
    // ts::Tensor<int> xxx = ts::rand<int>(3, rand_size);
    // ts::Tensor<long long> xxx = ts::rand<long long>(3, rand_size);
    // ts::Tensor<float> xxx = ts::rand<float>(3, rand_size, 20, 100);
    // ts::Tensor<float> xxx = ts::rand<float>(3, rand_size);
    // ts::Tensor<double> xxx = ts::rand<double>(3, rand_size, 20, 100);
    ts::Tensor<double> xxx = ts::rand<double>(3, rand_size);

    cout << "Shape: " << xxx.getShape() << endl;

    cout << "Size: ";
    for (int i = 0; i < shape; ++i) {
        cout << xxx.getSize()[i] << " ";
    }
    cout << endl;

    cout << "Total Size: " << xxx.getTotalSize() << endl;

    cout << "dtype: " << xxx.type_name() << endl;

    cout << "data_ptr: " << xxx.data_ptr() << endl;

    cout << endl << xxx << endl;    

    return 0;
}
