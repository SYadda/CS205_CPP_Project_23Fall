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

    return 0;
}
