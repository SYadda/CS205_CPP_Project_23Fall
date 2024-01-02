#include <iostream>
#include <cstdlib>
using namespace std;

enum dtype {
    dtype_bool = 0,
    dtype_char = 1,
    dtype_int = 2,
    dtype_long_long = 3,
    dtype_float = 4,
    dtype_double = 5
};

template <typename T>
class Tensor {
private:
    int shape;  // 张量的维数（形状）
    int *size;  // 张量每一维的宽度
    int total_size;  // 张量的总大小
    int dtype;  // 张量的数据类型
    int *permute;  // 张量的维度排列
    T *data;    // 张量的数据

public:
    // 构造函数
    Tensor(int s, int *sz, int dt, T *d) : shape(s), dtype(dt) {
        size = new int[shape];
        permute = new int[shape];

        total_size = 1;
        for (int i = 0; i < shape; ++i) {
            size[i] = sz[i];
            permute[i] = i;
            total_size *= size[i];
        }

        data = new T[total_size];
        for (int i = 0; i < total_size; ++i) {
            data[i] = d[i];
        }
    }

    // 析构函数
    ~Tensor() {
        delete[] size;
        delete[] permute;
        delete[] data;
    }

    // 获取张量的维数（形状）
    int getShape() const {
        return shape;
    }

    // 获取张量每一维的宽度
    int *getSize() const {
        return size;
    }

    // 获取张量的总大小
    int getTotalSize() const {
        return total_size;
    }

    // 获取张量数据类型
    int type() const {
        return dtype;
    }

    string type_name() const {
        switch (dtype) {
            case dtype_bool:         return "bool";
            case dtype_char:         return "char";
            case dtype_int:          return "int";
            case dtype_long_long:    return "long long";
            case dtype_float:        return "float";
            case dtype_double:       return "double";
            default:                 return "unknown";
        }
    }


    // 获取张量数据的首地址
    T *data_ptr() const {
        return data;
    }
};

int main() {
    // 示例用法
    int shape = 3;
    int size[] = {2, 3, 4};
    int dtype = dtype_float;
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

    Tensor<float> tensor(shape, size, dtype, data);

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

    cout << "Data: ";
    for (int i = 0; i < tensor.getTotalSize(); ++i) {
        cout << tensor.data_ptr()[i] << " ";
    }
    cout << endl;

    return 0;
}
