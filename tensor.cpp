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
    int *permute;  // 张量的转置状况（permute[i]，表示第i维的数据，在原张量数据结构中存储的位置）
    int *offset; // 张量的偏移量（offset[i]，表示第i维的数据，其内存地址，到邻近的下一个数据的偏移量）
    // 每次使用<<打印输出时，先根据当前的permute，计算出offset，然后根据offset，计算出data的地址，然后打印data的值
    T *data;    // 张量的数据

    void printTensor(ostream &os, T *data, int *size, int virtual_index, bool is_first) const {
        if (virtual_index == shape) { // virtual_index超出shape数组范围，即输出单个数据
            os << *data;
            return;
        }
        
        // 转置时只改变permute，不修改数据
        // virtual_index：从0向后遍历
        // index：数据结构存储中的真实index
        int index = permute[virtual_index];

        if (! is_first) { // 非第一维需要缩进
            for (int i = 0; i < index; ++i) {
                os << " ";
            }
        }
        os << "[";

        // i == 0
        printTensor(os, data, size, virtual_index + 1, true);

        if (index == shape-1) { // 最后一维不需要换行
            for (int i = 1; i < size[index]; ++i) {
                os << ", " << *(data + i * offset[index]);
            }
        } else { 
            for (int i = 1; i < size[index]; ++i) {
                os << "," << endl;
                printTensor(os, data + i * offset[index], size, virtual_index + 1, false);
            }
        }
        os << "]";
    }

public:
    // 构造函数
    Tensor(int s, int *sz, int dt, T *d) : shape(s), dtype(dt) {
        size = new int[shape];
        permute = new int[shape];
        offset = new int[shape];

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
        delete[] offset;
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

    int *getPermute() const {
        return permute;
    }

    // 获取张量数据的首地址
    T *data_ptr() const {
        return data;
    }

    /* 
        逐级打印，例如：
        [[0.1000, 1.2000],
         [2.2000, 3.10001],
         [4.9000, 5.20001]]
    */
    friend ostream &operator << (ostream &os, const Tensor &t) {
        //根据t的转置情况permute，计算出t的offset
        int *tp = t.getPermute();
        t.offset[t.shape - 1] = 1;
        for (int i = t.shape - 2; i >= 0; --i) {
            t.offset[i] = t.offset[i + 1] * t.size[tp[i + 1]];
        }

        //根据t的offset，计算出t的每一个data的地址，并打印
        t.printTensor(os, t.data, t.size, 0, true);
        return os;
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

    cout << endl << tensor << endl;

    return 0;
}
