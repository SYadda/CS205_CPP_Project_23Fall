#include <iostream>
#include <exception>
#include <random>
#include <limits>
using namespace std;

namespace ts {

    class UnsupportedTypesException: public exception {
    public:
        const char* what() const noexcept override {
            return "Unsupported types";
        }
    };

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
        void *data_ptr() const {
            return static_cast<void *>(data);
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

    template <typename T>
    Tensor<T> rand (int shape, int *size, T range_min = 0, T range_max = std::numeric_limits<T>::max()) { 

        int total_size = 1;
        for (int i = 0; i < shape; ++i) {
            total_size *= size[i];
        }

        T *data = new T[total_size];

        std::random_device rd;  // 用于获取随机数种子
        std::mt19937_64 gen(rd()); // 初始化64位Mersenne Twister随机数生成器

        // 识别 T 的类型
        int dtype;
        T temp;
        string typeName = typeid(temp).name();

        if (typeName == "b") {
            dtype = dtype_bool;
            std::uniform_int_distribution<> dis(0, 1); // 定义分布范围
            for (int i = 0; i < total_size; ++i) {
                data[i] = dis(gen); // 生成随机数
            }
        } else if (typeName == "c") {
            if (range_min == 0 && range_max == 127) {
                range_min = 32;  // 可打印字符范围
                range_max = 126;            
            }

            dtype = dtype_char;
            std::uniform_int_distribution<char> dis(range_min, range_max); // 定义分布范围
            for (int i = 0; i < total_size; ++i) {
                data[i] = dis(gen); // 生成随机数
            }
        } else if (typeName == "i") {
            dtype = dtype_int;
            std::uniform_int_distribution<int> dis(range_min, range_max); // 定义分布范围
            for (int i = 0; i < total_size; ++i) {
                data[i] = dis(gen); // 生成随机数
            }
        } else if (typeName == "x") {
            dtype = dtype_long_long;
            std::uniform_int_distribution<long long> dis(range_min, range_max); // 定义分布范围
            for (int i = 0; i < total_size; ++i) {
                data[i] = dis(gen); // 生成随机数
            }
        } else if (typeName == "f") {
            dtype = dtype_float;
            std::uniform_real_distribution<float> dis(range_min, range_max); // 定义分布范围
            for (int i = 0; i < total_size; ++i) {
                data[i] = dis(gen); // 生成随机数
            }
        } else if (typeName == "d") {
            dtype = dtype_double;
            std::uniform_real_distribution<double> dis(range_min, range_max); // 定义分布范围
            for (int i = 0; i < total_size; ++i) {
                data[i] = dis(gen); // 生成随机数
            }
        } else {
            throw UnsupportedTypesException();
            return Tensor<T>(0, nullptr, 0, nullptr);
        }

        return Tensor<T>(shape, size, dtype, data);
    }

}
