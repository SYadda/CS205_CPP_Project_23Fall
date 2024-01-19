#include <iostream>
#include <exception>
#include <stdexcept>
#include <random>
#include <limits>
#include <cstring>
#include <typeinfo>
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
          template <typename> friend class Tensor; //便于不同数据类型实例化的类互访

        void printTensor(ostream &os, T *data, int *size, int virtual_index, bool is_first) const {
            if (virtual_index == shape) { // virtual_index超出shape数组范围，即输出单个数据
                if (typeid(T).name() == typeid(bool).name()) {
                    if (*data) {
                        os << "true";
                    } else {
                        os << "false";
                    }
                } else {
                    os << *data;
                }
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
                    if (typeid(T).name() == typeid(bool).name()) {
                        if (*(data + i * offset[index])) {
                            os << ", true";
                        } else {
                            os << ", false";
                        }    
                    } else {
                        os << ", " << *(data + i * offset[index]);
                    }
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
        Tensor(int s, const int *sz, int dt, const T *d = nullptr) : shape(s), dtype(dt) {
            size = new int[shape];
            permute = new int[shape];
            offset = new int[shape];

            total_size = 1;
            for (int i = 0; i < shape; ++i) {
                size[i] = sz[i];
                permute[i] = i;
                total_size *= size[i];
            }

                 if (shape > 0) {
                offset[shape - 1] = 1;
                for (int i = shape - 2; i >= 0; --i) {
                    offset[i] = offset[i + 1] * size[i + 1];
                }
            }


            data = new T[total_size];
            if (d != nullptr) {
                for (int i = 0; i < total_size; ++i) {
                    data[i] = d[i];
                }
            } else {
                for (int i = 0; i < total_size; ++i) {
                    data[i] = 0;
                }
            }
        }

        // copy constructor
        Tensor(const Tensor &t) : Tensor(t.shape, t.size, t.dtype, t.data) {
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
 void setPermute(int *p) {
            for (int i = 0; i < shape; ++i) {
                permute[i] = p[i];
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

            //根据t的offset，计算出t的每一个data的地址，并打印
            t.printTensor(os, t.data, t.size, 0, true);
            return os;
        }
         //尺寸验证
        void assert_shape_same(const Tensor &t) const {
            if (shape != t.shape) {
                throw std::runtime_error("Shape mismatch");
            }

            for (int i = 0; i < shape; ++i) {
                if (this->size[i] != t.size[i]) {
                    throw std::runtime_error("Size mismatch");
                }
            }
        }
        //将index数组返回成int类型的data索引
        int get_index(int *index) const {
            int real_index = 0;
            for (int i = 0; i < shape; ++i) {
                real_index += index[i] * offset[permute[i]];
            }
            return real_index;
        }
        //获得data的元素引用，便于修改元素
        T &access(int *index) {
            const int real_index = get_index(index);
            return data[real_index];
        }
        //获得dataa的元素的值
        T access(int *index) const {
            const int real_index = get_index(index);
            return data[real_index];
        }
         //函数模板，只不过不确定的类型是函数，这样不同的+ —法等都可以当输入参数
         //实现对某一维度进行操作
        template<typename ReduceFunc>
        Tensor reduce(int dim, ReduceFunc reduce_fuc) const {
            if (dim < 0 || dim >= shape) {
                throw std::runtime_error("Dimension out of range");
            }

            int *new_size = new int[shape - 1];
            for (int i = 0; i < shape; ++i) {
                if (i < dim) {
                    new_size[i] = size[i];
                } else if (i > dim) {
                    new_size[i - 1] = size[i];
                }
            }

            Tensor res(shape - 1, new_size, dtype);


            
            // index索引的是原tensor中的数据
            int *index = new int[shape]{};

            // target_index索引目标tensor中的一个数据
            int *target_index = new int[res.shape]{};

            const int sum_size = size[permute[dim]];

            // (3,4,5)
            // [0,0,0]
            // 0,1,2

            // sum(1)
            // [0,0] [0,1] [0,2] [0,3] [0,4] [1,0]
            // 

            // [1,3]
            // [1,0,3] [1,1,3] [1,2,3] [1,3,3] 
            // 0,1

            
            for (int i = 0; i < res.total_size; ++i) {
                {
                    // 根据目标的index去得到原tensor中的index
                    int k = 0;
                    for (int j = 0; j < shape; ++j) {
                        if (j != dim) {
                            index[j] = target_index[k++];
                        }
                    }
                }

                index[dim] = 0;

                T val = access(index);
                for (int j = 1; j < sum_size; ++j) {
                    index[dim] = j;
                    val = reduce_fuc(val, access(index));
                }

                res.access(target_index) = val;

                // [3, 5] 
                // [0, 0] -> [0, 1] -> [0, 2] -> [0, 3] -> [0, 4] -> [1, 0] -> [1, 1] -> [1, 2]
                // 更新target_index,指向下一个目标位置
                for (int j = res.shape - 1; j >= 0; --j) {
                    if (target_index[j] < res.size[j] - 1) {
                        ++target_index[j];
                        break;
                    } else {
                        target_index[j] = 0;
                    }
                }
            }

            delete[] index;
            delete[] target_index;
            delete[] new_size;
            return res;
        }

    
    
       

    
// 函数调用
        Tensor add(const Tensor &t) const {
            assert_shape_same(t);

            Tensor res(shape, this->size, dtype, data);

            int total_size = this->total_size;
            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] + t.data[i];
            }

            return res;
        }

        Tensor add(T t) const {
            Tensor res(shape, size, dtype, data);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = data[i] + t;
            }

            return res;
        }

        Tensor sub(const Tensor &t) const {
            assert_shape_same(t);

            Tensor res(shape, this->size, dtype, data);

            int total_size = this->total_size;
            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] - t.data[i];
            }

            return res;
        }

        Tensor sub(T t) const {
            Tensor res(shape, size, dtype, data);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = data[i] - t;
            }

            return res;
        }

        
        Tensor mul(const Tensor &t) const {
            assert_shape_same(t);

            Tensor res(shape, this->size, dtype, data);

            int total_size = this->total_size;
            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] * t.data[i];
            }

            return res;
        }

        Tensor mul(T t) const {
            Tensor res(shape, size, dtype, data);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = data[i] * t;
            }

            return res;
        }

        Tensor div(const Tensor &t) const {
            assert_shape_same(t);

            Tensor res(shape, this->size, dtype, data);

            int total_size = this->total_size;
            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] / t.data[i];
            }

            return res;
        }

        Tensor div(T t) const {
            Tensor res(shape, size, dtype, data);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = data[i] / t;
            }

            return res;
        }

        Tensor<double> log() const {
            Tensor<double> res(shape, size, dtype, dtype_double);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = std::log(static_cast<double>(data[i]));
            }

            return res;
        }

        Tensor<bool> eq(const Tensor &t) const {
            assert_shape_same(t);

            Tensor<bool> res(t.shape, t.size, dtype_bool);

            for (int i = 0; i < t.total_size; ++i) {
                res.data[i] = this->data[i] == t.data[i];
            }

            return res;
        }

        Tensor<bool> eq(T t) const {
            Tensor<bool> res(shape, size, dtype_bool);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] == t;
            }

            return res;
        }

        Tensor<bool> ne(const Tensor &t) const {
            assert_shape_same(t);

            Tensor<bool> res(t.shape, t.size, dtype_bool);

            for (int i = 0; i < t.total_size; ++i) {
                res.data[i] = this->data[i] != t.data[i];
            }

            return res;
        }

        Tensor<bool> ne(T t) const {
            Tensor<bool> res(shape, size, dtype_bool);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] != t;
            }

            return res;
        }

        Tensor<bool> gt(const Tensor &t) const {
            assert_shape_same(t);

            Tensor<bool> res(t.shape, t.size, dtype_bool);

            for (int i = 0; i < t.total_size; ++i) {
                res.data[i] = this->data[i] > t.data[i];
            }

            return res;
        }

        Tensor<bool> gt(T t) const {
            Tensor<bool> res(shape, size, dtype_bool);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] > t;
            }

            return res;
        }

        Tensor<bool> ge(const Tensor &t) const {
            assert_shape_same(t);

            Tensor<bool> res(t.shape, t.size, dtype_bool);

            for (int i = 0; i < t.total_size; ++i) {
                res.data[i] = this->data[i] >= t.data[i];
            }

            return res;
        }

        Tensor<bool> ge(T t) const {
            Tensor<bool> res(shape, size, dtype_bool);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] >= t;
            }

            return res;
        }

        Tensor<bool> lt(const Tensor &t) const {
            assert_shape_same(t);

            Tensor<bool> res(t.shape, t.size, dtype_bool);

            for (int i = 0; i < t.total_size; ++i) {
                res.data[i] = this->data[i] < t.data[i];
            }

            return res;
        }

        Tensor<bool> lt(T t) const {
            Tensor<bool> res(shape, size, dtype_bool);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] < t;
            }

            return res;
        }

        Tensor<bool> le(const Tensor &t) const {
            assert_shape_same(t);

            Tensor<bool> res(t.shape, t.size, dtype_bool);

            for (int i = 0; i < t.total_size; ++i) {
                res.data[i] = this->data[i] <= t.data[i];
            }

            return res;
        }

        Tensor<bool> le(T t) const {
            Tensor<bool> res(shape, size, dtype_bool);

            for (int i = 0; i < total_size; ++i) {
                res.data[i] = this->data[i] <= t;
            }

            return res;
        }

        Tensor sum(int dim) const {
            return reduce(dim, [](T a, T b) { return a + b; });
        }

        Tensor max(int dim) const {
            return reduce(dim, [](T a, T b) { return a >= b ? a : b; });
        }

        Tensor min(int dim) const {
            return reduce(dim, [](T a, T b) { return a <= b ? a : b; });
        }

        Tensor mean(int dim) const {
            Tensor res = sum(dim);
            int dim_size = size[permute[dim]];
            for (int i = 0; i < res.total_size; ++i) {
                res.data[i] /= dim_size;
            }
            return res;
        }
        //3.4
    void check_equation_lhs(const char *equation, int &n_uniq_vars, int *vars, int *lhs_vars, int *var_size) const {
            for (int i = 0; i < shape; ++i) {
                if (equation[i] < 'a' || equation[i] > 'z') {
                    throw std::runtime_error("Invalid equation");
                }
                int var = equation[i] - 'a';

                lhs_vars[i] = var;

                bool var_appeared = false;
                for (int j = 0; j < n_uniq_vars; ++j) {
                    if (vars[j] == var) {
                        var_appeared = true;
                        break;
                    }
                }

                if (!var_appeared) {
                    // 这个变量首次出现
                    var_size[var] = size[permute[i]];
                    vars[n_uniq_vars] = var;
                    ++n_uniq_vars;
                } else {
                    // 这个变量之前出现过
                    if (size[permute[i]] != var_size[var]) {
                        throw std::runtime_error("Invalid equation");
                    }
                }
            }
        }

       Tensor einsum(const char *equation) const {
            for (const char *s = equation; s[0] != '\0'; ++s) {
                if (!((s[0] >= 'a' && s[0] <= 'z') || s[0] == '-' || s[0] == '>')) {
                    throw std::runtime_error("Invalid equation");
                }
            }

            const char *arrow = strstr(equation, "->");
            const int eq_len = strlen(equation);
            int res_shape = eq_len - (arrow - equation) - 2;

            if (arrow == NULL || arrow - equation != shape || res_shape < 0) {
                throw std::runtime_error("Invalid equation");
            }

            int *lhs_vars = new int[shape];
            int vars[26];
            int var_size[26]{};
            int n_vars = 0;

            check_equation_lhs(equation, n_vars, vars, lhs_vars, var_size);

            const char *rhs_equation = equation + (shape + 2);
            int rhs_size[26]{};
            bool is_rhs_var[26]{};
            int rhs_vars[26]{};

            for (int i = 0; i < res_shape; ++i) {
                char var = rhs_equation[i] - 'a';
                const char *ptr = strchr(equation, var + 'a');
                if (ptr == &rhs_equation[i]) {
                    // 右边的变量必须要在左边出现过
                    throw std::runtime_error("Invalid equation");
                }

                if (strchr(rhs_equation + i + 1, var + 'a') != NULL) {
                    // 右边的变量不能重复
                    throw std::runtime_error("Invalid equation");
                }

                rhs_size[i] = var_size[var];
                rhs_vars[i] = var;
                is_rhs_var[var] = true;
            }

            int sum_size = 1;
            int n_sum_vars = 0;
            int sum_vars[26]{};
            for (int i = 0; i < n_vars; ++i) {
                if (!is_rhs_var[vars[i]]) {
                    sum_vars[n_sum_vars++] = vars[i];
                    sum_size *= var_size[vars[i]];
                }
            }

            int *rhs_idx = new int[res_shape]{}; // 用于遍历res的index
            int *lhs_idx = new int[shape]{}; // 用于遍历lhs的index
            int var_vals[26]{};

            Tensor res(res_shape, rhs_size, dtype);

            for (int i = 0; i < res.total_size; ++i) {
                T sum{};
                for (int j = 0; j < n_sum_vars; ++j) {
                    var_vals[sum_vars[j]] = 0;
                }

                for (int j = 0; j < sum_size; ++j) {
                    for (int k = 0; k < shape; ++k) {
                        lhs_idx[k] = var_vals[lhs_vars[k]];
                    }

                    sum += access(lhs_idx);

                    for (int j = 0; j < n_sum_vars; ++j) {
                        const int var = sum_vars[j];
                        if (var_vals[var] < var_size[var] - 1) {
                            ++var_vals[var];
                            break;
                        } else {
                            var_vals[var] = 0;
                        }
                    }
                }

                for (int k = 0; k < res_shape; ++k) {
                    rhs_idx[k] = var_vals[rhs_vars[k]];
                }

                res.access(rhs_idx) = sum;

                for (int j = res_shape - 1; j >= 0; --j) {
                    const int var = rhs_vars[j];
                    if (var_vals[var] < var_size[var] - 1) {
                        ++var_vals[var];
                        break;
                    } else {
                        var_vals[var] = 0;
                    }
                }
            }

            delete[] lhs_vars;
            delete[] rhs_idx;
            delete[] lhs_idx;

            return res;
        }

        Tensor einsum(const char *equation, const Tensor &t) const {
            for (const char *s = equation; s[0] != '\0'; ++s) {
                if (!((s[0] >= 'a' && s[0] <= 'z') || s[0] == ',' || s[0] == '-' || s[0] == '>')) {
                    throw std::runtime_error("Invalid equation");
                }
            }

            if (strlen(equation) < shape + 1 + t.shape + 2) {
                throw std::runtime_error("Invalid equation");
            }


            if (equation[shape] != ',' || equation[shape + 1 + t.shape] != '-' || equation[shape + 1 + t.shape + 1] != '>') {
                throw std::runtime_error("Invalid equation");
            }

            const int res_shape = strlen(equation) - (shape + 1 + t.shape + 2);

            int *lhs_vars1 = new int[shape];
            int *lhs_vars2 = new int[t.shape];
            int vars[26];
            int var_size[26]{};
            int n_vars = 0;

            check_equation_lhs(equation, n_vars, vars, lhs_vars1, var_size);
            t.check_equation_lhs(equation + shape + 1, n_vars, vars, lhs_vars2, var_size);

            const char *rhs_equation = equation + (shape + 1 + t.shape + 2);
            int rhs_size[26]{};
            bool is_rhs_var[26]{};
            int rhs_vars[26]{};

            for (int i = 0; i < res_shape; ++i) {
                char var = rhs_equation[i] - 'a';
                const char *ptr = strchr(equation, var + 'a');
                if (ptr == &rhs_equation[i]) {
                    // 右边的变量必须要在左边出现过
                    throw std::runtime_error("Invalid equation");
                }

                if (strchr(rhs_equation + i + 1, var + 'a') != NULL) {
                    // 右边的变量不能重复
                    throw std::runtime_error("Invalid equation");
                }

                rhs_size[i] = var_size[var];
                rhs_vars[i] = var;
                is_rhs_var[var] = true;
            }

            int sum_size = 1;
            int n_sum_vars = 0;
            int sum_vars[26]{};
            for (int i = 0; i < n_vars; ++i) {
                if (!is_rhs_var[vars[i]]) {
                    sum_vars[n_sum_vars++] = vars[i];
                    sum_size *= var_size[vars[i]];
                }
            }

            int *rhs_idx = new int[res_shape]{}; // 用于遍历res的index
            int *lhs_idx1 = new int[shape]{}; // 用于遍历lhs的index
            int *lhs_idx2 = new int[t.shape]{}; // 用于遍历lhs的index
            int var_vals[26]{};

            Tensor res(res_shape, rhs_size, dtype);

            for (int i = 0; i < res.total_size; ++i) {
                T sum{};
                for (int j = 0; j < n_sum_vars; ++j) {
                    var_vals[sum_vars[j]] = 0;
                }

                for (int j = 0; j < sum_size; ++j) {
                    for (int k = 0; k < shape; ++k) {
                        lhs_idx1[k] = var_vals[lhs_vars1[k]];
                    }
                    for (int k = 0; k < t.shape; ++k) {
                        lhs_idx2[k] = var_vals[lhs_vars2[k]];
                    }

                    const T v1 = access(lhs_idx1);
                    const T v2 = t.access(lhs_idx2);
                    sum += v1 * v2;

                    for (int j = 0; j < n_sum_vars; ++j) {
                        const int var = sum_vars[j];
                        if (var_vals[var] < var_size[var] - 1) {
                            ++var_vals[var];
                            break;
                        } else {
                            var_vals[var] = 0;
                        }
                    }
                }

                for (int k = 0; k < res_shape; ++k) {
                    rhs_idx[k] = var_vals[rhs_vars[k]];
                }

                res.access(rhs_idx) = sum;

                for (int j = res_shape - 1; j >= 0; --j) {
                    const int var = rhs_vars[j];
                    if (var_vals[var] < var_size[var] - 1) {
                        ++var_vals[var];
                        break;
                    } else {
                        var_vals[var] = 0;
                    }
                }
            }

            delete[] lhs_vars1;
            delete[] lhs_vars2;
            delete[] rhs_idx;
            delete[] lhs_idx1;
            delete[] lhs_idx2;

            return res;
        }
        //运算符重载
        Tensor operator+(const Tensor &t) const {
            return add(t);
        }

        Tensor operator+(T t) const {
            return add(t);
        }

        friend Tensor operator+(T t, const Tensor &tensor) {
            return tensor.add(t);
        }

        Tensor operator-(const Tensor &t) const {
            return sub(t);
        }

        Tensor operator-(T t) const {
            return sub(t);
        }

        friend Tensor operator-(T t, const Tensor &tensor) {
            Tensor res(tensor.shape, tensor.size, tensor.dtype, tensor.data);

            for (int i = 0; i < tensor.total_size; ++i) {
                res.data[i] = t - tensor.data[i];
            }

            return res;
        }

        Tensor operator*(const Tensor &t) const {
            return mul(t);
        }

        Tensor operator*(T t) const {
            return mul(t);
        }

        friend Tensor operator*(T t, const Tensor &tensor) {
            return tensor.mul(t);
        }

        Tensor operator/(T t) const {
            return div(t);
        }

        Tensor operator/(const Tensor &t) const {
            return div(t);
        }

        friend Tensor operator/(T t, const Tensor &tensor) {
            Tensor res(tensor.shape, tensor.size, tensor.dtype, tensor.data);

            for (int i = 0; i < tensor.total_size; ++i) {
                res.data[i] = t / tensor.data[i];
            }

            return res;
        }

        Tensor<bool> operator==(const Tensor &t) const {
            return eq(t);
        }

        Tensor<bool> operator==(T t) const {
            return eq(t);
        }

        Tensor<bool> operator!=(const Tensor &t) const {
            return ne(t);
        }

        Tensor<bool> operator!=(T t) const {
            return ne(t);
        }

        Tensor<bool> operator>(const Tensor &t) const {
            return gt(t);
        }

        Tensor<bool> operator>(T t) const {
            return gt(t);
        }

        Tensor<bool> operator>=(const Tensor &t) const {
            return ge(t);
        }

        Tensor<bool> operator>=(T t) const {
            return ge(t);
        }

        Tensor<bool> operator<(const Tensor &t) const {
            return lt(t);
        }

        Tensor<bool> operator<(T t) const {
            return lt(t);
        }

        Tensor<bool> operator<=(const Tensor &t) const {
            return le(t);
        }

        Tensor<bool> operator<=(T t) const {
            return le(t);
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

        Tensor<T> res(shape, size, dtype, data);
        delete[] data;
        return res;
    }


    template<typename T>
    Tensor<T> add(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.add(t2);
    }

    template<typename T>
    Tensor<T> add(const Tensor<T> &t1, T t2) {
        return t1.add(t2);
    }

    template<typename T>
    Tensor<T> add(T t1, const Tensor<T> &t2) {
        return t1 + t2;
    }

    template<typename T>
    Tensor<T> sub(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.sub(t2);
    }

    template<typename T>
    Tensor<T> sub(const Tensor<T> &t1, T t2) {
        return t1.sub(t2);
    }

    template<typename T>
    Tensor<T> sub(T t1, const Tensor<T> &t2) {
        return t1 - t2;
    }

    template<typename T>
    Tensor<T> mul(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.mul(t2);
    }

    template<typename T>
    Tensor<T> mul(const Tensor<T> &t1, T t2) {
        return t1.mul(t2);
    }

    template<typename T>
    Tensor<T> mul(T t1, const Tensor<T> &t2) {
        return t1 * t2;
    }

    template<typename T>
    Tensor<T> div(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.div(t2);
    }

    template<typename T>
    Tensor<T> div(const Tensor<T> &t1, T t2) {
        return t1.div(t2);
    }

    template<typename T>
    Tensor<T> div(T t1, const Tensor<T> &t2) {
        return t1 / t2;
    }

    // eq , ne , gt , ge , lt , le
    template<typename T>
    Tensor<bool> eq(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.eq(t2);
    }

    template<typename T>
    Tensor<bool> eq(const Tensor<T> &t1, T t2) {
        return t1.eq(t2);
    }

    template<typename T>
    Tensor<bool> eq(T t1, const Tensor<T> &t2) {
        return t2.eq(t1);
    }

    template<typename T>
    Tensor<bool> ne(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.ne(t2);
    }

    template<typename T>
    Tensor<bool> operator==(T t1, const Tensor<T> &t2) {
        return t2.eq(t1);
    }

    template<typename T>
    Tensor<bool> ne(const Tensor<T> &t1, T t2) {
        return t1.ne(t2);
    }

    template<typename T>
    Tensor<bool> ne(T t1, const Tensor<T> &t2) {
        return t2.ne(t1);
    }

    template<typename T>
    Tensor<bool> operator!=(T t1, const Tensor<T> &t2) {
        return t2.ne(t1);
    }

    template<typename T>
    Tensor<bool> gt(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.gt(t2);
    }

    template<typename T>
    Tensor<bool> gt(const Tensor<T> &t1, T t2) {
        return t1.gt(t2);
    }

    template<typename T>
    Tensor<bool> gt(T t1, const Tensor<T> &t2) {
        return t2.lt(t1);
    }

    template<typename T>
    Tensor<bool> operator>(T t1, const Tensor<T> &t2) {
        return t2.lt(t1);
    }

    template<typename T>
    Tensor<bool> ge(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.ge(t2);
    }

    template<typename T>
    Tensor<bool> ge(const Tensor<T> &t1, T t2) {
        return t1.ge(t2);
    }

    template<typename T>
    Tensor<bool> ge(T t1, const Tensor<T> &t2) {
        return t2.le(t1);
    }

    template<typename T>
    Tensor<bool> operator>=(T t1, const Tensor<T> &t2) {
        return t2.le(t1);
    }

    template<typename T>
    Tensor<bool> lt(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.lt(t2);
    }

    template<typename T>
    Tensor<bool> lt(const Tensor<T> &t1, T t2) {
        return t1.lt(t2);
    }

    template<typename T>
    Tensor<bool> lt(T t1, const Tensor<T> &t2) {
        return t2.gt(t1);
    }

    template<typename T>
    Tensor<bool> operator<(T t1, const Tensor<T> &t2) {
        return t2.gt(t1);
    }

    template<typename T>
    Tensor<bool> le(const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.le(t2);
    }

    template<typename T>
    Tensor<bool> le(const Tensor<T> &t1, T t2) {
        return t1.le(t2);
    }

    template<typename T>
    Tensor<bool> le(T t1, const Tensor<T> &t2) {
        return t2.ge(t1);
    }

    template<typename T>
    Tensor<bool> operator<=(T t1, const Tensor<T> &t2) {
        return t2.ge(t1);
    }

    template<typename T>
    Tensor<T> sum(const Tensor<T> &t, int dim) {
        return t.sum(dim);
    }

    template<typename T>
    Tensor<T> mean(const Tensor<T> &t, int dim) {
        return t.mean(dim);
    }

    template<typename T>
    Tensor<T> max(const Tensor<T> &t, int dim) {
        return t.max(dim);
    }

    template<typename T>
    Tensor<T> min(const Tensor<T> &t, int dim) {
        return t.min(dim);
    }

    template<typename T>
    Tensor<T> einsum(const char *equation, const Tensor<T> &t) {
        return t.einsum(equation);
    }

    template<typename T>
    Tensor<T> einsum(const char *equation, const Tensor<T> &t1, const Tensor<T> &t2) {
        return t1.einsum(equation, t2);
    }
}