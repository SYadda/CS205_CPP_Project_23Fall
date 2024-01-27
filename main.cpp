#include <iostream>
#include "tensor.cpp"
using namespace std;
using namespace ts;

int main()
{
     {

          // 示例用法
          int shape = 3;
          int size[] = {2, 3, 4};
          int dtype = dtype_float;
          float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

          ts::Tensor<float> tensor(shape, size, dtype, data);

          // 示例输出
          cout << "Shape: " << tensor.getShape() << endl;

          cout << "Size: ";
          for (int i = 0; i < shape; ++i)
          {
               cout << tensor.getSize()[i] << " ";
          }
          cout << endl;

          cout << "Total Size: " << tensor.getTotalSize() << endl;

          cout << "dtype: " << tensor.type_name() << endl;

          cout << "data_ptr: " << tensor.data_ptr() << endl;

          cout << endl
               << tensor << endl;

          cout << endl
               << endl
               << endl
               << endl;

          int *p = new int[3]{1, 2, 0};
          tensor.setPermute(p);

          // 示例输出
          cout << "Shape: " << tensor.getShape() << endl;

          cout << "Size: ";
          for (int i = 0; i < shape; ++i)
          {
               cout << tensor.getSize()[i] << " ";
          }
          cout << endl;

          cout << "Total Size: " << tensor.getTotalSize() << endl;

          cout << "dtype: " << tensor.type_name() << endl;

          cout << "data_ptr: " << tensor.data_ptr() << endl;

          cout << endl
               << tensor << endl;

          cout << endl
               << endl
               << endl
               << endl;

          cout << "----------------- 1.2 Random Init -----------------" << endl;

          // ts::rand<long>();

          int *rand_size = new int[3]{3, 4, 5};
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
          for (int i = 0; i < shape; ++i)
          {
               cout << xxx.getSize()[i] << " ";
          }
          cout << endl;

          cout << "Total Size: " << xxx.getTotalSize() << endl;

          cout << "dtype: " << xxx.type_name() << endl;

          cout << "data_ptr: " << xxx.data_ptr() << endl;

          cout << endl
               << xxx << endl;

          cout << endl
               << endl
               << endl
               << endl;

          cout << "rand bool" << endl;
          ts::Tensor<bool> test_rand_b = ts::rand<bool>(3, rand_size);
          cout << test_rand_b << endl
               << endl;

          cout << "rand char" << endl;
          ts::Tensor<char> test_rand_c = ts::rand<char>(3, rand_size);
          cout << test_rand_c << endl
               << endl;

          cout << "rand int" << endl;
          ts::Tensor<int> test_rand_i = ts::rand<int>(3, rand_size);
          cout << test_rand_i << endl
               << endl;

          cout << "rand long long" << endl;
          ts::Tensor<long long> test_rand_ll = ts::rand<long long>(3, rand_size);
          cout << test_rand_ll << endl
               << endl;

          cout << "rand float" << endl;
          ts::Tensor<float> test_rand_f = ts::rand<float>(3, rand_size);
          cout << test_rand_f << endl
               << endl;

          cout << "rand double" << endl;
          ts::Tensor<double> test_rand_d = ts::rand<double>(3, rand_size);
          cout << test_rand_d << endl
               << endl;

          cout << endl
               << endl
               << endl
               << endl;

          cout << "----------------- 1.3.1 Zeros Init -----------------" << endl;

          cout << "zeros bool" << endl;
          ts::Tensor<bool> test_zeros_b = ts::zeros<bool>(3, rand_size);
          cout << test_zeros_b << endl
               << endl;

          cout << "zeros char" << endl;
          ts::Tensor<char> test_zeros_c = ts::zeros<char>(3, rand_size);
          cout << test_zeros_c << endl
               << endl;

          cout << "zeros int" << endl;
          ts::Tensor<int> test_zeros_i = ts::zeros<int>(3, rand_size);
          cout << test_zeros_i << endl
               << endl;

          cout << endl
               << endl
               << endl
               << endl;

          cout << "----------------- 1.3.2 Ones Init -----------------" << endl;

          cout << "ones bool" << endl;
          ts::Tensor<bool> test_ones_b = ts::ones<bool>(3, rand_size);
          cout << test_ones_b << endl
               << endl;

          cout << "ones char" << endl;
          ts::Tensor<char> test_ones_c = ts::ones<char>(3, rand_size);
          cout << test_ones_c << endl
               << endl;

          cout << "ones int" << endl;
          ts::Tensor<int> test_ones_i = ts::ones<int>(3, rand_size);
          cout << test_ones_i << endl
               << endl;

          cout << endl
               << endl
               << endl
               << endl;

          cout << "----------------- 1.3.3 Value Init -----------------" << endl;

          cout << "full bool" << endl;
          ts::Tensor<bool> test_full_b = ts::full<bool>(3, rand_size, true);
          cout << test_full_b << endl
               << endl;

          cout << "full char" << endl;
          ts::Tensor<char> test_full_c = ts::full<char>(3, rand_size, 'a');
          cout << test_full_c << endl
               << endl;

          cout << "full int" << endl;
          ts::Tensor<int> test_full_i = ts::full<int>(3, rand_size, 1234);
          cout << test_full_i << endl
               << endl;

          cout << "full long long" << endl;
          ts::Tensor<long long> test_full_ll = ts::full<long long>(3, rand_size, 1145141919810);
          cout << test_full_ll << endl
               << endl;

          cout << "full float" << endl;
          ts::Tensor<float> test_full_f = ts::full<float>(3, rand_size, 3.1415926);
          cout << test_full_f << endl
               << endl;

          cout << endl
               << endl
               << endl
               << endl;

          cout << "----------------- 1.4 Pattern Init -----------------" << endl;

          cout << "eye 3 3" << endl;
          ts::Tensor<int> test_eye_33 = ts::eye<int>(3, 3);
          cout << test_eye_33 << endl
               << endl;

          cout << "eye 4 5" << endl;
          ts::Tensor<int> test_eye_45 = ts::eye<int>(4, 5);
          cout << test_eye_45 << endl
               << endl;

          cout << "eye 5 2" << endl;
          ts::Tensor<int> test_eye_52 = ts::eye<int>(5, 2);
          cout << test_eye_52 << endl
               << endl;

          cout << endl
               << endl
               << endl
               << endl;
     }

     // 第二部分的测试
     cout << "----------------------------第二部分测试----------------------------" << endl;

     {
          // 示例用法
          int shape = 3;
          int size1[] = {2, 3, 4};
          int dtype = dtype_float;
          float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                           9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                           17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

          ts::Tensor<float> op_t1(shape, size1, dtype, data1);

          int size2[] = {2, 2, 4};
          float data2[] = {1, 2, 3, 4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13, 14, 15, 16};

          ts::Tensor<float> op_t2(shape, size2, dtype, data2);

          cout << "op_t1 : " << endl
               << endl
               << op_t1 << endl
               << endl;

          cout << "op_t2 : " << endl
               << endl
               << op_t2 << endl
               << endl;

          // 测试index_select
          cout << endl
               << "--------------------2.1: 测试index_select----------------------" << endl
               << endl;

          vector<int> index = {1};
          cout << "---2.1.1: 对于op_t1, index = {1}---" << endl
               << endl;
          ts::Tensor<float> index_op_t1 = op_t1(index);
          cout << index_op_t1 << endl;

          cout << endl
               << "---Memory sharing---"
               << endl;
          cout << "address of op_t1 = " << op_t1.data_ptr() << endl;
          cout << "address of index_op_t1 = " << index_op_t1.data_ptr() << endl
               << endl;

          cout << "---2.1.2: 对于op_t2, index = {1}, range = {1, 2}---" << endl
               << endl;
          ts::Tensor<float> index_op_t2 = op_t2(index, {1, 2});
          cout << index_op_t2 << endl;

          cout << endl
               << "---Memory sharing---"
               << endl;
          cout << "address of op_t2 = " << op_t2.data_ptr() << endl;
          cout << "address of index_op_t2 = " << index_op_t2.data_ptr() << endl
               << endl;

          // 测试join
          cout << endl
               << "--------------------2.2.1: 测试Cat, 将 op_t1 和 op_t2 沿着 dim = 1 方向进行拼接----------------------" << endl
               << endl;
          ts::Tensor<float> cat_op_t1 = cat(op_t1, op_t2, 1);
          cout << cat_op_t1 << endl;

          // 测试tile
          cout << endl
               << "--------------------2.2.2: 测试tile, 将 op_t2 按照{2, 1, 2, 3}进行拼接----------------------" << endl
               << endl;
          vector<int> reps = {2, 1, 2, 3};
          ts::Tensor<float> tile_op_t2 = tile(op_t2, reps);
          cout << tile_op_t2 << endl;

          // 测试mutating1
          cout << endl
               << "--------------------2.3: 测试mutating1, op_t2(1) = 1----------------------" << endl
               << endl;
          vector<int> goal;
          vector<float> data = {1};
          goal.push_back(1);
          op_t2(goal) = data;
          cout << op_t2 << endl;

          // 测试mutating2
          cout << endl
               << "--------------------2.3: 测试mutating2, op_t2({0, 1}, {0, 2}) = {100, 200}, 其中{0, 1} 为多级索引, {0, 2} 为范围----------------------" << endl
               << endl;
          vector<int> goal2 = {0, 1};
          pair<int, int> range = {0, 2};
          vector<float> data_2 = {100, 200};
          op_t2(goal2, range) = data_2;
          cout << op_t2 << endl;

          // 测试transpose
          cout << endl
               << "--------------------2.4.1: 测试transpose----------------------" << endl
               << endl;
          cout << "---将op_t1的第1维和第2维交换, 并拷贝到 transpose_op_t1 ---" << endl
               << endl;
          ts::Tensor<float> transpose_op_t1 = op_t1.tensor_transpose(2, 1);
          cout << transpose_op_t1 << endl;

          cout << endl
               << "Memory sharing"
               << endl;
          cout << "address of op_t1 = " << op_t1.data_ptr() << endl;
          cout << "address of transpose_op_t1 = " << transpose_op_t1.data_ptr() << endl
               << endl;

          // 测试permute
          cout << endl
               << "--------------------2.4.2: 测试permute. 将op_t1按照 {2, 0, 1} 转置后浅赋值给 permute_op_t1 ----------------------" << endl
               << endl;
          int permute[] = {2, 0, 1};
          ts::Tensor<float> permute_op_t1 = op_t1.tensor_permute(permute);
          cout << permute_op_t1 << endl;

          cout << endl
               << "Memory sharing"
               << endl;
          cout << "address of op_t1 = " << op_t1.data_ptr() << endl;
          cout << "address of permute_op_t1 = " << permute_op_t1.data_ptr() << endl
               << endl;

          // 测试view
          cout << endl
               << "--------------------2.5: 测试view, 将原本size为 {2, 2, 4} 的op_t2 变为 {2, 2, 2, 2} 的 view_t ----------------------" << endl
               << endl;
          vector<int> size = {2, 2, 2, 2};
          ts::Tensor<float> view_t = op_t2.view(size);
          cout << view_t << endl;

          cout << endl
               << "Memory sharing"
               << endl;
          cout << "address of op_t2 = " << op_t2.data_ptr() << endl;
          cout << "address of view_t = " << view_t.data_ptr() << endl
               << endl;
     }

    {
        int size[] = { 3, 2 };
        double data1[] = { 0.1, 1.2, 2.2, 3.1, 4.9, 5.2 };
        double data2[] = { 0.2, 1.3, 2.2, 3.2, 4.8, 5.2 };

        ts::Tensor<double> t1(2, size, dtype_double, data1);
        ts::Tensor<double> t2(2, size, dtype_double, data2);

        cout << "----------------- 3.1 testing add -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.add(t2):\n" << t1.add(t2) << "\n" << endl;
        cout << "ts::add(t1, t2):\n" << ts::add(t1, t2) << "\n" << endl;
        cout << "t1 + t2:\n" << (t1 + t2) << "\n" << endl;
        cout << "t1.add(1.8):\n" << t1.add(1.8) << "\n" << endl;
        cout << "ts::add(t1, 1.8):\n" << ts::add(t1, 1.8) << "\n" << endl;
        cout << "ts::add(1.8, t1):\n" << ts::add(1.8, t1) << "\n" << endl;
        cout << "t1 + 1.8:\n" << (t1 + 1.8) << "\n" << endl;
        cout << "1.8 + t1:\n" << (1.8 + t1) << "\n" << endl;

        cout << "\n";
        cout << "----------------- 3.1 testing sub -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.sub(t2):\n" << t1.sub(t2) << "\n" << endl;
        cout << "ts::sub(t1, t2):\n" << ts::sub(t1, t2) << "\n" << endl;
        cout << "t1 - t2:\n" << (t1 - t2) << "\n" << endl;
        cout << "t1.sub(1.8):\n" << t1.sub(1.8) << "\n" << endl;
        cout << "ts::sub(t1, 1.8):\n" << ts::sub(t1, 1.8) << "\n" << endl;
        cout << "ts::sub(1.8, t1):\n" << ts::sub(1.8, t1) << "\n" << endl;
        cout << "t1 - 1.8:\n" << (t1 - 1.8) << "\n" << endl;
        cout << "1.8 - t1:\n" << (1.8 - t1) << "\n" << endl;

        cout << "----------------- 3.1 testing mul -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.mul(t2):\n" << t1.mul(t2) << "\n" << endl;
        cout << "ts::mul(t1, t2):\n" << ts::mul(t1, t2) << "\n" << endl;
        cout << "t1 * t2:\n" << (t1 * t2) << "\n" << endl;
        cout << "t1.mul(1.8):\n" << t1.mul(1.8) << "\n" << endl;
        cout << "ts::mul(t1, 1.8):\n" << ts::mul(t1, 1.8) << "\n" << endl;
        cout << "ts::mul(1.8, t1):\n" << ts::mul(1.8, t1) << "\n" << endl;
        cout << "t1 * 1.8:\n" << (t1 * 1.8) << "\n" << endl;
        cout << "1.8 * t1:\n" << (1.8 * t1) << "\n" << endl;

        cout << "----------------- 3.1 testing div -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.div(t2):\n" << t1.div(t2) << "\n" << endl;
        cout << "ts::div(t1, t2):\n" << ts::div(t1, t2) << "\n" << endl;
        cout << "t1 / t2:\n" << (t1 / t2) << "\n" << endl;
        cout << "t1.div(1.8):\n" << t1.div(1.8) << "\n" << endl;
        cout << "ts::div(t1, 1.8):\n" << ts::div(t1, 1.8) << "\n" << endl;
        cout << "ts::div(1.8, t1):\n" << ts::div(1.8, t1) << "\n" << endl;
        cout << "t1 / 1.8:\n" << (t1 / 1.8) << "\n" << endl;
        cout << "1.8 / t1:\n" << (1.8 / t1) << "\n" << endl;

        cout << "----------------- 3.1 testing log -----------------" << endl;
        cout << "t1:\n" << t1 << "\n" << endl;
        cout << "t1.log():\n" << t1.log() << endl;
    }

    {
        // 3.2
        // sum, mean, max, min
        int size1[] = { 3, 2 };
        double data1[] = { 0.1, 1.2, 2.2, 3.1, 4.9, 5.2 };
        ts::Tensor<double> t1(2, size1, dtype_double, data1);

        int size2[] = { 3, 4, 5 };
        double data2[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 };
        ts::Tensor<double> t2(3, size2, dtype_double, data2);

        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "----------------- 3.2 testing sum -----------------" << endl;
        cout << "t1.sum(0):\n" << t1.sum(0) << endl;
        cout << "sum(t1,1):\n" << sum(t1,1) << endl;

        cout << "t2.sum(0):\n" << t2.sum(0) << endl;
        cout << "t2.sum(1):\n" << t2.sum(1) << endl;
        cout << "t2.sum(2):\n" << t2.sum(2) << endl;
        cout << "----------------- 3.2 testing mean -----------------" << endl;
        cout << "t1.mean(0):\n" << t1.mean(0) << endl;
        cout << "mean(t1,1):\n" << mean(t1,1) << endl;
        cout << "t2.mean(0):\n" << t2.mean(0) << endl;
        cout << "t2.mean(1):\n" << t2.mean(1) << endl;
        cout << "t2.mean(2):\n" << t2.mean(2) << endl;
        cout << "----------------- 3.2 testing max -----------------" << endl;
        cout << "t1.max(0):\n" << t1.max(0) << endl;
        cout << "max(t1,1):\n" << max(t1,1) << endl;
        cout << "t2.max(0):\n" << t2.max(0) << endl;
        cout << "t2.max(1):\n" << t2.max(1) << endl;
        cout << "t2.max(2):\n" << t2.max(2) << endl;
        cout << "----------------- 3.2 testing min -----------------" << endl;
        cout << "t1.min(0):\n" << t1.min(0) << endl;
        cout << "min(t1,1):\n" << min(t1,1) << endl;
        cout << "t2.min(0):\n" << t2.min(0) << endl;
        cout << "t2.min(1):\n" << t2.min(1) << endl;
        cout << "t2.min(2):\n" << t2.min(2) << endl;
    }

    {
        int size[] = { 3, 2 };
        double data1[] = { 0.1, 1.2, 2.2, 3.1, 4.9, 5.2 };
        double data2[] = { 0.2, 1.3, 2.2, 2.2, 4.8, 5.2 };

        ts::Tensor<double> t1(2, size, dtype_double, data1);
        ts::Tensor<double> t2(2, size, dtype_double, data2);

        cout << "\n";
        cout << "----------------- 3.3 testing eq -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.eq(t2):\n" << t1.eq(t2) << "\n" << endl;
        cout << "ts::eq(t1, t2):\n" << ts::eq(t1, t2) << "\n" << endl;
        cout << "t1 == t2:\n" << (t1 == t2) << "\n" << endl;
        cout << "t2.eq(2.2):\n" << t2.eq(2.2) << "\n" << endl;
        cout << "ts::eq(t2, 2.2):\n" << ts::eq(t2, 2.2) << "\n" << endl;
        cout << "t2 == 2.2:\n" << (t2 == 2.2) << "\n" << endl;
        cout << "ts::eq(2.2, t2):\n" << ts::eq(2.2, t2) << "\n" << endl;
        cout << "2.2 == t2:\n" << (2.2 == t2) << "\n" << endl;

        // ne
        cout << "\n";
        cout << "----------------- 3.3 testing ne -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.ne(t2):\n" << t1.ne(t2) << "\n" << endl;
        cout << "ts::ne(t1, t2):\n" << ts::ne(t1, t2) << "\n" << endl;
        cout << "t1 != t2:\n" << (t1 != t2) << "\n" << endl;
        cout << "t2.ne(2.2):\n" << t2.ne(2.2) << "\n" << endl;
        cout << "ts::ne(t2, 2.2):\n" << ts::ne(t2, 2.2) << "\n" << endl;
        cout << "t2 != 2.2:\n" << (t2 != 2.2) << "\n" << endl;
        cout << "ts::ne(2.2, t2):\n" << ts::ne(2.2, t2) << "\n" << endl;
        cout << "2.2 != t2:\n" << (2.2 != t2) << "\n" << endl;

        // gt
        cout << "\n";
        cout << "----------------- 3.3 testing gt -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.gt(t2):\n" << t1.gt(t2) << "\n" << endl;
        cout << "ts::gt(t1, t2):\n" << ts::gt(t1, t2) << "\n" << endl;
        cout << "t1 > t2:\n" << (t1 > t2) << "\n" << endl;
        cout << "t2.gt(2.2):\n" << t2.gt(2.2) << "\n" << endl;
        cout << "ts::gt(t2, 2.2):\n" << ts::gt(t2, 2.2) << "\n" << endl;
        cout << "t2 > 2.2:\n" << (t2 > 2.2) << "\n" << endl;
        cout << "ts::gt(2.2, t2):\n" << ts::gt(2.2, t2) << "\n" << endl;
        cout << "2.2 > t2:\n" << (2.2 > t2) << "\n" << endl;

        // ge
        cout << "\n";
        cout << "----------------- 3.3 testing ge -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.ge(t2):\n" << t1.ge(t2) << "\n" << endl;
        cout << "ts::ge(t1, t2):\n" << ts::ge(t1, t2) << "\n" << endl;
        cout << "t1 >= t2:\n" << (t1 >= t2) << "\n" << endl;
        cout << "t2.ge(2.2):\n" << t2.ge(2.2) << "\n" << endl;
        cout << "ts::ge(t2, 2.2):\n" << ts::ge(t2, 2.2) << "\n" << endl;
        cout << "t2 >= 2.2:\n" << (t2 >= 2.2) << "\n" << endl;
        cout << "ts::ge(2.2, t2):\n" << ts::ge(2.2, t2) << "\n" << endl;
        cout << "2.2 >= t2:\n" << (2.2 >= t2) << "\n" << endl;

        // lt
        cout << "\n";
        cout << "----------------- 3.3 testing lt -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.lt(t2):\n" << t1.lt(t2) << "\n" << endl;
        cout << "ts::lt(t1, t2):\n" << ts::lt(t1, t2) << "\n" << endl;
        cout << "t1 < t2:\n" << (t1 < t2) << "\n" << endl;
        cout << "t2.lt(2.2):\n" << t2.lt(2.2) << "\n" << endl;
        cout << "ts::lt(t2, 2.2):\n" << ts::lt(t2, 2.2) << "\n" << endl;
        cout << "t2 < 2.2:\n" << (t2 < 2.2) << "\n" << endl;
        cout << "ts::lt(2.2, t2):\n" << ts::lt(2.2, t2) << "\n" << endl;
        cout << "2.2 < t2:\n" << (2.2 < t2) << "\n" << endl;

        // le
        cout << "\n";
        cout << "----------------- 3.3 testing le -----------------" << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << "\n" << endl;
        cout << "t1.le(t2):\n" << t1.le(t2) << "\n" << endl;
        cout << "ts::le(t1, t2):\n" << ts::le(t1, t2) << "\n" << endl;
        cout << "t1 <= t2:\n" << (t1 <= t2) << "\n" << endl;
        cout << "t2.le(2.2):\n" << t2.le(2.2) << "\n" << endl;
        cout << "ts::le(t2, 2.2):\n" << ts::le(t2, 2.2) << "\n" << endl;
        cout << "t2 <= 2.2:\n" << (t2 <= 2.2) << "\n" << endl;
        cout << "ts::le(2.2, t2):\n" << ts::le(2.2, t2) << "\n" << endl;
        cout << "2.2 <= t2:\n" << (2.2 <= t2) << "\n" << endl;
    }

    {
        int size[] = { 3, 3 };
        double data[] = { -0.6274, -0.8041, 0.2895, 0.2361, 0.2403, 0.0249,
            -1.1858, 0.0942, 0.8567 };
        ts::Tensor<double> t(2, size, dtype_double, data);
        cout << "\n";
        cout << "----------------- 3.4 testing einsum 1), 2), 4), 5) -----------------"
             << endl;
        cout << "t:\n" << t << endl;
        // ii->i
        cout << "\n";
        cout << "1) Extracting elements along diagonal:\n";
        cout << "einsum('ii->i', t):\n" << einsum("ii->i", t) << endl;
        // ij->ji
        cout << "\n";
        cout << "2) Transpose\n";
        cout << "einsum('ij->ji', t):\n" << einsum("ij->ji", t) << endl;
        cout << "\n";
        // mn->nm
        cout << "einsum('mn->nm', t):\n" << einsum("mn->nm", t) << endl;
        cout << "\n";
        // ij->
        cout << "4) Reduce sum\n";
        cout << "einsum('ij->', t):\n" << einsum("ij->", t) << endl;
        cout << "\n";
        // ij->j
        cout << "5) Sum along dimension,\n";
        cout << "einsum('ij->j', t):\n" << einsum("ij->j", t) << endl;
        cout << "\n";
    }
    {
        int size1[] = { 3, 4 };
        double data1[] = { 0.2164, 0.5688, 0.9395, 0.5708, 0.3704, 0.9071,
            0.2359, 0.4737, 0.4256, 0.5464, 0.2862, 0.0466 };
        ts::Tensor<double> t1(2, size1, dtype_double, data1);

        int size2[] = { 4 };
        double data2[] = { 0.1809, 0.5410, 0.1090, 0.2674 };
        ts::Tensor<double> t2(1, size2, dtype_double, data2);

        int size3[] = { 4, 3 };
        double data3[] = { 0.6041, 0.3206, 0.7300, 0.7074, 0.0419, 0.0804,
            0.6089, 0.1982, 0.8061, 0.5899, 0.0815, 0.7246 };
        ts::Tensor<double> t3(2, size3, dtype_double, data3);

        cout << "\n";
        cout << "----------------- 3.4 testing einsum 6), 7) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "t3:\n" << t3 << endl;
        cout << "\n";
        cout << "6) Matrix and vector mul:\n";
        // ik,k->i
        cout << "einsum('ik,k->i', t1, t2):\n" << einsum("ik,k->i", t1, t2)
             << endl;
        cout << "\n";

        // ik,kj->ij  t1, t3
        cout << "7) Matrix mul:\n";
        cout << "einsum('ik,kj->ij', t1, t3):\n" << einsum("ik,kj->ij", t1, t3)
             << endl;
        cout << "\n";
    }

    {
        int size1[] = { 6 };
        double data1[] = { 0.2164, 0.5688, 0.9395, 0.5708, 0.3704, 0.9071 };
        ts::Tensor<double> t1(1, size1, dtype_double, data1);

        int size3[] = { 6 };
        double data3[] = { 0.6041, 0.3206, 0.7300, 0.7074, 0.0419, 0.0804 };
        ts::Tensor<double> t2(1, size3, dtype_double, data3);

        cout << "\n";
        cout << "----------------- 3.4 testing einsum 8) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "\n";
        // i,i->
        cout << "8) Dot product\n";
        cout << "einsum('i,i->', t1, t2):\n" << einsum("i,i->", t1, t2)
             << endl;
        cout << "\n";
    }
    {
        int size1[] = {3, 4};
        double data1[] = {0.2164, 0.5688, 0.9395, 0.5708, 0.3704, 0.9071,
                          0.2359, 0.4737, 0.4256, 0.5464, 0.2862, 0.0466};
        ts::Tensor<double> t1(2, size1, dtype_double, data1);

        int size2[] = {3, 4};
        double data2[] = {0.1809, 0.5410, 0.1090, 0.2674, 0.6041, 0.3206,
                          0.7300, 0.7074, 0.0419, 0.0804, 0.6089, 0.1982};
        ts::Tensor<double> t2(2, size2, dtype_double, data2);

        cout << "\n";
        cout << "----------------- 3.4 testing einsum 9) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "\n";
        // ij,ij->
        cout << "9) Pointwise mul and reduce sum\n";
        cout << "einsum('ij,ij->', t1, t2):\n" << einsum("ij,ij->", t1, t2)
             << endl;
        cout << "\n";
    }
    {
        // i,j->ij
        int size1[] = { 3 };
        double data1[] = { 0.2164, 0.5688, 0.9395 };
        ts::Tensor<double> t1(1, size1, dtype_double, data1);
        int size2[] = { 4 };
        double data2[] = { 0.5708, 0.3704, 0.9071, 0.2359 };
        ts::Tensor<double> t2(1, size2, dtype_double, data2);

        cout << "\n";
        cout << "----------------- 3.4 testing einsum 10) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "\n";
        cout << "10) Outer product\n";
        cout << "einsum('i,j->ij', t1, t2):\n" << einsum("i,j->ij", t1, t2)
             << endl;
        cout << "\n";
    }

    {
        // ijk,ikl->ijl
        int size1[] = { 3, 4, 5 };
        double data1[] = { 0.7475, 0.2420, 0.1905, 0.9840, 0.2215, 0.6647, 0.6045, 0.7222, 0.4154,
        0.5037, 0.7241, 0.0319, 0.0975, 0.2166, 0.2514, 0.2391, 0.4143, 0.7320,
        0.9538, 0.0899, 0.6249, 0.2612, 0.5859, 0.8104, 0.5143, 0.1083, 0.8269,
        0.4314, 0.6574, 0.8443, 0.9273, 0.8301, 0.8928, 0.9377, 0.6064, 0.1680,
        0.1360, 0.8315, 0.1189, 0.5792, 0.9235, 0.5039, 0.8804, 0.8270, 0.5636,
        0.1966, 0.8364, 0.1656, 0.6791, 0.0840, 0.1440, 0.6058, 0.9837, 0.1285,
        0.5152, 0.4353, 0.2792, 0.6164, 0.0983, 0.8235 };
        ts::Tensor<double> t1(3, size1, dtype_double, data1);

        int size2[] = { 3, 5, 2 };
        double data2[] = { 0.3080, 0.2942, 0.1771, 0.1675, 0.1604, 0.2737, 0.8589, 0.3297, 0.0218,
        0.3657, 0.9732, 0.7385, 0.1233, 0.4695, 0.1999, 0.0510, 0.1127, 0.1438,
        0.5219, 0.7963, 0.2384, 0.0884, 0.2820, 0.9798, 0.5506, 0.2151, 0.5204,
        0.5458, 0.6886, 0.5882 };
        ts::Tensor<double> t2(3, size2, dtype_double, data2);

        cout << "\n";
        cout << "----------------- 3.4 testing einsum 11) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "\n";
        cout << "11) Batch matrix mul\n";
        cout << "einsum('ijk,ikl->ijl', t1, t2):\n" << einsum("ijk,ikl->ijl", t1, t2)
             << endl;
        cout << "\n";
    }

    {
        // pqrs,tuqvr->pstuv
        int size1[] = { 2, 3, 3, 4 };
        double data1[] = { 0.2729, 0.9709, 0.2534, 0.1098, 0.4694, 0.8283,
            0.6258, 0.4763, 0.6209, 0.5457, 0.6035, 0.6015, 0.5163, 0.7208,
            0.7135, 0.7038, 0.2723, 0.4679, 0.0127, 0.0915, 0.5699, 0.4765,
            0.6856, 0.8199, 0.6746, 0.9718, 0.6641, 0.2906, 0.8492, 0.0980,
            0.5609, 0.3358, 0.1992, 0.6030, 0.8240, 0.7837, 0.4130, 0.2668,
            0.1504, 0.9733, 0.9515, 0.5840, 0.5659, 0.1637, 0.2658, 0.2519,
            0.1837, 0.2568, 0.8910, 0.3223, 0.4161, 0.6597, 0.6888, 0.6115,
            0.3052, 0.1518, 0.3345, 0.3663, 0.1732, 0.5728, 0.4113, 0.8895,
            0.5957, 0.1411, 0.3236, 0.5834, 0.3013, 0.1013, 0.0156, 0.6714,
            0.9324, 0.7913 };
        ts::Tensor<double> t1(4, size1, dtype_double, data1);

        int size2[] = { 3, 2, 3, 4, 3 };
        double data2[] = { 0.9627, 0.6719, 0.6777, 0.8530, 0.4622, 0.5804,
            0.1863, 0.3729, 0.1315, 0.1413, 0.8996, 0.4034, 0.6090, 0.6922,
            0.2529, 0.6881, 0.9567, 0.0129, 0.9290, 0.4677, 0.0178, 0.4069,
            0.5031, 0.9346, 0.4490, 0.8495, 0.0923, 0.0345, 0.6827, 0.9780,
            0.3886, 0.0855, 0.6775, 0.7474, 0.1034, 0.1131, 0.6632, 0.2300,
            0.4707, 0.5231, 0.8979, 0.2612, 0.6941, 0.0814, 0.6417, 0.9336,
            0.9007, 0.7563, 0.2808, 0.3146, 0.3268, 0.6989, 0.6289, 0.9619,
            0.8761, 0.1856, 0.3378, 0.2370, 0.8426, 0.2152, 0.2492, 0.9020,
            0.7972, 0.6395, 0.3853, 0.2558, 0.4736, 0.6177, 0.7403, 0.8576,
            0.6529, 0.1128, 0.5875, 0.7218, 0.7765, 0.0823, 0.7393, 0.7281,
            0.5569, 0.8374, 0.5893, 0.9431, 0.5472, 0.9591, 0.9434, 0.9105,
            0.2237, 0.4274, 0.5569, 0.1389, 0.2286, 0.6436, 0.6472, 0.9780,
            0.9911, 0.9204, 0.3622, 0.9622, 0.8969, 0.6399, 0.9433, 0.3323,
            0.1158, 0.8814, 0.9198, 0.4874, 0.8867, 0.6108, 0.5678, 0.2339,
            0.8166, 0.8242, 0.2140, 0.5135, 0.9296, 0.9503, 0.7716, 0.5973,
            0.9200, 0.9716, 0.1863, 0.1211, 0.2803, 0.2406, 0.3629, 0.5806,
            0.5097, 0.6703, 0.6591, 0.3901, 0.2902, 0.0629, 0.9209, 0.0797,
            0.3269, 0.4131, 0.1168, 0.7850, 0.6855, 0.7958, 0.5554, 0.8128,
            0.8838, 0.9206, 0.4277, 0.0276, 0.9018, 0.2563, 0.0818, 0.7899,
            0.2784, 0.3405, 0.7473, 0.8566, 0.4055, 0.8043, 0.5774, 0.6414,
            0.9214, 0.1546, 0.7883, 0.9434, 0.5076, 0.5464, 0.4802, 0.5306,
            0.2603, 0.9650, 0.7713, 0.3977, 0.2189, 0.2747, 0.5012, 0.5638,
            0.7854, 0.0730, 0.8901, 0.9167, 0.1237, 0.7609, 0.4108, 0.5486,
            0.7437, 0.8047, 0.6408, 0.7050, 0.4823, 0.6296, 0.2072, 0.5536,
            0.1922, 0.5408, 0.9299, 0.5381, 0.4533, 0.4776, 0.1334, 0.0599,
            0.3162, 0.2735, 0.6917, 0.4130, 0.6812, 0.0181, 0.5298, 0.9826,
            0.6020, 0.7374, 0.8755, 0.0028, 0.3099, 0.2916, 0.0672, 0.4212,
            0.1689, 0.4229 };
        ts::Tensor<double> t2(5, size2, dtype_double, data2);

        cout << "\n";
        cout << "----------------- 3.4 testing einsum 12) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "\n";
        cout << "12) Tensor contraction\n";
        cout << "einsum('pqrs,tuqvr->pstuv', t1, t2): p=2,s=4,t=3,u=2,v=4\n"
             << einsum("pqrs,tuqvr->pstuv", t1, t2) << endl;
        cout << "\n";
    }

    {
        // ik,jkl,il->ij
        int size1[] = { 3, 4 };
        double data1[] = { 0.4447, 0.2936, 0.2852, 0.1430, 0.7893, 0.2406,
            0.7225, 0.9517, 0.2114, 0.6327, 0.9359, 0.9987 };
        ts::Tensor<double> t1(2, size1, dtype_double, data1);

        int size2[] = { 4, 4, 2 };
        double data2[] = { 0.4438, 0.2215, 0.3127, 0.8174, 0.2970, 0.6483,
            0.3394, 0.3280, 0.4439, 0.8616, 0.4786, 0.6281, 0.7202, 0.2462,
            0.2810, 0.1781, 0.0733, 0.6727, 0.5010, 0.6281, 0.3971, 0.7764,
            0.2451, 0.5115, 0.9404, 0.4647, 0.7387, 0.5485, 0.4958, 0.4800,
            0.9261, 0.0035 };
        ts::Tensor<double> t2(3, size2, dtype_double, data2);
        int size3[] = { 3,2 };
        double data3[]={0.4447, 0.2936, 0.2852, 0.1430, 0.7893, 0.2406, };
            ts::Tensor<double>t3(2,size3,dtype_double,data3);
        cout << "\n";
        cout << "----------------- 3.4 testing einsum 13) -----------------"
             << endl;
        cout << "t1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
         cout << "t3:\n" << t3 << endl;
        cout << "\n";
        cout << "13) Bilinear transformation\n";
        cout << "einsum('ik,jkl,il->ij', t1, t2,t3):\n"
             << einsum("ik,jkl,il->ij", t1, t2,t3) << endl;
        cout << "\n";
    }


     // bonus部分
     {
          int shape = 3;
          int size1[] = {2, 3, 4};
          int dtype = dtype_float;
          float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                           9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                           17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

          ts::Tensor<float> op_t1(shape, size1, dtype, data1);

          int size2[] = {2, 2, 4};
          float data2[] = {1, 2, 3, 4, 5, 6, 7, 8,
                           9, 10, 11, 12, 13, 14, 15, 16};

          ts::Tensor<float> op_t2(shape, size2, dtype, data2);

          // 测试Serialization
          cout << endl
               << "--------------------bonus1: 测试Serialization, 将op_t1序列化后, 再反序列化为 deserialized_op_t1 ----------------------" << endl
               << endl;
          const string filename = "tensor_data.bin";
          ts::save(op_t1, filename);

          // 从文件加载Tensor
          Tensor<float> loadedTensor = ts::load<float>(filename);

          // 打印加载的Tensor
          cout << loadedTensor << endl;
     }

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
     
     
}
return 0;
}