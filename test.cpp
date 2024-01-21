#include <iostream>
#include <vector>
#include "tensor.cpp"

using namespace std;
using namespace ts;

int main()
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

     cout << "---2.1.2: 对于op_t2, index = {1}, range = {0, 2}---" << endl
          << endl;
     ts::Tensor<float> index_op_t2 = op_t2(index, {0, 2});
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
          << "--------------------2.2.2: 测试tile, 将 op_t1 沿着{2, 2}进行拼接----------------------" << endl
          << endl;
     vector<int> reps = {2, 2};
     ts::Tensor<float> tile_op_t1 = tile(op_t2, reps);
     cout << tile_op_t1 << endl;

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

     // 测试Serialization
     cout << endl
          << "--------------------3.2.1: 测试Serialization, 将op_t1序列化后, 再反序列化为 deserialized_op_t1 ----------------------" << endl
          << endl;
     const string filename = "tensor_data.bin";
     ts::save(op_t1, filename);

     // 从文件加载Tensor
     Tensor<float> loadedTensor = ts::load<float>(filename);

     // 打印加载的Tensor
     cout << loadedTensor << endl;

     return 0;
}
