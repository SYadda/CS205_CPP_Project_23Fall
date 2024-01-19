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
    float data1[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

    ts::Tensor<float> t1(shape, size1, dtype, data1);

    int size2[] = {2, 2, 4};
    float data2[] = {1, 2, 3, 4, 5, 6, 7, 8,
                     9, 10, 11, 12, 13, 14, 15, 16};

    ts::Tensor<float> t2(shape, size2, dtype, data2);

    cout << t1 << endl
         << endl;
    cout << t2 << endl
         << endl;

    // 测试join
    cout << endl
         << "--------------------测试join----------------------" << endl
         << endl;
    ts::Tensor<float> t3 = cat(t1, t2, 1);
    cout << t3 << endl;

    // 测试tile
    cout << endl
         << "--------------------测试tile----------------------" << endl
         << endl;
    vector<int> reps = {2, 2, 2};
    ts::Tensor<float> t4 = tile(t2, reps);
    cout << t4 << endl;

    // 测试index_select
    cout << endl
         << "--------------------测试index_select----------------------" << endl
         << endl;
    vector<int> index = {0};
    ts::Tensor<float> t5 = t1(index, {0, 2});
    cout << t5 << endl;

    // 测试permute
    cout << endl
         << "--------------------测试permute----------------------" << endl
         << endl;
    int permute[] = {2, 0, 1};
    ts::Tensor<float> t6 = t1.tensor_permute(permute);
    cout << t6 << endl;

    // 测试view
    cout << endl
         << "--------------------测试view----------------------" << endl
         << endl;
    vector<int> size = {2, 12};
    ts::Tensor<float> t7 = t1.view(size);
    cout << t7 << endl;

    return 0;
}