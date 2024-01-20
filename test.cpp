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

    // 测试index_select
    cout << endl
         << "--------------------测试index_select----------------------" << endl
         << endl;
    vector<int> index = {0};
    cout << "---index = {0}---" << endl
         << endl;
    ts::Tensor<float> t10 = t1(index);
    cout << t10 << endl
         << endl;
    cout << "---index = {0}, range = {0, 2}---" << endl
         << endl;
    ts::Tensor<float> t5 = t1(index, {0, 2});
    cout << t5 << endl;

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

    // 测试mutating1
    cout << endl
         << "--------------------测试mutating1----------------------" << endl
         << endl;
    vector<int> goal;
    vector<float> data = {1};
    goal.push_back(1);
    t2(goal) = data;
    cout << t2 << endl;

    // 测试mutating2
    cout << endl
         << "--------------------测试mutating2----------------------" << endl
         << endl;
    vector<int> goal2 = {0, 1};
    vector<float> data_2 = {100, 200};
    pair<int, int> range = {0, 2};
    t2(goal2, range) = data_2;
    cout << t2 << endl;

    // 测试transpose
    cout << endl
         << "--------------------测试transpose----------------------" << endl
         << endl;
    cout << "---将t1的第1维和第2维交换---" << endl
         << endl;
    ts::Tensor<float> t8 = t1.tensor_transpose(2, 1);
    cout << t8 << endl
         << endl;
    cout << "---将t1复原---" << endl;
    ts::Tensor<float> t9 = t8.tensor_transpose(2, 1);
    cout << endl
         << t9 << endl;

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
    vector<int> size = {1, 2, 2, 4};
    ts::Tensor<float> t7 = t2.view(size);
    cout << t7 << endl;

    return 0;
}
