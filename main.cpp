#include <iostream>
#include "tensor.cpp"
using namespace std;
using namespace ts;

int main() {
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
    for (int i = 0; i < shape; ++i) {
        cout << tensor.getSize()[i] << " ";
    }
    cout << endl;

    cout << "Total Size: " << tensor.getTotalSize() << endl;

    cout << "dtype: " << tensor.type_name() << endl;

    cout << "data_ptr: " << tensor.data_ptr() << endl;

    cout << endl << tensor << endl;


    cout << endl << endl << endl << endl;


    int * p = new int[3]{1, 2, 0};
    tensor.setPermute(p);

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

    cout << endl << endl << endl << endl;




    cout << "rand bool" << endl;
    ts::Tensor<bool> test_rand_b = ts::rand<bool>(3, rand_size);
    cout << test_rand_b << endl << endl;

    cout << "rand char" << endl;
    ts::Tensor<char> test_rand_c = ts::rand<char>(3, rand_size);
    cout << test_rand_c << endl << endl;

    cout << "rand int" << endl;
    ts::Tensor<int> test_rand_i = ts::rand<int>(3, rand_size);
    cout << test_rand_i << endl << endl;

    cout << "rand long long" << endl;
    ts::Tensor<long long> test_rand_ll = ts::rand<long long>(3, rand_size);
    cout << test_rand_ll << endl << endl;

    cout << "rand float" << endl;
    ts::Tensor<float> test_rand_f = ts::rand<float>(3, rand_size);
    cout << test_rand_f << endl << endl;

    cout << "rand double" << endl;
    ts::Tensor<double> test_rand_d = ts::rand<double>(3, rand_size);
    cout << test_rand_d << endl << endl;

    cout << endl << endl << endl << endl;




    cout << "zeros bool" << endl;
    ts::Tensor<bool> test_zeros_b = ts::zeros<bool>(3, rand_size);
    cout << test_zeros_b << endl << endl;

    cout << "zeros char" << endl;
    ts::Tensor<char> test_zeros_c = ts::zeros<char>(3, rand_size);
    cout << test_zeros_c << endl << endl;

    cout << "zeros int" << endl;
    ts::Tensor<int> test_zeros_i = ts::zeros<int>(3, rand_size);
    cout << test_zeros_i << endl << endl;

    cout << endl << endl << endl << endl;



    cout << "ones bool" << endl;
    ts::Tensor<bool> test_ones_b = ts::ones<bool>(3, rand_size);
    cout << test_ones_b << endl << endl;

    cout << "ones char" << endl;
    ts::Tensor<char> test_ones_c = ts::ones<char>(3, rand_size);
    cout << test_ones_c << endl << endl;

    cout << "ones int" << endl;
    ts::Tensor<int> test_ones_i = ts::ones<int>(3, rand_size);
    cout << test_ones_i << endl << endl;

    cout << endl << endl << endl << endl;




    cout << "full bool" << endl;
    ts::Tensor<bool> test_full_b = ts::full<bool>(3, rand_size, true);
    cout << test_full_b << endl << endl;

    cout << "full char" << endl;
    ts::Tensor<char> test_full_c = ts::full<char>(3, rand_size, 'a');
    cout << test_full_c << endl << endl;

    cout << "full int" << endl;
    ts::Tensor<int> test_full_i = ts::full<int>(3, rand_size, 1234);
    cout << test_full_i << endl << endl;

    cout << "full long long" << endl;
    ts::Tensor<long long> test_full_ll = ts::full<long long>(3, rand_size, 1145141919810);
    cout << test_full_ll << endl << endl;

    cout << "full float" << endl;
    ts::Tensor<float> test_full_f = ts::full<float>(3, rand_size, 3.1415926);
    cout << test_full_f << endl << endl;

    cout << endl << endl << endl << endl;



    cout << "eye 3 3" << endl;
    ts::Tensor<int> test_eye_33 = ts::eye<int>(3, 3);
    cout << test_eye_33 << endl << endl;

    cout << "eye 4 5" << endl;
    ts::Tensor<int> test_eye_45 = ts::eye<int>(4, 5);
    cout << test_eye_45 << endl << endl;

    cout  << "eye 5 2" << endl;
    ts::Tensor<int> test_eye_52 = ts::eye<int>(5, 2);
    cout << test_eye_52 << endl << endl;

    cout << endl << endl << endl << endl;

    }

    {
        int size[] = { 3, 2 };
        double data1[] = { 0.1, 1.2, 2.2, 3.1, 4.9, 5.2 };
        double data2[] = { 0.2, 1.3, 2.2, 3.2, 4.8, 5.2 };

        ts::Tensor<double> t1(2, size, dtype_double, data1);
        ts::Tensor<double> t2(2, size, dtype_double, data2);

        cout << "\n\nt1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "t1 + t2:\n" << (t1 + t2) << endl;
        cout << "t1 - t2:\n" << (t1 - t2) << endl;
        cout << "t1 * t2:\n" << (t1 * t2) << endl;
        cout << "t1 / t2:\n" << (t1 / t2) << endl;
        cout << "t1 + 1.3:\n" << (t1 + 1.3) << endl;
        cout << "t1 + 0.0:\n" << (t1 + 0.0) << endl;
        cout << "t1 * 0.0:\n" << (t1 * 0.0) << endl;
        cout << "t1 - 0.5:\n" << (t1 - 0.5) << endl;
        cout << "t1 * 2.0:\n" << (t1 * 2.0) << endl;
        cout << "t1 / 2.5:\n" << (t1 / 2.5) << endl;
        cout << "t1 == t2:\n" << (t1 == t2) << endl;
        cout << "t1 != t2:\n" << (t1 != t2) << endl;
        cout << "t1 < t2:\n" << (t1 < t2) << endl;
        cout << "t1 <= t2:\n" << (t1 <= t2) << endl;
        cout << "t1 > t2:\n" << (t1 > t2) << endl;
        cout << "t1 >= t2:\n" << (t1 >= t2) << endl;
        cout << "t1 < 1.0:\n" << (t1 < 1.0) << endl;
        cout << "t1 <= 2.2:\n" << (t1 <= 2.2) << endl;
        cout << "t1 > 1.0:\n" << (t1 > 1.0) << endl;
        cout << "t1 >= 2.2:\n" << (t1 >= 2.2) << endl;
    }

    {
        // sum, mean, max, min
        int size1[] = { 3, 2 };
        double data1[] = { 0.1, 1.2, 2.2, 3.1, 4.9, 5.2 };
        ts::Tensor<double> t1(2, size1, dtype_double, data1);

        cout << "t:\n" << t1 << endl;
        cout << "t.sum(0):\n" << t1.sum(0) << endl;
        cout << "t.sum(1):\n" << t1.sum(1) << endl;
        cout << "t.mean(0):\n" << t1.mean(0) << endl;
        cout << "t.mean(1):\n" << t1.mean(1) << endl;
        cout << "t.max(0):\n" << t1.max(0) << endl;
        cout << "t.max(1):\n" << t1.max(1) << endl;
        cout << "t.min(0):\n" << t1.min(0) << endl;
        cout << "t.min(1):\n" << t1.min(1) << endl;

        int size2[] = { 3, 4, 5 };
        double data2[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 };
        ts::Tensor<double> t2(3, size2, dtype_double, data2);

        cout << "t:\n" << t2 << endl;
        cout << "t.sum(0):\n" << t2.sum(0) << endl;
        cout << "t.sum(1):\n" << t2.sum(1) << endl;
        cout << "t.sum(2):\n" << t2.sum(2) << endl;
        cout << "t.mean(0):\n" << t2.mean(0) << endl;
        cout << "t.mean(1):\n" << t2.mean(1) << endl;
        cout << "t.mean(2):\n" << t2.mean(2) << endl;
        cout << "t.max(0):\n" << t2.max(0) << endl;
        cout << "t.max(1):\n" << t2.max(1) << endl;
        cout << "t.max(2):\n" << t2.max(2) << endl;
        cout << "t.min(0):\n" << t2.min(0) << endl;
        cout << "t.min(1):\n" << t2.min(1) << endl;
        cout << "t.min(2):\n" << t2.min(2) << endl;
    }

    {
        int size[] = { 3, 3, 3 };
        double data[] = { -0.6274, -0.8041, 0.2895, 0.2361, 0.2403, 0.0249,
            -1.1858, 0.0942, 0.8567, 0.7559, -0.0828, -1.0056, 0.1684, -0.4167,
            0.8091, -2.4442, 0.4162, -1.5877, -1.2795, -0.7375, -0.5490,
            0.4779, 0.2398, -0.3176, -0.7226, 0.1408, -0.3850 };
        ts::Tensor<double> t(3, size, dtype_double, data);
        cout << "\n\nt:\n" << t << endl;
        cout << "einsum('iii->i', t):\n" << einsum("iii->i", t) << endl;
        cout << "einsum('iij->i', t):\n" << einsum("iij->i", t) << endl;
        cout << "einsum('iij->j', t):\n" << einsum("iij->j", t) << endl;
        cout << "einsum('iji->j', t):\n" << einsum("iji->j", t) << endl;
        cout << "einsum('ijk->i', t):\n" << einsum("ijk->i", t) << endl;
        cout << "einsum('ijk->ij', t):\n" << einsum("ijk->ij", t) << endl;
        cout << "einsum('ijk->k', t):\n" << einsum("ijk->k", t) << endl;
        cout << "einsum('ijk->ijk', t):\n" << einsum("ijk->ijk", t) << endl;
        cout << "einsum('ijk->kji', t):\n" << einsum("ijk->kji", t) << endl;
    }

    {
        int size1[] = { 3, 2, 4 };
        int size2[] = { 3, 4, 3 };
        double data1[] = { 0.4547, 0.8822, 0.8478, 0.6656, 0.1323, 0.7476,
            0.2541, 0.7416, 0.5142, 0.2962, 0.5716, 0.2463, 0.5226, 0.1680,
            0.8201, 0.5469, 0.6911, 0.3628, 0.8087, 0.9809, 0.3090, 0.3547,
            0.1759, 0.9699 };
        double data2[] = { 0.1592, 0.8073, 0.7585, 0.1458, 0.2934, 0.3100,
            0.1755, 0.4466, 0.5505, 0.7787, 0.4602, 0.0362, 0.5743, 0.6436,
            0.4860, 0.0690, 0.9785, 0.1603, 0.4690, 0.0975, 0.2710, 0.4157,
            0.1001, 0.6953, 0.6438, 0.1162, 0.7043, 0.4370, 0.3398, 0.5229,
            0.0482, 0.4570, 0.3660, 0.6159, 0.0598, 0.0279 };

        Tensor<double> t1(3, size1, dtype_double, data1);
        Tensor<double> t2(3, size2, dtype_double, data2);

        cout << "\n\nt1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "einsum('bij,bjk->bik', t1, t2):\n"
             << einsum("bij,bjk->bik", t1, t2) << endl;
        cout << "einsum('bij,bjb->ib', t1, t2):\n"
             << einsum("bij,bjb->ib", t1, t2) << endl;
        cout << "einsum('ijk,ijk->ijk', t1, t1):\n"
             << einsum("ijk,ijk->ijk", t1, t1) << endl;
    }

    {
        int size1[] = { 24 };
        int size2[] = { 24 };
        double data1[] = { 0.4547, 0.8822, 0.8478, 0.6656, 0.1323, 0.7476,
            0.2541, 0.7416, 0.5142, 0.2962, 0.5716, 0.2463, 0.5226, 0.1680,
            0.8201, 0.5469, 0.6911, 0.3628, 0.8087, 0.9809, 0.3090, 0.3547,
            0.1759, 0.9699 };
        double data2[] = { 0.1592, 0.8073, 0.7585, 0.1458, 0.2934, 0.3100,
            0.1755, 0.4466, 0.5505, 0.7787, 0.4602, 0.0362, 0.5743, 0.6436,
            0.4860, 0.0690, 0.9785, 0.1603, 0.4690, 0.0975, 0.2710, 0.4157,
            0.1001, 0.6953 };

        Tensor<double> t1(1, size1, dtype_double, data1);
        Tensor<double> t2(1, size2, dtype_double, data2);

        cout << "\n\nt1:\n" << t1 << endl;
        cout << "t2:\n" << t2 << endl;
        cout << "einsum('i->', t1):\n" << einsum("i->", t1) << endl;
        cout << "einsum('i,i->', t1, t2):\n"
             << einsum("i,i->", t1, t2) << endl;
    }

    return 0;
}