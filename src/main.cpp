#include <iostream>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
    cout << "Hello, World!" << endl;
    Matrix3f a = Matrix3f::Zero();
    cout << "Matrix[0][0]:" << a(0, 0) << endl;
    return 0;
}
