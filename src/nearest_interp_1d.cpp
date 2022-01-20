#ifndef NEAREST_INTERP_1D_CPP
#define NEAREST_INTERP_1D_CPP

#include <iostream>
#include <cmath>

using namespace std;

//****************************************************************************80

tuple<vector<double>, vector<int>> nearest_interp_1d(vector<double> xd, vector<double> yd, vector<double> xi)

//****************************************************************************80
//
//  Purpose:
//
//    NEAREST_INTERP_1D evaluates the nearest neighbor interpolant.
//
//  Discussion:
//
//    The nearest neighbor interpolant L(ND,XD,YD)(X) is the piecewise
//    constant function which interpolates the data (XD(I),YD(I)) for I = 1
//    to ND.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 September 2012
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int ND, the number of data points.
//    ND must be at least 1.
//
//    Input, double XD[ND], the data points.
//
//    Input, double YD[ND], the data values.
//
//    Input, int NI, the number of interpolation points.
//
//    Input, double XI[NI], the interpolation points.
//
//    Output, double NEAREST_INTERP_1D[NI], the interpolated values.
//
{
    int nd = xd.size();
    int ni = xi.size();
    double d;
    double d2;
    int i;
    int j;
    int k;
    vector<double> yi;
    vector<int> indices;

    for (i = 0; i < ni; i++) {
        k = 0;
        d = abs(xi[i] - xd[k]);
        for (j = 1; j < nd; j++) {
            d2 = abs(xi[i] - xd[j]);
            if (d2 < d) {
                k = j;
                d = d2;
            }
        }
        indices.push_back(k);
        yi.push_back(yd[k]);
    }

    return make_tuple(yi, indices);
}

#endif