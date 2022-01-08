//
// Created by witek on 07.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_MAP3D_H
#define BUNDLE_ADJUSTMENT_MAP3D_H

#include "../headers/VirtualSensor.h"
#include "opencv2/calib3d.hpp"
#include "Matching.h"
//#include <Eigen/Dense>
using namespace std;
using namespace cv;
using namespace Eigen;



vector<Vector3f> getPoints3D(const frame_correspondences& cs1,
                             const Mat& depth_frame1, const Matrix3f& intrinsics);


pair<Mat, Mat> getPose(const Mat& E, const vector<Point2f>& matched_points1,
                       const vector<Point2f>& matched_points2, const Mat& intrinsics);



#endif //BUNDLE_ADJUSTMENT_MAP3D_H
