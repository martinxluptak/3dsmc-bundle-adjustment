//
// Created by witek on 07.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_MAP3D_H
#define BUNDLE_ADJUSTMENT_MAP3D_H

#include "../headers/VirtualSensor.h"
#include "CommonTypes.h"
#include "Matching.h"
#include <Eigen/Dense>
#include <sophus/se3.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;

vector<Vector3d> getLocalPoints3D(const vector<KeyPoint> &correspondences,
                                  const Mat &depth_frame1,
                                  const Matrix3f &intrinsics);

vector<Vector3d> getGlobalPoints3D(const KeyFrame &frame);

Sophus::SE3d getExtrinsics(const Mat &E, const vector<Point2d> &matched_points1,
                           const vector<Point2d> &matched_points2,
                           const Mat &intrinsics);

#endif // BUNDLE_ADJUSTMENT_MAP3D_H
