//
// Created by witek on 07.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_MAP3D_H
#define BUNDLE_ADJUSTMENT_MAP3D_H

#include "../headers/VirtualSensor.h"
#include "Matching.h"
#include <Eigen/Dense>
using namespace std;
using namespace cv;
using namespace Eigen;


struct frame1_geometry{
    Matrix4f pose;
    vector<Vector3f> points3d_local; // relative to the current frame
    vector<Vector3f> points3d_global; // relative to frame 1
};


vector<Vector3f> getLocalPoints3D(const frame_correspondences& correspondences, const Mat& depth_frame1,
                                  const Matrix3f& intrinsics);


vector<Vector3f> getGlobalPoints3D(const frame1_geometry& frame);


Matrix4f getExtrinsics(const Mat& E, const vector<Point2f>& matched_points1,
                             const vector<Point2f>& matched_points2, const Mat& intrinsics);




#endif //BUNDLE_ADJUSTMENT_MAP3D_H
