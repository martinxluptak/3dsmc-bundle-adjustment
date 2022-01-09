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
    Matrix4f extrinsics;
    vector<Vector3f> points3d_before; //for frame2 of the pair, before applying extrinsics
    vector<Vector3f> points3d_after; // for frame2 of the pair, after applying extrinsics
};


vector<Vector3f> getPoints3D_before(const frame_correspondences& correspondences, const Mat& depth_frame1,
                                    const Matrix3f& intrinsics);


vector<Vector3f> getPoints3D_after(const frame1_geometry& frame);


Matrix4f getExtrinsics(const Mat& E, const vector<Point2f>& matched_points1,
                             const vector<Point2f>& matched_points2, const Mat& intrinsics);




#endif //BUNDLE_ADJUSTMENT_MAP3D_H
