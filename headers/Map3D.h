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

void updateLandmarks(KeyFrame &old_frame, KeyFrame &new_frame,
                     const vector<DMatch> &matches, Map3D &map,
                     LandmarkId &landmark_id);

void addNewLandmark(KeyFrame &kf_1, KeyFrame &kf_2, const DMatch &match,
                    Map3D &map, const LandmarkId &landmark_id);

void addLandmarkObservation(KeyFrame &old_frame, KeyFrame &new_frame,
                            const DMatch &match, Map3D &map);

vector<Vector3d> getLocalPoints3D(const vector<KeyPoint> &correspondences,
                                  const Mat &depth_frame1,
                                  const Vector4d &intrinsics);

Sophus::SE3d getExtrinsics(const Mat &E, const vector<Point2d> &matched_points1,
                           const vector<Point2d> &matched_points2,
                           const Mat &intrinsics);

#endif // BUNDLE_ADJUSTMENT_MAP3D_H
