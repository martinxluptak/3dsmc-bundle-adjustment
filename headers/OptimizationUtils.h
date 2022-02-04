//
// Created by lupta on 1/26/2022.
//

#ifndef BUNDLE_ADJUSTMENT_OPTIMIZATIONUTILS_H
#define BUNDLE_ADJUSTMENT_OPTIMIZATIONUTILS_H

#include <string>
#include "Eigen/Dense"
#include "CommonTypes.h"
#include "BundleAdjustmentConfig.h"

using namespace Eigen;

Vector4d read_camera_intrinsics_from_file(const string &file_path);


/**
 * Write camera poses from keyframes to file.
 * Use this to save the output of the algorithm in a standardized format
 * for comparison with RGB-D dataset tools.
 * Useful link: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
 * file format: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
 *
 * @param file_path
 * @param keyframes
 */
void write_keyframe_poses_to_file(const string &file_path,
                                  const vector<KeyFrame> & keyframes);

/**
 * Count the number of reprojection and unprojection constraints.
 * These values are later used for normalization of weights of individual constraints.
 * These values are returned via reprojection_constraints_result and unprojection_constraints_result.
 *
 * @param cfg BundleAdjustmentConfig
 * @param map Map3D
 * @param keyframes KeyFrame vector
 * @param reprojection_constraints_result returns # of reprojection constraints
 * @param unprojection_constraints_result returns # of unprojection constraints
 */
int countConstraints(const Map3D &map, const vector<KeyFrame> &keyframes, int kf_i, int kf_f);

/**
 *
 * Searches global_points_map matching to find the index of local 2D point
 * to its corresponding global 3D point.
 *
 * @param keyframe keyframe containing the global_points_map mapping.
 * @param landmarkId ID of sought landmark.
 * @return local index of 2D point in keyframe.points3d_local vector.
 */
int findLocalPointIndex(const KeyFrame &keyframe, const int landmarkId);

bool windowOptimize(ceresGlobalProblem & globalProblem, int kf_i, int kf_f, vector<KeyFrame> & keyframes, Map3D & map, const Vector4d &intrinsics_initial, Vector4d & intrinsics_optimized);

Sophus::SE3d getFirstPose(const string & first_timestamp,
                          const string & ground_truth_file_path);

void poseOffset(vector<KeyFrame> &keyframes, const Sophus::SE3d & initial_pose);

bool
optimizeDebug(ceresGlobalProblem &globalProblem, const Vector4d &intrinsics_initial, Vector4d &intrinsics_optimized,
              int opt_type);


#endif //BUNDLE_ADJUSTMENT_OPTIMIZATIONUTILS_H
