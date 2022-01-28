//
// Created by lupta on 1/26/2022.
//

#ifndef BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
#define BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H

#include "OptimizationUtils.h"
#include <string>
#include "Eigen/Dense"
#include <fstream>
#include <utility>
#include <ceres/ceres.h>
#include "sophus/local_parameterization_se3.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;


class BundleAdjustmentConfig {

public:
    /*
     *  File paths
     */
    const string CAMERA_DEFAULT_INTRINSICS_PATH = "../../Data/ros_default_intrinsics.txt";
    const string DATASET_FILEPATH = "../../Data/rgbd_dataset_freiburg1_xyz/"; // SET TO <your_path>/rgbd_dataset_freiburg1_xyz/
    const string GROUND_TRUTH_PATH = "../../Data/rgbd_dataset_freiburg1_xyz/groundtruth.txt";
    const string OUTPUT_POSES_PATH = "../../output/freiburg1_xyz_poses.txt"; // output:

    /*
     *  Keypoint extraction + feature matching
     */
    const string DETECTOR = "SIFT"; // options: ORB, SIFT, SURF
    int NUM_FEATURES = 1000; // hyperparameter for ORB and SIFT
    int HESSIAN_THRES = 100; // hyperparameter for SURF; default value: 100
    const uint KEYFRAME_INCREMENT = 5;

};

class ceresGlobalProblem {
public:

    const double HUB_P_REPR = 1e-2; // Huber loss parameter for reprojection constraints
    const double WEIGHT_INTRINSICS = 1e-6;
    const double WEIGHT_UNPR = 5; // weight for unprojection constraint. Relative to the reprojection constraints, who have a weight of 1
    const double HUB_P_UNPR = 1e-2; // Huber loss parameter for depth prior (i.e. unprojection constraints)

    const int frame_frequency = 10;  // wait 5 frames to do optimization again
    const int window_size = 20; // how many keyframes are we optimizing for every window? Put -1 to have a unique window, put 0 to skip optimization

    ceres::Solver::Options options;

    ceresGlobalProblem() {
        initialize_options();
    }

    void initialize_options() {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 200;
    }

};

#endif //BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
