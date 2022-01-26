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
    const string CAMERA_DEFAULT_INTRINSICS_PATH = "../../Data/freiburg1_intrinsics.txt";
    const string DATASET_FILEPATH = "../../Data/rgbd_dataset_freiburg1_xyz/"; // SET TO <your_path>/rgbd_dataset_freiburg1_xyz/
    const string OUTPUT_POSES_PATH = "../../output/freiburg1_poses.txt"; // output:

    /*
     *  Keypoint extraction + feature matching
     */
    int NUM_FEATURES = 1000;
    const uint KEYFRAME_INCREMENT = 10;

};

class ceresGlobalProblem{
public:

    const double HUB_P_REPR = 1e-2; // Huber loss parameter for reprojection constraints
    const double WEIGHT_INTRINSICS = 1e-4;
    const double WEIGHT_UNPR = 5; // weight for unprojection constraint. Relative to the reprojection constraints, who have a weight of 1
    const double HUB_P_UNPR = 1e-2; // Huber loss parameter for depth prior (i.e. unprojection constraints)

    ceres::Solver::Options options;

    ceres::LossFunctionWrapper *loss_function_repr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(HUB_P_REPR),
                                                              ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::LossFunctionWrapper *loss_function_unpr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(HUB_P_UNPR),
                                                              ceres::DO_NOT_TAKE_OWNERSHIP);

    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;

    ceresGlobalProblem() {
        initialize_options();
    }

    void initialize_options(){
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 200;
    }

};

#endif //BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
