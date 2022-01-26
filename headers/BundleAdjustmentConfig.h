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
    ceresGlobalProblem(const double hubPRepr, const double weightIntrinsics, const double weightUnpr,
                       const double hubPUnpr, const ceres::Solver::Options &options,
                       ceres::LossFunctionWrapper *lossFunctionRepr, ceres::LossFunctionWrapper *lossFunctionUnpr,
                       ceres::LocalParameterization *localParametrizationSe3)
            : HUB_P_REPR(hubPRepr), WEIGHT_INTRINSICS(weightIntrinsics), WEIGHT_UNPR(weightUnpr), HUB_P_UNPR(hubPUnpr),
              options(options), loss_function_repr(lossFunctionRepr), loss_function_unpr(lossFunctionUnpr),
              local_parametrization_se3(localParametrizationSe3) {}

    const double HUB_P_REPR; // Huber loss parameter for reprojection constraints
    const double WEIGHT_INTRINSICS;
    const double WEIGHT_UNPR; // weight for unprojection constraint. Relative to the reprojection constraints, who have a weight of 1
    const double HUB_P_UNPR; // Huber loss parameter for depth prior (i.e. unprojection constraints)

    ceres::Solver::Options options;

    ceres::LossFunctionWrapper *loss_function_repr;
    ceres::LossFunctionWrapper *loss_function_unpr;

    ceres::LocalParameterization *local_parametrization_se3;

};

#endif //BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
