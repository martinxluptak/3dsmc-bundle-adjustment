//
// Created by lupta on 1/26/2022.
//

#include <iostream>
#include <string>

#ifndef BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
#define BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H

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

    /*
     *  Define parameters for optimization.
     */
    const double HUB_P_REPR = 1e-2; // Huber loss parameter for reprojection constraints
    const double HUB_P_UNPR = 1e-2; // Huber loss parameter for depth prior (i.e. unprojection constraints)
    const double WEIGHT_UNPR = 5; // weight for unprojection constraint
    const double WEIGHT_INTRINSICS = 1e-4;
    static constexpr double NEG_INF = std::numeric_limits<double>::lowest();   // -inf
};

#endif //BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
