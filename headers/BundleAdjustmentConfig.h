//
// Created by lupta on 1/26/2022.
//

#include <iostream>
#include <string>

#ifndef BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
#define BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H

class OptimizationConfig {

    /*
     *  File paths
     */

    const string DATASET_NAME = "rgbd_dataset_freiburg1_xyz"
    const string CAMERA_DEFAULT_INTRINSICS_PATH = "../../Data/freiburg1_intrinsics.txt";
    const string DATASET_FILEPATH = "../../Data//"; // SET TO <your_path>/rgbd_dataset_freiburg1_xyz/
    const string OUTPUT_POSES_PATH = "../../output/freiburg"

    /*
     *  Keypoint extraction + feature matching
     */
    int NUM_FEATURES = 1000;
    const uint KEYFRAME_INCREMENT = 10;


};

#endif //BUNDLE_ADJUSTMENT_OPTIMIZATIONCONFIG_H
