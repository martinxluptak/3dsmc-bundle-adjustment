#include "CommonTypes.h"
#include "Detection.h"
#include "Eigen/Dense"
#include "Map3D.h"
#include "Matching.h"
#include "VirtualSensor.h"
#include "opencv2/core.hpp"
#include <iostream>
#include "sophus/local_parameterization_se3.hpp"
#include "OptimizationUtils.h"
#include "BundleAdjustmentConfig.h"

using namespace cv;
using namespace std;
using namespace Eigen;

void tracking_step(VirtualSensor & sensor, vector<KeyFrame> & keyframes, Map3D & map, BundleAdjustmentConfig & cfg, int & landmark_id, Vector4d & intrinsics_initial){
    // Transfromation from frame 2 to 1
    Sophus::SE3d T_1_2;
    vector<Vector3d> points3d_before, point3d_after;

    KeyFrame current_frame;
    current_frame.frame_id = sensor.GetCurrentFrameCnt();
    current_frame.timestamp = sensor.GetCurrentRGBTimestamp();
    const auto &rgb = sensor.GetGrayscaleFrame();
    const auto &depth = sensor.GetDepthFrame();

    // detect keypoints and compute descriptors using ORB
    getORB(rgb, depth, current_frame.keypoints, current_frame.descriptors,
           cfg.NUM_FEATURES);
    current_frame.points3d_local =
            getLocalPoints3D(current_frame.keypoints, depth, intrinsics_initial);

    if (current_frame.frame_id == 0) {
        keyframes.push_back(current_frame);
        return;
    }
    auto &previous_frame = keyframes.back();
    vector<vector<DMatch>> knn_matches;
    // match keypoints

    // uncomment relevant lines in the function, depending if ORB/SIFT/SURF
    matchKeypoints(previous_frame.descriptors, current_frame.descriptors,
                   knn_matches);

    // filter matches using Lowe's ratio test
    vector<DMatch> lowe_matches = filterMatchesLowe(knn_matches, 0.7);

    vector<DMatch> inliers;
    initializeRelativePose(previous_frame.points3d_local,
                           current_frame.points3d_local, lowe_matches, inliers,
                           T_1_2);

    cout << "inliers found: " << inliers.size() << endl;

    // Update pose, track on the last saved pose inside keyframes, which could possibly have been optimized
    auto last_pose = keyframes[keyframes.size()-1].T_w_c;
    current_frame.T_w_c =
            last_pose * T_1_2;           // global pose for the current frame

    updateLandmarks(previous_frame, current_frame, inliers, map, landmark_id);

    keyframes.push_back(current_frame);

}

int main() {
    /*
     * Create an optimization config object.
     */
    auto cfg = BundleAdjustmentConfig();

    /////////////////////////////
    //  WINDOWED OPTIMIZATION  //
    /////////////////////////////

    // The things we're interested in
    Map3D map;
    vector<KeyFrame> keyframes;
    Vector4d intrinsics_initial = read_camera_intrinsics_from_file(cfg.CAMERA_DEFAULT_INTRINSICS_PATH);
    auto intrinsics_optimized(intrinsics_initial);  // to be later optimized

    // Temporary variables
    LandmarkId landmark_id = 0;

    // Load the sequence
    cout << "Initialize virtual sensor..." << endl;
    VirtualSensor sensor(cfg.KEYFRAME_INCREMENT);
    if (!sensor.Init(cfg.DATASET_FILEPATH)) {
        cout << "Failed to initialize the sensor.\nCheck file path." << endl;
        return -1;
    }

    // Process every frame in the sequence
    while (sensor.ProcessNextFrame()) {
        tracking_step(sensor, keyframes, map, cfg, landmark_id, intrinsics_initial);
        // todo: insert optimization here! and not below, for better precision of map and what not
    }
    cout << endl << "End of the sequence.\n" << endl;

    // todo: will be removed
    ///////////////////////////////
    //  OPTIMIZATION WITH CERES  //
    ///////////////////////////////

    runOptimization(cfg, map, keyframes, intrinsics_initial, intrinsics_optimized);
    return 0;
}
