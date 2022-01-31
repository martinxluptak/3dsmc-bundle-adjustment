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

void tracking_step(VirtualSensor &sensor, vector<KeyFrame> &keyframes, Map3D &map, BundleAdjustmentConfig &cfg,
                   int &landmark_id, Vector4d &intrinsics_initial) {
    // Transformation from frame 2 to 1
    Sophus::SE3d T_1_2;
    vector<Vector3d> points3d_before, point3d_after;

    KeyFrame current_frame;
    current_frame.frame_id = sensor.GetCurrentFrameCnt();
    current_frame.timestamp = sensor.GetCurrentRGBTimestamp();
    const auto &rgb = sensor.GetGrayscaleFrame();
    const auto &depth = sensor.GetDepthFrame();

    getKeypointsAndDescriptors(cfg.DETECTOR,
                               cfg.NUM_FEATURES, cfg.HESSIAN_THRES,
                               rgb, depth,
                               current_frame.keypoints, current_frame.descriptors);

    current_frame.points3d_local =
            getLocalPoints3D(current_frame.keypoints, depth, intrinsics_initial);

    if (current_frame.frame_id == 0) {
        keyframes.push_back(current_frame);
        return;
    }
    auto &previous_frame = keyframes.back();
    vector<vector<DMatch>> knn_matches;

    matchKeypoints(cfg.DETECTOR,
                   previous_frame.descriptors, current_frame.descriptors,
                   knn_matches);

    // filter matches using Lowe's ratio test
    vector<DMatch> lowe_matches = filterMatchesLowe(knn_matches, cfg.LOWE_TRESHOLD);

    vector<DMatch> inliers;
    initializeRelativePose(previous_frame.points3d_local,
                           current_frame.points3d_local, lowe_matches, inliers,
                           T_1_2);

    cout << "Found this many inliers: " << inliers.size() << endl;

    // Update pose, track on the last saved pose inside keyframes, which could possibly have been optimized
    auto last_pose = keyframes[keyframes.size() - 1].T_w_c;
    current_frame.T_w_c =
            last_pose * T_1_2;           // global pose for the current frame

    updateLandmarks(previous_frame, current_frame, inliers, map, landmark_id);

    keyframes.push_back(current_frame);

}

int main() {

    // Some configurations
    auto cfg = BundleAdjustmentConfig();
    auto cfg_optimization = ceresGlobalProblem();

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

    // Process every frame in the sequence, optimize once in a while
    bool do_windowed_optimization = (cfg_optimization.window_size > 0), do_global_optimization = (
            cfg_optimization.window_size < 0);
    while (sensor.ProcessNextFrame()) {
        tracking_step(sensor, keyframes, map, cfg, landmark_id, intrinsics_initial);    // Tracking step
        if (do_windowed_optimization && keyframes.size() % cfg_optimization.frame_frequency == 0 &&
            keyframes.size() >= cfg_optimization.window_size) {
            windowOptimize(cfg_optimization, keyframes.size() - cfg_optimization.window_size, keyframes.size() - 1,
                           keyframes, map, intrinsics_initial, intrinsics_optimized);
        }   // Optimization step
    }
    if (do_windowed_optimization && keyframes.size() % cfg_optimization.frame_frequency != 0) {
        windowOptimize(cfg_optimization,
                       keyframes.size() - cfg_optimization.window_size,
                       keyframes.size() - 1, keyframes, map, intrinsics_initial, intrinsics_optimized);
    }   // Leftovers

    // Run instead global optimization
    if (do_global_optimization) {
        windowOptimize(cfg_optimization, 0, keyframes.size() - 1, keyframes, map, intrinsics_initial,
                       intrinsics_optimized);
    }

    cout << endl << "End of optimization. Writing output to file..." << endl;

    // Adjust the output so that the first pose aligns with ground_truth.txt, and is not just identity
    // (for ATE evaluation, and trajectory plotting).
    auto firstPose = getFirstPose(keyframes[0].timestamp, cfg.GROUND_TRUTH_PATH);
    poseOffset(keyframes, firstPose);

    // Write output to file
    write_keyframe_poses_to_file(cfg.OUTPUT_POSES_PATH, keyframes);


    return 0;
}
