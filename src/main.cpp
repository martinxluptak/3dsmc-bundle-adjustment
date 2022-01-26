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

int main() {
    /*
     * Create an optimization config object.
     */
    auto cfg = BundleAdjustmentConfig();

    /////////////////////////////////////
   //  FEATURE EXTRACTION + MATCHING  //
  /////////////////////////////////////
  LandmarkId landmark_id = 0;
  vector<KeyFrame> keyframes;
  Sophus::SE3d current_pose;
  Map3D map;

  /*
   * Read intrinsics_initial from file.
   * Always read from file as we might load different file
   * for a different data set.
   * format: fx, fy, cx, cy
   */
  Vector4d intrinsics_initial = read_camera_intrinsics_from_file(cfg.CAMERA_DEFAULT_INTRINSICS_PATH);

  // load video
  cout << "Initialize virtual sensor..." << endl;

  VirtualSensor sensor(cfg.KEYFRAME_INCREMENT);

  if (!sensor.Init(cfg.DATASET_FILEPATH)) {
    cout << "Failed to initialize the sensor.\nCheck file path." << endl;
    return -1;
  }

  while (sensor.ProcessNextFrame()) {

    // Transfromation from frame 2 to 1
    Sophus::SE3d T_1_2;
    vector<Vector3d> points3d_before, point3d_after;

    KeyFrame current_frame;
    current_frame.frame_id = sensor.GetCurrentFrameCnt();
    const auto &rgb = sensor.GetGrayscaleFrame();
    const auto &depth = sensor.GetDepthFrame();

    // detect keypoints and compute descriptors using ORB
    getORB(rgb, current_frame.keypoints, current_frame.descriptors,
           cfg.NUM_FEATURES);
    current_frame.points3d_local =
        getLocalPoints3D(current_frame.keypoints, depth, intrinsics_initial);

    if (current_frame.frame_id == 0) {
      keyframes.push_back(current_frame);
      continue;
    }
    auto &previous_frame = keyframes.back();
    vector<vector<DMatch>> knn_matches;
    // match keypoints

    matchKeypoints(previous_frame.descriptors, current_frame.descriptors,
                   knn_matches);

    // filter matches using Lowe's ratio test
    vector<DMatch> lowe_matches = filterMatchesLowe(knn_matches, 0.7);

    vector<DMatch> inliers;
    initializeRelativePose(previous_frame.points3d_local,
                           current_frame.points3d_local, lowe_matches, inliers,
                           T_1_2);

    cout << "inliers found: " << inliers.size() << endl;

    current_frame.T_w_c =
        current_pose * T_1_2;           // global pose for the current frame
    current_pose = current_frame.T_w_c; // global pose for the next frame

    updateLandmarks(previous_frame, current_frame, inliers, map, landmark_id);

    keyframes.push_back(current_frame);
  }
  cout << endl << "No more keyframe pairs.\n" << endl;

  runOptimization(cfg, map, keyframes, intrinsics_initial);
    return 0;
}
