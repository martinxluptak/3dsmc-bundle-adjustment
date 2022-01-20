#include "CommonTypes.h"
#include "Detection.h"
#include "Eigen/Dense"
#include "Map3D.h"
#include "Matching.h"
#include "VirtualSensor.h"
#include "opencv2/core.hpp"
#include <iostream>
#include <ceres/ceres.h>
#include "sophus/local_parameterization_se3.hpp"
#include "OptimizationUtils.cpp"

using namespace cv;
using namespace std;
using namespace Eigen;

/**
 * Reads camera intrinsics data from the target file.
 *
 * Intrinsics format: (fx, fy, cx, cy)
 *
 * @return a vector of doubles containing intrinsics in aforementioned format.
 */
Vector4d read_camera_intrinsics_from_file(const string &file_path) {
    ifstream infile;
    infile.open(file_path);
    double fx, fy, cx, cy, d0, d1, d2, d3, d4;
    while (infile >> fx >> fy >> cx >> cy >> d0 >> d1 >> d2 >> d3 >> d4) {
        // reads only the first line and then closes.
    }
    infile.close();

    Vector4d intrinsics;
    intrinsics << fx, fy, cx, cy;
    return intrinsics;
}

int main() {
    /////////////////////////////////////
   //  FEATURE EXTRACTION + MATCHING  //
  /////////////////////////////////////
  LandmarkId landmark_id = 0;
  vector<KeyFrame> keyframes;
  Sophus::SE3d current_pose;
  Map3D map;
  int NUM_FEATURES = 1000;
  const uint KEYFRAME_INCREMENT = 10;

  /*
   * Read intrinsics_initial from file.
   * Always read from file as we might load different file
   * for a different data set.
   * format: fx, fy, cx, cy
   */
  const string CAMERA_DEFAULT_INTRINSICS_PATH = "../../Data/freiburg1_intrinsics.txt";
  const string DATASET_FILEPATH = "../../Data/rgbd_dataset_freiburg1_xyz/"; // SET TO <your_path>/rgbd_dataset_freiburg1_xyz/
  Vector4d intrinsics_initial = read_camera_intrinsics_from_file(CAMERA_DEFAULT_INTRINSICS_PATH);

  // load video
  cout << "Initialize virtual sensor..." << endl;

  VirtualSensor sensor(KEYFRAME_INCREMENT);

  if (!sensor.Init(DATASET_FILEPATH)) {
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
           NUM_FEATURES);
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

    ////////////////////////////////
    //  OPTIMIZATION WITH CERES  //
    //////////////////////////////

    /*
     *  Define constants and ceres problem options.
     */
    const double HUB_B_REPR = 1e-2;
    const double WEIGHT_INTRINSICS = 1e-4;

    ceres::Problem problem; // Optimization variables, poses, map and intrinsics_initial
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;

    /*
     * Add camera poses as parameter blocks.
     * (Saved in keyframes as T_w_c)
     */
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;
    for (auto &keyframe: keyframes) {
        problem.AddParameterBlock(keyframe.T_w_c.data(),
                                  Sophus::SE3d::num_parameters,
                                  local_parametrization_se3
        );
    }

    /*
     * Add world 3D points as parameter blocks.
     * Note: the parameter block must always be an array of doubles!
     */
    for (auto& it: map) {
        auto landmark_3d_point = it.second.point;
        problem.AddParameterBlock(landmark_3d_point.data(), 3);
    }

    /*
     *  Add camera intrinsics as parameter block.
     *  Initialize intrinsics_optimized from intrinsics_initial
     *  Keep a copy of intrinsics_initial to compare the values after optimization.
     */
    auto intrinsics_optimized(intrinsics_initial);
    problem.AddParameterBlock(intrinsics_optimized.data(), 4);

    /*
     * Define loss function for reprojection error.
     */
    auto *loss_function_repr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(HUB_B_REPR),
                                                              ceres::TAKE_OWNERSHIP);

    /*
     * Iterate over map of landmarks
     */
    for (auto& it : map) {
        auto landmark = it.second;
        /*
         * Iterate over each observation of the same landmark.
         */
        for (auto & observation : landmark.observations) {
            auto keyframe_index = observation.first / KEYFRAME_INCREMENT;
            auto associated_keyframe = keyframes[keyframe_index];
            auto observed_point = observation.second;
            Vector2d observed_point_vec2d(observed_point.x, observed_point.y); // Conversion from Point2d to Vector2d

            problem.AddResidualBlock(
                    ReprojectionConstraint::create_cost_function(observed_point_vec2d),
                    loss_function_repr,
                    associated_keyframe.T_w_c.data(), // (global) camera pose during observation
                    landmark.point.data(), // 3D point
                    intrinsics_optimized.data()
                );
        }
    }

    // Compare with ground truth
    vector<double> pose_errors;
    vector<double> map_errors;

    // Optimize for the intrinsics_initial -> we must add a (small) prior
    problem.AddResidualBlock(
            IntrinsicsPrior::create_cost_function(intrinsics_initial, WEIGHT_INTRINSICS),
            nullptr, // squared loss
            intrinsics_optimized.data()
    );

    // Constrain the problem
    problem.SetParameterBlockConstant(keyframes[0].T_w_c.data()); // any pose, kept constant, will do

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    return 0;
}
