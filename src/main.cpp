#include "CommonTypes.h"
#include "Detection.h"
#include "Eigen/Dense"
#include "Map3D.h"
#include "Matching.h"
#include "VirtualSensor.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "sophus/se3.hpp"
#include <iostream>
using namespace cv;
using namespace std;
using namespace Eigen;

int main() {
  TrackId track_id = 0;
  vector<frame_correspondences> video_correspondences;
  vector<KeyFrame> keyframes;
  Sophus::SE3d current_pose;
  Map3D map;
  int num_features = 1000;
  int keyframe_increment = 10;

  // TODO: unify intrinsics data formats
  Matrix3f intr;
  intr << 517.3, 0.0, 319.5, 0.0, 516.5, 255.3, 0.0, 0.0, 1.0;
  double intrinsics_data[] = {517.3, 0.0, 319.5, 0.0, 516.5,
                              255.3, 0.0, 0.0,   1.0};
  Mat intrinsics(3, 3, CV_64FC1, intrinsics_data);

  string filename = string(
      "../../Data/rgbd_dataset_freiburg1_xyz/"); // SET TO
                                                 // <your_path>/rgbd_dataset_freiburg1_xyz/

  // load video
  cout << "Initialize virtual sensor..." << endl;

  VirtualSensor sensor(keyframe_increment);

  if (!sensor.Init(filename)) {
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
           num_features);
    current_frame.points3d_local =
        getLocalPoints3D(current_frame.keypoints, depth, intr);

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

    current_frame.pose =
        T_1_2.inverse() * current_pose; // global pose for the current frame
    current_pose = current_frame.pose;  // global pose for the next frame

    //    current_frame.points3d_global = getGlobalPoints3D(current_frame);
    keyframes.push_back(current_frame);

    // Map initialization with the first two keyframes
    if (keyframes.size() == 2) {
      // Add inliers to map
      for (auto &inlier : inliers) {
        // Update map
        Observation obs_1, obs_2;
        obs_1.first = previous_frame.frame_id;
        obs_1.second = previous_frame.keypoints[inlier.queryIdx].pt;
        obs_2.first = current_frame.frame_id;
        obs_2.second = current_frame.keypoints[inlier.trainIdx].pt;

        Vector3d point_3d = previous_frame.points3d_local[inlier.queryIdx];

        Track track;
        track.observations.push_back(obs_1);
        track.observations.push_back(obs_2);

        map[track_id] = track;

        // Update keyframe -> map correspondences
        previous_frame.global_points_map[inlier.trainIdx] = track_id;
        current_frame.global_points_map[inlier.queryIdx] = track_id;

        track_id++;
      }
    }
  }
  cout << endl << "No more keyframe pairs.\n" << endl;

  return 0;
}
