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

  vector<frame_correspondences> video_correspondences;
  vector<frame1_geometry> video_geometry;
  Sophus::SE3d current_pose;
  int num_features = 1000;
  int keyframe_increment = 10;
  int iterations = 5;
  bool frame2_exists;

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

  // choose keyframes
  VirtualSensor sensor(keyframe_increment);
  if (!sensor.Init(filename)) {
    cout << "Failed to initialize the sensor.\nCheck file path." << endl;
    return -1;
  }

  sensor.ProcessNextFrame(); // load frame 0

  //    for(int i=0; i<iterations; i++)
  while (true) {
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2, mask_ransac, E, mask_default;
    vector<vector<DMatch>> knn_matches;
    frame_correspondences correspondences;
    frame1_geometry frame;
    vector<DMatch> lowe_matches, ransac_matches;
    vector<Point2d> matched_points_lowe1, matched_points_lowe2;
    vector<Point2d> matched_points_ransac1, matched_points_ransac2;
    // Transfromation from frame 2 to 1
    Sophus::SE3d T_1_2;
    vector<Vector3d> points3d_before, point3d_after;

    const auto &frame1 = sensor.GetGrayscaleFrame();
    const auto &depth_frame1 = sensor.GetDepthFrame();

    frame2_exists = sensor.ProcessNextFrame();
    const auto &frame2 = sensor.GetGrayscaleFrame();
    const auto &depth_frame2 = sensor.GetDepthFrame();
    //        int num = sensor.GetCurrentFrameCnt();
    //        cout << num << endl;

    if (frame1.empty()) {
      cout << "Could not open or find the image.\n" << endl;
      break;
    }

    if (!frame2_exists) {
      cout << endl << "No more keyframe pairs.\n" << endl;
      break;
    }

    // detect keypoints and compute descriptors using ORB
    getORB(frame1, frame2, keypoints1, keypoints2, descriptors1, descriptors2,
           num_features);

    // match keypoints
    matchKeypoints(descriptors1, descriptors2, knn_matches);

    // filter matches using Lowe's ratio test
    lowe_matches = filterMatchesLowe(knn_matches, 0.9);
    //    cout << "detected Lowe matches: " << lowe_matches.size() << endl;
    tie(matched_points_lowe1, matched_points_lowe2) =
        getMatchedPoints(lowe_matches, keypoints1, keypoints2);

    // register 3d points
    frame.points3d_local = getLocalPoints3D(keypoints2, depth_frame2, intr);
    vector<Vector3d> frame1_points_3d =
        getLocalPoints3D(keypoints1, depth_frame1, intr);
    vector<DMatch> inliers;
    initializeRelativePose(frame1_points_3d, frame.points3d_local, lowe_matches,
                           inliers, T_1_2);

    cout << "inliers found: " << inliers.size() << endl;
    // register corresponding points
    //    correspondences.frame1 = matched_points_ransac1;
    //    correspondences.frame2 = matched_points_ransac2;
    //    video_correspondences.push_back(correspondences);

    // debugging TODO: fix getLocalPoints3D - stops loading the data
    // (even though the loop below works)
    //    for (auto &point2d : correspondences.frame1) {
    //      cout << "corresp. point: " << point2d << endl;
    //    }

    // display matches
    //        mask_default = Mat::ones(1, lowe_matches.size(), CV_64F);
    //        displayMatches("Matches Lowe", frame1, keypoints1, frame2,
    //        keypoints2, lowe_matches, mask_default); displayMatches("Matches
    //        Lowe & RANSAC", frame1, keypoints1, frame2, keypoints2,
    //        lowe_matches, mask_ransac);

    // get rotation and translation between 2 neighbouring frames
    frame.pose =
        T_1_2.inverse() * current_pose; // global pose for the current frame
    current_pose = frame.pose;          // global pose for the next frame

    frame.points3d_global = getGlobalPoints3D(frame);
    video_geometry.push_back(frame);
  }
  return 0;
}

// debugging
//    cv::Mat flat = mask_ransac.reshape(1,
//    mask_ransac.total()*mask_ransac.channels()); vector<uchar> mask_ransac_vec
//    = mask_ransac.isContinuous()? flat : flat.clone(); cout <<
//    mask_ransac_vec.size() << endl; cout << count(mask_ransac_vec.begin(),
//    mask_ransac_vec.end(), 1) << endl;
