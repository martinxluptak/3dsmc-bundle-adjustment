
#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <sophus/se3.hpp>

#include "Eigen.h"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

struct KeyFrame {
  int frame_id;
  Sophus::SE3d pose; // relative to frame 1
  vector<KeyPoint> keypoints;
  Mat descriptors;
  // Keypoints with the ground truth depth added
  vector<Vector3d> points3d_local;

  // Maps the locals 3d points to their corresponding global 3d point in the Map
  // if it exists
  unordered_map<
      Vector3d, Vector3d,
      Eigen::aligned_allocator<pair<Eigen::Vector3d, Eigen::Vector3d>>>
      points3d_map;

  // relative to the current frame
  //  vector<Vector3d> points3d_global; // relative to frame 1
};

typedef pair<int, Point2d> Observation;

typedef vector<Observation> Observations;

typedef unordered_map<Vector3d, Observations> Map;

#endif // COMMON_TYPES_H
