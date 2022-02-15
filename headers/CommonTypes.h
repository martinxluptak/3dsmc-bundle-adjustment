
#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <sophus/se3.hpp>

#include "Eigen.h"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

typedef int LandmarkId;

struct KeyFrame {
  uint frame_id;
  string timestamp;
  Sophus::SE3d T_w_c; // relative to frame 1
  vector<KeyPoint> keypoints;
  Mat descriptors;
  // Keypoints with the ground truth depth added
  vector<Vector3d> points3d_local;

  // Maps the locals 3d points to their corresponding global 3d point in the Map
  // if it exists
  // Note: This data is redundant with the map, but makes it faster to index
  // and implement
  unordered_map<int, LandmarkId> global_points_map;

  // relative to the current frame
  //  vector<Vector3d> points3d_global; // relative to frame 1
};

typedef pair<int, Point2d> Observation;

typedef vector<Observation> Observations;

struct Landmark {
  Observations observations;
  Vector3d point;
};

typedef unordered_map<LandmarkId, Landmark> Map3D;

#endif // COMMON_TYPES_H
