//
// Created by witek on 07.01.22.
//

#include "Map3D.h"

vector<Vector3f> getLocalPoints3D(const frame_correspondences &correspondences,
                                  const Mat &depth_frame1,
                                  const Matrix3f &intrinsics) {
  Vector3f point3d;
  vector<Vector3f> points3d;

  for (auto &point2d : correspondences.frame1) {
    // debugging
    cout << "corresp. point func: " << point2d << endl;

    int u = static_cast<int>(point2d.x);
    int v = static_cast<int>(point2d.y);

    float z = depth_frame1.at<float>(v, u);
    float x = z * (u - intrinsics(0, 2)) / intrinsics(0, 0);
    float y = z * (v - intrinsics(1, 2)) / intrinsics(1, 1);

    point3d << x, y, z;
    points3d.push_back(point3d);
  }
  return points3d;
}

vector<Vector3f> getGlobalPoints3D(const frame1_geometry &frame) {
  Vector3f point3d;
  vector<Vector3f> points3d;

  for (auto &point3d_local : frame.points3d_local) {
    point3d = frame.pose * point3d_local;
    points3d.push_back(point3d);
  }
  return points3d;
}

Sophus::SE3f getExtrinsics(const Mat &E, const vector<Point2f> &matched_points1,
                           const vector<Point2f> &matched_points2,
                           const Mat &intrinsics) {
  Mat R, T;
  Matrix3f eigenR;
  Vector3f eigenT;
  Sophus::SE3f extrinsics;
  recoverPose(E, matched_points1, matched_points2, intrinsics, R, T);
  //    cout << "rotation: " << R << endl;
  //    cout << "translation: " << T << endl;

  cv2eigen(R, eigenR);
  cv2eigen(T, eigenT);

  return Sophus::SE3f(eigenR, eigenT);
}
