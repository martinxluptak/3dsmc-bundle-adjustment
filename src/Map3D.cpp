//
// Created by witek on 07.01.22.
//

#include "Map3D.h"

vector<Vector3d> getLocalPoints3D(const vector<KeyPoint> &points,
                                  const Mat &depth_frame1,
                                  const Matrix3f &intrinsics) {
  Vector3d point3d;
  vector<Vector3d> points3d;

  for (auto &point2d : points) {
    // debugging
    //    cout << "corresp. point func: " << point2d.pt << endl;

    int u = static_cast<int>(point2d.pt.x);
    int v = static_cast<int>(point2d.pt.y);

    float z = depth_frame1.at<float>(v, u);
    float x = z * (u - intrinsics(0, 2)) / intrinsics(0, 0);
    float y = z * (v - intrinsics(1, 2)) / intrinsics(1, 1);

    point3d << x, y, z;
    points3d.push_back(point3d);
  }
  return points3d;
}

vector<Vector3d> getGlobalPoints3D(const KeyFrame &frame) {
  Vector3d point3d;
  vector<Vector3d> points3d;

  for (auto &point3d_local : frame.points3d_local) {
    point3d = frame.pose * point3d_local;
    points3d.push_back(point3d);
  }
  return points3d;
}

Sophus::SE3d getExtrinsics(const Mat &E, const vector<Point2d> &matched_points1,
                           const vector<Point2d> &matched_points2,
                           const Mat &intrinsics) {
  Mat R, T;
  Matrix3d eigenR;
  Vector3d eigenT;
  Sophus::SE3d extrinsics;
  recoverPose(E, matched_points1, matched_points2, intrinsics, R, T);
  //    cout << "rotation: " << R << endl;
  //    cout << "translation: " << T << endl;

  cv2eigen(R, eigenR);
  cv2eigen(T, eigenT);

  return Sophus::SE3d(eigenR, eigenT);
}
