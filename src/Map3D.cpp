//
// Created by witek on 07.01.22.
//

#include "Map3D.h"


// saves points with -Inf depth
// TODO: skip those image points!
vector<Vector3f> getPoints3D(const frame_correspondences& cs1,
                             const Mat& depth_frame1, const Matrix3f& intrinsics)
{
    vector<Vector3f> points3d;
    for (auto& point2d : cs1.frame1)
    {
        int u = static_cast<int>(point2d.x);
        int v = static_cast<int>(point2d.y);

        v = point2d.y;
        float z = depth_frame1.at<float>(u, v);
        float x = z * (u - intrinsics(0,2)) / intrinsics(0,0);
        float y = z * (v - intrinsics(1,2)) / intrinsics(1,1);
        Vector3f point3d(x, y, z);
        points3d.push_back(point3d);
    }
    return points3d;
}

pair<Mat, Mat> getPose(const Mat& E, const vector<Point2f>& matched_points1,
                       const vector<Point2f>& matched_points2, const Mat& intrinsics)
{
    Mat R, T;
    recoverPose(E, matched_points1, matched_points2, intrinsics, R, T);
//    cout << "rotation: " << R << endl;
//    cout << "translation: " << T << endl;
    return make_pair(R, T);
}
