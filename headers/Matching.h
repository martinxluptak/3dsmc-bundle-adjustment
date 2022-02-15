//
// Created by witek on 08.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_MATCHING_H
#define BUNDLE_ADJUSTMENT_MATCHING_H

#include "Eigen/Dense"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opengv/point_cloud/PointCloudAdapter.hpp"
#include "opengv/sac/Ransac.hpp"
#include "opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp"
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;
using namespace opengv;

void matchKeypoints(const string &detector,
                    const Mat &descriptors1, const Mat &descriptors2,
                    vector<vector<DMatch>> &knn_matches);

vector<DMatch> filterMatchesLowe(const vector<vector<DMatch>> &knn_matches,
                                 const float &ratio);

vector<DMatch> filterMatchesRANSAC(const vector<DMatch> &matches,
                                   const Mat &mask);

pair<vector<Point2d>, vector<Point2d>>
getMatchedPoints(const vector<DMatch> &matches,
                 const vector<KeyPoint> &keypoints1,
                 const vector<KeyPoint> &keypoints2);

void displayMatches(string img_name, const Mat &frame1,
                    const vector<KeyPoint> &keypoints1, const Mat &frame2,
                    const vector<KeyPoint> &keypoints2,
                    const vector<DMatch> &matches, const Mat &mask);

void initializeRelativePose(const vector<Vector3d> &points1,
                            const vector<Vector3d> &points2,
                            const vector<DMatch> &matches,
                            vector<DMatch> &inliers, Sophus::SE3d &pose);

#endif // BUNDLE_ADJUSTMENT_MATCHING_H
