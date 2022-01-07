#include <iostream>

#include "../headers/VirtualSensor.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
//#include <Eigen/Dense>
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;


vector<DMatch> filterMatchesLowe(const vector<vector<DMatch>>& knn_matches, const float& ratio)
{
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance <
            ratio * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

vector<DMatch> filterMatchesRANSAC(const vector<DMatch>& matches, const Mat& mask){
    vector<DMatch> good_matches;
    for (int i=0; i<matches.size(); i++)
    {
        if (mask.at<int>(0, i) == 1) { // keep inliers only
            good_matches.push_back(matches[i]);
        }
    }
    return good_matches;
}

pair<vector<Point2f>, vector<Point2f>> getMatchedPoints(const vector<DMatch>& matches,
                                                         const vector<KeyPoint>& keypoints1,
                                                         const vector<KeyPoint>& keypoints2)
{
    vector<Point2f> matched_points1, matched_points2; // points that match
    for (int i=0; i<matches.size(); i++)
    {
        int idx1=matches[i].trainIdx;
        int idx2=matches[i].queryIdx;
        //use match indices to get the keypoints, add to the two lists of points
        matched_points1.push_back(keypoints1[idx1].pt);
        matched_points2.push_back(keypoints2[idx2].pt);
    }
    return make_pair(matched_points1, matched_points2);
}

pair<Mat, Mat> getPose(const Mat& E, const vector<Point2f>& matched_points1,
                       const vector<Point2f>& matched_points2, const Mat& intrinsics)
{
    Mat R, T;
    recoverPose(E, matched_points1, matched_points2, intrinsics, R, T);
    cout << "rotation: " << R << endl;
    cout << "translation: " << T << endl;
    return make_pair(R, T);
}

void displayMatches(const Mat& frame1, const vector<KeyPoint>& keypoints1,
                    const Mat& frame2, const vector<KeyPoint>& keypoints2,
                    const vector<DMatch>& matches){
    Mat img_out;
    drawMatches(frame1, keypoints1, frame2, keypoints2, matches,
                img_out, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // show detected matches
    imshow("Matches", img_out);
    waitKey();
};


int main() {
    std::string filenameIn =
            std::string("/home/witek/Desktop/TUM/3d_scanning/3dsmc-bundle-adjustment/data/rgbd_dataset_freiburg1_xyz/");

    // load video
    cout << "Initialize virtual sensor..." << endl;
    VirtualSensor sensor(5);
    if (!sensor.Init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!"
                  << std::endl;
        return -1;
    }

    // store a first frame as a reference frame;
    // all next frames are tracked relatively to the first frame
    sensor.ProcessNextFrame();
    const auto &frame1 = sensor.GetGrayscaleFrame();
    sensor.ProcessNextFrame();
    const auto &frame2 = sensor.GetGrayscaleFrame();
    waitKey();

    if (frame1.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    // detect the keypoints and compute descriptors using ORB
    int nFeatures = 1000;
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    vector<vector<DMatch>> knn_matches;
    Ptr<ORB> detector = ORB::create(nFeatures);
    detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(frame2, noArray(), keypoints2, descriptors2);

    // match keypoints
    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // filter matches using Lowe's ratio test
    vector<DMatch> good_matches = filterMatchesLowe(knn_matches, 1.0);

    vector<Point2f> matched_points1, matched_points2; // points that match
    tie(matched_points1, matched_points2) = getMatchedPoints(good_matches, keypoints1, keypoints2);

    double intrinsics_data[] = {517.3, 0.0, 319.5,
                                0.0, 516.5, 255.3,
                                0.0, 0.0, 1.0};
    Mat intrinsics(3, 3, CV_64FC1, intrinsics_data);

    // filter matches using RANSAC, get essential matrix
    Mat mask_ransac;
    Mat E = findEssentialMat(matched_points1, matched_points2, intrinsics, RANSAC, 0.9999, 1.0, mask_ransac);
    vector<DMatch> better_matches = filterMatchesRANSAC(good_matches, mask_ransac);

    // get rotation and translation
    Mat R, T;
    pair<Mat, Mat> pose;
    tie(R, T) = getPose(E, matched_points1, matched_points2, intrinsics);

    displayMatches(frame1, keypoints1, frame2, keypoints2, better_matches);

    return 0;
}

