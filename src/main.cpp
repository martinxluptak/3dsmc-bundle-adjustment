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

int main() {
    std::string filenameIn =
            std::string("/home/witek/Desktop/TUM/3d_scanning/3dsmc-bundle-adjustment/data/rgbd_dataset_freiburg1_xyz/");

    // Load video
    std::cout << "Initialize virtual sensor..." << std::endl;
    VirtualSensor sensor(5);
    if (!sensor.Init(filenameIn)) {
        std::cout << "Failed to initialize the sensor!\nCheck file path!"
                  << std::endl;
        return -1;
    }

    // We store a first frame as a reference frame. All next frames are tracked
    // relatively to the first frame.
    sensor.ProcessNextFrame();
    const auto &frame_1 = sensor.GetGrayscaleFrame();
    sensor.ProcessNextFrame();
    const auto &frame_2 = sensor.GetGrayscaleFrame();
    waitKey();

    if (frame_1.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }
    //-- Step 1: Detect the keypoints using ORB Detector
    int nFeatures = 1000;
    Ptr<ORB> detector = ORB::create(nFeatures);
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(frame_1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(frame_2, noArray(), keypoints2, descriptors2);

    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // filter matches using the Lowe's ratio test
    const float ratio_thresh = 1.0f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance <
            ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    vector<Point2f> matched_points1, matched_points2; // points that match

    for (int i=0; i<good_matches.size(); i++)
    {
        int idx1=good_matches[i].trainIdx;
        int idx2=good_matches[i].queryIdx;
        //use match indices to get the keypoints, add to the two lists of points
        matched_points1.push_back(keypoints1[idx1].pt);
        matched_points2.push_back(keypoints2[idx2].pt);
    }

    double intrinsics_data[] = {517.3, 0.0, 319.5,
                                0.0, 516.5, 255.3,
                                0.0, 0.0, 1.0};
    Mat intrinsics(3, 3, CV_64FC1, intrinsics_data);

/*    Point2d principal_point(319.5, 255.3);
    float focal = 517.0;*/
    Mat mask;
    Mat E = findEssentialMat(matched_points1, matched_points2, intrinsics, RANSAC, 0.9999, 1.0, mask);
//    Mat F = findFundamentalMat(matched_points1, matched_points2, RANSAC, 5.0, 0.999, mask);
    Mat R, T; // rotation and translation
    recoverPose(E, matched_points1, matched_points2, intrinsics, R, T);
    cout << "rotation: " << R << endl;
    cout << "translation: " << T << endl;
    cout << mask.size() << endl; // Mat
    cout << good_matches.size() << endl; // vector

    std::vector<DMatch> better_matches;
    for (int i=0; i<good_matches.size(); i++)
    {
        if (mask.at<int>(0, i) == 1) { // keep inliers only
            better_matches.push_back(good_matches[i]);
        }
    }
    cout << better_matches.size();

    // draw matches
    Mat img_matches;
    drawMatches(frame_1, keypoints1, frame_2, keypoints2, better_matches,
                img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // show detected matches
    imshow("Good Matches", img_matches);
    waitKey();

    return 0;
}
