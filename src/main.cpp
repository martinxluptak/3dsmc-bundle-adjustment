#include <iostream>
#include "../headers/VirtualSensor.h"
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "Detection.cpp" // TODO: switch to Detection.h!
#include "Matching.cpp" // TODO: switch to Matching.h!
#include "Map3D.cpp" // TODO: switch to Map3D.h!
//#include <Eigen/Dense>
using namespace cv;
using namespace std;
using namespace Eigen;


int main() {

    vector<frame_correspondences> video_correspondences;
    int num_features = 1000;
    int keyframe_increment = 100;
    bool frame2_exists;

    // TODO: unify intrinsics data formats
    Matrix3f intr;
    intr << 517.3, 0.0, 319.5,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;
    double intrinsics_data[] = {517.3, 0.0, 319.5,
                                0.0, 516.5, 255.3,
                                0.0, 0.0, 1.0};
    Mat intrinsics(3, 3, CV_64FC1, intrinsics_data);

    string filename = string(".../rgbd_dataset_freiburg1_xyz/"); // SET TO <your_path>/rgbd_dataset_freiburg1_xyz/

    // load video
    cout << "Initialize virtual sensor..." << endl;

    // choose keyframes
    VirtualSensor sensor(keyframe_increment);
    if (!sensor.Init(filename)) {
        cout << "Failed to initialize the sensor.\nCheck file path." << endl;
        return -1;
    }

    sensor.ProcessNextFrame(); // load frame 0

    while(true){
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2, mask_ransac, E, R, T, mask_default;
        vector<vector<DMatch>> knn_matches;
        frame_correspondences correspondences;
        vector<DMatch> lowe_matches, ransac_matches;
        vector<Point2f> matched_points_lowe1, matched_points_lowe2;
        vector<Point2f> matched_points_ransac1, matched_points_ransac2;

        const auto &frame1 = sensor.GetGrayscaleFrame();
        const auto &depth_frame1 = sensor.GetDepthFrame(); // TODO: correct corresponding depth frame?

        frame2_exists = sensor.ProcessNextFrame();
        const auto &frame2 = sensor.GetGrayscaleFrame();
        const auto &depth_frame2 = sensor.GetDepthFrame(); // TODO: correct corresponding depth frame?
//        int num = sensor.GetCurrentFrameCnt();
//        cout << num << endl;

        if (frame1.empty()) {
            cout << "Could not open or find the image.\n" << endl;
            break;
        }

        if (!frame2_exists) {
            cout << "No more keyframe pairs.\n" << endl;
            break;
        }

        // detect keypoints and compute descriptors using ORB
        getORB(frame1, frame2, keypoints1, keypoints2, descriptors1, descriptors2, num_features);

        // match keypoints
        matchKeypoints(descriptors1, descriptors2, knn_matches);

        // filter matches using Lowe's ratio test
        lowe_matches = filterMatchesLowe(knn_matches, 0.9);
        cout << "detected Lowe matches: " << lowe_matches.size() << endl;
        tie(matched_points_lowe1, matched_points_lowe2) = getMatchedPoints(lowe_matches, keypoints1, keypoints2);

        // get essential matrix, perform RANSAC
        E = findEssentialMat(matched_points_lowe1, matched_points_lowe2, intrinsics, RANSAC, 0.9999, 2.0, mask_ransac);

        // filter matches using RANSAC
        ransac_matches = filterMatchesRANSAC(lowe_matches, mask_ransac);
        cout << "detected matches after RANSAC: " << ransac_matches.size() << endl;
        tie(matched_points_ransac1, matched_points_ransac2) = getMatchedPoints(ransac_matches, keypoints1, keypoints2);

        // register corresponding points

        cout << "here1" << endl;

        correspondences.frame1 = matched_points_ransac1;
        correspondences.frame2 = matched_points_ransac2;
        video_correspondences.push_back(correspondences);

        // get rotation and translation
        tie(R, T) = getPose(E, matched_points_lowe1, matched_points_lowe2, intrinsics);

        // display matches
//        mask_default = Mat::ones(1, lowe_matches.size(), CV_64F);
//        displayMatches("Matches Lowe", frame1, keypoints1, frame2, keypoints2, lowe_matches, mask_default);
//        displayMatches("Matches Lowe & RANSAC", frame1, keypoints1, frame2, keypoints2, lowe_matches, mask_ransac);

        // register 3d point (very "manual" solution for now)
//        vector<Vector3f> points3d = getPoints3D(correspondences, depth_frame1, intr);

//    debugging
//        for(int i=0; i<points3d.size(); i++){
//            cout << points3d[i] << endl << endl;
//        }
//        cout << points3d[2] << endl;

    }

    return 0;
}

// debugging
//    cv::Mat flat = mask_ransac.reshape(1, mask_ransac.total()*mask_ransac.channels());
//    vector<uchar> mask_ransac_vec = mask_ransac.isContinuous()? flat : flat.clone();
//    cout << mask_ransac_vec.size() << endl;
//    cout << count(mask_ransac_vec.begin(), mask_ransac_vec.end(), 1) << endl;