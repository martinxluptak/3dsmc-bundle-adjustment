#include <iostream>

#include "../headers/VirtualSensor.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
//#include <Eigen/Dense>
#include "opencv2/calib3d.hpp"

#include "Map3D.cpp" // TODO: switch to Map3D.h!

using namespace cv;
using namespace std;
using namespace Eigen;


vector<DMatch> filterMatchesLowe(const vector<vector<DMatch>>& knn_matches, const float& ratio)
{
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

vector<DMatch> filterMatchesRANSAC(const vector<DMatch>& matches, const Mat& mask){
    vector<DMatch> good_matches;
    cv::Mat flat = mask.reshape(1, mask.total() * mask.channels());
    vector<uchar> mask_vec = mask.isContinuous()? flat : flat.clone();

    for (int i=0; i<mask_vec.size(); i++)
    {
        if (mask_vec[i] == 1) { // keep inliers only
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
//    cout << "rotation: " << R << endl;
//    cout << "translation: " << T << endl;
    return make_pair(R, T);
}

void displayMatches(string img_name, const Mat& frame1, const vector<KeyPoint>& keypoints1,
                    const Mat& frame2, const vector<KeyPoint>& keypoints2,
                    const vector<DMatch>& matches, const Mat& mask){
    Mat img_out;
    drawMatches(frame1, keypoints1, frame2, keypoints2, matches,
                img_out, Scalar::all(-1), Scalar::all(-1),
                mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // show detected matches
    imshow(img_name, img_out);
    waitKey();
};

struct frame_correspondences{
    vector<Point2f> frame1;
    vector<Point2f> frame2;
};

// saves points with -Inf depth
// TODO: skip those points!
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

void getORB(const Mat& frame1, const Mat& frame2,
            vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
            Mat& descriptors1, Mat& descriptors2, int& num_features) {
    Ptr<ORB> detector = ORB::create(num_features);
    detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(frame2, noArray(), keypoints2, descriptors2);
}

void matchKeypoints(const Mat& descriptors1, const Mat& descriptors2, vector<vector<DMatch>>& knn_matches){
    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
};









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

//    while(true){

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