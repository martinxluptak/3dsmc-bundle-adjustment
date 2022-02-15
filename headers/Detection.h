//
// Created by witek on 08.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_DETECTION_H
#define BUNDLE_ADJUSTMENT_DETECTION_H

#include "iostream"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
using namespace std;
using namespace cv; // SIFT and ORB
using namespace cv::xfeatures2d; // SURF

void getKeypointsAndDescriptors(const string &detector,
                              int &num_features, int &hessian_threshold,
                              const Mat &rgb, const Mat &depth,
                              vector<KeyPoint> &keypoints, Mat &descriptors);

void getORB(const Mat &rgb, const Mat &depth, vector<KeyPoint> &keypoints, Mat &descriptors,
            int &num_features);

void getSIFT(const Mat &rgb, const Mat &depth, vector<KeyPoint> &keypoints, Mat &descriptors,
             int &num_features);

void getSURF(const Mat &rgb, const Mat &depth, vector<KeyPoint> &keypoints, Mat &descriptors,
             int &hessian_threshold);

// keep keypoints and descriptors with a positive depth value
void pruneMINF(Mat depth, vector<KeyPoint> &keypoints, Mat &descriptors);

#endif // BUNDLE_ADJUSTMENT_DETECTION_H
