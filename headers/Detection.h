//
// Created by witek on 08.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_DETECTION_H
#define BUNDLE_ADJUSTMENT_DETECTION_H

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
using namespace std;
using namespace cv; // SIFT and ORB
using namespace cv::xfeatures2d; // SURF

void getORB(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
            int &num_features);

void getSIFT(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
             int &num_features);

void getSURF(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
             int &hessian_threshold); // default hessian_threshold = 100

#endif // BUNDLE_ADJUSTMENT_DETECTION_H
