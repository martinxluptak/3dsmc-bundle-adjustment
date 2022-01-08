//
// Created by witek on 08.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_DETECTION_H
#define BUNDLE_ADJUSTMENT_DETECTION_H


#include "opencv2/features2d.hpp"
using namespace std;
using namespace cv;


void getORB(const Mat& frame1, const Mat& frame2,
            vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
            Mat& descriptors1, Mat& descriptors2, int& num_features);


#endif //BUNDLE_ADJUSTMENT_DETECTION_H
