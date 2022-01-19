//
// Created by witek on 08.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_DETECTION_H
#define BUNDLE_ADJUSTMENT_DETECTION_H

#include "opencv2/features2d.hpp"
using namespace std;
using namespace cv;

void getORB(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
            int &num_features);

#endif // BUNDLE_ADJUSTMENT_DETECTION_H
