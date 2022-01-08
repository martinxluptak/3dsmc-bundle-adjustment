//
// Created by witek on 08.01.22.
//

#include <Detection.h>


void getORB(const Mat& frame1, const Mat& frame2,
            vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2,
            Mat& descriptors1, Mat& descriptors2, int& num_features) {
    Ptr<ORB> detector = ORB::create(num_features);
    detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(frame2, noArray(), keypoints2, descriptors2);
}