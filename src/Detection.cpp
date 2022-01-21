//
// Created by witek on 08.01.22.
//

#include <Detection.h>

void getORB(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
            int &num_features) {
    Ptr<ORB> detector = ORB::create(num_features);
    detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
}

void getSIFT(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
            int &num_features) {
    Ptr<SIFT> detector = SIFT::create(num_features);
    detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
}

void getSURF(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
             int &hessian_threshold) {
    Ptr<SURF> detector = SURF::create(hessian_threshold);
    detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
}