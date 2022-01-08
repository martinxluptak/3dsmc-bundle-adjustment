//
// Created by witek on 08.01.22.
//

#ifndef BUNDLE_ADJUSTMENT_MATCHING_H
#define BUNDLE_ADJUSTMENT_MATCHING_H


#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;


void matchKeypoints(const Mat& descriptors1, const Mat& descriptors2, vector<vector<DMatch>>& knn_matches);


vector<DMatch>  filterMatchesLowe(const vector<vector<DMatch>>& knn_matches, const float& ratio);


vector<DMatch>  filterMatchesRANSAC(const vector<DMatch>& matches, const Mat& mask);


pair<vector<Point2f>, vector<Point2f>> getMatchedPoints(const vector<DMatch>& matches,
                                                        const vector<KeyPoint>& keypoints1,
                                                        const vector<KeyPoint>& keypoints2);


void displayMatches(string img_name, const Mat& frame1, const vector<KeyPoint>& keypoints1,
                    const Mat& frame2, const vector<KeyPoint>& keypoints2,
                    const vector<DMatch>& matches, const Mat& mask);


struct frame_correspondences{
    vector<Point2f> frame1;
    vector<Point2f> frame2;
};


#endif //BUNDLE_ADJUSTMENT_MATCHING_H
