//
// Created by martinxluptak on 12/15/2021.
//

// Source copied from OpenCV: opencv/samples/cpp/tutorial_code/features2D/feature_flann_matcher/SURF_FLANN_matching_Demo.cpp

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <gtest/gtest.h>
#include <filesystem>

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
using std::filesystem::current_path;

int main(int argc, char* argv[]) {
    Mat img1 = imread( "<project_directory>/samples/data/box.png", IMREAD_GRAYSCALE); // TODO change to your project directory
    Mat img2 = imread( "<project_directory/samples/data/box_in_scene.png", IMREAD_GRAYSCALE ); //TODO change to your project directory
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    // namedWindow("Good Matches", WINDOW_AUTOSIZE);// Create a window for display.
    imshow("Good Matches", img_matches );

    waitKey();

    return 0;
}
