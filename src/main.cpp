#include <iostream>

#include "../headers/VirtualSensor.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;

int main() {
  std::string filenameIn =
      std::string("../../Data/rgbd_dataset_freiburg1_xyz/");

  // Load video
  std::cout << "Initialize virtual sensor..." << std::endl;
  VirtualSensor sensor(100);
  if (!sensor.Init(filenameIn)) {
    std::cout << "Failed to initialize the sensor!\nCheck file path!"
              << std::endl;
    return -1;
  }

  // We store a first frame as a reference frame. All next frames are tracked
  // relatively to the first frame.
  sensor.ProcessNextFrame();
  const auto &frame_1 = sensor.GetGrayscaleFrame();
  sensor.ProcessNextFrame();
  const auto &frame_2 = sensor.GetGrayscaleFrame();
  waitKey();

  if (frame_1.empty()) {
    cout << "Could not open or find the image!\n" << endl;
    return -1;
  }
  //-- Step 1: Detect the keypoints using ORB Detector
  int nFeatures = 250;
  Ptr<ORB> detector = ORB::create(nFeatures);
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detectAndCompute(frame_1, noArray(), keypoints1, descriptors1);
  detector->detectAndCompute(frame_2, noArray(), keypoints2, descriptors2);

  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
  std::vector<std::vector<DMatch>> knn_matches;
  matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }

  //-- Draw matches
  Mat img_matches;
  drawMatches(frame_1, keypoints1, frame_2, keypoints2, good_matches,
              img_matches, Scalar::all(-1), Scalar::all(-1),
              std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  //-- Show detected matches
  imshow("Good Matches", img_matches);
  waitKey();
  return 0;
}
