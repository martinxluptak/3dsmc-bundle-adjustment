//
// Created by witek on 08.01.22.
//

#include <Detection.h>

void getORB(const Mat &frame1, vector<KeyPoint> &keypoints1, Mat &descriptors1,
            int &num_features) {
  Ptr<ORB> detector = ORB::create(num_features);
  detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
}
