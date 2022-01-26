//
// Created by witek on 08.01.22.
//

#include <Detection.h>

void getORB(const Mat &rgb, const Mat &depth,  vector<KeyPoint> &keypoints, Mat &descriptors,
            int &num_features) {
    Ptr<ORB> detector = ORB::create(num_features);
    detector->detectAndCompute(rgb, noArray(), keypoints, descriptors);
//    pruneMINF(depth, keypoints, descriptors);
//    cout << "key: " << keypoints.size() << endl;
//    cout << "desc: " << descriptors.size() << endl;
}

void getSIFT(const Mat &rgb, const Mat &depth, vector<KeyPoint> &keypoints, Mat &descriptors,
            int &num_features) {
    Ptr<SIFT> detector = SIFT::create(num_features);
    detector->detectAndCompute(rgb, noArray(), keypoints, descriptors);
    pruneMINF(depth, keypoints, descriptors);
}

void getSURF(const Mat &rgb, const Mat &depth, vector<KeyPoint> &keypoints, Mat &descriptors,
             int &hessian_threshold) {
    Ptr<SURF> detector = SURF::create(hessian_threshold);
    detector->detectAndCompute(rgb, noArray(), keypoints, descriptors);
    pruneMINF(depth, keypoints, descriptors);
}

void pruneMINF(Mat depth, vector<KeyPoint> &keypoints, Mat &descriptors){
    vector<int> good_ids;
    vector<KeyPoint> good_keypoints;
    Mat good_descriptors;

    for (int id=0; id < keypoints.size(); id++){
        int x = static_cast<int>(keypoints[id].pt.x);
        int y = static_cast<int>(keypoints[id].pt.y);
        if (depth.at<int>(y, x) > 0)
            good_ids.push_back(id);
    }

    for (int i=0; i<good_ids.size(); i++){
        good_keypoints.push_back(keypoints[i]);
        good_descriptors.push_back(descriptors.row(i));
    }

    keypoints = good_keypoints;
    descriptors = good_descriptors;
}
