//
// Created by witek on 08.01.22.
//

#include <Detection.h>

void getKeypointsAndDescriptors(const string &detector,
                                int &num_features, int &hessian_threshold,
                                const Mat &rgb, const Mat &depth,
                                vector <KeyPoint> &keypoints, Mat &descriptors) {
    if (detector == "ORB")
        getORB(rgb, depth, keypoints, descriptors, num_features);
    else if (detector == "SIFT")
        getSIFT(rgb, depth, keypoints, descriptors, num_features);
    else if (detector == "SURF")
        getSURF(rgb, depth, keypoints, descriptors, hessian_threshold);
    else
        cout << "Provide correct detector name" << endl;
}

void getORB(const Mat &rgb, const Mat &depth, vector <KeyPoint> &keypoints, Mat &descriptors,
            int &num_features) {
    Ptr <ORB> detector = ORB::create(num_features);
    detector->detectAndCompute(rgb, noArray(), keypoints, descriptors);
    pruneMINF(depth, keypoints, descriptors);
//    cout << "key: " << keypoints.size() << endl;
//    cout << "desc: " << descriptors.size() << endl;
}

void getSIFT(const Mat &rgb, const Mat &depth, vector <KeyPoint> &keypoints, Mat &descriptors,
             int &num_features) {
    Ptr <SIFT> detector = SIFT::create(num_features);
    detector->detectAndCompute(rgb, noArray(), keypoints, descriptors);
    pruneMINF(depth, keypoints, descriptors);
}

void getSURF(const Mat &rgb, const Mat &depth, vector <KeyPoint> &keypoints, Mat &descriptors,
             int &hessian_threshold) {
    Ptr <SURF> detector = SURF::create(hessian_threshold);
    detector->detectAndCompute(rgb, noArray(), keypoints, descriptors);
    pruneMINF(depth, keypoints, descriptors);
}

void pruneMINF(Mat depth, vector <KeyPoint> &keypoints, Mat &descriptors) {
    vector <KeyPoint> good_keypoints;
    Mat good_descriptors;

    for (int id = 0; id < keypoints.size(); id++) {
        int x = static_cast<int>(keypoints[id].pt.x);
        int y = static_cast<int>(keypoints[id].pt.y);
        if (depth.at<float>(y, x) > 1e-15) {
            good_keypoints.push_back(keypoints[id]);
            good_descriptors.push_back(descriptors.row(id));
        }
    }

    keypoints = good_keypoints;
    descriptors = good_descriptors;
}
