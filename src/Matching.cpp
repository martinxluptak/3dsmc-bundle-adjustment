//
// Created by witek on 08.01.22.
//

#include<Matching.h>


void matchKeypoints(const Mat& descriptors1, const Mat& descriptors2, vector<vector<DMatch>>& knn_matches){
    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
};

vector<DMatch> filterMatchesLowe(const vector<vector<DMatch>>& knn_matches, const float& ratio)
{
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

vector<DMatch> filterMatchesRANSAC(const vector<DMatch>& matches, const Mat& mask){
    vector<DMatch> good_matches;
    cv::Mat flat = mask.reshape(1, mask.total() * mask.channels());
    vector<uchar> mask_vec = mask.isContinuous()? flat : flat.clone();

    for (int i=0; i<mask_vec.size(); i++)
    {
        if (mask_vec[i] == 1) { // keep inliers only
            good_matches.push_back(matches[i]);
        }
    }
    return good_matches;
}

pair<vector<Point2f>, vector<Point2f>> getMatchedPoints(const vector<DMatch>& matches,
                                                        const vector<KeyPoint>& keypoints1,
                                                        const vector<KeyPoint>& keypoints2)
{
    vector<Point2f> matched_points1, matched_points2; // points that match
    for (int i=0; i<matches.size(); i++)
    {
        int idx1=matches[i].trainIdx;
        int idx2=matches[i].queryIdx;
        //use match indices to get the keypoints, add to the two lists of points
        matched_points1.push_back(keypoints1[idx1].pt);
        matched_points2.push_back(keypoints2[idx2].pt);
    }
    return make_pair(matched_points1, matched_points2);
}

void displayMatches(string img_name, const Mat& frame1, const vector<KeyPoint>& keypoints1,
                    const Mat& frame2, const vector<KeyPoint>& keypoints2,
                    const vector<DMatch>& matches, const Mat& mask){
    Mat img_out;
    drawMatches(frame1, keypoints1, frame2, keypoints2, matches,
                img_out, Scalar::all(-1), Scalar::all(-1),
                mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // show detected matches
    imshow(img_name, img_out);
    waitKey();
};
