//
// Created by witek on 08.01.22.
//

#include <Matching.h>

void matchKeypoints(const string &detector,
                    const Mat &descriptors1, const Mat &descriptors2,
                    vector<vector<DMatch>> &knn_matches) {
    if (detector == "ORB") {
        Ptr <DescriptorMatcher> matcher =
                DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    } else if (detector == "SIFT" || detector == "SURF") {
        Ptr <DescriptorMatcher> matcher =
                DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    } else
        cout << "Provide correct detector name" << endl;
};

vector<DMatch> filterMatchesLowe(const vector<vector<DMatch>> &knn_matches,
                                 const float &ratio) {
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    return good_matches;
}

vector<DMatch> filterMatchesRANSAC(const vector<DMatch> &matches,
                                   const Mat &mask) {
    vector<DMatch> good_matches;
    cv::Mat flat = mask.reshape(1, mask.total() * mask.channels());
    vector<uchar> mask_vec = mask.isContinuous() ? flat : flat.clone();

    for (int i = 0; i < mask_vec.size(); i++) {
        if (mask_vec[i] == 1) { // keep inliers only
            good_matches.push_back(matches[i]);
        }
    }
    return good_matches;
}

// Computes transformation from the second camera to the first and a set of
// inliers Ransac and Arun's method
void initializeRelativePose(const vector<Vector3d> &points1,
                            const vector<Vector3d> &points2,
                            const vector<DMatch> &matches,
                            vector<DMatch> &inliers, Sophus::SE3d &pose) {
    // create a 3D-3D adapter

    points_t points_1, points_2;
    for (unsigned int i = 0; i < matches.size(); i++) {
        // queryIdx -> 1st frame
        int idx1 = matches[i].queryIdx;
        //  trainIdx -> 2nd frame
        int idx2 = matches[i].trainIdx;
        points_1.push_back(points1[idx1]);
        points_2.push_back(points2[idx2]);
    }

    point_cloud::PointCloudAdapter adapter(points_1, points_2);
    // create a RANSAC object
    sac::Ransac<sac_problems::point_cloud::PointCloudSacProblem> ransac;
    // create the sample consensus problem
    std::shared_ptr<sac_problems::point_cloud::PointCloudSacProblem>
            relposeproblem_ptr(
            new sac_problems::point_cloud::PointCloudSacProblem(adapter));
    // run ransac
    ransac.sac_model_ = relposeproblem_ptr;
    ransac.threshold_ = 0.1;
    ransac.probability_ = 0.999;
//  ransac.max_iterations_ = 100;
    ransac.computeModel(0);
    // return the result
    for (int inlier: ransac.inliers_) {
        inliers.push_back(matches[inlier]);
    }

    transformation_t best_transformation = ransac.model_coefficients_;
    pose.setRotationMatrix(best_transformation.leftCols<3>());
    pose.translation() = best_transformation.rightCols<1>();
}

pair<vector<Point2d>, vector<Point2d>>
getMatchedPoints(const vector<DMatch> &matches,
                 const vector<KeyPoint> &keypoints1,
                 const vector<KeyPoint> &keypoints2) {
    vector<Point2d> matched_points1, matched_points2; // points that match
    for (int i = 0; i < matches.size(); i++) {
        int idx1 = matches[i].queryIdx;
        int idx2 = matches[i].trainIdx;
        // use match indices to get the keypoints, add to the two lists of points
        matched_points1.push_back(keypoints1[idx1].pt);
        matched_points2.push_back(keypoints2[idx2].pt);
    }
    return make_pair(matched_points1, matched_points2);
}

void displayMatches(string img_name, const Mat &frame1,
                    const vector<KeyPoint> &keypoints1, const Mat &frame2,
                    const vector<KeyPoint> &keypoints2,
                    const vector<DMatch> &matches, const Mat &mask) {
    Mat img_out;
    drawMatches(frame1, keypoints1, frame2, keypoints2, matches, img_out,
                Scalar::all(-1), Scalar::all(-1), mask,
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // show detected matches
    imshow(img_name, img_out);
    waitKey();
};
