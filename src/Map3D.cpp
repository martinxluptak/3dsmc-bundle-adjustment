//
// Created by witek on 07.01.22.
//

#include "Map3D.h"

void updateLandmarks(KeyFrame &old_frame, KeyFrame &new_frame,
                     const vector<DMatch> &matches, Map3D &map,
                     LandmarkId &landmark_id) {
    for (auto &match: matches) {

        int idx1 = match.queryIdx;

        // Add new landmark to the map in case:
        // 1. Initializing the map with the first two keyframes
        // 2. Landmark wasn't observed in the previous frame
        if (old_frame.frame_id == 0 ||
            !old_frame.global_points_map.count(idx1)) {
            addNewLandmark(old_frame, new_frame, match, map, landmark_id);
            landmark_id++;
        } else if (old_frame.global_points_map.count(idx1)) {
            // Else if it was observed in the previous frame, update the landmark
            // observations to include the new frame
            addLandmarkObservation(old_frame, new_frame, match, map);
        }
    }
}

void addNewLandmark(KeyFrame &kf_1, KeyFrame &kf_2, const DMatch &match,
                    Map3D &map, const LandmarkId &landmark_id) {

    int idx1 = match.queryIdx;
    int idx2 = match.trainIdx;

    // Update map
    Observation obs_1, obs_2;
    obs_1.first = kf_1.frame_id;
    obs_1.second = kf_1.keypoints[idx1].pt;
    obs_2.first = kf_2.frame_id;
    obs_2.second = kf_2.keypoints[idx2].pt;

    Landmark landmark;
    // Transform local point to world coordinates
    landmark.point = kf_1.T_w_c * kf_1.points3d_local[idx1];

    landmark.observations.push_back(obs_1);
    landmark.observations.push_back(obs_2);

    map[landmark_id] = landmark;

    // Update keyframe -> map correspondences
    kf_1.global_points_map[idx1] = landmark_id;
    kf_2.global_points_map[idx2] = landmark_id;
}

///
/// \brief updateLandmarkObservation Add a new frame as an observation of an
/// existing landmark
/// \param old_frame the frame that has already observed the landmark
/// \param new_frame The frame that we want to add as observation
///
void addLandmarkObservation(KeyFrame &old_frame, KeyFrame &new_frame,
                            const DMatch &match, Map3D &map) {
    int idx1 = match.queryIdx;
    int idx2 = match.trainIdx;

    Observation obs;
    obs.first = new_frame.frame_id;
    obs.second = new_frame.keypoints[idx2].pt;
    LandmarkId &landmark_id = old_frame.global_points_map[idx1];
    map[landmark_id].observations.push_back(obs);

    new_frame.global_points_map[idx2] = landmark_id;
}

vector<Vector3d> getLocalPoints3D(const vector<KeyPoint> &points,
                                  const Mat &depth_frame1,
                                  const Vector4d &intrinsics) {
    Vector3d point3d;
    vector<Vector3d> points3d;

    for (auto &point2d: points) {
        // debugging
        //    cout << "corresp. point func: " << point2d.pt << endl;

        const auto u = static_cast<double>(point2d.pt.x);
        const auto v = static_cast<double>(point2d.pt.y);

        const double z = static_cast<double> (depth_frame1.at<float>(trunc(v), trunc(u)));
        const double x = z * (u - intrinsics[2]) / intrinsics[0];
        const double y = z * (v - intrinsics[3]) / intrinsics[1];

        point3d << x, y, z;
        points3d.push_back(point3d);
    }
    return points3d;
}

Sophus::SE3d getExtrinsics(const Mat &E, const vector<Point2d> &matched_points1,
                           const vector<Point2d> &matched_points2,
                           const Mat &intrinsics) {
    Mat R, T;
    Matrix3d eigenR;
    Vector3d eigenT;
    Sophus::SE3d extrinsics;
    recoverPose(E, matched_points1, matched_points2, intrinsics, R, T);
    //    cout << "rotation: " << R << endl;
    //    cout << "translation: " << T << endl;

    cv2eigen(R, eigenR);
    cv2eigen(T, eigenT);

    return Sophus::SE3d(eigenR, eigenT);
}
