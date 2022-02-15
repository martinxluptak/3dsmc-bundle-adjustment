//
// Created by lupta on 1/20/2022.
//

#include "OptimizationUtils.h"
#include <string>
#include "Eigen/Dense"
#include <fstream>
#include <utility>
#include <ceres/ceres.h>
#include <rotation.h>
#include "sophus/local_parameterization_se3.hpp"
#include "BundleAdjustmentConfig.h"
#include "Map3D.h"
#include "nearest_interp_1d.cpp"


using namespace std;
using namespace Eigen;

class ReprojectionConstraint {
public:
    ReprojectionConstraint(Vector2d pPix, const double &weight) : p_pix(std::move(pPix)), weight(weight) {}

    template<typename T>
    bool operator()(const T *const pose, const T *const x_w, const T *const intr,
                    T *residuals) const {   //constant array of type constant T

        Matrix<T, 3, 3> intrinsics_tmp = Matrix<T, 3, 3>::Identity();
        intrinsics_tmp(0, 0) = intr[0];
        intrinsics_tmp(1, 1) = intr[1];
        intrinsics_tmp(0, 2) = intr[2];
        intrinsics_tmp(1, 2) = intr[3];
        const Matrix<T, 3, 3> intrinsics = intrinsics_tmp;

        const Quaternion<T> q(pose[3], pose[0], pose[1],
                              pose[2]);  // pose0123 = quaternionxyzw, and quaterniond requires wxyz input
        const Vector<T, 3> t(pose[4], pose[5], pose[6]);
        const Vector<T, 3> p_W(x_w[0], x_w[1], x_w[2]);

        const Vector<T, 3> p_C = q.matrix().transpose()  * (p_W - t);
        const Vector<T, 2> p_pix_est = (intrinsics * p_C / p_C.z())(seq(0, 1));

        // Reprojection error
        residuals[0] = T(sqrt(weight)) * (T(p_pix_est[0]) - T(p_pix[0]));
        residuals[1] = T(sqrt(weight)) * (T(p_pix_est[1]) - T(p_pix[1]));

        return true;
    }

    // Hide the implementation from the user
    static ceres::CostFunction *create_cost_function(const Vector2d &pPix, const double &weight) {
        return new ceres::AutoDiffCostFunction<ReprojectionConstraint, 2, 7, 3, 4>(
                new ReprojectionConstraint(pPix, weight)
        );
    }

private:
    Vector2d p_pix;
    double weight;  // to weight this residual
};


// We make sure not to get too far away from the depths measured from the depth sensor
class DepthPrior {
public:
    DepthPrior(Vector2d pPix, const double &depth, const double &weight) :
            p_pix(std::move(pPix)),
            depth(depth),
            weight(weight) {}

    template<typename T>
    bool operator()(const T *const pose, const T *const x_w, const T *const intr,
                    T *residuals) const {   //constant array of type constant T

        Matrix<T, 3, 3> intrinsics_tmp = Matrix<T, 3, 3>::Identity();
        intrinsics_tmp(0, 0) = intr[0];
        intrinsics_tmp(1, 1) = intr[1];
        intrinsics_tmp(0, 2) = intr[2];
        intrinsics_tmp(1, 2) = intr[3];
        const Matrix<T, 3, 3> intrinsics = intrinsics_tmp;

        const Quaternion<T> q(pose[3], pose[0], pose[1],
                              pose[2]);  // pose0123 = quaternionxyzw, and quaterniond requires wxyz input
        const Vector<T, 3> t(pose[4], pose[5], pose[6]);
        const Vector<T, 3> p_W(x_w[0], x_w[1], x_w[2]);

        const Vector<T, 3> p_C = q.matrix().transpose() * (p_W - t);
        T d = p_C[2];

        residuals[0] = T(sqrt(weight)) * (T(depth) - d);

        return true;
    }

    static ceres::CostFunction *
    create_cost_function(const Vector2d &pPix, const double &depth, const double &weight) {
        return new ceres::AutoDiffCostFunction<DepthPrior, 1, 7, 3, 4>(
                new DepthPrior(pPix, depth, weight)
        );
    }

private:
    Vector2d p_pix;
    double weight;  // to weight this residual
    double depth;
};

// We make sure not to get too far away from the ROS default intrinsics
class IntrinsicsPrior {
public:
    IntrinsicsPrior(Vector4d intr_prior, const double &weight) :
            intr_prior(std::move(intr_prior)),
            weight(weight) {}

    template<typename T>
    bool operator()(const T *const intr,
                    T *residuals) const {   //constant array of type constant T

        for (int i = 0; i < 4; i++) {
            residuals[i] = T(sqrt(weight)) * (T(intr_prior[i] - intr[i]));
        }

        return true;
    }

    static ceres::CostFunction *
    create_cost_function(const Vector4d &intr_prior, const double &weight) {
        return new ceres::AutoDiffCostFunction<IntrinsicsPrior, 4, 4>(
                new IntrinsicsPrior(intr_prior, weight)
        );
    }

private:
    Vector4d intr_prior;
    double weight;  // to weight this residual
};

/**
 * Reads camera intrinsics data from the target file.
 *
 * Intrinsics format: (fx, fy, cx, cy)
 *
 * @return a vector of doubles containing intrinsics in aforementioned format.
 */
Vector4d read_camera_intrinsics_from_file(const string &file_path) {
    ifstream infile;
    infile.open(file_path);
    double fx, fy, cx, cy, d0, d1, d2, d3, d4;
    while (infile >> fx >> fy >> cx >> cy >> d0 >> d1 >> d2 >> d3 >> d4) {
        // reads only the first line and then closes.
    }
    infile.close();

    Vector4d intrinsics;
    intrinsics << fx, fy, cx, cy;
    return intrinsics;
}

void write_keyframe_poses_to_file(const string &file_path, const vector<KeyFrame> & keyframes) {
    ofstream outfile(file_path);
    for (auto &keyframe: keyframes) {
        auto & t = keyframe.T_w_c.translation();
        auto & q = keyframe.T_w_c.unit_quaternion();
        outfile << keyframe.timestamp << " ";
        outfile << t.x() << " " << t.y() << " " << t.z() << " ";
        outfile << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
        outfile << "\n";
    }
    outfile.close();
    cout << "Writing keyframe poses to file: " << file_path << " successful.";
}

int findLocalPointIndex(const KeyFrame &keyframe, const int landmarkId) {
    int local_index = -1;
    for (auto &map_it: keyframe.global_points_map) {
        if (map_it.second == landmarkId) {
            local_index = map_it.first;
        }
    }
    return local_index;
}

int countConstraints(const Map3D &map, const vector<KeyFrame> &keyframes, int kf_i, int kf_f) {

    int admissible_obs = 0;

    // Poses, map points (only the relevant ones)
    for (int kf_n = kf_i; kf_n <= kf_f; kf_n++) {

        // Modify the poses, make them relative to the first frame, add them to the problem
        auto &curr_kf = keyframes[kf_n];

        // Run over all observations of this keyframe
        for (auto index_pair: curr_kf.global_points_map) {
            int localId = index_pair.first;

            auto depth = curr_kf.points3d_local[localId](2);

            // Discard this observation if it has negative depth. todo: to be removed
            if (depth <= 1e-15) {
                cout << "Negative and thus unadmissible depth " << depth << endl;
                continue;
            }

            admissible_obs++;

        }
    }

    return admissible_obs;

}

bool windowOptimize(ceresGlobalProblem &globalProblem, int kf_i, int kf_f, vector<KeyFrame> &keyframes, Map3D &map,
                    const Vector4d &intrinsics_initial, Vector4d &intrinsics_optimized) {

    ceres::Problem problem; // Optimization variables, poses, map and intrinsics_initial
    ceres::Solver::Summary summary;

    // Some optimization related stuff which is owned by the problem, as ceres says (so, can't be put into config)
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;
    auto *loss_function_repr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(globalProblem.HUB_P_REPR),
                                                              ceres::TAKE_OWNERSHIP);
    auto *loss_function_unpr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(globalProblem.HUB_P_UNPR),
                                                              ceres::TAKE_OWNERSHIP);


    set<int> already_observed_pts;  // we will visit a 3D map point more than once below. We need this not to apply T1->i twice

    auto initialPose = keyframes[kf_i].T_w_c;
    auto initialPoseInv = keyframes[kf_i].T_w_c.inverse();

    // Add all parameter blocks to the problem, create the residuals
    // Intrinsics
    problem.AddParameterBlock(intrinsics_optimized.data(), 4);
    problem.AddResidualBlock(
            IntrinsicsPrior::create_cost_function(intrinsics_initial, globalProblem.WEIGHT_INTRINSICS),
            nullptr, // squared loss
            intrinsics_optimized.data()
    );
    int admissible_obs = countConstraints(map, keyframes, kf_i, kf_f);
    // Poses, map points (only the relevant ones)
    for (int kf_n = kf_i; kf_n <= kf_f; kf_n++) {

        // Modify the poses, make them relative to the first frame, add them to the problem
        auto &curr_kf = keyframes[kf_n];
        curr_kf.T_w_c = Sophus::SE3d(initialPoseInv * curr_kf.T_w_c);
        // This is the respective pose to be optimized
        auto &pose = curr_kf.T_w_c;
        problem.AddParameterBlock(pose.data(),
                                  Sophus::SE3d::num_parameters,
                                  local_parametrization_se3
        );

        // Run over all observations of this keyframe
        for (auto index_pair: curr_kf.global_points_map) {
            int landmarkId = index_pair.second;   // id into the 3d map of this observed point
            int localId = index_pair.first;

            auto depth = curr_kf.points3d_local[localId](2);
            Vector2d pix_coords(curr_kf.keypoints[localId].pt.x, curr_kf.keypoints[localId].pt.y);

            // Discard this observation if it has negative depth. todo: to be removed
            if (depth <= 1e-15) {
//                cout << "Negative and thus unadmissible depth" << endl;
                continue;
            }
            // Check if we never observed such a point. If so, move it to 1st frame of reference
            auto &map_point = map.at(landmarkId);
            if (already_observed_pts.find(landmarkId) == already_observed_pts.end()) {
                already_observed_pts.insert(landmarkId);
                // Move this point into frame kf_i
                map_point.point = initialPoseInv * map_point.point;   //yes, this works
                problem.AddParameterBlock(map_point.point.data(), 3);
            }

            // Reprojection
            problem.AddResidualBlock(
                    ReprojectionConstraint::create_cost_function(pix_coords, 1.0/admissible_obs),
                    loss_function_repr,
                    pose.data(), // (global) camera pose during observation
                    map_point.point.data(), // 3D point
                    intrinsics_optimized.data()
            );

            // Unprojection
            problem.AddResidualBlock(
                    DepthPrior::create_cost_function(pix_coords,
                                                     depth, globalProblem.WEIGHT_UNPR/admissible_obs),
                    loss_function_unpr,
                    pose.data(),
                    map_point.point.data(),
                    intrinsics_optimized.data());
        }

    }

    problem.SetParameterBlockConstant(keyframes[kf_i].T_w_c.data()); // any pose, kept constant, will do
    ceres::Solve(globalProblem.options, &problem, &summary);

    // Re-put everything in the correct frame, both poses and map points which we messed up with
    for (int kf_n = kf_i; kf_n <= kf_f; kf_n++) {
        // Remodify the poses
        auto &curr_kf = keyframes[kf_n];
        curr_kf.T_w_c = Sophus::SE3d(initialPose * curr_kf.T_w_c);  // 1->W * C->1 = C->W
    }
    for (int lId: already_observed_pts) {
        map.at(lId).point = initialPose * map.at(lId).point;
    }

    return true;
}

/**
 * Read the ground_truth.txt file and fetch the pose from ground_truth.txt timestamp, which
 * is nearest to first_timestamp parameter.
 *
 * @param first_timestamp timestamp of the pose to fetch.
 * @param ground_truth_file_path ground_truth.txt file to read.
 * @return
 */
Sophus::SE3d getFirstPose(const string &first_timestamp, const string &ground_truth_file_path) {

    // Read gt file
    ifstream infile(ground_truth_file_path);
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    vector<double> timestamps, txs, tys, tzs, qxs, qys, qzs, qws;
    // ignore file header.
    string line;
    // ignore file header
    getline(infile, line);
    getline(infile, line);
    getline(infile, line);
    while (infile >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
        timestamps.push_back(timestamp);
        txs.push_back(tx);
        tys.push_back(ty);
        tzs.push_back(tz);
        qxs.push_back(qx);
        qys.push_back(qy);
        qzs.push_back(qz);
        qws.push_back(qw);
    }
    infile.close();

    // Get the right pose index
    vector<double> interpolatedTimestamps;
    vector<int> timestampIndices;
    vector<double> firstTimestampVector;
    firstTimestampVector.emplace_back(stod(first_timestamp));
    tie(interpolatedTimestamps, timestampIndices) =
            nearest_interp_1d(timestamps, timestamps, firstTimestampVector);
    int firstPoseIndex = timestampIndices[0];

    // Create the first pose
    Sophus::SE3d firstPose;
    Quaternion q(
            qws[ firstPoseIndex ],
            qxs[ firstPoseIndex ],
            qys[ firstPoseIndex ],
            qzs[ firstPoseIndex ]);  // pose0123 = quaternionxyzw, and quaterniond requires wxyz input
    Vector3d t(
            txs[ firstPoseIndex ],
            tys[ firstPoseIndex ],
            tzs[ firstPoseIndex ]);
    Sophus::SE3d::Point tr(t.x(), t.y(), t.z());
    Sophus::SE3d initialPose(q, tr);
//    cout.precision(17);
//    cout << "Input timestamp: " << first_timestamp << endl;
//    cout << "Closest timestamp: " << interpolatedTimestamps[0] << endl;
//    cout << "Pose (eigen, sophus): " << t.transpose() << " " << initialPose.translation().transpose() << " | " << initialPose.unit_quaternion() << endl;

    return initialPose;
}

void poseOffset(vector<KeyFrame> &keyframes, const Sophus::SE3d &initial_pose) {
    auto delta_pose = initial_pose * keyframes[0].T_w_c.inverse();  // C0 -> W * (C0 -> W_fictitious)^{-1} = W_fictitious -> W

    // Given a sequence of keyframes with identity being the first pose, make initial_pose the first pose
    for (auto & kf : keyframes){
        kf.T_w_c =  delta_pose *  kf.T_w_c;
    }
}
