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

class ReprojectionConstraintChd {
public:
    ReprojectionConstraintChd(double _observed_x, double _observed_y, const Eigen::Matrix3d& _intrinsics) :
            observed_x{_observed_x}, observed_y{_observed_y}, intrinsics{_intrinsics} {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const
    {
        Sophus::SE3d se3;
        for(int i = 0; i < 7; ++i)
            se3.data()[i] = camera[i];

        Eigen::Vector4d vec {point[0], point[1], point[2], 1.0};
        Eigen::Vector4d p = se3.inverse().matrix() * vec;

        T predicted_x = p[0] / p[2] * intrinsics(0, 0) + intrinsics(0, 2);
        T predicted_y = p[1] / p[2] * intrinsics(1, 1) + intrinsics(1, 2);

//        std::cout << predicted_x << "," << predicted_y << " vs. " << observed_x << "," << observed_y << std::endl;

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

private:
    double observed_x, observed_y;
    const Eigen::Matrix3d intrinsics;
};

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
            : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {

        // Camera is W->C

        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);


        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Compute final projected point position.
        T predicted_x = fx * xp + cx;
        T predicted_y = fy * yp + cy;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const double fx, const double fy, const double cx, const double cy) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6, 3>(
                new SnavelyReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
    }

    double observed_x;
    double observed_y;
    double fx{}, fy{}, cx{}, cy{};
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

bool
optimizeDebug(ceresGlobalProblem &globalProblem, const Vector4d &intrinsics_initial, Vector4d &intrinsics_optimized,
              int opt_type) {

    // opt_type: 0 = ours, 1 = chd, 2 = ceres

    srand(time(0));
    ceres::Problem problem; // Optimization variables, poses, map and intrinsics_initial
    ceres::Solver::Summary summary;

    // Some optimization related stuff which is owned by the problem, as ceres says (so, can't be put into config)
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;
    auto *loss_function_repr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(globalProblem.HUB_P_REPR),
                                                              ceres::TAKE_OWNERSHIP);
    auto *loss_function_unpr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(globalProblem.HUB_P_UNPR),
                                                              ceres::TAKE_OWNERSHIP);

    // The intrinsics
    Matrix3d intrinsics = Matrix3d::Identity();
    intrinsics(0,0) = intrinsics_initial(0);
    intrinsics(1,1) = intrinsics_initial(1);
    intrinsics(0,2) = intrinsics_initial(2);
    intrinsics(1,2) = intrinsics_initial(3);

    // The poses
    Quaternion q1(1.0,0.0,0.0,0.0);  // pose0123 = quaternionxyzw, and quaterniond requires wxyz input
    Vector3d t(0.0,0.0,0.0);
    Sophus::SE3d::Point tr(t.x(), t.y(), t.z());
    Quaternion q2(0.8775826, 0.4794255, 0.0, 0.0);  // around 30 °
    Quaternion q3(0.5403023, 0.841471, 0.0, 0.0);  // around 60 ° (all relative to first frame)
    Sophus::SE3d T1(q1, tr);
    Sophus::SE3d T2(q2, tr);
    Sophus::SE3d T3(q3, tr);
    vector<Sophus::SE3d> poses = {T1, T2, T3}, gtPoses = {T1, T2, T3};

    // The map, in frame 1
    vector<double> mapVector= {
            .0, .0, 1.0,
            .2,.3, 2.0,
            -1.0,-3, .5,
            .3, -.5, .3,
            1.0, -.9, 4.0,
            -1.0, -1.0, .3,
            2.0, -2.2, 1.5,
            .2, .5,.5,
            -.9, -.01, 1.2,
            .7, .02, .2,
            1.2, 2.4, 2.0,
            -.4, .4, 1.2,
            -.2, -.8, 1.02,
            -.7, -.6, .3,
            1.6, -.2, .9,
            -0.1, 2.0, 2.4,
    };
    vector<Vector3d> map;
    for(int i=0; i< mapVector.size()/3; i++){
        map.emplace_back(mapVector[3*i], mapVector[3*i+1], mapVector[3*i+2]);
    }
    vector<Vector3d> gtMap = map;

    // Generate observations in every frame, and depth maps, plus, perturbations
    vector<vector<Vector2d>> O; // it will be O1, O2, O3
    vector<vector<double>> D; // it will be O1, O2, O3

    for (auto pose : poses){
        // Map into camera coordinates
        vector<Vector2d> O_temp;
        vector<double> D_temp;

        for (auto p_W : map){
            auto p_C = pose.inverse() * p_W;    // (C->W)^t * W
            D_temp.push_back(p_C(2));
            O_temp.emplace_back((intrinsics * p_C / p_C(2))(seq(0,1)) + Vector2d::Random());
        }

        O.push_back(O_temp);
        D.push_back(D_temp);

    }

    // More perturbations
    for (auto & ps : poses){
        Vector3d translationPert = Vector3d::Random(3) * .3;
        Vector4d quatPert = Vector4d::Random(4) * .1; // perturbation
        Vector4d quat_eigen( // perturbed quaternion, correct like this!
                ps.unit_quaternion().x(),
                ps.unit_quaternion().y(),
                ps.unit_quaternion().z(),
                ps.unit_quaternion().w()
        );
        Vector3d translation_eigen(ps.translation().data());
        translation_eigen += translationPert;
        quat_eigen += quatPert;
        quat_eigen /= quat_eigen.norm();

        Sophus::SE3d::Point translation = translation_eigen;
        Quaterniond quat(quat_eigen);
        Sophus::SE3d new_pose(quat, translation);

        ps = new_pose;

    }

    auto initial_poses = poses;

    vector<Vector<double, 6>> generalized_poses;
    if (opt_type==2){   // when using ceres suggested implementation

        for(auto ps : poses){
            auto quat = ps.inverse().unit_quaternion(); // Snavely reprojection wants the inverse
            auto trans = ps.inverse().translation();
            Vector3d angle_axis;
            Vector4d quat_eigen(quat.w(), quat.x(), quat.y(), quat.z());
            ceres::QuaternionToAngleAxis(quat_eigen.data(), angle_axis.data());  // accepts wxyz

            Vector<double, 6> generalized_pose;
            generalized_pose <<
                angle_axis.x(),
                angle_axis.y(),
                angle_axis.z(),
                trans.x(),
                trans.y(),
                trans.z(),
            generalized_poses.push_back(generalized_pose);
        }
    }

    // Debug, did we do everything correctly up to here?
//    for(int i = 0; i < O.size(); i++){
//        double E = .0;
//
//        auto O_curr = O[i];
//        auto D_curr = D[i];
//
//        for(int j = 0; j < O_curr.size(); j++){
//            Vector3d p_W = poses[i]*((intrinsics.inverse() * O_curr[j].homogeneous()) * D_curr[j]);
//            E += (p_W - map[j]).norm();
//        }
//    }

    // Add all parameter blocks to the problem, create the residuals
    // Intrinsics
    if (opt_type==0){   // only if we're using our method
        problem.AddParameterBlock(intrinsics_optimized.data(), 4);
        problem.AddResidualBlock(
                IntrinsicsPrior::create_cost_function(intrinsics_initial, globalProblem.WEIGHT_INTRINSICS),
                nullptr, // squared loss
                intrinsics_optimized.data()
        );
    }

    // Add map
    for(auto & pt : map){
        problem.AddParameterBlock(pt.data(), 3);
    }

    for(int i = 0; i < O.size(); i++){

        if(opt_type!=2){
            auto &pose = poses[i];
            problem.AddParameterBlock(pose.data(),
                                      Sophus::SE3d::num_parameters,
                                      local_parametrization_se3
            );
        }
        else{
            auto & gp = generalized_poses[i];
            problem.AddParameterBlock(gp.data(), 6);
        }


        // Run over all observations of this keyframe
        for (int j = 0; j < O[i].size(); j++) {

            Vector2d pix_coords = O[i][j];
            auto & map_point = map[j];

            if (opt_type==0)   // our method
            {
                auto & pose = poses[i];
                // Reprojection
                problem.AddResidualBlock(
                        ReprojectionConstraint::create_cost_function(pix_coords, 1.0),
                        loss_function_repr,
                        pose.data(), // (global) camera pose during observation
                        map_point.data(), // 3D point
                        intrinsics_optimized.data()
                );

                // Unprojection
                problem.AddResidualBlock(
                        DepthPrior::create_cost_function(pix_coords,
                                                         D[i][j], globalProblem.WEIGHT_UNPR),
                        loss_function_unpr,
                        pose.data(),
                        map_point.data(),
                        intrinsics_optimized.data());
            }
            else if(opt_type==1){
                auto & pose = poses[i];
                auto* constraint = new ReprojectionConstraintChd(pix_coords.x(), pix_coords.y(), intrinsics);
                auto cost_func_numeric = new ceres::NumericDiffCostFunction<ReprojectionConstraintChd, ceres::CENTRAL, 2, 7, 3>(constraint);
                problem.AddResidualBlock(cost_func_numeric,
                                         nullptr /* squared loss */,
                                         pose.data(),
                                         map_point.data());
            }
            else if(opt_type==2){
                auto & gp = generalized_poses[i];
                ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
                        pix_coords.x(), pix_coords.y(), intrinsics_initial.x(), intrinsics_initial.y(), intrinsics_initial.z(), intrinsics_initial.w());
                problem.AddResidualBlock(cost_function,
                                         nullptr /* squared loss */,
                                         gp.data(),
                                         map_point.data());
            }
        }
    }

    if (opt_type!=2){
        problem.SetParameterBlockConstant(poses[0].data()); // any pose, kept constant, will do
    }
    else    problem.SetParameterBlockConstant(generalized_poses[0].data());

    if(opt_type==0){problem.SetParameterBlockConstant(intrinsics_optimized.data());} // any pose, kept constant, will do
    ceres::Solve(globalProblem.options, &problem, &summary);

    if(opt_type==2){
        poses.clear();

        // This will convert at least the poses into their original format
        for(auto gp : generalized_poses){
            auto aa = gp(seq(0,2));
            auto trans = gp(seq(3,5));
            auto intr = gp(seq(6,9));
            Vector4d quat;
            ceres::AngleAxisToQuaternion(aa.data(), quat.data());
            Quaternion quat_quat(quat.x(), quat.y(), quat.z(), quat.w());
            Sophus::SE3d::Point trans_point = trans;
            Sophus::SE3d pose_sophus(quat_quat, trans_point);
            pose_sophus = pose_sophus.inverse();
            poses.push_back(pose_sophus);
        }
    }

    cout << "Pose errors after optimization: " << endl;
    for (int i = 1; i < poses.size(); i++) {
        Sophus::SE3d p_err((gtPoses[i].inverse() * poses[i]).matrix());
        cout << " Angle errors: " << p_err.angleX() / M_PI * 180 << ", " << p_err.angleY() / M_PI * 180 << ", "
             << p_err.angleZ() / M_PI * 180 << endl;
        cout << " Translational error: " << p_err.translation().norm() << endl
             << endl;
    }


    cout << "Map errors after optimization: " << endl;
    for (int i = 0; i < map.size(); i++) {
        auto m_err = (map[i] - gtMap[i]).norm();
        cout << i + 1 << endl << m_err << " | " << gtMap[i].transpose() << endl << endl;
    }
    cout << "Error in intrinsics: " << endl;
    for (int i = 0; i < 4; i++) {
        cout << intrinsics_optimized.data()[i] - intrinsics_initial.data()[i] << endl;
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
