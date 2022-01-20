//
// Created by lupta on 12/21/2021.
//

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>
#include "Eigen.h" // TODO include all header files in all targets
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "sophus/local_parameterization_se3.hpp"
#include "nearest_interp_1d.cpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <cmath>
#include <algorithm>
#include <rotation.h>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace cv::xfeatures2d;

class Correspondence {
public:

    int getIndex() const {
        return index;
    }

    const vector<string> &getRgbFile() const {
        return rgb_file;
    }

    const vector<string> &getDepthFile() const {
        return depth_file;
    }

    const vector<Vector2d> &getObsPix() const {
        return obs_pix;
    }

    const vector<double> &getDepths() const {
        return depths;
    }

    const vector<Sophus::SE3d> &getPoses() const {
        return poses;
    }

    const vector<int> &getPoseIndices() const {
        return pose_indices;
    }

    Correspondence(int index, const vector<string> &rgbFile, const vector<string> &depthFile,
                   const vector<string> &gtFile, const vector<Vector2d> &obsPix, const vector<Sophus::SE3d> &poses,
                   const vector<int> &poseIndices, const vector<double> &depths) : index(index), rgb_file(rgbFile),
                                                                                   depth_file(depthFile),
                                                                                   gt_file(gtFile), obs_pix(obsPix),
                                                                                   poses(poses),
                                                                                   pose_indices(poseIndices),
                                                                                   depths(depths) {}

protected:
    int index;  // of the 3D point being observed
    vector<string> rgb_file, depth_file, gt_file;
    vector<Vector2d> obs_pix;   // a vector of pixel coordinates, the observations of our 3D point
    vector<Sophus::SE3d> poses; // vector of poses, for every camera which made an observation of the 3D points, format B->W
    vector<int> pose_indices;   // inside some vector of poses (the one we'll optimize)
    vector<double> depths;  // depth for every observation

public:

    // For debugging, it modifies the pixel correspondences to be perfect.
    void makeSynthetic(Matrix<double, 3, 3> &intrinsics) {

        auto base_pose = poses[0];
        auto base_pix = obs_pix[0];
        auto base_depth = depths[0];
        Vector3d p_W = base_pose.rotationMatrix() * base_depth * intrinsics.inverse() * base_pix.homogeneous() +
                       base_pose.translation();

        for (int i = 0; i < depths.size(); i++) {
            // Take pixel of the first frame, reproject it into this frame
            auto pose = poses[i];
            Vector3d p_C = pose.rotationMatrix().transpose() * (p_W - pose.translation());
            Vector3d h_pix = (intrinsics * p_C / p_C[2]);
            obs_pix[i] = h_pix(seq(0, 1));
            obs_pix[i][0] += (rand() % 100 - 50) / 100.0 * .0;
            obs_pix[i][1] += (rand() % 100 - 50) / 100.0 * .0;
            depths[i] = p_C[2] + (rand() % 100 - 50) / 100.0 * .0;

        }
    }

    // Change the world into the frame T, so that the poses are B->T, not B->W
    void putHere(Matrix<double, 4, 4> T) {
        for (int i = 0; i < depths.size(); i++) {
            // Take pixel of the first frame, reproject it into this frame

            auto pose = poses[i];
            poses[i] = Sophus::SE3d(T.inverse() * pose.matrix());

        }
    }

    void setFirstDepth(double d) {
        depths[0] = d;
    }

};

// Reprojection error
class ReprojectionConstraint {
public:
    ReprojectionConstraint(const Vector2d &pPix) : p_pix(pPix) {}

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

        const Vector<T, 3> p_C = q.inverse().matrix() * (p_W - t);
        const Vector<T, 2> p_pix_est = (intrinsics * p_C / p_C.z())(seq(0, 1));

        // Reprojection error
        residuals[0] = T(p_pix_est[0]) - T(p_pix[0]);
        residuals[1] = T(p_pix_est[1]) - T(p_pix[1]);

        return true;
    }

    // Hide the implementation from the user
    static ceres::CostFunction *create_cost_function(const Vector2d &pPix) {
        return new ceres::AutoDiffCostFunction<ReprojectionConstraint, 2, 7, 3, 4>(
                new ReprojectionConstraint(pPix)
        );
    }

private:
    Vector2d p_pix;
};

// We make sure not to get too far away from the depths measured from the depth sensor
class DepthPrior {
public:
    DepthPrior(const Vector2d &pPix, const double &depth, const double &weight) :
            p_pix(pPix),
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
    Matrix3d intrinsics;
    Vector2d p_pix;
    double weight;  // to weight this residual
    double depth;
};

// We make sure not to get too far away from the ROS default intrinsics
class IntrinsicsPrior {
public:
    IntrinsicsPrior(const Vector4d &intr_prior, const double &weight) :
            intr_prior(intr_prior),
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

int main(int argc, char *argv[]) {

    srand(time(0)); // Set random generator

    //-------------------------------------------------- TAKE CARE OF THE INTRINSICS

    ifstream infile;
    const string DEFAULT_INTRINSICS = "../../Data/ros_default_intrinsics.txt";
    infile.open(DEFAULT_INTRINSICS);
    Matrix3d intrinsics = Matrix3d::Identity();
    double fx, fy, cx, cy, d0, d1, d2, d3, d4;
    while (infile >> fx >> fy >> cx >> cy >> d0 >> d1 >> d2 >> d3 >> d4) {
        // reads only one line
    }
    infile.close();

    intrinsics(0, 0) = fx;
    intrinsics(1, 1) = fy;
    intrinsics(0, 2) = cx;
    intrinsics(1, 2) = cy;

    auto intrinsics_inv = intrinsics.inverse();

    //-------------------------------------------------- FILL THE CORRESPONDENCES ARRAY

    infile.open("../../Data/three_frames_teddy.txt");
    // Ignore file header
    string line;
    getline(infile, line);

    vector<Sophus::SE3d> gtPoses;   // vector of the poses of the cameras which are used in this frames. Will be an optimization variable

    std::vector<Correspondence> correspondences;
    int current_correspondence_id = 1;
    vector<string> rgb_files, depth_files, gt_files;
    vector<Vector2d> obs_pixs;
    vector<double> depths;
    vector<Sophus::SE3d> poses_v;
    int correspondence_id;
    string rgb_file_ts, depth_file_ts, gt_ts;
    double corr_x, corr_y, tx, ty, tz, qx, qy, qz, qw, d;
    vector<string> uniqueNeededImageNames;
    int ui = 0;
    vector<int> pose_indices;

    while (infile >> correspondence_id >> rgb_file_ts >> depth_file_ts >> gt_ts >> corr_x >> corr_y >> tx >> ty >> tz
                  >> qx >> qy >> qz >> qw) {

        if (correspondence_id != current_correspondence_id) {
            correspondences.emplace_back(
                    current_correspondence_id,
                    rgb_files,
                    depth_files,
                    gt_files,
                    obs_pixs,
                    poses_v,
                    pose_indices,
                    depths
            );
            current_correspondence_id = correspondence_id;
            rgb_files.clear();
            depth_files.clear();
            gt_files.clear();
            obs_pixs.clear();
            poses_v.clear();
            pose_indices.clear();
            depths.clear();
        }

        Quaterniond q(qw, qx, qy,
                      qz);
        Sophus::SE3d::Point tr(tx, ty, tz);
        poses_v.emplace_back(q, tr);

        // To fill gtPoses
        auto iter_found = std::find(uniqueNeededImageNames.begin(), uniqueNeededImageNames.end(), rgb_file_ts);
        // If we've already seen this image
        if (iter_found != uniqueNeededImageNames.end()) {
            pose_indices.push_back(distance(uniqueNeededImageNames.begin(), iter_found));
        } else {
            if (uniqueNeededImageNames.begin() != uniqueNeededImageNames.end())
                ui++;
            pose_indices.push_back(ui);
            uniqueNeededImageNames.emplace_back(rgb_file_ts);
            gtPoses.emplace_back(q, tr);
        }

        // Let's also get the depth
        rgb_files.push_back(rgb_file_ts);
        depth_files.push_back(depth_file_ts);
        obs_pixs.emplace_back(corr_x, corr_y);
        gt_files.emplace_back(gt_ts);
        auto depth_img = imread("../../Data/rgbd_dataset_freiburg3_teddy/depth/" + depth_file_ts + ".png",
                                IMREAD_UNCHANGED);
        int j = floor(corr_x);
        int i = floor(corr_y);
        d = depth_img.at<uint16_t>(i, j) / 5000.0;
        if (d < 1e-6) {
            cout << "Small depth, check it out again" << endl;
            break;
        }
        depths.push_back(d);

    }
    // Account for last line in file
    correspondences.emplace_back(
            current_correspondence_id,
            rgb_files,
            depth_files,
            gt_files,
            obs_pixs,
            poses_v,
            pose_indices,
            depths
    );

    infile.close();

    //-------------------------------------------------- PERTURB GROUND TRUTH DATA FOR INITIALIZATION

    vector<Sophus::SE3d> initialPoses;  // initial guess for optimization algorithm

    auto base_pose = gtPoses[0].matrix();

    // Put everything into the first frame of reference, for simplicity
    for (int i = 0; i < gtPoses.size(); i++) {
        Vector<double, 7> p(gtPoses[i].data());
        gtPoses[i] = Sophus::SE3d(base_pose.inverse() * gtPoses[i].matrix());

    }
    // Update also the poses in the correspondence structure
    for (auto &cs: correspondences) {
        cs.putHere(base_pose);
    }

    // Add some perturbation
    double PERTURBATION_RATIO = 1e-1;
    bool perturb = false;
    for (auto &ps: gtPoses) {

        Vector3d translationPert = Vector3d::Random(3) * PERTURBATION_RATIO;
        Vector4d quatPert = Vector4d::Random(4) * PERTURBATION_RATIO; // perturbation
        Vector4d quat_eigen(
                ps.unit_quaternion().x(),
                ps.unit_quaternion().y(),
                ps.unit_quaternion().z(),
                ps.unit_quaternion().w()
        );
        Vector3d translation_eigen(ps.translation().data());

        // Perturbation and creation
        if (perturb) {  // not perturbing the first guy, it is kept fixed in the optimization
            translation_eigen += translationPert;
            quat_eigen += quatPert;
            quat_eigen /= quat_eigen.norm();

        }
        perturb = true;
        Sophus::SE3d::Point translation = translation_eigen;
        Quaterniond quat(quat_eigen);
        initialPoses.emplace_back(quat, translation);

    }

    //-------------------------------------------------- INITIALIZE 3D POINTS FROM THE DEPTH FILES


    // We initialize a 3D point with an average of all its observation
    vector<Vector3d> initialWorldMap;   // will be optimization varibale
    vector<Vector3d> gtWorldMap;

    double MAP_PERTURBATION_RATIO = 2e-1;
    // TODO: this works because of the order in which the correspondences are stored in the frames .txr file. Inform other people about this
    for (auto &correspondence: correspondences) {

        // Let's initialize the 3D point using a mean of all observations
        Vector3d point_world = Vector3d::Zero();
        Vector3d point_world_tmp;
        int count(0);

        for (int i = 0; i < correspondence.getObsPix().size(); i++) {

            point_world_tmp.setOnes();
            auto point_pix = correspondence.getObsPix()[i];
            point_world_tmp(seq(0, 1)) = point_pix; // homogenous pixel coordinates
            point_world_tmp = correspondence.getDepths()[i] * intrinsics_inv * point_world_tmp; // camera coordinates
            Sophus::SE3d pose = correspondence.getPoses()[i];
            point_world_tmp = (pose.matrix() * point_world_tmp.homogeneous())(seq(0,
                                                                                  2));    // This is done with gt data. And by plotting this one can tell that the gt rotations are R_{B->W}, probably
            point_world = point_world + point_world_tmp;
            count++;

        }
        point_world = point_world / count;

//        cout << "Result: " << point_world.transpose() << endl << endl;
        // Perturb, add to map, save ground truth data
        Vector3d pointPert = Vector3d::Random(3) * MAP_PERTURBATION_RATIO;
        gtWorldMap.push_back(point_world);
        point_world += pointPert;
        initialWorldMap.push_back(point_world);

    }

    //-------------------------------------------------- CERES OPTIMIZATION
    ceres::Problem problem;// Optimization variables, poses, map and intrinsics

    auto poses(initialPoses);
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;

    for (auto &pose: poses) {
        problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, local_parametrization_se3);
    }

    auto worldMap(initialWorldMap);
    for (auto &pw: worldMap) {
        problem.AddParameterBlock(pw.data(), 3);    // Note, the parameter block must always be an array of doubles!
    }

    Vector4d intr(fx, fy, cx, cy);
    Vector4d initial_intr = intr;
    problem.AddParameterBlock(intr.data(), 4);

    // Define two loss functions (for reprojection error and depth prior)
    double hub_p_repr = 1e-2;
    ceres::LossFunctionWrapper *loss_function_repr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(hub_p_repr),
                                                                                    ceres::TAKE_OWNERSHIP);
    double hub_p_unpr = 1e-2;
    ceres::LossFunctionWrapper *loss_function_unpr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(hub_p_unpr),
                                                                                    ceres::TAKE_OWNERSHIP);
    double weight_unpr = 5; // weight for unprojection constraint
    // Remember, we have associated a world map index (correspondence index -1) and a pose index (more complicated) to every correspondence
//    cout << "Reprojection errors before optimization: " << endl;
    for (auto &correspondence: correspondences) {
        for (int l = 0; l < correspondence.getPoses().size(); l++) {
            problem.AddResidualBlock(
                    ReprojectionConstraint::create_cost_function(correspondence.getObsPix()[l]),
                    loss_function_unpr,//new ceres::HuberLoss(1.0)
                    poses[correspondence.getPoseIndices()[l]].data(),
                    worldMap[correspondence.getIndex() - 1].data(),
                    intr.data());

//            // Write reprojection error
//            auto pw = worldMap[correspondence.getIndex() - 1];
//            auto pos = poses[correspondence.getPoseIndices()[l]];
//            Vector3d p_c = (pos.matrix().inverse() * pw.homogeneous())(seq(0, 2));
//            Vector2d p_p = (intrinsics * (p_c / p_c.z()))(seq(0, 1));
//            cout << (p_p - correspondence.getObsPix()[l]).norm() << endl;

            problem.AddResidualBlock(
                    DepthPrior::create_cost_function(correspondence.getObsPix()[l],
                                                     correspondence.getDepths()[l], weight_unpr),
                    loss_function_unpr,
                    poses[correspondence.getPoseIndices()[l]].data(),
                    worldMap[correspondence.getIndex() - 1].data(),
                    intr.data());
        }
    }

    // Compare with ground truth
    vector<double> pose_errors;
    vector<double> map_errors;
//    cout << "Pose errors before optimization: " << endl;
//    for (int i = 0; i < gtPoses.size(); i++) {
//        auto p_err = ((gtPoses[i].inverse() * poses[i]).matrix() - Matrix4d::Identity()).norm();
//        pose_errors.push_back(p_err);
//        cout << p_err << endl;
//    }
//    cout << "Map errors before optimization: " << endl;
//    for (int i = 0; i < gtWorldMap.size(); i++) {
//        auto m_err = (worldMap[i] - gtWorldMap[i]).norm();
//        map_errors.push_back(m_err);
//        cout << m_err << endl;
//    }

    // Optimize for the intrinsics -> we must add a (small) prior
    double weight_intr = 1e-4;
    problem.AddResidualBlock(IntrinsicsPrior::create_cost_function(initial_intr, weight_intr),
                             nullptr,
                             intr.data());

    // Constrain the problem
    problem.SetParameterBlockConstant(poses[0].data()); // any pose, kept constant, will do
//    problem.SetParameterBlockConstant(intr.data()); // any pose, kept constant, will do

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

//    cout << "Reprojection errors after optimization: " << endl;
//    for (auto &correspondence: correspondences) {
//        for (int l = 0; l < correspondence.getPoses().size(); l++) {
//            auto pw = worldMap[correspondence.getIndex() - 1];
//            auto pos = poses[correspondence.getPoseIndices()[l]];
//            Vector3d p_c = (pos.matrix().inverse() * pw.homogeneous())(seq(0, 2));
//            Vector2d p_p = (intrinsics * (p_c / p_c.z()))(seq(0, 1));
//            cout << (p_p - correspondence.getObsPix()[l]).transpose() << endl;
////            cout << (p_p - correspondence.getObsPix()[l]).norm() << " | " << pw.transpose() << endl;
//        }
//    }

    cout << endl << "Pose errors after optimization: " << endl;
    for (int i = 1; i < gtPoses.size(); i++) {
        Sophus::SE3d p_err((gtPoses[i].inverse() * poses[i]).matrix());
        cout << " Angle errors: " << p_err.angleX() / M_PI * 180 << ", " << p_err.angleY() / M_PI * 180 << ", "
             << p_err.angleZ() / M_PI * 180 << endl;
        cout << " Translational % error: " << p_err.translation().norm() / gtPoses[i].translation().norm() << endl
             << endl;

    }
//
//    cout << "Map errors after optimization: " << endl;
//    for (int i = 0; i < gtWorldMap.size(); i++) {
//        auto m_err = (worldMap[i] - gtWorldMap[i]).norm();
//        map_errors.push_back(m_err);
//        cout << i + 1 << endl << m_err << " | " << initialWorldMap[i].transpose() << endl << endl;
//    }
//    cout << "New intrinsics: " << endl;
//    for (int i = 0; i < 4; i++) {
//        cout << intr.data()[i] << endl;
//    }

}