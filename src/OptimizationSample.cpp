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

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace cv::xfeatures2d;

double cutPngSuffix(const string &filename) {
    return stod(filename.substr(0, filename.size() - 4));
}


class Correspondence {
public:

    Correspondence(int index, const vector<string> &rgbFile, const vector<string> &depthFile,
                   const vector<Vector2d> &obsPix) : index(index), rgb_file(rgbFile), depth_file(depthFile),
                                                     obs_pix(obsPix) {}

    int getIndex() const {
        return index;
    }

    void setIndex(int index) {
        Correspondence::index = index;
    }

    const vector<string> &getRgbFile() const {
        return rgb_file;
    }

    void setRgbFile(const vector<string> &rgbFile) {
        rgb_file = rgbFile;
    }

    const vector<string> &getDepthFile() const {
        return depth_file;
    }

    void setDepthFile(const vector<string> &depthFile) {
        depth_file = depthFile;
    }

    const vector<Vector2d> &getObsPix() const {
        return obs_pix;
    }

    void setObsPix(const vector<Vector2d> &obsPix) {
        obs_pix = obsPix;
    }

private:
    int index;
    vector<string> rgb_file, depth_file; // TODO rename variable names AND getter/setters to plural
    vector<Vector2d> obs_pix; // observations in pixel coordinates

};

//class gtPose{
//public:
//    gtPose(const Sophus::SE3d &pose, const string &rgbFile) : pose(pose), rgb_file(rgbFile) {}
//
//    const Sophus::SE3d &getPose() const {
//        return pose;
//    }
//
//    const string &getRgbFile() const {
//        return rgb_file;
//    }
//
//private:
//    Sophus::SE3d pose;
//    string rgb_file;
//};

/**
 * Optimization constraints.
 * TODO: this code copied from Exercise5. Change code into a constraint that can be used for optimization.
 */
class SparseTrackingConstraint {
public:
    SparseTrackingConstraint(const Vector3f &sourcePoint, const Vector3f &targetPoint, const float weight) :
            m_sourcePoint{sourcePoint},
            m_targetPoint{targetPoint},
            m_weight{weight} {}

    template<typename T>
    bool operator()(const T *const pose, T *residuals) const {
//
//        PoseIncrement<T> poseIncrement = PoseIncrement<T>(const_cast<T* const>(pose));
//
//        // The resulting 3D residual should be stored in the residuals array. To apply the pose
//        // increment (pose parameters) to the source point, you can use the PoseIncrement class.
//        // Important: Ceres automatically squares the cost function.
//        T source_T[3];
//        fillVector(m_sourcePoint,source_T);
//        T targ_T[3];
//        fillVector(m_targetPoint, targ_T);
//        T trans_T[3];
//        poseIncrement.apply(source_T,trans_T);
//
//        // Also squaring the weights ?
//        for (int i = 0; i < 3; i++) {
//            residuals[i] = T(m_weight)* (trans_T[i] - targ_T[i]);
//        }
//
        return true;
    }

    static ceres::CostFunction *create(const Vector3f &sourcePoint, const Vector3f &targetPoint, const float weight) {
        return new ceres::AutoDiffCostFunction<SparseTrackingConstraint, 3, 6>(
                new SparseTrackingConstraint(sourcePoint, targetPoint, weight)
        );
    }

protected:
    const Vector3f m_sourcePoint;
    const Vector3f m_targetPoint;
    const float m_weight;
    const float LAMBDA = 0.1f;
};

int main(int argc, char *argv[]) {

    //-------------------------------------------------- FILL THE CORRESPONDENCES ARRAY

    int correspondence_id;
    string rgb_file, depth_file;
    double corr_x, corr_y;
    ifstream infile;
    const string DEFAULT_INTRINSICS = "../../Data/ros_default_intrinsics.txt";
    const string FREIBURG1_INTRINSICS = "../../Data/freiburg1_intrinsics.txt";
    const string GROUND_TRUTH = "../../Data/rgbd_dataset_freiburg1_xyz/groundtruth.txt";

//    cout << "Current working directory: " << filesystem::current_path() << endl;

    infile.open(DEFAULT_INTRINSICS);    // TODO: optimize over the intrinsics, too, as they are the ROS standard ones
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

    infile.open("../../Data/freiburg1_xyz_sample_five_frames.txt");
    std::vector<Correspondence> correspondences;
    int current_correspondence_id = 1;

    vector<string> rgb_files, depth_files;
    vector<Vector2d> obs_pixs;
    while (infile >> correspondence_id >> rgb_file >> depth_file >> corr_x >> corr_y) {
//        cout << correspondence_id << "," << rgb_file << "," << depth_file << "," << corr_x << "," << corr_y << endl;

        if (correspondence_id != current_correspondence_id) { // TODO does not read the last line
            correspondences.emplace_back(
                    current_correspondence_id,
                    rgb_files,
                    depth_files,
                    obs_pixs
            );
            current_correspondence_id = correspondence_id;
            rgb_files.clear();
            depth_files.clear();
            obs_pixs.clear();
        }
        rgb_files.push_back(rgb_file);
        depth_files.push_back(depth_file);
        obs_pixs.emplace_back(corr_x, corr_y);
    }
    infile.close();

    // Print correspondences
//    for (auto &correspondence: correspondences) {
//
//        cout << correspondence.getIndex() << " : ";
//        for (auto &obsPix: correspondence.getObsPix()) {
//            cout << obsPix;
//        }
//        cout << endl;
//    }

    // get set of RGB image filenames in which correspondences are observed
    // get depth images too
    set<string> neededImageNamesSet;
    vector<string> neededImageNames;
    set<string> neededDepthNamesSet;
    vector<string> neededDepthNames;
    for (auto &correspondence: correspondences) {
        for (auto &imageName: correspondence.getRgbFile())
            neededImageNamesSet.insert(imageName);
        neededDepthNamesSet.insert(correspondence.getDepthFile()[0]);
    }
    neededDepthNames.assign(neededDepthNamesSet.begin(), neededDepthNamesSet.end());
    neededImageNames.assign(neededImageNamesSet.begin(), neededImageNamesSet.end());

    vector<double> imageTimestamps;
    cout.precision(17);
    for (auto &imageName: neededImageNames) {   // Note, imageTimestamps has the same ordering of neededImageNames
        imageTimestamps.push_back(cutPngSuffix(imageName)); // TODO is this precise enough? 1e+9 + 5 decimal places
    }

    //-------------------------------------------------- READ GROUND TRUTH

    infile.open(GROUND_TRUTH);
    double timestamp, tx, ty, tz, qx, qy, qz, qw;
    vector<double> timestamps, txs, tys, tzs, qxs, qys, qzs, qws;
    // ignore file header.
    string line;
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

    vector<double> interpolatedTimestamps;
    vector<int> timestampIndices;
    tie(interpolatedTimestamps, timestampIndices) =
            nearest_interp_1d(timestamps, timestamps, imageTimestamps);

    //-------------------------------------------------- PERTURB GROUND TRUTH DATA FOR INITIALIZATION

    vector<Sophus::SE3d> initialPoses;
    vector<Sophus::SE3d> gtPoses;

    const double PERTURBATION_RATIO = 1e-2;
    for (int i = 0; i < interpolatedTimestamps.size(); i++) {
        auto index = timestampIndices[i];

        // Unperturbed quantities
        Vector3d translationPert = Vector3d::Random(3) * PERTURBATION_RATIO;
        Vector4d quatPert = Vector4d::Random(4) * PERTURBATION_RATIO; // perturbation
        Vector4d quat_eigen( // perturbed quaternion
                qws[index],
                qxs[index],
                qys[index],
                qzs[index]
        );
        Vector3d translation_eigen(txs[index], tys[index], tzs[index]);
        Quaterniond quat_gt(quat_eigen);
        Sophus::SE3d::Point translation_gt = translation_eigen;
        gtPoses.emplace_back(quat_gt, translation_gt);

        // Perturbation and creation
        translation_eigen += translationPert;
        Sophus::SE3d::Point translation = translation_eigen;
        quat_eigen += quatPert;
        quat_eigen /= quat_eigen.norm();
        Quaterniond quat(quat_eigen);
        initialPoses.emplace_back(quat, translation);
    }

    // Note, there is now a correspondence between gtPoses and neededImageNames
//    for(int i=0; i < neededImageNames.size(); i++){
//        cout << neededImageNames[i] << " " << interpolatedTimestamps[i] << endl;
//    }
    // Let's make this explicit
    tuple<vector<string>, vector<Sophus::SE3d>> gt_info(neededImageNames, gtPoses);

    //-------------------------------------------------- INITIALIZE 3D POINTS FROM THE DEPTH FILES

    // Let's get just the needed depth files
    vector<Mat> neededDepthImages;
    for (auto &filename: neededDepthNames) {
        neededDepthImages.emplace_back(
                imread("../../Data/rgbd_dataset_freiburg1_xyz/depth/" + filename, IMREAD_GRAYSCALE));
    }

    // We initialize a 3D point with the depth map of the first frame it is observed in (to reduce drift)
    vector<Vector3d> world_map;
    double MAP_PERTURBATION_RATIO = 1e-2;
    // TODO: this works because of the order in which the correspondences are stored in five_frames.txt. Inform other people about this
    for (auto &correspondence: correspondences) { // TODO we should run over the depth frames and not over correspondences for efficient image loading

        // Get the right depth image
        auto depth_filename = correspondence.getDepthFile()[0];
        auto itr = find(neededDepthNames.begin(), neededDepthNames.end(), depth_filename);
        auto ind = distance(neededDepthNames.begin(), itr);
        auto depthImage = neededDepthImages[ind];

        // TODO here we rely on the correctness of the ground truth poses to initializer 3D positions in the world frame.
        //      Implement an initialization procedure independent of the ground truth.

        // Get the depth
        int j = floor(correspondence.getObsPix()[0][0]);
        int i = floor(correspondence.getObsPix()[0][1]);
        double d = depthImage.at<uint16_t>(i, j) / 5000.0; // 16-bit depth, 1 channel

        // Reconstruct the 3D point (from the real ground truth, not the perturbed one)
        Vector3d point_world;
        point_world.setOnes();
        auto point_pix = correspondence.getObsPix()[0];
        point_world(seq(0, 1)) = point_pix; // homogenous pixel coordinates
        point_world = d * intrinsics_inv * point_world; // camera coordinates
        // Put into world coordinates
        auto rgb_filename = correspondence.getRgbFile()[0];
        auto it = find(get<0>(gt_info).begin(), get<0>(gt_info).end(), rgb_filename);
        auto index = distance(get<0>(gt_info).begin(), it);
        auto pose = get<1>(gt_info)[index]; // TODO: all of this is terrible and slow, make it better!
        point_world = (pose.matrix() * point_world.homogeneous())(seq(0, 2));    // This is done with gt data

        // Perturb, add to map
        Vector3d pointPert = Vector3d::Random(3) * MAP_PERTURBATION_RATIO;
        point_world += pointPert;
        world_map.push_back(point_world);

    }

    for (auto &pw: world_map) {
        cout << pw.transpose() << endl;
    }

    //-------------------------------------------------- CERES OPTIMIZATION

    ceres::Problem problem;
    auto poses(initialPoses);
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;

    for (auto &pose: poses) {
        problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, local_parametrization_se3);
    }

//    for (auto &obsPix: correspondence.getObsPix()) {
////            problem.AddResidualBlock(
////                    cost_func,
////                    nullptr, // L2
////                    poses[poseIndex].data()
////            )
//    }


}