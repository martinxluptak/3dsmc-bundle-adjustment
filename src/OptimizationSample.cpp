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

double cutPngSuffix(const string &filename) {
    return stod(filename.substr(0, filename.size() - 4));
}


class Correspondence {
public:
//
//    Correspondence(int index, const vector<string> &rgbFile, const vector<string> &depthFile,
//                   const vector<Vector2d> &obsPix) : index(index), rgb_file(rgbFile), depth_file(depthFile),
//                                                     obs_pix(obsPix) {}

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
    vector<string> rgb_file, depth_file, gt_file; // TODO rename variable names AND getter/setters to plural
    vector<Vector2d> obs_pix; // observations in pixel coordinates
    vector<Sophus::SE3d> poses;
    vector<int> pose_indices;
public:
    Correspondence(int index, const vector<string> &rgbFile, const vector<string> &depthFile,
                   const vector<string> &gtFile, const vector<Vector2d> &obsPix, const vector<Sophus::SE3d> &poses,
                   const vector<int> &poseIndices) : index(index), rgb_file(rgbFile), depth_file(depthFile),
                                                     gt_file(gtFile), obs_pix(obsPix), poses(poses),
                                                     pose_indices(poseIndices) {}

    const vector<int> &getPoseIndices() const {
        return pose_indices;
    }

    void setPoseIndices(const vector<int> &poseIndices) {
        pose_indices = poseIndices;
    }
    // in an array of images = poses, to be later created
public:
    const vector<string> &getGtFile() const {
        return gt_file;
    }

    void setGtFile(const vector<string> &gtFile) {
        gt_file = gtFile;
    }

    const vector<Sophus::SE3d> &getPoses() const {
        return poses;
    }

    void setPoses(const vector<Sophus::SE3d> &poses) {
        Correspondence::poses = poses;
    }

};

class ReprojectionConstraint {
public:
    ReprojectionConstraint(const Matrix3d &intrinsics, const Vector2d &pPix) : intrinsics(intrinsics), p_pix(pPix) {}

    template<typename T>
    bool operator()(const T *const pose, const T *const x_w, T *residuals) const {   //constant array of type constant T
        // remember, pose is B->W

        // Create some nicer variables
        // The below test shows that .data() returns qxywz, txyz
//        cout << pose.unit_quaternion() << " " << pose.unit_quaternion().w() << endl;
//        cout << pose.data()[0] << " " << pose.data()[1] << " " << pose.data()[2] << " " << pose.data()[3] << endl
//             << endl;
//        cout << pose[0] << " "<< pose[1] << " "<< pose[2] << " "<< pose[3] << " "<< pose[4] << " "<< pose[5] << " "<< pose[6] <<  endl;
//        cout << x_w[0] << " " << x_w[1] << " " << x_w[2] << endl;
//        cout << p_pix.transpose() << endl;
//        cout << endl;
//        // Testing rotation.h, not needed anymore
//        T qi[] = {pose[3], -pose[0], -pose[1], -pose[2]};   // rotation of the inverse transformation
//        T t[3] = {pose[4], pose[5], pose[6]}; // minus translation of the original pose
//        T ti[3];    // translation of the inverse transformation, to be filled in
//        UnitQuaternionRotatePoint(qi, mt, ti);

        // Autodiff way
        const Quaternion<T> q(pose[3], pose[0], pose[1],
                              pose[2]);  // pose0123 = quaternionxyzw, and quaterniond requires wxyz input
        const Vector<T, 3> t(pose[4], pose[5], pose[6]);
        const Vector<T, 3> p_W(x_w[0], x_w[1], x_w[2]);

        const Vector<T, 3> p_C = q.inverse().matrix() * (p_W - t);
        const Vector<T, 2> p_pix_est = (intrinsics.cast<T>() * p_C / p_C.z())(seq(0, 1));

//        // Numeric way
//        Quaterniond q(pose[3], pose[0], pose[1],
//                      pose[2]);  // pose0123 = quaternionxyzw, and quaterniond requires wxyz input
//        Sophus::SE3d::Point t(pose[4], pose[5], pose[6]);
//        Sophus::SE3d pose_CW(q, t);
//        Vector3d p_w(x_w[0], x_w[1], x_w[2]);
//        // Now, rotate from world to camera
//        Vector3d p_C = (pose_CW.matrix().inverse() * p_w.homogeneous())(seq(0, 2));
//        // Project and use intrinsics
//        Vector2d p_pix_est = (intrinsics * p_C / p_C.z())(seq(0, 1));

        // Reprojection error
        residuals[0] = T(p_pix_est[0]) - T(p_pix[0]);
        residuals[1] = T(p_pix_est[1]) - T(p_pix[1]);

        return true;
    }

//    // Hide the implementation from the user
//    static ceres::CostFunction *create_cost_function(const Matrix3d &intrinsics, const Vector2d &pPix) {
//        return new ceres::NumericDiffCostFunction<ReprojectionConstraint, ceres::CENTRAL, 2, 7, 3>(
//                new ReprojectionConstraint(intrinsics, pPix));
//    }
//    static ceres::CostFunction * create_cost_function(const Matrix3d &intrinsics, const Vector2d &pPix){
//        return new ceres::NumericDiffCostFunction<ReprojectionConstraint, ceres::CENTRAL, 2, 7, 3>(
//                    new ReprojectionConstraint(intrinsics, pPix)
//                );
//    }
    static ceres::CostFunction *create_cost_function(const Matrix3d &intrinsics, const Vector2d &pPix) {
        return new ceres::AutoDiffCostFunction<ReprojectionConstraint, 2, 7, 3>(
                new ReprojectionConstraint(intrinsics, pPix)
        );
    }

private:
    Matrix3d intrinsics;
    Vector2d p_pix;
};

int main(int argc, char *argv[]) {

    //-------------------------------------------------- FILL THE CORRESPONDENCES ARRAY

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
    // Ignore file header
    string line;
    getline(infile, line);

    vector<string> rgb_files, depth_files, gt_files;
    vector<Vector2d> obs_pixs;
    vector<Sophus::SE3d> poses_v;
    int correspondence_id;
    string rgb_file_ts, depth_file_ts, gt_ts;
    double corr_x, corr_y, tx, ty, tz, qx, qy, qz, qw;
    vector<string> uniqueNeededImageNames;
    int ui = 0;
    vector<int> pose_indices; // Here I want to create a set, but with an order (so, a vector)
    while (infile >> correspondence_id >> rgb_file_ts >> depth_file_ts >> gt_ts >> corr_x >> corr_y >> tx >> ty >> tz
                  >> qx >> qy >> qz >> qw) {

        if (correspondence_id != current_correspondence_id) { // TODO does not read the last line
            correspondences.emplace_back(
                    current_correspondence_id,
                    rgb_files,
                    depth_files,
                    gt_files,
                    obs_pixs,
                    poses_v,
                    pose_indices
            );
            current_correspondence_id = correspondence_id;
            rgb_files.clear();
            depth_files.clear();
            gt_files.clear();
            obs_pixs.clear();
            poses_v.clear();
            pose_indices.clear();
        }


        auto iter_found = std::find(uniqueNeededImageNames.begin(), uniqueNeededImageNames.end(), rgb_file_ts);
        // If we've already seen this image
        if (iter_found != uniqueNeededImageNames.end()) {
            pose_indices.push_back(distance(uniqueNeededImageNames.begin(), iter_found));
        } else {
            if (uniqueNeededImageNames.begin() != uniqueNeededImageNames.end())
                ui++;
            pose_indices.push_back(ui);
            uniqueNeededImageNames.emplace_back(rgb_file_ts);
        }

        rgb_files.push_back(rgb_file_ts);
        depth_files.push_back(depth_file_ts);
        obs_pixs.emplace_back(corr_x, corr_y);
        gt_files.emplace_back(gt_ts);
        Quaterniond q(qw, qx, qy,
                      qz); // TODO: (former) hyper error, the constructor from a vector expects the vector xyzw
        Sophus::SE3d::Point tr(tx, ty, tz);
        poses_v.emplace_back(q, tr);
    }
    infile.close();

////     Print correspondences
//    for (auto &correspondence: correspondences) {
//        for (int l = 0; l < correspondence.getObsPix().size(); l++) {
//            cout << correspondence.getIndex()
//                 << " | " << correspondence.getObsPix()[l].transpose() << " | "
//                 << uniqueNeededImageNames[correspondence.getPoseIndices()[l]];
//            cout << endl;
//        }
//    }

    // get set of RGB image filenames in which correspondences are observed
    // get depth images too
//    set<string> neededImageNamesSet;
    set<string> neededDepthNamesSet;
    vector<string> neededDepthNames;
    for (auto &correspondence: correspondences) {
//        for (auto &imageName: correspondence.getRgbFile())
//            neededImageNamesSet.insert(imageName);
        neededDepthNamesSet.insert(correspondence.getDepthFile()[0]);
    }
    neededDepthNames.assign(neededDepthNamesSet.begin(), neededDepthNamesSet.end());
//    uniqueNeededImageNames.assign(neededImageNamesSet.begin(), neededImageNamesSet.end());

    vector<double> uniqueImageTimestamps;
    cout.precision(17);
    for (auto &imageName: uniqueNeededImageNames) {   // Note, uniqueImageTimestamps has the same ordering of uniqueNeededImageNames
        uniqueImageTimestamps.push_back(stod(imageName)); // TODO is this precise enough? 1e+9 + 5 decimal places
    }

    //-------------------------------------------------- READ GROUND TRUTH

    infile.open(GROUND_TRUTH);
    double timestamp;
    vector<double> timestamps, txs, tys, tzs, qxs, qys, qzs, qws;
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

    vector<double> interpolatedTimestamps;
    vector<int> timestampIndices;
    tie(interpolatedTimestamps, timestampIndices) =
            nearest_interp_1d(timestamps, timestamps, uniqueImageTimestamps);

    //-------------------------------------------------- PERTURB GROUND TRUTH DATA FOR INITIALIZATION

    vector<Sophus::SE3d> initialPoses;
    vector<Sophus::SE3d> gtPoses;

    double PERTURBATION_RATIO = .5e-1;
    for (int i = 0; i < interpolatedTimestamps.size(); i++) {
        auto index = timestampIndices[i];

        // Unperturbed quantities
        Vector3d translationPert = Vector3d::Random(3) * PERTURBATION_RATIO;
        Vector4d quatPert = Vector4d::Random(4) * PERTURBATION_RATIO; // perturbation
        Vector4d quat_eigen( // perturbed quaternion, correct like this!
                qxs[index],
                qys[index],
                qzs[index],
                qws[index]
        );
        Vector3d translation_eigen(txs[index], tys[index], tzs[index]);
        Quaterniond quat_gt(quat_eigen);
        Sophus::SE3d::Point translation_gt = translation_eigen;
        gtPoses.emplace_back(quat_gt, translation_gt);

        // Perturbation and creation
        if (i > 0) {  // not perturbing the first guy, it is kept fixed in the optimization
            translation_eigen += translationPert;
            quat_eigen += quatPert;
            quat_eigen /= quat_eigen.norm();
        }
        Sophus::SE3d::Point translation = translation_eigen;
        Quaterniond quat(quat_eigen);
        initialPoses.emplace_back(quat, translation);

        // Optional: add even more drift as the sequence progresses
//        PERTURBATION_RATIO+=PERTURBATION_RATIO/3;

    }

    // Note, there is now a correspondence between gtPoses and uniqueNeededImageNames
//    for(int i=0; i < uniqueNeededImageNames.size(); i++){
//        cout << uniqueNeededImageNames[i] << " " << interpolatedTimestamps[i] << endl;
//    }
    // Let's make this explicit
    tuple<vector<string>, vector<Sophus::SE3d>> gt_info(uniqueNeededImageNames, gtPoses);

    //-------------------------------------------------- INITIALIZE 3D POINTS FROM THE DEPTH FILES

    // Let's get just the needed depth files
    vector<Mat> neededDepthImages;
    for (auto &filename: neededDepthNames) {
        neededDepthImages.emplace_back(
                imread("../../Data/rgbd_dataset_freiburg1_xyz/depth/" + filename + ".png",
                       IMREAD_UNCHANGED)); // TODO: there seems to be a LITTLE difference with IMREAD_UNCHANGED
    }

    // We initialize a 3D point with the depth map of the first frame it is observed in (to reduce drift)
    vector<Vector3d> initialWorldMap;
    vector<Vector3d> gtWorldMap;
    double MAP_PERTURBATION_RATIO = 1e-1;
    double p_outlier = 0.0; // generate an outlier with p_outlier % probability (but not our job to eliminate outliers)
    double outlier_scaling = 5;   // increase perturbation by a factor of 10 (a lot!)
    int keep_constant = 15;
    int count = 0;
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
        cout << d << endl;

        // Reconstruct the 3D point (from the real ground truth, not the perturbed one)
        Vector3d point_world;
        point_world.setOnes();
        auto point_pix = correspondence.getObsPix()[0];
        point_world(seq(0, 1)) = point_pix; // homogenous pixel coordinates
        point_world = d * intrinsics_inv * point_world; // camera coordinates
        // Put into world coordinates
//        auto rgb_filename = correspondence.getRgbFile()[0];
//        auto it = find(get<0>(gt_info).begin(), get<0>(gt_info).end(), rgb_filename);
//        auto index = distance(get<0>(gt_info).begin(), it);
//        auto pose = get<1>(gt_info)[index]; // all of this is terrible and slow, make it better!
        auto pose = correspondence.getPoses()[0];
//        auto pose2 = get<1>(gt_info)[correspondence.getPoseIndices()[0]] // yes, the indices work

        point_world = (pose.matrix() * point_world.homogeneous())(seq(0,
                                                                      2));    // This is done with gt data. And by plotting this one can tell that the gt rotations are R_{B->W}, probably

        // Perturb, add to map, save ground truth data
        // Outliers
        int p = (rand() % 100) / 100;
        double scaling = 1;
        if (p < p_outlier)
            scaling *= outlier_scaling;
        if (count == keep_constant) {
            scaling = 0.0;
        }
        count++;
        Vector3d pointPert = Vector3d::Random(3) * scaling * MAP_PERTURBATION_RATIO;
        gtWorldMap.push_back(point_world);
        point_world += pointPert;
        initialWorldMap.push_back(point_world);

    }

    //-------------------------------------------------- CERES OPTIMIZATION
    ceres::Problem problem;// Optimization variables, poses and map
    auto poses(initialPoses);
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;
    for (auto &pose: poses) {
//        Vector4d quat_test(pose.data()[0],pose.data()[1], pose.data()[2], pose.data()[3]);
//        cout << quat_test.norm() << endl;
        problem.AddParameterBlock(pose.data(), Sophus::SE3d::num_parameters, local_parametrization_se3);
    }
    auto worldMap(initialWorldMap);
    for (auto &pw: worldMap) {
        problem.AddParameterBlock(pw.data(), 3);    // Note, the parameter block must always be an array of doubles!
    }

    // Remember, we have associated a world map index (correspondence index -1) and a pose index (more complicated) to every correspondence
    cout << "Reprojection errors before optimization: " << endl;
    for (auto &correspondence: correspondences) {
        for (int l = 0; l < correspondence.getPoses().size(); l++) {
            problem.AddResidualBlock(
                    ReprojectionConstraint::create_cost_function(intrinsics, correspondence.getObsPix()[l]),
                    new ceres::HuberLoss(1.0),
                    poses[correspondence.getPoseIndices()[l]].data(),
                    worldMap[correspondence.getIndex() - 1].data());
            auto pw = worldMap[correspondence.getIndex() - 1];
            auto pos = poses[correspondence.getPoseIndices()[l]];
            Vector3d p_c = (pos.matrix().inverse() * pw.homogeneous())(seq(0, 2));
            Vector2d p_p = (intrinsics * (p_c / p_c.z()))(seq(0, 1));
            cout << (p_p - correspondence.getObsPix()[l]).norm() << endl;
//            cout << (p_p - correspondence.getObsPix()[l]).norm() << " | " << pw.transpose() << endl;
        }
    }

    // Compare with ground truth
    vector<double> pose_errors;
    vector<double> map_errors;
    cout << "Pose errors before optimization: " << endl;
    for (int i = 0; i < gtPoses.size(); i++) {
        auto p_err = ((gtPoses[i].inverse() * poses[i]).matrix() - Matrix4d::Identity()).norm();
        pose_errors.push_back(p_err);
        cout << p_err << endl;
    }
    cout << "Map errors before optimization: " << endl;
    for (int i = 0; i < gtWorldMap.size(); i++) {
        auto m_err = (worldMap[i] - gtWorldMap[i]).norm();
        map_errors.push_back(m_err);
        cout << m_err << endl;
    }

    // Constrain the problem
    problem.SetParameterBlockConstant(poses[0].data()); // any pose, kept constant, will do
    problem.SetParameterBlockConstant(worldMap[keep_constant].data()); // any pose, kept constant, will do

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.BriefReport() << "\n";

    cout << "Reprojection errors after optimization: " << endl;
    for (auto &correspondence: correspondences) {
        for (int l = 0; l < correspondence.getPoses().size(); l++) {
            auto pw = worldMap[correspondence.getIndex() - 1];
            auto pos = poses[correspondence.getPoseIndices()[l]];
            Vector3d p_c = (pos.matrix().inverse() * pw.homogeneous())(seq(0, 2));
            Vector2d p_p = (intrinsics * (p_c / p_c.z()))(seq(0, 1));
            cout << (p_p - correspondence.getObsPix()[l]).norm() << endl;
//            cout << (p_p - correspondence.getObsPix()[l]).norm() << " | " << pw.transpose() << endl;
        }
    }
    cout << "Pose errors after optimization: " << endl;
    for (int i = 0; i < gtPoses.size(); i++) {
        auto p_err = ((gtPoses[i].inverse() * poses[i]).matrix() - Matrix4d::Identity()).norm();
        pose_errors.push_back(p_err);
        cout << p_err << endl;
    }
    cout << "Map errors after optimization: " << endl;
    for (int i = 0; i < gtWorldMap.size(); i++) {
        auto m_err = (worldMap[i] - gtWorldMap[i]).norm();
        map_errors.push_back(m_err);
        cout << m_err << " | " << worldMap[i].transpose() << endl;
    }

    // Notes
    // 1) autodiff runs twice as fast
    // 2) perturbations above a certain treshold make optimization not converge
    // 3) adding large outliers make optimization not converge
    // 4) not even using a Huber loss helps
    // 5) pose errors are averaged out and slightly reduced
    // 6) map errors increase
    // 7) reprojection errors decrease drastically
    // 8) -> scale ambiguity, fix a pose and a map point
    // 9) now pose errors are much lower (they were .2), and they increase as time increases, to be expected?).
    //    Map points are better apart from 2 large outliers (?)
    //    Pose errors after optimization:
    //    0
    //    0.01791161869422029
    //    0.02346511678199089
    //    0.026982826378678926
    //    0.0371614501770633
    //    Map errors after optimization:
    //    0
    //    0.024840395914591456
    //    0.093004283293719486
    //    0.049394236301493352
    //    0.028589798394583421
    //    0.15847544563464272
    //    0.10603251824330331
    //    0.066467253281677532
    //    0.66709619830411182
    //    0.042044582560245493
    //    0.10740345080902573
    //    1.9959865304447988
    //    0.30224908467118672
    //    0.088936535129609945
    //    0.010218351660408742
    //    0.15897312101451591
    //    3.9754683674080531
    //    0.45031045582085988
    //    0.079442148764633361
    //    Even worse if keep_constant = 15
    //    Pose errors after optimization:
    //    0
    //    0.016424707006456873
    //    0.021445867328087592
    //    0.027286074810309393
    //    0.038493247348242332
    //    Map errors after optimization:
    //    0.36424188460076729 | 0.72688446871620971 0.19295236199472934  1.0006107406876317
    //    0.33569179594020004 | 0.71440530112624556 0.31490037734909404 0.99747799127947689
    //    0.48244118065288505 | 0.57716932945071753 0.33319562182083079  1.0276424925006935
    //    0.46601562995688067 | 0.54154877377083344 0.18870233837013795 0.99988794689157789
    //    0.28437290672193072 | 0.7105047880788733 0.8536380459254993 1.0785024097277756
    //    0.14057464220234531 | 0.60902632781967347 0.93521191068384868 0.98669807038259405
    //    0.39058233524787267 | 0.87578975133244708 0.68689877605071703  1.0532952924015371
    //    0.1781773733856051 | 0.94331907799840509 0.73274669494646416 0.90236433119915171
    //    0.17869067391362314 | 0.45633098821654355  1.2758423232468488 0.63673247791319709
    //    0.25681347950645306 | 0.81659108865352481 0.47572158021871935  1.1078436921037733
    //    0.19110205583054327 |  0.8287411944078148 0.53238125176025353  1.0696471947114767
    //    1.8582732084451083 |  -3.3981902238558779 -0.62392379344118509 0.058915575756753505
    //    0.96312088997284584 |  -0.23246423301959715 -0.010677030598000382   0.91452416890146704
    //    0.26801101187541093 | 0.34001387139197792 0.84964785580594815  1.1966751768976516
    //    0.31854717065106314 | 0.37591155067784454  0.9185350185405432  1.2152490309104735
    //    0 | 0.76581549209968713  1.0625169735356792 0.76779446823853792
    //    1480.8199040598579 | -1083.4685579068214  115.10309180664919 -1001.4343639708552
    //    0.1188009283575841 |  0.3526032414753616  1.2036970603269825 0.68535566890512967
    //    0.45796098845305783 | 0.64050595899442486 0.56306030004999741  1.0461500399329871

}