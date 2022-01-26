//
// Created by lupta on 1/20/2022.
//

#include "OptimizationUtils.h"
#include <string>
#include "Eigen/Dense"
#include <fstream>
#include <utility>
#include <ceres/ceres.h>
#include "sophus/local_parameterization_se3.hpp"

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

        const Vector<T, 3> p_C = q.inverse().matrix() * (p_W - t);
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
    Matrix3d intrinsics;
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

void write_keyframe_poses_to_file(const string &file_path, const vector<KeyFrame> keyframes) {
    ofstream outfile(file_path);
    for (auto &keyframe: keyframes) {
        // TODO NOT IMPLEMENTED
    }
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

void countConstraints(const BundleAdjustmentConfig &cfg, const Map3D &map, const vector<KeyFrame> &keyframes,
                      int &reprojection_constraints_result, int &unprojection_constraints_result) {
    for (auto &it: map) {
        auto landmark = it.second;
        auto landmarkId = it.first;

        /*
         * Iterate over each observation of the same landmark.
         */
        for (auto &observation: landmark.observations) {
            auto keyframe_index = observation.first / cfg.KEYFRAME_INCREMENT;
            auto associated_keyframe = keyframes[keyframe_index];
            auto observed_pix = observation.second;
            Vector2d observed_pix_vec2d(observed_pix.x, observed_pix.y); // Conversion from Point2d to Vector2d

            reprojection_constraints_result++; // increment reprojection constraints counter.

            int local_index = findLocalPointIndex(associated_keyframe, landmarkId);
            if (local_index == -1) {
                cout << "Global point index not matching to any observation in current keyframe" << endl;
                continue;
            }

            auto local_depth = associated_keyframe.points3d_local[local_index][2];

            if (local_depth > BundleAdjustmentConfig::NEG_INF && local_depth < 0) {
                cout << local_depth << " is < 0" << endl;
                continue;
            }

            if (local_depth <= BundleAdjustmentConfig::NEG_INF) {
                cout << "Local depth is -inf" << endl;
                continue;
            }

            unprojection_constraints_result++; // increment unprojection constraints counter.
        }
    }
}


void runOptimization(const BundleAdjustmentConfig &cfg, Map3D &map, vector<KeyFrame> &keyframes,
                     const Vector4d &intrinsics_initial) {
    ////////////////////////////////
    //  OPTIMIZATION WITH CERES  //
    //////////////////////////////

    /*
     * Define ceres problem options.
     */
    ceres::Problem problem; // Optimization variables, poses, map and intrinsics_initial
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;

    /*
     * Count residuals before adding them.
     * This allows us to normalize the weight of each constraint by the number of constraints.
     * The following for-loops ONLY count the number of constraints and do not do anything
     * with regards to the optimization algorithm.
     */
    int reprojection_constraints_count = 0;
    int unprojection_constraints_count = 0;
    countConstraints(cfg, map, keyframes, reprojection_constraints_count, unprojection_constraints_count);

    /*
     * Add camera poses as parameter blocks.
     * (Saved in keyframes as T_w_c)
     */
    ceres::LocalParameterization *local_parametrization_se3 = new Sophus::LocalParameterizationSE3;
    for (auto &keyframe: keyframes) {
        problem.AddParameterBlock(keyframe.T_w_c.data(),
                                  Sophus::SE3d::num_parameters,
                                  local_parametrization_se3
        );
    }

    /*
     * Add world 3D points as parameter blocks.
     * Note: the parameter block must always be an array of doubles!
     */
    for (auto &it: map) {
        auto &landmark_3d_point = it.second.point;
        problem.AddParameterBlock(landmark_3d_point.data(), 3);
    }

    /*
     *  Add camera intrinsics as parameter block.
     *  Initialize intrinsics_optimized from intrinsics_initial
     *  Keep a copy of intrinsics_initial to compare the values after optimization.
     */
    auto intrinsics_optimized(intrinsics_initial);
    problem.AddParameterBlock(intrinsics_optimized.data(), 4);


    /*
     * Define loss function for reprojection error.
     */
    auto *loss_function_repr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(cfg.HUB_P_REPR),
                                                              ceres::TAKE_OWNERSHIP);
    auto *loss_function_unpr = new ceres::LossFunctionWrapper(new ceres::HuberLoss(cfg.HUB_P_UNPR),
                                                              ceres::TAKE_OWNERSHIP);

    /*
     * Iterate over map of landmarks
     */
    for (auto &it: map) {
        auto &landmarkId = it.first;
        auto &landmark = it.second;
        /*
         * Iterate over each observation of the same landmark.
         */
        for (auto &observation: landmark.observations) {
            auto keyframe_index = observation.first / cfg.KEYFRAME_INCREMENT;
            auto &associated_keyframe = keyframes[keyframe_index];
            auto &observed_pixel = observation.second;
            Vector2d observed_pixel_vec2d(observed_pixel.x, observed_pixel.y); // Conversion from Point2d to Vector2d

            /*
             *  Add residual block - reprojection constraint.
             */
            problem.AddResidualBlock(
                    ReprojectionConstraint::create_cost_function(
                            observed_pixel_vec2d,
                            1.0 / reprojection_constraints_count // divide weight by # of reprojection constraints
                    ),
                    loss_function_repr,
                    associated_keyframe.T_w_c.data(), // (global) camera pose during observation
                    landmark.point.data(), // 3D point
                    intrinsics_optimized.data()
            );

            int local_index = findLocalPointIndex(associated_keyframe, landmarkId);
            double local_depth = associated_keyframe.points3d_local[local_index][2]; // get Z coordinate from this vector.

            /*
             * Only add the depth constraint if local_depth has a valid value.
             * Valid values EXCLUDE:
             * - negative depth values
             * - depth set to negative infinity
             */
            if (local_depth <= BundleAdjustmentConfig::NEG_INF || local_depth < 0) {
                continue;
            }

            /*
             *  Add residual block - depth constraint.
             */
            problem.AddResidualBlock(
                    DepthPrior::create_cost_function(
                            observed_pixel_vec2d,
                            local_depth,
                            cfg.WEIGHT_UNPR / unprojection_constraints_count
                    ),  // divide by # of residuals for normalizing
                    loss_function_unpr,
                    associated_keyframe.T_w_c.data(),
                    landmark.point.data(),
                    intrinsics_optimized.data()
            );
        }
    }

    // Optimize for the intrinsics_initial -> we must add a (small) prior
    problem.AddResidualBlock(
            IntrinsicsPrior::create_cost_function(intrinsics_initial, cfg.WEIGHT_INTRINSICS),
            nullptr, // squared loss
            intrinsics_optimized.data()
    );

    // Constrain the problem
    problem.SetParameterBlockConstant(keyframes[0].T_w_c.data()); // any pose, kept constant, will do

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}
