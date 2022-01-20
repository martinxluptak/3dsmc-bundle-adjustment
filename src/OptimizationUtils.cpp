//
// Created by lupta on 1/20/2022.
//

#include <string>
#include "Eigen/Dense"
#include <fstream>
#include <ceres/ceres.h>

using namespace std;
using namespace Eigen;

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
