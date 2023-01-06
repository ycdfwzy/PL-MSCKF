#ifndef MSCKF_MONO_SENSOR_TYPES_H_
#define MSCKF_MONO_SENSOR_TYPES_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>

namespace line_detector {
template <typename _Scalar>
  class PluckerLine
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    inline void normalize() {
      _Scalar t = std::sqrt(v.squaredNorm() + n.squaredNorm());
      n /= t;
      v /= t;
    }

    PluckerLine()
    {
      n << 1, 1, 1;
      v << 0, 0, 0;
    }
    // construct from plucker coordinate
    PluckerLine(const Eigen::Matrix<_Scalar, 3, 1> &n, const Eigen::Matrix<_Scalar, 3, 1> &v) : n(n), v(v) {}
    // construct from tow homogeneous points
    PluckerLine(const Eigen::Matrix<_Scalar, 4, 1> &p, const Eigen::Matrix<_Scalar, 4, 1> &q)
    {
      Eigen::Matrix<_Scalar, 4, 4> L = q * p.transpose() - p * p.transpose();
      v = L.block<3, 1>(0, 3);
      n << L(2, 1), L(0, 2), L(1, 0);
    }
    // construct from tow homogeneous points
    PluckerLine(_Scalar x1, _Scalar y1, _Scalar z1, _Scalar w1,
                _Scalar x2, _Scalar y2, _Scalar z2, _Scalar w2)
    {
      Eigen::Matrix<_Scalar, 4, 1> p(x1, y1, z1, w1);
      Eigen::Matrix<_Scalar, 4, 1> q(x2, y2, z2, w2);
      Eigen::Matrix<_Scalar, 4, 4> L = q * p.transpose() - p * p.transpose();
      v = L.block<3, 1>(0, 3);
      n << L(2, 1), L(0, 2), L(1, 0);
    }
    // construct from orthogonal expression
    PluckerLine(const Eigen::Matrix<_Scalar, 3, 3> &U, const Eigen::Matrix<_Scalar, 2, 2> &W)
    {
      n = W(0, 0) * U.template block<3, 1>(0, 0);
      v = W(1, 0) * U.template block<3, 1>(0, 1);
    }
    PluckerLine(const PluckerLine<_Scalar> &c)
    {
      this->n = c.N();
      this->v = c.V();
      // this->normalize();
    }
    ~PluckerLine() {}

    inline PluckerLine &operator=(const PluckerLine<_Scalar> &c)
    {
      this->n = c.N();
      this->v = c.V();
      // this->normalize();
      return *this;
    }

    inline Eigen::Matrix<_Scalar, 3, 1> N() const { return n; }
    inline Eigen::Matrix<_Scalar, 3, 1> V() const { return v; }

    bool reset(const Eigen::Matrix<_Scalar, 3, 1> &n, const Eigen::Matrix<_Scalar, 3, 1> &v)
    {
      // todo remove comments
      // if (n.dot(v) == 0)
      // {
        this->n = n;
        this->v = v;
        return true;
      // }
      // return false;
    }

    // Orthogonal Expression of Plucker line
    bool OrthogonalExpression(Eigen::Matrix<_Scalar, 3, 3> &U, Eigen::Matrix<_Scalar, 2, 2> &W) const
    {
      // if (!this->validate()) return false;
      U.template block<3, 1>(0, 0) = n / n.norm();
      U.template block<3, 1>(0, 1) = v / v.norm();
      Eigen::Matrix<_Scalar, 3, 1> c = n.cross(v);
      U.template block<3, 1>(0, 2) = c / c.norm();
      // _Scalar theta = std::atan2(v.norm(), n.norm());
      // W << std::cos(theta), -std::sin(theta),
      //      std::sin(theta),  std::cos(theta);
      W << n.norm(), -v.norm(),
           v.norm(), n.norm();
      W = W / std::sqrt(v.squaredNorm() + n.squaredNorm());
      return true;
    }

    // coordinates transformation
    bool transform(const Eigen::Matrix<_Scalar, 3, 3> &R,
                  const Eigen::Matrix<_Scalar, 3, 1> &T,
                  PluckerLine<_Scalar> &res) const
    {
      // if (!this->validate()) return false;
      // [T]_X
      Eigen::Matrix<_Scalar, 3, 3> TX;
      TX << 0, -T(2), T(1),
            T(2), 0, -T(0),
            -T(1), T(0), 0;
      Eigen::Matrix<_Scalar, 3, 1> res_n = R * this->n + TX * R * this->v;
      Eigen::Matrix<_Scalar, 3, 1> res_v = R * this->v;
      // res.reset(res_n, res_v);
      res.n = res_n;
      res.v = res_v;
      return true;
    }

    PluckerLine<_Scalar> update(const Eigen::Matrix<_Scalar, 3, 1> &psi, const _Scalar theta) const {
      Eigen::AngleAxis<_Scalar> angleAxis(psi.norm(), psi.normalized());
      Eigen::Matrix<_Scalar, 3, 3> U;
      Eigen::Matrix<_Scalar, 2, 2> W, dW;
      this->OrthogonalExpression(U, W);
      U = U * angleAxis.toRotationMatrix();
      dW << std::cos(theta), -std::sin(theta),
            std::sin(theta),  std::cos(theta);
      W = W * dW;
      return PluckerLine<_Scalar>(U, W);
    }

    PluckerLine<_Scalar> update(const Eigen::Matrix<_Scalar, 4, 1> &delta) const  {
      Eigen::Matrix<_Scalar, 3, 1> psi(delta(0), delta(1), delta(2));
      _Scalar theta = delta(3);
      return update(psi, theta);
    }

    bool validate() const
    {
      return n.dot(v) == 0;
    }

  private:
    Eigen::Matrix<_Scalar, 3, 1> n, v;
  };
} // namespace line_detector

namespace msckf_mono {
  template <typename _Scalar>
    using Quaternion = Eigen::Quaternion<_Scalar>;

  template <typename _Scalar>
    using Matrix2 = Eigen::Matrix<_Scalar, 2, 2>;

  template <typename _Scalar>
    using Matrix3 = Eigen::Matrix<_Scalar, 3, 3>;

  template <typename _Scalar>
    using Matrix4 = Eigen::Matrix<_Scalar, 4, 4>;

  template <typename _Scalar>
    using MatrixX = Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  template <typename _Scalar>
    using RowVector3 = Eigen::Matrix<_Scalar, 1, 3>;

  template <typename _Scalar>
    using Vector2 = Eigen::Matrix<_Scalar, 2, 1>;

  template <typename _Scalar>
    using Vector3 = Eigen::Matrix<_Scalar, 3, 1>;

  template <typename _Scalar>
    using Vector4 = Eigen::Matrix<_Scalar, 4, 1>;

  template <typename _Scalar>
    using VectorX = Eigen::Matrix<_Scalar, Eigen::Dynamic, 1>;

  template <typename _Scalar>
    using Point = Vector3<_Scalar>;

  template <typename _Scalar>
    using GyroscopeReading = Vector3<_Scalar>;

  template <typename _Scalar>
    using AccelerometerReading = Vector3<_Scalar>;

  template <typename _Scalar>
    using Isometry3 = Eigen::Transform<_Scalar,3,Eigen::Isometry>;

  template <typename _Scalar>
    struct lineMeasurement
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vector2<_Scalar> start, end;
    Vector3<_Scalar> toCartesianCoordinates() const
    {
      Vector3<_Scalar> l;
      Vector3<_Scalar> s(start(0), start(1), 1);
      Vector3<_Scalar> e(end(0), end(1), 1);
      l = s.cross(e);
      // l << start(1) - end(1),
      //     end(0) - start(0),
      //     start(0) * end(1) - start(1) * end(0);
      return l;
    }
  };

  template <typename _Scalar>
    struct Camera {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        _Scalar c_u, c_v, f_u, f_v, b;

      Quaternion<_Scalar> q_CI;
      Point<_Scalar> p_C_I;
    };

  template <typename _Scalar>
    struct camState {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Point<_Scalar> p_C_G;
      Quaternion<_Scalar> q_CG;
      _Scalar time;
      int state_id;
      int last_correlated_id;
      std::vector<size_t> tracked_feature_ids;
      std::vector<size_t> tracked_line_feature_ids;
    };

  template <typename _Scalar>
    struct imuState {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

      Point<_Scalar> p_I_G, p_I_G_null;
      Vector3<_Scalar> v_I_G, b_g, b_a, g, v_I_G_null;
      Quaternion<_Scalar> q_IG, q_IG_null;
    };

  template <typename _Scalar>
    struct imuReading {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        GyroscopeReading<_Scalar> omega;
      AccelerometerReading<_Scalar> a;
      _Scalar dT;
    };

  template <typename _Scalar>
    struct noiseParams {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        _Scalar u_var_prime, v_var_prime;
        _Scalar r1_var_prime, r2_var_prime;
      Eigen::Matrix<_Scalar, 12, 12> Q_imu;
      Eigen::Matrix<_Scalar, 15, 15> initial_imu_covar;
    };

  template <typename _Scalar>
    struct MSCKFParams {
      _Scalar max_gn_line_cost_norm;
      _Scalar max_gn_cost_norm, min_rcond, translation_threshold;
      _Scalar redundancy_angle_thresh, redundancy_distance_thresh;
      int min_track_length, max_track_length, max_cam_states;
    };

  template <typename _Scalar>
    struct featureTrackToResidualize {
      size_t feature_id;
      std::vector<Vector2<_Scalar>,
        Eigen::aligned_allocator<Vector2<_Scalar>>> observations;

      std::vector<camState<_Scalar>> cam_states;
      std::vector<size_t> cam_state_indices;

      bool initialized;
      Vector3<_Scalar> p_f_G;

      featureTrackToResidualize() : initialized(false) {}
    };

  template <typename _Scalar>
    struct featureTrack {
      size_t feature_id;
      std::vector<Vector2<_Scalar>,
        Eigen::aligned_allocator<Vector2<_Scalar>>> observations;

      std::vector<size_t> cam_state_indices; // state_ids of cam states corresponding to observations

      bool initialized = false;
      Vector3<_Scalar> p_f_G;
    };

  template <typename _Scalar>
    struct lineFeatureTrack {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      
      size_t feature_id;
      std::vector<lineMeasurement<_Scalar>> observations;

      std::vector<size_t> cam_state_indices; // state_ids of cam states corresponding to observations

      bool initialized = false;
      line_detector::PluckerLine<_Scalar> p_f_G;
    };

    template <typename _Scalar>
    struct lineFeatureTrackToResidualize {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      size_t feature_id;
      std::vector<lineMeasurement<_Scalar>> observations;

      std::vector<camState<_Scalar>> cam_states;
      std::vector<size_t> cam_state_indices;

      bool initialized = false;
      line_detector::PluckerLine<_Scalar> p_f_G;
    };
}

#endif
