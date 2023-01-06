#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>
// #include <opencv2/line_descriptor.hpp>
#include <cv_bridge/cv_bridge.h>
// #include <opencv2/ximgproc/fast_line_detector.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <msckf_mono/types.h>
#include <msckf_mono/matrix_utils.h>

#include <opencv2/line_descriptor_individual.hpp>
#include <msckf_mono/line_feature/matching.h>
#include <msckf_mono/line_feature/config.h>

namespace line_detector {

// to remove distortion
template <typename _S>
class CameraModel {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CameraModel(const cv::Mat &K, const cv::Mat &distortion_coeffs) {
        K_ << K.at<_S>(0, 0), K.at<_S>(0, 1), K.at<_S>(0, 2),
            K.at<_S>(1, 0), K.at<_S>(1, 1), K.at<_S>(1, 2),
            K.at<_S>(2, 0), K.at<_S>(2, 1), K.at<_S>(2, 2);
        K_inv_ = K_.inverse();
        k1 = distortion_coeffs.at<_S>(0);
        k2 = distortion_coeffs.at<_S>(1);
        p1 = distortion_coeffs.at<_S>(2);
        p2 = distortion_coeffs.at<_S>(3);
    }

    /**
     * \brief Lifts a point from the image plane to its projective ray
     *
     * \param p image coordinates
     * \param P coordinates of the projective ray
     */
    void liftProjective(const Eigen::Matrix<_S, 2, 1> &p, Eigen::Matrix<_S, 3, 1> &P) const {
        Eigen::Matrix<_S, 3, 1> p_(p(0), p(1), 1);
        p_ = K_inv_ * p_;
        _S mx_d = p_(0), my_d = p_(1);

        // Recursive distortion model
        int n = 8;
        Eigen::Matrix<_S, 2, 1> d_u;
        this->distortion(Eigen::Matrix<_S, 2, 1>(mx_d, my_d), d_u);
        // Approximate value
        _S mx_u = mx_d - d_u(0);
        _S my_u = my_d - d_u(1);

        for (int i = 1; i < n; ++i) {
            this->distortion(Eigen::Matrix<_S, 2, 1>(mx_u, my_u), d_u);
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }

        P << mx_u, my_u, 1.0;
    }

private:
    Eigen::Matrix<_S, 3, 3> K_, K_inv_;
    _S k1, k2, p1, p2;

    void distortion(const Eigen::Matrix<_S, 2, 1> &p_u, Eigen::Matrix<_S, 2, 1> &d_u) const {
        _S mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

        mx2_u = p_u(0) * p_u(0);
        my2_u = p_u(1) * p_u(1);
        mxy_u = p_u(0) * p_u(1);
        rho2_u = mx2_u + my2_u;
        rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
               p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
    }
};

class LineDetector {
public:
    static void detect_lines(const cv::Mat &image, std::vector<cv::line_descriptor::KeyLine> &lines, cv::Mat &lines_descriptor);
    static void match_lines(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, std::vector<cv::DMatch> &matches);
    static void match_lines(const cv::Mat& queryDescriptors, const cv::Mat& trainDescriptors, std::vector<int> &matches);

    static bool judge_distance(const cv::line_descriptor::KeyLine &line1, const cv::line_descriptor::KeyLine &line2, float distance_threshold = 60.);
    static bool judge_angle(const cv::line_descriptor::KeyLine &line1, const cv::line_descriptor::KeyLine &line2, float angle_threshold = 15.);

    static void compute_descriptors(const cv::Mat &image, std::vector<cv::line_descriptor::KeyLine> &lines, cv::Mat &lines_descriptor);
};

class LineVisualizer {
public:
    LineVisualizer() { cur_time_stamp = 0; lines.clear(); }
    ~LineVisualizer() {}
    void update_old_features(const std::vector<cv::line_descriptor::KeyLine>&, const std::vector<int>&);
    void add_new_features(const std::vector<cv::line_descriptor::KeyLine>&, const std::vector<int>&);
    cv::Mat get_image(const cv::Mat& image);

private:
    int cur_time_stamp;
    std::unordered_map<int, std::pair<cv::line_descriptor::KeyLine, int>> lines;
};

class LineTrackHandler {
public:
    LineTrackHandler(const cv::Mat &K, const cv::Mat &distortion_coeffs, std::ofstream &debug_info, int COL, int ROW);
    ~LineTrackHandler() { log.close(); }

    void set_current_image(cv::Mat &image, double time);

    void process_current_image(bool drawlines = false);

    cv::Mat get_feature_image();

    inline void undistortPoint(std::vector<cv::Point2f> &in,
                               std::vector<cv::Point2f> &out) const
    {
        cv::undistortPoints(in, out, K_, distortion_coeffs_);
    }

    template <typename _S>
    inline void undistortLine(const cv::Point2f &start, const cv::Point2f &end,
                              msckf_mono::lineMeasurement<_S> &lineMeas,
                              const CameraModel<_S> &camera) const
    {
        Eigen::Matrix<_S, 2, 1> start_(start.x, start.y);
        Eigen::Matrix<_S, 2, 1> end_(end.x, end.y);
        Eigen::Matrix<_S, 3, 1> tmp;
        camera.liftProjective(start_, tmp);
        lineMeas.start << tmp(0) / tmp(2), tmp(1) / tmp(2);
        camera.liftProjective(end_, tmp);
        lineMeas.end << tmp(0) / tmp(2), tmp(1) / tmp(2);
        // lineMeas.start << start.x, start.y;
        // lineMeas.end << end.x, end.y;
    }

    template <typename _S>
    void get(std::vector<msckf_mono::lineMeasurement<_S>> &cur_lines_,
             std::vector<size_t> &cur_line_ids_,
             std::vector<msckf_mono::lineMeasurement<_S>> &new_lines_,
             std::vector<size_t> &new_line_ids_) const
    {
        CameraModel<_S> camera(this->K_, this->distortion_coeffs_);
        cur_lines_.resize(this->cur_lines_.size());
        cur_line_ids_.resize(this->cur_lines_.size());

        // debug_info << "tracked lines: " << this->cur_lines_.size() << std::endl;
        for (int i = 0; i < this->cur_lines_.size(); i++) {
            // debug_info << "start distortion: " << this->cur_lines_[i].startPointX << " " << this->cur_lines_[i].startPointY << std::endl;
            // debug_info << "end distortion: " << this->cur_lines_[i].endPointX << " " << this->cur_lines_[i].endPointY << std::endl;

            // std::vector<cv::Point2f> in, out;
            // in.clear(); out.clear();
            // in.push_back(this->cur_lines_[i].getStartPoint());
            // in.push_back(this->cur_lines_[i].getEndPoint());
            // this->undistortPoint(in, out);
            // debug_info << "start undistortion by opencv: " <<  out[0].x << " " << out[0].y << std::endl;
            // debug_info << "end undistortion by opencv: " <<  out[1].x << " " << out[1].y << std::endl;

            this->undistortLine(this->cur_lines_[i].getStartPoint(),
                                this->cur_lines_[i].getEndPoint(),
                                cur_lines_[i], camera);

            // debug_info << "start undistortion by custom: " << cur_lines_[i].start(0) << " " << cur_lines_[i].start(1) << std::endl;
            // debug_info << "end undistortion by custom: " << cur_lines_[i].end(0) << " " << cur_lines_[i].end(1) << std::endl;

            cur_line_ids_[i] = this->cur_line_ids_[i];
        }

        new_lines_.resize(this->new_lines_.size());
        new_line_ids_.resize(this->new_lines_.size());
        // debug_info << "New lines: " << this->new_lines_.size() << std::endl;

        for (int i = 0; i < this->new_lines_.size(); i++) {
            // debug_info << "start distortion: " << this->new_lines_[i].startPointX << " " << this->new_lines_[i].startPointY << std::endl;
            // debug_info << "end distortion: " << this->new_lines_[i].endPointX << " " << this->new_lines_[i].endPointY << std::endl;

            // std::vector<cv::Point2f> in, out;
            // in.clear(); out.clear();
            // in.push_back(this->new_lines_[i].getStartPoint());
            // in.push_back(this->new_lines_[i].getEndPoint());
            // this->undistortPoint(in, out);
            // new_lines_[i].start << out[0].x, out[0].y;
            // new_lines_[i].end << out[1].x, out[1].y;
            // debug_info << "start undistortion by opencv: " << out[0].x << " " << out[0].y << std::endl;
            // debug_info << "end undistortion by opencv: " << out[1].x << " " << out[1].y << std::endl;

            this->undistortLine(this->new_lines_[i].getStartPoint(),
                                this->new_lines_[i].getEndPoint(),
                                new_lines_[i], camera);

            // debug_info << "start undistortion by custom: " << new_lines_[i].start(0) << " " << new_lines_[i].start(1) << std::endl;
            // debug_info << "end undistortion by custom: " << new_lines_[i].end(0) << " " << new_lines_[i].end(1) << std::endl;

            new_line_ids_[i] = this->new_line_ids_[i];
        }
    }

private:
    bool inBorder(const cv::line_descriptor::KeyLine&) const;

    void drawTrackedLineFeatures(const cv::line_descriptor::KeyLine&, int) const;

    cv::Mat K_, distortion_coeffs_;

    cv::Mat cur_image_;
    double cur_time_;
    std::vector<cv::line_descriptor::KeyLine> cur_lines_;
    std::vector<int> cur_line_ids_;
    cv::Mat cur_line_descs_;

    cv::Mat prev_image_;
    double prev_time_;
    std::vector<cv::line_descriptor::KeyLine> prev_lines_;
    std::vector<int> prev_line_ids_;
    cv::Mat prev_line_descs_;

    std::vector<cv::line_descriptor::KeyLine> new_lines_;
    std::vector<int> new_line_ids_;

    int line_cnt;       // count lines
    bool initialized;   // if has first image

    LineVisualizer visualizer;

    std::ofstream log;

    std::ofstream &debug_info;

    int ROW, COL; // row: image height
                  // col: image width
};

} // namespace line_detector

namespace cv {
namespace line_descriptor {


}
}