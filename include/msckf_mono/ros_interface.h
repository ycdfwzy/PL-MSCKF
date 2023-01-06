#ifndef MSCKF_MONO_ROS_INTERFACE_H_
#define MSCKF_MONO_ROS_INTERFACE_H_

#include <fstream>
#include <iostream>
#include <vector>
#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <msckf_mono/types.h>
#include <msckf_mono/msckf.h>
#include <msckf_mono/corner_detector.h>
#include <msckf_mono/line_detector.h>
#include <atomic>
#include <chrono>
#include <unistd.h>

namespace msckf_mono
{
  class RosInterface {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      RosInterface(ros::NodeHandle nh);
      ~RosInterface() { output.close(); debug_info.close(); }

      void imuCallback(const sensor_msgs::ImuConstPtr& imu);

      void imageCallback(const sensor_msgs::ImageConstPtr& msg);

      void pathCallback(const geometry_msgs::PointStampedConstPtr& path_msg);

      void publish_core(const ros::Time& publish_time);

      void publish_extra(const ros::Time& publish_time);

      void publish_extra_line(const ros::Time& publish_time);

    private:
      std::ofstream output;
      std::ofstream debug_info;

      ros::NodeHandle nh_;
      image_transport::ImageTransport it_;

      image_transport::Subscriber image_sub_;
      image_transport::Publisher track_image_pub_;
      ros::Publisher odom_pub_;
      ros::Publisher groundtruth_pub_;

      ros::Subscriber imu_sub_;
      ros::Subscriber groundtruth_sub_;

      void load_parameters();

      bool debug_;

      std::vector<std::tuple<double, imuReading<float>>> imu_queue_;
      double prev_imu_time_;

      void setup_track_handler();
      std::shared_ptr<corner_detector::TrackHandler> track_handler_;
      std::shared_ptr<line_detector::LineTrackHandler> line_track_handler_;

      Matrix3<float> R_imu_cam_;
      Vector3<float> p_imu_cam_;

      Matrix3<float> R_cam_imu_;
      Vector3<float> p_cam_imu_;

      std::string camera_model_;
      cv::Mat K_;
      std::string distortion_model_;
      cv::Mat dist_coeffs_;

      int n_grid_cols_;
      int n_grid_rows_;
      float ransac_threshold_;

      enum CalibrationMethod { TimedStandStill };
      CalibrationMethod imu_calibration_method_;

      double stand_still_time_;
      double done_stand_still_time_;

      std::atomic<bool> imu_calibrated_;
      bool can_initialize_imu();
      void initialize_imu();

      int state_k_;
      void setup_msckf();
      MSCKF<float> msckf_;
      Camera<float> camera_;
      noiseParams<float> noise_params_;
      MSCKFParams<float> msckf_params_;
      imuState<float> init_imu_state_;
      bool using_line_feature;
      int image_height, image_width;
  };
}

#endif
