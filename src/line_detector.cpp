#include <msckf_mono/line_detector.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace line_detector {

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using cv::Mat;
using namespace cv::line_descriptor;
using namespace std;

// inline Matrix3d skew_symmetric(const Vector3d &x) {
//     Matrix3d X;
//     X <<   0, -x(2),  x(1),
//         x(2),     0, -x(0),
//        -x(1),  x(0),     0;
//     return X;
// }

void LineDetector::detect_lines(const Mat &image,
                                vector<KeyLine> &lines,
                                Mat &lines_descriptor) {
    // LSD + LBD
    auto lsd_detector = LSDDetectorC::createLSDDetector();
    LSDDetectorC::LSDOptions lsdOptions;
    // todo: put these parameters into configuration file
    // lsd_nfeatures     = 100;        // number of LSD lines detected (set to 0 if keeping all lines)
    // lsd_refine        = 0;          // the way of refining or not the detected lines
    // lsd_scale         = 1.2;        // scale of the image that will be used to find the lines
    // lsd_sigma_scale   = 0.6;        // sigma for Gaussian filter
    // lsd_quant         = 2.0;        // bound to the quantization error on the gradient norm
    // lsd_ang_th        = 22.5;       // gradient angle tolerance in degrees
    // lsd_log_eps       = 1.0;        // detection threshold (only for advanced refinement)
    // lsd_density_th    = 0.6;        // minimal density of aligned region points in the enclosing rectangle
    // lsd_n_bins        = 1024;       // number of bins in pseudo-ordering of gradient modulus
    // lsd_min_length    = 0.5
    lsdOptions.refine = Config::lsdRefine();
    lsdOptions.scale = Config::lsdScale();
    lsdOptions.sigma_scale = Config::lsdSigmaScale();
    lsdOptions.quant = Config::lsdQuant();
    lsdOptions.ang_th = Config::lsdAngTh();
    lsdOptions.log_eps = Config::lsdLogEps();
    lsdOptions.density_th = Config::lsdDensityTh();
    lsdOptions.n_bins = Config::lsdNBins();
    lsdOptions.min_length = 1.0;
    lsd_detector->detect(image, lines, lsdOptions.scale, 1, lsdOptions);

    // eliminate some lines with low response if too many lines
    if (lines.size() > Config::lsdNFeatures() && Config::lsdNFeatures() != 0) {
        sort(lines.begin(), lines.end(), [](const KeyLine &a,
                                            const KeyLine &b){
            return a.response > b.response;
        });
        lines.resize(Config::lsdNFeatures());
        int cnt = 0;
        for (auto& line: lines)
            line.class_id = cnt++;
    }

    // cout << "lines size: " << lines.size() << endl; 
    compute_descriptors(image, lines, lines_descriptor);
}

void LineDetector::match_lines(const Mat& queryDescriptors,
                               const Mat& trainDescriptors,
                               vector<cv::DMatch> &matches) {
    auto line_mather = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    line_mather->match(queryDescriptors, trainDescriptors, matches);
}

void LineDetector::match_lines(const cv::Mat& queryDescriptors,
                               const cv::Mat& trainDescriptors,
                               vector<int> &matches) {
    StVO::match(queryDescriptors, trainDescriptors, Config::minRatio12L(), matches);
}

bool LineDetector::judge_distance(const KeyLine &line1,
                                  const KeyLine &line2,
                                  float distance_threshold) {
    float mid1_x = (line1.startPointX + line1.endPointX) / 2;
    float mid1_y = (line1.startPointY + line1.endPointY) / 2;
    float mid2_x = (line2.startPointX + line2.endPointX) / 2;
    float mid2_y = (line2.startPointY + line2.endPointY) / 2;

    return (mid1_x - mid2_x) * (mid1_x - mid2_x) +
           (mid1_y - mid2_y) * (mid1_y - mid2_y)
           <= distance_threshold * distance_threshold;
}

bool LineDetector::judge_angle(const KeyLine &line1,
                               const KeyLine &line2,
                               float angle_threshold) {
    float alpha1 = std::atan2(line1.endPointY - line1.startPointY,
                              line1.endPointX - line1.startPointX);
    float alpha2 = std::atan2(line2.endPointY - line2.startPointY,
                              line2.endPointX - line2.startPointX);
    return std::fabs(alpha1 - alpha2) <= angle_threshold;
}

void LineDetector::compute_descriptors(const Mat &image,
                                       vector<KeyLine> &lines,
                                       Mat &lines_descriptor) {
    auto lbd_computer = BinaryDescriptor::createBinaryDescriptor();
    lbd_computer->compute(image, lines, lines_descriptor);
}

LineTrackHandler::LineTrackHandler(const Mat &K,
                                   const Mat &distortion_coeffs,
                                   std::ofstream &debug_info,
                                   int COL, int ROW) : 
                                   K_(K), distortion_coeffs_(distortion_coeffs),
                                   debug_info(debug_info), 
                                   COL(COL), ROW(ROW)
{
    line_cnt = 0;
    initialized = false;
    // camera = CameraModel<float>(K, distortion_coeffs);
    log.open("/home/ycdfwzy/dataset/processed_images/log.txt");

    // Mat img = cv::imread("/home/ycdfwzy/dataset/MH_03_medium/cam0/data/1403637130538319104.png");
    // vector<KeyLine> lines_;
    // Mat descs_;
    // LineDetector::detect_lines(img, lines_, descs_);
    
    // for (auto& line: lines_) {
    //     debug_info << line.startPointX << " " << line.startPointY << " " << line.endPointX << " " << line.endPointY << endl;
    // }
}

void LineTrackHandler::set_current_image(Mat &image, double time) {
    this->prev_time_       = this->cur_time_;
    this->prev_image_      = this->cur_image_;
    this->prev_lines_      = this->cur_lines_;
    this->prev_line_ids_   = this->cur_line_ids_;
    // this->prev_line_descs_ = this->cur_line_descs_;
    copy(this->new_lines_.begin(),
         this->new_lines_.end(),
         back_inserter(this->prev_lines_));
    copy(this->new_line_ids_.begin(),
         this->new_line_ids_.end(),
         back_inserter(this->prev_line_ids_));

    if (!this->prev_lines_.empty())
        LineDetector::compute_descriptors(this->prev_image_, this->prev_lines_, this->prev_line_descs_);

    this->cur_time_ = time;
    this->cur_image_ = image;
    this->cur_lines_.clear();
    this->cur_line_ids_.clear();

    this->new_lines_.clear();
    this->new_line_ids_.clear();
}

void draw_lines(const Mat& inImage, const vector<KeyLine> &lines, Mat& outImage) {
    if( inImage.type() == CV_8UC3 ){
        inImage.copyTo( outImage );
    }else if( inImage.type() == CV_8UC1 ){
        cvtColor( inImage, outImage, cv::COLOR_GRAY2BGR );
    }else{
        CV_Error( cv::Error::StsBadArg, "Incorrect type of input image.\n" );
        return;
    }

    for (const auto& line_: lines) {
        auto color = cv::Scalar(255, 0, 255);

        cv::line(outImage,
                 cv::Point(line_.startPointX, line_.startPointY),
                 cv::Point(line_.endPointX, line_.endPointY),
                 color, 2);
    }
}

bool LineTrackHandler::inBorder(const cv::line_descriptor::KeyLine& line) const {
    const int BORDER_SIZE = 2;
    int start_x = cvRound(line.startPointX);
    int start_y = cvRound(line.startPointY);
    int end_x = cvRound(line.endPointX);
    int end_y = cvRound(line.endPointY);
    return (BORDER_SIZE <= start_x && start_x < COL - BORDER_SIZE && BORDER_SIZE <= start_y && start_y < ROW - BORDER_SIZE) ||
           (BORDER_SIZE <= end_x && end_x < COL - BORDER_SIZE && BORDER_SIZE <= end_y && end_y < ROW - BORDER_SIZE);
}

string prefix = "/home/ycdfwzy/dataset/processed_images/";

void LineTrackHandler::drawTrackedLineFeatures(const KeyLine &line,
                                               int line_id) const {
    // check file path
    string path = prefix + "tracked_feature/" + to_string(line_id) + "/";
    int mode;
    if (access(path.c_str(), F_OK) == -1) {
        if (mkdir(path.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
            debug_info << "can't create directory " << path << endl;
            return;
        }
        if (chmod(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
            debug_info << "can't chmod directory " << path << endl;
            return;
        }
    }

    // draw feature on image
    Mat outImage;
    if (this->cur_image_.type() == CV_8UC3) {
        this->cur_image_.copyTo(outImage);
    } else
    if (this->cur_image_.type() == CV_8UC1) {
        cvtColor(this->cur_image_, outImage, cv::COLOR_GRAY2BGR);
    } else
    {
        CV_Error(cv::Error::StsBadArg, "Incorrect type of input image.\n");
        return;
    }

    // write to file
    auto color = cv::Scalar(255, 0, 255);
    cv::line(outImage,
             line.getStartPoint(),
             line.getEndPoint(),
             color, 2);
    cv::imwrite(path + to_string(this->cur_time_) + ".jpg", outImage);
}

// process image:
//   extract features
//   match features
//   set current features and new features
void LineTrackHandler::process_current_image(bool drawlines) {
    LineDetector::detect_lines(this->cur_image_, this->cur_lines_, this->cur_line_descs_);

    if (!this->initialized) { // first process
        this->initialized = true;

        copy(this->cur_lines_.begin(),
             this->cur_lines_.end(),
             back_inserter(this->new_lines_));
        this->new_line_ids_.resize(this->new_lines_.size());
        for (int i = 0; i < this->new_line_ids_.size(); i++) {
            this->new_line_ids_[i] = this->line_cnt;
            this->line_cnt++;
        }
        this->cur_lines_.clear();
        this->cur_line_ids_.clear();
        visualizer.add_new_features(this->new_lines_, this->new_line_ids_);

        if (drawlines) {
            for (int i = 0; i < this->new_lines_.size(); i++) {
                this->drawTrackedLineFeatures(this->new_lines_[i], this->new_line_ids_[i]);
            }
        }
        return;
    }

    vector<int> matches;
    LineDetector::match_lines(this->cur_line_descs_, this->prev_line_descs_, matches);

    // filter
    int total_matched = 0;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i] != -1) {
            if (inBorder(this->cur_lines_.at(i)) &&
                LineDetector::judge_distance(this->prev_lines_.at(matches[i]),
                                             this->cur_lines_.at(i)) &&
                LineDetector::judge_angle(this->prev_lines_.at(matches[i]),
                                          this->cur_lines_.at(i))) {
                total_matched++;
            } else
            {
                matches[i] = -1;
            }
        }
    }

    vector<KeyLine> tmp_lines(this->cur_lines_);
    this->cur_lines_.clear();
    this->cur_line_ids_.clear();
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i] == -1) { // new feature
            this->new_lines_.push_back(tmp_lines.at(i));
            this->new_line_ids_.push_back(this->line_cnt);
            this->line_cnt++;
        } else
        { // tracked_feature
            this->cur_lines_.push_back(tmp_lines.at(i));
            this->cur_line_ids_.push_back(this->prev_line_ids_[matches[i]]);
        }
    }

    visualizer.update_old_features(this->cur_lines_, this->cur_line_ids_);
    visualizer.add_new_features(this->new_lines_, this->new_line_ids_);

    if (drawlines) {
        for (int i = 0; i < this->cur_lines_.size(); i++) {
            this->drawTrackedLineFeatures(this->cur_lines_[i], this->cur_line_ids_[i]);
        }
        for (int i = 0; i < this->new_lines_.size(); i++) {
            this->drawTrackedLineFeatures(this->new_lines_[i], this->new_line_ids_[i]);
        }
    }
}

Mat LineTrackHandler::get_feature_image() {
    return visualizer.get_image(this->cur_image_);
}

void LineVisualizer::update_old_features(const vector<KeyLine> &tracked_line_features,
                                         const vector<int> &tracked_line_features_id) {
    unordered_set<int> ids;
    ids.clear();
    for (const auto& iter: this->lines)
        ids.insert(iter.first);
    // update tracked features
    for (int i = 0; i < tracked_line_features.size(); i++) {
        if (this->lines.find(tracked_line_features_id[i]) == this->lines.end()) {
            cout << "can't track unexisting features!" << endl;
            continue;
        }
        int time_stamp = this->lines[tracked_line_features_id[i]].second;
        this->lines[tracked_line_features_id[i]] = make_pair(tracked_line_features[i], time_stamp);
        ids.erase(tracked_line_features_id[i]);
    }

    // remove untracked features
    for (size_t id: ids)
        this->lines.erase(id);
}

void LineVisualizer::add_new_features(const vector<KeyLine>& new_line_features,
                                      const vector<int>& new_line_features_id) {
    for (int i = 0; i < new_line_features.size(); i++) {
        if (this->lines.find(new_line_features_id[i]) != this->lines.end()) {
            cout << "can't add new features which has been added!" << endl;
            continue;
        }
        this->lines[new_line_features_id[i]] = make_pair(new_line_features[i], this->cur_time_stamp);
    }
}

const int thickness = 2;
Mat LineVisualizer::get_image(const Mat& image) {
    // update global time stamp
    this->cur_time_stamp++;

    Mat outImage;

    if( image.type() == CV_8UC3 ){
        image.copyTo( outImage );
    }else if( image.type() == CV_8UC1 ){
        cvtColor( image, outImage, cv::COLOR_GRAY2BGR );
    }else{
        CV_Error( cv::Error::StsBadArg, "Incorrect type of input image.\n" );
        return outImage;
    }

    for (const auto &line_data: this->lines) {
        const KeyLine& line_ = line_data.second.first;
        int time_stamp = line_data.second.second;

        double t = std::min(10, this->cur_time_stamp - time_stamp);
        auto color = cv::Scalar(25 * (10 - t), 0, 25 * t);

        cv::line(outImage,
                 cv::Point(line_.startPointX, line_.startPointY),
                 cv::Point(line_.endPointX, line_.endPointY),
                 color, thickness);
    }

    return outImage;
}

} // namespaced line_detector