/**
 * @file vio.h
 * @brief Visual-Inertial Odometry module for Omni-LIVO
 *
 * @copyright Copyright (c) 2026 Hangzhou Institute for Advanced Study,
 *            University of Chinese Academy of Sciences
 *
 * @par Project: Omni-LIVO
 * Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry
 * via Photometric Migration and ESIKF Fusion
 *
 * This work is based on FAST-LIVO2:
 * C. Zheng, W. Xu, Q. Guo, and F. Zhang, "FAST-LIVO2: Fast, direct LiDAR-inertial-visual
 * odometry," IEEE Trans. Robot., vol. 40, pp. 1529-1546, 2024.
 *
 * @author Yinong Cao (cyn_688@163.com), Chenyang Zhang, Xin He*, Yuwei Chen, Chengyu Pu,
 *         Bingtao Wang, Kaile Wu, Shouzheng Zhu, Fei Han, Shijie Liu, Chunlai Li, Jianyu Wang
 * @author *Corresponding author: Xin He (xinhe@ucas.ac.cn)
 *
 * @par Repository
 * https://github.com/elon876/Omni-LIVO
 *
 * @par Citation
 * If you use this code in your research, please cite our paper:
 * Y. Cao et al., "Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR
 * Odometry via Photometric Migration and ESIKF Fusion," IEEE Robotics and Automation
 * Letters, 2026.
 */

#ifndef VIO_H_
#define VIO_H_

#include "voxel_map.h"
#include "feature.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <sstream>
#include <iomanip>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>
#include <unordered_set>
#include <deque>

struct SubSparseMap {
    vector<float> propa_errors;
    vector<float> errors;
    vector<vector<float>> warp_patch;
    vector<int> search_levels;
    vector<VisualPoint *> voxel_points;
    vector<double> inv_expo_list;
    vector<pointWithVar> add_from_voxel_map;
    vector<int> camera_ids;

    SubSparseMap() {
        propa_errors.reserve(SIZE_LARGE);
        errors.reserve(SIZE_LARGE);
        warp_patch.reserve(SIZE_LARGE);
        search_levels.reserve(SIZE_LARGE);
        voxel_points.reserve(SIZE_LARGE);
        inv_expo_list.reserve(SIZE_LARGE);
        add_from_voxel_map.reserve(SIZE_SMALL);
        camera_ids.reserve(SIZE_LARGE);
    };

    void reset() {
        propa_errors.clear();
        errors.clear();
        warp_patch.clear();
        search_levels.clear();
        voxel_points.clear();
        inv_expo_list.clear();
        add_from_voxel_map.clear();
        camera_ids.clear();
    }
};

class Warp {
public:
    Matrix2d A_cur_ref;
    int search_level;

    Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}

    ~Warp() {}
};

class VOXEL_POINTS {
public:
    std::vector<VisualPoint *> voxel_points;
    int count;
    double creation_timestamp_;

    VOXEL_POINTS(int num) : count(num), creation_timestamp_(-1.0) {}

    ~VOXEL_POINTS() {
        for (VisualPoint *vp: voxel_points) {
            if (vp != nullptr) {
                delete vp;
                vp = nullptr;
            }
        }
    }
};

class VIOManager {
public:
    int grid_size{};
    int max_total_points = 300;
    int points_per_camera_min = 50;
    int points_per_camera_max = 150;
    vector<vk::AbstractCamera *> cams;
    StatesGroup *state{};
    StatesGroup *state_propagat{};
    bool raycast_en = false;
    std::vector<std::vector<std::vector<V3D>>> rays_with_sample_points; // [cam_id][grid_idx][sample_points]
    std::vector<std::vector<int>> border_flag; // [cam_id][grid_idx]


    std::vector<Eigen::Matrix3d> Rci_vec;
    std::vector<Eigen::Matrix3d> Rcl_vec;
    std::vector<Eigen::Matrix3d> Rcw_vec;
    std::vector<Eigen::Vector3d> Pci_vec;
    std::vector<Eigen::Vector3d> Pcl_vec;
    std::vector<Eigen::Vector3d> Pcw_vec;

    std::vector<Eigen::Matrix3d> Jdphi_dR_vec;
    std::vector<Eigen::Matrix3d> Jdp_dt_vec;
    std::vector<Eigen::Matrix3d> Jdp_dR_vec;

    M3D Rli;
    V3D Pli;
    vector<int> grid_num;
    vector<int> map_index;
    vector<int> update_flag;
    vector<float> map_dist;
    vector<float> scan_value;
    vector<float> patch_buffer;
    bool normal_en{}, inverse_composition_en{}, exposure_estimate_en{}, has_ref_patch_cache{};
    bool ncc_en = false, colmap_output_en = false;

    int width{}, height{}, grid_n_width{}, grid_n_height{}, length{};
    double image_resize_factor{};
    int patch_pyrimid_level{}, patch_size{}, patch_size_total{}, patch_size_half{}, border{}, warp_len{};
    int max_iterations{}, total_points{};

    double img_point_cov{}, outlier_threshold{}, ncc_thre{};

    std::vector<std::vector<int>> grid_num_per_cam_;
    std::vector<std::vector<float>> map_dist_per_cam_;
    std::vector<std::vector<VisualPoint*>> retrieve_voxel_points_per_cam_;
    std::vector<std::vector<int>> scan_grid_num_per_cam_;
    std::vector<std::vector<float>> scan_value_per_cam_;
    std::vector<std::vector<pointWithVar>> scan_append_points_per_cam_;

    std::vector<std::unordered_set<VisualPoint*>> retrieve_voxel_points_list_buffer_;
    std::vector<cv::Mat> depth_imgs_buffer_;
    std::vector<int> grid_num_buffer_;
    std::vector<float> map_dist_buffer_;
    std::vector<VisualPoint*> retrieve_voxel_points_buffer_;
    std::vector<float> scan_value_buffer_;

    SubSparseMap *visual_submap{};

    double compute_jacobian_time{}, update_ekf_time{};
    double ave_total = 0;
    // double ave_build_residual_time = 0;
    // double ave_ekf_time = 0;

    int frame_count = 0;
    bool plot_flag{};

    Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
    MatrixXd K, H_sub_inv;

    ofstream fout_camera, fout_colmap;
    unordered_map<VOXEL_LOCATION, VOXEL_POINTS *> feat_map;
    unordered_map<VOXEL_LOCATION, int> sub_feat_map;
    unordered_map<int, Warp *> warp_map;
    vector<VisualPoint *> retrieve_voxel_points;
    vector<pointWithVar> append_voxel_points;
    FramePtr new_frame_;
    std::vector<cv::Mat> imgs_cp, imgs_rgb;
    cv::Mat panorama_image;

    struct CameraPhotoParams {
        double exposure_factor;
        std::vector<double> vignetting;
        bool parameters_initialized;
    };
    std::vector<CameraPhotoParams> camera_photo_params;

    struct CrossCameraObservation {
        int source_cam_id;
        int target_cam_id;
        V2D source_px;
        V2D target_px;
        double photometric_error;
        Matrix2d warp_matrix;
        std::vector<float> source_patch;
        bool is_valid;
    };
    void initializeCameraPhotoParams();
    Matrix2d computeCrossCameraWarpMatrix(
            int source_cam_id, int target_cam_id,
            const Vector2d& source_px, const V3D& point_pos,
            const V3D& point_normal);
    float applyCameraPhotoCorrection(float intensity, int cam_id, const V2D& pixel_pos);
    void addCrossCameraConsistencyConstraint(
            VisualPoint* pt, int source_cam_id, int target_cam_id,
            MatrixXd& H_sub, VectorXd& z, int level, int& row_offset);
    double evaluateCrossCameraConsistency(const CrossCameraObservation& obs, int level);

    int total_cross_camera_observations = 0;
    int successful_cross_camera_tracks = 0;
    bool enable_cross_camera_tracking = false;
    enum CellType {
        TYPE_MAP = 1,
        TYPE_POINTCLOUD,
        TYPE_UNKNOWN
    };

    VIOManager();

    ~VIOManager();
    void updateStateInverse(const std::vector<cv::Mat> &imgs, int level);

    void updateState(const std::vector<cv::Mat> &imgs, int level);

    void processFrame(const std::vector<cv::Mat> &imgs, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
                  double frame_timestamp);

    void retrieveFromVisualSparseMap(const std::vector<cv::Mat> imgs, vector<pointWithVar> &pg, const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);

    void generateVisualMapPoints(const std::vector<cv::Mat> &imgs, vector<pointWithVar> &pg);

    void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);

    void setLidarToCameraExtrinsic(std::vector<std::vector<double>> &R, std::vector<std::vector<double>> &P);

    void initializeVIO();

    void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);

    void computeProjectionJacobian(int cam_idx, V3D p, MD(2, 3) &J);

    void computeJacobianAndUpdateEKF(const std::vector<cv::Mat> imgs);

    void resetGrid();

    void updateVisualMapPoints(std::vector<cv::Mat>& imgs);

    void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref,
                             const double depth_ref, const SE3 &T_cur_ref,
                             const int level_ref,
                             const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);

    void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                       const V3D &xyz_ref, const V3D &normal_ref, const SE3 &T_cur_ref,
                                       const int level_ref, Matrix2d &A_cur_ref);

    void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref,
                    const int search_level,
                    const int pyramid_level, const int halfpatch_size, float *patch);

    void insertPointIntoVoxelMap(VisualPoint *pt_new);
    void setCurrentTimestamp(double timestamp) { current_timestamp_ = timestamp; }
    double current_timestamp_ = -1.0;

    void plotTrackedPoints();

    void updateFrameState(StatesGroup state);

    void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);

    void applyTrajectoryTransformToVisualMap(
        std::function<SE3(double)> transform_function,
        double match_timestamp,
        double query_timestamp);

    void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);

    void precomputeReferencePatches(int level);

    void dumpDataForColmap();

    double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);

    int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);

    V3F getInterpolatedPixel(cv::Mat img, V2D pc);

    void resetAfterLoopClosure();

    void cleanupCrossCameraData();

    void cleanupOldVisualPoints();
    void cleanupVisualMapByTimestamp(double oldest_kept_timestamp);
    int max_point_age_frames = 50;
    int cleanup_interval_frames = 10;
    int last_cleanup_frame_id = 0;

    void cleanupOldFrames();
    std::deque<FramePtr> frame_history_;
    int max_frame_history = 3;

    // Bound visual-map memory by capping feat_map voxel count. When enabled and
    // feat_map.size() > max_visual_voxels_, drop oldest voxels (by creation_timestamp_)
    // until the cap is satisfied.
    bool map_sliding_en_ = false;
    int max_visual_voxels_ = 10000;
    size_t capVisualMap();

    bool enable_dynamic_covariance_ = false;
    int dynamic_cov_warmup_frames = 200;
    double warmup_cov_scale = 500.0;
    int ideal_total_points = 150;
    double min_cov_scale = 10.0;
    double max_cov_scale = 2000.0;
    double dynamic_cov_error_max = 50.0;

    void initializeRaycast();

private:
    std::vector<MatrixXd> H_sub_all;
    std::vector<VectorXd> z_all;

    std::vector<double> prev_cov_scale_per_cam_;
    std::vector<double> prev_avg_error_per_cam_;
    std::vector<int> prev_n_meas_per_cam_;

    double prev_cov_scale_ = 10.0;
    double prev_avg_error_ = 5.0;
    int prev_n_meas_ = 0;

    double calculateCoVarianceScale(double realtime_avg_error, int realtime_n_meas);
    double calculateCoVarianceScalePerCam(int cam_idx, double realtime_avg_error, int realtime_n_meas);

    bool extractPatchSafely(const cv::Mat &img, const V2D &center, vector<float> &patch);

    bool isRealCrossCameraPoint(VisualPoint *pt, int current_cam_id);

    void updateCrossCameraHistory(VisualPoint *pt, int cam_id);


};

typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_