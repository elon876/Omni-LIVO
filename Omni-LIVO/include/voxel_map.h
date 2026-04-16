/**
 * @file voxel_map.h
 * @brief Voxel Map module for Omni-LIVO
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

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <ros/ros.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <unordered_set>


#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

class LIVMapper;  // Forward declaration

static int voxel_plane_id = 0;

typedef struct VoxelMapConfig {
    double max_voxel_size_;
    int max_layer_;
    int max_iterations_;
    std::vector<int> layer_init_num_;
    int max_points_num_;
    double planner_threshold_;
    double beam_err_;
    double dept_err_;
    double sigma_num_;
    bool is_pub_plane_map_;
    // Local sliding map (FAST-LIO style) — bound memory by keeping only a cube
    // of side 2*half_map_size_ around the robot; slide when |pos-origin|>sliding_thresh_
    bool map_sliding_en_;
    double half_map_size_;
    double sliding_thresh_;
    int max_visual_voxels_;
} VoxelMapConfig;

typedef struct PointToPlane {
    Eigen::Vector3d point_b_;
    Eigen::Vector3d point_w_;
    Eigen::Vector3d normal_;
    Eigen::Vector3d center_;
    Eigen::Matrix<double, 6, 6> plane_var_;
    M3D body_cov_;
    int layer_;
    double d_;
    double eigen_value_;
    bool is_valid_;
    float dis_to_plane_;
    double voxel_timestamp_;
} PointToPlane;

typedef struct VoxelPlane {
    Eigen::Vector3d center_;
    Eigen::Vector3d normal_;
    Eigen::Vector3d y_normal_;
    Eigen::Vector3d x_normal_;
    Eigen::Matrix3d covariance_;
    Eigen::Matrix<double, 6, 6> plane_var_;
    float radius_ = 0;
    float min_eigen_value_ = 1;
    float mid_eigen_value_ = 1;
    float max_eigen_value_ = 1;
    float d_ = 0;
    int points_size_ = 0;
    bool is_plane_ = false;
    bool is_init_ = false;
    int id_ = 0;
    bool is_update_ = false;

    VoxelPlane() {
        plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
        covariance_ = Eigen::Matrix3d::Zero();
        center_ = Eigen::Vector3d::Zero();
        normal_ = Eigen::Vector3d::Zero();
    }
} VoxelPlane;

class VOXEL_LOCATION {
public:
    int64_t x, y, z;

    VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std {
    template<>
    struct hash<VOXEL_LOCATION> {
        int64_t operator()(const VOXEL_LOCATION &s) const {
            using std::hash;
            using std::size_t;
            return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
        }
    };
} // namespace std


void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree {

public:
    VoxelOctoTree() = default;

    std::vector<pointWithVar> temp_points_;
    VoxelPlane *plane_ptr_;
    int layer_;
    int octo_state_; // 0 is end of tree, 1 is not
    VoxelOctoTree *leaves_[8];
    double voxel_center_[3]; // x, y, z
    std::vector<int> layer_init_num_;
    float quater_length_;
    float planer_threshold_;
    int points_size_threshold_;
    int update_size_threshold_;
    int max_points_num_;
    int max_layer_;
    int new_points_;
    bool init_octo_;
    bool update_enable_;
    double creation_timestamp_;  // When this voxel was first created


    VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
            : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold),
              max_points_num_(max_points_num),
              planer_threshold_(planer_threshold) {
        temp_points_.clear();
        octo_state_ = 0;
        new_points_ = 0;
        update_size_threshold_ = 5;
        init_octo_ = false;
        update_enable_ = true;
        for (int i = 0; i < 8; i++) {
            leaves_[i] = nullptr;
        }
        plane_ptr_ = new VoxelPlane;
        creation_timestamp_ = -1.0;
    }

    ~VoxelOctoTree() {
        for (int i = 0; i < 8; i++) {
            delete leaves_[i];
        }
        delete plane_ptr_;
    }

    void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);

    void init_octo_tree();

    void cut_octo_tree();

    void UpdateOctoTree(const pointWithVar &pv);

    VoxelOctoTree *find_correspond(Eigen::Vector3d pw);

    VoxelOctoTree *Insert(const pointWithVar &pv);
};

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config);
struct PlaneWithVoxel {
    VoxelPlane plane;
    const VoxelOctoTree* voxel_ptr;

    PlaneWithVoxel() : voxel_ptr(nullptr) {}
};
class VoxelMapManager {
public:
    VoxelMapManager() = default;

    VoxelMapConfig config_setting_;
    int current_frame_id_ = 0;
    ros::Publisher voxel_map_pub_;
    std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;
    typedef std::function<PointCloudXYZI::Ptr(double)> HistoricalPointCloudCallback;

    PointCloudXYZI::Ptr feats_undistort_;
    PointCloudXYZI::Ptr feats_down_body_;
    PointCloudXYZI::Ptr feats_down_world_;

    M3D extR_;
    V3D extT_;

    StatesGroup state_;
    V3D position_last_;

    geometry_msgs::Quaternion geoQuat_;

    int feats_down_size_;
    int effct_feat_num_;
    std::vector<M3D> cross_mat_list_;
    std::vector<M3D> body_cov_list_;
    std::vector<pointWithVar> pv_list_;
    std::vector<PointToPlane> ptpl_list_;

    VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
            : config_setting_(config_setting), voxel_map_(voxel_map) {
        current_frame_id_ = 0;
        feats_undistort_.reset(new PointCloudXYZI());
        feats_down_body_.reset(new PointCloudXYZI());
        feats_down_world_.reset(new PointCloudXYZI());
    };

    void setLIVMapperReference(LIVMapper* mapper);


    void setHistoricalPointCloudCallback(HistoricalPointCloudCallback callback) {
        historical_point_cloud_callback_ = callback;
    }

    void StateEstimation(StatesGroup &state_propagat);

    void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

    void BuildVoxelMap();

    V3F RGBFromVoxel(const V3D &input_point);

    void UpdateVoxelMap(const std::vector<pointWithVar> &input_points, double timestamp);

    void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

    void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                               double &prob, PointToPlane &single_ptpl);

    HistoricalPointCloudCallback historical_point_cloud_callback_ = nullptr;

    // Method to get the current transformation
    Eigen::Matrix4d getCurrentTransform() const {
        return current_transform_;
    }

    // Method to transform a point from world to map coordinates
    Eigen::Vector3d worldToMapCoordinates(const Eigen::Vector3d& point_world) const {
        return worldToMapCoordinates(point_world, -1.0);
    }

    void initializeNewMapWithPoints(const vector<pointWithVar> &initial_points);

    void BuildResidualListWithMap(vector<pointWithVar> &pv_list, vector<PointToPlane> &ptpl_list,
                                  unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &map_to_use);

    auto& getActiveMap() {
        return voxel_map_;
    }

    void clearMemOutOfMapForNew(const int& x_max, const int& x_min,
                                const int& y_max, const int& y_min,
                                const int& z_max, const int& z_min) {
        // Not needed with new approach
    }
    // 根据时间戳计算特定变换
    SE3 getTransformForTimestamp(double timestamp) const;

    // 修改世界坐标到地图坐标的转换，考虑时间戳
    Eigen::Vector3d worldToMapCoordinates(const Eigen::Vector3d& point_world, double timestamp) const;

    // 修改地图坐标到世界坐标的转换
    Eigen::Vector3d mapToWorldCoordinates(const Eigen::Vector3d& point_map, double timestamp) const;

    // 接受匹配帧和查询帧时间戳的地图变换函数
    void applyTrajectoryTransformToMap(function<SE3(double)> transform_function, double match_timestamp,
                                       double query_timestamp);
    ~VoxelMapManager() {
        // Clean up
        for (auto& pair : voxel_map_) {
            if (pair.second) delete pair.second;
        }
        voxel_map_.clear();
    }

    bool has_correction_ = false;
    void initVoxelMapPublisher(ros::NodeHandle& nh);
    void republishAfterLoopClosure();
    void pubVoxelMap();
    void clearOldMarkers();
    ros::Publisher voxel_pointcloud_pub_;

    double oldest_kept_timestamp_ = -1.0;  // 保留的最旧时间戳（公开给VIO使用）

    // Sliding local map: recenter to `pos` when it drifts > sliding_thresh_ from
    // local_map_origin_, erasing voxels outside the cube of side 2*half_map_size_.
    // Returns number of voxels erased.
    size_t mapSliding(const V3D& pos);
    V3D local_map_origin_ = V3D::Zero();
    bool local_map_inited_ = false;

private:
    // Store the map correction transformation
    SE3 map_correction_ = SE3();
    double query_frame_timestamp_ = -1.0;  // 回环查询帧时间戳
    double match_frame_timestamp_ = -1.0;
    // Matrix to track the current active transformation
    // This allows us to handle sequential corrections
    Eigen::Matrix4d current_transform_ = Eigen::Matrix4d::Identity();

    void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer,
                        std::vector<VoxelPlane> &plane_list);


    void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
    mutable std::atomic<bool> is_applying_transform_{false};

    // 基于轨迹的变换函数
    std::function<SE3(double)> trajectory_transform_function_ = nullptr;
    bool has_trajectory_transform_ = false;
    LIVMapper* liv_mapper_ = nullptr; // Pointer to LIVMapper
    ros::Timer voxel_map_timer_;
    bool need_clear_markers_ = false;

    // 帧数滑动窗口相关成员变量
    std::map<int, double> frame_to_timestamp_;  // 帧ID到时间戳的映射
    int current_frame_count_ = 0;  // 当前帧计数
    void GetUpdatePlaneWithVoxel(const VoxelOctoTree *current_octo,
                                 const int pub_max_voxel_layer,
                                 std::vector<PlaneWithVoxel> &plane_list);

    void updateVoxelGeometry(VoxelOctoTree* voxel, const SE3& transform);

    bool isVoxelSafeToAccess(VoxelOctoTree* voxel);

    void saveVoxelMapPCD(const string &suffix);
    void applyTransform(
            std::function<SE3(double)>& transform_function,
            double match_timestamp,
            double query_timestamp);

};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_