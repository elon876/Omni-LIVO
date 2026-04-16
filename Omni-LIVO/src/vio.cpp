/**
 * @file vio.cpp
 * @brief Visual-Inertial Odometry implementation for Omni-LIVO
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

#include "vio.h"

VIOManager::VIOManager() {
}

VIOManager::~VIOManager() {
    delete visual_submap;
    for (auto &pair: warp_map) delete pair.second;
    warp_map.clear();
    for (auto &pair: feat_map) delete pair.second;
    feat_map.clear();
}

size_t VIOManager::capVisualMap() {
    if (!map_sliding_en_) return 0;
    if ((int) feat_map.size() <= max_visual_voxels_) return 0;

    // Drop oldest voxels first (smallest creation_timestamp_).
    std::vector<std::pair<double, VOXEL_LOCATION>> aged;
    aged.reserve(feat_map.size());
    for (const auto &kv: feat_map) {
        aged.emplace_back(kv.second->creation_timestamp_, kv.first);
    }
    const size_t to_drop = feat_map.size() - (size_t) max_visual_voxels_;
    std::nth_element(aged.begin(), aged.begin() + to_drop, aged.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });

    size_t erased = 0;
    for (size_t i = 0; i < to_drop; ++i) {
        auto it = feat_map.find(aged[i].second);
        if (it != feat_map.end()) {
            delete it->second;  // ~VOXEL_POINTS frees VisualPoints + Features
            feat_map.erase(it);
            sub_feat_map.erase(aged[i].second);
            ++erased;
        }
    }
    std::cout << "[ VIO ] Cap visual map: erased " << erased
              << " oldest voxels, remaining " << feat_map.size() << std::endl;
    return erased;
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot) {
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void VIOManager::setLidarToCameraExtrinsic(std::vector<std::vector<double>> &R,
                                           std::vector<std::vector<double>> &P) {
    Rcl_vec.resize(cams.size());
    Pcl_vec.resize(cams.size());
    for (size_t i = 0; i < cams.size(); i++) {
        Rcl_vec[i] << R[i][0], R[i][1], R[i][2],
                R[i][3], R[i][4], R[i][5],
                R[i][6], R[i][7], R[i][8];
        Pcl_vec[i] << P[i][0], P[i][1], P[i][2];
    }
}


void VIOManager::initializeVIO() {
    if (cams.empty()) {
        std::cerr << "[VIOManager::initializeVIO] Error: no cameras!\n";
        return;
    }
    width = cams[0]->width();
    height = cams[0]->height();


    visual_submap = new SubSparseMap;
    image_resize_factor = cams[0]->scale();

    Rci_vec.resize(cams.size());
    Rcl_vec.resize(cams.size());
    Rcw_vec.resize(cams.size());
    Pci_vec.resize(cams.size());
    Pcl_vec.resize(cams.size());
    Pcw_vec.resize(cams.size());

    Jdphi_dR_vec.resize(cams.size());
    Jdp_dt_vec.resize(cams.size());
    Jdp_dR_vec.resize(cams.size());

    
    prev_cov_scale_per_cam_.resize(cams.size(), 10.0);
    prev_avg_error_per_cam_.resize(cams.size(), 5.0);
    prev_n_meas_per_cam_.resize(cams.size(), 0);

    
    if (state && state->inv_expo_time_per_cam.empty()) {
        state->inv_expo_time_per_cam.resize(cams.size(), 1.0);
        ROS_INFO("[VIO] Initialized %zu camera exposure parameters (inv_expo=1.0)", cams.size());
    }
    if (state_propagat && state_propagat->inv_expo_time_per_cam.empty()) {
        state_propagat->inv_expo_time_per_cam.resize(cams.size(), 1.0);
    }

    for(size_t i = 0; i < cams.size(); i++)
    {
        Rci_vec[i] = Rcl_vec[i] * Rli;
        Pci_vec[i] = Rcl_vec[i] * Pli + Pcl_vec[i];
        Jdphi_dR_vec[i] = Rci_vec[i];
        Eigen::Vector3d Pic_i = - Rci_vec[i].transpose() * Pci_vec[i];
        Eigen::Matrix3d tmp;
        tmp << 0,        -Pic_i.z(),  Pic_i.y(),
                Pic_i.z(),  0,        -Pic_i.x(),
                -Pic_i.y(),  Pic_i.x(), 0;

        Jdp_dR_vec[i] = - Rci_vec[i] * tmp;

    }
    if (grid_size > 10) {
        grid_n_width = ceil(static_cast<double>(width / grid_size));
        grid_n_height = ceil(static_cast<double>(height / grid_size));
    } else {
        grid_size = static_cast<int>(height / grid_n_height);
        grid_n_height = ceil(static_cast<double>(height / grid_size));
        grid_n_width = ceil(static_cast<double>(width / grid_size));
    }
    length = grid_n_width * grid_n_height;

    if (colmap_output_en) {
        std::vector<vk::PinholeCamera*> pinhole_cams;
        for (size_t i = 0; i < cams.size(); i++) {
            vk::PinholeCamera* cam = dynamic_cast<vk::PinholeCamera*>(cams[i]);
            if (cam) {
                pinhole_cams.push_back(cam);
            }
        }
        fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
        fout_colmap << "# Image list with two lines of data per image:\n";
        fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
        fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
        fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
        fout_camera << "# Camera list with one line of data per camera:\n";
        fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
        for (size_t i = 0; i < cams.size(); i++) {
            fout_camera << i+1 << " PINHOLE "
                        << cams[i]->width() << " " << cams[i]->height() << " "
                        << std::fixed << std::setprecision(9)  
                        << cams[i]->fx() << " " << cams[i]->fy() << " "
                        << cams[i]->cx() << " " << cams[i]->cy() << std::endl;
        }
        fout_camera.close();
    }

    grid_num.resize(length);
    map_index.resize(length);
    map_dist.resize(length);
    update_flag.resize(length);
    scan_value.resize(length);

    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size / 2);
    patch_buffer.resize(patch_size_total);
    warp_len = patch_size_total * patch_pyrimid_level;
    border = (patch_size_half + 2) * 8;

    retrieve_voxel_points.reserve(length);
    append_voxel_points.reserve(length);

    sub_feat_map.clear();

    initializeCameraPhotoParams();
    total_cross_camera_observations = 0;
    successful_cross_camera_tracks = 0;

    if (enable_cross_camera_tracking) {
        ROS_INFO("Cross-camera direct tracking enabled!");
    } else {
        ROS_INFO("Cross-camera direct tracking disabled!");
    }

    if (enable_dynamic_covariance_) {
        ROS_INFO("Dynamic covariance enabled! (warmup=%d frames, min=%.1f, max=%.1f)",
                 dynamic_cov_warmup_frames, min_cov_scale, max_cov_scale);
    } else {
        ROS_INFO("Dynamic covariance disabled!");
    }

}
void VIOManager::initializeRaycast() {
    ROS_INFO("[ VIO ] Starting raycast initialization...");
    raycast_en = true;
    int num_cameras = cams.size();
    if (num_cameras == 0) {
        ROS_ERROR("[ VIO ] No cameras available for raycast initialization!");
        raycast_en = false;
        return;
    }
    if (grid_size <= 0 || grid_n_height <= 0 || grid_n_width <= 0 || length <= 0) {
        ROS_ERROR("[ VIO ] Grid parameters not properly initialized: grid_size=%d, grid_n_height=%d, grid_n_width=%d, length=%d",
                  grid_size, grid_n_height, grid_n_width, length);
        raycast_en = false;
        return;
    }

    border_flag.clear();
    rays_with_sample_points.clear();
    border_flag.resize(num_cameras);
    rays_with_sample_points.resize(num_cameras);

    ROS_INFO("[ VIO ] Initializing raycast for %d cameras: grid_size=%d, grid_n_height=%d, grid_n_width=%d, length=%d",
             num_cameras, grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    int total_sample_points = 0;
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        

        border_flag[cam_idx].resize(length, 0);
        rays_with_sample_points[cam_idx].clear();
        rays_with_sample_points[cam_idx].reserve(length);

        int camera_sample_points = 0;
        for (int grid_row = 1; grid_row <= grid_n_height; grid_row++) {
            for (int grid_col = 1; grid_col <= grid_n_width; grid_col++) {
                std::vector<V3D> SamplePointsEachGrid;
                int index = (grid_row - 1) * grid_n_width + grid_col - 1;

                if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width)
                    border_flag[cam_idx][index] = 1;

                int u = grid_size / 2 + (grid_col - 1) * grid_size;
                int v = grid_size / 2 + (grid_row - 1) * grid_size;

                for (float d_temp = d_min; d_temp <= d_max; d_temp += step) {
                    V3D xyz;
                    try {
                        xyz = cams[cam_idx]->cam2world(u, v);
                        xyz *= d_temp / xyz[2];
                        SamplePointsEachGrid.push_back(xyz);
                        camera_sample_points++;
                    } catch (const std::exception& e) {
                        ROS_ERROR("[ VIO ] Exception in camera %d raycast computation: %s", cam_idx, e.what());
                    }
                }
                rays_with_sample_points[cam_idx].push_back(SamplePointsEachGrid);
            }
        }
        total_sample_points += camera_sample_points;
    }

    ROS_INFO("[ VIO ] Raycast initialization completed! Total sample points: %d", total_sample_points);
}
void VIOManager::resetGrid() {
    fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
    fill(map_index.begin(), map_index.end(), 0);
    fill(map_dist.begin(), map_dist.end(), 10000.0f);
    fill(update_flag.begin(), update_flag.end(), 0);
    fill(scan_value.begin(), scan_value.end(), 0.0f);

    retrieve_voxel_points.clear();
    retrieve_voxel_points.resize(length);

    append_voxel_points.clear();
    append_voxel_points.resize(length);

    total_points = 0;
}



void VIOManager::computeProjectionJacobian(int cam_idx, V3D p, MD(2, 3) &J)
{
    double fx_i = cams[cam_idx]->fx();
    double fy_i = cams[cam_idx]->fy();
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1.0 / p[2];
    const double z_inv_2 = z_inv * z_inv;

    J(0, 0) = fx_i * z_inv;   
    J(0, 1) = 0.0;            
    J(0, 2) = -fx_i * x * z_inv_2;  

    J(1, 0) = 0.0;
    J(1, 1) = fy_i * z_inv;
    J(1, 2) = -fy_i * y * z_inv_2;
}


void VIOManager::getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level) {
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale = (1 << level);
    const int u_ref_i = floorf(pc[0] / scale) * scale;
    const int v_ref_i = floorf(pc[1] / scale) * scale;
    const float subpix_u_ref = (u_ref - u_ref_i) / scale;
    const float subpix_v_ref = (v_ref - v_ref_i) / scale;
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    for (int x = 0; x < patch_size; x++) {
        uint8_t *img_ptr = (uint8_t *) img.data + (v_ref_i - patch_size_half * scale + x * scale) * width +
                           (u_ref_i - patch_size_half * scale);
        for (int y = 0; y < patch_size; y++, img_ptr += scale) {
            patch_tmp[patch_size_total * level + x * patch_size + y] =
                    w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                    w_ref_br * img_ptr[scale * width + scale];
        }
    }
}

void VIOManager::insertPointIntoVoxelMap(VisualPoint *pt_new) {
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
        loc_xyz[j] = pt_w[j] / voxel_size;
        if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
        iter->second->voxel_points.push_back(pt_new);
        iter->second->count++;
    } else {
        VOXEL_POINTS *ot = new VOXEL_POINTS(0);
        ot->creation_timestamp_ = current_timestamp_;  
        ot->voxel_points.push_back(pt_new);
        feat_map[position] = ot;
    }
}

void VIOManager::getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref, const V3D &xyz_ref,
                                               const V3D &normal_ref,
                                               const SE3 &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref) {
    const V3D t = T_cur_ref.inverse().translation();
    const Eigen::Matrix3d H_cur_ref =
            T_cur_ref.rotation_matrix() *
            (normal_ref.dot(xyz_ref) * Eigen::Matrix3d::Identity() - t * normal_ref.transpose());
    const int kHalfPatchSize = 4;
    V3D f_du_ref(cam.cam2world(px_ref + Eigen::Vector2d(kHalfPatchSize, 0) * (1 << level_ref)));
    V3D f_dv_ref(cam.cam2world(px_ref + Eigen::Vector2d(0, kHalfPatchSize) * (1 << level_ref)));
    const V3D f_cur(H_cur_ref * xyz_ref);
    const V3D f_du_cur = H_cur_ref * f_du_ref;
    const V3D f_dv_cur = H_cur_ref * f_dv_ref;
    V2D px_cur(cam.world2cam(f_cur));
    V2D px_du_cur(cam.world2cam(f_du_cur));
    V2D px_dv_cur(cam.world2cam(f_dv_cur));
    A_cur_ref.col(0) = (px_du_cur - px_cur) / kHalfPatchSize;
    A_cur_ref.col(1) = (px_dv_cur - px_cur) / kHalfPatchSize;
}

void VIOManager::getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref,
                                     const double depth_ref,
                                     const SE3 &T_cur_ref, const int level_ref, const int pyramid_level,
                                     const int halfpatch_size,
                                     Matrix2d &A_cur_ref) {
    const Vector3d xyz_ref(f_ref * depth_ref);
    Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size, 0) * (1 << level_ref) * (1 << pyramid_level)));
    Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0, halfpatch_size) * (1 << level_ref) * (1 << pyramid_level)));
    xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
    const Vector2d px_cur(cam.world2cam(T_cur_ref * (xyz_ref)));
    const Vector2d px_du(cam.world2cam(T_cur_ref * (xyz_du_ref)));
    const Vector2d px_dv(cam.world2cam(T_cur_ref * (xyz_dv_ref)));
    A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

void
VIOManager::warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref,
                       const int search_level,
                       const int pyramid_level, const int halfpatch_size, float *patch) {
    const int patch_size = halfpatch_size * 2;
    const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
    if (isnan(A_ref_cur(0, 0))) {
        printf("Affine warp is NaN, probably camera has no translation\n"); 
        return;
    }

    float *patch_ptr = patch;
    for (int y = 0; y < patch_size; ++y) {
        for (int x = 0; x < patch_size; ++x) 
        {
            Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
            px_patch *= (1 << search_level);
            px_patch *= (1 << pyramid_level);
            const Vector2f px(A_ref_cur * px_patch + px_ref.cast<float>());
            if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 || px[1] >= img_ref.rows - 1)
                patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = 0;
            else
                patch_ptr[patch_size_total * pyramid_level + y * patch_size + x] = (float) vk::interpolateMat_8u(
                        img_ref, px[0], px[1]);
        }
    }
}

int VIOManager::getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level) {
    int search_level = 0;
    double D = A_cur_ref.determinant();
    while (D > 3.0 && search_level < max_level) {
        search_level += 1;
        D *= 0.25;
    }
    return search_level;
}

double VIOManager::calculateNCC(float *ref_patch, float *cur_patch, int patch_size) {
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref = sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr = sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}
bool VIOManager::isRealCrossCameraPoint(VisualPoint* pt, int current_cam_id) {
    
    
    
    
    
    
    
    
    

    if (!pt || !pt->ref_patch) return false;

    
    CrossCameraData& data = pt->cross_cam_data_;

    
    int ref_cam_id = pt->ref_patch->cam_id_;

    
    int cur_cam_id = current_cam_id;

    
    if (ref_cam_id == cur_cam_id) return false;

    
    
    if (data.has_migration_history && data.migration_source_cam == ref_cam_id) {
        return true;
    }

    
    
    for (auto obs : pt->obs_) {
        if (obs->cam_id_ == ref_cam_id && obs->cam_id_ != cur_cam_id) {
            
            return true;
        }
    }

    return false;
}

void VIOManager::updateCrossCameraHistory(VisualPoint* pt, int cam_id) {
    if (!pt) return;

    
    CrossCameraData& data = pt->cross_cam_data_;
    if (data.primary_cam_idx != -1 && data.primary_cam_idx != cam_id) {
        
        if (!data.currently_visible.test(data.primary_cam_idx) && data.currently_visible.test(cam_id)) {
            data.cross_camera_migrations++;
        }
    }
    data.primary_cam_idx = cam_id;
}
void VIOManager::retrieveFromVisualSparseMap(const std::vector<cv::Mat> imgs,
                                             vector<pointWithVar> &pg,
                                             const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map) {
    
    if (feat_map.empty()) {
        ROS_WARN_THROTTLE(5.0, "[VIO] feat_map is empty, skipping visual point retrieval");
        return;
    }
    if (camera_photo_params.empty()) {
        initializeCameraPhotoParams();
    }

    double ts0 = omp_get_wtime();
    visual_submap->reset();
    sub_feat_map.clear();

    
    

    float voxel_size = 0.5f;
    if (!normal_en)
        warp_map.clear();

    
    if (depth_imgs_buffer_.size() != cams.size()) {
        depth_imgs_buffer_.resize(cams.size());
        for (size_t i = 0; i < cams.size(); i++) {
            depth_imgs_buffer_[i] = cv::Mat::zeros(height, width, CV_32FC1);
        }
    } else {
        for (auto& depth_img : depth_imgs_buffer_) {
            depth_img.setTo(0);  
        }
    }
    std::vector<cv::Mat>& depth_imgs = depth_imgs_buffer_;

    
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    int cells_per_camera = grid_n_width * grid_n_height;
    int num_cameras = (int)cams.size();
    int total_cells = cells_per_camera * num_cameras;  

    
    std::vector<int> camera_grid_offset(num_cameras);
    for (int i = 0; i < num_cameras; i++) {
        camera_grid_offset[i] = i * cells_per_camera;
    }

    grid_num.assign(total_cells, TYPE_UNKNOWN);
    map_dist.assign(total_cells, std::numeric_limits<float>::max());
    retrieve_voxel_points.assign(total_cells, nullptr);
    scan_value.assign(total_cells, 0.0f);  

    
    
    if (retrieve_voxel_points_list_buffer_.size() != total_cells) {
        retrieve_voxel_points_list_buffer_.resize(total_cells);
    }
    for (auto& set : retrieve_voxel_points_list_buffer_) {
        set.clear();  
    }
    std::vector<std::unordered_set<VisualPoint*>>& retrieve_voxel_points_list = retrieve_voxel_points_list_buffer_;

    
    for (int i_point = 0; i_point < (int)pg.size(); i_point++) {
        V3D pt_w = pg[i_point].point_w;
        int loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = (int)std::floor(pt_w[j] / voxel_size);
            if (loc_xyz[j] < 0)
                loc_xyz[j] -= 1;
        }
        VOXEL_LOCATION position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        sub_feat_map[position] = 0;

        
#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int cam_idx = 0; cam_idx < (int)cams.size(); cam_idx++)
        {
            V3D pt_c = new_frame_->w2f(pt_w, cam_idx);
            if (pt_c[2] > 0)
            {
                V2D px = new_frame_->w2c(pt_w, cam_idx);
                if (cams[cam_idx]->isInFrame(px.cast<int>(), border))
                {
                    float depth = (float)pt_c[2];
                    int col = (int)px[0];
                    int row = (int)px[1];
                    if (row >= 0 && row < height && col >= 0 && col < width) {
                        float *it = (float*)depth_imgs[cam_idx].data;
                        it[row * width + col] = depth;  
                    }
                }
            }
        }
    }
    
    std::vector<VOXEL_LOCATION> DeleteKeyList;
    int total_visible_points = 0;  
    int total_projections = 0;     
    int filtered_behind_camera = 0;  
    int filtered_out_of_frame = 0;   
    int filtered_invalid_grid = 0;   
    std::map<int, int> grid_overwrite_count; 

    for (auto &iter_sf : sub_feat_map) {
        VOXEL_LOCATION position = iter_sf.first;
        auto corre_voxel = feat_map.find(position);
        if (corre_voxel != feat_map.end()) {
            bool voxel_in_fov = false;
            std::vector<VisualPoint*> &voxel_points = corre_voxel->second->voxel_points;

            for (VisualPoint* pt : voxel_points) {
                
                if (!pt || pt->obs_.empty())
                    continue;

                
                if (!pt->is_normal_initialized_)
                    continue;

                total_visible_points++;  

                
                if (enable_cross_camera_tracking) {
                    pt->cross_cam_data_.previously_visible = pt->cross_cam_data_.currently_visible;
                    pt->cross_cam_data_.currently_visible.reset();
                }

                bool this_pt_in_fov = false;
                
                for (int cam_idx = 0; cam_idx < (int)cams.size(); cam_idx++) {
                    V3D dir_c = new_frame_->w2f(pt->pos_, cam_idx);
                    if (dir_c[2] < 0) {
                        filtered_behind_camera++;
                        continue;
                    }

                    V2D pc = new_frame_->w2c(pt->pos_, cam_idx);
                    if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border)) {
                        filtered_out_of_frame++;
                        continue;
                    }

                    this_pt_in_fov = true;

                    
                    if (enable_cross_camera_tracking) {
                        pt->cross_cam_data_.currently_visible.set(cam_idx);
                    }

                    
                    int grid_col = (int)(pc[0] / grid_size);
                    int grid_row = (int)(pc[1] / grid_size);
                    int local_index = grid_row * grid_n_width + grid_col;

                    
                    int index = camera_grid_offset[cam_idx] + local_index;

                    if (local_index < 0 || local_index >= cells_per_camera) {
                        filtered_invalid_grid++;
                        continue;
                    }

                    total_projections++;  

                    grid_num[index] = TYPE_MAP;

                    
                    
                    retrieve_voxel_points_list[index].insert(pt);

                    
                    V3D obs_vec = new_frame_->pos(cam_idx) - pt->pos_;
                    float cur_dist = (float)obs_vec.norm();
                    if (cur_dist <= map_dist[index]) {
                        map_dist[index] = cur_dist;
                        retrieve_voxel_points[index] = pt;
                    }
                }

                if (this_pt_in_fov)
                    voxel_in_fov = true;
            }

            if (!voxel_in_fov)
                DeleteKeyList.push_back(position);
        }
    }
    for (auto &key : DeleteKeyList) {
        sub_feat_map.erase(key);
    }

    
    int total_points_in_grids = 0;
    int max_points_in_grid = 0;
    int grids_with_multiple_points = 0;
    for (int i = 0; i < total_cells; i++) {
        int count = (int)retrieve_voxel_points_list[i].size();
        total_points_in_grids += count;
        if (count > max_points_in_grid) {
            max_points_in_grid = count;
        }
        if (count > 1) {
            grids_with_multiple_points++;
        }
    }

    
    
    int filter_not_map = 0;
    int filter_no_pt = 0;
    int filter_no_normal = 0;
    int filter_no_valid_cam = 0;
    int filter_depth_continuous = 0;
    int filter_no_ref_ftr = 0;
    int filter_ncc_fail = 0;
    int filter_error_too_large = 0;
    int success_count = 0;

    
    struct CandidatePoint {
        VisualPoint* pt;
        float error;
        int search_level;
        Matrix2d A_cur_ref_zero;
        std::vector<float> patch_wrap;
        Feature* ref_ftr;
        V2D pc;
    };

    
    
    
    int num_threads = 1;
#ifdef MP_EN
    num_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<std::unordered_set<VisualPoint*>>> thread_camera_processed_points(num_threads);
    std::vector<std::vector<std::vector<CandidatePoint>>> thread_camera_candidates(num_threads);

    
    std::vector<std::vector<float>> thread_patch_wrap_buffer(num_threads);
    std::vector<std::vector<float>> thread_patch_buffer_buffer(num_threads);

    for (int t = 0; t < num_threads; t++) {
        thread_camera_processed_points[t].resize(num_cameras);  
        thread_camera_candidates[t].resize(num_cameras);
        thread_patch_wrap_buffer[t].resize(warp_len);  
        thread_patch_buffer_buffer[t].resize(patch_size_total);  
    }

    
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for schedule(dynamic, 16) \
    reduction(+:filter_not_map, filter_no_pt, filter_no_normal, filter_no_valid_cam, \
              filter_depth_continuous, filter_no_ref_ftr, filter_ncc_fail, \
              filter_error_too_large, success_count)
#endif
    for (int i = 0; i < total_cells; i++) {
        int thread_id = 0;
#ifdef MP_EN
        thread_id = omp_get_thread_num();
#endif
        std::vector<std::vector<CandidatePoint>>& camera_candidates = thread_camera_candidates[thread_id];
        std::vector<std::unordered_set<VisualPoint*>>& camera_processed_points = thread_camera_processed_points[thread_id];

        
        std::vector<float>& patch_wrap = thread_patch_wrap_buffer[thread_id];
        std::vector<float>& patch_buffer_local = thread_patch_buffer_buffer[thread_id];

        if (grid_num[i] != TYPE_MAP) {
            filter_not_map++;
            continue;
        }

        
        int cam_idx = i / cells_per_camera;  

        
        for (auto pt : retrieve_voxel_points_list[i]) {
            if (!pt) {
                filter_no_pt++;
                continue;
            }

            
            if (camera_processed_points[cam_idx].find(pt) != camera_processed_points[cam_idx].end()) {
                continue;
            }
            camera_processed_points[cam_idx].insert(pt);

            
            if (!pt->is_normal_initialized_) {
                filter_no_normal++;
                continue;
            }

        
        V2D pc = new_frame_->w2c(pt->pos_, cam_idx);

        
        if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border)) {
            filter_no_valid_cam++;
            continue;
        }

        
        V3D pt_cam = new_frame_->w2f(pt->pos_, cam_idx);
        bool depth_continous = false;
        float *depth_ptr = reinterpret_cast<float*>(depth_imgs[cam_idx].data);

        
        int base_col = int(pc[0]) - patch_size_half;
        int base_row = int(pc[1]) - patch_size_half;
        int center_offset = patch_size_half;
        float pt_depth = static_cast<float>(pt_cam[2]);

        
        
        const int sample_offsets[][2] = {
            {0, 0}, {patch_size, 0}, {0, patch_size}, {patch_size, patch_size},  
            {center_offset, 0}, {patch_size, center_offset},  
            {0, center_offset}, {center_offset, patch_size}   
        };

        for (int i = 0; i < 8; i++) {
            int col = base_col + sample_offsets[i][0];
            int row = base_row + sample_offsets[i][1];
            if (row < 0 || row >= height || col < 0 || col >= width) continue;
            if (i == 0) continue;  

            float depth = depth_ptr[row * width + col];
            if (depth == 0.) continue;
            if (abs(pt_depth - depth) > 0.5) {
                depth_continous = true;
                break;
            }
        }
        if (depth_continous) {
            filter_depth_continuous++;
            continue;
        }

        
        int previous_cam_idx = pt->cross_cam_data_.previous_cam_idx;

        
        Feature *ref_ftr = nullptr;
        bool camera_migrated = (previous_cam_idx != -1 && previous_cam_idx != cam_idx);

        if (normal_en) {
            
            
            
            

            if (pt->obs_.size() == 1) {
                
                ref_ftr = *pt->obs_.begin();
                pt->ref_patch = ref_ftr;
                pt->has_ref_patch_ = true;
            }
            else if (pt->has_ref_patch_ && pt->ref_patch) {
                
                int num_cameras = cams.size();
                if (num_cameras == 1) {
                    
                    ref_ftr = pt->ref_patch;
                } else {
                    
                    bool found_current_cam = false;
                    for (auto it = pt->obs_.begin(); it != pt->obs_.end(); ++it) {
                        if ((*it)->cam_id_ == cam_idx) {
                            ref_ftr = *it;
                            found_current_cam = true;
                            break;
                        }
                    }
                    
                    if (!found_current_cam) {
                        ref_ftr = pt->ref_patch;
                    } else {
                        
                        pt->ref_patch = ref_ftr;
                    }
                }
            }
            else {
                
                
                for (auto it = pt->obs_.begin(); it != pt->obs_.end(); ++it) {
                    if ((*it)->cam_id_ == cam_idx) {
                        ref_ftr = *it;
                        break;
                    }
                }

                
                
                if (!ref_ftr && pt->obs_.size() > 0) {
                    
                    if (pt->obs_.size() <= 3) {
                        
                        ref_ftr = *pt->obs_.begin();
                    } else {
                        
                        float photometric_errors_min = std::numeric_limits<float>::max();
                        int compare_count = 0;
                        const int max_compare = 2;  

                        for (auto it = pt->obs_.begin(); it != pt->obs_.end() && compare_count < max_compare + 1; ++it, ++compare_count) {
                            Feature* ref_patch_temp = *it;
                            float *patch_temp = ref_patch_temp->patch_;
                            float photometric_errors = 0.0f;
                            int sample_count = 0;

                            
                            auto itm = pt->obs_.begin();
                            for (int i = 0; i < max_compare && itm != pt->obs_.end(); ++itm, ++i) {
                                if ((*itm)->id_ == ref_patch_temp->id_)
                                    continue;
                                float *patch_cache = (*itm)->patch_;
                                Eigen::Map<Eigen::VectorXf> patch_a(patch_temp, patch_size_total);
                                Eigen::Map<Eigen::VectorXf> patch_b(patch_cache, patch_size_total);
                                photometric_errors += (patch_a - patch_b).squaredNorm();
                                sample_count++;
                            }
                            if (sample_count > 0)
                                photometric_errors /= (float)sample_count;

                            if (photometric_errors < photometric_errors_min) {
                                photometric_errors_min = photometric_errors;
                                ref_ftr = ref_patch_temp;
                            }
                        }
                    }
                }

                pt->ref_patch = ref_ftr;
                pt->has_ref_patch_ = (ref_ftr != nullptr);
            }
        } else {
            if (!pt->getCloseViewObs(new_frame_->pos(cam_idx), ref_ftr, pc)) {
                filter_no_ref_ftr++;
                continue;
            }
        }
        if (!ref_ftr) {
            filter_no_ref_ftr++;
            continue;
        }

        
        int search_level;
        Matrix2d A_cur_ref_zero;
        if (normal_en) {
            V3D norm_vec = (ref_ftr->T_f_w_.rotation_matrix() * pt->normal_).normalized();
            V3D pf = ref_ftr->T_f_w_ * pt->pos_;
            SE3 T_cur_ref = new_frame_->T_f_w_[cam_idx] * ref_ftr->T_f_w_.inverse();
            getWarpMatrixAffineHomography(*cams[cam_idx], ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref_zero);
            search_level = getBestSearchLevel(A_cur_ref_zero, 2);
        } else {
            auto iter_warp = warp_map.find(ref_ftr->id_);
            if (iter_warp != warp_map.end()) {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            } else {
                getWarpMatrixAffine(*cams[cam_idx], ref_ftr->px_, ref_ftr->f_,
                                    (ref_ftr->pos() - pt->pos_).norm(),
                                    new_frame_->T_f_w_[cam_idx] * ref_ftr->T_f_w_.inverse(),
                                    ref_ftr->level_, 0, patch_size_half, A_cur_ref_zero);
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);
                Warp *ot = new Warp(search_level, A_cur_ref_zero);
                warp_map[ref_ftr->id_] = ot;
            }
        }

        
        for (int pyramid_level = 0; pyramid_level <= patch_pyrimid_level - 1; pyramid_level++) {
            warpAffine(A_cur_ref_zero, ref_ftr->img_, ref_ftr->px_, ref_ftr->level_,
                       search_level, pyramid_level, patch_size_half, patch_wrap.data());
        }
        getImagePatch(imgs[cam_idx], pc, patch_buffer_local.data(), 0);

        
        
        float error = 0.0;
        if (exposure_estimate_en) {
            
            double cur_inv_expo = (cam_idx < state->inv_expo_time_per_cam.size())
                                ? state->inv_expo_time_per_cam[cam_idx] : 1.0;
            
            for (int ind = 0; ind < patch_size_total; ind++) {
                float diff = ref_ftr->inv_expo_time_ * patch_wrap[ind] -
                             cur_inv_expo * patch_buffer_local[ind];
                error += diff * diff;
            }
        } else {
            
            for (int ind = 0; ind < patch_size_total; ind++) {
                float diff = patch_wrap[ind] - patch_buffer_local[ind];
                error += diff * diff;
            }
        }

        if (ncc_en) {
            double ncc = calculateNCC(patch_wrap.data(), patch_buffer_local.data(), patch_size_total);
            if (ncc < ncc_thre) {
                filter_ncc_fail++;
                continue;
            }
        }

        
        bool is_cross_camera = (ref_ftr && ref_ftr->cam_id_ != cam_idx);
        float threshold_multiplier = 1.0f;  
        float error_threshold = outlier_threshold * patch_size_total * threshold_multiplier;

        if (error > error_threshold) {
            filter_error_too_large++;
            continue;
        }

        
        CandidatePoint candidate;
        candidate.pt = pt;
        candidate.error = error;
        candidate.search_level = search_level;
        candidate.A_cur_ref_zero = A_cur_ref_zero;
        candidate.patch_wrap = patch_wrap;
        candidate.ref_ftr = ref_ftr;
        candidate.pc = pc;
        camera_candidates[cam_idx].push_back(candidate);
        success_count++;

        
        CrossCameraData& cc_data = pt->cross_cam_data_;

        
        cc_data.currently_visible.set(cam_idx);

        
        
        if (cc_data.primary_cam_idx != -1 && cc_data.primary_cam_idx != cam_idx) {
            
            
            if (!cc_data.has_migration_history || cc_data.migration_source_cam != cc_data.primary_cam_idx) {
                cc_data.migration_source_cam = cc_data.primary_cam_idx;
                cc_data.has_migration_history = true;
                cc_data.cross_camera_migrations++;
            }
        }

        cc_data.previous_cam_idx = cam_idx;
        cc_data.primary_cam_idx = cam_idx;  
        } 
    } 

    
    
    std::vector<std::vector<CandidatePoint>> per_camera_candidates(num_cameras);
    for (int t = 0; t < num_threads; t++) {
        for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
            per_camera_candidates[cam_idx].insert(per_camera_candidates[cam_idx].end(),
                                                  thread_camera_candidates[t][cam_idx].begin(),
                                                  thread_camera_candidates[t][cam_idx].end());
        }
    }

    
    
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        size_t n_candidates = per_camera_candidates[cam_idx].size();
        if (n_candidates == 0) continue;

        
        size_t n_to_sort = std::min(n_candidates, (size_t)points_per_camera_max);

        std::partial_sort(per_camera_candidates[cam_idx].begin(),
                         per_camera_candidates[cam_idx].begin() + n_to_sort,
                         per_camera_candidates[cam_idx].end(),
                         [](const CandidatePoint &a, const CandidatePoint &b) {
                             return a.error < b.error;
                         });
    }

















    
    std::vector<int> points_per_camera(num_cameras, 0);
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        int n_to_keep = std::min((int)per_camera_candidates[cam_idx].size(), points_per_camera_min);
        for (int i = 0; i < n_to_keep; i++) {
            const auto &cand = per_camera_candidates[cam_idx][i];
            visual_submap->voxel_points.push_back(cand.pt);
            visual_submap->propa_errors.push_back(cand.error);
            visual_submap->search_levels.push_back(cand.search_level);
            visual_submap->errors.push_back(cand.error);
            visual_submap->warp_patch.push_back(cand.patch_wrap);
            visual_submap->inv_expo_list.push_back(cand.ref_ftr->inv_expo_time_);
            visual_submap->camera_ids.push_back(cam_idx);  
            points_per_camera[cam_idx]++;
        }
        
        per_camera_candidates[cam_idx].erase(per_camera_candidates[cam_idx].begin(),
                                             per_camera_candidates[cam_idx].begin() + n_to_keep);
    }

    
    std::vector<std::pair<CandidatePoint, int>> remaining_candidates;  
    remaining_candidates.reserve(max_total_points);  
    for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
        
        int max_remaining = points_per_camera_max - points_per_camera[cam_idx];
        int n_remaining = std::min((int)per_camera_candidates[cam_idx].size(), max_remaining);
        for (int i = 0; i < n_remaining; i++) {
            remaining_candidates.push_back({per_camera_candidates[cam_idx][i], cam_idx});
        }
    }

    
    std::sort(remaining_candidates.begin(), remaining_candidates.end(),
              [](const auto &a, const auto &b) {
                  return a.first.error < b.first.error;
              });

    
    int current_total = visual_submap->voxel_points.size();
    int remaining_quota = max_total_points - current_total;
    int n_add = std::min((int)remaining_candidates.size(), remaining_quota);

    for (int i = 0; i < n_add; i++) {
        const auto &cand = remaining_candidates[i].first;
        int cam_idx = remaining_candidates[i].second;
        visual_submap->voxel_points.push_back(cand.pt);
        visual_submap->propa_errors.push_back(cand.error);
        visual_submap->search_levels.push_back(cand.search_level);
        visual_submap->errors.push_back(cand.error);
        visual_submap->warp_patch.push_back(cand.patch_wrap);
        visual_submap->inv_expo_list.push_back(cand.ref_ftr->inv_expo_time_);
        visual_submap->camera_ids.push_back(cam_idx);  
        points_per_camera[cam_idx]++;
    }

    int total_selected = visual_submap->voxel_points.size();
    total_points = total_selected;  



    if (raycast_en) {
        float voxel_size = 0.5f;
        int loc_xyz[3];
        int total_raycast_points = 0;
        int total_found_features = 0;
        
        for (int cam_idx = 0; cam_idx < num_cameras; cam_idx++) {
            int cam_offset = camera_grid_offset[cam_idx];
            for (int local_i = 0; local_i < cells_per_camera; local_i++) {
                int i = cam_offset + local_i;
                if (grid_num[i] == TYPE_MAP) continue;

                
                bool is_border = false;
                if (cam_idx < border_flag.size() && local_i < border_flag[cam_idx].size() &&
                    border_flag[cam_idx][local_i] == 1) {
                    is_border = true;
                }
                if (is_border) continue;

                
                bool found_feature = false;
                if (cam_idx >= rays_with_sample_points.size()) {
                    continue;
                }

                if (local_i >= rays_with_sample_points[cam_idx].size()) {
                    continue;
                }

                for (const auto &sample_point_cam : rays_with_sample_points[cam_idx][local_i]) {
                    total_raycast_points++;
                    V3D sample_point_w = new_frame_->f2w(sample_point_cam, cam_idx);

                    for (int j = 0; j < 3; j++) {
                        loc_xyz[j] = floor(sample_point_w[j] / voxel_size);
                        if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
                    }

                    VOXEL_LOCATION sample_pos(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

                    auto corre_sub_feat_map = sub_feat_map.find(sample_pos);
                    if (corre_sub_feat_map != sub_feat_map.end()) break;

                    auto corre_feat_map = feat_map.find(sample_pos);
                    if (corre_feat_map != feat_map.end()) {
                        std::vector<VisualPoint *> &voxel_points = corre_feat_map->second->voxel_points;
                        int voxel_num = voxel_points.size();
                        if (voxel_num == 0) continue;

                        for (int j = 0; j < voxel_num; j++) {
                            VisualPoint *pt = voxel_points[j];
                            if (pt == nullptr) continue;
                            if (pt->obs_.size() == 0) continue;
                            bool point_visible = false;
                            for (int check_cam = 0; check_cam < cams.size(); check_cam++) {
                                V3D dir = new_frame_->w2f(pt->pos_, check_cam);
                                if (dir[2] < 0) continue;

                                V2D pc = new_frame_->w2c(pt->pos_, check_cam);
                                if (new_frame_->cams_[check_cam]->isInFrame(pc.cast<int>(), border)) {
                                    point_visible = true;
                                    total_found_features++;

                                    grid_num[i] = TYPE_MAP;

                                    Vector3d obs_vec = new_frame_->pos(check_cam) - pt->pos_;
                                    float cur_dist = obs_vec.norm();
                                    if (cur_dist <= map_dist[i]) {
                                        map_dist[i] = cur_dist;
                                        retrieve_voxel_points[i] = pt;
                                    }
                                    break;
                                }
                            }
                            if (point_visible) {
                                found_feature = true;
                                break;
                            }
                        }

                        if (found_feature) {
                            sub_feat_map[sample_pos] = 0;
                            break;
                        }
                    } else {
                        auto iter = plane_map.find(sample_pos);
                        if (iter != plane_map.end()) {
                            VoxelOctoTree *current_octo;
                            current_octo = iter->second->find_correspond(sample_point_w);
                            if (current_octo->plane_ptr_->is_plane_) {
                                pointWithVar plane_center;
                                VoxelPlane &plane = *current_octo->plane_ptr_;
                                plane_center.point_w = plane.center_;
                                plane_center.normal = plane.normal_;
                                visual_submap->add_from_voxel_map.push_back(plane_center);
                                found_feature = true;
                                break;
                            }
                        }
                    }
                }
                if (found_feature) break;
            } 
        } 
    }




}

double VIOManager::calculateCoVarianceScale(double realtime_avg_error, int realtime_n_meas) {
    
    if (frame_count < dynamic_cov_warmup_frames) {
        return warmup_cov_scale;
    }

    
    
    double t = (realtime_avg_error - 1.0) / (dynamic_cov_error_max - 1.0);
    t = std::min(1.0, std::max(0.0, t));  
    double cov_scale = min_cov_scale + t * (max_cov_scale - min_cov_scale);

    
    cov_scale = std::max(min_cov_scale, std::min(max_cov_scale, cov_scale));

    
    double alpha = 0.6;  
    double smoothed_scale = alpha * cov_scale + (1.0 - alpha) * prev_cov_scale_;
    prev_cov_scale_ = smoothed_scale;

    
    if (frame_count % 30 == 0 || smoothed_scale > min_cov_scale * 1.5) {
        ROS_INFO("[DynCov] frame=%d, MSE=%.1f, cov=%.1f (range: %.1f~%.1f)",
                 frame_count, realtime_avg_error, smoothed_scale, min_cov_scale, max_cov_scale);
    }

    return smoothed_scale;
}

double VIOManager::calculateCoVarianceScalePerCam(int cam_idx, double realtime_avg_error, int realtime_n_meas) {
    
    if (frame_count < dynamic_cov_warmup_frames) {
        return warmup_cov_scale;
    }

    
    
    const int min_required_meas = 5;
    if (realtime_n_meas < min_required_meas) {




        return max_cov_scale * 10.0;  
    }

    
    double t = (realtime_avg_error - 1.0) / (dynamic_cov_error_max - 1.0);
    t = std::min(1.0, std::max(0.0, t));
    double cov_scale = min_cov_scale + t * (max_cov_scale - min_cov_scale);

    
    cov_scale = std::max(min_cov_scale, std::min(max_cov_scale, cov_scale));

    
    double alpha = 0.6;
    double smoothed_scale = alpha * cov_scale + (1.0 - alpha) * prev_cov_scale_per_cam_[cam_idx];
    prev_cov_scale_per_cam_[cam_idx] = smoothed_scale;

    return smoothed_scale;
}

void VIOManager::computeJacobianAndUpdateEKF(const std::vector<cv::Mat> imgs)
{
    if (total_points < 2) {
        compute_jacobian_time = update_ekf_time = 0.0;
        return;  
    }
    double original_img_point_cov = img_point_cov;
    if (enable_dynamic_covariance_) {
        
        double cov_scale = calculateCoVarianceScale(prev_avg_error_, prev_n_meas_);
        img_point_cov = cov_scale;  
    }

    compute_jacobian_time = update_ekf_time = 0.0;
    for (int level = patch_pyrimid_level - 1; level >= 0; level--)
    {
        if (inverse_composition_en)
        {
            updateStateInverse(imgs, level);
        }
        else{
            updateState(imgs, level);
        }
    }
    state->cov -= G * state->cov;
    updateFrameState(*state);
    img_point_cov = original_img_point_cov;
}


void VIOManager::generateVisualMapPoints(const std::vector<cv::Mat>& imgs, std::vector<pointWithVar> &pg)
{
    if (pg.size() <= 10)
        return;

    
    int num_cameras = static_cast<int>(imgs.size());
    int cells_per_camera = grid_n_width * grid_n_height;
    int total_cells = cells_per_camera * num_cameras;

    std::vector<int> best_cam_idx(total_cells, -1);  
    if (append_voxel_points.size() != total_cells) {
        ROS_WARN("Resizing append_voxel_points from %zu to %d", append_voxel_points.size(), total_cells);
        append_voxel_points.resize(total_cells);
    }

    
    std::vector<int> camera_grid_offset(num_cameras);
    for (int i = 0; i < num_cameras; i++) {
        camera_grid_offset[i] = i * cells_per_camera;
    }

    for (size_t i = 0; i < pg.size(); i++) {
        if (pg[i].normal == V3D(0, 0, 0))
            continue;

        V3D pt = pg[i].point_w;
        for (size_t cam_idx = 0; cam_idx < imgs.size(); cam_idx++) {
            V2D pc = new_frame_->w2c(pt, cam_idx);
            if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border))
                continue;
            int grid_col = static_cast<int>(pc[0] / grid_size);
            int grid_row = static_cast<int>(pc[1] / grid_size);
            if (grid_col < 0 || grid_col >= grid_n_width ||
                grid_row < 0 || grid_row >= grid_n_height)
                continue;

            
            int local_index = grid_row * grid_n_width + grid_col;
            int index = camera_grid_offset[cam_idx] + local_index;

            if (index < 0 || index >= total_cells) {
                ROS_WARN("Invalid grid index: %d (max: %d)", index, total_cells-1);
                continue;
            }
            if (grid_num[index] == TYPE_MAP)
                continue;
            float cur_value = vk::shiTomasiScore(imgs[cam_idx], pc[0], pc[1]);
            if (cur_value > scan_value[index]) {
                scan_value[index] = cur_value;
                append_voxel_points[index] = pg[i]; 
                best_cam_idx[index] = static_cast<int>(cam_idx);
                grid_num[index] = TYPE_POINTCLOUD;
            }
        } 
    } 
    int add_count = 0;
    for (int i = 0; i < total_cells; i++) {  
        if (grid_num[i] != TYPE_POINTCLOUD)
            continue;
        if (best_cam_idx[i] < 0 || best_cam_idx[i] >= imgs.size()) {
            ROS_WARN("Invalid camera index %d for grid %d", best_cam_idx[i], i);
            continue;
        }
        pointWithVar pt_var = append_voxel_points[i];
        V3D pt = pt_var.point_w;
        if (pt.norm() < 0.0001) {
            ROS_WARN("Skip point with near-zero position in grid %d", i);
            continue;
        }
        int cam_idx = best_cam_idx[i];
        V2D pc = new_frame_->w2c(pt, cam_idx);
        if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border)) {
            continue;
        }
        V3D pt_cam = new_frame_->w2f(pt, cam_idx);
        if (pt_cam[2] <= 0) {
            continue;
        }

        try {
            float *patch = new float[patch_size_total];
            getImagePatch(imgs[cam_idx], pc, patch, 0);
            VisualPoint *pt_new = new VisualPoint(pt);
            Vector3d f = cams[cam_idx]->cam2world(pc);
            Feature *ftr_new = new Feature(pt_new, patch, pc, f, new_frame_->T_f_w_[cam_idx], 0, cam_idx);
            ftr_new->cam_id_ = cam_idx;
            ftr_new->img_ = imgs[cam_idx];
            ftr_new->id_ = new_frame_->id_;
            
            ftr_new->inv_expo_time_ = (cam_idx < state->inv_expo_time_per_cam.size())
                                    ? state->inv_expo_time_per_cam[cam_idx] : 1.0;
            pt_new->addFrameRef(ftr_new);
            
            if (enable_cross_camera_tracking) {
                pt_new->cross_cam_data_.primary_cam_idx = cam_idx;  
                pt_new->cross_cam_data_.cross_camera_migrations = 0;
                pt_new->cross_cam_data_.currently_visible.reset();
                pt_new->cross_cam_data_.currently_visible.set(cam_idx);  
            }
            pt_new->covariance_ = pt_var.var;
            pt_new->is_normal_initialized_ = true;
            V3D norm_vec = new_frame_->T_f_w_[cam_idx].rotation_matrix() * pt_var.normal;
            V3D dir = new_frame_->T_f_w_[cam_idx] * pt;
            if (dir.dot(norm_vec) < 0)
                pt_new->normal_ = -pt_var.normal;
            else
                pt_new->normal_ = pt_var.normal;
            pt_new->previous_normal_ = pt_new->normal_;
            insertPointIntoVoxelMap(pt_new);
            add_count++;
        }
        catch (const std::exception& e) {
            ROS_ERROR("Exception in generateVisualMapPoints: %s", e.what());
            continue;
        }
    }

    
}

void VIOManager::updateVisualMapPoints(std::vector<cv::Mat> &imgs)
{
    if (total_points == 0)
        return;

    int update_num = 0;
    for (int i = 0; i < total_points; i++)
    {
        VisualPoint* pt = visual_submap->voxel_points[i];
        if (pt == nullptr)
            continue;
        if (pt->is_converged_)
        {
            pt->deleteNonRefPatchFeatures();
            continue;
        }
        if (pt->obs_.empty())
            continue; 
        Feature *last_feature = pt->obs_.back();
        for (int cam_idx = 0; cam_idx < cams.size(); cam_idx++)
        {
            SE3 pose_cur = new_frame_->T_f_w_[cam_idx];
            V2D pc = new_frame_->w2c(pt->pos_, cam_idx);
            if (!cams[cam_idx]->isInFrame(pc.cast<int>(), border))
                continue;  
            SE3 pose_ref = last_feature->T_f_w_;
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();
            double cos_val = 0.5 * (delta_pose.rotation_matrix().trace() - 1);
            if (cos_val > 1.0)  cos_val = 1.0;
            if (cos_val < -1.0) cos_val = -1.0;
            double delta_theta = std::acos(cos_val);
            bool add_flag = false;
            if (delta_p > 0.5 || delta_theta > 0.3)
                add_flag = true;
            V2D last_px = last_feature->px_;
            double pixel_dist = (pc - last_px).norm();
            if (pixel_dist > 40.0)
                add_flag = true;
            if (pt->obs_.size() >= 30)
            {
                Feature *ref_ftr = nullptr;
                pt->findMinScoreFeature(new_frame_->pos(cam_idx), ref_ftr);
                if (ref_ftr) pt->deleteFeatureRef(ref_ftr);
            }
            if (add_flag)
            {
                float* patch_temp = new float[patch_size_total];
                getImagePatch(imgs[cam_idx], pc, patch_temp, 0);
                Vector3d f = cams[cam_idx]->cam2world(pc);
                int search_level = (visual_submap->search_levels.size() > (size_t)i)
                                   ? visual_submap->search_levels[i] : 0;

                Feature *ftr_new = new Feature(
                        pt,                         
                        patch_temp,                 
                        pc,                         
                        f,                          
                        new_frame_->T_f_w_[cam_idx],
                        search_level,
                        cam_idx
                );
                ftr_new->cam_id_ = cam_idx;
                ftr_new->img_           = imgs[cam_idx];
                ftr_new->id_            = new_frame_->id_;
                
                ftr_new->inv_expo_time_ = (cam_idx < state->inv_expo_time_per_cam.size())
                                        ? state->inv_expo_time_per_cam[cam_idx] : 1.0;
                pt->addFrameRef(ftr_new);
                update_num++;
                update_flag[i] = 1; 
            }
        } 
    } 

}

void VIOManager::updateReferencePatch(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree*> &plane_map)
{
    if (total_points == 0)
        return;
    for (int i = 0; i < (int)visual_submap->voxel_points.size(); i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];
        if (!pt->is_normal_initialized_ || pt->is_converged_ || pt->obs_.size() <= 5)
            continue;
        if (update_flag[i] == 0)
            continue;
        {
            const V3D &p_w = pt->pos_;
            float loc_xyz[3];
            for (int j = 0; j < 3; j++)
            {
                loc_xyz[j] = p_w[j] / 0.5;   
                if (loc_xyz[j] < 0)
                    loc_xyz[j] -= 1.0f;
            }
            VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

            auto it_plane = plane_map.find(position);
            if (it_plane != plane_map.end())
            {
                VoxelOctoTree *current_octo = it_plane->second->find_correspond(p_w);
                if (current_octo->plane_ptr_->is_plane_)
                {
                    VoxelPlane &plane = *current_octo->plane_ptr_;
                    float dis_to_plane =
                            plane.normal_.dot(p_w) + plane.d_;
                    float dis_to_plane_abs = std::fabs(dis_to_plane);

                    float dis_to_center = (plane.center_ - p_w).squaredNorm();
                    float range_dis = std::sqrt(dis_to_center - dis_to_plane*dis_to_plane);

                    if (range_dis <= 3.0f * plane.radius_)
                    {
                        Eigen::Matrix<double, 1, 6> J_nq;
                        J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
                        J_nq.block<1, 3>(0, 3) = -plane.normal_.transpose();
                        double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
                        sigma_l += plane.normal_.transpose() * pt->covariance_ * plane.normal_;

                        if (dis_to_plane_abs < 3.0 * std::sqrt(sigma_l))
                        {
                            if (pt->previous_normal_.dot(plane.normal_) < 0)
                                pt->normal_ = -plane.normal_;
                            else
                                pt->normal_ = plane.normal_;

                            double normal_update = (pt->normal_ - pt->previous_normal_).norm();
                            pt->previous_normal_ = pt->normal_;
                            if (normal_update < 0.0001 && pt->obs_.size() > 10)
                            {
                                pt->is_converged_ = true;
                            }
                        }
                    }
                }
            }
        }
        float score_max = -1e6;
        Feature *best_ref_ftr = nullptr;
        for (auto it = pt->obs_.begin(); it != pt->obs_.end(); ++it)
        {
            Feature *ref_patch_temp = *it;
            if (!ref_patch_temp)
                continue;
            float *patch_temp = ref_patch_temp->patch_;
            V3D pf = ref_patch_temp->T_f_w_ * pt->pos_;  
            V3D norm_vec = ref_patch_temp->T_f_w_.rotation_matrix() * pt->normal_;
            pf.normalize();
            double cos_angle = pf.dot(norm_vec);
            if (std::fabs(ref_patch_temp->mean_) < 1e-6)
            {
                float sum_val = std::accumulate(patch_temp, patch_temp + patch_size_total, 0.0f);
                float mean_val = sum_val / (float)patch_size_total;
                ref_patch_temp->mean_ = mean_val;
            }
            float ref_mean = ref_patch_temp->mean_;
            float sumNCC = 0.0f;
            int countNCC = 0;

            for (auto itm = pt->obs_.begin(); itm != pt->obs_.end(); ++itm)
            {
                if((*itm)->id_ == ref_patch_temp->id_)
                    continue;
                float *patch_cache = (*itm)->patch_;
                if (std::fabs((*itm)->mean_) < 1e-6)
                {
                    float sum_val2 = std::accumulate(patch_cache, patch_cache + patch_size_total, 0.0f);
                    (*itm)->mean_ = sum_val2 / (float)patch_size_total;
                }
                float other_mean = (*itm)->mean_;
                
                Eigen::Map<Eigen::VectorXf> patch_a(patch_temp, patch_size_total);
                Eigen::Map<Eigen::VectorXf> patch_b(patch_cache, patch_size_total);
                Eigen::VectorXf diff1 = (patch_a.cast<double>().array() - ref_mean).matrix().cast<float>();
                Eigen::VectorXf diff2 = (patch_b.cast<double>().array() - other_mean).matrix().cast<float>();
                double numerator = diff1.dot(diff2);
                double denominator1 = diff1.squaredNorm();
                double denominator2 = diff2.squaredNorm();
                double ncc_val = numerator / std::sqrt(denominator1*denominator2 + 1e-10);
                sumNCC += std::fabs(ncc_val);
                countNCC++;
            }
            float NCC_avg = (countNCC>0) ? (sumNCC / (float)countNCC) : 0.0f;
            float score = NCC_avg + (float)cos_angle;
            ref_patch_temp->score_ = score;
            if (score > score_max)
            {
                score_max     = score;
                best_ref_ftr  = ref_patch_temp;
            }
        } 
        if (best_ref_ftr)
        {
            pt->ref_patch = best_ref_ftr;
            pt->has_ref_patch_ = true;
        }
    } 
}
void VIOManager::projectPatchFromRefToCur(const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map)
{
    if (total_points == 0) return;

    int patch_size = 25;
    string dir = string(ROOT_DIR) + "Log/ref_cur_combine/";

    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat result_normal = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat result_dense = cv::Mat::zeros(height, width, CV_8UC1);

    cv::Mat img_photometric_error = new_frame_->imgs_[0].clone();

    uchar *it = (uchar *)result.data;
    uchar *it_normal = (uchar *)result_normal.data;
    uchar *it_dense = (uchar *)result_dense.data;

    struct pixel_member
    {
        Vector2f pixel_pos;
        uint8_t pixel_value;
    };

    int num = 0;
    for (int i = 0; i < visual_submap->voxel_points.size(); i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];

        if (pt->is_normal_initialized_)
        {
            Feature *ref_ftr;
            ref_ftr = pt->ref_patch;
            V2D pc(new_frame_->w2c(pt->pos_,0));
            V2D pc_prior(new_frame_->w2c_prior(pt->pos_,0));

            V3D norm_vec(ref_ftr->T_f_w_.rotation_matrix() * pt->normal_);
            V3D pf(ref_ftr->T_f_w_ * pt->pos_);

            if (pf.dot(norm_vec) < 0) norm_vec = -norm_vec;
            cv::Mat img_cur = new_frame_->imgs_[0];
            cv::Mat img_ref = ref_ftr->img_;

            SE3 T_cur_ref = new_frame_->T_f_w_[0] * ref_ftr->T_f_w_.inverse();
            Matrix2d A_cur_ref;
            getWarpMatrixAffineHomography(*cams[0], ref_ftr->px_, pf, norm_vec, T_cur_ref, 0, A_cur_ref);
            int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);

            double D = A_cur_ref.determinant();
            if (D > 3) continue;

            num++;

            cv::Mat ref_cur_combine_temp;
            int radius = 20;
            cv::hconcat(img_cur, img_ref, ref_cur_combine_temp);
            cv::cvtColor(ref_cur_combine_temp, ref_cur_combine_temp, CV_GRAY2BGR);

            getImagePatch(img_cur, pc, patch_buffer.data(), 0);

            float error_est = 0.0;
            float error_gt = 0.0;

            
            {
                
                double cur_inv_expo = (!state->inv_expo_time_per_cam.empty())
                                    ? state->inv_expo_time_per_cam[0] : 1.0;
                Eigen::Map<Eigen::VectorXf> patch_warp(visual_submap->warp_patch[i].data(), patch_size_total);
                Eigen::Map<Eigen::VectorXf> patch_buf(patch_buffer.data(), patch_size_total);
                Eigen::VectorXf diff = ref_ftr->inv_expo_time_ * patch_warp - cur_inv_expo * patch_buf;
                error_est = diff.squaredNorm();
            }
            std::string ref_est = "ref_est " + std::to_string(1.0 / ref_ftr->inv_expo_time_);
            double cur_inv_expo_str = (!state->inv_expo_time_per_cam.empty())
                                    ? state->inv_expo_time_per_cam[0] : 1.0;
            std::string cur_est = "cur_est " + std::to_string(1.0 / cur_inv_expo_str);
            std::string cur_propa = "cur_gt " + std::to_string(error_gt);
            std::string cur_optimize = "cur_est " + std::to_string(error_est);

            cv::putText(ref_cur_combine_temp, ref_est, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - 40, ref_ftr->px_[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4,
                        cv::Scalar(0, 255, 0), 1, 8, 0);

            cv::putText(ref_cur_combine_temp, cur_est, cv::Point2f(pc[0] - 40, pc[1] + 40), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8, 0);
            cv::putText(ref_cur_combine_temp, cur_propa, cv::Point2f(pc[0] - 40, pc[1] + 60), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255), 1, 8,
                        0);
            cv::putText(ref_cur_combine_temp, cur_optimize, cv::Point2f(pc[0] - 40, pc[1] + 80), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 255, 0), 1, 8,
                        0);

            cv::rectangle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols - radius, ref_ftr->px_[1] - radius),
                          cv::Point2f(ref_ftr->px_[0] + img_cur.cols + radius, ref_ftr->px_[1] + radius), cv::Scalar(0, 0, 255), 1);
            cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc[0] - radius, pc[1] - radius), cv::Point2f(pc[0] + radius, pc[1] + radius),
                          cv::Scalar(0, 255, 0), 1);
            cv::rectangle(ref_cur_combine_temp, cv::Point2f(pc_prior[0] - radius, pc_prior[1] - radius),
                          cv::Point2f(pc_prior[0] + radius, pc_prior[1] + radius), cv::Scalar(255, 255, 255), 1);
            cv::circle(ref_cur_combine_temp, cv::Point2f(ref_ftr->px_[0] + img_cur.cols, ref_ftr->px_[1]), 1, cv::Scalar(0, 0, 255), -1, 8);
            cv::circle(ref_cur_combine_temp, cv::Point2f(pc[0], pc[1]), 1, cv::Scalar(0, 255, 0), -1, 8);
            cv::circle(ref_cur_combine_temp, cv::Point2f(pc_prior[0], pc_prior[1]), 1, cv::Scalar(255, 255, 255), -1, 8);
            cv::imwrite(dir + std::to_string(new_frame_->id_) + "_" + std::to_string(ref_ftr->id_) + "_" + std::to_string(num) + ".png",
                        ref_cur_combine_temp);

            std::vector<std::vector<pixel_member>> pixel_warp_matrix;

            for (int y = 0; y < patch_size; ++y)
            {
                vector<pixel_member> pixel_warp_vec;
                for (int x = 0; x < patch_size; ++x) 
                {
                    Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
                    px_patch *= (1 << search_level);
                    const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
                    uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

                    const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
                    if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
                        continue;
                    else
                    {
                        pixel_member pixel_warp;
                        pixel_warp.pixel_pos << px[0], px[1];
                        pixel_warp.pixel_value = pixel_value;
                        pixel_warp_vec.push_back(pixel_warp);
                    }
                }
                pixel_warp_matrix.push_back(pixel_warp_vec);
            }

            float x_min = 1000;
            float y_min = 1000;
            float x_max = 0;
            float y_max = 0;

            for (int i = 0; i < pixel_warp_matrix.size(); i++)
            {
                vector<pixel_member> pixel_warp_row = pixel_warp_matrix[i];
                for (int j = 0; j < pixel_warp_row.size(); j++)
                {
                    float x_temp = pixel_warp_row[j].pixel_pos[0];
                    float y_temp = pixel_warp_row[j].pixel_pos[1];
                    if (x_temp < x_min) x_min = x_temp;
                    if (y_temp < y_min) y_min = y_temp;
                    if (x_temp > x_max) x_max = x_temp;
                    if (y_temp > y_max) y_max = y_temp;
                }
            }
            int x_min_i = floor(x_min);
            int y_min_i = floor(y_min);
            int x_max_i = ceil(x_max);
            int y_max_i = ceil(y_max);
            Matrix2f A_cur_ref_Inv = A_cur_ref.inverse().cast<float>();
            for (int i = x_min_i; i < x_max_i; i++)
            {
                for (int j = y_min_i; j < y_max_i; j++)
                {
                    Eigen::Vector2f pc_temp(i, j);
                    Vector2f px_patch = A_cur_ref_Inv * (pc_temp - pc.cast<float>());
                    if (px_patch[0] > (-patch_size / 2 * (1 << search_level)) && px_patch[0] < (patch_size / 2 * (1 << search_level)) &&
                        px_patch[1] > (-patch_size / 2 * (1 << search_level)) && px_patch[1] < (patch_size / 2 * (1 << search_level)))
                    {
                        const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
                        uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);
                        it_normal[width * j + i] = pixel_value;
                    }
                }
            }
        }
    }
    for (int i = 0; i < visual_submap->voxel_points.size(); i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];

        if (!pt->is_normal_initialized_) continue;

        Feature *ref_ftr;
        V2D pc(new_frame_->w2c(pt->pos_,0));
        ref_ftr = pt->ref_patch;

        Matrix2d A_cur_ref;
        getWarpMatrixAffine(*cams[0], ref_ftr->px_, ref_ftr->f_, (ref_ftr->pos() - pt->pos_).norm(), new_frame_->T_f_w_ [0]* ref_ftr->T_f_w_.inverse(), 0, 0,
                            patch_size_half, A_cur_ref);
        int search_level = getBestSearchLevel(A_cur_ref.inverse(), 2);
        double D = A_cur_ref.determinant();
        if (D > 3) continue;

        cv::Mat img_cur = new_frame_->imgs_[0];
        cv::Mat img_ref = ref_ftr->img_;
        for (int y = 0; y < patch_size; ++y)
        {
            for (int x = 0; x < patch_size; ++x) 
            {
                Vector2f px_patch(x - patch_size / 2, y - patch_size / 2);
                px_patch *= (1 << search_level);
                const Vector2f px_ref(px_patch + ref_ftr->px_.cast<float>());
                uint8_t pixel_value = (uint8_t)vk::interpolateMat_8u(img_ref, px_ref[0], px_ref[1]);

                const Vector2f px(A_cur_ref.cast<float>() * px_patch + pc.cast<float>());
                if (px[0] < 0 || px[1] < 0 || px[0] >= img_cur.cols - 1 || px[1] >= img_cur.rows - 1)
                    continue;
                else
                {
                    int col = int(px[0]);
                    int row = int(px[1]);
                    it[width * row + col] = pixel_value;
                }
            }
        }
    }
    cv::Mat ref_cur_combine;
    cv::Mat ref_cur_combine_normal;
    cv::Mat ref_cur_combine_error;

    cv::hconcat(result, new_frame_->imgs_[0], ref_cur_combine);
    cv::hconcat(result_normal, new_frame_->imgs_[0], ref_cur_combine_normal);

    cv::cvtColor(ref_cur_combine, ref_cur_combine, CV_GRAY2BGR);
    cv::cvtColor(ref_cur_combine_normal, ref_cur_combine_normal, CV_GRAY2BGR);
    cv::absdiff(img_photometric_error, result_normal, img_photometric_error);
    cv::hconcat(img_photometric_error, new_frame_->imgs_[0], ref_cur_combine_error);

    cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + ".png", ref_cur_combine);
    cv::imwrite(dir + std::to_string(new_frame_->id_) + +"_0_" +
                "photometric"
                ".png",
                ref_cur_combine_error);
    cv::imwrite(dir + std::to_string(new_frame_->id_) + "_0_" + "normal" + ".png", ref_cur_combine_normal);
}

void VIOManager::precomputeReferencePatches(int level)
{
    double t1 = omp_get_wtime();
    if (total_points == 0) return;

    const int num_cams = (int)cams.size();
    const int H_DIM = total_points * patch_size_total;  

    H_sub_inv.resize(H_DIM, 6);
    H_sub_inv.setZero();

    
    for (int i = 0; i < total_points; i++)
    {
        VisualPoint *pt = visual_submap->voxel_points[i];
        if (!pt) {
            ROS_WARN(" pt for Feature reference at point %d is empty, skipping.", i);
            continue;
        }

        
        int cam_idx = visual_submap->camera_ids[i];
        if (cam_idx < 0 || cam_idx >= num_cams) {
            ROS_WARN(" Invalid camera index for point %d, skipping.", i);
            continue;
        }

        double fx_i = cams[cam_idx]->fx();
        double fy_i = cams[cam_idx]->fy();

        Feature *ref_ftr = pt->ref_patch;
        if (!ref_ftr) {
            ROS_WARN(" ref_ftr for Feature reference at point %d is empty, skipping.", i);
            continue;
        }

        cv::Mat &img = ref_ftr->img_;
        if (img.empty()) {
            ROS_WARN(" Image for Feature reference at point %d is empty, skipping.", i);
            continue;
        }


        double depth = (pt->pos_ - ref_ftr->pos()).norm();
        V3D pf = ref_ftr->f_ * depth;

        V2D pc = ref_ftr->px_;
        M3D R_ref_w = ref_ftr->T_f_w_.rotation_matrix();

        MD(2, 3) Jdpi;
        computeProjectionJacobian(cam_idx, pf, Jdpi);

        M3D p_hat;
        p_hat << SKEW_SYM_MATRX(pt->pos_);

        int search_level   = visual_submap->search_levels[i];
        int pyramid_level  = level + search_level;
        int scale          = (1 << pyramid_level);
        float inv_scale    = 1.0f / scale;

        float u_ref = pc[0];
        float v_ref = pc[1];
        int u_ref_i = (int)std::floor(u_ref / scale) * scale;
        int v_ref_i = (int)std::floor(v_ref / scale) * scale;
        float subpix_u_ref = (u_ref - u_ref_i) / scale;
        float subpix_v_ref = (v_ref - v_ref_i) / scale;
        float w_ref_tl = (1.0f - subpix_u_ref) * (1.0f - subpix_v_ref);
        float w_ref_tr =          subpix_u_ref  * (1.0f - subpix_v_ref);
        float w_ref_bl = (1.0f - subpix_u_ref) *          subpix_v_ref;
        float w_ref_br =          subpix_u_ref  *          subpix_v_ref;

        int row_offset_pt = i * patch_size_total;  

            for (int px_y = 0; px_y < patch_size; px_y++)
            {
                int row_img = v_ref_i + px_y * scale - patch_size_half * scale;
                if (row_img < 1 || row_img >= img.rows - 1)
                {
                    continue;
                }

                int col_start = (u_ref_i - patch_size_half * scale);
                if (col_start < 1 || col_start >= img.cols - 1)
                {
                    continue;
                }

                uint8_t* img_ptr = (uint8_t*)img.data + row_img * img.cols + col_start;

                for (int px_x = 0; px_x < patch_size; px_x++, img_ptr += scale)
                {
                    if ((col_start + px_x * scale) < 1 || (col_start + px_x * scale) >= (img.cols - 1))
                    {
                        continue;
                    }

                    float du = 0.5f * (
                            (w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2]
                             + w_ref_bl * img_ptr[scale * img.cols + scale]
                             + w_ref_br * img_ptr[scale * img.cols + scale * 2])
                            - (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0]
                               + w_ref_bl * img_ptr[scale * img.cols - scale]
                               + w_ref_br * img_ptr[scale * img.cols])
                    );
                    float dv = 0.5f * (
                            (w_ref_tl * img_ptr[scale * img.cols] + w_ref_tr * img_ptr[scale + scale * img.cols]
                             + w_ref_bl * img_ptr[scale * img.cols * 2] + w_ref_br * img_ptr[scale * img.cols * 2 + scale])
                            - (w_ref_tl * img_ptr[-scale * img.cols] + w_ref_tr * img_ptr[-scale * img.cols + scale]
                               + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale])
                    );

                    MD(1,2) Jimg;
                    Jimg << du, dv;
                    Jimg *= inv_scale;

                    MD(1,3) J_dphi = Jimg * Jdpi * (R_ref_w * p_hat);
                    MD(1,3) J_dp   = -Jimg * Jdpi * R_ref_w;

                    MD(1,3) JdR = J_dphi * Jdphi_dR_vec[cam_idx] + J_dp * Jdp_dR_vec[cam_idx];
                    MD(1,3) Jdt = J_dp * Jdp_dt_vec[cam_idx];

                    int row_cur = row_offset_pt + px_y * patch_size + px_x;
                    if (row_cur >= 0 && row_cur < H_DIM)
                    {
                        H_sub_inv.block<1,6>(row_cur, 0) << JdR, Jdt;
                    }
                    else
                    {
                        ROS_WARN("[precomputeReferencePatches] row_cur out of bounds: %d", row_cur);
                    }
                }
            }
    }

    has_ref_patch_cache = true;
    compute_jacobian_time += (omp_get_wtime() - t1);
}

void VIOManager::updateStateInverse(const std::vector<cv::Mat>& imgs, int level)
{
    if (total_points == 0) return;
    StatesGroup old_state = (*state);
    const int num_cams = static_cast<int>(imgs.size());
    const int H_DIM = total_points * patch_size_total;  

    
    if (state->inv_expo_time_per_cam.size() != cams.size()) {
        state->inv_expo_time_per_cam.resize(cams.size(), 1.0);
    }

    VectorXd z(H_DIM);
    z.setZero();
    MatrixXd H_sub(H_DIM, 6); 
    H_sub.setZero();

    bool EKF_end = false;
    float last_error = std::numeric_limits<float>::max();
    int n_meas = 0;  

    
    std::vector<float> error_per_cam(num_cams, 0.0f);
    std::vector<int> n_meas_per_cam(num_cams, 0);
    std::vector<float> last_error_per_cam(num_cams, 0.0f);

    compute_jacobian_time = 0.0;
    update_ekf_time = 0.0;
    M3D P_wi_hat;
    P_wi_hat << SKEW_SYM_MATRX(state->pos_end);
    if (!has_ref_patch_cache) {
        precomputeReferencePatches(level);
    }
    for (int iter = 0; iter < max_iterations; iter++)
    {
        double t1 = omp_get_wtime();
        n_meas = 0;  
        float error = 0.0f;
        
        for (int c = 0; c < num_cams; c++) {
            error_per_cam[c] = 0.0f;
            n_meas_per_cam[c] = 0;
        }
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);

        
        for (int cam_idx = 0; cam_idx < num_cams; cam_idx++)
        {
            Rcw_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();
            Pcw_vec[cam_idx] = -Rci_vec[cam_idx] * Rwi.transpose() * Pwi + Pci_vec[cam_idx];
            Jdp_dt_vec[cam_idx] = Rci_vec[cam_idx] * Rwi.transpose();
        }

        
#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for schedule(dynamic, 4) reduction(+:error, n_meas)
#endif
        for (int i_pt = 0; i_pt < total_points; i_pt++)
        {
            float patch_error = 0.0f;
            const int scale = (1 << level);

            VisualPoint *pt = visual_submap->voxel_points[i_pt];
            if (!pt) continue;

            
            int cam_idx = visual_submap->camera_ids[i_pt];
            if (cam_idx < 0 || cam_idx >= num_cams) continue;
            if (imgs[cam_idx].empty()) continue;

            const cv::Mat &img_cur = imgs[cam_idx];
            
            double cur_inv_expo = (cam_idx < state->inv_expo_time_per_cam.size())
                                ? state->inv_expo_time_per_cam[cam_idx] : 1.0;

            
            V3D pf = Rcw_vec[cam_idx] * pt->pos_ + Pcw_vec[cam_idx];
            if (pf.z() < 1e-6) continue;
            V2D pc = cams[cam_idx]->world2cam(pf);
                int u_ref_i = static_cast<int>(std::floor(pc[0] / scale)) * scale;
                int v_ref_i = static_cast<int>(std::floor(pc[1] / scale)) * scale;
                float subpix_u_ref = (pc[0] - u_ref_i) / scale;
                float subpix_v_ref = (pc[1] - v_ref_i) / scale;
                float w_ref_tl = (1.f - subpix_u_ref) * (1.f - subpix_v_ref);
                float w_ref_tr =         subpix_u_ref  * (1.f - subpix_v_ref);
                float w_ref_bl = (1.f - subpix_u_ref) *        subpix_v_ref;
                float w_ref_br =         subpix_u_ref  *       subpix_v_ref;
                const std::vector<float> &P_patch = visual_submap->warp_patch[i_pt];
                double inv_ref_expo = visual_submap->inv_expo_list[i_pt];
                int row_offset_pt = i_pt * patch_size_total;  
                for (int px_y = 0; px_y < patch_size; px_y++)
                {
                    int row_img = v_ref_i + px_y * scale - patch_size_half * scale;
                    if (row_img < 1 || row_img >= (img_cur.rows - 1))
                        continue;
                    int col_start = u_ref_i - patch_size_half * scale;
                    if (col_start < 1 || col_start >= (img_cur.cols - 1))
                        continue;
                    uint8_t *img_ptr = (uint8_t*)img_cur.data + row_img * img_cur.cols + col_start;

                    for (int px_x = 0; px_x < patch_size; px_x++, img_ptr += scale)
                    {
                        int cur_col = col_start + px_x * scale;
                        if (cur_col < 1 || cur_col >= (img_cur.cols - 1))
                            continue;
                        float du = 0.5f * (
                                (w_ref_tl * img_ptr[scale] +
                                 w_ref_tr * img_ptr[scale * 2] +
                                 w_ref_bl * img_ptr[scale * img_cur.cols + scale] +
                                 w_ref_br * img_ptr[scale * img_cur.cols + scale * 2])
                                -
                                (w_ref_tl * img_ptr[-scale] +
                                 w_ref_tr * img_ptr[0] +
                                 w_ref_bl * img_ptr[scale * img_cur.cols - scale] +
                                 w_ref_br * img_ptr[scale * img_cur.cols])
                        );
                        float dv = 0.5f * (
                                (w_ref_tl * img_ptr[scale * img_cur.cols] +
                                 w_ref_tr * img_ptr[scale + scale * img_cur.cols] +
                                 w_ref_bl * img_ptr[scale * img_cur.cols * 2] +
                                 w_ref_br * img_ptr[scale * img_cur.cols * 2 + scale])
                                -
                                (w_ref_tl * img_ptr[-scale * img_cur.cols] +
                                 w_ref_tr * img_ptr[-scale * img_cur.cols + scale] +
                                 w_ref_bl * img_ptr[0] +
                                 w_ref_br * img_ptr[scale])
                        );

                        MD(1,2) Jimg;
                        Jimg << du, dv;
                        Jimg = Jimg * cur_inv_expo;
                        Jimg = Jimg * (1.0f / scale);
                        MD(2,3) Jdpi;
                        computeProjectionJacobian(cam_idx, pf, Jdpi);
                        M3D p_hat;
                        p_hat << SKEW_SYM_MATRX(pf);
                        MD(1,3) J_dphi = Jimg * Jdpi * p_hat;
                        MD(1,3) J_dp   = -Jimg * Jdpi;
                        MD(1,3) JdR_local = J_dphi * Jdphi_dR_vec[cam_idx] + J_dp * Jdp_dR_vec[cam_idx];
                        MD(1,3) Jdt_local = J_dp * Jdp_dt_vec[cam_idx];
                        int row_idx = row_offset_pt + px_y * patch_size + px_x;
                        if (row_idx < 0 || row_idx >= H_DIM) continue;
                        MD(1,3) J_dR_ref = H_sub_inv.block<1,3>(row_idx, 0);
                        MD(1,3) J_dt_ref = H_sub_inv.block<1,3>(row_idx, 3);
                        MD(1,3) JdR_final = J_dR_ref * Rwi + J_dt_ref * P_wi_hat * Rwi;
                        MD(1,3) Jdt_final = J_dt_ref * Rwi;
                        H_sub.block<1,6>(row_idx, 0) << JdR_final, Jdt_final;
                        float cur_val = w_ref_tl * img_ptr[0] +
                                        w_ref_tr * img_ptr[scale] +
                                        w_ref_bl * img_ptr[scale * img_cur.cols] +
                                        w_ref_br * img_ptr[scale * img_cur.cols + scale];

                        int patch_idx = px_y * patch_size + px_x;
                        double res = cur_inv_expo * cur_val - inv_ref_expo * P_patch[patch_size_total * level + patch_idx];
                        z(row_idx) = res;
                        patch_error += res * res;
                        n_meas++;


                    } 
                } 

                visual_submap->errors[i_pt] = patch_error;
#pragma omp atomic
                error += patch_error;
        } 

        
        for (int i_pt = 0; i_pt < total_points; i_pt++) {
            int cam_idx = visual_submap->camera_ids[i_pt];
            if (cam_idx >= 0 && cam_idx < num_cams) {
                error_per_cam[cam_idx] += visual_submap->errors[i_pt];
                
                n_meas_per_cam[cam_idx] += patch_size_total;
            }
        }

        if (n_meas > 0)
            error /= n_meas;

        
        for (int c = 0; c < num_cams; c++) {
            if (n_meas_per_cam[c] > 0) {
                error_per_cam[c] /= n_meas_per_cam[c];
            }
        }

        
        double rmse_per_pixel = sqrt(error);  
        double avg_points = (double)n_meas / (patch_size_total > 0 ? patch_size_total : 1);




        compute_jacobian_time += omp_get_wtime() - t1;
        double t3 = omp_get_wtime();
        if (error <= last_error && error > 0)
        {
            old_state = (*state);
            last_error = error;
            for (int c = 0; c < num_cams; c++) {
                last_error_per_cam[c] = error_per_cam[c];
            }


            std::vector<double> cov_per_cam(num_cams, img_point_cov);
            if (enable_dynamic_covariance_) {
                for (int c = 0; c < num_cams; c++) {
                    cov_per_cam[c] = calculateCoVarianceScalePerCam(c, prev_avg_error_per_cam_[c], prev_n_meas_per_cam_[c]);
                }

                // Output per-camera dynamic covariance info
                if (frame_count % 30 == 0) {
                    std::stringstream ss;
                    ss << "[DynCov-PerCam] frame=" << frame_count << ", cov=[";
                    for (int c = 0; c < num_cams; c++) {
                        ss << "cam" << c << ":" << std::fixed << std::setprecision(1) << cov_per_cam[c];
                        if (c < num_cams - 1) ss << ", ";
                    }
                    ss << "]";
                    ROS_INFO("%s", ss.str().c_str());
                }
            }


            MatrixXd H_weighted = H_sub;
            VectorXd z_weighted = z;
            for (int i_pt = 0; i_pt < total_points; i_pt++) {
                int cam_idx = visual_submap->camera_ids[i_pt];
                if (cam_idx < 0 || cam_idx >= num_cams) continue;
                double weight = 1.0 / std::sqrt(cov_per_cam[cam_idx]);
                int row_start = i_pt * patch_size_total;
                for (int k = 0; k < patch_size_total; k++) {
                    int row_idx = row_start + k;
                    if (row_idx < H_DIM) {
                        H_weighted.row(row_idx) *= weight;
                        z_weighted(row_idx) *= weight;
                    }
                }
            }

            auto H_weighted_T = H_weighted.transpose();
            H_T_H.setZero();
            G.setZero();
            H_T_H.block<6,6>(0,0) = H_weighted_T * H_weighted;
            double det = H_T_H.determinant();

            
            MD(DIM_STATE, DIM_STATE) K_1 = (H_T_H + state->cov.inverse()).inverse();
            auto HTz = H_weighted_T * z_weighted;
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            MD(DIM_STATE,1) solution = -K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);

            (*state) += solution;

            
            if (exposure_estimate_en) {
                std::vector<double> expo_H(num_cams, 0.0);  
                std::vector<double> expo_b(num_cams, 0.0);  

                
                for (int i_pt = 0; i_pt < total_points; i_pt++) {
                    int cam_idx = visual_submap->camera_ids[i_pt];
                    if (cam_idx < 0 || cam_idx >= num_cams) continue;

                    double inv_ref_expo = visual_submap->inv_expo_list[i_pt];
                    const std::vector<float>& P_patch = visual_submap->warp_patch[i_pt];
                    int row_offset_pt = i_pt * patch_size_total;

                    for (int px_idx = 0; px_idx < patch_size_total; px_idx++) {
                        int row_idx = row_offset_pt + px_idx;
                        if (row_idx >= H_DIM) continue;

                        
                        
                        double residual = z_weighted(row_idx);
                        double ref_val = P_patch[patch_size_total * level + px_idx];
                        
                        double cur_inv_expo = state->inv_expo_time_per_cam[cam_idx];
                        double cur_val = (residual + inv_ref_expo * ref_val) / (cur_inv_expo + 1e-10);

                        double J_expo = cur_val;
                        expo_H[cam_idx] += J_expo * J_expo;
                        expo_b[cam_idx] += J_expo * residual;
                    }
                }

                
                for (int c = 0; c < num_cams; c++) {
                    if (expo_H[c] > 1e-6) {
                        double expo_before = state->inv_expo_time_per_cam[c];
                        double delta_expo = -expo_b[c] / expo_H[c];
                        state->inv_expo_time_per_cam[c] += delta_expo;
                        double expo_after = state->inv_expo_time_per_cam[c];
                        ROS_ERROR("[InvComp Cam%d] Before=%.6f, Delta=%.6f, After=%.6f, RealExpo=%.4fms, H=%.3e",
                                  c, expo_before, delta_expo, expo_after, 1000.0/expo_after, expo_H[c]);
                    }
                }
            }

            auto rot_add = solution.block<3,1>(0,0);
            auto t_add   = solution.block<3,1>(3,0);

            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
                EKF_end = true;
        }
        else
        {
            (*state) = old_state;
            EKF_end = true;
        }
        update_ekf_time += omp_get_wtime() - t3;

        if (iter == max_iterations - 1 || EKF_end)
            break;
    } 

    
    prev_avg_error_ = last_error;
    prev_n_meas_ = n_meas;

    
    for (int c = 0; c < num_cams; c++) {
        if (c < (int)prev_avg_error_per_cam_.size()) {
            prev_avg_error_per_cam_[c] = last_error_per_cam[c];
            prev_n_meas_per_cam_[c] = n_meas_per_cam[c];
        }
    }

    
    double final_rmse = sqrt(last_error);
    if (enable_dynamic_covariance_) {
        std::string cov_info = "";
        for (int c = 0; c < num_cams; c++) {
            double cov_c = calculateCoVarianceScalePerCam(c, last_error_per_cam[c], n_meas_per_cam[c]);
            char buf[64];
            snprintf(buf, sizeof(buf), " cam%d:%.0f", c, cov_c);
            cov_info += buf;
        }
        if (frame_count % 30 == 0) {
            ROS_INFO("[VIO Final] frame=%d, MSE=%.3f, RMSE=%.3f, meas=%d, cov:[%s]",
                     frame_count, last_error, final_rmse, n_meas, cov_info.c_str());
        }
    } else {
        if (frame_count % 30 == 0) {
            ROS_INFO("[VIO Final] frame=%d, final_MSE=%.3f, final_RMSE=%.3f (per-pixel gray), total_meas=%d, img_point_cov=%.1f",
                     frame_count, last_error, final_rmse, n_meas, img_point_cov);
        }
    }

    
}

inline float huberWeight(float r, float huber_delta = 5.0f)
{
    float abs_r = fabs(r);
    if (abs_r < huber_delta) {
        return 1.0f;
    } else {
        return huber_delta / abs_r;
    }
}

void VIOManager::updateState(const std::vector<cv::Mat> &imgs, int level) {
    if (total_points == 0) return;
    StatesGroup old_state = (*state);
    const int num_cams = (int)cams.size();
    int base_measurement_size = total_points * patch_size_total * num_cams;
    int cross_cam_measurements = total_points * patch_size_total * num_cams * (num_cams - 1) / 2;
    int total_measurements = base_measurement_size + cross_cam_measurements;

    VectorXd z(total_measurements);
    z.setZero();
    
    MatrixXd H_sub(total_measurements, 6 + num_cams);
    H_sub.setZero();

    bool EKF_end = false;
    float last_error = std::numeric_limits<float>::max();
    int n_meas = 0;  

    
    std::vector<float> error_per_cam(num_cams, 0.0f);
    std::vector<int> n_meas_per_cam(num_cams, 0);
    std::vector<float> last_error_per_cam(num_cams, 0.0f);
    std::vector<int> row_start_per_cam(num_cams, 0);  

    
    successful_cross_camera_tracks = 0;

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        double t1 = omp_get_wtime();
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        
        if (state->inv_expo_time_per_cam.size() != cams.size()) {
            state->inv_expo_time_per_cam.resize(cams.size(), 1.0);
        }
        float error = 0.0f;
        n_meas = 0;  
        
        for (int c = 0; c < num_cams; c++) {
            error_per_cam[c] = 0.0f;
            n_meas_per_cam[c] = 0;
        }
        int row_offset_cam = 0;
        
        int cross_cam_row_offset = base_measurement_size;

        
        struct CrossCameraPair {
            VisualPoint* pt;
            int cam_a;
            int cam_b;
        };
        std::vector<CrossCameraPair> cross_camera_pairs;
        cross_camera_pairs.reserve(total_points);  

        
        for (int c = 0; c < num_cams; c++) {
            row_start_per_cam[c] = c * total_points * patch_size_total;
            if (!imgs[c].empty()) {
                Rcw_vec[c] = Rci_vec[c] * Rwi.transpose();
                Pcw_vec[c] = -Rci_vec[c] * Rwi.transpose() * Pwi + Pci_vec[c];
                Jdp_dt_vec[c] = Rci_vec[c] * Rwi.transpose();
            }
        }

        
#ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for schedule(dynamic, 4) reduction(+:error, n_meas)
#endif
        for (int i_pt = 0; i_pt < total_points; i_pt++) {
            float patch_error = 0.0f;

            VisualPoint *pt = visual_submap->voxel_points[i_pt];
            if (!pt) continue;

            
            int cam_idx = visual_submap->camera_ids[i_pt];
            if (cam_idx < 0 || cam_idx >= num_cams) continue;
            if (imgs[cam_idx].empty()) continue;

            const cv::Mat &img_cur = imgs[cam_idx];

            V3D pf = Rcw_vec[cam_idx] * pt->pos_ + Pcw_vec[cam_idx];
            if (pf.z() < 1e-6)
                continue;

            V2D pc = cams[cam_idx]->world2cam(pf);
            MD(2, 3) Jdpi;
            computeProjectionJacobian(cam_idx, pf, Jdpi);
            M3D p_hat;
            p_hat << SKEW_SYM_MATRX(pf);
            int scale = (1 << level);
            float inv_scale = 1.0f / scale;

            float u_ref = pc[0];
            float v_ref = pc[1];
            int u_ref_i = floorf(u_ref / scale) * scale;
            int v_ref_i = floorf(v_ref / scale) * scale;
            float subpix_u_ref = (u_ref - u_ref_i) * inv_scale;
            float subpix_v_ref = (v_ref - v_ref_i) * inv_scale;
            float w_ref_tl = (1.f - subpix_u_ref) * (1.f - subpix_v_ref);
            float w_ref_tr = subpix_u_ref * (1.f - subpix_v_ref);
            float w_ref_bl = (1.f - subpix_u_ref) * subpix_v_ref;
            float w_ref_br = subpix_u_ref * subpix_v_ref;
            const std::vector<float> &P = visual_submap->warp_patch[i_pt];
            double inv_ref_expo = visual_submap->inv_expo_list[i_pt];
            
            float cur_inv_expo = (cam_idx < state->inv_expo_time_per_cam.size())
                               ? state->inv_expo_time_per_cam[cam_idx] : 1.0;
            if ((int)P.size() <= patch_size_total * level)
                continue;

            
            
            bool will_have_cross_camera_constraint = false;
            if (enable_cross_camera_tracking) {
                for (int other_cam = 0; other_cam < num_cams; other_cam++) {
                    if (other_cam == cam_idx) continue;
                    
                    if (pt->cross_cam_data_.currently_visible.test(cam_idx) &&
                        pt->cross_cam_data_.currently_visible.test(other_cam)) {
                        will_have_cross_camera_constraint = true;
                        break;
                    }
                }
            }

            
            int row_offset_pt = row_start_per_cam[cam_idx] + i_pt * patch_size_total;
                for (int px_x = 0; px_x < patch_size; px_x++) {
                    int row_img = v_ref_i + px_x * scale - patch_size_half * scale;
                    if (row_img < 1 || row_img >= (img_cur.rows - 1))
                        continue;
                    uint8_t *img_ptr =
                            (uint8_t *)img_cur.data + row_img * width + (u_ref_i - patch_size_half * scale);

                    for (int px_y = 0; px_y < patch_size; px_y++, img_ptr += scale) {
                        int col_img = (u_ref_i - patch_size_half * scale) + px_y * scale;
                        if (col_img < 1 || col_img >= (img_cur.cols - 1))
                            continue;
                        float du = 0.5f * (
                                (w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2]
                                 + w_ref_bl * img_ptr[scale * width + scale]
                                 + w_ref_br * img_ptr[scale * width + scale * 2])
                                - (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0]
                                   + w_ref_bl * img_ptr[scale * width - scale]
                                   + w_ref_br * img_ptr[scale * width])
                        );
                        float dv = 0.5f * (
                                (w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width]
                                 + w_ref_bl * img_ptr[scale * width * 2] +
                                 w_ref_br * img_ptr[scale * width * 2 + scale])
                                - (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale]
                                   + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale])
                        );
                        MD(1, 2) Jimg;
                        Jimg << du, dv;
                        Jimg *= (float)cur_inv_expo;
                        Jimg *= inv_scale;
                        MD(1, 3) Jdphi = Jimg * Jdpi * p_hat;
                        MD(1, 3) Jdp = -Jimg * Jdpi;
                        MD(1, 3) JdR = Jdphi * Jdphi_dR_vec[cam_idx]
                                       + Jdp * Jdp_dR_vec[cam_idx];
                        MD(1, 3) Jdt = Jdp * Jdp_dt_vec[cam_idx];
                        float cur_val = w_ref_tl * img_ptr[0]
                                        + w_ref_tr * img_ptr[scale]
                                        + w_ref_bl * img_ptr[scale * width]
                                        + w_ref_br * img_ptr[scale * width + scale];
                        int idx_patch = px_x * patch_size + px_y;
                        float ref_val = P[patch_size_total * level + idx_patch];
                        
                        float res;
                        if (exposure_estimate_en) {
                            res = cur_inv_expo * cur_val - inv_ref_expo * ref_val;
                        } else {
                            res = ref_val - cur_val;
                        }
                        float corrected_res = res;
                        

                        if (pt->ref_patch && isRealCrossCameraPoint(pt, cam_idx)) {
                            V2D px_pos(pc);
                            float source_intensity = applyCameraPhotoCorrection(
                                    inv_ref_expo * ref_val, pt->ref_patch->cam_id_, pt->ref_patch->px_);
                            float target_intensity = applyCameraPhotoCorrection(
                                    cur_inv_expo * cur_val, cam_idx, px_pos);
                            corrected_res = source_intensity - target_intensity;
                        }

                        float w = 1.0f;
                        float abs_res = std::abs(corrected_res);
                        if (abs_res > outlier_threshold) {
                            w = 0.0f;  
                        } else {
                            float ratio = corrected_res / outlier_threshold;
                            float temp = 1.0f - ratio * ratio;
                            w = temp * temp;  
                        }

                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        

                        float sqrt_w = sqrtf(w);
                        int row_here = row_offset_pt + idx_patch;
                        z(row_here) = sqrt_w * corrected_res;
                        H_sub(row_here, 0) = sqrt_w * JdR(0);
                        H_sub(row_here, 1) = sqrt_w * JdR(1);
                        H_sub(row_here, 2) = sqrt_w * JdR(2);

                    H_sub(row_here, 3) = sqrt_w * Jdt(0);
                    H_sub(row_here, 4) = sqrt_w * Jdt(1);
                    H_sub(row_here, 5) = sqrt_w * Jdt(2);
                    
                    if (exposure_estimate_en) {
                        
                        
                        float expo_jacobian = cur_val;
                        int expo_col = 6 + cam_idx;  
                        H_sub(row_here, expo_col) = sqrt_w * expo_jacobian;
                    }
                    patch_error += w * (corrected_res * corrected_res);
                    n_meas++;
                }
            }
            visual_submap->errors[i_pt] = patch_error;
            error += patch_error;
#pragma omp critical
            {
                error_per_cam[cam_idx] += patch_error;
                n_meas_per_cam[cam_idx] += patch_size_total;  
            }
            
            
            if (enable_cross_camera_tracking) {
                for (int other_cam = cam_idx + 1; other_cam < num_cams; other_cam++) {
                    if (pt->cross_cam_data_.currently_visible.test(cam_idx) &&
                        pt->cross_cam_data_.currently_visible.test(other_cam)) {
#pragma omp critical
                        {
                            cross_camera_pairs.push_back({pt, cam_idx, other_cam});
                        }
                    }
                }
            }
        } 


        successful_cross_camera_tracks = 0;
        for (const auto& pair : cross_camera_pairs) {
            addCrossCameraConsistencyConstraint(
                pair.pt, pair.cam_a, pair.cam_b, H_sub, z, level, cross_cam_row_offset);
            successful_cross_camera_tracks++;
        }

        // Output cross-camera constraint info
        if (cross_camera_pairs.size() > 0 && frame_count % 30 == 0) {
            ROS_INFO("[CrossCam] frame=%d, pairs=%zu, level=%d",
                     frame_count, cross_camera_pairs.size(), level);
        }

        if (n_meas > 0) {
            error /= (float)n_meas;
        }

        
        for (int c = 0; c < num_cams; c++) {
            if (n_meas_per_cam[c] > 0) {
                error_per_cam[c] /= n_meas_per_cam[c];
            }
        }

        compute_jacobian_time += (omp_get_wtime() - t1);
        double t3 = omp_get_wtime();

        if (error <= last_error) {
            old_state = (*state);
            last_error = error;
            for (int c = 0; c < num_cams; c++) {
                last_error_per_cam[c] = error_per_cam[c];
            }


            std::vector<double> cov_per_cam(num_cams, img_point_cov);
            if (enable_dynamic_covariance_) {
                for (int c = 0; c < num_cams; c++) {
                    cov_per_cam[c] = calculateCoVarianceScalePerCam(c, prev_avg_error_per_cam_[c], prev_n_meas_per_cam_[c]);
                }

                // Output per-camera dynamic covariance info
                if (frame_count % 30 == 0) {
                    std::stringstream ss;
                    ss << "[DynCov-PerCam] frame=" << frame_count << ", cov=[";
                    for (int c = 0; c < num_cams; c++) {
                        ss << "cam" << c << ":" << std::fixed << std::setprecision(1) << cov_per_cam[c];
                        if (c < num_cams - 1) ss << ", ";
                    }
                    ss << "]";
                    ROS_INFO("%s", ss.str().c_str());
                }
            }


            MatrixXd H_weighted = H_sub;
            VectorXd z_weighted = z;
            int cam_meas_size = total_points * patch_size_total;
            for (int c = 0; c < num_cams; c++) {
                double weight = 1.0 / std::sqrt(cov_per_cam[c]);
                int row_start = row_start_per_cam[c];
                int row_end = row_start + cam_meas_size;
                for (int row = row_start; row < row_end && row < total_measurements; row++) {
                    H_weighted.row(row) *= weight;
                    z_weighted(row) *= weight;
                }
            }

            auto H_weighted_T = H_weighted.transpose();
            H_T_H.setZero();
            G.setZero();

            
            int calib_dim = 6 + num_cams;
            MatrixXd H_T_H_expo(calib_dim, calib_dim);
            H_T_H_expo = H_weighted_T * H_weighted;

            auto HTz = H_weighted_T * z_weighted;
            auto vec = (*state_propagat) - (*state);

            
            Matrix<double, 6, 6> K_pose =
                    (H_T_H_expo.block<6,6>(0,0) + state->cov.block<6,6>(0,0).inverse()).inverse();

            
            VectorXd solution_pose = -K_pose * HTz.head<6>() + vec.head<6>();

            
            for (int c = 0; c < num_cams; c++) {
                int expo_idx = 6 + c;
                double H_ii = H_T_H_expo(expo_idx, expo_idx);
                if (H_ii > 1e-6) {
                    double expo_before = state->inv_expo_time_per_cam[c];
                    double delta_expo = -HTz(expo_idx) / H_ii;
                    double expo_after = expo_before + delta_expo;

                    
                    const double min_inv_expo = 0.001;
                    const double max_inv_expo = 100.0;

                    if (expo_after < min_inv_expo) {
                        expo_after = min_inv_expo;
                    } else if (expo_after > max_inv_expo) {
                        expo_after = max_inv_expo;
                    }

                    state->inv_expo_time_per_cam[c] = expo_after;

                }
            }

            
            MD(DIM_STATE, 1) solution = MD(DIM_STATE, 1)::Zero();
            solution.head<6>() = solution_pose;
            (*state) += solution;


            auto rot_add = solution.block<3,1>(0,0);
            auto t_add   = solution.block<3,1>(3,0);

            
            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.f < 0.001f)) {
                EKF_end = true;
            }
        } else {
            (*state) = old_state;
            EKF_end = true;
        }

        update_ekf_time += (omp_get_wtime() - t3);
        if (iteration == max_iterations - 1 || EKF_end)
            break;
    }
    
    prev_avg_error_ = last_error;
    prev_n_meas_ = n_meas;

    
    for (int c = 0; c < num_cams; c++) {
        if (c < (int)prev_avg_error_per_cam_.size()) {
            prev_avg_error_per_cam_[c] = last_error_per_cam[c];
            prev_n_meas_per_cam_[c] = n_meas_per_cam[c];
        }
    }
}

void VIOManager::updateFrameState(StatesGroup state) {
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    assert(Rci_vec.size() == cams.size());
    assert(Pci_vec.size() == cams.size());
    if (new_frame_->T_f_w_.size() != cams.size()) {
        new_frame_->T_f_w_.resize(cams.size(), SE3());
    }
    for (int cam_idx = 0; cam_idx < cams.size(); cam_idx++) {
        M3D Rcw = Rci_vec[cam_idx] * Rwi.transpose();

        V3D Pcw = -Rci_vec[cam_idx] * Rwi.transpose() * Pwi + Pci_vec[cam_idx];
        new_frame_->T_f_w_[cam_idx] = SE3(Rcw, Pcw);
    }
}
void VIOManager::plotTrackedPoints() {
    if (visual_submap->voxel_points.empty() || imgs_rgb.empty()) return;
    int num_cams = std::min(cams.size(), imgs_rgb.size());
    if (num_cams == 0) return;

    int grid_rows, grid_cols;
    if (num_cams <= 1) {
        grid_rows = grid_cols = 1;
    } else {
        grid_rows = (int)ceil(sqrt(num_cams));
        grid_cols = (int)ceil((double)num_cams / grid_rows);
        if ((grid_rows - 1) * (grid_cols + 1) >= num_cams &&
            (grid_rows - 1) > 0 &&
            abs((grid_rows - 1) - (grid_cols + 1)) < abs(grid_rows - grid_cols)) {
            grid_rows--;
            grid_cols++;
        }
    }

    
    const double SCALE_FACTOR = 0.25;  
    int img_width = imgs_rgb[0].cols;
    int img_height = imgs_rgb[0].rows;
    int display_width = static_cast<int>(img_width * SCALE_FACTOR);
    int display_height = static_cast<int>(img_height * SCALE_FACTOR);

    cv::Mat display = cv::Mat(grid_rows * display_height, grid_cols * display_width, CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar normal_color(0, 255, 0);     

    
    const int POINT_SKIP = 1;  

    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        
        cv::Mat cam_img_resized;
        cv::resize(imgs_rgb[cam_idx], cam_img_resized, cv::Size(display_width, display_height));

        int row = cam_idx / grid_cols;
        int col = cam_idx % grid_cols;
        cv::Rect roi(col * display_width, row * display_height, display_width, display_height);

        
        for (size_t i = 0; i < visual_submap->voxel_points.size(); i += POINT_SKIP) {  
            VisualPoint* pt = visual_submap->voxel_points[i];
            if (!pt) continue;

            
            int pt_cam_id = visual_submap->camera_ids[i];
            if (pt_cam_id != cam_idx) continue;  

            
            V3D pt_cam = new_frame_->w2f(pt->pos_, cam_idx);
            if (pt_cam[2] > 0) {  
                V2D pc = new_frame_->w2c(pt->pos_, cam_idx);

                
                int x = static_cast<int>(pc[0] * SCALE_FACTOR);
                int y = static_cast<int>(pc[1] * SCALE_FACTOR);

                if (x >= 0 && x < display_width && y >= 0 && y < display_height) {
                    
                    
                    cv::circle(cam_img_resized, cv::Point(x, y), 1, normal_color, -1);  
                }
            }
        }

        cam_img_resized.copyTo(display(roi));
    }

    panorama_image = display;
}

V3F VIOManager::getInterpolatedPixel(cv::Mat img, V2D pc) {
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]);
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref - u_ref_i);
    const float subpix_v_ref = (v_ref - v_ref_i);
    const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t *img_ptr = (uint8_t *) img.data + ((v_ref_i) * width + (u_ref_i)) * 3;
    float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] +
              w_ref_br * img_ptr[width * 3 + 0 + 3];
    float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] +
              w_ref_br * img_ptr[width * 3 + 1 + 3];
    float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] +
              w_ref_br * img_ptr[width * 3 + 2 + 3];
    V3F pixel(B, G, R);
    return pixel;
}

void VIOManager::dumpDataForColmap() {
    static int cnt = 1; 
    std::ostringstream ss;
    ss << std::setw(5) << std::setfill('0') << cnt;
    std::string cnt_str = ss.str();
    if (cams.empty()) {
        ROS_ERROR("No cameras available for COLMAP export");
        return;
    }
    for (size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx) {
        std::string image_path = std::string(ROOT_DIR) + "Log/Colmap/images/" + cnt_str + "_cam" + std::to_string(cam_idx) + ".png";
        if (cam_idx >= imgs_rgb.size() || imgs_rgb[cam_idx].empty()) {
            ROS_WARN("Missing or empty image for camera %zu", cam_idx);
            continue;
        }

        cv::Mat img_rgb = imgs_rgb[cam_idx];
        cv::Mat img_rgb_undistort;
        vk::PinholeCamera* pinhole_cam_ptr = dynamic_cast<vk::PinholeCamera*>(cams[cam_idx]);
        if (!pinhole_cam_ptr) {
            ROS_WARN("Camera %zu is not a pinhole camera, skipping undistortion", cam_idx);
            img_rgb_undistort = img_rgb.clone(); 
        } else {
            pinhole_cam_ptr->undistortImage(img_rgb, img_rgb_undistort);
        }
        cv::imwrite(image_path, img_rgb_undistort);
    }
    static bool cameras_written = false;
    if (!cameras_written) {
        fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), std::ios::out);
        fout_camera << "# Camera list with one line of data per camera:\n";
        fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
        for (size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx) {
            fout_camera << (cam_idx + 1) << " PINHOLE "
                        << cams[cam_idx]->width() << " " << cams[cam_idx]->height() << " "
                        << std::fixed << std::setprecision(6)  
                        << cams[cam_idx]->fx() << " " << cams[cam_idx]->fy() << " "
                        << cams[cam_idx]->cx() << " " << cams[cam_idx]->cy() << std::endl;
        }
        fout_camera.close();
        cameras_written = true;
    }
    if (!fout_colmap.is_open()) {
        fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), std::ios::out);
        fout_colmap << "# Image list with two lines of data per image:\n";
        fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
        fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    }
    for (size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx) {
        if (cam_idx >= new_frame_->T_f_w_.size()) {
            ROS_WARN("Missing pose for camera %zu", cam_idx);
            continue;
        }
        SE3 cam_pose = new_frame_->T_f_w_[cam_idx];
        Eigen::Quaterniond q(cam_pose.rotation_matrix());
        Eigen::Vector3d t = cam_pose.translation();
        int image_id = cnt * 100 + static_cast<int>(cam_idx);
        std::string image_name = cnt_str + "_cam" + std::to_string(cam_idx) + ".png";
        fout_colmap << image_id << " "
                    << std::fixed << std::setprecision(6)  
                    << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                    << t.x() << " " << t.y() << " " << t.z() << " "
                    << (cam_idx + 1) << " "  
                    << image_name << std::endl;
        fout_colmap << "0.0 0.0 -1.0" << std::endl;
    }

    cnt++; 
}

void VIOManager::initializeCameraPhotoParams() {
    camera_photo_params.resize(cams.size());

    for (size_t i = 0; i < cams.size(); i++) {
        camera_photo_params[i].exposure_factor = 1.0; 
        camera_photo_params[i].vignetting = {0.0, 0.0, 0.0}; 
        camera_photo_params[i].parameters_initialized = false;
    }

    
}

float VIOManager::applyCameraPhotoCorrection(float intensity, int cam_id, const V2D& pixel_pos) {
    if (cam_id < 0 || cam_id >= (int)camera_photo_params.size()) {
        return intensity; 
    }
    float width = cams[cam_id]->width();
    float height = cams[cam_id]->height();

    float dx = (pixel_pos[0] / width) - 0.5f;
    float dy = (pixel_pos[1] / height) - 0.5f;
    float r2 = dx*dx + dy*dy;
    float vignette_factor = 1.0f;
    if (!camera_photo_params[cam_id].vignetting.empty()) {
        vignette_factor = 1.0f +
                          camera_photo_params[cam_id].vignetting[0] * r2 +
                          camera_photo_params[cam_id].vignetting[1] * r2*r2 +
                          camera_photo_params[cam_id].vignetting[2] * r2*r2*r2;
    }
    return (intensity * vignette_factor) * camera_photo_params[cam_id].exposure_factor;
}



void VIOManager::cleanupOldVisualPoints() {
    if (feat_map.empty()) return;

    int current_frame_id = new_frame_ ? new_frame_->id_ : 0;

    std::vector<VOXEL_LOCATION> voxels_to_delete;
    voxels_to_delete.reserve(feat_map.size() / 10);  
    int total_points_before = 0;
    int total_points_deleted = 0;

    for (auto& [voxel_loc, voxel_pts] : feat_map) {
        if (!voxel_pts) continue;

        auto& points = voxel_pts->voxel_points;
        total_points_before += points.size();

        
        points.erase(
            std::remove_if(points.begin(), points.end(),
                [this, current_frame_id, &total_points_deleted](VisualPoint* pt) {
                    if (!pt) return true;

                    
                    int last_seen = pt->cross_cam_data_.last_seen_frame_id;
                    bool too_old = (last_seen > 0) &&
                                   ((current_frame_id - last_seen) > max_point_age_frames);

                    
                    if (pt->is_converged_ && pt->obs_.size() > 5) {
                        too_old = (last_seen > 0) &&
                                  ((current_frame_id - last_seen) > max_point_age_frames * 2);
                    }

                    if (too_old) {
                        delete pt;
                        total_points_deleted++;
                        return true;
                    }
                    return false;
                }),
            points.end()
        );

        
        if (points.empty()) {
            voxels_to_delete.push_back(voxel_loc);
        }
    }

    
    for (auto& loc : voxels_to_delete) {
        auto it = feat_map.find(loc);
        if (it != feat_map.end()) {
            delete it->second;
            feat_map.erase(it);
        }
    }

    if (total_points_deleted > 0) {
        ROS_INFO("[VIO Memory] Cleaned up %d old points (%.1f%%), %zu empty voxels. Remaining: %d points in %zu voxels",
                 total_points_deleted,
                 100.0 * total_points_deleted / std::max(1, total_points_before),
                 voxels_to_delete.size(),
                 total_points_before - total_points_deleted,
                 feat_map.size());
    }

    last_cleanup_frame_id = current_frame_id;
}


void VIOManager::cleanupVisualMapByTimestamp(double oldest_kept_timestamp) {
    if (feat_map.empty() || oldest_kept_timestamp < 0) {
        return;
    }

    std::vector<VOXEL_LOCATION> voxels_to_delete;
    voxels_to_delete.reserve(feat_map.size() / 10);
    int total_voxels_before = feat_map.size();
    int deleted_voxels = 0;

    for (auto it = feat_map.begin(); it != feat_map.end(); ) {
        if (!it->second) {
            it = feat_map.erase(it);
            continue;
        }

        
        double voxel_timestamp = it->second->creation_timestamp_;

        
        if (voxel_timestamp < 0) {
            ++it;
            continue;
        }

        
        if (voxel_timestamp < oldest_kept_timestamp) {
            delete it->second;
            it = feat_map.erase(it);
            deleted_voxels++;
        } else {
            ++it;
        }
    }

    if (deleted_voxels > 0) {
        ROS_INFO("[VIO Sliding] Deleted %d visual voxels (%.1f%%), remaining: %zu voxels",
                 deleted_voxels,
                 100.0 * deleted_voxels / std::max(1, total_voxels_before),
                 feat_map.size());
    }
}


void VIOManager::cleanupOldFrames() {
    
    while (frame_history_.size() > max_frame_history) {
        FramePtr old_frame = frame_history_.front();
        frame_history_.pop_front();

        
        if (old_frame) {
            old_frame->imgs_shared_.clear();
            
        }
    }
}

void VIOManager::addCrossCameraConsistencyConstraint(
        VisualPoint* pt, int source_cam_id, int target_cam_id,
        MatrixXd& H_sub, VectorXd& z, int level, int& row_offset) {

    if (!pt || !enable_cross_camera_tracking ||
        source_cam_id < 0 || source_cam_id >= (int)cams.size() ||
        target_cam_id < 0 || target_cam_id >= (int)cams.size() ||
        source_cam_id == target_cam_id) {
        return;
    }

    
    if (source_cam_id >= (int)new_frame_->imgs_.size() ||
        target_cam_id >= (int)new_frame_->imgs_.size() ||
        new_frame_->imgs_[source_cam_id].empty() ||
        new_frame_->imgs_[target_cam_id].empty()) {
        return;
    }

    
    V2D source_px = new_frame_->w2c(pt->pos_, source_cam_id);
    V2D target_px = new_frame_->w2c(pt->pos_, target_cam_id);

    if (!cams[source_cam_id]->isInFrame(source_px.cast<int>(), border) ||
        !cams[target_cam_id]->isInFrame(target_px.cast<int>(), border)) {
        return;
    }

    
    
    float source_patch[256];  
    float target_patch[256];
    memset(source_patch, 0, sizeof(source_patch));
    memset(target_patch, 0, sizeof(target_patch));

    
    getImagePatch(new_frame_->imgs_[source_cam_id], source_px, source_patch, level);
    getImagePatch(new_frame_->imgs_[target_cam_id], target_px, target_patch, level);

    
    if (source_cam_id >= (int)camera_photo_params.size() ||
        target_cam_id >= (int)camera_photo_params.size()) {
        return;
    }

    
    float mean_source = 0, mean_target = 0;
    for (int i = 0; i < patch_size_total; i++) {
        mean_source += source_patch[i];
        mean_target += target_patch[i];
    }
    mean_source /= patch_size_total;
    mean_target /= patch_size_total;

    
    if (mean_source > 20 && mean_source < 235 &&
        mean_target > 20 && mean_target < 235) {

        double ratio = mean_target / mean_source;
        double alpha = 0.005; 

        
        if (source_cam_id == 0 && target_cam_id > 0) {
            camera_photo_params[target_cam_id].exposure_factor =
                (1-alpha) * camera_photo_params[target_cam_id].exposure_factor +
                alpha * ratio;
        } else if (target_cam_id == 0 && source_cam_id > 0) {
            camera_photo_params[source_cam_id].exposure_factor =
                (1-alpha) * camera_photo_params[source_cam_id].exposure_factor +
                alpha / ratio;
        }
    }

    double source_to_target_factor = camera_photo_params[target_cam_id].exposure_factor /
                                     camera_photo_params[source_cam_id].exposure_factor;

    
    double source_inv_expo = (source_cam_id < state->inv_expo_time_per_cam.size())
                           ? state->inv_expo_time_per_cam[source_cam_id] : 1.0;
    double target_inv_expo = (target_cam_id < state->inv_expo_time_per_cam.size())
                           ? state->inv_expo_time_per_cam[target_cam_id] : 1.0;
    
    V3D p_source = new_frame_->w2f(pt->pos_, source_cam_id);
    V3D p_target = new_frame_->w2f(pt->pos_, target_cam_id);

    
    if (p_source.z() < 1e-6 || p_target.z() < 1e-6) {
        return;
    }

    
    MD(2, 3) Jdpi_source, Jdpi_target;
    computeProjectionJacobian(source_cam_id, p_source, Jdpi_source);
    computeProjectionJacobian(target_cam_id, p_target, Jdpi_target);

    M3D p_hat_source, p_hat_target;
    p_hat_source << SKEW_SYM_MATRX(p_source);
    p_hat_target << SKEW_SYM_MATRX(p_target);

    float scale = 1 << level;
    float inv_scale = 1.0f / scale;

    
    cv::Mat &source_img = new_frame_->imgs_[source_cam_id];
    cv::Mat &target_img = new_frame_->imgs_[target_cam_id];

    
    for (int px_y = 0; px_y < patch_size; px_y++) {
        for (int px_x = 0; px_x < patch_size; px_x++) {
            int patch_idx = px_y * patch_size + px_x;

            
            float source_intensity, target_intensity;
            if (exposure_estimate_en) {
                
                
                
                source_intensity = source_inv_expo * source_patch[patch_idx];
                target_intensity = target_inv_expo * target_patch[patch_idx];  
            } else {
                
                source_intensity = source_patch[patch_idx];
                target_intensity = target_patch[patch_idx] / source_to_target_factor;
            }
            float residual = source_intensity - target_intensity;
            float abs_res = std::abs(residual);

            
            float weight = (abs_res <= outlier_threshold) ?
                           (1.0f - (residual/outlier_threshold)*(residual/outlier_threshold)) *
                           (1.0f - (residual/outlier_threshold)*(residual/outlier_threshold)) : 0.0f;
            float sqrt_weight = std::sqrt(weight);

            
            int u_source = std::floor(source_px[0]);
            int v_source = std::floor(source_px[1]);
            float du_source = 0, dv_source = 0;

            if (u_source-1 >= 0 && u_source+1 < source_img.cols &&
                v_source-1 >= 0 && v_source+1 < source_img.rows) {
                du_source = (float)(source_img.at<uint8_t>(v_source, u_source+1) -
                                   source_img.at<uint8_t>(v_source, u_source-1)) * 0.5f;
                dv_source = (float)(source_img.at<uint8_t>(v_source+1, u_source) -
                                   source_img.at<uint8_t>(v_source-1, u_source)) * 0.5f;
            }

            
            int u_target = std::floor(target_px[0]);
            int v_target = std::floor(target_px[1]);
            float du_target = 0, dv_target = 0;

            if (u_target-1 >= 0 && u_target+1 < target_img.cols &&
                v_target-1 >= 0 && v_target+1 < target_img.rows) {
                du_target = (float)(target_img.at<uint8_t>(v_target, u_target+1) -
                                   target_img.at<uint8_t>(v_target, u_target-1)) * 0.5f;
                dv_target = (float)(target_img.at<uint8_t>(v_target+1, u_target) -
                                   target_img.at<uint8_t>(v_target-1, u_target)) * 0.5f;
            }

            
            MD(1, 2) Jimg_source;
            Jimg_source << du_source, dv_source;
            Jimg_source *= (float)source_inv_expo;  
            Jimg_source *= inv_scale;

            MD(1, 3) Jdphi_source = Jimg_source * Jdpi_source * p_hat_source;
            MD(1, 3) Jdp_source = -Jimg_source * Jdpi_source;

            MD(1, 3) JdR_source = Jdphi_source * Jdphi_dR_vec[source_cam_id] +
                                  Jdp_source * Jdp_dR_vec[source_cam_id];
            MD(1, 3) Jdt_source = Jdp_source * Jdp_dt_vec[source_cam_id];

            
            MD(1, 2) Jimg_target;
            Jimg_target << du_target, dv_target;
            Jimg_target *= (float)target_inv_expo;  
            Jimg_target *= inv_scale;  

            MD(1, 3) Jdphi_target = Jimg_target * Jdpi_target * p_hat_target;
            MD(1, 3) Jdp_target = -Jimg_target * Jdpi_target;

            MD(1, 3) JdR_target = Jdphi_target * Jdphi_dR_vec[target_cam_id] +
                                  Jdp_target * Jdp_dR_vec[target_cam_id];
            MD(1, 3) Jdt_target = Jdp_target * Jdp_dt_vec[target_cam_id];

            
            
            
            
            MD(1, 3) JdR_coupled = JdR_source - JdR_target;
            MD(1, 3) Jdt_coupled = Jdt_source - Jdt_target;

            
            int curr_row = row_offset + patch_idx;
            if (curr_row >= H_sub.rows() || curr_row >= z.size()) {
                continue;
            }

            H_sub(curr_row, 0) = sqrt_weight * JdR_coupled(0);
            H_sub(curr_row, 1) = sqrt_weight * JdR_coupled(1);
            H_sub(curr_row, 2) = sqrt_weight * JdR_coupled(2);
            H_sub(curr_row, 3) = sqrt_weight * Jdt_coupled(0);
            H_sub(curr_row, 4) = sqrt_weight * Jdt_coupled(1);
            H_sub(curr_row, 5) = sqrt_weight * Jdt_coupled(2);

            
            if (exposure_estimate_en) {
                
                
                
                
                float expo_jac_source = source_patch[patch_idx];
                float expo_jac_target = -target_patch[patch_idx];  

                int source_expo_col = 6 + source_cam_id;
                int target_expo_col = 6 + target_cam_id;
                H_sub(curr_row, source_expo_col) = sqrt_weight * expo_jac_source;
                H_sub(curr_row, target_expo_col) = sqrt_weight * expo_jac_target;
            }
            z(curr_row) = sqrt_weight * residual;
        }
    }

    
    row_offset += patch_size_total;
}
void VIOManager::processFrame(const std::vector<cv::Mat> &imgs,
                              std::vector<pointWithVar> &pg,
                              const std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map, double frame_timestamp)
{
    
    current_timestamp_ = frame_timestamp;

    if (imgs.size() != cams.size()) {
        ROS_ERROR( "[ VIO ] Error: input imgs.size() != cams.size().  is  %zu %zu", imgs.size(), cams.size());
        return;
    }
    imgs_cp.resize(cams.size());
    imgs_rgb.resize(cams.size());
    for (size_t i = 0; i < cams.size(); i++) {
        if (imgs[i].empty()) {
            ROS_WARN_STREAM("Camera " << i << " image is empty, skipping.");
            continue;
        }
        if (width != imgs[i].cols || height != imgs[i].rows) {
            cv::Mat resized;
            cv::resize(imgs[i], resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            imgs_cp[i]  = resized.clone();
            imgs_rgb[i] = resized.clone();
        } else {
            imgs_cp[i]  = imgs[i].clone();
            imgs_rgb[i] = imgs[i].clone();
        }
        if (imgs_cp[i].channels() == 3) {
            cv::cvtColor(imgs_cp[i], imgs_cp[i], cv::COLOR_BGR2GRAY);
        }
    }
    
    new_frame_.reset(new Frame(cams, imgs_cp, frame_timestamp));
    updateFrameState(*state);

    
    frame_history_.push_back(new_frame_);
    cleanupOldFrames();

    double t1 = omp_get_wtime();
    retrieveFromVisualSparseMap(imgs_cp, pg, feat_map);
    double t2 = omp_get_wtime();
    computeJacobianAndUpdateEKF(imgs_cp);
    double t3 = omp_get_wtime();

    generateVisualMapPoints(imgs_cp, pg);
    double t4 = omp_get_wtime();
    plotTrackedPoints();
    if (plot_flag)
        projectPatchFromRefToCur(feat_map);
    double t5 = omp_get_wtime();
    updateVisualMapPoints(imgs_cp);
    double t6 = omp_get_wtime();
    updateReferencePatch(feat_map);

    
    if (new_frame_ && (new_frame_->id_ - last_cleanup_frame_id) >= cleanup_interval_frames) {
        cleanupOldVisualPoints();
    }

    double t7 = omp_get_wtime();


    if (colmap_output_en)
        dumpDataForColmap();

    frame_count++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;


    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m|                         VIO Time (MultiCam)                 |\033[0m\n");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
    printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
    printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
    printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
}

