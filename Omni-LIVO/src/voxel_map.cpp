/**
 * @file voxel_map.cpp
 * @brief Voxel Map implementation for Omni-LIVO
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

#include "voxel_map.h"

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(ros::NodeHandle &nh, VoxelMapConfig &voxel_config)
{
  nh.param<bool>("publish/pub_plane_en", voxel_config.is_pub_plane_map_, false);

  nh.param<int>("lio/max_layer", voxel_config.max_layer_, 1);
  nh.param<double>("lio/voxel_size", voxel_config.max_voxel_size_, 0.5);
  nh.param<double>("lio/min_eigen_value", voxel_config.planner_threshold_, 0.01);
  nh.param<double>("lio/sigma_num", voxel_config.sigma_num_, 3);
  nh.param<double>("lio/beam_err", voxel_config.beam_err_, 0.02);
  nh.param<double>("lio/dept_err", voxel_config.dept_err_, 0.05);
  nh.param<vector<int>>("lio/layer_init_num", voxel_config.layer_init_num_, vector<int>{5,5,5,5,5});
  nh.param<int>("lio/max_points_num", voxel_config.max_points_num_, 50);
  nh.param<int>("lio/max_iterations", voxel_config.max_iterations_, 5);

  nh.param<bool>("local_map/map_sliding_en", voxel_config.map_sliding_en_, false);
  nh.param<double>("local_map/half_map_size", voxel_config.half_map_size_, 100.0);
  nh.param<double>("local_map/sliding_thresh", voxel_config.sliding_thresh_, 8.0);
  nh.param<int>("local_map/max_visual_voxels", voxel_config.max_visual_voxels_, 10000);
}

size_t VoxelMapManager::mapSliding(const V3D& pos)
{
  if (!config_setting_.map_sliding_en_) return 0;

  if (!local_map_inited_) {
    local_map_origin_ = pos;
    local_map_inited_ = true;
    return 0;
  }

  if ((pos - local_map_origin_).norm() < config_setting_.sliding_thresh_) return 0;

  const double voxel_size = config_setting_.max_voxel_size_;
  const double half = config_setting_.half_map_size_;
  const int64_t x_min = (int64_t)std::floor((pos.x() - half) / voxel_size);
  const int64_t x_max = (int64_t)std::floor((pos.x() + half) / voxel_size);
  const int64_t y_min = (int64_t)std::floor((pos.y() - half) / voxel_size);
  const int64_t y_max = (int64_t)std::floor((pos.y() + half) / voxel_size);
  const int64_t z_min = (int64_t)std::floor((pos.z() - half) / voxel_size);
  const int64_t z_max = (int64_t)std::floor((pos.z() + half) / voxel_size);

  size_t erased = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); ) {
    const VOXEL_LOCATION& loc = it->first;
    if (loc.x < x_min || loc.x > x_max ||
        loc.y < y_min || loc.y > y_max ||
        loc.z < z_min || loc.z > z_max) {
      delete it->second;
      it = voxel_map_.erase(it);
      ++erased;
    } else {
      ++it;
    }
  }
  local_map_origin_ = pos;
  std::cout << "[ LIO ] Sliding map: erased " << erased
            << " voxels, remaining " << voxel_map_.size() << std::endl;
  return erased;
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::init_octo_tree()
{
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cut_octo_tree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree()
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      // 继承父体素的创建时间戳
      leaves_[leafnum]->creation_timestamp_ = creation_timestamp_;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          init_plane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          // 继承父体素的创建时间戳
          leaves_[leafnum]->creation_timestamp_ = creation_timestamp_;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      // 继承父体素的创建时间戳
      leaves_[leafnum]->creation_timestamp_ = creation_timestamp_;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

void VoxelMapManager::StateEstimation(StatesGroup &state_propagat)
{
    cross_mat_list_.clear();
    cross_mat_list_.reserve(feats_down_size_);
    body_cov_list_.clear();
    body_cov_list_.reserve(feats_down_size_);

    for (size_t i = 0; i < feats_down_body_->size(); i++)
    {
        V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
        if (point_this[2] == 0) { point_this[2] = 0.001; }
        M3D var;
        calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
        body_cov_list_.push_back(var);
        point_this = extR_ * point_this + extT_;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);
        cross_mat_list_.push_back(point_crossmat);
    }

    vector<pointWithVar>().swap(pv_list_);
    pv_list_.resize(feats_down_size_);

    int rematch_num = 0;
    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
    {
        double total_residual = 0.0;
        pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);

        // **简化：直接使用当前状态变换点云**
        // 因为体素已经在回环校正后的坐标系中，不需要动态变换
        TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);

        M3D rot_var = state_.cov.block<3, 3>(0, 0);
        M3D t_var = state_.cov.block<3, 3>(3, 3);

        for (size_t i = 0; i < feats_down_body_->size(); i++)
        {
            pointWithVar &pv = pv_list_[i];
            pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
            pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

            M3D cov = body_cov_list_[i];
            M3D point_crossmat = cross_mat_list_[i];
            cov = state_.rot_end * cov * state_.rot_end.transpose() +
                  (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
            pv.var = cov;
            pv.body_var = body_cov_list_[i];
        }

        ptpl_list_.clear();

        // 使用当前活动地图（现在已经是校正后的坐标系）
        auto& active_map = getActiveMap();
        BuildResidualListWithMap(pv_list_, ptpl_list_, active_map);

        for (int i = 0; i < ptpl_list_.size(); i++)
        {
            total_residual += fabs(ptpl_list_[i].dis_to_plane_);
        }
        effct_feat_num_ = ptpl_list_.size();

        // **后续EKF更新逻辑保持不变**
        MatrixXd Hsub(effct_feat_num_, 6);
        MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
        VectorXd R_inv(effct_feat_num_);
        VectorXd meas_vec(effct_feat_num_);
        meas_vec.setZero();

        for (int i = 0; i < effct_feat_num_; i++)
        {
            auto &ptpl = ptpl_list_[i];
            V3D point_this(ptpl.point_b_);
            point_this = extR_ * point_this + extT_;
            V3D point_body(ptpl.point_b_);
            M3D point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_this);

            V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
            J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;

            M3D var;
            var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();

            double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();
            R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);

            V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
            Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
            Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i),
                    ptpl_list_[i].normal_[0] * R_inv(i),
                    ptpl_list_[i].normal_[1] * R_inv(i),
                    ptpl_list_[i].normal_[2] * R_inv(i);
            meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
        }

        bool EKF_stop_flg = false;
        bool flg_EKF_converged = false;

        // EKF更新逻辑保持不变
        MatrixXd K(DIM_STATE, effct_feat_num_);
        auto &&HTz = Hsub_T_R_inv * meas_vec;
        H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
        MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) +
                                          state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
        G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
        auto vec = state_propagat - state_;
        VD(DIM_STATE) solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz +
                                 vec.block<DIM_STATE, 1>(0, 0) -
                                 G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

        state_ += solution;
        auto rot_add = solution.block<3, 1>(0, 0);
        auto t_add = solution.block<3, 1>(3, 0);
        if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) {
            flg_EKF_converged = true;
        }

        V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

        if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2)))) {
            rematch_num++;
        }

        if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
        {
            state_.cov.block<DIM_STATE, DIM_STATE>(0, 0) =
                    (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) -
                     G.block<DIM_STATE, DIM_STATE>(0, 0)) * state_.cov.block<DIM_STATE, DIM_STATE>(0, 0);
            position_last_ = state_.pos_end;
            geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
            EKF_stop_flg = true;
        }
        if (EKF_stop_flg) break;
    }
}

void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void VoxelMapManager::BuildVoxelMap()
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  std::vector<pointWithVar> input_points;

  for (size_t i = 0; i < feats_down_world_->size(); i++)
  {
    pointWithVar pv;
    pv.point_w << feats_down_world_->points[i].x, feats_down_world_->points[i].y, feats_down_world_->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    var = (state_.rot_end * extR_) * var * (state_.rot_end * extR_).transpose() +
          (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state_.cov.block<3, 3>(3, 3);
    pv.var = var;
    input_points.push_back(pv);
  }

  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.point_w[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->init_octo_tree();
  }
}

V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)
{
  int64_t loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = floor(input_point[j] / config_setting_.max_voxel_size_);
  }

  VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  int64_t ind = loc_xyz[0] + loc_xyz[1] + loc_xyz[2];
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}
void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points, double timestamp) {
    if (input_points.empty()) {
        ROS_WARN("UpdateVoxelMap called with empty point list");
        return;
    }

    // Use the active map (now we always use voxel_map_ since we don't create a new map)
    auto& current_map = voxel_map_;

    float voxel_size = config_setting_.max_voxel_size_;
    float planer_threshold = config_setting_.planner_threshold_;
    int max_layer = config_setting_.max_layer_;
    int max_points_num = config_setting_.max_points_num_;
    std::vector<int> layer_init_num = config_setting_.layer_init_num_;

    uint plsize = input_points.size();
    int processed_count = 0;

    try {
        for (uint i = 0; i < plsize; i++) {
            const pointWithVar p_v_orig = input_points[i];

            // Skip invalid points
            if (!p_v_orig.point_w.allFinite()) {
                continue;
            }

            // Transform the point to map coordinates if we have a correction
            Eigen::Vector3d map_point = worldToMapCoordinates(p_v_orig.point_w);

            // Create a copy of the point with transformed coordinates
            pointWithVar p_v = p_v_orig;
            p_v.point_w = map_point;

            // The rest of the voxel update logic remains unchanged
            float loc_xyz[3];
            for (int j = 0; j < 3; j++) {
                loc_xyz[j] = p_v.point_w[j] / voxel_size;
                if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
            }

            VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            auto iter = current_map.find(position);

            if (iter != current_map.end()) {
                // Existing voxel
                if (iter->second) {  // Check for null pointer
                    iter->second->UpdateOctoTree(p_v);
                    processed_count++;
                }
            } else {
                // Create new voxel
                try {
                    VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
                    if (octo_tree) {
                        octo_tree->quater_length_ = voxel_size / 4;
                        octo_tree->voxel_center_[0] = (0.5 + position.x) * voxel_size;
                        octo_tree->voxel_center_[1] = (0.5 + position.y) * voxel_size;
                        octo_tree->voxel_center_[2] = (0.5 + position.z) * voxel_size;
                        octo_tree->layer_init_num_ = layer_init_num;

                        // 使用点云的时间戳作为体素创建时间
                        octo_tree->creation_timestamp_ = timestamp;

                        octo_tree->UpdateOctoTree(p_v);
                        current_map[position] = octo_tree;
                        processed_count++;
                    }
                } catch (const std::exception& e) {
                    ROS_ERROR("Exception creating voxel: %s", e.what());
                    // Continue processing other points
                }
            }
        }

        ROS_INFO("UpdateVoxelMap processed %d points out of %d at timestamp %.6f",
                 processed_count, plsize, timestamp);
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in UpdateVoxelMap: %s", e.what());
    } catch (...) {
        ROS_ERROR("Unknown exception in UpdateVoxelMap");
    }
}

void VoxelMapManager::BuildResidualListWithMap(
        std::vector<pointWithVar> &pv_list,
        std::vector<PointToPlane> &ptpl_list,
        std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &map_to_use) {

    // 诊断信息
    static int debug_call_count = 0;
    debug_call_count++;
    bool should_debug = (debug_call_count % 10 == 1);


    int max_layer = config_setting_.max_layer_;
    double voxel_size = config_setting_.max_voxel_size_;
    double sigma_num = config_setting_.sigma_num_;
    std::mutex mylock;
    ptpl_list.clear();
    std::vector<PointToPlane> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size());
    std::vector<size_t> index(pv_list.size());

    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
        useful_ptpl[i] = false;
    }

    // 诊断计数
    int found_voxel_count = 0;
    int successful_match_count = 0;
    int coordinate_transform_count = 0;
    int near_voxel_successes = 0;

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < index.size(); i++) {
        pointWithVar &pv = pv_list[i];

        try {
            // **关键恢复：坐标变换逻辑**
            Eigen::Vector3d map_point = worldToMapCoordinates(pv.point_w);

            if (has_correction_) {
                coordinate_transform_count++;
            }



            float loc_xyz[3];
            for (int j = 0; j < 3; j++) {
                loc_xyz[j] = map_point[j] / voxel_size;
                if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
            }

            VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

            auto iter = map_to_use.find(position);
            bool match_found = false;
            PointToPlane single_ptpl;

            if (iter != map_to_use.end() && iter->second != nullptr) {
                found_voxel_count++;
                VoxelOctoTree *current_octo = iter->second;
                bool is_sucess = false;
                double prob = 0;

                // **关键恢复：使用体素时间戳构建残差**
                build_single_residual(pv, current_octo, 0, is_sucess, prob, single_ptpl);

                if (is_sucess) {
                    match_found = true;
                    successful_match_count++;
                } else {
                    // **恢复：原始的近邻搜索逻辑**
                    VOXEL_LOCATION near_position = position;
                    if (map_point[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
                        near_position.x = near_position.x + 1;
                    }
                    else if (map_point[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) {
                        near_position.x = near_position.x - 1;
                    }
                    if (map_point[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
                        near_position.y = near_position.y + 1;
                    }
                    else if (map_point[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) {
                        near_position.y = near_position.y - 1;
                    }
                    if (map_point[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
                        near_position.z = near_position.z + 1;
                    }
                    else if (map_point[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) {
                        near_position.z = near_position.z - 1;
                    }

                    auto iter_near = map_to_use.find(near_position);
                    if (iter_near != map_to_use.end() && iter_near->second != nullptr) {
                        build_single_residual(pv, iter_near->second, 0, is_sucess, prob, single_ptpl);
                        if (is_sucess) {
                            match_found = true;
                            near_voxel_successes++;
                            successful_match_count++;
                        }
                    }
                }
            }

            if (match_found) {
                // **关键恢复：正确的平面参数变换逻辑**
                if (has_correction_) {
                    // 获取体素特定的时间戳变换
                    double voxel_timestamp = single_ptpl.voxel_timestamp_;

                    try {
                        // **关键修复：使用mapToWorldCoordinates将平面参数转换回世界坐标系**
                        single_ptpl.normal_ = mapToWorldCoordinates(single_ptpl.normal_, voxel_timestamp);
                        single_ptpl.center_ = mapToWorldCoordinates(single_ptpl.center_, voxel_timestamp);

                        // 重新计算平面方程常数项
                        single_ptpl.d_ = -(single_ptpl.normal_.dot(single_ptpl.center_));

                        if (should_debug && i < 2) {
                            ROS_INFO("Point %d - Transformed plane normal: [%.3f, %.3f, %.3f], center: [%.3f, %.3f, %.3f]",
                                     i, single_ptpl.normal_.x(), single_ptpl.normal_.y(), single_ptpl.normal_.z(),
                                     single_ptpl.center_.x(), single_ptpl.center_.y(), single_ptpl.center_.z());
                        }
                    } catch (const std::exception& e) {
                        ROS_ERROR_THROTTLE(1.0, "Transform error for point %d: %s", i, e.what());
                        match_found = false;
                    }
                }

                if (match_found) {
                    std::lock_guard<std::mutex> lock(mylock);
                    useful_ptpl[i] = true;
                    all_ptpl_list[i] = single_ptpl;
                }
            }

        } catch (const std::exception& e) {
            ROS_ERROR_THROTTLE(1.0, "Exception processing point %d: %s", i, e.what());
            std::lock_guard<std::mutex> lock(mylock);
            useful_ptpl[i] = false;
        }
    }

    // 构建最终结果
    try {
        ptpl_list.reserve(pv_list.size());
        for (size_t i = 0; i < useful_ptpl.size(); i++) {
            if (useful_ptpl[i]) {
                ptpl_list.push_back(all_ptpl_list[i]);
            }
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Exception building final ptpl_list: %s", e.what());
        ptpl_list.clear();
    }


}

void VoxelMapManager::build_single_residual(
        pointWithVar &pv, const VoxelOctoTree *current_octo,
        const int current_layer, bool &is_sucess,
        double &prob, PointToPlane &single_ptpl) {

    int max_layer = config_setting_.max_layer_;
    double sigma_num = config_setting_.sigma_num_;
    double radius_k = 3;

    // 使用体素时间戳进行坐标变换
    double voxel_timestamp = current_octo->creation_timestamp_;
    Eigen::Vector3d p_w_map = worldToMapCoordinates(pv.point_w, voxel_timestamp);

    if (current_octo->plane_ptr_->is_plane_) {
        VoxelPlane &plane = *current_octo->plane_ptr_;
        Eigen::Vector3d p_world_to_center = p_w_map - plane.center_;
        float dis_to_plane = fabs(plane.normal_(0) * p_w_map(0) +
                                  plane.normal_(1) * p_w_map(1) +
                                  plane.normal_(2) * p_w_map(2) + plane.d_);
        float dis_to_center = (plane.center_(0) - p_w_map(0)) * (plane.center_(0) - p_w_map(0)) +
                              (plane.center_(1) - p_w_map(1)) * (plane.center_(1) - p_w_map(1)) +
                              (plane.center_(2) - p_w_map(2)) * (plane.center_(2) - p_w_map(2));
        float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

        if (range_dis <= radius_k * plane.radius_) {
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w_map - plane.center_;
            J_nq.block<1, 3>(0, 3) = -plane.normal_;
            double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
            sigma_l += plane.normal_.transpose() * pv.var * plane.normal_;

            if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
                is_sucess = true;
                double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);

                if (this_prob > prob) {
                    prob = this_prob;
                    pv.normal = plane.normal_;
                    single_ptpl.body_cov_ = pv.body_var;
                    single_ptpl.point_b_ = pv.point_b;
                    single_ptpl.point_w_ = pv.point_w;
                    single_ptpl.plane_var_ = plane.plane_var_;

                    // 存储平面参数（地图坐标系）
                    single_ptpl.normal_ = plane.normal_;
                    single_ptpl.center_ = plane.center_;
                    single_ptpl.d_ = plane.d_;
                    single_ptpl.layer_ = current_layer;
                    single_ptpl.voxel_timestamp_ = voxel_timestamp;  // 保存体素时间戳

                    // 使用变换后的点计算平面距离
                    single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w_map(0) +
                                                plane.normal_(1) * p_w_map(1) +
                                                plane.normal_(2) * p_w_map(2) + plane.d_;
                }
                return;
            } else {
                return;
            }
        } else {
            return;
        }
    } else {
        if (current_layer < max_layer) {
            for (size_t leafnum = 0; leafnum < 8; leafnum++) {
                if (current_octo->leaves_[leafnum] != nullptr) {
                    VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
                    build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl);
                }
            }
            return;
        } else {
            return;
        }
    }
}






void VoxelMapManager::updateVoxelGeometry(VoxelOctoTree* voxel, const SE3& transform) {
    if (!voxel || !voxel->plane_ptr_) return;

    VoxelPlane* plane = voxel->plane_ptr_;

    // 1. 变换平面参数（保持原有逻辑）
    if (plane->is_plane_ && plane->is_init_) {
        // 变换平面中心
        plane->center_ = transform * plane->center_;

        // 变换平面法向量（只旋转，不平移）
        plane->normal_ = transform.rotation_matrix() * plane->normal_;
        plane->x_normal_ = transform.rotation_matrix() * plane->x_normal_;
        plane->y_normal_ = transform.rotation_matrix() * plane->y_normal_;

        // 重新计算平面方程常数项
        plane->d_ = -(plane->normal_.dot(plane->center_));

        // 变换协方差矩阵
        Eigen::Matrix3d R = transform.rotation_matrix();
        plane->covariance_ = R * plane->covariance_ * R.transpose();

        // 变换平面方差矩阵（6x6）
        Eigen::Matrix<double, 6, 6> T_adj = Eigen::Matrix<double, 6, 6>::Identity();
        T_adj.block<3, 3>(0, 0) = R;  // 旋转部分
        T_adj.block<3, 3>(3, 3) = R;  // 平移部分
        plane->plane_var_ = T_adj * plane->plane_var_ * T_adj.transpose();

        // 标记需要更新
        plane->is_update_ = true;
    }

    // **🆕 关键添加：同步变换关联的点云**
    if (!voxel->temp_points_.empty()) {
        try {
            for (auto& point_var : voxel->temp_points_) {
                // 检查点是否有效
                if (!point_var.point_w.allFinite()) {
                    continue;
                }

                // 变换世界坐标点
                Eigen::Vector3d old_point = point_var.point_w;
                point_var.point_w = transform * old_point;

                // 变换协方差矩阵（旋转变换）
                Eigen::Matrix3d R = transform.rotation_matrix();
                point_var.var = R * point_var.var * R.transpose();

                // 注意：point_b（body坐标）不需要变换，因为它是传感器坐标系下的固定值
                // body_var 也不需要变换，因为它是传感器不确定性
            }

            // 添加调试信息（限流）
            static int transform_count = 0;
            if (++transform_count % 100 == 1) {
                ROS_INFO("Transformed %zu points in voxel (total transforms: %d)",
                         voxel->temp_points_.size(), transform_count);
            }

        } catch (const std::exception& e) {
            ROS_ERROR("Exception transforming points in voxel: %s", e.what());
        }
    }

    // 3. 递归处理子体素（保持原有逻辑）
    if (voxel->layer_ < voxel->max_layer_) {
        for (int i = 0; i < 8; i++) {
            if (voxel->leaves_[i] != nullptr) {
                // 更新子体素中心
                Eigen::Vector3d old_leaf_center(voxel->leaves_[i]->voxel_center_[0],
                                                voxel->leaves_[i]->voxel_center_[1],
                                                voxel->leaves_[i]->voxel_center_[2]);
                Eigen::Vector3d new_leaf_center = transform * old_leaf_center;

                if (new_leaf_center.allFinite()) {
                    voxel->leaves_[i]->voxel_center_[0] = new_leaf_center.x();
                    voxel->leaves_[i]->voxel_center_[1] = new_leaf_center.y();
                    voxel->leaves_[i]->voxel_center_[2] = new_leaf_center.z();

                    // 递归更新子体素
                    updateVoxelGeometry(voxel->leaves_[i], transform);
                }
            }
        }
    }
}
void VoxelMapManager::initVoxelMapPublisher(ros::NodeHandle& nh) {
    // 只有在启用平面发布时才创建定时器
    if (config_setting_.is_pub_plane_map_) {
        voxel_map_timer_ = nh.createTimer(ros::Duration(1.0),
                                          [this](const ros::TimerEvent&) {
                                              if (need_clear_markers_) {
                                                  clearOldMarkers();
                                                  need_clear_markers_ = false;
                                              }
                                              pubVoxelMap();
                                          });
        ROS_INFO("VoxelMap publisher initialized at 1Hz");
    }
}



void VoxelMapManager::clearOldMarkers() {
    visualization_msgs::MarkerArray clear_markers;
    visualization_msgs::Marker delete_marker;
    delete_marker.header.frame_id = "camera_init";
    delete_marker.header.stamp = ros::Time::now();
    delete_marker.ns = "plane";
    delete_marker.action = visualization_msgs::Marker::DELETEALL;
    clear_markers.markers.push_back(delete_marker);
    voxel_map_pub_.publish(clear_markers);
    ros::Duration(0.1).sleep(); // 短暂等待确保删除完成
}

void VoxelMapManager::pubVoxelMap() {
    if (voxel_map_.empty()) {
        ROS_WARN("voxel_map_ is empty, size: %zu", voxel_map_.size());
        return;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    std::vector<PlaneWithVoxel> pub_plane_list;
    for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++) {
        GetUpdatePlaneWithVoxel(iter->second, config_setting_.max_layer_, pub_plane_list);
    }

    int valid_planes = 0;
    int corrected_planes = 0;
    int uncorrected_planes = 0;

    for (size_t i = 0; i < pub_plane_list.size(); i++) {
        const PlaneWithVoxel& pwv = pub_plane_list[i];
        if (!pwv.plane.is_plane_) continue;

        pcl::PointXYZRGB point;

        // 先记录原始中心点
        Eigen::Vector3d original_center = pwv.plane.center_;

            point.x = original_center[0];
            point.y = original_center[1];
            point.z = original_center[2];
            uncorrected_planes++;

        // 计算颜色（保持原有逻辑）
        V3D plane_cov = pwv.plane.plane_var_.block<3, 3>(0, 0).diagonal();
        double trace = plane_cov.sum();
        double max_trace = 0.25;
        if (trace >= max_trace) { trace = max_trace; }
        trace = trace * (1.0 / max_trace);
        trace = pow(trace, 0.2);

        uint8_t r, g, b;
        mapJet(trace, 0, 1, r, g, b);
        point.r = r;
        point.g = g;
        point.b = b;

        voxel_cloud->push_back(point);
        valid_planes++;
    }

    // 发布点云
    if (!voxel_cloud->empty()) {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*voxel_cloud, cloud_msg);
        cloud_msg.header.frame_id = "camera_init";
        cloud_msg.header.stamp = ros::Time::now();

        voxel_pointcloud_pub_.publish(cloud_msg);
    } else {
        ROS_WARN("voxel_cloud is empty, not publishing");
    }
}
void VoxelMapManager::saveVoxelMapPCD(const std::string& suffix) {
    if (voxel_map_.empty()) return;

    // 创建保存目录
    std::string save_dir = std::string(ROOT_DIR) + "Log/";
    system(("mkdir -p " + save_dir).c_str());

    // 文件名
    std::string filename = save_dir + "voxel_map_" + suffix + ".pcd";

    // 创建点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (const auto& [location, voxel] : voxel_map_) {
        if (voxel) {
            pcl::PointXYZ point;
            point.x = voxel->voxel_center_[0];
            point.y = voxel->voxel_center_[1];
            point.z = voxel->voxel_center_[2];
            cloud->push_back(point);
        }
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;

    pcl::io::savePCDFileBinary(filename, *cloud);
    ROS_INFO("Saved %zu voxels to: %s", cloud->size(), filename.c_str());
}
void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}
void VoxelMapManager::GetUpdatePlaneWithVoxel(const VoxelOctoTree *current_octo,
                                              const int pub_max_voxel_layer,
                                              std::vector<PlaneWithVoxel> &plane_list) {
    if (current_octo->layer_ > pub_max_voxel_layer) {
        return;
    }

    if (current_octo->plane_ptr_->is_update_) {
        PlaneWithVoxel pwv;
        pwv.plane = *current_octo->plane_ptr_;
        pwv.voxel_ptr = current_octo;  // 保存体素指针
        plane_list.push_back(pwv);
    }

    if (current_octo->layer_ < current_octo->max_layer_) {
        if (!current_octo->plane_ptr_->is_plane_) {
            for (size_t i = 0; i < 8; i++) {
                if (current_octo->leaves_[i] != nullptr) {
                    GetUpdatePlaneWithVoxel(current_octo->leaves_[i], pub_max_voxel_layer, plane_list);
                }
            }
        }
    }
    return;
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

SE3 VoxelMapManager::getTransformForTimestamp(double timestamp) const {
    if (!has_trajectory_transform_ || !trajectory_transform_function_) {
        return SE3(); // 返回单位变换
    }

    // **关键修复：正确处理无效时间戳**
    if (timestamp < 0) {
        // 对于无效时间戳，假设是旧体素，使用匹配帧时间戳
        timestamp = match_frame_timestamp_;

        static int debug_count = 0;
        if (debug_count < 3) {  // 减少debug输出
            ROS_INFO("getTransformForTimestamp: Invalid timestamp, using match_frame_timestamp %.6f",
                     timestamp);
            debug_count++;
        }
    }

    try {
        SE3 result = trajectory_transform_function_(timestamp);
        return result;
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in getTransformForTimestamp: %s", e.what());
        return SE3(); // 返回单位变换
    }
}

Eigen::Vector3d VoxelMapManager::worldToMapCoordinates(const Eigen::Vector3d& point_world, double timestamp) const {
    if (!has_correction_) {
        return point_world;
    }

    // **修复：简化时间戳处理**
    double use_timestamp = (timestamp < 0) ? match_frame_timestamp_ : timestamp;
    SE3 transform = getTransformForTimestamp(use_timestamp);

    // **修复：减少debug输出，使用thread-safe计数器**
    static std::atomic<int> call_count{0};
    int current_call = call_count.fetch_add(1);

    // 只在前3次和每1000次调用时输出debug信息
    bool should_debug = (current_call < 3) || (current_call % 1000 == 0);

    // 逆向变换
    Eigen::Vector3d result = transform.inverse() * point_world;
    return result;
}
Eigen::Vector3d VoxelMapManager::mapToWorldCoordinates(const Eigen::Vector3d& point_map, double timestamp) const {
    if (!has_correction_) {
        return point_map;
    }

    double use_timestamp = (timestamp < 0) ? query_frame_timestamp_ : timestamp;
    SE3 transform = getTransformForTimestamp(use_timestamp);

    static int call_count = 0;
    call_count++;

    // **关键修正：从地图坐标系转换回世界坐标系**
    // 如果 point_search = T^(-1) * point_world，那么 point_world = T * point_search
    Eigen::Vector3d result = transform * point_map;

    return result;
}
