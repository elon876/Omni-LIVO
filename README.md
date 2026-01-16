# Omni-LIVO

**Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2509.15673)
[![Dataset](https://img.shields.io/badge/Dataset-Download-green)](https://pan.baidu.com/s/11SaVzvJGM1juWH-zAbGySg?pwd=328S)

This repository contains the official implementation of **Omni-LIVO**, a tightly-coupled multi-camera LiDAR-Inertial-Visual Odometry system that extends FAST-LIVO2 with multi-view photometric constraints for enhanced robustness and accuracy.

## News
- **[2026-01]** 🎉 Our paper has been accepted by IEEE Robotics and Automation Letters (RA-L)!
- **[2026-01]** 📦 Full source code is now publicly available
- **[2026-01]** 🗂️ Custom dataset with 3/4-camera configuration released

## Abstract

Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but existing LiDAR-inertial-visual odometry (LIVO) systems generally rely on a single camera, limiting their ability to fully exploit LiDAR-derived depth for photometric alignment and scene colorization. We present **Omni-LIVO**, a tightly coupled multi-camera LIVO system that leverages multi-view observations to comprehensively utilize LiDAR geometric information across extended spatial regions.

### Key Contributions

1. **Cross-View Temporal Migration**: A direct alignment strategy preserving photometric consistency across non-overlapping camera views
2. **Adaptive Multi-View ESIKF**: Enhanced Error-State Iterated Kalman Filter with adaptive covariance for improved robustness
3. **Extensive Evaluation**: Comprehensive testing on Hilti SLAM Challenge, Newer College, and custom datasets


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{cao2025omnilivo,
  title={Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion},
  author={Cao, Yinong and Zhang, Chenyang and He, Xin and Chen, Yuwei and Pu, Chengyu and Wang, Bingtao and Wu, Kaile and Zhu, Shouzheng and Han, Fei and Liu, Shijie and Li, Chunlai and Wang, Jianyu},
  year={2025},
  eprint={2509.15673},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  note={Accepted by IEEE Robotics and Automation Letters (RA-L)}
}
```

## Features

- ✨ **Multi-Camera Support**: Seamless integration of multiple cameras with non-overlapping FoVs
- 🔄 **Cross-View Migration**: Photometric patch tracking across different camera views
- 📊 **Adaptive Covariance**: Dynamic weighting based on real-time photometric error
- 🗺️ **Unified Voxel Map**: Integrated LiDAR geometry and multi-view visual observations
- 🎨 **Dense RGB Colorization**: 3.5× more colored points than FAST-LIVO2 in complex environments

## Omni-LIVO Dataset

**Download Link**: [Baidu Netdisk](https://pan.baidu.com/s/11SaVzvJGM1juWH-zAbGySg?pwd=328S)

### Sensor Configuration

Our multi-sensor data collection platform features the following specifications:

| **Component** | **Model/Type** | **Specifications** | **Frequency** |
|---------------|----------------|-------------------|---------------|
| **LiDAR** | LIVOX MID360 | • 360° horizontal FOV<br>• 59° vertical FOV<br>• 0.1-70m range<br>• 200,000 pts/s | 10 Hz |
| **IMU** | ICM40609 (built-in MID360) | • 6-DOF inertial measurement<br>• Accelerometer + Gyroscope | 200 Hz |
| **Camera Array** | JHEM306GC-HM | • 1024×768 resolution<br>• Cross-pattern configuration<br>• Hardware-level synchronization | 10 Hz |

### Camera Configuration

The four cameras are strategically arranged in a **cross-pattern configuration** to achieve near-omnidirectional visual coverage:

```
    Camera Front
         |
Camera ----- Camera
Left         Right
         |
    Camera Rear
```

This configuration synergistically complements the 360° LiDAR perception, enabling robust tracking even when objects transition between different camera views.

### Dataset Sequences

Our custom dataset includes diverse scenarios:
- **Indoor**: Corridors, basements, classrooms with varying illumination
- **Outdoor**: Open spaces with dynamic lighting conditions
- **Staircase**: Multi-level vertical transitions with loop closures
- **Mixed**: Indoor-outdoor transitions

## Environment Requirements

Omni-LIVO environment is compatible with FAST-LIVO2 requirements.

### Ubuntu and ROS
- Ubuntu 18.04 / 20.04
- [ROS Installation](http://wiki.ros.org/ROS/Installation)

### Dependencies

| Library | Version | Installation |
|---------|---------|--------------|
| PCL | >=1.8 | [PCL Installation](https://pointclouds.org/downloads/) |
| Eigen | >=3.3.4 | [Eigen Installation](https://eigen.tuxfamily.org/index.php?title=Main_Page) |
| OpenCV | >=4.2 | [OpenCV Installation](https://opencv.org/get-started/) |

### Sophus

Sophus installation for the non-templated/double-only version:

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```

### Vikit

Vikit contains camera models, math and interpolation functions. Download it into your catkin workspace source folder:

```bash
cd catkin_ws/src
git clone https://github.com/xuankuzcr/rpg_vikit.git
```

## Build

Clone the repository and build with catkin:

```bash
cd ~/catkin_ws/src
git clone https://github.com/elon876/Omni-LIVO.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## Configuration

### Multi-Camera Setup

1. **Camera Intrinsics**: Configure in `config/your_dataset_cam.yaml`
2. **Camera-to-LiDAR Extrinsics**: Set transformations in main config file
3. **ROS Topics**: Update image topics for each camera

Example multi-camera configuration:

```yaml
common:
  num_of_cameras: 4

extrin_calib:
  cameras:
    - cam_id: 0
      img_topic: "/cam0/image_raw"
      Rcl: [1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]
      Pcl: [0.0, 0.0, 0.0]
    - cam_id: 1
      img_topic: "/cam1/image_raw"
      Rcl: [rotation_matrix_3x3]
      Pcl: [translation_vector_3x1]
    # Additional cameras...
```

### VIO Parameters

```yaml
vio:
  adaptive_cov_en: true        # Enable adaptive covariance
  cross_view_migration_en: true # Enable cross-view temporal migration
  max_visual_points: 150       # Maximum visual points per frame
  photometric_threshold: 50.0  # Photometric error threshold
```

## Run

### On Our Custom Dataset

```bash
roslaunch fast_livo mapping_mid360.launch
rosbag play YOUR_DOWNLOADED.bag
```

### On Hilti SLAM Challenge

**⚠️ Important**: Hilti datasets have cameras running at 40Hz while Omni-LIVO requires 10Hz. You must first convert the bag files using the provided script:

```bash
# Step 1: Convert camera frame rate from 40Hz to 10Hz
python Omni-LIVO/scripts/cvt10hz.py /path/to/hilti/dataset/

# This will generate *_10hz.bag files with synchronized 10Hz cameras

# Step 2: Run Omni-LIVO
roslaunch fast_livo mapping_Hilti2022.launch
rosbag play hilti_dataset_10hz.bag
```

The `cvt10hz.py` script:
- Downsamples 4 camera topics from 40Hz to 10Hz
- Maintains temporal synchronization across all cameras
- Preserves LiDAR and IMU data unchanged
- Supports batch processing of multiple bag files


**Note**: Hilti datasets require preprocessing with `cvt10hz.py` script to convert cameras from 40Hz to 10Hz. See "Run → On Hilti SLAM Challenge" section for details.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Sensor Input                          │
│  LiDAR (10Hz) │ IMU (200Hz) │ Multi-Cam (10Hz)          │
└────────┬────────────┬────────────┬────────────────────────┘
         │            │            │
         v            v            v
┌────────────────────────────────────────────────────────┐
│              Temporal Synchronization                  │
└────────┬───────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────┐
│              IMU Propagation                           │
└────────┬───────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────┐
│        LiDAR Update (Point-to-Plane ESIKF)            │
│        └─ Voxel-based Plane Extraction                │
└────────┬───────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────┐
│         Multi-Camera VIO Update                        │
│  ├─ Patch Selection from Multi-View                   │
│  ├─ Cross-View Temporal Migration                     │
│  ├─ Adaptive Covariance Computation                   │
│  └─ Iterative ESIKF Update                            │
└────────┬───────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────┐
│        Unified Voxel Map Update                        │
│  └─ LiDAR Geometry + Multi-View RGB Patches           │
└────────────────────────────────────────────────────────┘
```



## Acknowledgments

We sincerely thank the authors of [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) for their outstanding work and for making their code open source. This project builds upon their excellent foundation in LiDAR-Inertial-Visual Odometry.

Special thanks to:
- The FAST-LIVO2 development team at HKU-MARS
- Hilti SLAM Challenge and Newer College Dataset organizers
- The entire open-source robotics and SLAM community

## License

This project is released under the same license as FAST-LIVO2. Please refer to the LICENSE file for details.

## Contact

For questions and collaboration opportunities, please contact:

- **Yinong Cao** (First Author): cyn_688@163.com

## Related Projects

- [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) - Our baseline system
- [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) - LiDAR-Inertial Odometry
- [R³LIVE](https://github.com/hku-mars/r3live) - RGB-Colored LiDAR-Inertial-Visual fusion

---

**If you find this work useful, please consider starring ⭐ this repository and citing our paper!**
