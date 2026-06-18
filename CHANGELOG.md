# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-18

### Added
- SLAM module: EKF-SLAM, FastSLAM, GraphSLAM, Cartographer-ESKF-UKF fusion
- Perception module: Voxel filter, SDF adaptive filter, Euclidean cluster, Kalman tracker, Costmap
- PathPlanning module: A*, Adaptive A*, Dijkstra, DWA, RRT, TEB
- Control module: LQR, Stanley, Pure Pursuit, SMC, speed control
- Evaluation module: metrics, monte carlo, sensitivity analysis
- Sensor synchronization module

### Changed
- SLAM degradation manager simplified to 2-state FSM
- UKF adaptive alpha and innovation-driven Q
- Author attribution unified to Kat-yuan-eng (RuiWen Liao)
- Code comments optimized to PythonRobotics style with physical unit annotations

## [0.9.0] - 2023-12-01

### Added
- Initial release with 4 core modules
