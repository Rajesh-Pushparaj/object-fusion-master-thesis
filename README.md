# Deep Object-Level Fusion for Autonomous Perception

This repository contains my Master's thesis project titled:

**"Deep High-Level Object Fusion and Multi-Object Tracking"**  
Submitted to: Technische Universität Dortmund  
Date: December 27, 2023  
Author: Rajesh Pushparaj

The project focuses on transformer-based architectures for fusing object-level detections from multi-modal sensors (camera and radar) to enhance perception in autonomous driving systems.

## 🧠 Abstract
> This work proposes a transformer-based architecture for high-level multi-sensor object fusion in the context of autonomous driving. Compared to classical methods such as Kalman filters, the approach uses attention-based mechanisms to dynamically combine detections from diverse sensors under varying environmental conditions. The model significantly improves mean average precision (mAP), outperforming baseline fusion methods across multiple IoU thresholds.

## 🚀 Key Highlights
- Transformer architecture adapted from DETR and BEVFormer
- Custom dataset with 175 annotated highway scenes
- Achieved ~68.5 mAP @ 0.5 IoU vs. 22.1 (baseline)
