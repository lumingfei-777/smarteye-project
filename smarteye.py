#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#七百八十多行记得输入相机内参 调整参数

import cv2
import numpy as np
import torch
import time
import json
import socket
import threading
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from ultralytics import YOLO

try:
    from mmcv.utils import Config, DictAction
except ImportError:
    from mmengine import Config, DictAction

from mono.model.monodepth_model import get_configured_monodepth_model

# ============== 日志配置 ==============
logging.basicConfig(
    level=logging.WARNING,  # 生产环境建议使用WARNING级别，开发调试时可以改为DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drone_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============== 帧率计算 ==============
class FPSCounter:
    """帧率计算器"""
    #采用最近的三十帧的时间来计算平均FPS
    def __init__(self, max_samples: int = 30):
        self.frame_times = deque(maxlen=max_samples)
    
    def tick(self):
        """记录一帧的时间"""
        self.frame_times.append(time.time())
    
    def get_fps(self) -> float:
        """计算当前FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0.0

# ============== 全局配置 ==============
METRIC3D_DIR = Path('./Metric3D')

MODEL_TYPE = {
    'ViT-Large': {
        'cfg_file': f'{METRIC3D_DIR}/mono/configs/HourglassDecoder/vit.raft5.large.py',
        'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
    },
}

GT_DEPTH_SCALE = 256.0
INPUT_SIZE = (616, 1064)
DEPTH_PADDING = (123.675, 116.28, 103.53)
DEPTH_MEAN = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
DEPTH_STD = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

MAX_TRACK_HISTORY = 30
VELOCITY_THRESHOLD = 40.0
DISTANCE_NEAR = 30.0
DISTANCE_MID = 90.0

# 卡尔曼滤波配置
MAX_OCCLUSION_FRAMES = 10  # 最大遮挡帧数
PREDICTION_HORIZON = 5      # 预测未来帧数

# ============== 卡尔曼滤波器 ==============
class KalmanFilter3D:
    """
    3D空间卡尔曼滤波器
    状态向量: [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]
    - 位置 (X, Y, Z)
    - 速度 (Vx, Vy, Vz)
    - 加速度 (Ax, Ay, Az)
    """
    
    def __init__(self, dt: float = 0.033):
        """
        初始化卡尔曼滤波器
        
        Args:
            dt: 时间步长(秒), 默认30fps → 0.033s
        """
        self.dt = dt
        self.n_states = 9  # [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]
        self.n_measurements = 3  # [X, Y, Z]
        
        # 状态向量
        self.x = np.zeros((self.n_states, 1))
        
        # 状态协方差矩阵 (初始不确定性)
        self.P = np.eye(self.n_states) * 1000
        
        # 状态转移矩阵 (运动学模型: 匀加速运动)
        self.F = np.eye(self.n_states)
        # 位置 = 位置 + 速度*dt + 0.5*加速度*dt^2
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.F[0, 6] = 0.5 * dt**2
        self.F[1, 7] = 0.5 * dt**2
        self.F[2, 8] = 0.5 * dt**2
        # 速度 = 速度 + 加速度*dt
        self.F[3, 6] = dt
        self.F[4, 7] = dt
        self.F[5, 8] = dt
        
        # 测量矩阵 (只能观测位置)
        self.H = np.zeros((self.n_measurements, self.n_states))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # 过程噪声协方差 (模型不确定性)
        self.Q = np.eye(self.n_states)
        # 位置过程噪声较小
        self.Q[0:3, 0:3] *= 0.01
        # 速度过程噪声中等
        self.Q[3:6, 3:6] *= 0.1
        # 加速度过程噪声较大 (机动性强)
        self.Q[6:9, 6:9] *= 1.0
        
        # 测量噪声协方差 (传感器不确定性)
        self.R = np.eye(self.n_measurements) * 0.5
        
        # 滤波器状态
        self.is_initialized = False
        self.age = 0  # 滤波器年龄(帧数)
        self.time_since_update = 0  # 自上次更新以来的帧数
        
        self.logger = logging.getLogger(f"{__name__}.KalmanFilter3D")
    
    def initialize(self, measurement: np.ndarray):
        """
        使用首次测量初始化滤波器
        
        Args:
            measurement: 初始位置 [X, Y, Z]
        """
        self.x[0:3] = measurement.reshape(-1, 1)
        self.x[3:9] = 0  # 初始速度和加速度为0
        self.is_initialized = True
        self.age = 1
        self.time_since_update = 0
        self.logger.debug(f"滤波器已初始化: {measurement}")
    
    def predict(self) -> np.ndarray:
        """
        预测步骤 (时间更新)
        
        Returns:
            预测的位置 [X, Y, Z]
        """
        # 状态预测: x = F * x
        self.x = self.F @ self.x
        
        # 协方差预测: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        
        # 返回预测位置
        return self.x[0:3].flatten()
    
    def update(self, measurement: np.ndarray):
        """
        更新步骤 (测量更新)
        
        Args:
            measurement: 观测位置 [X, Y, Z]
        """
        if not self.is_initialized:
            self.initialize(measurement)
            return
        
        z = measurement.reshape(-1, 1)
        
        # 创新 (残差): y = z - H * x
        y = z - self.H @ self.x
        
        # 创新协方差: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新: x = x + K * y
        self.x = self.x + K @ y
        
        # 协方差更新: P = (I - K * H) * P
        I = np.eye(self.n_states)
        self.P = (I - K @ self.H) @ self.P
        
        self.time_since_update = 0
        
        self.logger.debug(f"滤波器已更新: 测量={measurement}, 状态={self.x[0:3].flatten()}")
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        获取当前状态
        
        Returns:
            状态字典: {position, velocity, acceleration}
        """
        return {
            'position': self.x[0:3].flatten(),
            'velocity': self.x[3:6].flatten(),
            'acceleration': self.x[6:9].flatten()
        }
    
    def predict_future(self, n_steps: int) -> np.ndarray:
        """
        预测未来n步的轨迹
        
        Args:
            n_steps: 预测步数
        
        Returns:
            未来轨迹 shape=(n_steps, 3)
        """
        future_positions = []
        x_future = self.x.copy()
        F_future = self.F.copy()
        
        for _ in range(n_steps):
            x_future = F_future @ x_future
            future_positions.append(x_future[0:3].flatten())
        
        return np.array(future_positions)
    
    def get_velocity_magnitude(self) -> float:
        """获取速度标量"""
        velocity = self.x[3:6].flatten()
        return np.linalg.norm(velocity)
    
    def get_covariance_trace(self) -> float:
        """获取协方差矩阵迹 (不确定性指标)"""
        return np.trace(self.P)

# ============== Metric3D深度估计器 ==============
class Metric3DDepthEstimator:
    """Metric3D深度估计器"""
    
    def __init__(self, model_type: str = 'ViT-Large', device: str = 'cuda'):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.input_size = INPUT_SIZE
        self.logger = logging.getLogger(f"{__name__}.Metric3D")
        self._load_model()
    
    def _load_model(self):
        """加载Metric3D模型"""
        try:
            self.logger.info(f"正在加载Metric3D模型: {self.model_type}")
            cfg_file = MODEL_TYPE[self.model_type]['cfg_file']
            ckpt_file = MODEL_TYPE[self.model_type]['ckpt_file']
            
            cfg = Config.fromfile(cfg_file)
            self.model = get_configured_monodepth_model(cfg)
            state_dict = torch.hub.load_state_dict_from_url(ckpt_file)
            self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
            self.model.to(self.device).eval()
            
            self.logger.info("Metric3D模型加载成功")
        except Exception as e:
            self.logger.error(f"加载Metric3D模型失败: {e}", exc_info=True)
            raise
    
    def preprocess_image(self, img: np.ndarray, intrinsic: List[float]) -> Tuple[torch.Tensor, List[float], List[int]]:
        """图像预处理"""
        rgb = img[:, :, ::-1]
        h, w = rgb.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        new_intrinsic = [intrinsic[i] * scale for i in range(4)]
        
        pad_h = self.input_size[0] - new_h
        pad_w = self.input_size[1] - new_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        
        rgb = cv2.copyMakeBorder(
            rgb, pad_h_half, pad_h - pad_h_half,
            pad_w_half, pad_w - pad_w_half,
            cv2.BORDER_CONSTANT, value=DEPTH_PADDING
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        
        rgb_tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb_tensor = torch.div((rgb_tensor - DEPTH_MEAN), DEPTH_STD)
        rgb_tensor = rgb_tensor[None, :, :, :].to(self.device)
        
        return rgb_tensor, new_intrinsic, pad_info
    
    def estimate_depth(self, img: np.ndarray, intrinsic: List[float]) -> np.ndarray:
        """估计深度图"""
        try:
            h_orig, w_orig = img.shape[:2]
            rgb_tensor, new_intrinsic, pad_info = self.preprocess_image(img, intrinsic)
            
            with torch.no_grad():
                depth, confidence, output_dict = self.model.inference({'input': rgb_tensor})
            
            depth = depth.squeeze()
            depth = depth[
                pad_info[0]: depth.shape[0] - pad_info[1],
                pad_info[2]: depth.shape[1] - pad_info[3]
            ]
            
            depth = torch.nn.functional.interpolate(
                depth[None, None, :, :], (h_orig, w_orig), mode='bilinear'
            ).squeeze()
            
            canonical_to_real_scale = new_intrinsic[0] / 1000.0
            depth = depth * canonical_to_real_scale
            depth = torch.clamp(depth, 0, 300)
            
            return depth.cpu().numpy()
        except Exception as e:
            self.logger.error(f"深度估计失败: {e}", exc_info=True)
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

# ============== 无人机追踪器 ==============
class DroneTrack:
    """
    单个无人机追踪对象
    包含卡尔曼滤波器、历史轨迹、状态管理
    """
    
    def __init__(self, track_id: int, initial_position: np.ndarray, timestamp: float, dt: float = 0.033):
        self.track_id = track_id
        self.kf = KalmanFilter3D(dt=dt)
        self.kf.initialize(initial_position)
        
        # 历史记录
        self.raw_history = deque(maxlen=MAX_TRACK_HISTORY)  # 原始测量
        self.filtered_history = deque(maxlen=MAX_TRACK_HISTORY)  # 滤波后
        self.predicted_history = deque(maxlen=MAX_TRACK_HISTORY)  # 预测值
        
        # 状态
        self.last_detection_time = timestamp#最后一次记录的时间戳
        self.last_update_time = timestamp
        self.hits = 1  # 匹配次数
        self.misses = 0  # 未匹配次数
        self.is_occluded = False
        
        # 2D信息
        self.last_bbox = None
        self.last_center_2d = None
        self.last_confidence = 0.0
        
        self.logger = logging.getLogger(f"{__name__}.DroneTrack")
    
    def predict(self) -> np.ndarray:
        """预测下一帧位置"""
        predicted_pos = self.kf.predict()
        self.predicted_history.append(predicted_pos.copy())
        return predicted_pos
    
    def update(self, measurement: np.ndarray, bbox: List[int], 
               center_2d: List[int], confidence: float, timestamp: float):
        """更新滤波器"""
        self.kf.update(measurement)
        
        # 记录历史
        self.raw_history.append(measurement.copy())
        filtered_state = self.kf.get_state()
        self.filtered_history.append(filtered_state['position'].copy())
        
        # 更新状态
        self.last_detection_time = timestamp
        self.last_update_time = timestamp
        self.last_bbox = bbox
        self.last_center_2d = center_2d
        self.last_confidence = confidence
        self.hits += 1
        self.misses = 0
        self.is_occluded = False
    
    def mark_missed(self):
        """标记为未检测到"""
        self.misses += 1
        if self.misses > MAX_OCCLUSION_FRAMES:
            self.is_occluded = True
    
    def get_state(self) -> Dict:
        """获取当前状态"""
        state = self.kf.get_state()
        return {
            'track_id': self.track_id,
            'position': state['position'],
            'velocity': state['velocity'],
            'acceleration': state['acceleration'],
            'velocity_magnitude': self.kf.get_velocity_magnitude(),
            'is_occluded': self.is_occluded,
            'hits': self.hits,
            'misses': self.misses,
            'confidence': self.last_confidence,
            'uncertainty': self.kf.get_covariance_trace()
        }
    
    def predict_trajectory(self, n_steps: int = PREDICTION_HORIZON) -> np.ndarray:
        """预测未来轨迹"""
        return self.kf.predict_future(n_steps)
    
    def should_delete(self) -> bool:
        """判断是否应该删除该轨迹"""
        return self.misses > MAX_OCCLUSION_FRAMES

# ============== 无人机检测与追踪器 ==============
class DroneDetectorTracker:
    """
    无人机检测与追踪系统
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        intrinsic: List[float],
        tracker_type: str = 'bytetrack',
        enable_depth: bool = True,
        depth_model_type: str = 'ViT-Large',
        fps: float = 30.0
    ):
        self.logger = logging.getLogger(f"{__name__}.DroneDetector")
        
        # YOLO模型
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.logger.info(f"YOLO模型加载成功: {yolo_model_path}")
        except Exception as e:
            self.logger.error(f"YOLO模型加载失败: {e}", exc_info=True)
            raise
        
        self.tracker_type = tracker_type
        self.intrinsic = intrinsic
        self.dt = 1.0 / fps  # 时间步长
        
        # 深度估计器
        self.enable_depth = enable_depth
        if self.enable_depth:
            try:
                self.depth_estimator = Metric3DDepthEstimator(
                    model_type=depth_model_type,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            except Exception as e:
                self.logger.warning(f"深度估计器初始化失败: {e}")
                self.enable_depth = False

        self._cached_depth_map = None
        
        # 追踪器字典 {track_id: DroneTrack}
        self.tracks: Dict[int, DroneTrack] = {}
        self.next_track_id = 1
        
        # 统计
        self.frame_count = 0
        self.total_detections = 0
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """获取指定点深度"""
        h, w = depth_map.shape
        if 0 <= y < h and 0 <= x < w:
            return float(depth_map[y, x])
        return 0.0
    
    def pixel_to_3d(self, u: int, v: int, z: float) -> np.ndarray:
        """像素坐标转3D坐标"""
        fx, fy, cx, cy = self.intrinsic
        if z > 0:
            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            return np.array([X, Y, z])
        return np.array([0.0, 0.0, 0.0])
    
    def draw_track_info(self, frame: np.ndarray, track: DroneTrack):
        """绘制追踪信息"""
        state = track.get_state()
        position = state['position']
        velocity_mag = state['velocity_magnitude']
        distance = np.linalg.norm(position)
        
        # 根据距离设置颜色
        if distance < DISTANCE_NEAR:
            color = (0, 0, 255)
        elif distance < DISTANCE_MID:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        
        # 绘制边界框
        if track.last_bbox:
            x1, y1, x2, y2 = track.last_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            status = "遮挡" if track.is_occluded else "跟踪"
            label = f"ID:{track.track_id} {status} D:{distance:.1f}m V:{velocity_mag:.1f}m/s"
            
            # 绘制标签背景
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制滤波后的轨迹 (蓝色)
        if len(track.filtered_history) > 1:
            # 3D轨迹投影到2D
            filtered_2d = []
            for pos_3d in track.filtered_history:
                if pos_3d[2] > 0:
                    fx, fy, cx, cy = self.intrinsic
                    u = int(pos_3d[0] * fx / pos_3d[2] + cx)
                    v = int(pos_3d[1] * fy / pos_3d[2] + cy)
                    filtered_2d.append((u, v))
            
            if len(filtered_2d) > 1:
                pts = np.array(filtered_2d, dtype=np.int32)
                cv2.polylines(frame, [pts], False, (255, 0, 0), 2)
        
        # 绘制预测轨迹 (黄色虚线)
        future_traj = track.predict_trajectory(PREDICTION_HORIZON)
        if len(future_traj) > 0:
            predicted_2d = []
            for pos_3d in future_traj:
                if pos_3d[2] > 0:
                    fx, fy, cx, cy = self.intrinsic
                    u = int(pos_3d[0] * fx / pos_3d[2] + cx)
                    v = int(pos_3d[1] * fy / pos_3d[2] + cy)
                    predicted_2d.append((u, v))
            
            if len(predicted_2d) > 1:
                for i in range(len(predicted_2d) - 1):
                    cv2.line(frame, predicted_2d[i], predicted_2d[i+1], 
                            (0, 255, 255), 2, cv2.LINE_AA)
        
        # 绘制中心点
        if track.last_center_2d:
            center_x, center_y = track.last_center_2d
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
    
    def detect_and_track(
        self,
        frame: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        检测并追踪无人机 (卡尔曼滤波增强)
        """
        self.frame_count += 1
        current_time = time.time()
        annotated_frame = frame.copy()
        drone_data_list = []
        
        try:
            
            # 先用 YOLO 检测（不开深度）
            results = self.yolo_model.track(
                frame, persist=True, tracker=self.tracker_type,
                verbose=False, conf=0.5, classes=[0], imgsz=640
            )

            has_detections = (
                results[0].boxes is not None and 
                results[0].boxes.id is not None and
                len(results[0].boxes.id) > 0
            )

            if self.enable_depth and has_detections and depth_map is None:
                if self.frame_count % 2 == 1:  # 奇数帧重新估计
                    self._cached_depth_map = self.depth_estimator.estimate_depth(
                    frame, self.intrinsic
                    )
                depth_map = self._cached_depth_map  # 偶数帧复用缓存
        
            
            # 预测所有现有轨迹
            for track in self.tracks.values():
                track.predict()
            
            # 处理检测结果
            detected_ids = set()
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, conf, track_id in zip(boxes, confs, track_ids):
                    detected_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 获取深度
                    if self.enable_depth and depth_map is not None:
                        depth_value = self.get_depth_at_point(depth_map, center_x, y2)
                    else:
                        depth_value = 0.0
                    
                    # 计算3D位置 (测量值)
                    measurement_3d = self.pixel_to_3d(center_x, y2, depth_value)
                    
                    # 更新或创建轨迹
                    if track_id not in self.tracks:
                        # 新轨迹
                        self.tracks[track_id] = DroneTrack(
                            track_id, measurement_3d, current_time, self.dt
                        )
                        self.logger.info(f"新建轨迹 ID={track_id}")
                        self.tracks[track_id].update(
                            measurement_3d, [x1, y1, x2, y2],
                            [center_x, center_y], conf, current_time
                        )
                    else:
                        # 更新现有轨迹
                        self.tracks[track_id].update(
                            measurement_3d,
                            [x1, y1, x2, y2],
                            [center_x, center_y],
                            conf,
                            current_time
                        )
                    
                    self.total_detections += 1
            
            # 处理未检测到的轨迹 (遮挡处理)
            for track_id, track in list(self.tracks.items()):
                if track_id not in detected_ids:
                    track.mark_missed()
                    
                    # 删除长时间未检测到的轨迹
                    if track.should_delete():
                        self.logger.info(f"删除轨迹 ID={track_id} (长时间未检测)")
                        del self.tracks[track_id]
                        continue
            
            # 收集所有轨迹数据
            for track in self.tracks.values():
                state = track.get_state()
                position = state['position']
                velocity = state['velocity']
                
                drone_info = {
                    'timestamp': current_time,
                    'track_id': int(track.track_id),
                    'position_3d': position.tolist(),
                    'velocity_3d': velocity.tolist(),
                    'velocity': float(state['velocity_magnitude']),
                    'distance': float(np.linalg.norm(position)),
                    'confidence': float(state['confidence']),
                    'is_occluded': bool(state['is_occluded']),
                    'hits': int(state['hits']),
                    'misses': int(state['misses']),
                    'uncertainty': float(state['uncertainty'])
                }
                
                # 添加2D信息
                if track.last_bbox:
                    drone_info['bbox_2d'] = track.last_bbox
                if track.last_center_2d:
                    drone_info['center_2d'] = track.last_center_2d
                
                # 预测轨迹
                future_traj = track.predict_trajectory(PREDICTION_HORIZON)
                drone_info['predicted_trajectory'] = future_traj.tolist()
                
                drone_data_list.append(drone_info)
                
                # 绘制
                self.draw_track_info(annotated_frame, track)
            
            # 绘制统计信息
            active_tracks = len([t for t in self.tracks.values() if not t.is_occluded])
            occluded_tracks = len([t for t in self.tracks.values() if t.is_occluded])
            
            stats_text = (
                f"Frame: {self.frame_count} | "
                f"Active: {active_tracks} | "
                f"Occluded: {occluded_tracks} | "
                f"Total: {self.total_detections}"
            )
            cv2.putText(annotated_frame, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        except Exception as e:
            self.logger.error(f"检测追踪失败: {e}", exc_info=True)
        
        return annotated_frame, drone_data_list

# ============== Socket网络客户端 ==============
class SocketClient:
    """Socket网络客户端"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 65432):
        self.host = host
        self._cached_depth_map = None
        self.port = port
        self.socket = None
        self.connected = False
        self.logger = logging.getLogger(f"{__name__}.SocketClient")
        self.send_queue = deque(maxlen=100)
        self.send_thread = None#后台发送线程
        self.running = False#运行状态
        self.last_frame_send_time = 0
        self.frame_send_interval = 1.0 / 10  # 最多传10fps


    def should_send_frame(self) -> bool:
        now = time.time()
        if now - self.last_frame_send_time >= self.frame_send_interval:
            self.last_frame_send_time = now
            return True
        return False

    def send_frame(self, frame: np.ndarray, frame_id: int, quality: int = 50):
        """发送压缩视频帧"""
        if not self.connected:
            return
    
        # JPEG压缩
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            return
    
        import base64
        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
        packet = {
            'message_type': 'frame',
            'device_id': 'drone_detector',
            'frame_id': frame_id,
            'timestamp': time.time(),
            'width': frame.shape[1],
            'height': frame.shape[0],
            'encoding': 'jpeg_base64',
            'data': frame_b64
        }
        self.send_queue.append(packet)
    
    def connect(self) -> bool:#返回布尔值表示连接是否成功
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#参数表示使用IPv4和TCP协议
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info(f"已连接到服务器: {self.host}:{self.port}")
            
            # 发送设备注册
            device_info = {
                'message_type': 'device_info',
                'device_name': '无人机检测器',
                'device_type': 'drone_detector',
                'ip_address': socket.gethostbyname(socket.gethostname()),
            }
            self._send_json(device_info)#发送设备信息到服务器
            
            # 启动发送线程
            self.running = True
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()
            
            return True
        except Exception as e:
            self.logger.error(f"连接失败: {e}", exc_info=True)
            self.connected = False
            return False
    
    def _send_json(self, data: Dict) -> bool:
        """发送JSON数据"""
        try:
            if self.socket and self.connected:
                json_str = json.dumps(data, ensure_ascii=False)#将数据转换为JSON字符串，允许非ASCII字符
                self.socket.sendall((json_str + '\n').encode('utf-8'))#发送数据并添加换行符作为分隔
                return True
        except Exception as e:
            self.logger.error(f"发送失败: {e}")
            self.connected = False
        return False
    
    def _send_loop(self):
        """发送循环"""
        while self.running:
            try:
                if self.send_queue and self.connected:
                    data = self.send_queue.popleft()#从队列中获取最早的信息
                    self._send_json(data)
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"发送循环错误: {e}")
    
    def send_drone_data(self, drone_data_list: List[Dict]):
        """发送无人机数据"""
        if not self.connected:
            return
        
        packet = {
            'message_type': 'data',
            'device_id': 'drone_detector',
            'timestamp': time.time(),
            'readings': {
                'drone_count': len(drone_data_list),
                'active_count': len([d for d in drone_data_list if not d.get('is_occluded', False)]),
                #未被遮挡的无人机数量
                'occluded_count': len([d for d in drone_data_list if d.get('is_occluded', False)]),
                #遮挡的无人机数量
                'drones': drone_data_list
            }
        }
        self.send_queue.append(packet)
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        self.connected = False
        if self.send_thread:
            self.send_thread.join(timeout=2)#主线程等待发送线程结束，最多等待2秒
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.logger.info("已断开连接")


# ============== 主程序 ==============
def main():
    """主程序"""
    # 配置
    CAMERA_INTRINSIC = [800.0, 800.0, 640.0, 360.0]
    YOLO_MODEL_PATH = './models/yolov8n.pt'
    SERVER_HOST = '127.0.0.1'
    SERVER_PORT = 65432
    VIDEO_SOURCE = 0
    ENABLE_DEPTH = torch.cuda.is_available()
    SHOW_DISPLAY = True
    ENABLE_NETWORK = True
    FPS = 30.0 
    
    logger.info("="*60)
    logger.info("无人机检测系统启动")
    logger.info("="*60)
    
    from finetune_daemon import FinetuneDaemon, FinetuneDaemonConfig

    daemon_cfg = FinetuneDaemonConfig(
        base_model_path=YOLO_MODEL_PATH,
        dataset_dir='./finetune_dataset',
    )
    finetune_daemon = FinetuneDaemon(daemon_cfg, detector.yolo_model)
    finetune_daemon.start()

    
    # 初始化检测器
    try:
        detector = DroneDetectorTracker(
            yolo_model_path=YOLO_MODEL_PATH,
            intrinsic=CAMERA_INTRINSIC,
            tracker_type='bytetrack',#使用多目标追踪算法Bytetrack
            enable_depth=ENABLE_DEPTH,
            depth_model_type='ViT-Large',
            fps=FPS
        )
        logger.info("✓ 检测器初始化成功")
    except Exception as e:
        logger.error(f"✗ 检测器初始化失败: {e}")
        return
    
    # 连接服务器
    socket_client = None
    if ENABLE_NETWORK:
        socket_client = SocketClient(host=SERVER_HOST, port=SERVER_PORT)
        if socket_client.connect():
            logger.info("✓ 已连接到监控服务器")
        else:
            logger.warning("⚠ 无法连接服务器,仅本地运行")
            socket_client = None
    
    # 打开视频
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"无法打开相机: {VIDEO_SOURCE}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"✓ 视频源: {VIDEO_SOURCE} | {width}x{height} @ {FPS}fps")
    
    frame_count = 0
    start_time = time.time()
    fps_counter = deque(maxlen=30)
    
    try:
        logger.info("开始检测...")
        
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                logger.info("视频流结束")
                break
            
            frame_count += 1
            
            # 检测与追踪
            annotated_frame, drone_data = detector.detect_and_track(frame)
            
            finetune_daemon.on_detection_result(frame, drone_data, frame_count)
            
            if socket_client and socket_client.should_send_frame():
                socket_client.send_frame(annotated_frame, frame_count, quality=50)

            # 发送数据
            if socket_client and drone_data:
                socket_client.send_drone_data(drone_data)
            
            # 打印结果
            if drone_data:
                active_drones = [d for d in drone_data if not d.get('is_occluded', False)]
                occluded_drones = [d for d in drone_data if d.get('is_occluded', False)]
                
                logger.info(f"Frame {frame_count} - 活跃:{len(active_drones)} 遮挡:{len(occluded_drones)}")
                
                for drone in active_drones:
                    track_id = drone['track_id']
                    pos = drone['position_3d']
                    vel = drone['velocity']
                    dist = drone['distance']
                    logger.info(
                        f"  [活跃] ID{track_id}: "
                        f"位置({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})m | "
                        f"速度{vel:.1f}m/s | 距离{dist:.1f}m"
                    )
                
                for drone in occluded_drones:
                    track_id = drone['track_id']
                    pos = drone['position_3d']
                    logger.info(f"  [遮挡] ID{track_id}: 预测位置({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})m")
            
            # FPS
            loop_time = time.time() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0
            fps_counter.append(current_fps)
            avg_fps = np.mean(fps_counter)
            
            if SHOW_DISPLAY:
                fps_text = f"FPS: {avg_fps:.1f} | Kalman Filter: ON"
                cv2.putText(annotated_frame, fps_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow('Drone Detection - Kalman Enhanced', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户退出")
                    break
            
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"统计: {frame_count}帧 | {avg_fps:.1f}fps | "
                    f"活跃轨迹:{len(detector.tracks)} | 总检测:{detector.total_detections}"
                )
    
    except KeyboardInterrupt:
        logger.info("检测被中断")
    except Exception as e:
        logger.error(f"运行错误: {e}", exc_info=True)
    finally:
        logger.info("清理资源...")
        cap.release()
        cv2.destroyAllWindows()
        if socket_client:
            socket_client.disconnect()
        finetune_daemon.stop()
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info(f"检测完成 | 总帧数:{frame_count} | 总检测:{detector.total_detections}")
        logger.info(f"运行时间:{total_time:.1f}秒 | 平均FPS:{frame_count/total_time:.1f}")
        logger.info("="*60)

if __name__ == '__main__':
    main()
