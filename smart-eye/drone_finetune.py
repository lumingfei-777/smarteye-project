#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机检测自适应微调守护程序
功能：
  1. 采集高置信度无人机帧作为伪标签
  2. 多特征联合场景去重（亮度/对比度/纹理/能见度），支持雾天雨天识别
  3. Alpha混合+模糊合成新训练样本
  4. 空闲时（连续N秒无目标）触发自蒸馏微调
  5. 微调完成后热加载新权重覆盖原模型
"""

import cv2
import numpy as np
import torch
import time
import threading
import logging
import shutil
import yaml
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ============== 配置 ==============
@dataclass
class FinetuneDaemonConfig:
    # 路径
    base_model_path: str = './models/yolov8s.pt'
    dataset_dir: str = './finetune_dataset'
    weights_backup_dir: str = './weights_backup'

    # 采集参数
    conf_threshold: float = 0.85          # 伪标签置信度门槛
    max_dataset_size: int = 500           # 数据集最大样本数
    max_crops_per_scene: int = 10         # 每个场景最多保存的crop数量
    min_drone_size: int = 20              # 无人机bbox最小边长（像素），过小不采集

    # 场景去重参数（多特征联合）
    brightness_tol: float = 15.0          # 亮度容差（0~255）
    contrast_tol: float = 10.0            # 对比度容差（标准差）
    texture_tol_ratio: float = 0.30       # 纹理差异比例容差
    fog_tol: float = 8.0                  # 能见度特征容差

    # 合成参数
    alpha_min: float = 0.6               # Alpha混合最小透明度
    alpha_max: float = 0.95              # Alpha混合最大透明度
    blur_kernel_range: Tuple[int,int] = (3, 7)  # 模糊核大小范围（奇数）
    synthetic_ratio: float = 0.4         # 合成样本占训练集比例

    # 空闲触发参数
    idle_trigger_seconds: float = 5.0    # 连续无目标多少秒触发微调
    min_samples_to_train: int = 30       # 最少样本数才触发微调
    finetune_cooldown: float = 300.0     # 两次微调最短间隔（秒）

    # 微调参数
    finetune_epochs: int = 3
    finetune_batch_size: int = 4
    finetune_lr: float = 0.0001
    freeze_layers: int = 5              # 冻结前N层（只训练检测头）
    imgsz: int = 640
    device: str = 'cuda'


# ============== 场景特征库 ==============
class SceneFeatureDB:
    """
    多特征联合场景去重
    同时考虑亮度、对比度、纹理复杂度、能见度（雾/雨天识别）
    只有四个维度都相似才认为是重复场景，任意一个维度差异大就视为新场景
    这样大雾/雨天（低对比度、低能见度）能被正确识别为新场景
    """

    def __init__(self, cfg: FinetuneDaemonConfig):
        self.cfg = cfg
        self.feature_db: List[Dict[str, float]] = []
        self._lock = threading.Lock()

    def _extract_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        提取场景特征向量
        - brightness : 平均亮度，晴天高，雨天/夜间低
        - contrast   : 灰度标准差，大雾时极低
        - texture    : Laplacian方差，纹理丰富度，雾天细节消失所以很低
        - fog_score  : 能见度综合分，大雾=低对比度+少暗区
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(gray.mean())
        contrast   = float(gray.std())

        # 纹理复杂度：Laplacian方差越大纹理越丰富
        laplacian  = cv2.Laplacian(gray, cv2.CV_64F)#输出数据为64位浮点数
        texture    = float(laplacian.var())#var计算方差，数值越大纹理越丰富

        # 能见度估计：
        # 大雾 → 对比度极低 + 暗区域极少（整体发白）
        # 雨天 → 对比度中等 + 纹理模糊
        dark_ratio = float(np.sum(gray < 50) / gray.size)
        fog_score  = contrast * (dark_ratio + 0.01)  # 越小越像大雾

        return {
            'brightness': brightness,
            'contrast':   contrast,
            'texture':    texture,
            'fog_score':  fog_score,
        }

    def _is_duplicate(self, f_new: Dict[str, float], f_old: Dict[str, float]) -> bool:
        """
        判断两个特征是否属于重复场景
        必须四个维度都相似才算重复，任意一个差异大就是新场景
        """
        # 亮度差异
        if abs(f_new['brightness'] - f_old['brightness']) > self.cfg.brightness_tol:
            return False

        # 对比度差异
        if abs(f_new['contrast'] - f_old['contrast']) > self.cfg.contrast_tol:
            return False

        # 纹理差异（相对比例，避免绝对值量纲问题）
        base_texture = max(f_new['texture'], f_old['texture'], 1.0)
        if abs(f_new['texture'] - f_old['texture']) / base_texture > self.cfg.texture_tol_ratio:
            return False

        # 能见度差异（雾天 fog_score 会非常小，与晴天差异显著）
        if abs(f_new['fog_score'] - f_old['fog_score']) > self.cfg.fog_tol:
            return False

        return True  # 四个维度都相似 → 重复场景

    def is_new_scene(self, frame: np.ndarray) -> Tuple[bool, str]:
        """
        判断是否是新场景
        Returns:
            (True/False, 场景类型描述)
        """
        features = self._extract_features(frame)

        # 判断场景类型（用于日志）
        scene_type = self._classify_scene(features)

        with self._lock:
            for existing in self.feature_db:
                if self._is_duplicate(features, existing):
                    return False, scene_type
            self.feature_db.append(features)
            return True, scene_type

    def _classify_scene(self, f: Dict[str, float]) -> str:
        """根据特征粗略分类场景类型（仅用于日志显示）"""
        if f['fog_score'] < 5.0 and f['contrast'] < 20:
            return '大雾'
        elif f['contrast'] < 30 and f['brightness'] > 150:
            return '薄雾'
        elif f['brightness'] < 60:
            return '夜间/弱光'
        elif f['texture'] < 50:
            return '低纹理'
        else:
            return '正常'

    def size(self) -> int:
        with self._lock:
            return len(self.feature_db)


# ============== 样本数据结构 ==============
@dataclass
class DroneSample:
    """一个采集到的无人机样本"""
    frame: np.ndarray           # 原始帧
    bbox: List[int]             # [x1, y1, x2, y2]
    crop: np.ndarray            # 无人机crop
    confidence: float
    timestamp: float
    frame_id: int


@dataclass
class BackgroundSample:
    """一个新场景背景帧"""
    frame: np.ndarray
    timestamp: float
    scene_hash: str = ''


# ============== 数据集管理 ==============
class DatasetManager:
    """
    管理训练数据集（YOLO格式）
    目录结构:
        dataset_dir/
            images/train/
            labels/train/
            images/val/
            labels/val/
            data.yaml
    """

    def __init__(self, dataset_dir: str, max_size: int = 500):
        self.dataset_dir = Path(dataset_dir)
        self.max_size = max_size
        self._lock = threading.Lock()
        self._setup_dirs()

    def _setup_dirs(self):
        for split in ['train', 'val']:
            (self.dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        self._write_yaml()

    def _write_yaml(self):
        yaml_content = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 1,
            'names': ['drone']
        }
        with open(self.dataset_dir / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def _bbox_to_yolo(self, bbox: List[int], img_w: int, img_h: int) -> str:
        """将像素bbox转换为YOLO归一化格式"""
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def save_sample(self, image: np.ndarray, bboxes: List[List[int]],
                    split: str = 'train', prefix: str = '') -> bool:
        """
        保存一张图片和对应标签
        bboxes: [[x1,y1,x2,y2], ...]
        """
        with self._lock:
            # 超出上限时删除最旧的
            img_dir = self.dataset_dir / 'images' / split
            lbl_dir = self.dataset_dir / 'labels' / split
            existing = sorted(img_dir.glob('*.jpg'))
            while len(existing) >= self.max_size:
                oldest = existing.pop(0)
                oldest.unlink()
                lbl = lbl_dir / (oldest.stem + '.txt')
                if lbl.exists(): lbl.unlink()
                existing = sorted(img_dir.glob('*.jpg'))

            # 生成文件名
            ts = int(time.time() * 1000)#毫秒级时间戳
            stem = f"{prefix}_{ts}" if prefix else str(ts)#不同命名方式
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            h, w = image.shape[:2]
            cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

            label_lines = [self._bbox_to_yolo(bb, w, h) for bb in bboxes]
            lbl_path.write_text('\n'.join(label_lines))
            return True

    def count(self, split: str = 'train') -> int:
        with self._lock:
            return len(list((self.dataset_dir / 'images' / split).glob('*.jpg')))

    def yaml_path(self) -> str:
        return str(self.dataset_dir / 'data.yaml')


# ============== Alpha混合合成器 ==============
class AlphaBlendSynthesizer:
    """
    将无人机crop通过Alpha混合+模糊植入背景帧
    生成合成训练样本
    """

    def __init__(self, cfg: FinetuneDaemonConfig):
        self.cfg = cfg

    def _make_alpha_mask(self, crop: np.ndarray, alpha: float) -> np.ndarray:
        """生成椭圆形Alpha遮罩，边缘渐变，比矩形更自然"""
        h, w = crop.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        # 椭圆形遮罩
        for y in range(h):
            for x in range(w):
                dy = (y - cy) / (h / 2 + 1e-6)
                dx = (x - cx) / (w / 2 + 1e-6)
                dist = dy**2 + dx**2
                if dist <= 1.0:
                    mask[y, x] = alpha * (1.0 - dist * 0.3)
        return mask

    def synthesize(
        self,
        background: np.ndarray,
        crop: np.ndarray,
        position: Optional[Tuple[int,int]] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        将crop植入background
        position: (cx, cy) 中心点，None则随机选择
        Returns: (合成图, bbox [x1,y1,x2,y2])
        """
        bg = background.copy()
        bh, bw = bg.shape[:2]
        ch, cw = crop.shape[:2]

        # 随机缩放crop（模拟不同距离）
        scale = np.random.uniform(0.7, 1.3)
        new_cw = max(10, int(cw * scale))
        new_ch = max(10, int(ch * scale))
        crop_resized = cv2.resize(crop, (new_cw, new_ch))
        ch, cw = new_ch, new_cw

        # 确定植入位置
        if position is None:
            margin = 20
            cx = np.random.randint(cw//2 + margin, bw - cw//2 - margin)
            cy = np.random.randint(ch//2 + margin, bh - ch//2 - margin)
        else:
            cx, cy = position
            cx = np.clip(cx, cw//2, bw - cw//2)
            cy = np.clip(cy, ch//2, bh - ch//2)

        x1 = cx - cw // 2
        y1 = cy - ch // 2
        x2 = x1 + cw
        y2 = y1 + ch

        # 边界检查
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(bw, x2), min(bh, y2)
        if x2c <= x1c or y2c <= y1c:
            return bg, [x1, y1, x2, y2]

        # 裁剪到边界内
        crop_patch = crop_resized[y1c-y1:y2c-y1, x1c-x1:x2c-x1]

        # Alpha混合
        alpha = np.random.uniform(self.cfg.alpha_min, self.cfg.alpha_max)
        alpha_mask = self._make_alpha_mask(crop_patch, alpha)
        alpha_3ch = np.stack([alpha_mask]*3, axis=-1)

        roi = bg[y1c:y2c, x1c:x2c].astype(np.float32)
        blended = roi * (1 - alpha_3ch) + crop_patch.astype(np.float32) * alpha_3ch
        bg[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)

        # 轻微模糊（模拟运动模糊/焦距虚化）
        k = np.random.choice([k for k in range(
            self.cfg.blur_kernel_range[0],
            self.cfg.blur_kernel_range[1]+1, 2
        )])
        sigma = np.random.uniform(0.3, 1.0)
        bg[y1c:y2c, x1c:x2c] = cv2.GaussianBlur(
            bg[y1c:y2c, x1c:x2c], (k, k), sigma
        )

        return bg, [x1c, y1c, x2c, y2c]


# ============== 微调执行器 ==============
class FinetuneExecutor:
    """
    执行YOLO自蒸馏微调
    分层学习率策略：
      - 浅层 (0 ~ freeze_layers-1)   : 完全冻结，学习率=0
      - 中间层 (freeze_layers ~ mid_layer) : 解冻，学习率×0.1（极小）
      - 检测头 (mid_layer+1 ~ end)    : 解冻，正常学习率
    微调完成后热加载权重
    """

    def __init__(self, cfg: FinetuneDaemonConfig):
        self.cfg = cfg
        self.backup_dir = Path(cfg.weights_backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._is_finetuning = False
        self._lock = threading.Lock()

    @property
    def is_finetuning(self) -> bool:
        #获取当前微调状态
        return self._is_finetuning

    def _backup_weights(self, model_path: str):
        """备份当前权重,只保留最近5个"""
        src = Path(model_path)
        if src.exists():
            ts = int(time.time())
            dst = self.backup_dir / f"{src.stem}_backup_{ts}{src.suffix}"
            #stem文件名不带后缀，suffix是后缀
            shutil.copy2(src, dst)#复制文件并保留元数据
            backups = sorted(self.backup_dir.glob(f"{src.stem}_backup_*"))
            #sorted分类按时间戳，glob返回所有匹配的文件路径列表
            for old in backups[:-5]:
                old.unlink()#删除除了最后五个以外的备份文件
            logger.info(f"权重已备份: {dst}")

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """
        构建分层学习率优化器
        浅层冻结，中间层lr×0.1，检测头正常lr
        """
        freeze_end  = self.cfg.freeze_layers          # 0~4  完全冻结
        mid_end     = self.cfg.freeze_layers + 6      # 5~10 中间层（极小lr）
        # 11层以上为检测头，正常lr

        head_params   = []#存储检测头的参数
        middle_params = []#存储中间层的参数


        # 遍历模型参数
        for name, param in model.named_parameters():
            '''
            # name, parameter
            ('model.0.conv.weight', Parameter containing: tensor([[...]]))
            ('model.0.conv.bias', Parameter containing: tensor([...]))
            ('model.1.conv.weight', Parameter containing: tensor([[...]]))
            ('model.1.conv.bias', Parameter containing: tensor([...]))
            '''
            parts = name.split('.')
            #分割成列表
            # 取层编号，非数字的（如 'model'前缀）跳过编号判断
            layer_num = -1#默认-1，如果没有数字层编号则视为检测头
            for p in parts:
                #检测层数
                if p.isdigit():#检测字符串是否只包含数字
                    layer_num = int(p)
                    break

            if layer_num < freeze_end:
                # 浅层：完全冻结
                param.requires_grad = False
            elif layer_num < mid_end:
                # 中间层：解冻，极小学习率
                param.requires_grad = True
                middle_params.append(param)
            else:
                # 检测头：解冻，正常学习率
                param.requires_grad = True
                head_params.append(param)

        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        middle = len(middle_params)#中间层数
        head   = len(head_params)#检测头层数
        logger.info(
            f"分层学习率 | 冻结={frozen}组 | "
            f"中间层={middle}组(lr={self.cfg.finetune_lr*0.1:.2e}) | "
            f"检测头={head}组(lr={self.cfg.finetune_lr:.2e})"
        )

        #创建adamW优化器，设置不同学习率
        optimizer = torch.optim.AdamW([
            {'params': middle_params, 'lr': self.cfg.finetune_lr * 0.1,
             'weight_decay': 0.0005},
            {'params': head_params,   'lr': self.cfg.finetune_lr,
             'weight_decay': 0.0005},
        ])
        return optimizer

    def _build_dataloader(self, dataset_manager: DatasetManager):
        """构建训练DataLoader（YOLO格式图片+标签）"""
        from torch.utils.data import Dataset, DataLoader

        class YOLODataset(Dataset):
            def __init__(self, img_dir: Path, lbl_dir: Path, imgsz: int):
                self.imgsz = imgsz
                self.samples = []
                for img_path in sorted(img_dir.glob('*.jpg')):
                    lbl_path = lbl_dir / (img_path.stem + '.txt')
                    if lbl_path.exists():
                        self.samples.append((img_path, lbl_path))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, lbl_path = self.samples[idx]

                # 读取图片并resize
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (self.imgsz, self.imgsz))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 随机数据增强
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)   # 水平翻转

                # 归一化 → tensor
                img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0

                # 读取标签
                lines = lbl_path.read_text().strip().split('\n')
                labels = []
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            labels.append([float(x) for x in parts])

                labels_t = torch.tensor(labels, dtype=torch.float32) \
                    if labels else torch.zeros((0, 5), dtype=torch.float32)

                return img_t, labels_t

        img_dir = Path(dataset_manager.dataset_dir) / 'images' / 'train'
        lbl_dir = Path(dataset_manager.dataset_dir) / 'labels' / 'train'

        dataset = YOLODataset(img_dir, lbl_dir, self.cfg.imgsz)
        if len(dataset) == 0:
            return None

        def collate_fn(batch):
            imgs, labels = zip(*batch)
            imgs = torch.stack(imgs, 0)
            return imgs, list(labels)

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.finetune_batch_size,
            shuffle=True,
            num_workers=0,      # Jetson上不用多进程，避免资源竞争
            collate_fn=collate_fn,
            drop_last=True,
        )
        return loader

    def run(self, dataset_manager: DatasetManager, yolo_model: YOLO) -> bool:
        """
        执行一次分层学习率微调
        Returns True 表示成功
        """
        with self._lock:
            if self._is_finetuning:
                logger.warning("微调已在运行中，跳过")
                return False
            self._is_finetuning = True

        try:
            logger.info("="*50)
            logger.info("开始分层学习率自蒸馏微调")
            logger.info(f"训练样本数: {dataset_manager.count('train')}")
            logger.info(f"浅层冻结: 0~{self.cfg.freeze_layers-1}层")
            logger.info(f"中间层(×0.1 lr): {self.cfg.freeze_layers}~{self.cfg.freeze_layers+5}层")
            logger.info(f"检测头(正常lr): {self.cfg.freeze_layers+6}层以上")
            logger.info("="*50)

            # 备份当前权重
            self._backup_weights(self.cfg.base_model_path)

            # 创建独立的微调模型（不影响推理实例）
            finetune_model = YOLO(self.cfg.base_model_path)
            ft_model = finetune_model.model.to(self.cfg.device)
            ft_model.train()

            # 构建分层优化器
            optimizer = self._build_optimizer(ft_model)

            # 构建DataLoader
            loader = self._build_dataloader(dataset_manager)
            if loader is None:
                logger.error("数据集为空，跳过微调")
                return False

            # 学习率调度：余弦退火，防止震荡
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.finetune_epochs * len(loader),
                eta_min=self.cfg.finetune_lr * 0.01
            )

            # ── 训练循环 ──
            best_loss = float('inf')
            best_state = None

            for epoch in range(self.cfg.finetune_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for imgs, labels in loader:
                    imgs = imgs.to(self.cfg.device)

                    optimizer.zero_grad()

                    try:
                        # YOLO前向传播（训练模式返回loss）
                        loss, loss_items = ft_model(imgs, labels)

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning("Loss异常（NaN/Inf），跳过该batch")
                            continue

                        loss.backward()

                        # 梯度裁剪，防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(
                            ft_model.parameters(), max_norm=10.0
                        )

                        optimizer.step()
                        scheduler.step()

                        epoch_loss += loss.item()
                        n_batches += 1

                    except Exception as e:
                        logger.warning(f"Batch训练异常: {e}")
                        continue

                if n_batches == 0:
                    logger.warning(f"Epoch {epoch+1} 无有效batch，跳过")
                    continue

                avg_loss = epoch_loss / n_batches
                lr_head   = optimizer.param_groups[1]['lr']
                lr_middle = optimizer.param_groups[0]['lr']

                logger.info(
                    f"Epoch [{epoch+1}/{self.cfg.finetune_epochs}] "
                    f"loss={avg_loss:.4f} | "
                    f"lr_head={lr_head:.2e} | "
                    f"lr_mid={lr_middle:.2e}"
                )

                # 保存最优权重（loss最低的epoch）
                if avg_loss < best_loss:
                    best_loss  = avg_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in ft_model.state_dict().items()}

            if best_state is None:
                logger.error("训练未产生有效权重")
                return False

            # 保存最优权重到文件
            save_path = Path('./finetune_runs/adaptive_layerlr/best.pt')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': best_state}, str(save_path))
            logger.info(f"最优权重已保存 (loss={best_loss:.4f}): {save_path}")

            # 验证权重有效性
            if not self._validate_weights(save_path, best_state):
                logger.error("新权重验证失败，保留原权重")
                return False

            # 热加载到推理模型
            self._hotload_weights(best_state, yolo_model)

            # 覆盖原模型文件
            shutil.copy2(save_path, self.cfg.base_model_path)
            logger.info(f"新权重已覆盖原模型: {self.cfg.base_model_path}")

            # 释放微调模型显存
            del ft_model, finetune_model, optimizer
            torch.cuda.empty_cache()

            return True

        except Exception as e:
            logger.error(f"微调失败: {e}", exc_info=True)
            return False
        finally:
            with self._lock:
                self._is_finetuning = False

    def _validate_weights(self, weights_path: Path,
                          state_dict: Optional[Dict] = None) -> bool:
        """验证权重有效性：能加载并推理一张空白图"""
        try:
            test_model = YOLO(self.cfg.base_model_path)
            if state_dict is not None:
                test_model.model.load_state_dict(state_dict, strict=False)
            test_model.model.eval()
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                test_model.predict(dummy, verbose=False)
            del test_model
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"权重验证失败: {e}")
            return False

    def _hotload_weights(self, state_dict: Dict, yolo_model: YOLO):
        """热加载权重到正在运行的推理模型（不中断推理）"""
        try:
            # 推理模型切到CPU加载，加载完再切回来
            yolo_model.model.load_state_dict(state_dict, strict=False)
            yolo_model.model.eval()
            logger.info("权重热加载成功，推理模型已更新")
        except Exception as e:
            logger.warning(f"热加载失败（下次重启生效）: {e}")



# ============== 守护线程主体 ==============
class FinetuneDaemon:
    """
    微调守护线程
    与主检测程序并行运行，负责：
    1. 接收检测结果，采集高质量样本
    2. 场景去重
    3. 合成数据
    4. 空闲触发微调
    """

    def __init__(self, cfg: FinetuneDaemonConfig, yolo_model: YOLO):
        self.cfg = cfg
        self.yolo_model = yolo_model  # 引用主检测器的YOLO实例

        self.scene_db = SceneFeatureDB(cfg)
        self.dataset_manager = DatasetManager(cfg.dataset_dir, cfg.max_dataset_size)
        self.synthesizer = AlphaBlendSynthesizer(cfg)
        self.executor = FinetuneExecutor(cfg)

        # 无人机crop池（用于合成）
        self.crop_pool: deque = deque(maxlen=100)
        self.crop_pool_lock = threading.Lock()

        # 空闲检测
        self.last_detection_time = time.time()
        self.last_finetune_time = 0.0

        # 采集队列（主线程push，守护线程pop）
        self.sample_queue: deque = deque(maxlen=50)
        self.sample_queue_lock = threading.Lock()

        # 运行状态
        self.running = False
        self._thread: Optional[threading.Thread] = None

        self.logger = logging.getLogger(f"{__name__}.FinetuneDaemon")

    def start(self):
        """启动守护线程"""
        self.running = True
        self._thread = threading.Thread(
            target=self._daemon_loop,
            name='FinetuneDaemon',
            daemon=True
        )
        self._thread.start()
        self.logger.info("微调守护线程已启动")

    def stop(self):
        """停止守护线程"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=10)
        self.logger.info("微调守护线程已停止")

    def on_detection_result(
        self,
        frame: np.ndarray,
        drone_data: List[Dict],
        frame_id: int
    ):
        """
        主线程每帧调用此方法（非阻塞）
        将采集任务推入队列，由守护线程异步处理
        """
        has_drone = len(drone_data) > 0

        if has_drone:
            self.last_detection_time = time.time()
            # 推入队列（只传引用，不复制，守护线程里再复制）
            with self.sample_queue_lock:
                self.sample_queue.append({
                    'frame': frame,
                    'drone_data': drone_data,
                    'frame_id': frame_id,
                    'timestamp': time.time()
                })

    def _daemon_loop(self):
        """守护线程主循环"""
        self.logger.info("守护线程进入主循环")

        while self.running:
            try:
                # 1. 处理采集队列
                self._process_sample_queue()

                # 2. 检查是否触发微调
                self._check_finetune_trigger()

                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"守护线程异常: {e}", exc_info=True)
                time.sleep(1.0)

    def _process_sample_queue(self):
        """处理采集队列中的样本"""
        with self.sample_queue_lock:
            if not self.sample_queue:
                return
            item = self.sample_queue.popleft()

        frame = item['frame'].copy()
        drone_data = item['drone_data']
        frame_id = item['frame_id']

        for drone in drone_data:
            conf = drone.get('confidence', 0.0)
            bbox = drone.get('bbox_2d')
            is_occluded = drone.get('is_occluded', False)

            # 过滤条件
            if conf < self.cfg.conf_threshold:
                continue
            if is_occluded:
                continue
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            bw, bh = x2 - x1, y2 - y1

            # 过滤太小的目标
            if bw < self.cfg.min_drone_size or bh < self.cfg.min_drone_size:
                continue

            # 提取crop
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # 判断是否新场景
            is_new, scene_type = self.scene_db.is_new_scene(frame)

            if is_new:
                # 新场景：保存真实样本
                self.dataset_manager.save_sample(
                    frame, [bbox], split='train', prefix='real'
                )
                self.logger.info(
                    f"新场景样本已保存 | 类型={scene_type} | conf={conf:.3f} | "
                    f"场景库大小={self.scene_db.size()}"
                )

                # 同时生成合成样本填充val集
                if self.dataset_manager.count('val') < 50:
                    self.dataset_manager.save_sample(
                        frame, [bbox], split='val', prefix='real_val'
                    )

            # 无论是否新场景，都把高置信度crop加入crop池
            with self.crop_pool_lock:
                self.crop_pool.append({
                    'crop': crop,
                    'conf': conf,
                    'frame_id': frame_id
                })

            # 如果crop池够了，尝试生成合成样本
            self._try_synthesize(frame, bbox)

    def _try_synthesize(self, background: np.ndarray, original_bbox: List[int]):
        """
        用crop池里的无人机合成新样本
        随机选一个不同的crop植入当前背景
        """
        with self.crop_pool_lock:
            if len(self.crop_pool) < 3:
                return
            # 随机选一个crop（避免用原来同一帧的crop）
            candidates = list(self.crop_pool)

        crop_item = candidates[np.random.randint(0, len(candidates))]
        crop = crop_item['crop']

        # 随机位置（避开原始bbox区域）
        h, w = background.shape[:2]
        x1o, y1o, x2o, y2o = original_bbox
        for _ in range(10):  # 最多尝试10次找合适位置
            cx = np.random.randint(w // 6, w * 5 // 6)
            cy = np.random.randint(h // 6, h * 5 // 6)
            # 检查与原始bbox是否重叠过多
            overlap_x = max(0, min(cx, x2o) - max(cx, x1o))
            overlap_y = max(0, min(cy, y2o) - max(cy, y1o))
            if overlap_x * overlap_y < 500:  # 重叠面积小于500像素则接受
                break

        synth_frame, synth_bbox = self.synthesizer.synthesize(
            background, crop, position=(cx, cy)
        )

        # 合成样本只在训练集中存一定比例
        train_count = self.dataset_manager.count('train')
        synth_count = int(train_count * self.cfg.synthetic_ratio)
        current_synth = len(list(
            (Path(self.cfg.dataset_dir) / 'images' / 'train').glob('synth_*.jpg')
        ))

        if current_synth < synth_count + 10:
            self.dataset_manager.save_sample(
                synth_frame, [synth_bbox], split='train', prefix='synth'
            )
            self.logger.debug(f"合成样本已保存 | 当前合成数={current_synth+1}")

    def _check_finetune_trigger(self):
        """检查是否满足微调触发条件"""
        now = time.time()

        # 条件1：连续空闲超过阈值
        idle_time = now - self.last_detection_time
        if idle_time < self.cfg.idle_trigger_seconds:
            return

        # 条件2：距上次微调超过冷却时间
        if now - self.last_finetune_time < self.cfg.finetune_cooldown:
            return

        # 条件3：有足够的训练样本
        sample_count = self.dataset_manager.count('train')
        if sample_count < self.cfg.min_samples_to_train:
            self.logger.debug(
                f"样本不足，跳过微调 | 当前={sample_count} | "
                f"需要={self.cfg.min_samples_to_train}"
            )
            return

        # 条件4：没有正在进行的微调
        if self.executor.is_finetuning:
            return

        self.logger.info(
            f"触发微调 | 空闲={idle_time:.1f}s | 样本数={sample_count}"
        )
        self.last_finetune_time = now

        # 在新线程里执行微调（不阻塞守护线程）
        finetune_thread = threading.Thread(
            target=self._run_finetune,
            name='FinetuneWorker',
            daemon=True
        )
        finetune_thread.start()

    def _run_finetune(self):
        """执行微调（在独立线程中运行）"""
        success = self.executor.run(self.dataset_manager, self.yolo_model)
        if success:
            self.logger.info("✓ 微调完成，模型已更新")
        else:
            self.logger.warning("✗ 微调失败，继续使用原模型")

    def get_status(self) -> Dict:
        """获取当前状态（供主程序显示）"""
        return {
            'train_samples': self.dataset_manager.count('train'),
            'val_samples': self.dataset_manager.count('val'),
            'scene_count': self.scene_db.size(),
            'crop_pool_size': len(self.crop_pool),
            'is_finetuning': self.executor.is_finetuning,
            'idle_time': time.time() - self.last_detection_time,
            'last_finetune': self.last_finetune_time,
        }


if __name__ == '__main__':
    # 简单测试
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    cfg = FinetuneDaemonConfig()
    print("配置加载成功")
    print(f"  置信度门槛: {cfg.conf_threshold}")
    print(f"  空闲触发时间: {cfg.idle_trigger_seconds}s")
    print(f"  最少样本数: {cfg.min_samples_to_train}")
    print(f"  微调冷却时间: {cfg.finetune_cooldown}s")
    print(f"  冻结层数: {cfg.freeze_layers}")
    print(f"  batch size: {cfg.finetune_batch_size}")
