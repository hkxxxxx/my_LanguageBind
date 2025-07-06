# #!/usr/bin/env python
# """
# 修改后的build_datasets.py - 支持MSR-VTT数据集用于LanguageBind训练
# """
# import os
# import pandas as pd
# import json
# import logging
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist
# from typing import Optional

# def get_data(args, epoch=0):
#     """获取数据加载器"""
#     data = {}
    
#     if args.train_data and args.do_train:
#         # 训练数据 - MSR-VTT
#         train_dataset = MSRVTTTrainDataset(
#             csv_path=args.msrvtt_train_csv,
#             json_path=args.msrvtt_data_json,
#             video_path=args.msrvtt_video_path,
#             tokenizer=args.tokenizer if hasattr(args, 'tokenizer') else None,
#             max_words=getattr(args, 'max_words', 77),
#             max_frames=getattr(args, 'num_frames', 8),
#             image_resolution=getattr(args, 'image_resolution', 224),
#         )
        
#         # 创建分布式采样器
#         if hasattr(args, 'world_size') and args.world_size > 1:
#             sampler = DistributedSampler(
#                 train_dataset,
#                 num_replicas=args.world_size,
#                 rank=args.rank,
#                 shuffle=True,
#                 seed=args.seed,
#                 drop_last=True,
#             )
#         else:
#             sampler = None
            
#         dataloader = DataLoader(
#             train_dataset,
#             batch_size=args.batch_size,
#             shuffle=(sampler is None),
#             sampler=sampler,
#             num_workers=args.workers,
#             pin_memory=True,
#             drop_last=True,
#         )
        
#         dataloader.num_samples = len(train_dataset)
#         dataloader.num_batches = len(dataloader)
        
#         # 同时使用两个键名以兼容不同的检查条件
#         data_info = DataInfo(dataloader=dataloader, sampler=sampler)
#         data["train"] = data_info  # main.py的条件检查需要
#         data[f"{args.clip_type}_pt"] = data_info  # train.py的训练循环需要
        
#     if hasattr(args, 'do_eval') and args.do_eval:
#         # 验证数据
#         if hasattr(args, 'val_vl_ret_data') and args.val_vl_ret_data == 'msrvtt':
#             val_dataset = MSRVTTValDataset(
#                 csv_path=getattr(args, 'msrvtt_val_csv', ''),
#                 json_path=args.msrvtt_data_json,
#                 video_path=args.msrvtt_video_path,
#                 tokenizer=args.tokenizer if hasattr(args, 'tokenizer') else None,
#                 max_words=getattr(args, 'max_words', 77),
#                 max_frames=getattr(args, 'num_frames', 8),
#                 image_resolution=getattr(args, 'image_resolution', 224),
#             )
            
#             val_dataloader = DataLoader(
#                 val_dataset,
#                 batch_size=getattr(args, 'batch_size_val', args.batch_size),
#                 shuffle=False,
#                 num_workers=args.workers,
#                 pin_memory=True,
#                 drop_last=False,
#             )
            
#             val_dataloader.num_samples = len(val_dataset)
#             val_dataloader.num_batches = len(val_dataloader)
            
#             data["vl_ret"] = DataInfo(dataloader=val_dataloader, sampler=None)
    
#     return data

# class DataInfo:
#     """数据信息包装类"""
#     def __init__(self, dataloader, sampler):
#         self.dataloader = dataloader
#         self.sampler = sampler
    
#     def set_epoch(self, epoch):
#         """设置epoch，用于分布式训练"""
#         if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
#             self.sampler.set_epoch(epoch)

# class MSRVTTTrainDataset(Dataset):
#     """MSR-VTT训练数据集"""
    
#     def __init__(
#         self,
#         csv_path: str,
#         json_path: str,
#         video_path: str,
#         tokenizer=None,
#         max_words: int = 77,
#         max_frames: int = 8,
#         image_resolution: int = 224,
#         frame_order: int = 0,
#         slice_framepos: int = 2,
#     ):
#         """
#         Args:
#             csv_path: 训练集CSV文件路径，如 MSRVTT_train.9k.csv
#             json_path: MSR-VTT数据JSON文件路径，如 MSRVTT_data.json
#             video_path: 视频文件夹路径
#             tokenizer: 文本tokenizer
#             max_words: 最大词数
#             max_frames: 最大帧数
#             image_resolution: 图像分辨率
#             frame_order: 帧顺序 (0: 正常, 1: 反向, 2: 随机)
#             slice_framepos: 帧提取位置 (0: 头部, 1: 尾部, 2: 均匀提取)
#         """
#         self.csv_data = pd.read_csv(csv_path)
        
#         # 加载MSR-VTT的JSON数据
#         with open(json_path, 'r') as f:
#             self.json_data = json.load(f)
        
#         self.video_path = video_path
#         self.tokenizer = tokenizer
#         self.max_words = max_words
#         self.max_frames = max_frames
#         self.image_resolution = image_resolution
#         self.frame_order = frame_order
#         self.slice_framepos = slice_framepos
        
#         # 处理sentences数据，创建video_id到captions的映射
#         self.video_captions = {}
#         for sentence in self.json_data['sentences']:
#             video_id = sentence['video_id']
#             caption = sentence['caption']
#             if video_id not in self.video_captions:
#                 self.video_captions[video_id] = []
#             self.video_captions[video_id].append(caption)
        
#         # 创建训练样本列表：每个video_id对应多个caption
#         self.samples = []
#         for _, row in self.csv_data.iterrows():
#             video_id = row['video_id']
#             if video_id in self.video_captions:
#                 for caption in self.video_captions[video_id]:
#                     self.samples.append({
#                         'video_id': video_id,
#                         'caption': caption
#                     })
        
#         logging.info(f"MSR-VTT训练集加载完成: {len(self.samples)} 个video-caption对")
        
#         # 初始化视频处理器
#         self.video_processor = self._init_video_processor()
    
#     def _init_video_processor(self):
#         """初始化视频处理器"""
#         try:
#             from .rawvideo_util import RawVideoExtractor
#             return RawVideoExtractor(
#                 framerate=1.0,
#                 size=self.image_resolution
#             )
#         except ImportError:
#             logging.warning("RawVideoExtractor not found, using basic video processing")
#             return None
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         video_id = sample['video_id']
#         caption = sample['caption']
        
#         # 处理文本
#         text_tokens = self._process_text(caption)
        
#         # 处理视频
#         video_frames = self._process_video(video_id)
        
#         # 返回train.py期望的格式：(images, input_ids, attention_mask)
#         if isinstance(text_tokens, dict):
#             # 如果tokenizer返回字典格式
#             input_ids = text_tokens.get('input_ids', torch.zeros(self.max_words, dtype=torch.long))
#             attention_mask = text_tokens.get('attention_mask', torch.ones(self.max_words, dtype=torch.long))
#             if len(input_ids.shape) > 1:
#                 input_ids = input_ids.squeeze(0)
#             if len(attention_mask.shape) > 1:
#                 attention_mask = attention_mask.squeeze(0)
#         else:
#             # 如果tokenizer返回简单文本，创建假的token
#             import torch
#             input_ids = torch.zeros(self.max_words, dtype=torch.long)
#             attention_mask = torch.ones(self.max_words, dtype=torch.long)
        
#         return video_frames, input_ids, attention_mask
    
#     def _process_text(self, caption):
#         """处理文本"""
#         if self.tokenizer is not None:
#             # 使用LanguageBind的tokenizer
#             tokens = self.tokenizer(
#                 caption,
#                 max_length=self.max_words,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             return tokens
#         else:
#             # 简单的文本处理
#             return caption
    
#     def _process_video(self, video_id):
#         """处理视频"""
#         # 构建视频文件路径
#         video_file = f"{video_id}.mp4"
#         video_full_path = os.path.join(self.video_path, video_file)
        
#         if not os.path.exists(video_full_path):
#             logging.warning(f"Video file not found: {video_full_path}")
#             # 返回正确格式的随机数据: [frames, channels, height, width]
#             import torch
#             return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution)
        
#         try:
#             # 使用OpenCV直接处理视频，避免RawVideoExtractor的参数问题
#             import cv2
#             import torch
            
#             cap = cv2.VideoCapture(video_full_path)
#             frames = []
            
#             # 获取视频总帧数
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             if total_frames == 0:
#                 raise ValueError("Video has no frames")
            
#             # 均匀采样指定数量的帧
#             frame_indices = torch.linspace(0, total_frames - 1, self.max_frames).long().tolist()
            
#             for frame_idx in frame_indices:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#                 ret, frame = cap.read()
#                 if ret:
#                     # 转换BGR到RGB
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     # 调整大小
#                     frame = cv2.resize(frame, (self.image_resolution, self.image_resolution))
#                     # 转换为tensor: [H, W, C] -> [C, H, W]
#                     frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
#                     frames.append(frame_tensor)
#                 else:
#                     # 如果读取失败，复制最后一帧
#                     if frames:
#                         frames.append(frames[-1])
#                     else:
#                         # 如果没有任何帧，创建随机帧
#                         frames.append(torch.randn(3, self.image_resolution, self.image_resolution))
            
#             cap.release()
            
#             # 确保有足够的帧
#             while len(frames) < self.max_frames:
#                 if frames:
#                     frames.append(frames[-1])  # 重复最后一帧
#                 else:
#                     frames.append(torch.randn(3, self.image_resolution, self.image_resolution))
            
#             # 堆叠成 [num_frames, channels, height, width]
#             video_tensor = torch.stack(frames[:self.max_frames])
            
#             return video_tensor
            
#         except Exception as e:
#             logging.warning(f"Error processing video {video_id}: {e}")
#             import torch
#             return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution)

# class MSRVTTValDataset(Dataset):
#     """MSR-VTT验证数据集"""
    
#     def __init__(
#         self,
#         csv_path: str,
#         json_path: str,
#         video_path: str,
#         tokenizer=None,
#         max_words: int = 77,
#         max_frames: int = 8,
#         image_resolution: int = 224,
#     ):
#         if csv_path and os.path.exists(csv_path):
#             self.csv_data = pd.read_csv(csv_path)
#         else:
#             # 如果没有验证集CSV，使用JSON中的测试数据
#             with open(json_path, 'r') as f:
#                 json_data = json.load(f)
            
#             # 创建测试集数据
#             test_video_ids = set()
#             for sentence in json_data['sentences']:
#                 test_video_ids.add(sentence['video_id'])
            
#             # 假设最后1000个video作为测试集
#             test_video_ids = sorted(list(test_video_ids))[-1000:]
#             self.csv_data = pd.DataFrame({'video_id': test_video_ids})
        
#         with open(json_path, 'r') as f:
#             self.json_data = json.load(f)
        
#         self.video_path = video_path
#         self.tokenizer = tokenizer
#         self.max_words = max_words
#         self.max_frames = max_frames
#         self.image_resolution = image_resolution
        
#         # 处理验证集的video-caption对
#         self.video_captions = {}
#         for sentence in self.json_data['sentences']:
#             video_id = sentence['video_id']
#             caption = sentence['caption']
#             if video_id not in self.video_captions:
#                 self.video_captions[video_id] = []
#             self.video_captions[video_id].append(caption)
        
#         # 创建验证样本
#         self.samples = []
#         for _, row in self.csv_data.iterrows():
#             video_id = row['video_id']
#             if video_id in self.video_captions:
#                 # 验证时只使用第一个caption
#                 caption = self.video_captions[video_id][0]
#                 self.samples.append({
#                     'video_id': video_id,
#                     'caption': caption,
#                     'all_captions': self.video_captions[video_id]
#                 })
        
#         logging.info(f"MSR-VTT验证集加载完成: {len(self.samples)} 个样本")
        
#         # 初始化视频处理器
#         self.video_processor = self._init_video_processor()
    
#     def _init_video_processor(self):
#         """初始化视频处理器"""
#         try:
#             from .rawvideo_util import RawVideoExtractor
#             return RawVideoExtractor(
#                 framerate=1.0,
#                 size=self.image_resolution
#             )
#         except ImportError:
#             logging.warning("RawVideoExtractor not found, using basic video processing")
#             return None
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         video_id = sample['video_id']
#         caption = sample['caption']
        
#         # 处理文本
#         text_tokens = self._process_text(caption)
        
#         # 处理视频
#         video_frames = self._process_video(video_id)
        
#         # 返回与训练集相同的格式
#         if isinstance(text_tokens, dict):
#             input_ids = text_tokens.get('input_ids', torch.zeros(self.max_words, dtype=torch.long))
#             attention_mask = text_tokens.get('attention_mask', torch.ones(self.max_words, dtype=torch.long))
#             if len(input_ids.shape) > 1:
#                 input_ids = input_ids.squeeze(0)
#             if len(attention_mask.shape) > 1:
#                 attention_mask = attention_mask.squeeze(0)
#         else:
#             import torch
#             input_ids = torch.zeros(self.max_words, dtype=torch.long)
#             attention_mask = torch.ones(self.max_words, dtype=torch.long)
        
#         return video_frames, input_ids, attention_mask
    
#     def _process_text(self, caption):
#         """处理文本"""
#         if self.tokenizer is not None:
#             tokens = self.tokenizer(
#                 caption,
#                 max_length=self.max_words,
#                 padding='max_length',
#                 truncation=True,
#                 return_tensors='pt'
#             )
#             return tokens
#         else:
#             return caption
    
#     def _process_video(self, video_id):
#         """处理视频"""
#         video_file = f"{video_id}.mp4"
#         video_full_path = os.path.join(self.video_path, video_file)
        
#         if not os.path.exists(video_full_path):
#             logging.warning(f"Video file not found: {video_full_path}")
#             import torch
#             return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution)
        
#         try:
#             # 使用OpenCV直接处理视频
#             import cv2
#             import torch
            
#             cap = cv2.VideoCapture(video_full_path)
#             frames = []
            
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             if total_frames == 0:
#                 raise ValueError("Video has no frames")
            
#             frame_indices = torch.linspace(0, total_frames - 1, self.max_frames).long().tolist()
            
#             for frame_idx in frame_indices:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#                 ret, frame = cap.read()
#                 if ret:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frame = cv2.resize(frame, (self.image_resolution, self.image_resolution))
#                     frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
#                     frames.append(frame_tensor)
#                 else:
#                     if frames:
#                         frames.append(frames[-1])
#                     else:
#                         frames.append(torch.randn(3, self.image_resolution, self.image_resolution))
            
#             cap.release()
            
#             while len(frames) < self.max_frames:
#                 if frames:
#                     frames.append(frames[-1])
#                 else:
#                     frames.append(torch.randn(3, self.image_resolution, self.image_resolution))
            
#             video_tensor = torch.stack(frames[:self.max_frames])
#             return video_tensor
            
#         except Exception as e:
#             logging.warning(f"Error processing video {video_id}: {e}")
#             import torch
#             return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution)

# def create_msrvtt_dataloader(args, split='train'):
#     """创建MSR-VTT数据加载器的辅助函数"""
#     if split == 'train':
#         dataset = MSRVTTTrainDataset(
#             csv_path=args.msrvtt_train_csv,
#             json_path=args.msrvtt_data_json,
#             video_path=args.msrvtt_video_path,
#             tokenizer=getattr(args, 'tokenizer', None),
#             max_words=getattr(args, 'max_words', 77),
#             max_frames=getattr(args, 'num_frames', 8),
#             image_resolution=getattr(args, 'image_resolution', 224),
#         )
        
#         if hasattr(args, 'world_size') and args.world_size > 1:
#             sampler = DistributedSampler(dataset, shuffle=True)
#         else:
#             sampler = None
            
#         dataloader = DataLoader(
#             dataset,
#             batch_size=args.batch_size,
#             shuffle=(sampler is None),
#             sampler=sampler,
#             num_workers=args.workers,
#             pin_memory=True,
#             drop_last=True,
#         )
    
#     elif split == 'val':
#         dataset = MSRVTTValDataset(
#             csv_path=getattr(args, 'msrvtt_val_csv', ''),
#             json_path=args.msrvtt_data_json,
#             video_path=args.msrvtt_video_path,
#             tokenizer=getattr(args, 'tokenizer', None),
#             max_words=getattr(args, 'max_words', 77),
#             max_frames=getattr(args, 'num_frames', 8),
#             image_resolution=getattr(args, 'image_resolution', 224),
#         )
        
#         dataloader = DataLoader(
#             dataset,
#             batch_size=getattr(args, 'batch_size_val', args.batch_size),
#             shuffle=False,
#             num_workers=args.workers,
#             pin_memory=True,
#             drop_last=False,
#         )
    
#     return dataloader, dataset

#!/usr/bin/env python
"""
修改后的build_datasets.py - 支持MSR-VTT数据集用于LanguageBind训练
修复设备和数据类型不匹配问题
"""
import os
import random
import pandas as pd
import json
import logging
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import cv2
import numpy as np
from typing import Optional

def get_data(args, epoch=0):
    """获取数据加载器"""
    data = {}
    
    if args.train_data and args.do_train:
        # 训练数据 - MSR-VTT
        train_dataset = MSRVTTTrainDataset(
            csv_path=args.msrvtt_train_csv,
            json_path=args.msrvtt_data_json,
            video_path=args.msrvtt_video_path,
            tokenizer=args.tokenizer if hasattr(args, 'tokenizer') else None,
            max_words=getattr(args, 'max_words', 77),
            max_frames=getattr(args, 'num_frames', 8),
            image_resolution=getattr(args, 'image_resolution', 224),
        )
        
        # 创建分布式采样器
        if hasattr(args, 'world_size') and args.world_size > 1:
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
        else:
            sampler = None
            
        dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        
        dataloader.num_samples = len(train_dataset)
        dataloader.num_batches = len(dataloader)
        
        # 同时使用两个键名以兼容不同的检查条件
        data_info = DataInfo(dataloader=dataloader, sampler=sampler)
        data["train"] = data_info  # main.py的条件检查需要
        data[f"{args.clip_type}_pt"] = data_info  # train.py的训练循环需要
        
    if hasattr(args, 'do_eval') and args.do_eval:
        # 验证数据
        if hasattr(args, 'val_vl_ret_data') and args.val_vl_ret_data == 'msrvtt':
            val_dataset = MSRVTTValDataset(
                csv_path=getattr(args, 'msrvtt_val_csv', ''),
                json_path=args.msrvtt_data_json,
                video_path=args.msrvtt_video_path,
                tokenizer=args.tokenizer if hasattr(args, 'tokenizer') else None,
                max_words=getattr(args, 'max_words', 77),
                max_frames=getattr(args, 'num_frames', 8),
                image_resolution=getattr(args, 'image_resolution', 224),
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=getattr(args, 'batch_size_val', args.batch_size),
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            
            val_dataloader.num_samples = len(val_dataset)
            val_dataloader.num_batches = len(val_dataloader)
            
            data["vl_ret"] = DataInfo(dataloader=val_dataloader, sampler=None)
    
    return data

class DataInfo:
    """数据信息包装类"""
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler
    
    def set_epoch(self, epoch):
        """设置epoch，用于分布式训练"""
        if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

class MSRVTTTrainDataset(Dataset):
    """MSR-VTT训练数据集 - 修复数据类型和设备问题"""
    
    def __init__(
        self,
        csv_path: str,
        json_path: str,
        video_path: str,
        tokenizer=None,
        max_words: int = 77,
        max_frames: int = 8,
        image_resolution: int = 224,
        frame_order: int = 0,
        slice_framepos: int = 2,
    ):
        """
        Args:
            csv_path: 训练集CSV文件路径，如 MSRVTT_train.9k.csv
            json_path: MSR-VTT数据JSON文件路径，如 MSRVTT_data.json
            video_path: 视频文件夹路径
            tokenizer: 文本tokenizer
            max_words: 最大词数
            max_frames: 最大帧数
            image_resolution: 图像分辨率
            frame_order: 帧顺序 (0: 正常, 1: 反向, 2: 随机)
            slice_framepos: 帧提取位置 (0: 头部, 1: 尾部, 2: 均匀提取)
        """
        self.csv_data = pd.read_csv(csv_path)
        
        # 加载MSR-VTT的JSON数据
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.frame_order = frame_order
        self.slice_framepos = slice_framepos
        
        # 处理sentences数据，创建video_id到captions的映射
        self.video_captions = {}
        for sentence in self.json_data['sentences']:
            video_id = sentence['video_id']
            caption = sentence['caption']
            if video_id not in self.video_captions:
                self.video_captions[video_id] = []
            self.video_captions[video_id].append(caption)
        
        # 创建训练样本列表：每个video_id对应多个caption
        self.samples = []
        for _, row in self.csv_data.iterrows():
            video_id = row['video_id']
            if video_id in self.video_captions:
                caption = random.choice(self.video_captions[video_id])
                self.samples.append({
                    'video_id': video_id,
                    'caption': caption
                })
        
        logging.info(f"MSR-VTT训练集加载完成: {len(self.samples)} 个video-caption对")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        caption = sample['caption']
        
        # 处理文本
        text_tokens = self._process_text(caption)
        
        # 处理视频
        video_frames = self._process_video(video_id)
        
        # 确保返回正确的数据类型
        if isinstance(text_tokens, dict):
            input_ids = text_tokens.get('input_ids', torch.zeros(self.max_words, dtype=torch.long))
            attention_mask = text_tokens.get('attention_mask', torch.ones(self.max_words, dtype=torch.long))
            
            # 确保维度正确
            if len(input_ids.shape) > 1:
                input_ids = input_ids.squeeze(0)
            if len(attention_mask.shape) > 1:
                attention_mask = attention_mask.squeeze(0)
                
            # 确保长度正确
            if input_ids.shape[0] != self.max_words:
                if input_ids.shape[0] > self.max_words:
                    input_ids = input_ids[:self.max_words]
                    attention_mask = attention_mask[:self.max_words]
                else:
                    # 填充到指定长度
                    pad_length = self.max_words - input_ids.shape[0]
                    input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        else:
            # 如果tokenizer返回简单文本，创建假的token
            input_ids = torch.zeros(self.max_words, dtype=torch.long)
            attention_mask = torch.ones(self.max_words, dtype=torch.long)
        
        # 确保视频tensor是float32类型
        video_frames = video_frames.float()
        
        return video_frames, input_ids, attention_mask
    
    def _process_text(self, caption):
        """处理文本"""
        if self.tokenizer is not None:
            try:
                # 使用LanguageBind的tokenizer
                tokens = self.tokenizer(
                    caption,
                    max_length=self.max_words,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return tokens
            except Exception as e:
                logging.warning(f"Tokenizer failed: {e}, using simple processing")
                return caption
        else:
            # 简单的文本处理
            return caption
    
    def _process_video(self, video_id):
        """处理视频 - 修复数据类型问题"""
        # 构建视频文件路径
        video_file = f"{video_id}.mp4"
        video_full_path = os.path.join(self.video_path, video_file)
        
        if not os.path.exists(video_full_path):
            logging.warning(f"Video file not found: {video_full_path}")
            # 返回正确格式的随机数据: [frames, channels, height, width]
            return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution, dtype=torch.float32)
        
        try:
            # 使用OpenCV直接处理视频
            cap = cv2.VideoCapture(video_full_path)
            frames = []
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            # 均匀采样指定数量的帧
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames).astype(int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # 转换BGR到RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 调整大小
                    frame = cv2.resize(frame, (self.image_resolution, self.image_resolution))
                    # 转换为tensor: [H, W, C] -> [C, H, W], 归一化到[0,1]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                else:
                    # 如果读取失败，复制最后一帧或创建随机帧
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        # 如果没有任何帧，创建随机帧
                        frames.append(torch.randn(3, self.image_resolution, self.image_resolution, dtype=torch.float32))
            
            cap.release()
            
            # 确保有足够的帧
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1].clone())  # 重复最后一帧
                else:
                    frames.append(torch.randn(3, self.image_resolution, self.image_resolution, dtype=torch.float32))
            
            # 堆叠成 [num_frames, channels, height, width]，确保是float32
            video_tensor = torch.stack(frames[:self.max_frames]).float()
            
            return video_tensor
            
        except Exception as e:
            logging.warning(f"Error processing video {video_id}: {e}")
            return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution, dtype=torch.float32)

class MSRVTTValDataset(Dataset):
    """MSR-VTT验证数据集 - 修复数据类型和设备问题"""
    
    def __init__(
        self,
        csv_path: str,
        json_path: str,
        video_path: str,
        tokenizer=None,
        max_words: int = 77,
        max_frames: int = 8,
        image_resolution: int = 224,
    ):
        if csv_path and os.path.exists(csv_path):
            self.csv_data = pd.read_csv(csv_path)
        else:
            # 如果没有验证集CSV，使用JSON中的测试数据
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # 创建测试集数据
            test_video_ids = set()
            for sentence in json_data['sentences']:
                test_video_ids.add(sentence['video_id'])
            
            # 假设最后1000个video作为测试集
            test_video_ids = sorted(list(test_video_ids))[-1000:]
            self.csv_data = pd.DataFrame({'video_id': test_video_ids})
        
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        
        # 处理验证集的video-caption对
        self.video_captions = {}
        for sentence in self.json_data['sentences']:
            video_id = sentence['video_id']
            caption = sentence['caption']
            if video_id not in self.video_captions:
                self.video_captions[video_id] = []
            self.video_captions[video_id].append(caption)
        
        # 创建验证样本
        self.samples = []
        for _, row in self.csv_data.iterrows():
            video_id = row['video_id']
            if video_id in self.video_captions:
                # 验证时只使用第一个caption
                caption = self.video_captions[video_id][0]
                self.samples.append({
                    'video_id': video_id,
                    'caption': caption,
                    'all_captions': self.video_captions[video_id]
                })
        
        logging.info(f"MSR-VTT验证集加载完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        caption = sample['caption']
        
        # 处理文本
        text_tokens = self._process_text(caption)
        
        # 处理视频
        video_frames = self._process_video(video_id)
        
        # 返回与训练集相同的格式
        if isinstance(text_tokens, dict):
            input_ids = text_tokens.get('input_ids', torch.zeros(self.max_words, dtype=torch.long))
            attention_mask = text_tokens.get('attention_mask', torch.ones(self.max_words, dtype=torch.long))
            
            # 确保维度正确
            if len(input_ids.shape) > 1:
                input_ids = input_ids.squeeze(0)
            if len(attention_mask.shape) > 1:
                attention_mask = attention_mask.squeeze(0)
                
            # 确保长度正确
            if input_ids.shape[0] != self.max_words:
                if input_ids.shape[0] > self.max_words:
                    input_ids = input_ids[:self.max_words]
                    attention_mask = attention_mask[:self.max_words]
                else:
                    # 填充到指定长度
                    pad_length = self.max_words - input_ids.shape[0]
                    input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        else:
            input_ids = torch.zeros(self.max_words, dtype=torch.long)
            attention_mask = torch.ones(self.max_words, dtype=torch.long)
        
        # 确保视频tensor是float32类型
        video_frames = video_frames.float()
        
        return video_frames, input_ids, attention_mask
    
    def _process_text(self, caption):
        """处理文本"""
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer(
                    caption,
                    max_length=self.max_words,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return tokens
            except Exception as e:
                logging.warning(f"Tokenizer failed: {e}, using simple processing")
                return caption
        else:
            return caption
    
    def _process_video(self, video_id):
        """处理视频 - 修复数据类型问题"""
        video_file = f"{video_id}.mp4"
        video_full_path = os.path.join(self.video_path, video_file)
        
        if not os.path.exists(video_full_path):
            logging.warning(f"Video file not found: {video_full_path}")
            return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution, dtype=torch.float32)
        
        try:
            # 使用OpenCV直接处理视频
            cap = cv2.VideoCapture(video_full_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            frame_indices = np.linspace(0, total_frames - 1, self.max_frames).astype(int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_resolution, self.image_resolution))
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                else:
                    if frames:
                        frames.append(frames[-1].clone())
                    else:
                        frames.append(torch.randn(3, self.image_resolution, self.image_resolution, dtype=torch.float32))
            
            cap.release()
            
            while len(frames) < self.max_frames:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.randn(3, self.image_resolution, self.image_resolution, dtype=torch.float32))
            
            video_tensor = torch.stack(frames[:self.max_frames]).float()
            return video_tensor
            
        except Exception as e:
            logging.warning(f"Error processing video {video_id}: {e}")
            return torch.randn(self.max_frames, 3, self.image_resolution, self.image_resolution, dtype=torch.float32)

def create_msrvtt_dataloader(args, split='train'):
    """创建MSR-VTT数据加载器的辅助函数"""
    if split == 'train':
        dataset = MSRVTTTrainDataset(
            csv_path=args.msrvtt_train_csv,
            json_path=args.msrvtt_data_json,
            video_path=args.msrvtt_video_path,
            tokenizer=getattr(args, 'tokenizer', None),
            max_words=getattr(args, 'max_words', 77),
            max_frames=getattr(args, 'num_frames', 8),
            image_resolution=getattr(args, 'image_resolution', 224),
        )
        
        if hasattr(args, 'world_size') and args.world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None
            
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
    
    elif split == 'val':
        dataset = MSRVTTValDataset(
            csv_path=getattr(args, 'msrvtt_val_csv', ''),
            json_path=args.msrvtt_data_json,
            video_path=args.msrvtt_video_path,
            tokenizer=getattr(args, 'tokenizer', None),
            max_words=getattr(args, 'max_words', 77),
            max_frames=getattr(args, 'num_frames', 8),
            image_resolution=getattr(args, 'image_resolution', 224),
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=getattr(args, 'batch_size_val', args.batch_size),
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    
    return dataloader, dataset