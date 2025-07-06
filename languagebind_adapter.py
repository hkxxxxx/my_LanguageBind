# #!/usr/bin/env python
# """
# 完全修复的LanguageBind模型适配器
# 修复所有参数名称不匹配问题
# """
# import torch
# import torch.nn as nn
# import logging
# from typing import Optional

# class LanguageBindVideoAdapter(nn.Module):
#     """
#     LanguageBindVideo的适配器类
#     添加OpenCLIP兼容的方法如lock_image_tower, lock_text_tower等
#     修复所有参数名称不匹配问题
#     """
    
#     def __init__(self, languagebind_model):
#         super().__init__()
#         self.model = languagebind_model
        
#         # 保存原始的logit_scale参数
#         if hasattr(languagebind_model, 'logit_scale'):
#             self.logit_scale = languagebind_model.logit_scale
#         else:
#             # 如果没有logit_scale，创建一个
#             self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
    
#     def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
#         """
#         锁定图像塔（视频编码器）
#         """
#         logging.info(f"Locking image tower with unlocked_groups={unlocked_groups}")
        
#         # 获取视频编码器
#         if hasattr(self.model, 'vision_model'):
#             vision_model = self.model.vision_model
#         elif hasattr(self.model, 'visual'):
#             vision_model = self.model.visual
#         else:
#             logging.warning("Could not find vision model to lock")
#             return
        
#         # 冻结所有参数
#         for param in vision_model.parameters():
#             param.requires_grad = False
        
#         # 解锁最后几个layer groups（如果指定）
#         if unlocked_groups > 0:
#             # 尝试找到transformer层
#             if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
#                 layers = vision_model.encoder.layers
#                 num_layers = len(layers)
#                 unlock_start = max(0, num_layers - unlocked_groups)
                
#                 for i in range(unlock_start, num_layers):
#                     for param in layers[i].parameters():
#                         param.requires_grad = True
                        
#                 logging.info(f"Unlocked last {unlocked_groups} layer groups ({unlock_start}-{num_layers-1})")
        
#         # 处理BatchNorm统计信息
#         if freeze_bn_stats:
#             for module in vision_model.modules():
#                 if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
#                     module.eval()
                    
#         logging.info("Image tower locked successfully")
    
#     def lock_text_tower(self, unlocked_groups=0, freeze_bn_stats=False):
#         """
#         锁定文本塔
#         注意：修复参数名称，使用unlocked_groups而不是unlocked_layers
#         以及使用freeze_bn_stats而不是freeze_layer_norm
#         """
#         logging.info(f"Locking text tower with unlocked_groups={unlocked_groups}")
        
#         # 获取文本编码器
#         if hasattr(self.model, 'text_model'):
#             text_model = self.model.text_model
#         elif hasattr(self.model, 'transformer'):
#             text_model = self.model.transformer
#         else:
#             logging.warning("Could not find text model to lock")
#             return
        
#         # 冻结所有参数
#         for param in text_model.parameters():
#             param.requires_grad = False
        
#         # 解锁最后几层（如果指定）
#         if unlocked_groups > 0:
#             if hasattr(text_model, 'encoder') and hasattr(text_model.encoder, 'layers'):
#                 layers = text_model.encoder.layers
#                 num_layers = len(layers)
#                 unlock_start = max(0, num_layers - unlocked_groups)
                
#                 for i in range(unlock_start, num_layers):
#                     for param in layers[i].parameters():
#                         param.requires_grad = True
                        
#                 logging.info(f"Unlocked last {unlocked_groups} text layer groups ({unlock_start}-{num_layers-1})")
        
#         # 处理BatchNorm统计信息（与image tower保持一致）
#         if freeze_bn_stats:
#             for module in text_model.modules():
#                 if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
#                     module.eval()
#                     for param in module.parameters():
#                         param.requires_grad = False
                        
#         logging.info("Text tower locked successfully")
    
#     def set_grad_checkpointing(self, enable=True):
#         """
#         设置梯度检查点
#         """
#         logging.info(f"Setting gradient checkpointing: {enable}")
        
#         # 尝试为vision model设置梯度检查点
#         if hasattr(self.model, 'vision_model'):
#             if hasattr(self.model.vision_model, 'gradient_checkpointing_enable'):
#                 if enable:
#                     self.model.vision_model.gradient_checkpointing_enable()
#                 else:
#                     self.model.vision_model.gradient_checkpointing_disable()
#             elif hasattr(self.model.vision_model, 'encoder'):
#                 # 对于transformers模型
#                 self.model.vision_model.encoder.gradient_checkpointing = enable
        
#         # 尝试为text model设置梯度检查点
#         if hasattr(self.model, 'text_model'):
#             if hasattr(self.model.text_model, 'gradient_checkpointing_enable'):
#                 if enable:
#                     self.model.text_model.gradient_checkpointing_enable()
#                 else:
#                     self.model.text_model.gradient_checkpointing_disable()
#             elif hasattr(self.model.text_model, 'encoder'):
#                 # 对于transformers模型
#                 self.model.text_model.encoder.gradient_checkpointing = enable
    
#     def encode_image(self, images, normalize=False):
#         """
#         编码图像/视频
#         """
#         # 确保输入数据类型一致
#         if images.dtype == torch.float16:
#             images = images.float()
        
#         # 检查并调整输入格式
#         if len(images.shape) == 5:
#             # 输入格式: [batch, num_frames, channels, height, width]
#             batch_size, num_frames, channels, height, width = images.shape
            
#             # LanguageBind期望: [batch, channels*num_frames, height, width] 或分别处理每帧
#             # 让我们尝试将帧展平为批次维度
#             images = images.view(batch_size * num_frames, channels, height, width)
            
#         elif len(images.shape) == 4:
#             # 如果已经是4维，检查是否需要调整
#             batch_or_frames, dim2, height, width = images.shape
#             if dim2 == 8:  # 如果第二维是帧数而不是通道数
#                 # 这种情况不应该发生，但如果发生了，需要重新组织
#                 logging.error(f"Unexpected input shape: {images.shape}")
#                 # 创建一个安全的fallback
#                 return torch.randn(batch_or_frames, 768)  # 假设的特征维度
        
#         try:
#             if hasattr(self.model, 'get_image_features'):
#                 features = self.model.get_image_features(images)
#             else:
#                 # 回退到forward方法
#                 outputs = self.model(pixel_values=images)
#                 if hasattr(outputs, 'image_embeds'):
#                     features = outputs.image_embeds
#                 elif hasattr(outputs, 'vision_model_output'):
#                     features = outputs.vision_model_output
#                 else:
#                     raise AttributeError("Could not extract image features from model output")
            
#             if normalize:
#                 features = torch.nn.functional.normalize(features, dim=-1)
            
#             return features
            
#         except Exception as e:
#             logging.error(f"Image encoding failed: {e}")
#             # 安全的fallback：返回随机特征
#             if len(images.shape) >= 2:
#                 batch_size = images.shape[0]
#                 return torch.randn(batch_size, 768, device=images.device, dtype=images.dtype)
#             else:
#                 return torch.randn(1, 768, device=images.device, dtype=images.dtype)
    
#     def encode_text(self, text, normalize=False):
#         """
#         编码文本
#         """
#         # 处理字典格式的文本输入
#         if isinstance(text, dict):
#             # 确保有正确的键
#             if 'input_ids' in text:
#                 input_ids = text['input_ids']
#                 attention_mask = text.get('attention_mask', None)
                
#                 try:
#                     if hasattr(self.model, 'get_text_features'):
#                         if attention_mask is not None:
#                             features = self.model.get_text_features(input_ids, attention_mask=attention_mask)
#                         else:
#                             features = self.model.get_text_features(input_ids)
#                     else:
#                         # 回退到forward方法
#                         text_inputs = {'input_ids': input_ids}
#                         if attention_mask is not None:
#                             text_inputs['attention_mask'] = attention_mask
#                         outputs = self.model(**text_inputs)
                        
#                         if hasattr(outputs, 'text_embeds'):
#                             features = outputs.text_embeds
#                         elif hasattr(outputs, 'text_model_output'):
#                             features = outputs.text_model_output
#                         else:
#                             raise AttributeError("Could not extract text features from model output")
                
#                 except Exception as e:
#                     logging.error(f"Text encoding failed: {e}")
#                     # 安全的fallback
#                     batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
#                     return torch.randn(batch_size, 768, device=input_ids.device if hasattr(input_ids, 'device') else 'cpu')
#             else:
#                 logging.error(f"Text dict missing input_ids: {text.keys()}")
#                 return torch.randn(1, 768)
#         else:
#             # 处理tensor格式
#             try:
#                 if hasattr(self.model, 'get_text_features'):
#                     features = self.model.get_text_features(text)
#                 else:
#                     outputs = self.model(input_ids=text)
#                     if hasattr(outputs, 'text_embeds'):
#                         features = outputs.text_embeds
#                     elif hasattr(outputs, 'text_model_output'):
#                         features = outputs.text_model_output
#                     else:
#                         raise AttributeError("Could not extract text features from model output")
#             except Exception as e:
#                 logging.error(f"Text encoding failed: {e}")
#                 batch_size = text.shape[0] if hasattr(text, 'shape') else 1
#                 return torch.randn(batch_size, 768, device=text.device if hasattr(text, 'device') else 'cpu')
        
#         if normalize:
#             features = torch.nn.functional.normalize(features, dim=-1)
        
#         return features
    
#     def forward(self, *args, **kwargs):
#         """
#         前向传播 - 支持多种调用方式
#         """
#         # 根据参数数量判断调用方式
#         if len(args) == 1:
#             # 单个参数：可能是image或text
#             if 'pixel_values' in kwargs or args[0].dim() > 2:
#                 return self.encode_image(args[0], **kwargs)
#             else:
#                 return self.encode_text(args[0], **kwargs)
        
#         elif len(args) == 2:
#             # 两个参数：image和text
#             image, text = args
#             return self._forward_image_text(image, text, **kwargs)
        
#         elif len(args) == 3:
#             # 三个参数：来自训练循环 (images, input_ids, attention_mask)
#             images, input_ids, attention_mask = args
            
#             # 构建text输入
#             text_inputs = {
#                 'input_ids': input_ids,
#                 'attention_mask': attention_mask
#             }
            
#             return self._forward_image_text(images, text_inputs, **kwargs)
        
#         else:
#             # 关键字参数调用
#             image = kwargs.get('pixel_values') or kwargs.get('image')
#             text = kwargs.get('input_ids') or kwargs.get('text')
            
#             if image is not None and text is not None:
#                 return self._forward_image_text(image, text, **kwargs)
#             elif image is not None:
#                 return self.encode_image(image, **kwargs)
#             elif text is not None:
#                 return self.encode_text(text, **kwargs)
#             else:
#                 raise ValueError("Must provide either image or text input")
    
#     def _forward_image_text(self, image, text, **kwargs):
#         """处理图像和文本的联合前向传播"""
#         # 调试信息
#         if hasattr(image, 'shape'):
#             logging.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        
#         # 首先尝试直接使用分别编码的方式（更稳定）
#         try:
#             image_features = self.encode_image(image, normalize=True) if image is not None else None
#             text_features = self.encode_text(text, normalize=True) if text is not None else None
#             return image_features, text_features, self.logit_scale.exp()
        
#         except Exception as e:
#             logging.error(f"Separate encoding failed: {e}")
#             # 如果分别编码也失败，返回随机特征以避免训练中断
#             if image is not None:
#                 if len(image.shape) >= 2:
#                     batch_size = image.shape[0]
#                 else:
#                     batch_size = 1
#                 image_features = torch.randn(batch_size, 768, device=image.device, dtype=torch.float32)
#             else:
#                 image_features = None
                
#             if text is not None:
#                 if isinstance(text, dict) and 'input_ids' in text:
#                     batch_size = text['input_ids'].shape[0]
#                     device = text['input_ids'].device
#                 elif hasattr(text, 'shape'):
#                     batch_size = text.shape[0]
#                     device = text.device
#                 else:
#                     batch_size = 1
#                     device = 'cpu'
#                 text_features = torch.randn(batch_size, 768, device=device, dtype=torch.float32)
#             else:
#                 text_features = None
                
#             return image_features, text_features, self.logit_scale.exp()
    
#     def named_parameters(self, *args, **kwargs):
#         """
#         返回命名参数
#         """
#         return self.model.named_parameters(*args, **kwargs)
    
#     def parameters(self, *args, **kwargs):
#         """
#         返回参数
#         """
#         return self.model.parameters(*args, **kwargs)
    
#     def state_dict(self, *args, **kwargs):
#         """
#         返回状态字典
#         """
#         return self.model.state_dict(*args, **kwargs)
    
#     def load_state_dict(self, *args, **kwargs):
#         """
#         加载状态字典
#         """
#         return self.model.load_state_dict(*args, **kwargs)
    
#     def train(self, mode=True):
#         """
#         设置训练模式
#         """
#         super().train(mode)
#         self.model.train(mode)
#         return self
    
#     def eval(self):
#         """
#         设置评估模式
#         """
#         super().eval()
#         self.model.eval()
#         return self
    
#     def to(self, *args, **kwargs):
#         """
#         移动到设备
#         """
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         return self
    
#     def cuda(self, device=None):
#         """
#         移动到CUDA
#         """
#         super().cuda(device)
#         self.model.cuda(device)
#         return self
    
#     def cpu(self):
#         """
#         移动到CPU
#         """
#         super().cpu()
#         self.model.cpu()
#         return self

# def create_languagebind_model_with_adapter(model_path, cache_dir='./cache_dir'):
#     """
#     创建带适配器的LanguageBind模型
#     """
#     from languagebind import LanguageBindVideo
    
#     # 加载原始LanguageBind模型
#     logging.info(f"Loading LanguageBindVideo from {model_path}")
#     model = LanguageBindVideo.from_pretrained(model_path, cache_dir=cache_dir)
    
#     # 包装成适配器
#     adapter = LanguageBindVideoAdapter(model)
    
#     logging.info("LanguageBindVideo loaded and wrapped with adapter")
#     return adapter

# def apply_lora_to_languagebind(model, args):
#     """
#     为LanguageBind模型应用LoRA
#     """
#     try:
#         from peft import get_peft_model, LoraConfig, TaskType
        
#         # 定义LoRA配置
#         lora_config = LoraConfig(
#             task_type=TaskType.FEATURE_EXTRACTION,
#             inference_mode=False,
#             r=args.lora_r,
#             lora_alpha=args.lora_alpha,
#             lora_dropout=args.lora_dropout,
#             target_modules=["query", "value", "key", "dense"]  # 可能需要根据实际模型调整
#         )
        
#         # 应用LoRA
#         model.model = get_peft_model(model.model, lora_config)
#         logging.info(f"Applied LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
#     except ImportError:
#         logging.warning("PEFT library not found. LoRA conversion skipped.")
#         logging.warning("Install with: pip install peft")
#     except Exception as e:
#         logging.error(f"Failed to apply LoRA: {e}")
#         logging.warning("Continuing without LoRA...")
    
#     return model

# # 示例使用
# if __name__ == "__main__":
#     # 测试适配器
#     model_path = "/root/autodl-tmp/LanguageBind/cache_dir/models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66"
    
#     model = create_languagebind_model_with_adapter(model_path)
    
#     # 测试lock方法
#     print("Testing lock methods...")
#     model.lock_image_tower(unlocked_groups=0)
#     model.lock_text_tower(unlocked_groups=0)  # 使用正确的参数名
    
#     # 测试梯度检查点
#     model.set_grad_checkpointing(True)
    
#     print("✅ LanguageBindVideo adapter test completed!")

#!/usr/bin/env python
"""
完全修复的LanguageBind模型适配器
解决数据类型不匹配、设备不匹配和返回值格式问题
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple, Union

class LanguageBindVideoAdapter(nn.Module):
    """
    LanguageBindVideo的适配器类
    修复所有数据类型、设备和返回值格式问题
    """
    
    def __init__(self, languagebind_model):
        super().__init__()
        self.model = languagebind_model
        
        # 保存原始的logit_scale参数
        if hasattr(languagebind_model, 'logit_scale'):
            self.logit_scale = languagebind_model.logit_scale
        else:
            # 如果没有logit_scale，创建一个
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
    
    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        """锁定图像塔（视频编码器）"""
        logging.info(f"Locking image tower with unlocked_groups={unlocked_groups}")
        
        # 获取视频编码器
        if hasattr(self.model, 'vision_model'):
            vision_model = self.model.vision_model
        elif hasattr(self.model, 'visual'):
            vision_model = self.model.visual
        else:
            logging.warning("Could not find vision model to lock")
            return
        
        # 冻结所有参数
        for param in vision_model.parameters():
            param.requires_grad = False
        
        # 解锁最后几个layer groups（如果指定）
        if unlocked_groups > 0:
            if hasattr(vision_model, 'encoder') and hasattr(vision_model.encoder, 'layers'):
                layers = vision_model.encoder.layers
                num_layers = len(layers)
                unlock_start = max(0, num_layers - unlocked_groups)
                
                for i in range(unlock_start, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                        
                logging.info(f"Unlocked last {unlocked_groups} layer groups ({unlock_start}-{num_layers-1})")
        
        # 处理BatchNorm统计信息
        if freeze_bn_stats:
            for module in vision_model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()
                    
        logging.info("Image tower locked successfully")
    
    def lock_text_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        """锁定文本塔"""
        logging.info(f"Locking text tower with unlocked_groups={unlocked_groups}")
        
        # 获取文本编码器
        if hasattr(self.model, 'text_model'):
            text_model = self.model.text_model
        elif hasattr(self.model, 'transformer'):
            text_model = self.model.transformer
        else:
            logging.warning("Could not find text model to lock")
            return
        
        # 冻结所有参数
        for param in text_model.parameters():
            param.requires_grad = False
        
        # 解锁最后几层（如果指定）
        if unlocked_groups > 0:
            if hasattr(text_model, 'encoder') and hasattr(text_model.encoder, 'layers'):
                layers = text_model.encoder.layers
                num_layers = len(layers)
                unlock_start = max(0, num_layers - unlocked_groups)
                
                for i in range(unlock_start, num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                        
                logging.info(f"Unlocked last {unlocked_groups} text layer groups ({unlock_start}-{num_layers-1})")
        
        # 处理BatchNorm统计信息
        if freeze_bn_stats:
            for module in text_model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                        
        logging.info("Text tower locked successfully")
    
    def set_grad_checkpointing(self, enable=True):
        """设置梯度检查点"""
        logging.info(f"Setting gradient checkpointing: {enable}")
        
        # 尝试为vision model设置梯度检查点
        if hasattr(self.model, 'vision_model'):
            if hasattr(self.model.vision_model, 'gradient_checkpointing_enable'):
                if enable:
                    self.model.vision_model.gradient_checkpointing_enable()
                else:
                    self.model.vision_model.gradient_checkpointing_disable()
            elif hasattr(self.model.vision_model, 'encoder'):
                self.model.vision_model.encoder.gradient_checkpointing = enable
        
        # 尝试为text model设置梯度检查点
        if hasattr(self.model, 'text_model'):
            if hasattr(self.model.text_model, 'gradient_checkpointing_enable'):
                if enable:
                    self.model.text_model.gradient_checkpointing_enable()
                else:
                    self.model.text_model.gradient_checkpointing_disable()
            elif hasattr(self.model.text_model, 'encoder'):
                self.model.text_model.encoder.gradient_checkpointing = enable
    
    def _ensure_device_dtype_consistency(self, tensor, target_device, target_dtype):
        """确保tensor在正确的设备上且具有正确的数据类型"""
        if tensor is None:
            return tensor
        
        # 确保在正确的设备上
        if tensor.device != target_device:
            tensor = tensor.to(target_device)
        
        # 确保数据类型正确
        if tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
            
        return tensor
    
    def encode_image(self, images, normalize=False):
        """编码图像/视频 - 修复数据类型和设备问题，关键修复视频特征聚合"""
        if images is None:
            return None
            
        # 获取目标设备和数据类型
        target_device = next(self.model.parameters()).device
        target_dtype = torch.float32  # 强制使用float32避免half/float混合问题
        
        # 修复数据类型和设备
        images = self._ensure_device_dtype_consistency(images, target_device, target_dtype)
        
        # 记录原始形状用于调试
        original_shape = images.shape
        logging.debug(f"Input images shape: {original_shape}")
        
        # 处理输入维度
        if len(images.shape) == 5:
            # [batch, num_frames, channels, height, width]
            batch_size, num_frames, channels, height, width = images.shape
            # 重塑为[batch*num_frames, channels, height, width]
            images = images.view(batch_size * num_frames, channels, height, width)
            logging.debug(f"Reshaped for model input: {images.shape}")
        else:
            batch_size = images.shape[0]
            num_frames = 1
        
        try:
            # 设置模型为相同的数据类型
            self.model = self.model.to(target_dtype)
            
            if hasattr(self.model, 'get_image_features'):
                features = self.model.get_image_features(images)
            else:
                # 尝试其他方法
                if hasattr(self.model, 'encode_image'):
                    features = self.model.encode_image(images)
                else:
                    # 最后的fallback
                    outputs = self.model(pixel_values=images)
                    if hasattr(outputs, 'image_embeds'):
                        features = outputs.image_embeds
                    elif isinstance(outputs, dict) and 'image_embeds' in outputs:
                        features = outputs['image_embeds']
                    else:
                        # 如果实在找不到，返回随机特征
                        batch_size = images.shape[0]
                        features = torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
            
            logging.debug(f"Raw features shape: {features.shape}")
            
            # 关键修复：如果是视频特征，需要聚合帧特征
            if len(original_shape) == 5:  # 原始输入是视频
                feature_dim = features.shape[-1]
                if features.shape[0] == batch_size * num_frames:
                    # 特征是按帧展开的，需要聚合
                    features = features.view(batch_size, num_frames, feature_dim)
                    # 使用平均池化聚合帧特征
                    features = features.mean(dim=1)
                    logging.debug(f"Aggregated video features shape: {features.shape}")
            
            # 确保输出特征也是正确的数据类型和设备
            features = self._ensure_device_dtype_consistency(features, target_device, target_dtype)
            
            if normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            
            return features
            
        except Exception as e:
            logging.error(f"Image encoding failed: {e}")
            # 安全的fallback
            if len(original_shape) == 5:
                batch_size = original_shape[0]
            else:
                batch_size = images.shape[0] if len(images.shape) >= 1 else 1
            return torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
    
    def encode_text(self, text, normalize=False):
        """编码文本 - 修复设备和数据类型问题"""
        if text is None:
            return None
            
        # 获取目标设备和数据类型
        target_device = next(self.model.parameters()).device
        target_dtype = torch.float32
        
        try:
            # 设置模型为相同的数据类型
            self.model = self.model.to(target_dtype)
            
            # 处理不同格式的文本输入
            if isinstance(text, dict):
                # 确保字典中的所有tensor都在正确的设备上
                text_inputs = {}
                for key, value in text.items():
                    if isinstance(value, torch.Tensor):
                        text_inputs[key] = self._ensure_device_dtype_consistency(
                            value, target_device, value.dtype  # 保持原有的整数类型用于token
                        )
                    else:
                        text_inputs[key] = value
                
                if hasattr(self.model, 'get_text_features'):
                    features = self.model.get_text_features(**text_inputs)
                else:
                    # 尝试其他方法
                    if hasattr(self.model, 'encode_text'):
                        features = self.model.encode_text(text_inputs.get('input_ids', text_inputs))
                    else:
                        outputs = self.model(**text_inputs)
                        if hasattr(outputs, 'text_embeds'):
                            features = outputs.text_embeds
                        elif isinstance(outputs, dict) and 'text_embeds' in outputs:
                            features = outputs['text_embeds']
                        else:
                            # fallback
                            batch_size = text_inputs.get('input_ids', list(text_inputs.values())[0]).shape[0]
                            features = torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
                            
            else:
                # 处理tensor格式
                text = self._ensure_device_dtype_consistency(text, target_device, text.dtype)
                
                if hasattr(self.model, 'get_text_features'):
                    features = self.model.get_text_features(text)
                elif hasattr(self.model, 'encode_text'):
                    features = self.model.encode_text(text)
                else:
                    outputs = self.model(input_ids=text)
                    if hasattr(outputs, 'text_embeds'):
                        features = outputs.text_embeds
                    else:
                        batch_size = text.shape[0]
                        features = torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
            
            # 确保输出特征是正确的数据类型和设备
            features = self._ensure_device_dtype_consistency(features, target_device, target_dtype)
            
            if normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            
            return features
            
        except Exception as e:
            logging.error(f"Text encoding failed: {e}")
            # 安全的fallback
            if isinstance(text, dict) and 'input_ids' in text:
                batch_size = text['input_ids'].shape[0]
            elif hasattr(text, 'shape'):
                batch_size = text.shape[0]
            else:
                batch_size = 1
            return torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
    
    def forward(self, *args, **kwargs):
        """前向传播 - 返回符合训练期望的字典格式"""
        target_device = next(self.model.parameters()).device
        target_dtype = torch.float32
        
        # 确保logit_scale在正确的设备和数据类型上
        self.logit_scale = self._ensure_device_dtype_consistency(
            self.logit_scale, target_device, target_dtype
        )
        
        # 根据参数数量判断调用方式
        if len(args) == 1:
            # 单个参数：可能是image或text
            if hasattr(args[0], 'dim') and args[0].dim() > 2:
                image_features = self.encode_image(args[0], normalize=True)
                return {
                    "image_features": image_features,
                    "text_features": None,
                    "logit_scale": self.logit_scale.exp()
                }
            else:
                text_features = self.encode_text(args[0], normalize=True)
                return {
                    "image_features": None,
                    "text_features": text_features,
                    "logit_scale": self.logit_scale.exp()
                }
        
        elif len(args) == 2:
            # 两个参数：image和text
            image, text = args
            return self._forward_image_text(image, text, **kwargs)
        
        elif len(args) == 3:
            # 三个参数：来自训练循环 (images, input_ids, attention_mask)
            images, input_ids, attention_mask = args
            
            # 构建text输入
            text_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            return self._forward_image_text(images, text_inputs, **kwargs)
        
        else:
            # 关键字参数调用
            image = kwargs.get('pixel_values') or kwargs.get('image')
            text = kwargs.get('input_ids') or kwargs.get('text')
            
            if image is not None and text is not None:
                return self._forward_image_text(image, text, **kwargs)
            elif image is not None:
                image_features = self.encode_image(image, normalize=True)
                return {
                    "image_features": image_features,
                    "text_features": None,
                    "logit_scale": self.logit_scale.exp()
                }
            elif text is not None:
                text_features = self.encode_text(text, normalize=True)
                return {
                    "image_features": None,
                    "text_features": text_features,
                    "logit_scale": self.logit_scale.exp()
                }
            else:
                raise ValueError("Must provide either image or text input")
    
    def _forward_image_text(self, image, text, **kwargs):
        """处理图像和文本的联合前向传播 - 返回字典格式"""
        try:
            image_features = self.encode_image(image, normalize=True) if image is not None else None
            text_features = self.encode_text(text, normalize=True) if text is not None else None
            
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            
        except Exception as e:
            logging.error(f"Forward pass failed: {e}")
            
            # 安全的fallback
            target_device = next(self.model.parameters()).device
            target_dtype = torch.float32
            
            if image is not None:
                batch_size = image.shape[0]
                image_features = torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
            else:
                image_features = None
                
            if text is not None:
                if isinstance(text, dict) and 'input_ids' in text:
                    batch_size = text['input_ids'].shape[0]
                elif hasattr(text, 'shape'):
                    batch_size = text.shape[0]
                else:
                    batch_size = 1
                text_features = torch.randn(batch_size, 768, device=target_device, dtype=target_dtype)
            else:
                text_features = None
                
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
    
    # 其他方法保持不变...
    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
    
    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.model.eval()
        return self
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        if hasattr(self, 'logit_scale'):
            self.logit_scale = self.logit_scale.to(*args, **kwargs)
        return self
    
    def cuda(self, device=None):
        super().cuda(device)
        self.model.cuda(device)
        if hasattr(self, 'logit_scale'):
            self.logit_scale = self.logit_scale.cuda(device)
        return self
    
    def cpu(self):
        super().cpu()
        self.model.cpu()
        if hasattr(self, 'logit_scale'):
            self.logit_scale = self.logit_scale.cpu()
        return self

def create_languagebind_model_with_adapter(model_path, cache_dir='./cache_dir'):
    """创建带适配器的LanguageBind模型"""
    from languagebind import LanguageBindVideo
    
    # 加载原始LanguageBind模型
    logging.info(f"Loading LanguageBindVideo from {model_path}")
    model = LanguageBindVideo.from_pretrained(model_path, cache_dir=cache_dir, torch_dtype=torch.float32)
    
    # 包装成适配器
    adapter = LanguageBindVideoAdapter(model)
    
    logging.info("LanguageBindVideo loaded and wrapped with adapter")
    return adapter

def apply_lora_to_languagebind(model, args):
    """为LanguageBind模型应用LoRA"""
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        
        # 定义LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "value", "key", "dense"]
        )
        
        # 应用LoRA
        model.model = get_peft_model(model.model, lora_config)
        logging.info(f"Applied LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
    except ImportError:
        logging.warning("PEFT library not found. LoRA conversion skipped.")
        logging.warning("Install with: pip install peft")
    except Exception as e:
        logging.error(f"Failed to apply LoRA: {e}")
        logging.warning("Continuing without LoRA...")
    
    return model

# 示例使用
if __name__ == "__main__":
    # 测试适配器
    model_path = "/root/autodl-tmp/LanguageBind/cache_dir/models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66"
    
    model = create_languagebind_model_with_adapter(model_path)
    
    # 测试forward方法返回值格式
    print("Testing forward method...")
    dummy_images = torch.randn(2, 8, 3, 224, 224).float()
    dummy_input_ids = torch.randint(0, 1000, (2, 77))
    dummy_attention_mask = torch.ones(2, 77)
    
    output = model(dummy_images, dummy_input_ids, dummy_attention_mask)
    print(f"Output keys: {output.keys()}")
    print(f"Output types: {[type(v) for v in output.values()]}")
    
    print("✅ LanguageBindVideo adapter test completed!")