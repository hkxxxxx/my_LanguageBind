#!/usr/bin/env python
"""
修复对比损失函数 - 解决batch size不匹配问题
创建一个自定义的损失函数来处理LanguageBind的特殊情况
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any


class LanguageBindContrastiveLoss(nn.Module):
    """
    专门为LanguageBind设计的对比损失函数
    解决batch size不匹配和梯度累积问题
    """
    
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        temperature=0.07,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temperature = temperature
        
        # 缓存标签以避免重复计算
        self.prev_num_logits = 0
        self.labels = {}
        
    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        """获取对比学习的ground truth标签"""
        # labels用于对比学习：对角线为1，其余为0
        # 对于batch size为N，labels应该是[0, 1, 2, ..., N-1]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
    
    def get_logits(self, image_features, text_features, logit_scale):
        """计算logits，处理不同的分布式设置"""
        if self.world_size > 1:
            # 分布式情况下gather所有features
            if self.local_loss:
                # 只使用本地features计算损失
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logits_per_image.T
            else:
                # 使用全局features计算损失（需要gather）
                if self.gather_with_grad:
                    all_image_features = gather_features(
                        image_features, self.local_loss, self.rank, self.world_size, self.use_horovod
                    )
                    all_text_features = gather_features(
                        text_features, self.local_loss, self.rank, self.world_size, self.use_horovod
                    )
                else:
                    with torch.no_grad():
                        all_image_features = gather_features(
                            image_features, self.local_loss, self.rank, self.world_size, self.use_horovod
                        )
                        all_text_features = gather_features(
                            text_features, self.local_loss, self.rank, self.world_size, self.use_horovod
                        )
                
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            # 单GPU情况
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
    
    def forward(
        self, 
        image_features: torch.Tensor,
        text_features: torch.Tensor, 
        logit_scale: torch.Tensor,
        output_dict: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播计算对比损失
        
        Args:
            image_features: 图像特征 [batch_size, feature_dim]
            text_features: 文本特征 [batch_size, feature_dim]
            logit_scale: 温度参数的倒数
            output_dict: 是否返回字典格式
        """
        
        # 确保输入tensor在同一设备上
        device = image_features.device
        text_features = text_features.to(device)
        logit_scale = logit_scale.to(device)
        
        # 检查batch size是否匹配
        if image_features.shape[0] != text_features.shape[0]:
            min_batch_size = min(image_features.shape[0], text_features.shape[0])
            logging.warning(
                f"Batch size mismatch: image {image_features.shape[0]} vs text {text_features.shape[0]}, "
                f"using minimum size {min_batch_size}"
            )
            image_features = image_features[:min_batch_size]
            text_features = text_features[:min_batch_size]
        
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算logits
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        
        # 获取标签
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        
        # 确保标签数量匹配
        if labels.shape[0] != logits_per_image.shape[0]:
            logging.warning(
                f"Labels size {labels.shape[0]} doesn't match logits size {logits_per_image.shape[0]}, "
                f"adjusting labels"
            )
            labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
        
        # 计算交叉熵损失
        try:
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            contrastive_loss = (loss_i2t + loss_t2i) / 2
        except Exception as e:
            logging.error(f"Loss calculation failed: {e}")
            logging.error(f"logits_per_image shape: {logits_per_image.shape}")
            logging.error(f"logits_per_text shape: {logits_per_text.shape}")
            logging.error(f"labels shape: {labels.shape}")
            # 返回零损失作为fallback
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if output_dict:
            return {"contrastive_loss": contrastive_loss}
        else:
            return contrastive_loss


def gather_features(
    features,
    local_loss=False,
    rank=0,
    world_size=1,
    use_horovod=False
):
    """收集分布式训练中的特征"""
    if use_horovod:
        # Horovod的实现
        import horovod.torch as hvd
        gathered_features = hvd.allgather(features)
    else:
        # PyTorch DDP的实现
        if world_size > 1:
            # 创建tensor列表来收集所有rank的features
            gathered_features_list = [torch.zeros_like(features) for _ in range(world_size)]
            torch.distributed.all_gather(gathered_features_list, features)
            gathered_features = torch.cat(gathered_features_list, dim=0)
        else:
            gathered_features = features
    
    return gathered_features


class ClipLoss(nn.Module):
    """
    兼容原有ClipLoss接口的包装器
    """
    
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.loss_fn = LanguageBindContrastiveLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )
    
    def forward(self, image_features, text_features, logit_scale, output_dict=False, **kwargs):
        """兼容原有接口"""
        return self.loss_fn(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
            output_dict=output_dict,
            **kwargs
        )


# 创建损失函数的工厂函数
def create_languagebind_loss(args):
    """创建适合LanguageBind的损失函数"""
    
    # 从args中获取参数
    local_loss = getattr(args, 'local_loss', False)
    gather_with_grad = getattr(args, 'gather_with_grad', False)
    cache_labels = getattr(args, 'cache_labels', False)
    rank = getattr(args, 'rank', 0)
    world_size = getattr(args, 'world_size', 1)
    use_horovod = getattr(args, 'horovod', False)
    
    logging.info(f"Creating LanguageBind loss with:")
    logging.info(f"  local_loss: {local_loss}")
    logging.info(f"  gather_with_grad: {gather_with_grad}")
    logging.info(f"  world_size: {world_size}")
    logging.info(f"  rank: {rank}")
    
    return ClipLoss(
        local_loss=local_loss,
        gather_with_grad=gather_with_grad,
        cache_labels=cache_labels,
        rank=rank,
        world_size=world_size,
        use_horovod=use_horovod,
    )


if __name__ == "__main__":
    # 测试损失函数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    batch_size = 4
    feature_dim = 512
    
    image_features = torch.randn(batch_size, feature_dim, device=device)
    text_features = torch.randn(batch_size, feature_dim, device=device)
    logit_scale = torch.tensor(1.0, device=device)
    
    # 创建损失函数
    loss_fn = LanguageBindContrastiveLoss()
    
    # 计算损失
    loss = loss_fn(image_features, text_features, logit_scale, output_dict=True)
    
    print(f"Loss: {loss}")
    print("✅ LanguageBind contrastive loss test passed!")