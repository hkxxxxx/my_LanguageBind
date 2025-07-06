import torch
import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict
from more_itertools import chunked
from tqdm.auto import tqdm
from autoattack import AutoAttack
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import json
import random
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== 参数配置 (对齐CLIP RobustBench) ==========
parser = argparse.ArgumentParser(description="MSR-VTT对抗攻击评估 - 全数据集版本")

# 模型配置
parser.add_argument('--cache_dir', type=str, default='./cache_dir', help='模型缓存目录')
parser.add_argument('--clip_type', type=str, default='LanguageBind_Video_FT', help='LanguageBind模型类型')

# 数据集配置  
parser.add_argument('--csv_file', type=str, default='/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_JSFUSION_test.csv', help='MSR-VTT测试集CSV文件')
parser.add_argument('--video_base_path', type=str, default='/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/MSRVTT_Videos/video', help='视频文件基础路径')

# 攻击配置 (对齐RobustBench参数)
parser.add_argument('--norm', type=str, default='linf', help='攻击范数: linf, l2')
parser.add_argument('--eps', type=float, default=4., help='攻击强度 (0-255范围)')
parser.add_argument('--alpha', type=float, default=2., help='APGD alpha参数')
parser.add_argument('--n_attack_samples', type=int, default=50, help='每种攻击方法的测试样本数，设为-1则使用全数据集')
parser.add_argument('--batch_size', type=int, default=16, help='特征提取批处理大小')
parser.add_argument('--attack_batch_size', type=int, default=4, help='攻击时的批处理大小')
parser.add_argument('--full_dataset_attack', type=bool, default=False, help='是否在全数据集上进行攻击')

# 攻击方法选择
parser.add_argument('--blackbox_only', type=bool, default=False, help='仅运行黑盒攻击')
parser.add_argument('--attacks_to_run', type=str, nargs='+', default=None, 
                   help='指定运行的攻击方法，默认为所有白盒攻击')

# 实验配置
parser.add_argument('--experiment_name', type=str, default='msrvtt_full_attack_eval', help='实验名称')
parser.add_argument('--save_results', type=bool, default=True, help='保存详细结果')
parser.add_argument('--verbose', type=bool, default=True, help='详细输出')

def compute_metrics(sim_matrix):
    """
    计算检索指标 - 对齐原始检索脚本
    sim_matrix: [num_queries, num_targets]
    """
    # 按相似度降序排序，获得排名
    ranks = np.argsort(-sim_matrix, axis=1)  # [num_queries, num_targets]
    
    # 计算每个查询的正确答案排名 (假设第i个查询对应第i个目标)
    correct_ranks = []
    for i in range(sim_matrix.shape[0]):
        # 找到正确答案(第i个目标)在排序中的位置
        rank = np.where(ranks[i] == i)[0][0]
        correct_ranks.append(rank)
    
    correct_ranks = np.array(correct_ranks)
    
    # 计算指标
    metrics = {}
    metrics['R1'] = float(np.sum(correct_ranks == 0)) * 100 / len(correct_ranks)
    metrics['R5'] = float(np.sum(correct_ranks < 5)) * 100 / len(correct_ranks)
    metrics['R10'] = float(np.sum(correct_ranks < 10)) * 100 / len(correct_ranks)
    metrics['MR'] = np.median(correct_ranks) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(correct_ranks) + 1
    
    return metrics

class LanguageBindAttackWrapper(torch.nn.Module):
    """
    用于AutoAttack的模型包装器 - 对齐检索脚本的相似度计算
    """
    def __init__(self, model, text_embeddings, args, logit_scale=100.0):
        super().__init__()
        self.model = model 
        self.text_embeddings = text_embeddings.float()  # [num_texts, embed_dim]
        self.args = args
        self.logit_scale = logit_scale  # 模拟原脚本中的logit_scale
        
    def forward(self, video_tensor):
        """
        前向传播 - 对齐检索脚本的相似度计算
        video_tensor: [batch_size, C, T, H, W]
        返回: [batch_size, num_texts] 相似度logits
        """
        video_tensor = video_tensor.float()
        
        # 获取视频embedding
        video_inputs = {'video': {'pixel_values': video_tensor}}
        video_embeddings = self.model(video_inputs)['video']  # [batch_size, embed_dim]
        
        # 对齐原脚本: logit_scale * video @ text.T
        sim_logits = self.logit_scale * video_embeddings @ self.text_embeddings.T
        
        return sim_logits

def load_msrvtt_test_data(csv_file, video_base_path):
    """加载MSR-VTT测试数据 - 对齐检索脚本"""
    df = pd.read_csv(csv_file)
    
    # 检查视频文件是否存在
    valid_indices = []
    for idx, row in df.iterrows():
        video_path = os.path.join(video_base_path, f"{row['video_id']}.mp4")
        if os.path.exists(video_path):
            valid_indices.append(idx)
    
    if len(valid_indices) == 0:
        raise FileNotFoundError(f"在 {video_base_path} 中没有找到任何视频文件")
    
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    language_data = df_valid['sentence'].values.tolist()
    video_data = df_valid['video_id'].apply(
        lambda x: os.path.join(video_base_path, f'{x}.mp4')
    ).values.tolist()
    
    logging.info(f"成功加载 {len(df_valid)} 个有效的视频-文本对")
    return language_data, video_data, df_valid

def extract_all_features(model, tokenizer, language_data, video_data, modality_transform, batch_size, device):
    """
    提取所有特征 - 对齐检索脚本的特征提取流程
    """
    logging.info("开始提取所有特征...")
    model.eval()
    
    # 存储所有特征
    batch_sequence_output_list = []
    batch_visual_output_list = []
    
    with torch.no_grad():
        # 1. 提取文本特征
        logging.info("提取文本特征...")
        text_batches = list(chunked(language_data, batch_size))
        for batch_texts in tqdm(text_batches, desc="处理文本批次"):
            # 对齐检索脚本: model.encode_text()
            text_tokens = tokenizer(batch_texts, max_length=77, padding='max_length', 
                                  truncation=True, return_tensors='pt')
            text_tokens = to_device(text_tokens, device)
            
            # 使用LanguageBind的文本编码
            text_inputs = {'language': text_tokens}
            text_embeddings = model(text_inputs)['language']  # [batch_size, embed_dim]
            
            batch_sequence_output_list.append(text_embeddings.cpu())
        
        # 2. 提取视频特征 
        logging.info("提取视频特征...")
        video_batches = list(chunked(video_data, batch_size))
        for batch_videos in tqdm(video_batches, desc="处理视频批次"):
            # 对齐检索脚本: model.encode_image() / encode_video()
            try:
                video_tensor_dict = modality_transform['video'](batch_videos)
                video_tensor_dict = to_device(video_tensor_dict, device)
                
                video_inputs = {'video': video_tensor_dict}
                video_embeddings = model(video_inputs)['video']  # [batch_size, embed_dim]
                
                batch_visual_output_list.append(video_embeddings.cpu())
                
            except Exception as e:
                logging.warning(f"跳过视频批次: {str(e)}")
                # 创建零向量作为占位符
                dummy_embeddings = torch.zeros(len(batch_videos), 768)  # 假设768维
                batch_visual_output_list.append(dummy_embeddings)
                continue
    
    # 3. 合并所有特征
    all_text_embeddings = torch.cat(batch_sequence_output_list, dim=0)  # [num_texts, embed_dim]
    all_video_embeddings = torch.cat(batch_visual_output_list, dim=0)   # [num_videos, embed_dim]
    
    logging.info(f"文本特征形状: {all_text_embeddings.shape}")
    logging.info(f"视频特征形状: {all_video_embeddings.shape}")
    
    return all_text_embeddings.numpy(), all_video_embeddings.numpy()

def compute_full_similarity_matrix(text_embeddings, video_embeddings, logit_scale=100.0):
    """
    计算完整相似度矩阵 - 对齐检索脚本
    """
    logging.info("计算完整相似度矩阵...")
    
    # 对齐检索脚本: logit_scale * text @ video.T (注意这里是text在前)
    sim_matrix = logit_scale * text_embeddings @ video_embeddings.T  # [num_texts, num_videos]
    
    logging.info(f"相似度矩阵形状: {sim_matrix.shape}")
    return sim_matrix

def evaluate_baseline_retrieval(sim_matrix):
    """
    评估基线检索性能 - 对齐检索脚本
    """
    logging.info("评估基线检索性能...")
    
    # Text-to-Video检索 (每行是一个文本查询)
    tv_metrics = compute_metrics(sim_matrix)
    
    # Video-to-Text检索 (转置后每行是一个视频查询)
    vt_metrics = compute_metrics(sim_matrix.T)
    
    # 对齐检索脚本的日志格式
    logging.info("MSRVTT Text-to-Video:")
    logging.info(f'\t>>>  R@1: {tv_metrics["R1"]:.1f} - R@5: {tv_metrics["R5"]:.1f} - R@10: {tv_metrics["R10"]:.1f} - Median R: {tv_metrics["MR"]:.1f} - Mean R: {tv_metrics["MeanR"]:.1f}')
    
    logging.info("MSRVTT Video-to-Text:")
    logging.info(f'\t>>>  V2T$R@1: {vt_metrics["R1"]:.1f} - V2T$R@5: {vt_metrics["R5"]:.1f} - V2T$R@10: {vt_metrics["R10"]:.1f} - V2T$Median R: {vt_metrics["MR"]:.1f} - V2T$Mean R: {vt_metrics["MeanR"]:.1f}')
    
    return tv_metrics, vt_metrics

def evaluate_attack_on_subset(attack_method, model, modality_transform, video_data, 
                             text_embeddings, baseline_sim_matrix, args):
    """
    评估攻击效果 - 正确的全集检索评估方法
    关键思想：攻击视频，但在全部1000个文本中进行检索
    """
    logging.info(f"\n开始评估攻击方法: {attack_method}")
    
    device = next(model.parameters()).device
    total_samples = len(video_data)
    
    # 确定攻击样本范围
    if args.full_dataset_attack or args.n_attack_samples == -1:
        attack_indices = list(range(total_samples))
        logging.info(f"使用全数据集进行攻击 ({total_samples} 个样本)")
        logging.warning("⚠️  全数据集攻击可能需要数小时时间")
    else:
        attack_indices = random.sample(range(total_samples), 
                                     min(args.n_attack_samples, total_samples))
        logging.info(f"使用子集进行攻击 ({len(attack_indices)} 个样本)")
    
    # 为攻击创建包装模型 - 注意这里使用全部文本embeddings
    text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
    wrapped_model = LanguageBindAttackWrapper(model, text_embeddings_tensor, args)
    
    # 存储所有攻击结果用于最终的全集检索评估
    all_original_video_embeddings = []
    all_attacked_video_embeddings = []
    all_attack_labels = []  # 对应的真实文本索引
    
    total_attack_time = 0
    total_valid_samples = 0
    
    # 按attack_batch_size分批处理
    batch_size = args.attack_batch_size
    num_batches = (len(attack_indices) + batch_size - 1) // batch_size
    
    logging.info(f"将分 {num_batches} 个批次进行攻击，每批最多 {batch_size} 个样本")
    logging.info(f"每个攻击视频将在全部 {len(text_embeddings)} 个文本中进行检索")
    
    for batch_idx in tqdm(range(num_batches), desc=f"攻击批次 {attack_method}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(attack_indices))
        batch_indices = attack_indices[start_idx:end_idx]
        
        # 准备当前批次的数据
        batch_videos = []
        batch_labels = []
        
        for attack_idx in batch_indices:
            try:
                video_path = video_data[attack_idx]
                if not os.path.exists(video_path):
                    continue
                    
                # 预处理视频
                video_tensor_dict = modality_transform['video']([video_path])
                video_tensor_dict = to_device(video_tensor_dict, device)
                video_tensor = video_tensor_dict['pixel_values'].float()
                
                batch_videos.append(video_tensor.squeeze(0))
                batch_labels.append(attack_idx)  # 真实标签是对应的文本索引
                
            except Exception as e:
                if args.verbose and batch_idx < 5:  # 只在前几个批次输出错误
                    logging.warning(f"跳过样本 {attack_idx}: {str(e)}")
                continue
        
        if len(batch_videos) == 0:
            continue
        
        # 转换为tensor
        x_batch = torch.stack(batch_videos).to(device)
        y_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        # 获取原始视频embeddings (用于最终检索评估)
        with torch.no_grad():
            original_video_inputs = {'video': {'pixel_values': x_batch}}
            original_video_embeddings = model(original_video_inputs)['video']  # [batch_size, embed_dim]
        
        # 运行攻击
        adversary = AutoAttack(
            wrapped_model, 
            norm=args.norm.replace('l', 'L'),
            eps=args.eps,
            version='custom',
            attacks_to_run=[attack_method],
            alpha=args.alpha,
            verbose=False
        )
        
        start_time = time.time()
        x_adv, y_adv = adversary.run_standard_evaluation(
            x_batch, y_batch, bs=len(x_batch), return_labels=True
        )
        batch_attack_time = time.time() - start_time
        
        # 获取攻击后的视频embeddings
        with torch.no_grad():
            adv_video_inputs = {'video': {'pixel_values': x_adv}}
            adv_video_embeddings = model(adv_video_inputs)['video']  # [batch_size, embed_dim]
        
        # 保存embeddings用于最终的全集检索评估
        all_original_video_embeddings.append(original_video_embeddings.cpu().numpy())
        all_attacked_video_embeddings.append(adv_video_embeddings.cpu().numpy())
        all_attack_labels.extend(batch_labels)
        
        total_attack_time += batch_attack_time
        total_valid_samples += len(batch_labels)
        
        # 清理当前批次的内存
        del x_batch, y_batch, x_adv, y_adv, original_video_embeddings, adv_video_embeddings, batch_videos
        torch.cuda.empty_cache()
        
        # 输出进度信息
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            logging.info(f"  完成: {batch_idx + 1}/{num_batches} 批次, "
                        f"累计样本: {total_valid_samples}, "
                        f"累计耗时: {total_attack_time:.1f}s")
    
    if total_valid_samples == 0:
        logging.error(f"{attack_method}: 没有有效的攻击样本")
        return None
    
    # ========== 关键：在全集上进行检索评估 ==========
    logging.info(f"开始在全集上评估 {total_valid_samples} 个攻击样本的检索性能...")
    
    # 合并所有攻击样本的embeddings
    all_original_embeddings = np.concatenate(all_original_video_embeddings, axis=0)  # [total_valid_samples, embed_dim]
    all_attacked_embeddings = np.concatenate(all_attacked_video_embeddings, axis=0)   # [total_valid_samples, embed_dim]
    
    logging.info(f"原始视频embeddings形状: {all_original_embeddings.shape}")
    logging.info(f"攻击后视频embeddings形状: {all_attacked_embeddings.shape}")
    logging.info(f"文本embeddings形状: {text_embeddings.shape}")
    
    # 计算在全集上的相似度矩阵
    # 每个攻击的视频与全部1000个文本的相似度
    original_sim_matrix = 100.0 * all_original_embeddings @ text_embeddings.T  # [n_attack, 1000]
    attacked_sim_matrix = 100.0 * all_attacked_embeddings @ text_embeddings.T   # [n_attack, 1000]
    
    logging.info(f"原始相似度矩阵形状: {original_sim_matrix.shape}")
    logging.info(f"攻击后相似度矩阵形状: {attacked_sim_matrix.shape}")
    
    # 计算每个攻击样本的检索性能
    original_ranks = []
    attacked_ranks = []
    
    for i, true_label in enumerate(all_attack_labels):
        # 原始视频的检索排名
        original_similarities = original_sim_matrix[i, :]  # 与所有1000个文本的相似度
        original_sorted_indices = np.argsort(-original_similarities)  # 按相似度降序排序
        original_rank = np.where(original_sorted_indices == true_label)[0][0]  # 找到真实标签的排名
        original_ranks.append(original_rank)
        
        # 攻击后视频的检索排名
        attacked_similarities = attacked_sim_matrix[i, :]
        attacked_sorted_indices = np.argsort(-attacked_similarities)
        attacked_rank = np.where(attacked_sorted_indices == true_label)[0][0]
        attacked_ranks.append(attacked_rank)
    
    original_ranks = np.array(original_ranks)
    attacked_ranks = np.array(attacked_ranks)
    
    # 计算检索指标
    def compute_retrieval_metrics(ranks):
        metrics = {}
        metrics['R1'] = float(np.sum(ranks == 0)) * 100 / len(ranks)
        metrics['R5'] = float(np.sum(ranks < 5)) * 100 / len(ranks)
        metrics['R10'] = float(np.sum(ranks < 10)) * 100 / len(ranks)
        metrics['MR'] = np.median(ranks) + 1
        metrics['MeanR'] = np.mean(ranks) + 1
        return metrics
    
    original_metrics = compute_retrieval_metrics(original_ranks)
    attacked_metrics = compute_retrieval_metrics(attacked_ranks)
    
    # 计算攻击成功率 (基于排名变化)
    rank_degradation = np.mean(attacked_ranks - original_ranks)  # 平均排名下降
    successful_attacks = np.sum(attacked_ranks > original_ranks)  # 排名变差的样本数
    attack_success_rate = successful_attacks / len(original_ranks)
    
    # 计算准确率指标 (R@1)
    original_accuracy = original_metrics['R1']
    robust_accuracy = attacked_metrics['R1'] 
    accuracy_drop = original_accuracy - robust_accuracy
    
    # 对比基线性能 (从完整baseline中提取对应样本的性能)
    baseline_ranks = []
    for true_label in all_attack_labels:
        baseline_similarities = baseline_sim_matrix[true_label, :]  # 从基线相似度矩阵中提取
        baseline_sorted_indices = np.argsort(-baseline_similarities)
        baseline_rank = np.where(baseline_sorted_indices == true_label)[0][0]
        baseline_ranks.append(baseline_rank)
    
    baseline_ranks = np.array(baseline_ranks)
    baseline_metrics = compute_retrieval_metrics(baseline_ranks)
    
    # 整理结果
    results = {
        'attack_method': attack_method,
        'dataset_scope': '全数据集' if (args.full_dataset_attack or args.n_attack_samples == -1) else '子集',
        'total_samples': total_valid_samples,
        'attack_time': total_attack_time,
        'avg_time_per_sample': total_attack_time / total_valid_samples,
        
        # 基线性能 (这些样本在基线评估中的表现)
        'baseline_r1': baseline_metrics['R1'],
        'baseline_r5': baseline_metrics['R5'],
        'baseline_r10': baseline_metrics['R10'],
        
        # 攻击前性能 (应该与基线接近)
        'original_r1': original_metrics['R1'],
        'original_r5': original_metrics['R5'],
        'original_r10': original_metrics['R10'],
        'original_mean_rank': original_metrics['MeanR'],
        
        # 攻击后性能
        'attacked_r1': attacked_metrics['R1'],
        'attacked_r5': attacked_metrics['R5'],
        'attacked_r10': attacked_metrics['R10'],
        'attacked_mean_rank': attacked_metrics['MeanR'],
        
        # 攻击效果指标
        'r1_drop': original_metrics['R1'] - attacked_metrics['R1'],
        'r5_drop': original_metrics['R5'] - attacked_metrics['R5'],
        'r10_drop': original_metrics['R10'] - attacked_metrics['R10'],
        'mean_rank_increase': attacked_metrics['MeanR'] - original_metrics['MeanR'],
        'attack_success_rate': attack_success_rate,
        'mean_rank_degradation': rank_degradation
    }
    
    # 输出详细结果
    logging.info(f"\n{attack_method} 全集检索评估结果:")
    logging.info(f"  攻击范围: {results['dataset_scope']} ({total_valid_samples} 个样本)")
    logging.info(f"  总攻击耗时: {total_attack_time:.1f}秒")
    logging.info(f"  平均每样本: {total_attack_time / total_valid_samples:.2f}秒")
    logging.info(f"")
    logging.info(f"  基线检索性能:")
    logging.info(f"    R@1: {baseline_metrics['R1']:.2f}% | R@5: {baseline_metrics['R5']:.2f}% | R@10: {baseline_metrics['R10']:.2f}%")
    logging.info(f"")  
    logging.info(f"  攻击前检索性能:")
    logging.info(f"    R@1: {original_metrics['R1']:.2f}% | R@5: {original_metrics['R5']:.2f}% | R@10: {original_metrics['R10']:.2f}%")
    logging.info(f"")
    logging.info(f"  攻击后检索性能:")
    logging.info(f"    R@1: {attacked_metrics['R1']:.2f}% | R@5: {attacked_metrics['R5']:.2f}% | R@10: {attacked_metrics['R10']:.2f}%")
    logging.info(f"")
    logging.info(f"  攻击效果:")
    logging.info(f"    R@1下降: {original_metrics['R1'] - attacked_metrics['R1']:.2f}%")
    logging.info(f"    R@5下降: {original_metrics['R5'] - attacked_metrics['R5']:.2f}%")
    logging.info(f"    R@10下降: {original_metrics['R10'] - attacked_metrics['R10']:.2f}%")
    logging.info(f"    平均排名恶化: {rank_degradation:.2f}")
    logging.info(f"    攻击成功率: {attack_success_rate:.3f}")
    
    return results

def main():
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 转换参数范围 (对齐RobustBench)
    args.eps = args.eps / 255
    args.alpha = args.alpha / 255
    
    logging.info(f"实验配置:\n{'-' * 50}")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    logging.info(f"{'-' * 50}")
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 确定攻击方法
    if args.attacks_to_run is not None:
        attacks_to_run = args.attacks_to_run
    elif args.blackbox_only:
        attacks_to_run = ['square']
    else:
        # attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'apgd-t', 'fab-t']
        attacks_to_run = ['apgd-t']
    logging.info(f'计划运行的攻击方法: {attacks_to_run}')
    
    # 加载LanguageBind模型
    logging.info("加载LanguageBind模型...")
    clip_type = {'video': args.clip_type}
    model = LanguageBind(clip_type=clip_type, cache_dir=args.cache_dir).to(device)
    model.eval()
    
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        'lb203/LanguageBind_Image', 
        cache_dir=os.path.join(args.cache_dir, 'tokenizer_cache_dir')
    )
    
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}
    
    # 加载测试数据
    language_data, video_data, df_valid = load_msrvtt_test_data(args.csv_file, args.video_base_path)
    
    # ========== 第一阶段: 提取所有特征并计算基线性能 ==========
    text_embeddings, video_embeddings = extract_all_features(
        model, tokenizer, language_data, video_data, 
        modality_transform, args.batch_size, device
    )
    
    # 计算完整相似度矩阵
    sim_matrix = compute_full_similarity_matrix(text_embeddings, video_embeddings)
    
    # 评估基线检索性能
    baseline_tv_metrics, baseline_vt_metrics = evaluate_baseline_retrieval(sim_matrix)
    
    # ========== 第二阶段: 对每种攻击方法进行评估 ==========
    all_attack_results = []
    
    for attack_method in attacks_to_run:
        try:
            result = evaluate_attack_on_subset(
                attack_method, model, modality_transform, video_data,
                text_embeddings, sim_matrix, args
            )
            if result:
                all_attack_results.append(result)
        except Exception as e:
            logging.error(f"攻击方法 {attack_method} 执行失败: {str(e)}")
            continue
    
    # ========== 第三阶段: 结果汇总和分析 ==========
    logging.info("\n" + "="*100)
    logging.info("攻击评估结果汇总:")
    logging.info("="*100)
    
    if all_attack_results:
        # 按R@1下降排序 (主要攻击效果指标)
        all_attack_results.sort(key=lambda x: x.get('r1_drop', 0), reverse=True)
        
        logging.info(f"{'攻击方法':<12} {'样本数':<6} {'基线R@1':<8} {'攻击前R@1':<10} {'攻击后R@1':<10} {'R@1下降':<8} {'R@5下降':<8} {'成功率':<8}")
        logging.info("-"*100)
        
        for result in all_attack_results:
            logging.info(f"{result['attack_method']:<12} "
                        f"{result['total_samples']:<6} "
                        f"{result.get('baseline_r1', 0):.1f}%     "
                        f"{result.get('original_r1', 0):.1f}%       "
                        f"{result.get('attacked_r1', 0):.1f}%       "
                        f"{result.get('r1_drop', 0):.1f}%    "
                        f"{result.get('r5_drop', 0):.1f}%    "
                        f"{result.get('attack_success_rate', 0):.3f}")
        
        # 找出最有效的攻击方法
        best_method = all_attack_results[0]
        logging.info(f"\n🏆 最有效的攻击方法: {best_method['attack_method']}")
        logging.info(f"   R@1下降: {best_method.get('r1_drop', 0):.2f}%")
        logging.info(f"   R@5下降: {best_method.get('r5_drop', 0):.2f}%")
        logging.info(f"   攻击成功率: {best_method.get('attack_success_rate', 0):.3f}")
        logging.info(f"   平均排名恶化: {best_method.get('mean_rank_degradation', 0):.1f}")
    
    # ========== 保存结果 ==========
    if args.save_results:
        results_data = {
            'experiment_config': vars(args),
            'baseline_performance': {
                'text_to_video': baseline_tv_metrics,
                'video_to_text': baseline_vt_metrics
            },
            'attack_results': all_attack_results,
            'summary': {
                'best_attack': best_method['attack_method'] if all_attack_results else None,
                'max_r1_drop': best_method.get('r1_drop', 0) if all_attack_results else 0,
                'max_r5_drop': best_method.get('r5_drop', 0) if all_attack_results else 0,
                'max_success_rate': best_method.get('attack_success_rate', 0) if all_attack_results else 0
            } if all_attack_results else {}
        }
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f'{args.experiment_name}_{args.norm}_eps{int(args.eps*255)}_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logging.info(f"\n详细结果已保存到: {results_file}")

if __name__ == "__main__":
    main()