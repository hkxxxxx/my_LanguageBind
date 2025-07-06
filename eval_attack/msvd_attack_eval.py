import torch
import pandas as pd
import numpy as np
import os
import pickle
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

# ========== 参数配置 ==========
parser = argparse.ArgumentParser(description="MSVD对抗攻击评估")

# 模型配置
parser.add_argument('--cache_dir', type=str, default='./cache_dir', help='模型缓存目录')
parser.add_argument('--clip_type', type=str, default='LanguageBind_Video_FT', help='LanguageBind模型类型')

# MSVD数据集配置
parser.add_argument('--msvd_root', type=str, default='./', help='MSVD数据集根目录')
parser.add_argument('--video_dir', type=str, default='./datasets/MSVD_Zero_Shot_QA/videos', help='视频文件目录')
parser.add_argument('--descriptions_file', type=str, default='./datasets/MSVD_Zero_Shot_QA/AllVideoDescriptions.txt', help='视频描述文件路径')
parser.add_argument('--test_list', type=str, default='./datasets/MSVD_Zero_Shot_QA/test_list.txt', help='测试视频列表')

# 攻击配置
parser.add_argument('--norm', type=str, default='linf', help='攻击范数: linf, l2')
parser.add_argument('--eps', type=float, default=4., help='攻击强度 (0-255范围)')
parser.add_argument('--alpha', type=float, default=2., help='APGD alpha参数')
parser.add_argument('--n_attack_samples', type=int, default=50, help='攻击样本数量')
parser.add_argument('--batch_size', type=int, default=16, help='特征提取批处理大小')
parser.add_argument('--attack_batch_size', type=int, default=4, help='攻击批处理大小')

# 攻击方法选择
parser.add_argument('--attacks_to_run', type=str, nargs='+', default=None,
                   help='指定攻击方法，默认为所有白盒攻击')

# 实验配置
parser.add_argument('--experiment_name', type=str, default='msvd_attack_eval', help='实验名称')
parser.add_argument('--save_results', type=bool, default=True, help='保存详细结果')
parser.add_argument('--verbose', type=bool, default=True, help='详细输出')

def load_msvd_descriptions(descriptions_file):
    """加载MSVD视频描述数据从AllVideoDescriptions.txt"""
    logging.info(f"加载视频描述数据: {descriptions_file}")
    
    try:
        video_descriptions = {}
        with open(descriptions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 分割视频ID和描述
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        video_id, description = parts
                        if video_id not in video_descriptions:
                            video_descriptions[video_id] = []
                        video_descriptions[video_id].append(description)
        
        logging.info(f"成功加载 {len(video_descriptions)} 个视频的描述数据")
        
        # 统计信息
        total_descriptions = sum(len(descs) for descs in video_descriptions.values())
        avg_descriptions = total_descriptions / len(video_descriptions)
        logging.info(f"总描述数: {total_descriptions}, 平均每视频: {avg_descriptions:.1f} 个描述")
        
        return video_descriptions
        
    except Exception as e:
        logging.error(f"加载视频描述数据失败: {str(e)}")
        return None

def load_msvd_test_list(test_list_file):
    """加载测试视频列表"""
    logging.info(f"加载测试列表: {test_list_file}")
    
    try:
        with open(test_list_file, 'r') as f:
            test_videos = [line.strip() for line in f.readlines()]
        logging.info(f"成功加载 {len(test_videos)} 个测试视频")
        return test_videos
    except Exception as e:
        logging.error(f"加载测试列表失败: {str(e)}")
        return None

def prepare_msvd_data_from_descriptions(video_descriptions, test_videos, video_dir):
    """从AllVideoDescriptions.txt准备MSVD数据"""
    logging.info("准备MSVD数据...")
    
    valid_videos = []
    valid_captions = []
    video_paths = []
    
    # 调试信息
    print(f"DEBUG: test_videos数量: {len(test_videos)}")
    print(f"DEBUG: video_descriptions包含视频数: {len(video_descriptions)}")
    
    for video_id in test_videos:
        # 检查视频文件是否存在
        video_path_avi = os.path.join(video_dir, f"{video_id}.avi")
        video_path_mp4 = os.path.join(video_dir, f"{video_id}.mp4")
        
        video_path = None
        if os.path.exists(video_path_avi):
            video_path = video_path_avi
        elif os.path.exists(video_path_mp4):
            video_path = video_path_mp4
        else:
            continue
        
        # 检查是否有对应的描述数据
        if video_id in video_descriptions:
            descriptions_list = video_descriptions[video_id]
            if len(descriptions_list) > 0:
                # 选择第一个描述作为代表性描述
                valid_videos.append(video_id)
                valid_captions.append(descriptions_list[0])
                video_paths.append(video_path)
    
    logging.info(f"成功准备 {len(valid_videos)} 个有效的视频-文本对 (每个视频一个描述)")
    print(f"DEBUG: 最终视频数: {len(valid_videos)}")
    print(f"DEBUG: 最终描述数: {len(valid_captions)}")
    
    # 显示一些示例
    for i in range(min(3, len(valid_videos))):
        print(f"DEBUG: 示例 {i}: {valid_videos[i]} -> {valid_captions[i][:50]}...")
    
    return valid_videos, valid_captions, video_paths

def compute_metrics(sim_matrix):
    """计算检索指标 - 修复维度不匹配问题"""
    print(f"DEBUG: sim_matrix shape: {sim_matrix.shape}")
    
    # 检查矩阵是否为方阵，如果不是，说明有维度问题
    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        print(f"WARNING: 相似度矩阵不是方阵: {sim_matrix.shape}")
    
    ranks = np.argsort(-sim_matrix, axis=1)  # [num_queries, num_targets]
    correct_ranks = []
    
    for i in range(sim_matrix.shape[0]):
        # 检查第i个查询是否能在目标中找到对应的答案
        if i < sim_matrix.shape[1]:  # 确保索引在范围内
            rank_positions = np.where(ranks[i] == i)[0]
            if len(rank_positions) > 0:
                rank = rank_positions[0]
                correct_ranks.append(rank)
            else:
                # 如果找不到正确答案，说明索引超出范围，设为最差排名
                print(f"WARNING: 查询 {i} 没有找到对应的正确答案")
                correct_ranks.append(sim_matrix.shape[1] - 1)  # 最差排名
        else:
            # 如果查询索引超出目标范围，这种情况下需要重新思考数据对应关系
            print(f"ERROR: 查询索引 {i} 超出目标范围 {sim_matrix.shape[1]}")
            correct_ranks.append(sim_matrix.shape[1] - 1)  # 最差排名
    
    if len(correct_ranks) == 0:
        print("ERROR: 没有有效的排名数据")
        return {
            'R1': 0.0, 'R5': 0.0, 'R10': 0.0, 
            'MR': sim_matrix.shape[1], 'MedianR': sim_matrix.shape[1], 'MeanR': sim_matrix.shape[1]
        }
    
    correct_ranks = np.array(correct_ranks)
    
    # 计算指标
    metrics = {}
    metrics['R1'] = float(np.sum(correct_ranks == 0)) * 100 / len(correct_ranks)
    metrics['R5'] = float(np.sum(correct_ranks < 5)) * 100 / len(correct_ranks)
    metrics['R10'] = float(np.sum(correct_ranks < 10)) * 100 / len(correct_ranks)
    metrics['MR'] = np.median(correct_ranks) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(correct_ranks) + 1
    
    print(f"DEBUG: 计算了 {len(correct_ranks)} 个有效排名")
    return metrics

class LanguageBindAttackWrapper(torch.nn.Module):
    """用于AutoAttack的模型包装器"""
    def __init__(self, model, text_embeddings, args, logit_scale=100.0):
        super().__init__()
        self.model = model 
        self.text_embeddings = text_embeddings.float()
        self.args = args
        self.logit_scale = logit_scale
        
    def forward(self, video_tensor):
        video_tensor = video_tensor.float()
        video_inputs = {'video': {'pixel_values': video_tensor}}
        video_embeddings = self.model(video_inputs)['video']
        sim_logits = self.logit_scale * video_embeddings @ self.text_embeddings.T
        return sim_logits

def extract_all_features(model, tokenizer, captions, video_paths, modality_transform, batch_size, device):
    """提取所有特征 - 修复tokenizer调用方式"""
    logging.info("开始提取所有特征...")
    model.eval()
    
    batch_sequence_output_list = []
    batch_visual_output_list = []
    
    with torch.no_grad():
        # 提取文本特征 - 修复tokenizer调用
        logging.info("提取文本特征...")
        text_batches = list(chunked(captions, batch_size))
        for batch_texts in tqdm(text_batches, desc="处理文本批次"):
            # 🔧 修复：这里采用与你MSR-VTT脚本相同的方式
            try:
                # 方式1：标准transformers格式
                text_tokens = tokenizer(batch_texts, max_length=77, padding='max_length', 
                                      truncation=True, return_tensors='pt')
            except:
                # 方式2：如果上面失败，尝试逐个处理
                text_tokens = {'input_ids': [], 'attention_mask': []}
                for text in batch_texts:
                    tokens = tokenizer(text, max_length=77, padding='max_length', 
                                     truncation=True, return_tensors='pt')
                    text_tokens['input_ids'].append(tokens['input_ids'])
                    text_tokens['attention_mask'].append(tokens['attention_mask'])
                text_tokens['input_ids'] = torch.cat(text_tokens['input_ids'], dim=0)
                text_tokens['attention_mask'] = torch.cat(text_tokens['attention_mask'], dim=0)
            
            text_tokens = to_device(text_tokens, device)
            
            # 使用LanguageBind的文本编码
            text_inputs = {'language': text_tokens}
            text_embeddings = model(text_inputs)['language']
            
            batch_sequence_output_list.append(text_embeddings.cpu())
        
        # 提取视频特征
        logging.info("提取视频特征...")
        video_batches = list(chunked(video_paths, batch_size))
        for batch_videos in tqdm(video_batches, desc="处理视频批次"):
            try:
                video_tensor_dict = modality_transform['video'](batch_videos)
                video_tensor_dict = to_device(video_tensor_dict, device)
                
                video_inputs = {'video': video_tensor_dict}
                video_embeddings = model(video_inputs)['video']
                
                batch_visual_output_list.append(video_embeddings.cpu())
                
            except Exception as e:
                logging.warning(f"跳过视频批次: {str(e)}")
                dummy_embeddings = torch.zeros(len(batch_videos), 768)
                batch_visual_output_list.append(dummy_embeddings)
                continue
    
    # 合并所有特征
    all_text_embeddings = torch.cat(batch_sequence_output_list, dim=0)
    all_video_embeddings = torch.cat(batch_visual_output_list, dim=0)
    
    logging.info(f"文本特征形状: {all_text_embeddings.shape}")
    logging.info(f"视频特征形状: {all_video_embeddings.shape}")
    
    return all_text_embeddings.numpy(), all_video_embeddings.numpy()

def compute_full_similarity_matrix(text_embeddings, video_embeddings, logit_scale=100.0):
    """计算完整相似度矩阵 - 添加调试信息"""
    logging.info("计算完整相似度矩阵...")
    
    print(f"DEBUG: text_embeddings shape: {text_embeddings.shape}")
    print(f"DEBUG: video_embeddings shape: {video_embeddings.shape}")
    
    # 对齐检索脚本: logit_scale * text @ video.T (注意这里是text在前)
    sim_matrix = logit_scale * text_embeddings @ video_embeddings.T  # [num_texts, num_videos]
    
    logging.info(f"相似度矩阵形状: {sim_matrix.shape}")
    print(f"DEBUG: 最终相似度矩阵形状: {sim_matrix.shape}")
    
    # 检查是否为方阵
    if sim_matrix.shape[0] != sim_matrix.shape[1]:
        logging.warning(f"相似度矩阵不是方阵: {sim_matrix.shape[0]} x {sim_matrix.shape[1]}")
        logging.info("这可能是正常的，如果文本数量与视频数量不同")
    
    return sim_matrix

def evaluate_baseline_retrieval(sim_matrix):
    """评估基线检索性能 - 专门处理MSVD的方阵情况"""
    logging.info("评估基线检索性能...")
    
    print(f"DEBUG: 基线相似度矩阵形状: {sim_matrix.shape}")
    
    # 检查矩阵维度
    num_texts, num_videos = sim_matrix.shape
    
    if num_texts == num_videos:
        # 方阵情况：标准的视频-文本配对评估
        logging.info("检测到方阵，进行标准的配对评估")
        tv_metrics = compute_metrics(sim_matrix)      # Text-to-Video
        vt_metrics = compute_metrics(sim_matrix.T)   # Video-to-Text
        
        # 检查性能是否合理
        if tv_metrics['R1'] < 1.0 and vt_metrics['R1'] < 1.0:
            logging.error("⚠️  基线性能异常低，可能存在数据对应问题!")
            logging.error("建议检查:")
            logging.error("1. 视频和文本的对应关系是否正确")
            logging.error("2. LanguageBind模型是否正确加载")
            logging.error("3. 数据预处理是否有问题")
            
    else:
        # 非方阵情况：这表明数据有问题
        logging.error(f"❌ 检测到非方阵 ({num_texts} texts, {num_videos} videos)")
        logging.error("这通常表明数据准备有问题，每个视频应该对应一个文本")
        
        # 强制创建方阵进行评估
        min_size = min(num_texts, num_videos)
        logging.warning(f"强制使用前 {min_size} 个样本创建方阵进行评估")
        
        square_sim = sim_matrix[:min_size, :min_size]
        tv_metrics = compute_metrics(square_sim)
        vt_metrics = compute_metrics(square_sim.T)
    
    # 对齐检索脚本的日志格式
    logging.info("MSVD Text-to-Video:")
    logging.info(f'\t>>>  R@1: {tv_metrics["R1"]:.1f} - R@5: {tv_metrics["R5"]:.1f} - R@10: {tv_metrics["R10"]:.1f} - Median R: {tv_metrics["MR"]:.1f} - Mean R: {tv_metrics["MeanR"]:.1f}')
    
    logging.info("MSVD Video-to-Text:")
    logging.info(f'\t>>>  V2T$R@1: {vt_metrics["R1"]:.1f} - V2T$R@5: {vt_metrics["R5"]:.1f} - V2T$R@10: {vt_metrics["R10"]:.1f} - V2T$Median R: {vt_metrics["MR"]:.1f} - V2T$Mean R: {vt_metrics["MeanR"]:.1f}')
    
    # 性能检查
    if tv_metrics['R1'] > 10.0 or vt_metrics['R1'] > 10.0:
        logging.info("✅ 基线性能看起来正常，可以进行攻击评估")
    else:
        logging.warning(f"⚠️  基线性能较低 (T2V R@1: {tv_metrics['R1']:.1f}%, V2T R@1: {vt_metrics['R1']:.1f}%)")
        logging.warning("攻击效果可能不明显")
    
    return tv_metrics, vt_metrics

def evaluate_attack_on_subset(attack_method, model, modality_transform, video_paths, 
                             text_embeddings, baseline_sim_matrix, args):
    """评估攻击效果 - 简化版本，专注于解决tokenizer问题"""
    logging.info(f"\n开始评估攻击方法: {attack_method}")
    
    device = next(model.parameters()).device
    total_samples = len(video_paths)
    
    # 随机选择攻击样本
    attack_indices = random.sample(range(total_samples), 
                                 min(args.n_attack_samples, total_samples))
    
    logging.info(f"使用子集进行攻击 ({len(attack_indices)} 个样本)")
    logging.info(f"每个攻击视频将在全部 {len(text_embeddings)} 个文本中进行检索")
    
    # 为攻击创建包装模型
    text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
    wrapped_model = LanguageBindAttackWrapper(model, text_embeddings_tensor, args)
    
    # 存储攻击结果
    all_original_video_embeddings = []
    all_attacked_video_embeddings = []
    all_attack_labels = []
    
    total_attack_time = 0
    total_valid_samples = 0
    
    # 按批次处理攻击
    batch_size = args.attack_batch_size
    num_batches = (len(attack_indices) + batch_size - 1) // batch_size
    
    logging.info(f"将分 {num_batches} 个批次进行攻击，每批最多 {batch_size} 个样本")
    
    for batch_idx in tqdm(range(num_batches), desc=f"攻击批次 {attack_method}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(attack_indices))
        batch_indices = attack_indices[start_idx:end_idx]
        
        # 准备当前批次的数据
        batch_videos = []
        batch_labels = []
        
        for attack_idx in batch_indices:
            try:
                video_path = video_paths[attack_idx]
                if not os.path.exists(video_path):
                    continue
                
                # 预处理视频
                video_tensor_dict = modality_transform['video']([video_path])
                video_tensor_dict = to_device(video_tensor_dict, device)
                video_tensor = video_tensor_dict['pixel_values'].float()
                
                batch_videos.append(video_tensor.squeeze(0))
                batch_labels.append(attack_idx)
                
            except Exception as e:
                if args.verbose and batch_idx < 5:
                    logging.warning(f"跳过样本 {attack_idx}: {str(e)}")
                continue
        
        if len(batch_videos) == 0:
            continue
        
        # 转换为tensor
        x_batch = torch.stack(batch_videos).to(device)
        y_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)
        
        # 获取原始视频embeddings
        with torch.no_grad():
            original_video_inputs = {'video': {'pixel_values': x_batch}}
            original_video_embeddings = model(original_video_inputs)['video']
        
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
            adv_video_embeddings = model(adv_video_inputs)['video']
        
        # 保存embeddings
        all_original_video_embeddings.append(original_video_embeddings.cpu().numpy())
        all_attacked_video_embeddings.append(adv_video_embeddings.cpu().numpy())
        all_attack_labels.extend(batch_labels)
        
        total_attack_time += batch_attack_time
        total_valid_samples += len(batch_labels)
        
        # 清理内存
        del x_batch, y_batch, x_adv, y_adv, original_video_embeddings, adv_video_embeddings, batch_videos
        torch.cuda.empty_cache()
        
        # 输出进度
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            logging.info(f"  完成: {batch_idx + 1}/{num_batches} 批次, "
                        f"累计样本: {total_valid_samples}, "
                        f"累计耗时: {total_attack_time:.1f}s")
    
    if total_valid_samples == 0:
        logging.error(f"{attack_method}: 没有有效的攻击样本")
        return None
    
    # 在全集上进行检索评估
    logging.info(f"开始在全集上评估 {total_valid_samples} 个攻击样本的检索性能...")
    
    # 合并所有攻击样本的embeddings
    all_original_embeddings = np.concatenate(all_original_video_embeddings, axis=0)
    all_attacked_embeddings = np.concatenate(all_attacked_video_embeddings, axis=0)
    
    # 计算在全集上的相似度矩阵
    original_sim_matrix = 100.0 * all_original_embeddings @ text_embeddings.T
    attacked_sim_matrix = 100.0 * all_attacked_embeddings @ text_embeddings.T
    
    # 计算每个攻击样本的检索性能
    original_ranks = []
    attacked_ranks = []
    
    for i, true_label in enumerate(all_attack_labels):
        # 原始视频的检索排名
        original_similarities = original_sim_matrix[i, :]
        original_sorted_indices = np.argsort(-original_similarities)
        original_rank = np.where(original_sorted_indices == true_label)[0][0]
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
    
    # 计算攻击效果
    rank_degradation = np.mean(attacked_ranks - original_ranks)
    successful_attacks = np.sum(attacked_ranks > original_ranks)
    attack_success_rate = successful_attacks / len(original_ranks)
    
    # 对比基线性能
    baseline_ranks = []
    for true_label in all_attack_labels:
        baseline_similarities = baseline_sim_matrix[true_label, :]
        baseline_sorted_indices = np.argsort(-baseline_similarities)
        baseline_rank = np.where(baseline_sorted_indices == true_label)[0][0]
        baseline_ranks.append(baseline_rank)
    
    baseline_ranks = np.array(baseline_ranks)
    baseline_metrics = compute_retrieval_metrics(baseline_ranks)
    
    # 整理结果
    results = {
        'attack_method': attack_method,
        'total_samples': total_valid_samples,
        'attack_time': total_attack_time,
        'avg_time_per_sample': total_attack_time / total_valid_samples,
        
        # 基线性能
        'baseline_r1': baseline_metrics['R1'],
        'baseline_r5': baseline_metrics['R5'],
        'baseline_r10': baseline_metrics['R10'],
        
        # 攻击前性能
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
    logging.info(f"  攻击样本数: {total_valid_samples}")
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
    
    # 转换参数范围
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
    
    # 加载MSVD数据
    video_descriptions = load_msvd_descriptions(args.descriptions_file)
    if video_descriptions is None:
        return
    
    test_videos = load_msvd_test_list(args.test_list)
    if test_videos is None:
        return
    
    video_ids, captions, video_paths = prepare_msvd_data_from_descriptions(
        video_descriptions, test_videos, args.video_dir
    )
    
    if len(video_paths) == 0:
        logging.error("没有找到有效的视频文件")
        return
    
    # 提取所有特征并计算基线性能
    text_embeddings, video_embeddings = extract_all_features(
        model, tokenizer, captions, video_paths, 
        modality_transform, args.batch_size, device
    )
    
    # 计算完整相似度矩阵
    sim_matrix = compute_full_similarity_matrix(text_embeddings, video_embeddings)
    
    # 评估基线检索性能
    baseline_tv_metrics, baseline_vt_metrics = evaluate_baseline_retrieval(sim_matrix)
    
    # 对每种攻击方法进行评估
    all_attack_results = []
    
    for attack_method in attacks_to_run:
        try:
            result = evaluate_attack_on_subset(
                attack_method, model, modality_transform, video_paths,
                text_embeddings, sim_matrix, args
            )
            if result:
                all_attack_results.append(result)
        except Exception as e:
            logging.error(f"攻击方法 {attack_method} 执行失败: {str(e)}")
            continue
    
    # 结果汇总
    logging.info("\n" + "="*100)
    logging.info("MSVD攻击评估结果汇总:")
    logging.info("="*100)
    
    if all_attack_results:
        # 按R@1下降排序
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
    
    # 保存结果
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
    # 使用示例
    print("MSVD对抗攻击评估脚本")
    print("使用示例:")
    print("  python msvd_complete_script.py --descriptions_file ./AllVideoDescriptions.txt --video_dir ./videos --test_list ./test_list.txt --n_attack_samples 10")
    print("")
    
    main()