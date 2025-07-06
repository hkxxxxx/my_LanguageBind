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

# ========== 参数配置 (对齐CLIP RobustBench) ==========
parser = argparse.ArgumentParser(description="MSR-VTT对抗攻击评估")

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
parser.add_argument('--n_samples', type=int, default=200, help='每种攻击方法的测试样本数')
parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')

# 攻击方法选择
parser.add_argument('--blackbox_only', type=bool, default=False, help='仅运行黑盒攻击')
parser.add_argument('--attacks_to_run', type=str, nargs='+', default=None, 
                   help='指定运行的攻击方法，默认为所有白盒攻击')

# 实验配置
parser.add_argument('--experiment_name', type=str, default='msrvtt_attack_eval', help='实验名称')
parser.add_argument('--save_results', type=bool, default=True, help='保存详细结果')
parser.add_argument('--verbose', type=bool, default=True, help='详细输出')

def compute_metrics(x):
    """计算检索指标 (对齐原始验证脚本)"""
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    return metrics

class LanguageBindAttackWrapper(torch.nn.Module):
    """用于AutoAttack的模型包装器 (对齐CLIP ClassificationModel结构)"""
    def __init__(self, model, text_embeddings, args):
        super().__init__()
        self.model = model 
        self.text_embeddings = text_embeddings.float()  # [num_texts, embed_dim]
        self.args = args
        
    def forward(self, video_tensor):
        """
        前向传播 - 输出logits而不是相似度分数
        对齐CLIP RobustBench的ClassificationModel.forward()
        """
        # 确保输入是float32
        video_tensor = video_tensor.float()
        
        # 获取视频embedding
        video_inputs = {'video': {'pixel_values': video_tensor}}
        video_embeddings = self.model(video_inputs)['video']  # [batch_size, embed_dim]
        
        # 计算相似度矩阵 [batch_size, num_texts]
        logits = video_embeddings @ self.text_embeddings.T
        
        # 对齐CLIP的logit scaling (虽然LanguageBind可能没有这个参数)
        # 这里使用一个固定的scale factor来增强logits
        logits = logits * 100  # 经验性的scaling factor
        
        return logits

def load_msrvtt_test_data(csv_file, video_base_path):
    """加载MSR-VTT测试数据"""
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
    
    print(f"成功加载 {len(df_valid)} 个有效的视频-文本对")
    return language_data, video_data, df_valid

def get_embeddings(model, tokenizer, language_data, video_data, modality_transform, batch_size):
    """获取所有数据的embeddings"""
    print("计算embeddings...")
    
    def embed(x: list[list], dtypes: list[str]) -> dict:
        inputs = {}
        for data, dtype in zip(x, dtypes):
            if dtype == 'language':
                inputs['language'] = to_device(
                    tokenizer(data, max_length=77, padding='max_length', 
                             truncation=True, return_tensors='pt'), device)
            elif dtype in modality_transform:
                inputs[dtype] = to_device(modality_transform[dtype](data), device)
            else:
                raise ValueError(f"Unknown dtype: {dtype}")
        
        with torch.no_grad():
            embeddings = model(inputs)
        
        embeddings = {k: v.detach().cpu().numpy() for k, v in embeddings.items()}
        return embeddings
    
    results = defaultdict(lambda: np.empty((0, 768)))
    
    for batch in tqdm(list(zip(
            chunked(language_data, batch_size),
            chunked(video_data, batch_size)
        )), desc="处理批次"):
        
        embeddings = embed(batch, dtypes=['language', 'video'])
        results['language'] = np.concatenate([results['language'], embeddings['language']])
        results['video'] = np.concatenate([results['video'], embeddings['video']])
    
    return results['video'], results['language']

def compute_accuracy_no_dataloader(model, data, targets, device, batch_size=64):
    """计算准确率 (对齐CLIP eval_utils函数)"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            
            total += batch_targets.size(0)
            correct += (predicted == batch_targets).sum().item()
    
    return correct / total

def evaluate_single_attack(attack_method, model, tokenizer, modality_transform, 
                          language_data, video_data, video_embeddings, text_embeddings, args):
    """评估单个攻击方法 (对齐RobustBench评估流程)"""
    print(f"\n评估攻击方法: {attack_method}")
    
    # 随机选择测试样本 - 注意这里只是选择索引，不影响embeddings计算
    total_samples = len(video_data)
    test_indices = random.sample(range(total_samples), min(args.n_samples, total_samples))
    
    # 为攻击创建包装模型
    text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
    wrapped_model = LanguageBindAttackWrapper(model, text_embeddings_tensor, args)
    
    # 准备测试数据 - 这里才是真正影响内存的地方
    test_videos = []
    test_labels = []
    
    print(f"准备 {len(test_indices)} 个测试样本...")
    successful_loads = 0
    max_test_samples = min(args.n_samples, 1000)  # 限制最大样本数避免OOM
    
    for test_idx in tqdm(test_indices, desc="加载视频"):
        if successful_loads >= max_test_samples:  # 达到限制就停止
            break
            
        try:
            video_path = video_data[test_idx]
            if not os.path.exists(video_path):
                continue
                
            # 预处理视频
            video_tensor_dict = modality_transform['video']([video_path])
            video_tensor_dict = to_device(video_tensor_dict, device)
            video_tensor = video_tensor_dict['pixel_values'].float()  # [1, C, T, H, W]
            
            test_videos.append(video_tensor.squeeze(0))  # 移除batch维度
            test_labels.append(test_idx)
            successful_loads += 1
            
        except Exception as e:
            if args.verbose:
                print(f"跳过样本 {test_idx}: {str(e)}")
            continue
    
    if len(test_videos) == 0:
        print(f"{attack_method}: 没有有效的测试样本")
        return None
    
    # 转换为tensor
    x_test = torch.stack(test_videos).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.long).to(device)
    
    print(f"实际测试数据形状: {x_test.shape}, 标签数量: {len(y_test)}")
    
    # 释放不必要的内存
    del test_videos
    torch.cuda.empty_cache()
    
    # 计算原始准确率 (对齐RobustBench流程)
    acc = compute_accuracy_no_dataloader(wrapped_model, x_test, y_test, device, min(args.batch_size, 8)) * 100
    print(f'[{attack_method}] 原始准确率: {acc:.2f}%', flush=True)
    
    # 配置AutoAttack (对齐RobustBench参数)
    adversary = AutoAttack(
        wrapped_model, 
        norm=args.norm.replace('l', 'L'),  # 转换为大写 (linf -> Linf)
        eps=args.eps,  # 已经在main函数中转换为0-1范围
        version='custom',
        attacks_to_run=[attack_method],
        alpha=args.alpha,  # alpha也已经在main函数中转换
        verbose=args.verbose
    )
    
    # 运行攻击 (对齐RobustBench调用) - 使用更小的batch size
    start_time = time.time()
    attack_batch_size = min(args.batch_size, 4)  # 减小攻击时的batch size
    print(f"使用攻击batch size: {attack_batch_size}")
    
    x_adv, y_adv = adversary.run_standard_evaluation(
        x_test, y_test, bs=attack_batch_size, return_labels=True
    )
    attack_time = time.time() - start_time
    
    # 计算鲁棒准确率
    racc = compute_accuracy_no_dataloader(wrapped_model, x_adv, y_test, device, min(args.batch_size, 8)) * 100
    
    # 计算攻击成功率
    success_rate = (acc - racc) / acc if acc > 0 else 0
    
    # 计算检索指标 (额外的MSR-VTT特定指标)
    with torch.no_grad():
        # 原始相似度
        original_sim = wrapped_model(x_test).cpu().numpy()
        # 攻击后相似度  
        adv_sim = wrapped_model(x_adv).cpu().numpy()
    
    # 计算检索性能下降
    original_metrics = compute_metrics(original_sim)
    adv_metrics = compute_metrics(adv_sim)
    
    results = {
        'attack_method': attack_method,
        'clean_accuracy': acc,
        'robust_accuracy': racc, 
        'accuracy_drop': acc - racc,
        'success_rate': success_rate,
        'total_samples': len(y_test),
        'attack_time': attack_time,
        'original_r1': original_metrics['R1'],
        'adv_r1': adv_metrics['R1'],
        'r1_drop': original_metrics['R1'] - adv_metrics['R1'],
        'original_r5': original_metrics['R5'],
        'adv_r5': adv_metrics['R5'],
        'r5_drop': original_metrics['R5'] - adv_metrics['R5']
    }
    
    print(f"{attack_method} 结果:")
    print(f"  原始准确率: {acc:.2f}%")
    print(f"  鲁棒准确率: {racc:.2f}%") 
    print(f"  准确率下降: {acc - racc:.2f}%")
    print(f"  攻击成功率: {success_rate:.3f}")
    print(f"  R@1: {original_metrics['R1']:.2f}% → {adv_metrics['R1']:.2f}% (下降{original_metrics['R1'] - adv_metrics['R1']:.2f}%)")
    print(f"  攻击耗时: {attack_time:.2f}秒")
    
    # 清理内存
    del x_test, y_test, x_adv, y_adv, original_sim, adv_sim
    torch.cuda.empty_cache()
    
    return results

def main():
    global device
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # 转换参数到正确范围 (对齐RobustBench)
    args.eps = args.eps / 255  # 转换为0-1范围
    args.alpha = args.alpha / 255  # alpha也转换为0-1范围
    
    print(f"参数配置:\n{'-' * 40}")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print(f"{'-' * 40}")
    
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确定攻击方法 (对齐RobustBench逻辑)
    if args.attacks_to_run is not None:
        attacks_to_run = args.attacks_to_run
    elif args.blackbox_only:
        attacks_to_run = ['square']
    else:
        # 所有白盒攻击方法
        attacks_to_run = ['apgd-ce', 'apgd-dlr', 'apgd-t', 'fab-t']
        # attacks_to_run = ['fab-t']

    print(f'[attacks_to_run] {attacks_to_run}')
    
    # 加载LanguageBind模型
    print("加载LanguageBind模型...")
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
    
    # 获取所有embeddings
    video_embeddings, text_embeddings = get_embeddings(
        model, tokenizer, language_data, video_data, modality_transform, args.batch_size
    )
    
    # 计算基线性能
    print("\n计算基线检索性能...")
    sim_matrix = torch.tensor(video_embeddings @ text_embeddings.T)
    baseline_vt = compute_metrics(sim_matrix)
    baseline_tv = compute_metrics(sim_matrix.T)
    
    print(f"基线性能 - Video-to-Text: R@1={baseline_vt['R1']:.2f}%, R@5={baseline_vt['R5']:.2f}%, R@10={baseline_vt['R10']:.2f}%")
    print(f"基线性能 - Text-to-Video: R@1={baseline_tv['R1']:.2f}%, R@5={baseline_tv['R5']:.2f}%, R@10={baseline_tv['R10']:.2f}%")
    
    # 对每种攻击方法进行评估
    all_results = []
    
    for attack_method in attacks_to_run:
        result = evaluate_single_attack(
            attack_method, model, tokenizer, modality_transform,
            language_data, video_data, video_embeddings, text_embeddings, args
        )
        if result:
            all_results.append(result)
    
    # ========== 结果汇总 (对齐RobustBench输出格式) ==========
    print("\n" + "="*100)
    print("攻击方法对比结果汇总:")
    print("="*100)
    print(f"{'方法':<12} {'原始准确率':<10} {'鲁棒准确率':<10} {'准确率下降':<10} {'R@1下降':<8} {'攻击时间':<8}")
    print("-"*100)
    
    # 按准确率下降排序 (攻击效果)
    all_results.sort(key=lambda x: x['accuracy_drop'], reverse=True)
    
    for result in all_results:
        print(f"{result['attack_method']:<12} "
              f"{result['clean_accuracy']:.2f}%       "
              f"{result['robust_accuracy']:.2f}%       "
              f"{result['accuracy_drop']:.2f}%       "
              f"{result['r1_drop']:.2f}%    "
              f"{result['attack_time']:.1f}s")
    
    # 找出最有效的攻击方法
    if all_results:
        best_method = all_results[0]
        print(f"\n🏆 最有效的攻击方法: {best_method['attack_method']}")
        print(f"   准确率下降: {best_method['accuracy_drop']:.2f}%")
        print(f"   R@1下降: {best_method['r1_drop']:.2f}%")
        print(f"   攻击成功率: {best_method['success_rate']:.3f}")
    
    # 保存结果 (对齐RobustBench保存格式)
    if args.save_results:
        results_data = {
            'experiment_config': vars(args),
            'baseline_performance': {
                'video_to_text': baseline_vt,
                'text_to_video': baseline_tv
            },
            'attack_results': all_results,
            'summary': {
                'best_attack': best_method['attack_method'] if all_results else None,
                'max_accuracy_drop': best_method['accuracy_drop'] if all_results else 0,
                'max_r1_drop': best_method['r1_drop'] if all_results else 0
            }
        }
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f'{args.experiment_name}_{args.norm}_eps{int(args.eps*255)}_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {results_file}")

if __name__ == "__main__":
    main()