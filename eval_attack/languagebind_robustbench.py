import torch
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from more_itertools import chunked
from tqdm.auto import tqdm
from autoattack import AutoAttack
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import json
import random

# ========== 配置 ==========
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clip_type = {
    'video': 'LanguageBind_Video_FT',  # 使用FT版本
}

# 数据集路径
csv_file = '/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_JSFUSION_test.csv'
video_base_path = '../data/MSRVTT/videos/all'

import torch
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from more_itertools import chunked
from tqdm.auto import tqdm
from autoattack import AutoAttack
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import json
import random

# ========== 配置 ==========
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clip_type = {
    'video': 'LanguageBind_Video_FT',  # 使用FT版本
}

# 数据集路径
csv_file = '/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_JSFUSION_test.csv'
video_base_path = '/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/MSRVTT_Videos/video'

# 攻击配置
attack_methods = ['apgd-ce', 'apgd-dlr', 'fab', 'apgd-t', 'fab-t']
eps = 4/255  # 攻击强度
batch_size = 16
test_samples = 2  # 每种攻击方法测试的样本数

def compute_metrics(x):
    """计算检索指标"""
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
    """用于AutoAttack的模型包装器"""
    def __init__(self, model, text_embeddings):
        super().__init__()
        self.model = model 
        # 确保embeddings是float32类型
        self.text_embeddings = text_embeddings.float()  # [num_texts, embed_dim]
        
    def forward(self, video_tensor):
        # video_tensor: [batch_size, C, T, H, W]
        # 确保输入是float32
        video_tensor = video_tensor.float()
        
        # 获取视频embedding (不能用no_grad，因为需要梯度)
        video_inputs = {'video': {'pixel_values': video_tensor}}
        video_embeddings = self.model(video_inputs)['video']  # [batch_size, embed_dim]
        
        # 计算相似度矩阵 [batch_size, num_texts]
        sim_matrix = video_embeddings @ self.text_embeddings.T
        return sim_matrix

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

def get_embeddings(model, tokenizer, language_data, video_data, modality_transform):
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

def evaluate_single_attack(attack_method, model, tokenizer, modality_transform, 
                          language_data, video_data, video_embeddings, text_embeddings):
    """评估单个攻击方法"""
    print(f"\n评估攻击方法: {attack_method}")
    
    # 随机选择测试样本
    total_samples = len(video_data)
    test_indices = random.sample(range(total_samples), min(test_samples, total_samples))
    
    success_count = 0
    total_count = 0
    original_r1_scores = []
    attacked_r1_scores = []
    
    # 为攻击创建包装模型
    text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32).to(device)
    wrapped_model = LanguageBindAttackWrapper(model, text_embeddings_tensor)
    
    for test_idx in tqdm(test_indices, desc=f"测试 {attack_method}"):
        try:
            video_path = video_data[test_idx]
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                continue
            
            # 预处理单个视频
            video_tensor_dict = modality_transform['video']([video_path])
            video_tensor_dict = to_device(video_tensor_dict, device)
            video_tensor = video_tensor_dict['pixel_values'].float()  # [1, C, T, H, W] 确保float32
            video_tensor.requires_grad_()
            
            # 检查tensor形状和类型
            if video_tensor.dim() != 5 or video_tensor.size(0) != 1:
                print(f"视频tensor形状异常: {video_tensor.shape}")
                continue
            
            # 获取原始预测
            with torch.no_grad():
                original_sim = wrapped_model(video_tensor.detach())  # [1, num_texts]
                original_pred = torch.argmax(original_sim, dim=-1)  # [1]
            
            # 真实标签（对应的文本索引）
            true_label = torch.tensor([test_idx], dtype=torch.long).to(device)
            
            # 检查原始预测是否正确
            if original_pred.item() != test_idx:
                # 如果原始预测就是错的，我们仍然进行攻击测试
                pass
            
            # 配置AutoAttack - 使用更稳定的设置
            adversary = AutoAttack(
                wrapped_model, 
                norm='Linf', 
                eps=eps,
                version='custom',
                attacks_to_run=[attack_method],
                verbose=False,
                device=device
            )
            
            # 运行攻击
            x_adv = adversary.run_standard_evaluation(video_tensor, true_label, bs=1)
            
            # 获取攻击后的预测
            with torch.no_grad():
                adv_sim = wrapped_model(x_adv.detach())  # [1, num_texts]
                adv_pred = torch.argmax(adv_sim, dim=-1)  # [1]
            
            # 计算检索性能
            original_sim_cpu = original_sim.cpu().numpy()
            adv_sim_cpu = adv_sim.cpu().numpy()
            
            # 计算单样本的R@1 (排名是否为第一)
            original_rank = np.argsort(-original_sim_cpu[0])
            adv_rank = np.argsort(-adv_sim_cpu[0])
            
            original_r1 = 1.0 if original_rank[0] == test_idx else 0.0
            adv_r1 = 1.0 if adv_rank[0] == test_idx else 0.0
            
            original_r1_scores.append(original_r1)
            attacked_r1_scores.append(adv_r1)
            
            # 检查攻击是否成功（预测发生改变或性能下降）
            if original_pred.item() != adv_pred.item() or original_r1 > adv_r1:
                success_count += 1
            
            total_count += 1
            
        except Exception as e:
            print(f"处理样本 {test_idx} 时出错: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            continue
    
    # 计算统计结果
    if total_count > 0:
        success_rate = success_count / total_count
        original_r1_avg = np.mean(original_r1_scores) * 100
        attacked_r1_avg = np.mean(attacked_r1_scores) * 100
        r1_drop = original_r1_avg - attacked_r1_avg
        
        results = {
            'attack_method': attack_method,
            'success_rate': success_rate,
            'success_count': success_count,
            'total_count': total_count,
            'original_r1': original_r1_avg,
            'attacked_r1': attacked_r1_avg,
            'r1_drop': r1_drop
        }
        
        print(f"{attack_method} 结果:")
        print(f"  攻击成功率: {success_rate:.3f} ({success_count}/{total_count})")
        print(f"  原始R@1: {original_r1_avg:.2f}%")
        print(f"  攻击后R@1: {attacked_r1_avg:.2f}%")
        print(f"  R@1下降: {r1_drop:.2f}%")
        
        return results
    else:
        print(f"{attack_method}: 没有有效的测试样本")
        return None

def main():
    print("开始MSR-VTT对抗攻击评估...")
    
    # 加载模型
    print("加载LanguageBind模型...")
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir').to(device)
    model.eval()
    
    tokenizer = LanguageBindImageTokenizer.from_pretrained(
        'lb203/LanguageBind_Image', 
        cache_dir='./cache_dir/tokenizer_cache_dir'
    )
    
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}
    
    # 加载测试数据
    language_data, video_data, df_valid = load_msrvtt_test_data(csv_file, video_base_path)
    
    # 获取所有embeddings
    video_embeddings, text_embeddings = get_embeddings(
        model, tokenizer, language_data, video_data, modality_transform
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
    
    for attack_method in attack_methods:
        result = evaluate_single_attack(
            attack_method, model, tokenizer, modality_transform,
            language_data, video_data, video_embeddings, text_embeddings
        )
        if result:
            all_results.append(result)
    
    # ========== 结果汇总 ==========
    print("\n" + "="*80)
    print("攻击方法对比结果汇总:")
    print("="*80)
    print(f"{'方法':<12} {'成功率':<8} {'样本数':<8} {'原始R@1':<10} {'攻击后R@1':<12} {'R@1下降':<8}")
    print("-"*80)
    
    # 按R@1下降幅度排序（攻击效果）
    all_results.sort(key=lambda x: x['r1_drop'], reverse=True)
    
    for result in all_results:
        print(f"{result['attack_method']:<12} "
              f"{result['success_rate']:.3f}    "
              f"{result['success_count']}/{result['total_count']:<3} "
              f"{result['original_r1']:.2f}%      "
              f"{result['attacked_r1']:.2f}%        "
              f"{result['r1_drop']:.2f}%")
    
    # 找出最有效的攻击方法
    if all_results:
        best_method = all_results[0]
        print(f"\n🏆 最有效的攻击方法: {best_method['attack_method']}")
        print(f"   R@1性能下降: {best_method['r1_drop']:.2f}%")
        print(f"   攻击成功率: {best_method['success_rate']:.3f}")
    
    # 保存结果
    results_data = {
        'baseline_performance': {
            'video_to_text': baseline_vt,
            'text_to_video': baseline_tv
        },
        'attack_results': all_results,
        'config': {
            'eps': eps,
            'test_samples': test_samples,
            'batch_size': batch_size
        }
    }
    
    results_file = 'msrvtt_1k_attack_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: {results_file}")

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    main()