# # 1.安装huggingface_hub
# # pip install huggingface_hub
# import os
# from huggingface_hub import snapshot_download
 
# # 使用cache_dir参数，将模型/数据集保存到指定“本地路径”
# snapshot_download(repo_id="friedrichor/MSR-VTT", repo_type="dataset",
#                   cache_dir="/root/autodl-tmp/LanguageBind/datasets",
#                   local_dir_use_symlinks=False, resume_download=True)

#!/usr/bin/env python3
"""
简化的MSR-VTT数据处理脚本
直接使用已下载的数据
"""

import os
import json
import zipfile
import shutil
from pathlib import Path

def setup_msrvtt_data():
    """处理已下载的MSR-VTT数据"""
    
    # 源路径（你下载的数据位置）
    source_dir = Path("/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962")
    
    # 目标路径（整理后的数据位置）
    target_dir = Path("/root/autodl-tmp/LanguageBind/datasets/MSR-VTT")
    
    print(f"🔄 处理MSR-VTT数据...")
    print(f"源路径: {source_dir}")
    print(f"目标路径: {target_dir}")
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = target_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # 1. 解压视频文件
    video_zip = source_dir / "MSRVTT_Videos.zip"
    if video_zip.exists():
        print("🔄 解压视频文件...")
        with zipfile.ZipFile(video_zip, 'r') as zip_ref:
            zip_ref.extractall(videos_dir)
        print(f"✅ 视频文件解压完成")
        
        # 检查解压后的视频数量
        video_count = len(list(videos_dir.rglob("*.mp4")))
        print(f"📹 解压了 {video_count} 个视频文件")
    else:
        print("❌ 未找到MSRVTT_Videos.zip文件")
        return False
    
    # 2. 复制标注文件
    annotation_files = [
        "msrvtt_train_9k.json",
        "msrvtt_train_7k.json", 
        "msrvtt_test_1k.json"
    ]
    
    print("🔄 复制标注文件...")
    for file_name in annotation_files:
        source_file = source_dir / file_name
        target_file = target_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"✅ 复制: {file_name}")
        else:
            print(f"⚠️  未找到: {file_name}")
    
    # 3. 创建统一的数据文件（合并训练和测试数据）
    create_unified_annotation(target_dir)
    
    print(f"✅ MSR-VTT数据处理完成!")
    print(f"📁 数据位置: {target_dir}")
    
    return True

def create_unified_annotation(data_dir):
    """创建统一的标注文件"""
    print("🔄 创建统一标注文件...")
    
    unified_data = {
        'videos': [],
        'sentences': []
    }
    
    video_id_set = set()
    sentence_id = 0
    
    # 处理训练数据
    train_file = data_dir / "msrvtt_train_9k.json"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # 处理视频信息
        if 'videos' in train_data:
            for video in train_data['videos']:
                video_id = video['video_id']
                if video_id not in video_id_set:
                    unified_data['videos'].append({
                        'video_id': video_id,
                        'url': video.get('url', ''),
                        'start_time': video.get('start time', 0),
                        'end_time': video.get('end time', 0),
                        'split': 'train'
                    })
                    video_id_set.add(video_id)
        
        # 处理句子信息
        if 'sentences' in train_data:
            for sentence in train_data['sentences']:
                unified_data['sentences'].append({
                    'caption': sentence['caption'],
                    'video_id': sentence['video_id'],
                    'sen_id': sentence_id
                })
                sentence_id += 1
    
    # 处理测试数据
    test_file = data_dir / "msrvtt_test_1k.json"
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # 处理视频信息
        if 'videos' in test_data:
            for video in test_data['videos']:
                video_id = video['video_id']
                if video_id not in video_id_set:
                    unified_data['videos'].append({
                        'video_id': video_id,
                        'url': video.get('url', ''),
                        'start_time': video.get('start time', 0),
                        'end_time': video.get('end time', 0),
                        'split': 'test'
                    })
                    video_id_set.add(video_id)
        
        # 处理句子信息
        if 'sentences' in test_data:
            for sentence in test_data['sentences']:
                unified_data['sentences'].append({
                    'caption': sentence['caption'],
                    'video_id': sentence['video_id'],
                    'sen_id': sentence_id
                })
                sentence_id += 1
    
    # 保存统一的标注文件
    unified_file = data_dir / "MSRVTT_data.json"
    with open(unified_file, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 统一标注文件已创建: {unified_file}")
    print(f"📊 总视频数: {len(unified_data['videos'])}")
    print(f"📊 总句子数: {len(unified_data['sentences'])}")

def verify_data():
    """验证数据完整性"""
    data_dir = Path("/root/autodl-tmp/LanguageBind/datasets/MSR-VTT")
    
    print("🔍 验证数据完整性...")
    
    # 检查视频文件
    videos_dir = data_dir / "videos"
    video_files = list(videos_dir.rglob("*.mp4"))
    print(f"📹 视频文件数量: {len(video_files)}")
    
    # 检查标注文件
    annotation_file = data_dir / "MSRVTT_data.json"
    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📊 标注中的视频数: {len(data['videos'])}")
        print(f"📊 标注中的句子数: {len(data['sentences'])}")
        
        # 检查视频文件和标注的匹配度
        annotated_videos = set(video['video_id'] for video in data['videos'])
        actual_video_files = set(f.stem for f in video_files)
        
        missing_videos = annotated_videos - actual_video_files
        extra_videos = actual_video_files - annotated_videos
        
        print(f"📈 匹配的视频数: {len(annotated_videos & actual_video_files)}")
        if missing_videos:
            print(f"⚠️  缺失的视频数: {len(missing_videos)}")
        if extra_videos:
            print(f"ℹ️  额外的视频数: {len(extra_videos)}")
        
        return True
    else:
        print("❌ 标注文件不存在")
        return False

if __name__ == "__main__":
    print("🚀 开始处理MSR-VTT数据...")
    
    # 检查源数据是否存在
    source_dir = Path("/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962")
    if not source_dir.exists():
        print(f"❌ 源数据目录不存在: {source_dir}")
        print("请确认数据已正确下载")
        exit(1)
    
    # 处理数据
    if setup_msrvtt_data():
        # 验证数据
        if verify_data():
            print("✅ MSR-VTT数据处理成功!")
            print("\n现在可以运行对抗攻击脚本了:")
            print("python msrvtt_adversarial_attack.py")
        else:
            print("❌ 数据验证失败")
    else:
        print("❌ 数据处理失败")