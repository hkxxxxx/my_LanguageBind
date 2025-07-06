# #!/bin/bash

# # 基于LanguageBind在MSR-VTT数据集上训练的脚本
# # 使用你提供的数据路径

# # 设置环境变量
# export CUDA_VISIBLE_DEVICES=0  # 根据你的GPU数量调整
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# # 数据路径配置
# CACHE_DIR="/root/autodl-tmp/LanguageBind/cache_dir"
# LANGUAGEBIND_MODEL="/root/autodl-tmp/LanguageBind/cache_dir/models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66"

# # MSR-VTT数据集路径
# MSRVTT_TRAIN_CSV="/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_train.9k.csv"
# MSRVTT_DATA_JSON="/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_data.json"
# MSRVTT_VIDEO_PATH="/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/MSRVTT_Videos/video"

# # 日志和输出目录
# LOG_DIR="./logs/languagebind_msrvtt_$(date +%Y%m%d_%H%M%S)"
# mkdir -p $LOG_DIR

# # 训练参数配置
# NUM_GPUS=1  # 根据实际GPU数量调整
# BATCH_SIZE=32  # 每GPU的batch size
# ACCUMULATION_STEPS=2  # 梯度累积步数
# LEARNING_RATE=1e-4
# EPOCHS=10
# NUM_FRAMES=8
# MAX_WORDS=77

# echo "开始基于LanguageBind在MSR-VTT数据集上训练..."
# echo "使用GPU数量: $NUM_GPUS"
# echo "总batch size: $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
# echo "LanguageBind模型路径: $LANGUAGEBIND_MODEL"
# echo "MSR-VTT训练CSV: $MSRVTT_TRAIN_CSV"
# echo "MSR-VTT数据JSON: $MSRVTT_DATA_JSON"
# echo "MSR-VTT视频路径: $MSRVTT_VIDEO_PATH"
# echo "日志目录: $LOG_DIR"

# # 启动分布式训练
# torchrun --nproc_per_node=$NUM_GPUS \
#     -m main \
#     --model "languagebind_video" \
#     --pretrained "" \
#     --cache-dir "$CACHE_DIR" \
#     \
#     --msrvtt-train-csv "$MSRVTT_TRAIN_CSV" \
#     --msrvtt-data-json "$MSRVTT_DATA_JSON" \
#     --msrvtt-video-path "$MSRVTT_VIDEO_PATH" \
#     \
#     --train-data "msrvtt" \
#     --do_train \
#     --do_eval \
#     --val_vl_ret_data "msrvtt" \
#     \
#     --clip-type "vl_new" \
#     --num-frames $NUM_FRAMES \
#     --max_words $MAX_WORDS \
#     --image-resolution 224 \
#     \
#     --lock-text \
#     --lock-image \
#     --init-temp 0.07 \
#     --learn-temp \
#     \
#     --convert_to_lora \
#     --lora_r 16 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     \
#     --lr $LEARNING_RATE \
#     --coef-lr 1e-3 \
#     --beta1 0.9 \
#     --beta2 0.98 \
#     --wd 0.2 \
#     --eps 1e-6 \
#     \
#     --epochs $EPOCHS \
#     --batch-size $BATCH_SIZE \
#     --accum-freq $ACCUMULATION_STEPS \
#     --warmup 1000 \
#     \
#     --precision "amp" \
#     --workers 8 \
#     --video-decode-backend "imgs" \
#     \
#     --save-frequency 1 \
#     --log-every-n-steps 100 \
#     --report-to "tensorboard" \
#     --logs "$LOG_DIR" \
#     --name "languagebind_msrvtt_training" \
#     \
#     --resume "latest" \
#     --force-patch-dropout 0.5 \
#     --seed 42

# echo "训练完成！"
# echo "检查日志: $LOG_DIR"
# echo "TensorBoard: tensorboard --logdir=$LOG_DIR/languagebind_msrvtt_training/tensorboard"


#!/bin/bash

# 与LanguageBind源码完全兼容的训练配置
# 保持梯度累积，修复视频特征聚合问题

# 设置环境变量 - 与LanguageBind保持一致
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 数据路径配置
CACHE_DIR="/root/autodl-tmp/LanguageBind/cache_dir"
LANGUAGEBIND_MODEL="/root/autodl-tmp/LanguageBind/cache_dir/models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66"

# MSR-VTT数据集路径
MSRVTT_TRAIN_CSV="/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_train.9k.csv"
MSRVTT_DATA_JSON="/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/raw_data/MSRVTT_data.json"
MSRVTT_VIDEO_PATH="/root/autodl-tmp/LanguageBind/datasets/datasets--friedrichor--MSR-VTT/snapshots/c1af215a96934854f42683c19c51391aaee6f962/MSRVTT_Videos/video"

# 日志和输出目录
LOG_DIR="./logs/languagebind_compatible_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

# LanguageBind官方推荐配置 - 针对单GPU调整
NUM_GPUS=1

# 参考官方配置，针对48GB单GPU优化:
# 官方: 8 GPUs × batch_size=128 × accum_freq=1 = 1024 effective batch size
# 我们: 1 GPU × batch_size=64 × accum_freq=4 = 256 effective batch size (适中)
BATCH_SIZE=128             # 48GB显卡可以承受
ACCUMULATION_STEPS=8      # 保持梯度累积，与LanguageBind一致
LEARNING_RATE=1e-3        # 参考官方深度训练的lr
EPOCHS=10
NUM_FRAMES=8              # 视频帧数
MAX_WORDS=77              # 文本长度
IMAGE_RESOLUTION=224      # 图像分辨率

# 其他训练配置 - 与LanguageBind保持一致
WORKERS=10                # 官方使用10个workers
PRECISION="amp"           # 混合精度
WARMUP=2000              # 预热步数，官方使用2000
LR_COEF=1e-3             # coefficient learning rate

echo "=== LanguageBind兼容配置 ==="
echo "GPU数量: $NUM_GPUS"
echo "Batch Size: $BATCH_SIZE"
echo "梯度累积步数: $ACCUMULATION_STEPS"
echo "有效Batch Size: $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo "学习率: $LEARNING_RATE"
echo "预热步数: $WARMUP"
echo "============================="

# 文件检查
for file in "$MSRVTT_TRAIN_CSV" "$MSRVTT_DATA_JSON"; do
    if [[ ! -f "$file" ]]; then
        echo "错误: 文件不存在: $file"
        exit 1
    fi
done

for dir in "$MSRVTT_VIDEO_PATH" "$LANGUAGEBIND_MODEL"; do
    if [[ ! -d "$dir" ]]; then
        echo "错误: 目录不存在: $dir"
        exit 1
    fi
done

# 启动训练 - 完全遵循LanguageBind官方参数
python -m main \
    --model "languagebind_video" \
    --cache-dir "$CACHE_DIR" \
    \
    --msrvtt-train-csv "$MSRVTT_TRAIN_CSV" \
    --msrvtt-data-json "$MSRVTT_DATA_JSON" \
    --msrvtt-video-path "$MSRVTT_VIDEO_PATH" \
    \
    --train-data "msrvtt" \
    --do_train \
    --clip-type "vl_new" \
    --num-frames $NUM_FRAMES \
    --max_words $MAX_WORDS \
    --image-resolution $IMAGE_RESOLUTION \
    \
    --lock-text --lock-image \
    --lock-image-unlocked-groups 0 \
    --lock-text-unlocked-groups 0 \
    --lock-image-freeze-bn-stats \
    --lock-text-freeze-bn-stats \
    \
    --init-temp 0.07 --learn-temp \
    \
    --convert_to_lora \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    \
    --lr $LEARNING_RATE \
    --coef-lr $LR_COEF \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accum-freq $ACCUMULATION_STEPS \
    --warmup $WARMUP \
    --lr-scheduler "cosine" \
    \
    --precision $PRECISION \
    --workers $WORKERS \
    --video-decode-backend "imgs" \
    \
    --save-frequency 1 \
    --log-every-n-steps 20 \
    --report-to "tensorboard" \
    --logs "$LOG_DIR" \
    --name "languagebind_compatible_training" \
    \
    --resume "latest" \
    --force-patch-dropout 0.5 \
    --seed 42 \
    \
    --grad-clip-norm 1.0

echo "训练完成！"
echo "检查日志: $LOG_DIR"
echo "TensorBoard: tensorboard --logdir=$LOG_DIR/languagebind_compatible_training/tensorboard"

# 监控命令
echo ""
echo "=== 实时监控 ==="
echo "GPU状态: watch -n 1 nvidia-smi"
echo "训练日志: tail -f $LOG_DIR/languagebind_compatible_training/out.log"
echo "查看batch size警告: grep -i 'batch.*mismatch' $LOG_DIR/languagebind_compatible_training/out.log"