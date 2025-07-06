# #!/usr/bin/env python
# import glob
# import logging
# import os
# import re
# import subprocess
# import sys
# import random
# from datetime import datetime

# import numpy as np
# import torch
# from torch import optim
# from torch.cuda.amp import GradScaler
# from transformers import CLIPPreTrainedModel

# from a_cls.zeroshot_cls import evaluate_a_cls
# from al_ret.retrieval import evaluate_al_ret
# from i_cls.zeroshot_cls import evaluate_i_cls
# from d_cls.zeroshot_cls import evaluate_d_cls
# from t_cls.zeroshot_cls import evaluate_t_cls
# from v_cls.zeroshot_cls import evaluate_v_cls
# from vl_ret.retrieval import evaluate_vl_ret

# from model.process_clip import set_global_value, print_trainable_parameters

# try:
#     import wandb
# except ImportError:
#     wandb = None

# try:
#     import tensorboardX as tensorboard
# except ImportError:
#     tensorboard = None

# try:
#     import horovod.torch as hvd
# except ImportError:
#     hvd = None

# from data.build_datasets import get_data
# from open_clip import create_model_and_transforms, create_loss
# from training.distributed import is_master, init_distributed_device, broadcast_object
# from training.logger import setup_logging
# from training.params import parse_args
# from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
# from training.file_utils import pt_load, start_sync_process, remote_sync
# from train import train_one_epoch
# from model.build_model import create_vat_model

# LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

# # 原始ViT模型路径 - 注释掉，不再使用
# # MODEL_DICT = {"ViT-L-14": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
# #               "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
# # CHECKPOINT_DICT = {"ViT-L-14": "models--laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K/snapshots/84c9828e63dc9a9351d1fe637c346d4c1c4db341/pytorch_model.bin",
# #                    "ViT-H-14": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/94a64189c3535c1cb44acfcccd7b0908c1c8eb23/pytorch_model.bin"}

# # 修改为基于LanguageBind的模型路径
# LANGUAGEBIND_MODEL_PATH = "/root/autodl-tmp/LanguageBind/cache_dir/models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66"

# def random_seed(seed=42, rank=0):
#     torch.manual_seed(seed + rank)
#     np.random.seed(seed + rank)
#     random.seed(seed + rank)

# def natural_key(string_):
#     """See http://www.codinghorror.com/blog/archives/001018.html"""
#     return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

# def get_latest_checkpoint(path: str, remote: bool):
#     # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
#     if remote:
#         result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if result.returncode == 0:
#             remote_files = result.stdout.decode().split('\n')
#             files = [f.split(' ')[-1] for f in remote_files if len(f.split(' ')) > 3]
#             files = [f for f in files if f is not None and len(f) > 0]
#         else:
#             files = []
#     else:
#         files = glob.glob(os.path.join(path, '**/*.pt'), recursive=True)

#     if files:
#         files = sorted(files, key=natural_key)
#         return files[-1]
#     else:
#         return None

# def main(args):
#     args = parse_args(args)

#     # 添加distill参数初始化
#     if not hasattr(args, 'distill_model'):
#         args.distill_model = None
#     if not hasattr(args, 'distill_pretrained'):
#         args.distill_pretrained = None
    
#     # 计算distill标志
#     args.distill = args.distill_model is not None and args.distill_pretrained is not None
    
#     if torch.cuda.is_available():
#         # This enables tf32 on Ampere GPUs which is only 8% slower than
#         # float16 and almost as accurate as float32
#         # This was a default in pytorch until 1.12
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = False

#     # fully initialize distributed device environment
#     device = init_distributed_device(args)

#     # get the name of the experiments
#     if args.name is None:
#         # sanitize model name for filesystem / uri use, easier if we don't use / in name
#         model_name_safe = args.model.replace('/', '-')
#         date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
#         if args.distributed:
#             # sync date_str from master to all ranks
#             date_str = broadcast_object(args, date_str)
#         args.name = '-'.join([
#             date_str,
#             f"model_{model_name_safe}",
#             f"lr_{args.lr}",
#             f"b_{args.batch_size}",
#             f"j_{args.workers}",
#             f"p_{args.precision}",
#         ])

#     resume_latest = args.resume == 'latest'
#     log_base_path = os.path.join(args.logs, args.name)
#     args.log_path = None
#     if is_master(args, local=args.log_local):
#         os.makedirs(log_base_path, exist_ok=True)
#         log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
#         args.log_path = os.path.join(log_base_path, log_filename)
#         if os.path.exists(args.log_path) and not resume_latest:
#             print(
#                 "Error. Experiment already exists. Use --name {} to specify a new experiment."
#             )
#             return -1

#     # Setup text logger
#     args.log_level = logging.DEBUG if args.debug else logging.INFO
#     setup_logging(args.log_path, args.log_level)

#     # Setup wandb, tensorboard, checkpoint logging
#     args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
#     args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
#     args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
#     if is_master(args):
#         args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
#         for dirname in [args.tensorboard_path, args.checkpoint_path]:
#             if dirname:
#                 os.makedirs(dirname, exist_ok=True)
#     else:
#         args.tensorboard_path = ''

#     if resume_latest:
#         resume_from = None
#         checkpoint_path = args.checkpoint_path
#         # If using remote_sync, need to check the remote instead of the local checkpoints folder.
#         if args.remote_sync is not None:
#             checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
#             if args.save_most_recent:
#                 print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
#                 return -1
#             if args.remote_sync_protocol != 's3':
#                 print('Error. Sync protocol not supported when using resume latest.')
#                 return -1
#         if is_master(args):
#             # Checking for existing checkpoint via master rank only. It is possible for
#             # different rank processes to see different files if a shared file-system is under
#             # stress, however it's very difficult to fully work around such situations.
#             if args.save_most_recent:
#                 # if --save-most-recent flag is set, look for latest at a fixed filename
#                 resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
#                 if not os.path.exists(resume_from):
#                     # If no latest checkpoint has been saved yet, don't try to resume
#                     resume_from = None
#             else:
#                 # otherwise, list checkpoint dir contents and pick the newest checkpoint
#                 resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
#             if resume_from:
#                 logging.info(f'Found latest resume checkpoint at {resume_from}.')
#             else:
#                 logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
#         if args.distributed:
#             # sync found checkpoint path to all ranks
#             resume_from = broadcast_object(args, resume_from)
#         args.resume = resume_from

#     if args.copy_codebase:
#         copy_codebase(args)

#     # start the sync proces if remote-sync
#     remote_sync_process = None
#     if is_master(args) and args.remote_sync is not None:
#         # first make sure it works
#         result = remote_sync(
#             os.path.join(args.logs, args.name), 
#             os.path.join(args.remote_sync, args.name), 
#             args.remote_sync_protocol
#         )
#         if result:
#             logging.info('remote sync successful.')
#         else:
#             logging.info('Error: remote sync failed. Exiting.')
#             return -1
#         # if all looks good, start a process to do this every args.remote_sync_frequency seconds
#         remote_sync_process = start_sync_process(
#             args.remote_sync_frequency,
#             os.path.join(args.logs, args.name), 
#             os.path.join(args.remote_sync, args.name), 
#             args.remote_sync_protocol
#         )
#         remote_sync_process.start()

#     if args.precision == 'fp16':
#         logging.warning(
#             'It is recommended to use AMP mixed-precision instead of FP16. '
#             'FP16 support needs further verification and tuning, especially for train.'
#         )

#     if args.horovod:
#         logging.info(
#             f'Running in horovod mode with multiple processes / nodes. Device: {device}.'
#             f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
#         )
#     elif args.distributed:
#         logging.info(
#             f'Running in distributed mode with multiple processes. Device: {device}.'
#             f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
#         )
#     else:
#         logging.info(f'Running with a single process. Device {device}.')

#     random_seed(args.seed, 0)

#     # 修改模型创建部分 - 使用LanguageBind预训练模型
#     if args.model.lower() == 'languagebind_video':
#         model = create_languagebind_model(args)
#     else:
#         # 原始的模型创建逻辑保持不变
#         model = create_vat_model(args)

#     random_seed(args.seed, args.rank)

#     if args.trace:
#         model = trace_model(model, batch_size=args.batch_size, device=device)

#     if args.lock_image:
#         # lock image tower as done in CLIP
#         model.lock_image_tower(
#             unlocked_groups=args.lock_image_unlocked_groups,
#             freeze_bn_stats=args.lock_image_freeze_bn_stats)
#     if args.lock_text:
#         model.lock_text_tower(
#             unlocked_groups=args.lock_text_unlocked_groups,
#             freeze_bn_stats=args.lock_text_freeze_bn_stats)

#     if args.grad_checkpointing:
#         model.set_grad_checkpointing()

#     if is_master(args):
#         logging.info("Model:")
#         logging.info(f"{str(model)}")
#         logging.info("Params:")
#         params_file = os.path.join(args.logs, args.name, "params.txt")
#         with open(params_file, "w") as f:
#             for name in sorted(vars(args)):
#                 val = getattr(args, name)
#                 logging.info(f"  {name}: {val}")
#                 f.write(f"{name}: {val}\n")

#     if args.distributed and not args.horovod:
#         if args.use_bn_sync:
#             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         ddp_args = {}
#         if args.ddp_static_graph:
#             # this doesn't exist in older PyTorch, arg only added if enabled
#             ddp_args['static_graph'] = True
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

#     # create optimizer and scaler
#     optimizer = None
#     scaler = None

#     if args.train_data or args.dataset_type == "synthetic":
#         assert not args.trace, 'Cannot train with traced model'

#         exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
#         include = lambda n, p: not exclude(n, p)

#         named_parameters = list(model.named_parameters())
#         gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
#         rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

#         optimizer = optim.AdamW(
#             [
#                 {"params": gain_or_bias_params, "weight_decay": 0.},
#                 {"params": rest_params, "weight_decay": args.wd},
#             ],
#             lr=args.lr,
#             betas=(args.beta1, args.beta2),
#             eps=args.eps,
#         )
#         if args.horovod:
#             optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
#             hvd.broadcast_parameters(model.state_dict(), root_rank=0)
#             hvd.broadcast_optimizer_state(optimizer, root_rank=0)

#         scaler = GradScaler() if args.precision == "amp" else None

#     # optionally resume from a checkpoint
#     start_epoch = 0
#     if args.resume is not None:
#         checkpoint = pt_load(args.resume, map_location='cpu')
#         if 'epoch' in checkpoint:
#             # resuming a train checkpoint w/ epoch and optimizer state
#             start_epoch = checkpoint["epoch"]
#             sd = checkpoint["state_dict"]
#             if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
#                 sd = {k[len('module.'):]: v for k, v in sd.items()}
#             model.load_state_dict(sd)
#             if optimizer is not None:
#                 optimizer.load_state_dict(checkpoint["optimizer"])
#             if scaler is not None and 'scaler' in checkpoint:
#                 scaler.load_state_dict(checkpoint['scaler'])
#             logging.info(f"=> resumed checkpoint '{args.resume}' (epoch {start_epoch})")
#         else:
#             # loading a bare (model only) checkpoint for fine-tune or evaluation
#             model.load_state_dict(checkpoint)
#             logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

#     # initialize datasets
#     data = get_data(args, epoch=start_epoch)

#     assert len(data), 'At least one train or eval dataset must be specified.'

#     # create scheduler if train
#     scheduler = None
#     if 'train' in data and optimizer is not None:
#         total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
#         if args.lr_scheduler == "cosine":
#             scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
#         elif args.lr_scheduler == "const":
#             scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
#         elif args.lr_scheduler == "const-cooldown":
#             assert args.epochs_cooldown is not None,\
#                 "Please specify the number of cooldown epochs for this lr schedule."
#             cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
#             scheduler = const_lr_cooldown(
#                 optimizer, args.lr, args.warmup, total_steps,
#                 cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
#         else:
#             logging.error(
#                 f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
#             exit(1)

#     # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
#     args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
#     writer = None
#     if args.save_logs and args.tensorboard:
#         assert tensorboard is not None, "Please install tensorboard."
#         writer = tensorboard.SummaryWriter(args.tensorboard_path)

#     if args.wandb and is_master(args):
#         assert wandb is not None, 'Please install wandb.'
#         logging.debug('Starting wandb.')
#         args.train_sz = data["train"].dataloader.num_samples
#         if args.val_data is not None:
#             args.val_sz = data["val"].dataloader.num_samples
#         # you will have to configure this for your project!
#         wandb.init(
#             project=args.wandb_project_name,
#             name=args.name,
#             id=args.name,
#             notes=args.wandb_notes,
#             tags=[],
#             resume='auto' if args.resume == "latest" else None,
#             config=vars(args),
#         )
#         if args.debug:
#             wandb.watch(model, log='all')

#     # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
#     # For compatibility, we save state_dict() of the original model, which shares the
#     # weights without the prefix.
#     original_model = model
#     if args.torchcompile:
#         logging.info('Compiling model...')
#         model = torch.compile(original_model)

#     if 'train' not in data:
#         # If using int8, convert to inference mode.
#         if args.use_bnb_linear is not None:
#             from open_clip.utils import convert_int8_model_to_inference_mode
#             convert_int8_model_to_inference_mode(model)
#         evaluate(model, data, start_epoch, args, writer)
#         return

#     loss = create_loss(args)

#     for epoch in range(start_epoch, args.epochs):
#         if is_master(args):
#             logging.info(f'Start epoch {epoch}')

#         train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model=None, args=args, tb_writer=writer)
#         completed_epoch = epoch + 1

#         if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
#             evaluate(model, data, completed_epoch, args, writer)

#         # Saving checkpoints.
#         if args.save_logs:
#             checkpoint_dict = {
#                 "epoch": completed_epoch,
#                 "name": args.name,
#                 "state_dict": original_model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#             }
#             if scaler is not None:
#                 checkpoint_dict["scaler"] = scaler.state_dict()

#             if completed_epoch == args.epochs or (
#                 args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
#             ):
#                 torch.save(
#                     checkpoint_dict,
#                     os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
#                 )
#             if args.delete_previous_checkpoint:
#                 previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
#                 if os.path.exists(previous_checkpoint):
#                     os.remove(previous_checkpoint)

#             if args.save_most_recent:
#                 # try not to corrupt the latest checkpoint if save fails
#                 tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
#                 latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
#                 torch.save(checkpoint_dict, tmp_save_path)
#                 os.replace(tmp_save_path, latest_save_path)

#     if args.wandb and is_master(args):
#         wandb.finish()

#     # run a final sync.
#     if remote_sync_process is not None:
#         logging.info('Final remote sync.')
#         remote_sync_process.terminate()
#         result = remote_sync(
#             os.path.join(args.logs, args.name), 
#             os.path.join(args.remote_sync, args.name), 
#             args.remote_sync_protocol
#         )
#         if result:
#             logging.info('Final remote sync successful.')
#         else:
#             logging.info('Final remote sync failed.')

# def create_languagebind_model(args):
#     """创建基于LanguageBind的模型"""
#     from languagebind_adapter import create_languagebind_model_with_adapter, apply_lora_to_languagebind
    
#     # 加载LanguageBind预训练模型并包装成适配器
#     model = create_languagebind_model_with_adapter(
#         LANGUAGEBIND_MODEL_PATH, 
#         cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else './cache_dir'
#     )
    
#     # 设置温度参数
#     if hasattr(args, 'init_temp') and args.init_temp != 0:
#         with torch.no_grad():
#             model.logit_scale.fill_(np.log(1 / float(args.init_temp)))
#         if is_master(args):
#             logging.info(f'Reset logit scale to {args.init_temp} (log-scale).')
    
#     # 转换为LoRA如果需要
#     if hasattr(args, 'convert_to_lora') and args.convert_to_lora:
#         model = apply_lora_to_languagebind(model, args)
#         if is_master(args):
#             logging.info(f"Successfully convert LanguageBind model to lora style.")
    
#     return model

# def evaluate(model, data, epoch, args, writer):
#     """评估函数"""
#     metrics = {}
#     if not is_master(args):
#         return metrics
    
#     device = torch.device(args.device)
#     model.eval()

#     # 这里可以添加具体的评估逻辑
#     # 例如video-text retrieval评估
#     if 'vl_ret' in data:
#         vl_ret_metrics = evaluate_vl_ret(model, data['vl_ret'].dataloader, args)
#         metrics.update(vl_ret_metrics)
    
#     if writer is not None:
#         for name, val in metrics.items():
#             writer.add_scalar(f"val/{name}", val, epoch)
    
#     return metrics

# def copy_codebase(args):
#     """复制代码库到输出目录"""
#     # Implementation for copying codebase
#     pass

# def trace_model(model, batch_size, device):
#     """模型追踪"""
#     # Implementation for model tracing
#     return model

# def convert_model_to_lora(args, model):
#     """将模型转换为LoRA格式"""
#     # 这里需要根据具体的LoRA实现来修改
#     # 可能需要安装peft库或其他LoRA实现
#     pass

# if __name__ == "__main__":
#     main(sys.argv[1:])



#!/usr/bin/env python
"""
修复后的main.py - 解决LanguageBind模型创建和训练问题
"""
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from transformers import CLIPPreTrainedModel

from a_cls.zeroshot_cls import evaluate_a_cls
from al_ret.retrieval import evaluate_al_ret
from i_cls.zeroshot_cls import evaluate_i_cls
from d_cls.zeroshot_cls import evaluate_d_cls
from t_cls.zeroshot_cls import evaluate_t_cls
from v_cls.zeroshot_cls import evaluate_v_cls
from vl_ret.retrieval import evaluate_vl_ret

from model.process_clip import set_global_value, print_trainable_parameters

try:
    import wandb
except ImportError:
    wandb = None

try:
    import tensorboardX as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from data.build_datasets import get_data
from open_clip import create_model_and_transforms, create_loss
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.file_utils import pt_load, start_sync_process, remote_sync
from train import train_one_epoch
from model.build_model import create_vat_model

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

# 修改为基于LanguageBind的模型路径
LANGUAGEBIND_MODEL_PATH = "/root/autodl-tmp/LanguageBind/cache_dir/models--LanguageBind--LanguageBind_Video_FT/snapshots/13f52c20ce666a7d017bcd00522039f4ab034a66"
LANGUAGEBIND_TOKENIZER_PATH = "/root/autodl-tmp/LanguageBind/cache_dir/tokenizer_cache_dir/models--lb203--LanguageBind_Image/snapshots/d8c2e37b439f4fc47c649dc8b90cdcd3a4e0c80e"

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            remote_files = result.stdout.decode().split('\n')
            files = [f.split(' ')[-1] for f in remote_files if len(f.split(' ')) > 3]
            files = [f for f in files if f is not None and len(f) > 0]
        else:
            files = []
    else:
        files = glob.glob(os.path.join(path, '**/*.pt'), recursive=True)

    if files:
        files = sorted(files, key=natural_key)
        return files[-1]
    else:
        return None

def main(args):
    args = parse_args(args)

    # 添加distill参数初始化
    if not hasattr(args, 'distill_model'):
        args.distill_model = None
    if not hasattr(args, 'distill_pretrained'):
        args.distill_pretrained = None
    
    # 计算distill标志
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.'
        )

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
        )
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
        )
    else:
        logging.info(f'Running with a single process. Device {device}.')

    random_seed(args.seed, 0)

    # 修改模型创建部分 - 使用LanguageBind预训练模型
    if args.model.lower() == 'languagebind_video':
        model = create_languagebind_model(args)
    else:
        # 原始的模型创建逻辑保持不变
        model = create_vat_model(args)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as done in CLIP
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_groups=args.lock_text_unlocked_groups,
            freeze_bn_stats=args.lock_text_freeze_bn_stats)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resumed checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, epoch=start_epoch)

    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        evaluate(model, data, start_epoch, args, writer)
        return

    # 创建适合LanguageBind的损失函数
    if args.model.lower() == 'languagebind_video':
        loss = create_languagebind_loss(args)
    else:
        loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model=None, args=args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')

def create_languagebind_model(args):
    """创建基于LanguageBind的模型 - 修复所有问题"""
    # 导入修复后的适配器
    from languagebind_adapter import create_languagebind_model_with_adapter, apply_lora_to_languagebind
    
    # 加载LanguageBind预训练模型并包装成适配器
    model = create_languagebind_model_with_adapter(
        LANGUAGEBIND_MODEL_PATH, 
        cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else './cache_dir'
    )
    
    # 确保模型在正确的设备上且使用float32
    device = torch.device(args.device)
    model = model.to(device, dtype=torch.float32)
    
    # 设置温度参数
    if hasattr(args, 'init_temp') and args.init_temp != 0:
        with torch.no_grad():
            model.logit_scale.fill_(np.log(1 / float(args.init_temp)))
        if is_master(args):
            logging.info(f'Reset logit scale to {args.init_temp} (log-scale).')
    
    # 转换为LoRA如果需要
    if hasattr(args, 'convert_to_lora') and args.convert_to_lora:
        model = apply_lora_to_languagebind(model, args)
        if is_master(args):
            logging.info(f"Successfully convert LanguageBind model to lora style.")
    
    # 添加tokenizer到args中，用于数据处理
    try:
        from languagebind import LanguageBindVideoTokenizer
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(
            LANGUAGEBIND_TOKENIZER_PATH, 
            cache_dir=args.cache_dir if hasattr(args, 'cache_dir') else './cache_dir/tokenizer_cache_dir'
        )
        args.tokenizer = tokenizer
        if is_master(args):
            logging.info("LanguageBind tokenizer loaded successfully.")
    except Exception as e:
        logging.warning(f"Failed to load LanguageBind tokenizer: {e}")
        # 使用简单的tokenizer作为fallback
        args.tokenizer = None
    
    return model

def create_languagebind_loss(args):
    """创建适合LanguageBind的损失函数"""
    from fixed_contrastive_loss import create_languagebind_loss
    return create_languagebind_loss(args)

def evaluate(model, data, epoch, args, writer):
    """评估函数"""
    metrics = {}
    if not is_master(args):
        return metrics
    
    device = torch.device(args.device)
    model.eval()

    # 这里可以添加具体的评估逻辑
    # 例如video-text retrieval评估
    if 'vl_ret' in data:
        try:
            vl_ret_metrics = evaluate_vl_ret(model, data['vl_ret'].dataloader, args)
            metrics.update(vl_ret_metrics)
        except Exception as e:
            logging.warning(f"VL retrieval evaluation failed: {e}")
    
    if writer is not None:
        for name, val in metrics.items():
            writer.add_scalar(f"val/{name}", val, epoch)
    
    return metrics

def copy_codebase(args):
    """复制代码库到输出目录"""
    # Implementation for copying codebase
    pass

def trace_model(model, batch_size, device):
    """模型追踪"""
    # Implementation for model tracing
    return model

if __name__ == "__main__":
    main(sys.argv[1:])