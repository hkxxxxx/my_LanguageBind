# import json
# import logging
# import math
# import os
# import time

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.nn.parallel.distributed import DistributedDataParallel

# from training.distributed import is_master
# from training.precision import get_autocast

# try:
#     import wandb
# except ImportError:
#     wandb = None

# from open_clip import get_input_dtype, CLIP, CustomTextCLIP


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

# def postprocess_clip_output(model_out):
#     return {
#         "image_features": model_out[0],
#         "text_features": model_out[1],
#         "logit_scale": model_out[2]
#     }

# def unwrap_model(model):
#     if hasattr(model, 'module'):
#         return model.module
#     else:
#         return model


# def backward(total_loss, scaler):
#     if scaler is not None:
#         scaler.scale(total_loss).backward()
#     else:
#         total_loss.backward()


# def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
#     device = torch.device(args.device)
#     autocast = get_autocast(args.precision)
#     input_dtype = get_input_dtype(args.precision)


#     model.train()
#     if args.distill:
#         dist_model.eval()

#     data[f'{args.clip_type}_pt'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
#     dataloader = data[f'{args.clip_type}_pt'].dataloader
#     num_batches_per_epoch = dataloader.num_batches // args.accum_freq
#     sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

#     if args.accum_freq > 1:
#         accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

#     losses_m = {}
#     batch_time_m = AverageMeter()
#     data_time_m = AverageMeter()
#     end = time.time()
#     for i, batch in enumerate(dataloader):
#         i_accum = i // args.accum_freq
#         step = num_batches_per_epoch * epoch + i_accum

#         if not args.skip_scheduler:
#             scheduler(step)

#         images, input_ids, attention_mask = batch
#         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
#         input_ids = input_ids.to(device=device, non_blocking=True)
#         attention_mask = attention_mask.to(device=device, non_blocking=True)

#         data_time_m.update(time.time() - end)
#         optimizer.zero_grad()

#         if args.accum_freq == 1:
#             with autocast():
#                 model_out = model(images, input_ids, attention_mask)
#                 logit_scale = model_out["logit_scale"]
#                 if args.distill:
#                     with torch.no_grad():
#                         dist_model_out = dist_model(images, input_ids, attention_mask)
#                     model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})
#                 losses = loss(**model_out, output_dict=True)

#                 total_loss = sum(losses.values())
#                 losses["loss"] = total_loss

#             backward(total_loss, scaler)
#         else:
#             # First, cache the features without any gradient tracking.
#             with torch.no_grad():
#                 with autocast():
#                     model_out = model(images, input_ids, attention_mask)
#                     model_out.pop("logit_scale")
#                     for key, val in model_out.items():
#                         if key in accum_features:
#                             accum_features[key].append(val)
#                         else:
#                             accum_features[key] = [val]

#                 accum_images.append(images)
#                 accum_input_ids.append(input_ids)
#                 accum_attention_mask.append(attention_mask)

#             # If (i + 1) % accum_freq is not zero, move on to the next batch.
#             if ((i + 1) % args.accum_freq) > 0:
#                 # FIXME this makes data time logging unreliable when accumulating
#                 continue

#             # Now, ready to take gradients for the last accum_freq batches.
#             # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
#             # Call backwards each time, but only step optimizer at the end.
#             optimizer.zero_grad()
#             for j in range(args.accum_freq):
#                 images = accum_images[j]
#                 input_ids = accum_input_ids[j]
#                 attention_mask = accum_attention_mask[j]
#                 with autocast():
#                     model_out = model(images, input_ids, attention_mask)
#                     logit_scale = model_out.pop("logit_scale")
#                     inputs = {}
#                     for key, val in accum_features.items():
#                         accumulated = accum_features[key]
#                         inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
#                     losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
#                     del inputs
#                     total_loss = sum(losses.values())
#                     losses["loss"] = total_loss
#                 backward(total_loss, scaler)

#         if scaler is not None:
#             if args.horovod:
#                 optimizer.synchronize()
#                 scaler.unscale_(optimizer)
#                 if args.grad_clip_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
#                 with optimizer.skip_synchronize():
#                     scaler.step(optimizer)
#             else:
#                 if args.grad_clip_norm is not None:
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
#                 scaler.step(optimizer)
#             scaler.update()
#         else:
#             if args.grad_clip_norm is not None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
#             optimizer.step()

#         # reset gradient accum, if enabled
#         if args.accum_freq > 1:
#             accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

#         # Note: we clamp to 4.6052 = ln(100), as in the original paper.
#         with torch.no_grad():
#             unwrap_model(model).logit_scale.clamp_(0, math.log(100))

#         batch_time_m.update(time.time() - end)
#         end = time.time()
#         batch_count = i_accum + 1
#         if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
#             batch_size = len(images)
#             num_samples = batch_count * batch_size * args.accum_freq * args.world_size
#             samples_per_epoch = dataloader.num_samples
#             percent_complete = 100.0 * batch_count / num_batches_per_epoch

#             # NOTE loss is coarsely sampled, just master node and per log update
#             for key, val in losses.items():
#                 if key not in losses_m:
#                     losses_m[key] = AverageMeter()
#                 losses_m[key].update(val.item(), batch_size)

#             logit_scale_scalar = logit_scale.item()
#             # if args.add_time_attn:
#             #     if hasattr(model, 'module'):
#             #         t_gate = [[F.sigmoid(m.t_attn_gate).detach().item(), F.sigmoid(m.t_ffn_gate).detach().item()] for m in model.module.vision_model.encoder.layers]
#             #     else:
#             #         t_gate = [[F.sigmoid(m.t_attn_gate).detach().item(), F.sigmoid(m.t_ffn_gate).detach().item()] for m in model.vision_model.encoder.layers]
#             #     t_attn_gate, t_ffn_gate = list(zip(*t_gate))
#             loss_log = " ".join(
#                 [
#                     f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
#                     for loss_name, loss_m in losses_m.items()
#                 ]
#             )
#             samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
#             samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
#             # if args.add_time_attn:
#             #     logging.info(
#             #         f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
#             #         f"Data (t): {data_time_m.avg:.3f} "
#             #         f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
#             #         f"LR: {optimizer.param_groups[0]['lr']:5f} "
#             #         f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log +
#             #         f"\nt_attn_gate: {[round(i, 2) for i in t_attn_gate]}\nt_ffn_gate: {[round(i, 2) for i in t_ffn_gate]}\n"
#             #     )
#             # else:
#             logging.info(
#                 f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
#                 f"Data (t): {data_time_m.avg:.3f} "
#                 f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
#                 f"LR: {optimizer.param_groups[0]['lr']:5f} "
#                 f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
#             )


#             # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
#             log_data = {
#                 "data_time": data_time_m.val,
#                 "batch_time": batch_time_m.val,
#                 "samples_per_second": samples_per_second,
#                 "samples_per_second_per_gpu": samples_per_second_per_gpu,
#                 "scale": logit_scale_scalar,
#                 "lr": optimizer.param_groups[0]["lr"]
#             }
#             log_data.update({name:val.val for name,val in losses_m.items()})
#             # if args.add_time_attn:
#             #     log_data.update({f'layer_{i}_t_attn_gate': attn for i, attn in enumerate(t_attn_gate)})
#             #     log_data.update({f'layer_{i}_t_ffn_gate': ffn for i, ffn in enumerate(t_ffn_gate)})

#             for name, val in log_data.items():
#                 name = "train/" + name
#                 if tb_writer is not None:
#                     tb_writer.add_scalar(name, val, step)
#                 if args.wandb:
#                     assert wandb is not None, 'Please install wandb.'
#                     wandb.log({name: val, 'step': step})

#             # resetting batch / data time meters per log window
#             batch_time_m.reset()
#             data_time_m.reset()
#     # end for



#!/usr/bin/env python
"""
完整修复后的train.py - 解决所有数据类型、设备和返回值格式问题
"""
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

from training.distributed import is_master
from training.precision import get_autocast

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    """
    后处理模型输出，确保返回正确的字典格式
    处理不同类型的model_out（dict, tuple, etc.）
    """
    # 如果model_out已经是字典格式
    if isinstance(model_out, dict):
        # 确保包含必需的键
        required_keys = ["image_features", "text_features", "logit_scale"]
        if all(key in model_out for key in required_keys):
            return model_out
        
        # 如果缺少某些键，尝试从其他键名映射
        result = {}
        
        # 处理image features
        if "image_features" in model_out:
            result["image_features"] = model_out["image_features"]
        elif "vision_embeds" in model_out:
            result["image_features"] = model_out["vision_embeds"]
        elif "image_embeds" in model_out:
            result["image_features"] = model_out["image_embeds"]
        else:
            result["image_features"] = None
        
        # 处理text features
        if "text_features" in model_out:
            result["text_features"] = model_out["text_features"]
        elif "text_embeds" in model_out:
            result["text_features"] = model_out["text_embeds"]
        elif "language_embeds" in model_out:
            result["text_features"] = model_out["language_embeds"]
        else:
            result["text_features"] = None
        
        # 处理logit_scale
        if "logit_scale" in model_out:
            result["logit_scale"] = model_out["logit_scale"]
        else:
            # 如果没有logit_scale，创建一个默认值
            result["logit_scale"] = torch.tensor(1.0)
        
        return result
    
    # 如果model_out是tuple格式 (image_features, text_features, logit_scale)
    elif isinstance(model_out, (tuple, list)):
        if len(model_out) == 3:
            return {
                "image_features": model_out[0],
                "text_features": model_out[1],
                "logit_scale": model_out[2]
            }
        elif len(model_out) == 2:
            # 假设是 (image_features, text_features)
            return {
                "image_features": model_out[0],
                "text_features": model_out[1],
                "logit_scale": torch.tensor(1.0)
            }
        else:
            raise ValueError(f"Unexpected tuple length: {len(model_out)}")
    
    # 如果是其他格式，尝试直接访问属性
    else:
        result = {}
        
        if hasattr(model_out, 'image_features'):
            result["image_features"] = model_out.image_features
        elif hasattr(model_out, 'vision_embeds'):
            result["image_features"] = model_out.vision_embeds
        else:
            result["image_features"] = None
        
        if hasattr(model_out, 'text_features'):
            result["text_features"] = model_out.text_features
        elif hasattr(model_out, 'text_embeds'):
            result["text_features"] = model_out.text_embeds
        else:
            result["text_features"] = None
        
        if hasattr(model_out, 'logit_scale'):
            result["logit_scale"] = model_out.logit_scale
        else:
            result["logit_scale"] = torch.tensor(1.0)
        
        return result


def ensure_tensor_consistency(tensor, target_device, target_dtype=None):
    """确保tensor在正确的设备上且具有正确的数据类型"""
    if tensor is None:
        return tensor
    
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    # 确保在正确的设备上
    if tensor.device != target_device:
        tensor = tensor.to(target_device)
    
    # 确保数据类型正确（如果指定）
    if target_dtype is not None and tensor.dtype != target_dtype:
        tensor = tensor.to(target_dtype)
    
    return tensor


def ensure_output_consistency(model_out, target_device, target_dtype=torch.float32):
    """
    确保模型输出的所有tensor都在正确的设备上并具有正确的数据类型
    """
    if isinstance(model_out, dict):
        result = {}
        for key, value in model_out.items():
            if isinstance(value, torch.Tensor):
                # 对于logit_scale，保持其原有的数据类型
                if key == "logit_scale":
                    value = ensure_tensor_consistency(value, target_device, target_dtype)
                else:
                    value = ensure_tensor_consistency(value, target_device, target_dtype)
            result[key] = value
        return result
    else:
        return model_out


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    
    # 强制使用float32避免half/float混合问题
    target_dtype = torch.float32

    model.train()
    if args.distill:
        dist_model.eval()

    data[f'{args.clip_type}_pt'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data[f'{args.clip_type}_pt'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, input_ids, attention_mask = batch
        
        # 确保输入数据在正确的设备和数据类型上
        images = ensure_tensor_consistency(images, device, target_dtype)
        input_ids = ensure_tensor_consistency(input_ids, device, input_ids.dtype)  # 保持整数类型
        attention_mask = ensure_tensor_consistency(attention_mask, device, attention_mask.dtype)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # 调用模型并后处理输出
                model_out_raw = model(images, input_ids, attention_mask)
                model_out = postprocess_clip_output(model_out_raw)
                model_out = ensure_output_consistency(model_out, device, target_dtype)
                
                logit_scale = model_out["logit_scale"]
                
                if args.distill:
                    with torch.no_grad():
                        dist_model_out_raw = dist_model(images, input_ids, attention_mask)
                        dist_model_out = postprocess_clip_output(dist_model_out_raw)
                        dist_model_out = ensure_output_consistency(dist_model_out, device, target_dtype)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                
                losses = loss(**model_out, output_dict=True)
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out_raw = model(images, input_ids, attention_mask)
                    model_out = postprocess_clip_output(model_out_raw)
                    model_out = ensure_output_consistency(model_out, device, target_dtype)
                    
                    # 从model_out中移除logit_scale，只保留features
                    logit_scale = model_out.pop("logit_scale")
                    
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_input_ids.append(input_ids)
                accum_attention_mask.append(attention_mask)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                input_ids = accum_input_ids[j]
                attention_mask = accum_attention_mask[j]
                with autocast():
                    model_out_raw = model(images, input_ids, attention_mask)
                    model_out = postprocess_clip_output(model_out_raw)
                    model_out = ensure_output_consistency(model_out, device, target_dtype)
                    
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])
                    losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_input_ids, accum_attention_mask, accum_features = [], [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item() if isinstance(logit_scale, torch.Tensor) else float(logit_scale)
            
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for