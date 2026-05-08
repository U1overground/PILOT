# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import logging
import os
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate

from . import performance



def train(config, epoch, num_epoch, epoch_iters, base_lr,
          num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader, 0):
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        

        losses, _, acc, loss_list = model(images, labels, bd_gts)
        loss = losses.mean()
        acc  = acc.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}, BCE loss: {:.6f}, SB loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], ave_loss.average(),
                      ave_acc.average(), avg_sem_loss.average(), avg_bce_loss.average(),ave_loss.average()-avg_sem_loss.average()-avg_bce_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    
    # Original confusion matrix
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))

    # --- NEW ---
    # Initialize a list of metric calculators for each model output
    metric1_list = [performance.SegmentationMetric(config) for _ in range(nums)]
    metric2_list = [performance.SegmentationMetrics2(config) for _ in range(nums)]
    metric3_list = [performance.SegmentationMetrics3(config) for _ in range(nums)]
    # --- END NEW ---

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
                
            for i, x in enumerate(pred):
                # Original interpolation
                x_interp = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                # Original confusion matrix update
                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x_interp,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

                # --- NEW ---
                # Update professor's metrics
                # Metric 1 (handles its own interpolation, pass un-interpolated 'x')
                metric1_list[i].update(x, label) 
                
                # Metrics 2 and 3 (need interpolated 'x_interp')
                metric2_list[i].update(x_interp, label)
                metric3_list[i].update(x_interp, label)
                # --- END NEW ---


            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    # --- UPDATED LOGGING ---
    mean_IoU = 0 # To store the mIoU of the last output, for the return value
    IoU_array = np.zeros(config.DATASET.NUM_CLASSES) # for the return value

    for i in range(nums):
        # Original calculation
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        # Get results from professor's metrics
        pixAcc1, mIoU1 = metric1_list[i].get()
        pixAcc2, mIoU2 = metric2_list[i].get()
        pixAcc3, mIoU3 = metric3_list[i].get()
        
        # Log all 4 mIoUs
        logging.info('--- Validation Output {} ---'.format(i))
        logging.info('  Original mIoU: {:.6f}'.format(mean_IoU))
        logging.info('  Metric 1 mIoU: {:.6f} (PixAcc: {:.6f})'.format(mIoU1, pixAcc1))
        logging.info('  Metric 2 mIoU: {:.6f} (PixAcc: {:.6f})'.format(mIoU2, pixAcc2))
        logging.info('  Metric 3 mIoU: {:.6f} (PixAcc: {:.6f})'.format(mIoU3, pixAcc3))
        logging.info('  Original IoU Array: {}'.format(np.array2string(IoU_array, formatter={'float_kind':lambda x: "%.4f" % x})))
        logging.info('---------------------------------')
    # --- END UPDATED LOGGING ---

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    # Write the original mIoU (from the last output) to tensorboard
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps) 
    writer_dict['valid_global_steps'] = global_steps + 1
    
    # Return values are unchanged (uses the last output's mIoU)
    return ave_loss.average(), mean_IoU, IoU_array


def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    
    # --- NEW ---
    # You could add the professor's metrics here too, if you want.
    # For now, I will leave testval and test as-is, following your
    # request to just modify the validation part.
    # metric1 = performance.SegmentationMetric(config)
    # metric2 = performance.SegmentationMetrics2(config)
    # metric3 = performance.SegmentationMetrics3(config)
    # --- END NEW ---

    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            # --- NEW ---
            # If you add the metrics, you would update them here:
            # metric1.update(pred, label) # Note: pred is already interpolated here
            # metric2.update(pred, label)
            # metric3.update(pred, label)
            # --- END NEW ---

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    # --- NEW ---
    # If you add the metrics, you would log them here:
    # pixAcc1, mIoU1 = metric1.get()
    # pixAcc2, mIoU2 = metric2.get()
    # pixAcc3, mIoU3 = metric3.get()
    # logging.info('Metric 1 mIoU: {:.6f}'.format(mIoU1))
    # logging.info('Metric 2 mIoU: {:.6f}'.format(mIoU2))
    # logging.info('Metric 3 mIoU: {:.6f}'.format(mIoU3))
    # --- END NEW ---

    return mean_IoU, IoU_array, pixel_acc, mean_acc


def test(config, test_dataset, testloader, model,
         sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)