import argparse
import csv
import os
import shutil
import time
import glob
import numpy as np
from collections import OrderedDict
from datetime import datetime

from dataset import CDiscountDataset, dataset_scan
from models import model_factory, dense_sparse_dense, multi_target
from lr_scheduler import ReduceLROnPlateau
from utils import AverageMeter, get_outdir
from optim import nadam

import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax", "avgmaxc" (default: "avg")')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='Train/valid fold #. (default: 0')
parser.add_argument('--labels', default='all', type=str, metavar='NAME',
                    help='Label set (default: "all"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
parser.add_argument('--multi-target', '--mt', type=int, default=0, metavar='N',
                    help='multi-target classifier count (default: 0)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-s', '--initial-batch-size', type=int, default=0, metavar='N',
                    help='initial input batch size for training (default: 0)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--ft-epochs', type=float, default=0., metavar='LR',
                    help='Number of finetuning epochs (final layer only)')
parser.add_argument('--ft-opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--ft-lr', type=float, default=0.0001, metavar='N',
                    help='Finetune learning rates.')
parser.add_argument('--drop', type=float, default=0.0, metavar='DROP',
                    help='Dropout rate (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='path to init checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='enable sparsity masking for DSD training')
parser.add_argument('--class-weights', action='store_true', default=False,
                    help='Use class weights for specified labels as loss penalty')


def main():
    args = parser.parse_args()

    if args.output:
        output_base = args.output
    else:
        output_base = './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(args.img_size),
        'f'+str(args.fold)])
    output_dir = get_outdir(output_base, 'train', exp_name)

    train_input_root = os.path.join(args.data)
    batch_size = args.batch_size
    num_epochs = args.epochs
    img_size = (args.img_size, args.img_size)
    num_classes = 5270  # FIXME

    torch.manual_seed(args.seed)

    # FIXME hackish, really don't care about original class count here, need a way to not require that
    if 'inception' in args.model:
        num_classes_init = 1001
        normalize = 'le'
    else:
        num_classes_init = 1000
        normalize = 'torchvision'

    if '5k' in args.initial_checkpoint:
        num_classes_init = 4786

    model = model_factory.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes_init,
        drop_rate=args.drop,
        global_pool=args.gp,
        checkpoint_path=args.initial_checkpoint)

    if args.multi_target:
        print('Creating multi-target model')
        if args.multi_target == 2:
            model = multi_target.MultiTargetModel(model, [5270, 483], cascade=True)
        elif args.multi_target == 3:
            model = multi_target.MultiTargetModel(model, [5270, 483, 49])
        else:
            assert False, 'Invalid target count'
    else:
        model.reset_classifier(num_classes=num_classes)

    bootstrap = dataset_scan(
        train_input_root,
        metadata_file='prod_metadata-l2_6.csv',
        category_file='category_map-l2_6.csv'
    )

    dataset_train = CDiscountDataset(
        bootstrap=bootstrap[0],
        is_training=True,
        img_size=img_size,
        fold=args.fold,
        normalize=normalize,
        multi_target=args.multi_target)

    #sampler = WeightedRandomOverSampler(dataset_train.get_sample_weights
    if args.initial_batch_size:
        batch_size = adjust_batch_size(
            epoch=0, initial_bs=args.initial_batch_size, target_bs=args.batch_size)
        print('Setting batch-size to %d' % batch_size)
    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        #sampler=sampler,
        num_workers=args.workers
    )

    dataset_eval = CDiscountDataset(
        bootstrap=bootstrap[1],
        is_training=False,
        img_size=img_size,
        test_aug=args.tta,
        fold=args.fold,
        normalize=normalize,
    )

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=4 * args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=args.workers
    )

    train_loss_fn = validate_loss_fn = torch.nn.CrossEntropyLoss()
    if args.multi_target:
        print('Creating multi-target loss')
        if args.multi_target == 3:
            mt_scales = [0.8, 0.1, 0.1]
        else:
            mt_scales = [0.9, 0.1]
        train_loss_fn = multi_target.MultiTargetLoss(fixed_scales=mt_scales, loss_fn=train_loss_fn)
        #train_loss_fn = multi_target.MultiTargetLoss(learnable=3, loss_fn=train_loss_fn)
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = validate_loss_fn.cuda()

    opt_params = list(model.parameters())
    if args.multi_target and train_loss_fn.learnable:
        opt_params += list(train_loss_fn.parameters())
    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            opt_params, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            opt_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'nadam':
        optimizer = nadam.Nadam(
            opt_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            opt_params, lr=args.lr, weight_decay=args.weight_decay, eps=args.opt_eps)
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            opt_params, lr=args.lr, alpha=0.9, eps=args.opt_eps,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"
    del opt_params

    if not args.decay_epochs:
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=8)
    else:
        lr_scheduler = None

    # optionally resume from a checkpoint
    start_epoch = 0 if args.start_epoch is None else args.start_epoch
    sparse_checkpoint = False
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                if 'sparse' in checkpoint and checkpoint['sparse']:
                    sparse_checkpoint = True
                    print("Loading sparse model")
                    # ensure sparsity_masks exist in model definition before loading state
                    dense_sparse_dense.sparsify(model, sparsity=0.)
                if 'args' in checkpoint:
                    print(checkpoint['args'])
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('module'):
                        name = k[7:] # remove `module.`
                    else:
                        name = k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'loss' in checkpoint:
                    train_loss_fn.load_state_dict(checkpoint['loss'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                start_epoch = checkpoint['epoch'] if args.start_epoch is None else args.start_epoch
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(1)
    if not sparse_checkpoint and args.sparse:
        print("Sparsifying loaded model")
        dense_sparse_dense.sparsify(model, sparsity=0.5)
    elif sparse_checkpoint and not args.sparse:
        print("Densifying loaded model")
        dense_sparse_dense.densify(model)

    saver = CheckpointSaver(checkpoint_dir=output_dir)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()

    # Optional fine-tune of only the final classifier weights for specified number of epochs (or part of)
    if not args.resume and args.ft_epochs > 0.:
        if isinstance(model, torch.nn.DataParallel):
            classifier_params = model.module.get_classifier().parameters()
        else:
            classifier_params = model.get_classifier().parameters()
        if args.opt.lower() == 'adam':
            finetune_optimizer = optim.Adam(
                classifier_params,
                lr=args.ft_lr, weight_decay=args.weight_decay)
        else:
            finetune_optimizer = optim.SGD(
                classifier_params,
                lr=args.ft_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

        finetune_epochs_int = int(np.ceil(args.ft_epochs))
        finetune_final_batches = int(np.ceil((1 - (finetune_epochs_int - args.ft_epochs)) * len(loader_train)))
        print(finetune_epochs_int, finetune_final_batches)
        for fepoch in range(0, finetune_epochs_int):
            if fepoch == finetune_epochs_int - 1 and finetune_final_batches:
                batch_limit = finetune_final_batches
            else:
                batch_limit = 0
            train_epoch(
                fepoch, model, loader_train, finetune_optimizer, train_loss_fn, args,
                output_dir=output_dir, batch_limit=batch_limit)

    best_loss = None
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.decay_epochs:
                adjust_learning_rate(
                    optimizer, epoch, initial_lr=args.lr,
                    decay_rate=args.decay_rate, decay_epochs=args.decay_epochs)

            if args.initial_batch_size:
                next_batch_size = adjust_batch_size(
                    epoch, initial_bs=args.initial_batch_size, target_bs=args.batch_size)
                if next_batch_size > batch_size:
                    print("Changing batch size from %d to %d" % (batch_size, next_batch_size))
                    batch_size = next_batch_size
                    loader_train = data.DataLoader(
                        dataset_train,
                        batch_size=batch_size,
                        pin_memory=True,
                        shuffle=True,
                        # sampler=sampler,
                        num_workers=args.workers)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                saver=saver, output_dir=output_dir)

            # save a recovery in case validation blows up
            saver.save_recovery({
                'epoch': epoch + 1,
                'arch': args.model,
                'sparse': args.sparse,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss_fn.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch + 1,
                batch_idx=0)

            step = epoch * len(loader_train)
            eval_metrics = validate(
                step, model, loader_eval, validate_loss_fn, args,
                output_dir=output_dir)

            if lr_scheduler is not None:
                lr_scheduler.step(eval_metrics['eval_loss'])

            rowd = OrderedDict(epoch=epoch)
            rowd.update(train_metrics)
            rowd.update(eval_metrics)
            with open(os.path.join(output_dir, 'summary.csv'), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=rowd.keys())
                if best_loss is None:  # first iteration (epoch == 1 can't be used)
                    dw.writeheader()
                dw.writerow(rowd)

            # save proper checkpoint with eval metric
            best_loss = saver.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'sparse': args.sparse,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch + 1,
                metric=eval_metrics['eval_loss'])

    except KeyboardInterrupt:
        pass
    print('*** Best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        saver=None, output_dir='', batch_limit=0):

    epoch_step = (epoch - 1) * len(loader)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
        step = epoch_step + batch_idx
        data_time_m.update(time.time() - end)

        input = input.cuda()
        if isinstance(target, list):
            target = [t.cuda() for t in target]
        else:
            target = target.cuda()

        output = model(input)

        loss = loss_fn(output, target)
        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.sparse:
            dense_sparse_dense.apply_sparsity_mask(model)

        batch_time_m.update(time.time() - end)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                  '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx * len(input), len(loader.sampler),
                100. * batch_idx / len(loader),
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input.size(0) / batch_time_m.val,
                rate_avg=input.size(0) / batch_time_m.avg,
                data_time=data_time_m))

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if saver is not None and batch_idx % args.recovery_interval == 0:
            saver.save_recovery({
                'epoch': epoch,
                'arch': args.model,
                'sparse': args.sparse,
                'state_dict':  model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                epoch=epoch,
                batch_idx=batch_idx)

        end = time.time()

        if batch_limit and batch_idx >= batch_limit:
            break

    return OrderedDict([('train_loss', losses_m.avg)])


def validate(step, model, loader, loss_fn, args, output_dir=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target, _) in enumerate(loader):
            input = input.cuda()
            if isinstance(target, list):
                target = target[0].cuda()
            else:
                target = target.cuda()

            output = model(input)

            if isinstance(output, list):
                output = output[0]

            # augmentation reduction
            reduce_factor = loader.dataset.get_aug_factor()
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            # calc loss
            loss = loss_fn(output, target)
            losses_m.update(loss.item(), input.size(0))

            # metrics
            prec1, prec5 = accuracy(output, target, topk=(1, 3))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                      'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    i, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    top1=prec1_m, top5=prec5_m))

                if args.save_batches:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'validate-batch-%d.jpg' % i),
                        padding=0,
                        normalize=True)

    metrics = OrderedDict([('eval_loss', losses_m.avg), ('eval_prec1', prec1_m.avg)])

    return metrics


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate=0.1, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (decay_rate ** (epoch // decay_epochs))
    print('Setting LR to', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_batch_size(epoch, initial_bs, target_bs, decay_epochs=1):
    batch_size = min(target_bs, initial_bs * (2 ** (epoch // decay_epochs)))
    return batch_size


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class CheckpointSaver:
    def __init__(
            self,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            max_history=10):

        self.checkpoint_files = []
        self.best_metric = None
        self.worst_metric = None
        self.max_history = max_history
        assert self.max_history >= 1
        self.curr_recovery_file = ''
        self.last_recovery_file = ''
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'

    def save_checkpoint(self, state, epoch, metric=None):
        worst_metric = self.checkpoint_files[-1] if self.checkpoint_files else None
        if len(self.checkpoint_files) < self.max_history or metric < worst_metric[1]:
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)

            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            if metric is not None:
                state['metric'] = metric
            torch.save(state, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(self.checkpoint_files, key=lambda x: x[1])

            if metric is not None and (self.best_metric is None or metric < self.best_metric[1]):
                self.best_metric = (epoch, metric)
                shutil.copyfile(save_path, os.path.join(self.checkpoint_dir, 'model_best' + self.extension))
        return None if self.best_metric is None else self.best_metric[1]

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        print(self.checkpoint_files, to_delete)
        for d in to_delete:
            try:
                print('Cleaning checkpoint', d)
                os.remove(d[0])
            except Exception as e:
                print('Exception (%s) while deleting checkpoint' % str(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, state, epoch, batch_idx):
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        torch.save(state, save_path)
        if os.path.exists(self.last_recovery_file):
            try:
                print('Cleaning recovery', self.last_recovery_file)
                os.remove(self.last_recovery_file)
            except Exception as e:
                print("Exception (%s) while removing %s" % (str(e), self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''


if __name__ == '__main__':
    main()
