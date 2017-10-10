import argparse
import csv
import os
import shutil
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime

from dataset import CDiscountDataset
from models import model_factory, dense_sparse_dense
from lr_scheduler import ReduceLROnPlateau
from utils import AverageMeter, get_outdir

import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--loss', default='mlsm', type=str, metavar='LOSS',
                    help='Loss function (default: "nll"')
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
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--ft-epochs', type=float, default=0., metavar='LR',
                    help='Number of finetuning epochs (final layer only)')
parser.add_argument('--ft-opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--ft-lr', type=float, default=0.0001, metavar='N',
                    help='Finetune learning rates.')
parser.add_argument('--drop', type=float, default=0.1, metavar='DROP',
                    help='Dropout rate (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='M',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--no-tb', action='store_true', default=False,
                    help='disables tensorboard')
parser.add_argument('--tbh', default='127.0.0.1:8009', type=str, metavar='IP',
                    help='Tensorboard (Crayon) host')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
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

    train_input_root = os.path.join(args.data)
    train_labels_file = './data/labels.csv'

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

    batch_size = args.batch_size
    num_epochs = args.epochs
    img_size = (args.img_size, args.img_size)
    num_classes = 5270 #FIXME

    torch.manual_seed(args.seed)

    # FIXME hackish, really don't care about original class count here, need a way to not require that
    if 'inception' in args.model:
        num_classes_init = 1001
    else:
        num_classes_init = 1000

    model = model_factory.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes_init,
        drop_rate=args.drop,
        global_pool=args.gp)

    model.reset_classifier(num_classes=num_classes)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model.cuda()


    dataset_train = CDiscountDataset(
        train_input_root,
        train=True,
        img_size=img_size,
        fold=args.fold,
    )

    #sampler = WeightedRandomOverSampler(dataset_train.get_sample_weights())

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        #sampler=sampler,
        num_workers=args.workers
    )

    dataset_eval = CDiscountDataset(
        train_input_root,
        train=False,
        img_size=img_size,
        test_aug=args.tta,
        fold=args.fold,
    )

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=4*batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, alpha=0.9, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"

    if not args.decay_epochs:
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=8)
    else:
        lr_scheduler = None

    if args.class_weights:
        class_weights = torch.from_numpy(dataset_train.get_class_weights()).float().cuda()
        class_weights_norm = class_weights / class_weights.sum()
        class_weights_norm = class_weights_norm.cuda()
    else:
        class_weights = None
        class_weights_norm = None

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()

    # optionally resume from a checkpoint
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            sparse_checkpoint = True if 'sparse' in checkpoint and checkpoint['sparse'] else False
            if sparse_checkpoint:
                print("Loading sparse model")
                dense_sparse_dense.sparsify(model, sparsity=0.)  # ensure sparsity_masks exist in model definition

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']

            if args.sparse and not sparse_checkpoint:
                print("Sparsifying loaded model")
                dense_sparse_dense.sparsify(model, sparsity=0.5)
            elif sparse_checkpoint and not args.sparse:
                print("Densifying loaded model")
                dense_sparse_dense.densify(model)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(-1)
    else:
        if args.sparse:
            dense_sparse_dense.sparsify(model, sparsity=0.5)

    use_tensorboard = not args.no_tb and CrayonClient is not None
    if use_tensorboard:
        hostname = '127.0.0.1'
        port = 8889
        host_port = args.tbh.split(':')[:2]
        if len(host_port) == 1:
            hostname = host_port[0]
        elif len(host_port) >= 2:
            hostname, port = host_port[:2]
        try:
            cc = CrayonClient(hostname=hostname, port=port)
            try:
                cc.remove_experiment(exp_name)
            except ValueError:
                pass
            exp = cc.create_experiment(exp_name)
        except Exception as e:
            exp = None
            print("Error (%s) connecting to Tensoboard/Crayon server. Giving up..." % str(e))
    else:
        exp = None

    # Optional fine-tune of only the final classifier weights for specified number of epochs (or part of)
    if not args.resume and args.ft_epochs > 0.:
        if args.opt.lower() == 'adam':
            finetune_optimizer = optim.Adam(
                model.get_classifier().parameters(),
                lr=args.ft_lr, weight_decay=args.weight_decay)
        else:
            finetune_optimizer = optim.SGD(
                model.get_classifier().parameters(),
                lr=args.ft_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        finetune_epochs_int = int(np.ceil(args.ft_epochs))
        finetune_final_batches = int(np.ceil((1 - (finetune_epochs_int - args.ft_epochs)) * len(loader_train)))
        print(finetune_epochs_int, finetune_final_batches)
        for fepoch in range(1, finetune_epochs_int + 1):
            if fepoch == finetune_epochs_int and finetune_final_batches:
                batch_limit = finetune_final_batches
            else:
                batch_limit = 0
            train_epoch(
                fepoch, model, loader_train, finetune_optimizer, loss_fn, args,
                output_dir=output_dir, batch_limit=batch_limit)
            step = fepoch * len(loader_train)
            validate(step, model, loader_eval, loss_fn, args, output_dir)

    best_loss = None
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            if args.decay_epochs:
                adjust_learning_rate(optimizer, epoch, initial_lr=args.lr, decay_epochs=args.decay_epochs)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, loss_fn, args,
                output_dir=output_dir, exp=exp)

            step = epoch * len(loader_train)
            eval_metrics = validate(
                step, model, loader_eval, loss_fn, args,
                output_dir=output_dir, exp=exp)

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

            best = False
            if best_loss is None or eval_metrics['eval_loss'] < best_loss[1]:
                best_loss = (epoch, eval_metrics['eval_loss'])
                best = True

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'sparse': args.sparse,
                'state_dict':  model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
                'gp': args.gp,
                },
                is_best=best,
                filename='checkpoint-%d.pth.tar' % epoch,
                output_dir=output_dir)

    except KeyboardInterrupt:
        pass
    print('*** Best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        output_dir='', exp=None, batch_limit=0):

    epoch_step = (epoch - 1) * len(loader)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
        step = epoch_step + batch_idx
        data_time_m.update(time.time() - end)
        input, target = input.cuda(), target.cuda()
        input_var = autograd.Variable(input)
        target_var = autograd.Variable(target)

        output = model(input_var)

        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input_var.size(0))

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
                rate=input_var.size(0) / batch_time_m.val,
                rate_avg=input_var.size(0) / batch_time_m.avg,
                data_time=data_time_m))

            if exp is not None:
                exp.add_scalar_value('loss_train', losses_m.val, step=step)
                exp.add_scalar_value('learning_rate', optimizer.param_groups[0]['lr'], step=step)

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        end = time.time()

        if batch_limit and batch_idx >= batch_limit:
            break

    return OrderedDict([('train_loss', losses_m.avg)])


def validate(step, model, loader, loss_fn, args, output_dir='', exp=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    output_list = []
    target_list = []
    for i, (input, target, _) in enumerate(loader):
        input, target = input.cuda(), target.cuda()
        input_var = autograd.Variable(input, volatile=True)
        target_var = autograd.Variable(target, volatile=True)

        output = model(input_var)

        # augmentation reduction
        reduce_factor = loader.dataset.get_aug_factor()
        if reduce_factor > 1:
            output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
            target_var.data = target_var.data[0:target_var.size(0):reduce_factor]

        # calc loss
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input.size(0))

        # output non-linearities and metrics
        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
        prec1_m.update(prec1[0], output.size(0))
        prec5_m.update(prec5[0], output.size(0))

        # copy to CPU and collect
        target_list.append(target.cpu().numpy())
        output_list.append(output.data.cpu().numpy())

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

    if exp is not None:
        exp.add_scalar_value('loss_eval', losses_m.avg, step=step)
        exp.add_scalar_value('prec@1_eval', prec1_m.avg, step=step)

    return metrics


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output_dir=''):
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(output_dir, 'model_best.pth.tar'))


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


if __name__ == '__main__':
    main()
