import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import wandb

import deeplab
from pascal import VOCSegmentation
from cityscapes import Cityscapes
from utils import AverageMeter, inter_and_union

# Add parent older to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import weight_regularization as wr

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None,
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')

parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--wandb-off', action='store_true', default=False)
parser.add_argument('--extra-tags', type=str, default='')
parser.add_argument('--reg-type', type=str, default=None)
parser.add_argument('--ortho_decay', type=float, default=1e-2)

args = parser.parse_args()

PASCAL_PATH = '/mnt5/yoavkurtz/datasets/PASCAL_VOC_2012'


def main():
    assert torch.cuda.is_available()
    # TODO handle reproducibility?
    torch.backends.cudnn.benchmark = True

    use_wandb = not args.wandb_off
    if use_wandb:
        tags = [args.dataset, args.backbone]
        if args.extra_tags != '':
            tags += [args.extra_tags]
        wandb.init(project='group_ortho', config=vars(args), tags=tags)
        wandb.run.log_code(".")

    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
        args.backbone, args.dataset, args.exp)
    if args.dataset == 'pascal':
        dataset = VOCSegmentation(os.path.join(PASCAL_PATH, 'VOCdevkit'),
                                  train=args.train, crop_size=args.crop_size)
    elif args.dataset == 'cityscapes':
        dataset = Cityscapes('data/cityscapes',
                             train=args.train, crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    if args.backbone == 'resnet101':
        model = getattr(deeplab, 'resnet101')(
            pretrained=(not args.scratch),
            num_classes=len(dataset.CLASSES),
            num_groups=args.groups,
            weight_std=args.weight_std,
            beta=args.beta)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.groups and args.reg_type is not None:
        weight_groups_dict = wr.get_layers_to_regularize(model, num_groups_fn=(lambda x: args.groups))
    else:
        weight_groups_dict = None

    if args.train:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        model = nn.DataParallel(model).cuda()
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        backbone_params = (
                list(model.module.conv1.parameters()) +
                list(model.module.bn1.parameters()) +
                list(model.module.layer1.parameters()) +
                list(model.module.layer2.parameters()) +
                list(model.module.layer3.parameters()) +
                list(model.module.layer4.parameters()))
        last_params = list(model.module.aspp.parameters())
        optimizer = optim.SGD([
            {'params': filter(lambda p: p.requires_grad, backbone_params)},
            {'params': filter(lambda p: p.requires_grad, last_params)}],
            lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=args.train,
            pin_memory=True, num_workers=args.workers)
        max_iter = args.epochs * len(dataset_loader)
        losses = AverageMeter()
        reg_losses = AverageMeter()
        start_epoch = 0

        if args.resume:
            if os.path.isfile(args.resume):
                print('=> loading checkpoint {0}'.format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('=> loaded checkpoint {0} (epoch {1})'.format(
                    args.resume, checkpoint['epoch']))
            else:
                print('=> no checkpoint found at {0}'.format(args.resume))

        for epoch in range(start_epoch, args.epochs):
            for i, (inputs, target) in enumerate(dataset_loader):
                cur_iter = epoch * len(dataset_loader) + i
                lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = lr * args.last_mult

                inputs = Variable(inputs.cuda())
                target = Variable(target.cuda())
                outputs = model(inputs)
                loss = criterion(outputs, target)
                if args.reg_type and weight_groups_dict is not None:
                    ortho_loss = wr.weights_reg(model, args.reg_type, weight_groups_dict)
                    reg_losses.update(ortho_loss.item(), args.batch_size)
                    loss += args.ortho_decay * ortho_loss

                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    pdb.set_trace()
                losses.update(loss.item(), args.batch_size)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % args.print_freq == 0:
                    print('epoch: {0}\t'
                          'iter: {1}/{2}\t'
                          'lr: {3:.6f}\t'
                          'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                        epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

            if use_wandb:
                wandb.log({'tarin_loss': losses.avg, 'reg_loss': reg_losses.avg})

            if epoch % 10 == 9:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fname % (epoch + 1))

    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        model.eval()
        checkpoint = torch.load(model_fname % args.epochs)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
        model.load_state_dict(state_dict)
        cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

        inter_meter = AverageMeter()
        union_meter = AverageMeter()
        for i in range(len(dataset)):
            inputs, target = dataset[i]
            inputs = Variable(inputs.cuda())
            outputs = model(inputs.unsqueeze(0))
            _, pred = torch.max(outputs, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            mask = target.numpy().astype(np.uint8)
            imname = dataset.masks[i].split('/')[-1]
            mask_pred = Image.fromarray(pred)
            mask_pred.putpalette(cmap)
            mask_pred.save(os.path.join('data/val', imname))
            print('eval: {0}/{1}'.format(i + 1, len(dataset)))

            inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
            inter_meter.update(inter)
            union_meter.update(union)

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        for i, val in enumerate(iou):
            print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
        print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))

        if use_wandb:
            wandb.summary['Mean IoU'] = iou.mean() * 100

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
