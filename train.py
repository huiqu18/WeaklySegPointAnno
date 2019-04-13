import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
from tensorboardX import SummaryWriter

from model import ResUNet34
import utils
from dataset import DataFolder
from my_transforms import get_transforms
from options import Options
from crf_loss.crfloss import CRFLoss


def main():
    global opt, num_iter, tb_writer, logger, logger_results
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = setup_logging(opt)

    # ----- create model ----- #
    model = ResUNet34(pretrained=opt.model['pretrained'])
    # if not opt.train['checkpoint']:
    #     logger.info(model)
    model = nn.DataParallel(model)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    criterion = torch.nn.NLLLoss(ignore_index=2).cuda()
    if opt.train['crf_weight'] > 0:
        logger.info('=> Using CRF loss...')
        global criterion_crf
        criterion_crf = CRFLoss(opt.train['sigmas'][0], opt.train['sigmas'][1])

    # ----- load data ----- #
    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'test': get_transforms(opt.transform['test'])}

    img_dir = '{:s}/train'.format(opt.train['img_dir'])
    target_vor_dir = '{:s}/train'.format(opt.train['label_vor_dir'])
    target_cluster_dir = '{:s}/train'.format(opt.train['label_cluster_dir'])
    dir_list = [img_dir, target_vor_dir, target_cluster_dir]
    post_fix = ['label_vor.png', 'label_cluster.png']
    num_channels = [3, 3, 3]
    train_set = DataFolder(dir_list, post_fix, num_channels, data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])

    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    num_epoch = opt.train['train_epochs'] + opt.train['finetune_epochs']
    num_iter = num_epoch * len(train_loader)
    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d}".format(num_iter))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))
    logger.info("=> Fine-tune epochs using dense CRF loss: {:d}".format(opt.train['finetune_epochs']))
    logger.info("=> CRF loss weight: {:.2g}".format(opt.train['crf_weight']))

    for epoch in range(opt.train['start_epoch'], num_epoch):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epoch))
        finetune_flag = False if epoch < opt.train['train_epochs'] else True
        if epoch == opt.train['train_epochs']:
            logger.info("Fine-tune begins, lr = {:.2g}".format(opt.train['lr'] * 0.1))
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.train['lr'] * 0.1

        train_results = train(train_loader, model, optimizer, criterion, finetune_flag)
        train_loss, train_loss_vor, train_loss_cluster, train_loss_crf = train_results

        cp_flag = (epoch+1) % opt.train['checkpoint_freq'] == 0
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, opt.train['save_dir'], cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch+1, train_loss, train_loss_vor, train_loss_cluster,
                                    train_loss_crf))
        # tensorboard logs
        tb_writer.add_scalars('epoch_losses',
                              {'train_loss': train_loss, 'train_loss_vor': train_loss_vor,
                               'train_loss_cluster': train_loss_cluster,
                               'train_loss_crf': train_loss_crf}, epoch)
    tb_writer.close()
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(train_loader, model, optimizer, criterion, finetune_flag):
    # list to store the average loss for this epoch
    results = utils.AverageMeter(4)

    # switch to train mode
    model.train()
    for i, sample in enumerate(train_loader):
        input, target1, target2 = sample

        # for b in range(input.size(0)):
        #     utils.show_figures((input[b, 0, :, :].numpy(), target1[b,0,:,:].numpy(), target2[b,0,:,:]))

        if target1.dim() == 4:
            target1 = target1.squeeze(1)
        if target2.dim() == 4:
            target2 = target2.squeeze(1)

        input_var = input.cuda()

        # compute output
        output = model(input_var)
        prob_maps = F.softmax(output, dim=1)
        log_prob_maps = F.log_softmax(output, dim=1)
        loss_vor = criterion(log_prob_maps, target1.cuda())
        loss_cluster = criterion(log_prob_maps, target2.cuda())
        loss = loss_vor + loss_cluster

        if opt.train['crf_weight'] != 0 and finetune_flag:
            image = input.detach().clone().data.cpu()
            mean, std = np.load('{:s}/mean_std.npy'.format(opt.train['data_dir']))
            for k in range(image.size(0)):
                for t, m, s in zip(image[k], mean, std):
                    t.mul_(s).add_(m)

            loss_crf = criterion_crf(prob_maps.cpu(), image)
            loss_crf = loss_crf.cuda()
            loss = loss_vor + loss_cluster + opt.train['crf_weight'] * loss_crf

        if opt.train['crf_weight'] != 0 and finetune_flag:
            result = [loss.item(), loss_vor.item(), loss_cluster.item(), loss_crf.item()]
        else:
            result = [loss.item(), loss_vor.item(), loss_cluster.item(), -1]

        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_cluster {r[2]:.4f}'
                        '\tLoss_CRF {r[3]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_cluster {r[2]:.4f}'
                '\tloss_CRF {r[3]:.4f}'.format(r=results.avg))

    return results.avg


def save_checkpoint(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_vor\ttrain_loss_cluster')

    return logger, logger_results


if __name__ == '__main__':
    main()
