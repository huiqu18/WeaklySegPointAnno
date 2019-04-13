import os
import numpy as np
import argparse


class Options:
    def __init__(self, isTrain):
        self.dataset = 'MO'     # dataset: LC: Lung Cancer, MO: MultiOrgan
        self.isTrain = isTrain  # train or test mode

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'ResUNet34'
        self.model['pretrained'] = True
        self.model['fix_params'] = False
        self.model['in_c'] = 3  # input channel

        # --- training params --- #
        self.train = dict()
        self.train['data_dir'] = './data_for_train/{:s}'.format(self.dataset)  # path to data
        self.train['save_dir'] = './experiments/{:s}'.format(self.dataset)  # path to save results
        self.train['input_size'] = 224          # input size of the image
        self.train['train_epochs'] = 60         # number of training epochs
        self.train['finetune_epochs'] = 10      # number of refinement epochs
        self.train['batch_size'] = 8            # batch size
        self.train['crf_weight'] = 0.0005       # weight for crf loss
        self.train['sigmas'] = (10.0, 10.0/255)  # parameters in CRF loss
        self.train['checkpoint_freq'] = 10      # epoch to save checkpoints
        self.train['lr'] = 0.0001               # initial learning rate
        self.train['weight_decay'] = 1e-4       # weight decay
        self.train['log_interval'] = 30         # iterations to print training results
        self.train['workers'] = 1               # number of workers to load images
        self.train['gpus'] = [0, ]              # select gpu devices
        # --- resume training --- #
        self.train['start_epoch'] = 0    # start epoch
        self.train['checkpoint'] = ''

        # --- data transform --- #
        self.transform = dict()

        # --- test parameters --- #
        self.test = dict()
        self.test['test_epoch'] = 60
        self.test['gpus'] = [0, ]
        self.test['img_dir'] = './data_for_train/{:s}/images/test'.format(self.dataset)
        self.test['label_dir'] = './data/{:s}/labels_instance'.format(self.dataset)
        self.test['save_flag'] = True
        self.test['patch_size'] = 224
        self.test['overlap'] = 80
        self.test['save_dir'] = './experiments/{:s}/test_results'.format(self.dataset, self.test['test_epoch'])
        self.test['model_path'] = './experiments/{:s}/checkpoints/checkpoint_{:d}.pth.tar'.format(self.dataset, self.test['test_epoch'])
        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 20  # minimum area for an object

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'], help='input batch size for training')
            parser.add_argument('--epochs', type=int, default=self.train['train_epochs'], help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpus', type=list, default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'], help='directory to save training results')
            args = parser.parse_args()

            self.train['batch_size'] = args.batch_size
            self.train['train_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpus'] = args.gpus
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/images'.format(self.train['data_dir'])
            self.train['label_vor_dir'] = '{:s}/labels_voronoi'.format(self.train['data_dir'])
            self.train['label_cluster_dir'] = '{:s}/labels_cluster'.format(self.train['data_dir'])

            self.train['save_dir'] = args.save_dir
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

            # define data transforms for training
            self.transform['train'] = {
                'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'vertical_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': self.train['input_size'],
                'label_encoding': 2,
                'to_tensor': 1,
                'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }
            self.transform['test'] = {
                'to_tensor': 1,
                'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }

        else:
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'], help='flag to save the network outputs and predictions')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'], help='train model to be evaluated')
            args = parser.parse_args()
            self.test['save_flag'] = args.save_flag
            self.test['img_dir'] = args.img_dir
            self.test['label_dir'] = args.label_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path'] = args.model_path

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

            self.transform['test'] = {
                'to_tensor': 1,
                'normalize': np.load('{:s}/mean_std.npy'.format(self.train['data_dir']))
            }

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        file = open(filename, 'w')
        groups = ['model', 'train', 'transform'] if self.isTrain else ['model', 'test', 'post', 'transform']

        file.write("# ---------- Options ---------- #")
        file.write('\ndataset: {:s}\n'.format(self.dataset))
        file.write('isTrain: {}\n'.format(self.isTrain))
        for group, options in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in options.items():
                    if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                        file.write("{:s}:\n".format(name))
                        for t_name, t_val in val.items():
                            file.write("\t{:s}: {:s}\n".format(t_name, repr(t_val)))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()




