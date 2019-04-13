# Weakly Supervised Nuclei Segmentation using Points Annotation
## Description
This page contains the code of weakly supervised nuclei segmentation using points annotation proposed
 in [1]. 


## Dependencies
Ubuntu 16.04

Pytorch 0.4.1

Gcc >= 4.9



## Usage
### Build CRFLoss
GCC version >= 4.9 is required to build the CRF loss correctly.
```bash
cd ./crf_loss
python setup.py install
```
### Prepare data
* Put the original images in the folder *./data/MO/images* and instance labels in the folder *./data/MO/labels_instance*
* Specify the image names of train, val, test sets in the json file under *./data/MO*
* Run the code:
```bash
python prepare_data.py
```

### Train and test
Before training or testing the model, set the options and data transforms in ```options.py```. Most options are set as default values, 
and a part of them can be also parsed from the command line, for example:
```train
python train.py --lr 0.0001 --epochs 60 --log-interval 30
python test.py --model-path ./experiments/MO/checkpoints/checkpoint_60.pth.tar
```

## Citation 
If you find this code helpful, please cite our work:

[1] Hui Qu, Pengxiang Wu, Qiaoying Huang, Jingru Yi, Gregory M. Riedlinger, Subhajyoti De and Dimitris N. Metaxas, 
"Weakly Supervised Deep Nuclei Segmentation using Points Annotation in Histopathology Images", In
 *International Conference on Medical Imaging with Deep Learning (MIDL)*, 2019.