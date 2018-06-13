print '---- THIS CODE REQUIRES CHAINER V3 ----'

import warnings

warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

import numpy as np
import time, os, copy, random, h5py
from argparse import ArgumentParser
import chainer
import chainer.functions as F
from chainer import link
from chainer import optimizers, Variable, cuda, serializers
from chainer.iterators import MultiprocessIterator
from chainer.optimizer import WeightDecay, GradientClipping
import datachef as DataChef
import modelx as Models


def parse_args():
    # set extract_features to 0 for training or 1 for feature extraction
    def_extract_features = 0
    # batch size
    def_minibatch = 16
    # image size for semantic segmentation
    def_scales_tr = '512,512'
    # image size for re-identification
    def_scales_reid = '512,170'  # '778,255'
    # learning rates for fresh and pretrained layers
    def_optimizer = 'lr:0.01--lr_pretrained:0.01'
    # GPU ids
    def_GPUs = '0'
    # set checkpoint bigger than zero to load saved model from checkpoints folder
    def_checkpoint = 0
    # set pre-trained model path for finetuning using evaluation datasets
    def_model_path_for_ft = ''

    # label for the dataset
    def_dataset = 'ReID10Dx'
    # number of different ids in training data
    def_label_dim = '16803'
    def_label_dim_ft = '16803'
    # the image list for feature extraction
    def_eval_split = 'cuhk03_gallery'
    # the image list for training
    def_train_set = 'train_10d'

    # number of workers to load images parallel
    def_nb_processes = 4
    # maximum number of iterations
    def_max_iter = 200000
    # loss report interval
    def_report_interval = 50
    # number of iterations for checkpoints
    def_save_interval = 20000

    def_project_folder = '.'
    def_dataset_folder = '/home/basaran/Documents/experiments/SpindleNet'
    p = ArgumentParser()
    p.add_argument('--extract_features', default=def_extract_features, type=int)
    p.add_argument('--minibatch', default=def_minibatch, type=int)
    p.add_argument('--scales_tr', default=def_scales_tr, type=str)
    p.add_argument('--scales_reid', default=def_scales_reid, type=str)
    p.add_argument('--optimizer', default=def_optimizer, type=str)
    p.add_argument('--GPUs', default=def_GPUs, type=str)
    p.add_argument('--dataset', default=def_dataset, type=str)
    p.add_argument('--eval_split', default=def_eval_split, type=str)
    p.add_argument('--train_set', default=def_train_set, type=str)
    p.add_argument('--checkpoint', default=def_checkpoint, type=int)
    p.add_argument('--model_path_for_ft', default=def_model_path_for_ft, type=str)
    p.add_argument('--label_dim', default=def_label_dim, type=str)
    p.add_argument('--label_dim_ft', default=def_label_dim_ft, type=int)
    p.add_argument('--nb_processes', default=def_nb_processes, type=int)
    p.add_argument('--max_iter', default=def_max_iter, type=int)
    p.add_argument('--report_interval', default=def_report_interval, type=int)
    p.add_argument('--save_interval', default=def_save_interval, type=int)
    p.add_argument('--project_folder', default=def_project_folder, type=str)
    p.add_argument('--dataset_folder', default=def_dataset_folder, type=str)
    args = p.parse_args()
    return args


def Evaluation():
    # Creat data generator
    batch_tuple = MultiprocessIterator(
        DataChef.ReID10D(args, args.project_folder + '/evaluation_list/' + args.eval_split + '.txt',
                         image_size=args.scales_tr[0]),
        args.minibatch, n_prefetch=2, n_processes=args.nb_processes, shared_mem=20000000, repeat=False, shuffle=False)
    # Keep the log in history
    history = {args.dataset: {'features': []}}

    for dataBatch in batch_tuple:
        dataBatch = zip(*dataBatch)
        # Prepare batch data
        IMG = np.array_split(np.array(dataBatch[0]), len(Model), axis=0)
        LBL = np.array_split(np.array(dataBatch[1]), len(Model), axis=0)
        # Forward
        for device_id, img, lbl in zip(range(len(Model)), IMG, LBL):
            Model[device_id](img, lbl, args.dataset, train=False)
        # Aggregate reporters from all GPUs
        reporters = []
        for i in range(len(Model)):
            reporters.append(Model[i].reporter)
            Model[i].reporter = {}  # clear reporter
        # History
        for reporter in reporters:
            for k in reporter[args.dataset].keys():
                history[args.dataset][k].append(reporter[args.dataset][k])
    # storing features to an outputfile
    features = np.concatenate(history[args.dataset]['features'], axis=0)
    outfile = '%s/evaluation_features/%s_@%s_%s.csv' % (
    args.project_folder, args.dataset, args.checkpoint, args.eval_split)
    np.savetxt(outfile, features, delimiter=',', fmt='%0.12e')


def Train():
    # Create data generator
    batch_tuples, history = {}, {}
    for dataset in args.dataset.split('+'):
        batch_tuples.update({dataset: []})
        for image_size in args.scales_tr:
            iterator = MultiprocessIterator(
                DataChef.ReID10D(args, args.project_folder + '/train_list/' + args.train_set + '.txt',
                                 image_size=image_size),
                args.minibatch, n_prefetch=2, n_processes=args.nb_processes, shared_mem=20000000, repeat=True,
                shuffle=True)
            batch_tuples[dataset].append(iterator)
        # Keep the log in history
        history.update({dataset: {'loss': []}})
    # Random input image size (change it after every x minibatch)
    batch_tuple_indx = np.random.choice(range(len(args.scales_tr)), args.max_iter / 10)
    batch_tuple_indx = list(np.repeat(batch_tuple_indx, 10))
    # Train
    start_time = time.time()
    for iterk in range(args.checkpoint, len(batch_tuple_indx)):
        # Get a minibatch while sequentially rotating between datasets
        for dataset in args.dataset.split('+'):
            dataBatch = batch_tuples[dataset][batch_tuple_indx[iterk]].next()
            dataBatch = zip(*dataBatch)
            # Prepare batch data
            IMG = np.array_split(np.array(dataBatch[0]), len(Model), axis=0)
            LBL = np.array_split(np.array(dataBatch[1]), len(Model), axis=0)
            # Forward
            for device_id, img, lbl in zip(range(len(Model)), IMG, LBL):
                Model[device_id](img, lbl, dataset, train=True)
            # Aggregate reporters from all GPUs
            reporters = []
            for i in range(len(Model)):
                reporters.append(Model[i].reporter)
                Model[i].reporter = {}  # clear reporter
            # History
            for reporter in reporters:
                for k in reporter[dataset].keys():
                    history[dataset][k].append(reporter[dataset][k])
            # Accumulate grads
            for i in range(1, len(Model)):
                Model[0].addgrads(Model[i])
            # Update
            opt.update()
            # Update params of other models
            for i in range(1, len(Model)):
                Model[i].copyparams(Model[0])
        # Report
        if (iterk + 1) % args.report_interval == 0:
            DataChef.Report(
                history, args.report_interval * len(args.GPUs), (iterk + 1), time.time() - start_time, split='train')
        # Saving the model
        if (iterk + 1) % args.save_interval == 0 or (iterk + 1) == len(batch_tuple_indx):
            serializers.save_hdf5('%s/checkpoints/%s_%s_iter_%d.chainermodel' %
                                  (args.project_folder, args.dataset, args.train_set[6:], iterk + 1), Model[0])
            serializers.save_npz('%s/checkpoints/%s_%s_iter_%d.chaineropt' %
                                 (args.project_folder, args.dataset, args.train_set[6:], iterk + 1), opt)
        # Decrease learning rate (poly in 10 steps)
        if (iterk + 1) % int(args.max_iter / 10) == 0:
            decay_rate = (1.0 - float(iterk) / args.max_iter) ** 0.9
            # Learning rate of fresh layers
            opt.lr = args.optimizer['lr'] * decay_rate
            # Learning rate of pretrained layers
            for name, param in opt.target.namedparams():
                if name.startswith('/predictor/'):
                    param.update_rule.hyperparam.lr = args.optimizer['lr_pretrained'] * decay_rate


def SetupOptimizer(model):
    opt = optimizers.NesterovAG(
        lr=args.optimizer['lr'], momentum=0.9)
    opt.setup(model)
    return opt


def toGPU():
    # main model is always first
    Model = [opt.target]
    for i in range(1, len(args.GPUs)):
        _model = copy.deepcopy(opt.target)
        _model.to_gpu(args.GPUs[i])
        _model.gpu_id = args.GPUs[i]
        _model.reporter = {}
        Model.append(_model)
    # First GPU device is by default the main one
    opt.target.to_gpu(args.GPUs[0])
    opt.target.gpu_id = args.GPUs[0]
    opt.target.reporter = {}
    return Model


def ResumeFromCheckpoint(path_to_checkpoint, model):
    init_weights = h5py.File(path_to_checkpoint, 'r')
    for name, link in model.namedlinks():
        if name.endswith('/conv') or name.endswith('/bn'):
            path_to_link = ['init_weights']
            for i in name.split('/')[1:]:
                path_to_link.append('["%s"]' % i)
            f = eval(''.join(path_to_link))
            if name.endswith('/conv'):
                link.W.data = np.array(f['W'])
                if 'b' in f.keys():
                    link.b.data = np.array(f['b'])
            elif name.endswith('/bn'):
                link.beta.data = np.array(f['beta'])
                link.gamma.data = np.array(f['gamma'])
                link.avg_mean = np.array(f['avg_mean'])
                link.avg_var = np.array(f['avg_var'])
    return model


# MAIN BODY
args = parse_args()
args.optimizer = dict(zip(['lr', 'lr_pretrained'], [float(x.split(':')[-1]) for x in args.optimizer.split('--')]))

args.label_dim = map(int, args.label_dim.split('+'))
args.scales_tr = [map(int, x.split(',')) for x in args.scales_tr.split('--')]
args.scales_reid = map(int, args.scales_reid.split(','))

# Adjust params w.r.t number of GPUs
args.GPUs = map(int, args.GPUs.split('/'))
args.minibatch *= len(args.GPUs)
args.optimizer['lr'] /= len(args.GPUs)
args.optimizer['lr_pretrained'] /= len(args.GPUs)
args.report_interval /= len(args.GPUs)
args.save_interval /= len(args.GPUs)
print vars(args)

print 'Initialize Model'
predictor = Models.InceptionV3(args, dilated=False)
model = Models.ReIDClassifier(predictor, args.label_dim[0], args)
with model.init_scope():
    model.segmentation = Models.InceptionV3Classifier(
        Models.InceptionV3(args, dilated=True), [Models.Classifier(20)], args)

if len(args.model_path_for_ft) > 0:
    model = ResumeFromCheckpoint(args.model_path_for_ft, model)
    model.classifiers = link.ChainList(Models.Conv(2048*3, args.label_dim_ft, 1, 1, 0, init_weights=None, pool=None,
                                                   nobias=False))

print 'Setup optimizer'
opt = SetupOptimizer(model)

# Use lower learning rate for pretrained parts
for name, param in opt.target.namedparams():
    if name.startswith('/predictor/'):
        param.update_rule.hyperparam.lr = args.optimizer['lr_pretrained']
opt.add_hook(WeightDecay(0.0005))
# opt.add_hook(GradientClipping(2.0))


# Resume training from a checkpoint
if args.checkpoint > 0:
    print 'Resume training from checkpoint'
    # Load model weights
    model = ResumeFromCheckpoint('%s/checkpoints/%s_%s_iter_%d.chainermodel' %
                                 (args.project_folder, args.dataset, args.train_set[6:], args.checkpoint), model)
    # Load optimizer status
    serializers.load_npz('%s/checkpoints/%s_%s_iter_%d.chaineropt' %
                         (args.project_folder, args.dataset, args.train_set[6:], args.checkpoint), opt)
    # Adjust the learning rate
    decay_rate = 1.0
    for iterk in range(args.checkpoint):
        if (iterk + 1) % int(args.max_iter / 10) == 0:
            decay_rate = (1.0 - float(iterk) / args.max_iter) ** 0.9
    # Learning rate of fresh layers
    opt.lr = args.optimizer['lr'] * decay_rate
    # Learning rate of pretrained layers
    for name, param in opt.target.namedparams():
        if name.startswith('/predictor/'):
            param.update_rule.hyperparam.lr = args.optimizer['lr_pretrained'] * decay_rate

print 'Load segmentation weights'
model.segmentation = ResumeFromCheckpoint('LIP_iter_30000.chainermodel', model.segmentation)

print 'Push Model to GPU'
Model = toGPU()

print 'Main Begins'
if args.extract_features:
    Evaluation()
else:
    Train()
