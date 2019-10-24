from collections import OrderedDict
import math
import numpy as np
import os
import pandas as pd
import PIL
import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as bce_w_logits
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import sampler as samplers
import torchvision
from torchvision import transforms
from typing import Dict, List, Tuple
import utils
from utils import device, dict_drop, TransientDict
from utils import DATA_DIR, SNAPSHOT_DIR


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class ClipByNeuron:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, grad_tensor):
        if grad_tensor.dim() == 2:
            for row in grad_tensor:
                row_norm = row.norm()
                if row_norm > self.max_norm:
                    row *= self.max_norm / row_norm


def init_weights(module):
    if isinstance(module, nn.modules.conv._ConvNd):
        module.weight.data.normal_(0.0, 0.02)
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0.0)


class ScaleParamInit:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, param):
        param.data *= self.scale
        

def predict(net, X, gpu):
    X = X.to(device(gpu))
    net.to(device(gpu))
    _, pred = net.train(False)(X).max(dim=1)
    return pred.cpu().numpy()
    
    
def validate(loss_f, net, val_iter, gpu):
    metrics = []
    for val_tuple in val_iter:
        val_tuple = [t.to(device(gpu)) for t in val_tuple]
        metrics += [loss_f.metrics(net, *val_tuple)]
    return [sum(metric) / len(metric) for metric in zip(*metrics)]


def multiexit_agreement(net, n_exits, val_iter, gpu):
    agreements = torch.zeros(n_exits, n_exits, device=device(gpu))
    net.to(device(gpu))
    n = 0
    for X, y in val_iter:
        logits_list = net.train(False)(X.to(device(gpu)))
        pred = [logits.max(dim=1)[1] for logits in logits_list]
        n += len(pred[0])
        for i in range(n_exits):
            for j in range(i+1, n_exits):
                agreements[i][j] += (pred[i]==pred[j]).float().sum()
    del net
    for i in range(n_exits):
        agreements[i][i] = n
        for j in range(i):
            agreements[i][j] = agreements[j][i]
    return agreements.cpu()
        

def multiexit_error_coocc(net, n_exits, val_iter, gpu):
    errors = torch.zeros(n_exits, n_exits)
    net.to(device(gpu))
    for X, y in val_iter:
        logits_list = net.train(False)(X.to(device(gpu)))
        err = [logits.max(dim=1)[1].cpu() != y for logits in logits_list]
        for i in range(n_exits):
            for j in range(i, n_exits):
                errors[i][j] += (err[i]*err[j]).float().sum()
    del net
    for i in range(n_exits):
        for j in range(i):
            errors[i][j] = errors[j][i]
    return errors

#########################################################

def blur_transform(pil_image):
    return pil_image.filter(PIL.ImageFilter.BLUR)
        

def to_rgb_transform(pil_image):
    if pil_image.mode == 'RGB':
        return pil_image
    else:
        return pil_image.convert('RGB')


class Lighting:
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval)
        self.eigvec = torch.tensor(eigvec)

    def __call__(self, tensor):
        if self.alphastd == 0: return tensor
        alpha = torch.randn(3) * self.alphastd
        rgb = (self.eigvec * alpha * self.eigval).sum(dim=1)
        return tensor + rgb[:, None, None]
    
    
class PadCrop:
    def __init__(self, pad:int, crop:int):
        self.pad = pad
        self.crop = crop

    def __call__(self, tensor):
        out = torch.zeros(*tensor.shape[:-2], self.crop, self.crop)
        offset = np.random.randint(-self.pad, self.pad+1, size=2)
        
        lower = np.maximum(offset, 0)
        upper = np.minimum(tensor.shape[-2:], tensor.shape[-2:] + offset)
        tensor_crop = tensor[..., lower[0]:upper[0], lower[1]:upper[1]]

        lower = np.maximum(-offset, 0)
        upper = np.minimum(tensor.shape[-2:], tensor.shape[-2:] - offset)
        out[..., lower[0]:upper[0], lower[1]:upper[1]] = tensor_crop
        return out
        
    
#########################################################

def _onto_cpu(storage, tag):
    return storage


def edit_odict(odict, key_f=None, val_f=None):
    key_f = key_f or (lambda k: k)
    val_f = val_f or (lambda v: v)
    return OrderedDict((key_f(k), val_f(v)) for k, v in odict.items())


def edit_statedict(snapshot_ep, key_f=None, val_f=None):
    snapshot_fname = str(SNAPSHOT_DIR/snapshot_ep)
    state_dict = torch.load(snapshot_fname, _onto_cpu)
    edited_dict = edit_odict(state_dict, key_f, val_f)
    torch.save(edited_dict, snapshot_fname)


def load_net(cf, key_f=None, val_f=None):
    snapshot_ep = utils.get_snapname_ep(cf['snapshot_name'])
    Net = globals()[cf['cf_net']['call']]
    net = Net(**dict_drop(cf['cf_net'], 'call'))
    snapshot_fname = str(SNAPSHOT_DIR / snapshot_ep)
    state_dict = torch.load(snapshot_fname, _onto_cpu)
    net.load_state_dict(edit_odict(state_dict, key_f, val_f))
    return net
    

def _load_pretrained(snapshot):
    if snapshot == 'alexnet:0':
        return AlexNet(pretrained=True)
    if snapshot == 'resnet18:0':
        return Resnet18(pretrained=True)
    if snapshot == 'resnet152:0':
        return Resnet152(pretrained=True)

def _load_my_trained(snapshot, prompt='Snapshot to load:',
                     key_f=None, val_f=None):
    # snapshot may or may not end with _ep123
    cf = utils.snapshot_cf(snapshot)
    snapshot_ep = utils.get_snapname_ep(snapshot, prompt)
    Net = globals()[cf['cf_net']['call']]
    net = Net(**dict_drop(cf['cf_net'], 'call'))
    snapshot_fname = str(SNAPSHOT_DIR / snapshot_ep)
    state_dict = torch.load(snapshot_fname, _onto_cpu)
    try:
        net.load_state_dict(edit_odict(state_dict, key_f, val_f))
    except RuntimeError:
        net.load_state_dict(
            edit_odict(state_dict, lambda k: k.replace('module.', '')))
    return net

def load_snapshot(snapshot, prompt='Snapshot to load:', key_f=None, val_f=None):
    return (_load_pretrained(snapshot) or
            _load_my_trained(snapshot, prompt, key_f, val_f))

#########################################################

def data_as_tensors(call, split, gpu, **cf_data):
    data = globals()[call](split, batch_size=1000, gpu=gpu, **cf_data)
    tuples = [tup for tup in data]
    return [torch.cat(batches) for batches in zip(*tuples)]

##############

class MnistDataset(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_CLASSES = 10
        self.TARGETS = self.train_labels if self.train else self.test_labels


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_CLASSES = 10
        self.TARGETS = self.train_labels if self.train else self.test_labels
        

class Cifar100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_CLASSES = 100
        self.TARGETS = self.train_labels if self.train else self.test_labels


class ImageNetTrnDataset(Dataset):
    def __init__(self, transform, target_transform=None):
        self.N_CLASSES = 1000
        self.img_dir = DATA_DIR / 'imagenet' / 'trn_images'
        self.tform = transform
        self.target_transform = target_transform or (lambda y: y)

        d = pd.read_csv(DATA_DIR/'imagenet'/'classes.csv', sep='\t')
        i = np.arange(d.n.sum())
        cumsum = [0] + list(d.n.cumsum())
        self.TARGETS = np.digitize(i, cumsum) - 1
        self.idx = i - np.array(cumsum)[self.TARGETS]
        self.i2synset = d.synset.values[self.TARGETS]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        i_img = self.idx[i]
        synset = self.i2synset[i]
        img_fname = os.listdir(self.img_dir/synset)[i_img]
        image = PIL.Image.open(self.img_dir/synset/img_fname)
        return self.tform(image), self.target_transform(self.TARGETS[i])
    

class ImageNetTestDataset(Dataset):
    def __init__(self, transform, target_transform=None):
        self.N_CLASSES = 1000
        self.img_dir = DATA_DIR/ 'imagenet'/'val_images'
        self.tform = transform
        self.target_transform = target_transform or (lambda y: y)

        targets_fname = DATA_DIR/'imagenet'/'val_targets.csv'
        self.TARGETS = pd.read_csv(targets_fname)['target']

    def __len__(self):
        return 50000

    def __getitem__(self, i):
        fname = 'ILSVRC2012_val_{:0>8}.JPEG'.format(i+1)
        image = PIL.Image.open(self.img_dir/fname)
        return self.tform(image), self.target_transform(self.TARGETS[i])

    def image(self, i):
        fname = 'ILSVRC2012_val_{:0>8}.JPEG'.format(i+1)
        image = PIL.Image.open(self.img_dir/fname)
        tform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(to_rgb_transform),
        ])
        return tform(image)

#######################################################

def dataset2semisup(dataset, n_unl_per_class, seed):
    n = len(dataset.TARGETS)
    rs = np.random.RandomState(seed)
    i_unlabelled = []
    for k in range(dataset.N_CLASSES):
        ind_k = np.where(np.array(dataset.TARGETS) == k)[0]
        i_unl = rs.choice(ind_k, n_unl_per_class, replace=False)
        i_unlabelled += list(i_unl)
    for i in i_unlabelled:
        dataset.TARGETS[i] = -100

    
#######################################################

class SplitEpochSampler(samplers.Sampler):
    def __init__(self, sampler, epoch_len):
        "epoch_len: number of examples per epoch"
        self.sampler = sampler
        self.epoch_len = epoch_len
        self.q = list()

    def _fill_q(self):
        while len(self.q) < self.epoch_len:
            self.q += list(iter(self.sampler))
        
    def __len__(self):
        return self.epoch_len

    def __iter__(self):
        self._fill_q()
        ep = self.q[:self.epoch_len]
        del self.q[:self.epoch_len]
        return iter(ep)


class SubsetSampler(samplers.Sampler):
    def __init__(self, split, dataset, classes=None, n_per_class=None,
                 nval_per_class=None, seed=0, lab_unl_ratio=(0, 0)):

        rs = np.random.RandomState(seed)
        if isinstance(classes, (list, tuple)):
            class_indices = classes
        else:
            sel_classes = classes or dataset.N_CLASSES
            class_indices = rs.choice(dataset.N_CLASSES, size=sel_classes,
                                      replace=False)
            
        class2ind = {k: i for i, k in enumerate(class_indices)}
        class2ind[-100] = -100
        dataset.target_transform = (lambda k: class2ind[int(k)])

        assert split == 'test' or nval_per_class 
        self.indices = []
        for k in class_indices:
            ind_k = np.where(np.array(dataset.TARGETS) == k)[0]
            if n_per_class:
                ind_k = rs.choice(ind_k, n_per_class, replace=False)
                
            if split == 'test':
                self.indices += list(ind_k)
            elif split == 'val':
                i_val = rs.choice(ind_k, int(nval_per_class), replace=False)
                self.indices += list(i_val)
            elif split == 'train':
                i_val = rs.choice(ind_k, int(nval_per_class), replace=False)
                self.indices += list(set(ind_k) - set(i_val))

        self.indices_unl = []
        if split == 'train':
            ind_100 = np.where(np.array(dataset.TARGETS) == -100)[0]
            self.indices_unl = list(ind_100)

        if self.indices_unl:
            assert any(lab_unl_ratio), 'Specify lab_unl_ratio!'
            self.r_lab, self.r_unl = lab_unl_ratio
            n_lab = len(self.indices)
            n_unl = len(self.indices_unl)
            n_tup = min(n_lab//self.r_lab, n_unl//self.r_unl)
            self.n_lab = n_tup * self.r_lab
            self.n_unl = n_tup * self.r_unl

    def __len__(self):
        return len(self.indices) + self.n_unl

    def __iter__(self):
        if not self.indices_unl or self.r_unl == 0:
            out = np.random.permutation(self.indices)
        elif self.r_lab == 0:
            out = np.random.permutation(self.indices_unl)
        else:
            ind_lab = np.random.permutation(self.indices)[:self.n_lab]
            ind_unl = np.random.permutation(self.indices_unl)[:self.n_unl]
            ind_lab = ind_lab.reshape(self.r_lab, -1)
            ind_unl = ind_unl.reshape(self.r_unl, -1)
            out = np.vstack([ind_lab, ind_unl]).T.flatten()
        return iter(out)
    
#######################################################

class _FromDataset(DataLoader):
    Dset = None
    tforms = None
    target_tform = None
    
    def __init__(self, split, batch_size, gpu, classes=None,
                 nval_per_class=None, resize=None, crop=None, seed=0,
                 n_workers=4):
        
        tforms = []
        if resize: tforms += [transforms.Resize(resize)]
        if crop: tforms += [transforms.CenterCrop(crop)]
        tform = transforms.Compose(tforms + self.tforms)
        
        dataset = self.Dset(str(DATA_DIR), split in ['train', 'val'],
                               tform, self.target_tform)
        sampler = SubsetSampler(split, dataset, classes,
                                nval_per_class=nval_per_class, seed=seed)
        super().__init__(dataset, batch_size, sampler=sampler, drop_last=True,
                         num_workers=n_workers)

    
class Mnist(_FromDataset):
    Dset = MnistDataset
    tforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    
class BinarisedMnist(_FromDataset):
    Dset = MnistDataset
    tforms = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.round())
    ]


class BinaryMnist(_FromDataset):
    Dset = MnistDataset
    tforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    target_tform = (lambda y: y % 2)
    
    
class BlurryCifar100(_FromDataset):
    Dset = Cifar100Dataset
    tforms = [
        transforms.Lambda(blur_transform),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

###############################    

class _Cifar(DataLoader):
    def __init__(self, split, batch_size, gpu, augment=False, classes=None,
                 n_per_class=None, nval_per_class=None, n_unl_per_class=0,
                 lab_unl_ratio=(0, 0), seed=0):
            
        if augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                PadCrop(4, 32),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

        dataset = self.Dset(str(DATA_DIR), split in ['train', 'val'], transform,
                            self.target_tform)
        dataset2semisup(dataset, n_unl_per_class, seed)
        sampler = SubsetSampler(split, dataset, classes, n_per_class,
                                nval_per_class, seed, lab_unl_ratio)
        super().__init__(dataset, batch_size, sampler=sampler, drop_last=True)

        
class Cifar10(_Cifar):
    Dset = Cifar10Dataset
    mean = (0.491, 0.482, 0.447)
    std = (0.247, 0.244, 0.262)
    target_tform = None

    def id2label(self, i):
        meta = utils.unpickle(DATA_DIR / 'cifar-10-batches-py' / 'batches.meta')
        return meta[b'label_names'][i]

    
class BinaryCifar10(_Cifar):
    Dset = Cifar10Dataset
    mean = (0.491, 0.482, 0.447)
    std = (0.247, 0.244, 0.262)
    target_tform = (lambda y: 1 if y in [0, 1, 8, 9] else 0)

    
class Cifar100(_Cifar):
    Dset = Cifar100Dataset
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)
    target_tform = None

    def id2label(self, i):
        meta = utils.unpickle(DATA_DIR / 'cifar-100-python' / 'meta')
        return meta[b'fine_label_names'][i]

    
class BinaryCifar100(_Cifar):
    Dset = Cifar100Dataset
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)
    target_tform = (lambda y: 1 if y in
                    [1, 2, 3, 4, 6, 7, 11, 14, 15, 18, 19, 21, 24, 26, 27, 29,
                     30, 31, 32, 34, 35, 36, 38, 42, 43, 44, 45, 46, 50, 55, 63,
                     64, 65, 66, 67, 72, 73, 74, 75, 77, 78, 79, 80, 88, 91, 93,
                     95, 97, 98, 99] else 0)

class ImageNet(DataLoader):
    def __init__(self, split, batch_size, gpu, classes=None, n_per_class=None,
                 nval_per_class=None, n_unl_per_class=0, lab_unl_ratio=(0, 0),
                 seed=0, epoch_len=None, n_workers=16):

        if split == 'train':
            eigval = [0.2175, 0.0188, 0.0045]
            eigvec = [[-0.5675,  0.7192,  0.4009],
                      [-0.5808, -0.0045, -0.8140],
                      [-0.5836, -0.6948,  0.4203]]
            
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(to_rgb_transform),
                transforms.ToTensor(),
                Lighting(0.1, eigval, eigvec),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif split in ['test', 'val']:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Lambda(to_rgb_transform),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        
        dataset = (ImageNetTrnDataset(transform) if split in ['train', 'val']
                   else ImageNetTestDataset(transform))
        dataset2semisup(dataset, n_unl_per_class, seed)
        sampler = SubsetSampler(split, dataset, classes, n_per_class,
                                nval_per_class, seed, lab_unl_ratio)
        if epoch_len:
            sampler = SplitEpochSampler(sampler, epoch_len)
        super().__init__(dataset, batch_size, sampler=sampler, drop_last=True,
                         num_workers=n_workers)

#####################################################
# Custom DataLoaders:
# __init__(split, batch_size, gpu)
# __iter__(self)
# __len__(self)
#
####################
        
class RandomData(DataLoader):
    def __init__(self, split, batch_size, gpu, dim, n_classes, n_batches=0):
        self.batch_size = batch_size
        self.n_batches = n_batches or (500 if split == 'train' else 1)
        self.dim = dim
        self.n_classes = n_classes

    def __iter__(self):
        for i in range(self.n_batches):
            X = torch.rand(self.batch_size, *self.dim) * 2.0 - 1.0
            y = np.random.choice(self.n_classes, self.batch_size)
            y = torch.LongTensor(y)
            yield X, y

    def __len__(self):
        return self.n_batches
            
#########################################################

def binary_accuracy(logit, y, apply_sigmoid=True, reduce=True) -> float:
    prob = logit.sigmoid() if apply_sigmoid else logit
    pred = prob.round().long().view(-1)
    return (pred == y).float().mean() if reduce else (pred == y).float()


def multiclass_accuracy(scores, y, reduce=True):
    _, pred = scores.max(dim=1)
    return (pred == y).float().mean() if reduce else (pred == y).float()


def multiclass_accuracies(scores, y, tops: Tuple[int]):
    _, pred = scores.topk(k=max(tops), dim=1)
    labelled = (y != -100)
    if not any(labelled):
        return [1.0 for i in tops]
    hit = (pred[labelled] == y[labelled, None])
    topk_acc = hit.float().cumsum(dim=1).mean(dim=0)
    return [topk_acc[i-1] for i in tops]

#########################################################

class _Loss:
    def __call__(self, net, *args):
        raise NotImplementedError

    def metrics(self, net, *args):
        "Returns list. First element is used for comparison (higher = better)"
        raise NotImplementedError

    def trn_metrics(self):
        "Metrics from last call."
        raise NotImplementedError

    metric_names = []

    
class _MultiExitAccuracy(_Loss):
    def __init__(self, n_exits, acc_tops=(1,), _binary_clf=False):
        self.n_exits = n_exits
        self._binary_clf = _binary_clf
        self._acc_tops = acc_tops
        self._cache = dict()
        self.metric_names = [f'acc{i}_avg' for i in acc_tops]
        for i in acc_tops:
            self.metric_names += [f'acc{i}_clf{k}' for k in range(n_exits)]
            self.metric_names += [f'acc{i}_ens{k}' for k in range(1, n_exits)]
        self.metric_names += ['avg_maxprob']

    def __call__(self, net, *args):
        raise NotImplementedError

    def _metrics(self, logits_list, y):
        ensemble = torch.zeros_like(logits_list[0])
        acc_clf = np.zeros((self.n_exits, len(self._acc_tops)))
        acc_ens = np.zeros((self.n_exits, len(self._acc_tops)))
        
        for i, logits in enumerate(logits_list):
            if self._binary_clf:
                ensemble = ensemble*i/(i+1) + F.sigmoid(logits)/(i+1)
                acc_clf[i] = binary_accuracy(logits, y)
                acc_ens[i] = binary_accuracy(ensemble, y, apply_sigmoid=False)
            else:
                ensemble += F.softmax(logits, dim=1)
                acc_clf[i] = multiclass_accuracies(logits, y, self._acc_tops)
                acc_ens[i] = multiclass_accuracies(ensemble, y, self._acc_tops)
                
        maxprob = F.softmax(logits_list[-1].data, dim=1).max(dim=1)[0].mean()

        out = list(acc_clf.mean(axis=0))
        for i in range(acc_clf.shape[1]):
            out += list(acc_clf[:, i])
            out += list(acc_ens[1:, i])
        return out + [maxprob]

    def metrics(self, net, X, y, *args):
        logits_list = net.train(False)(X)
        return self._metrics(logits_list, y)

    def trn_metrics(self):
        return self._metrics(self._cache['logits_list'], self._cache['y'])
    
    
class ClassificationOnlyLoss(_MultiExitAccuracy):
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = net(X)
        self._cache['y'] = y
        return sum(F.cross_entropy(logits, y)
                   for logits in self._cache['logits_list'])

    
class DistillationBasedLoss(_MultiExitAccuracy):    
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.05,
                 global_scale=1.0):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.global_scale = global_scale
        
        self.metric_names += ['adj_maxprob', 'temperature'] 

    def __call__(self, net, X, y, *args):
        logits_list = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = (1.0 - self.C) * F.cross_entropy(logits_list[-1], y)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        for logits in logits_list[:-1]:
            logprob_s = F.log_softmax(logits / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        self._cache['adj_maxprob'] = adj_maxprob = prob_t.max(dim=1)[0].mean()
        if adj_maxprob > self.maxprob:
            self.T *= self.Tmult
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob, self.T]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        return out + [self._cache['adj_maxprob'], self.T]


class DistillationLossConstTemp(_MultiExitAccuracy):
    def __init__(self, C, T, n_exits, acc_tops=(1,), global_scale=1.0):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.T = T
        self.global_scale = global_scale
        self.metric_names += ['adj_maxprob']

    def __call__(self, net, X, y, *args):
        logits_list = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = (1.0 - self.C) * F.cross_entropy(logits_list[-1], y)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        self._cache['prob_t'] = prob_t
        for logits in logits_list[:-1]:
            logprob_s = F.log_softmax(logits / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        adj_maxprob = self._cache['prob_t'].max(dim=1)[0].mean()
        return out + [adj_maxprob]
    

############    

def ConvBnRelu2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(*self.shape)

####################

class _Pretrained(nn.Module):
    Model = None
    
    def __init__(self, pretrained, **kwargs):
        super().__init__()
        self._net = self.__class__.Model(pretrained, **kwargs)

    def forward(self, x):
        return self._net(x)

    def trace(self, x, keep_layers):
        x = TransientDict(x=x, _keep=keep_layers)
        x['logits'] = self._net(x[-1])
        return x

    
class AlexNet(_Pretrained):
    Model = torchvision.models.alexnet

    
class Resnet18(_Pretrained):
    Model = torchvision.models.resnet18
    
    def __init__(self, pretrained, reset_clf=False, **kwargs):
        super().__init__(pretrained, **kwargs)
        if reset_clf:
            self._net.fc.weight.data.normal_(0.0, 0.02)
            self._net.fc.bias.data.fill_(0.0)

            
class Resnet152(_Pretrained):
    Model = torchvision.models.resnet152
    
####################
    
class _TraceInForward(nn.Module):
    def forward(self, x, keep_layers=()):
        raise NotImplementedError
        self._trace = TransientDict()
    
    def trace(self, *inputs, keep_layers, **kw_inputs):
        self.forward(*inputs, keep_layers, **kw_inputs)
        return self._trace



class MsdJoinConv(nn.Module):
    def __init__(self, n_in, n_in_down, n_out, btneck, btneck_down):
        super().__init__()
        if n_in_down:
            assert n_out % 2 == 0
            n_out //= 2
            self.conv_down = self._btneck(n_in_down, n_out, stride=2,
                                          btneck=btneck_down)
        self.conv = self._btneck(n_in, n_out, stride=1, btneck=btneck)

    def _btneck(self, n_in, n_out, stride, btneck:int):
        if btneck:
            n_mid = min(n_in, btneck * n_out)
            return nn.Sequential(
                ConvBnRelu2d(n_in, n_mid, 1),
                ConvBnRelu2d(n_mid, n_out, 3, stride=stride, padding=1))
        else:
            return ConvBnRelu2d(n_in, n_out, 3, stride=stride, padding=1)

    def forward(self, x1, x2, x_down):
        out = [x1, self.conv(x2)]
        out += [self.conv_down(x_down)] if x_down is not None else []
        return torch.cat(out, dim=1)
    

class MsdLayer0(nn.Module):
    def __init__(self, nplanes_list, in_shape):
        super().__init__()
        in_channels = 3
        self.mods = nn.ModuleList()
        
        if in_shape == 32:
            self.mods += [ConvBnRelu2d(in_channels, nplanes_list[0],
                                       kernel_size=3, padding=1)]
        elif in_shape == 224:
            conv = ConvBnRelu2d(in_channels, nplanes_list[0],
                                kernel_size=7, stride=2, padding=3)
            pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.mods += [nn.Sequential(conv, pool)]

        for i in range(1, len(nplanes_list)):
            self.mods += [ConvBnRelu2d(nplanes_list[i-1], nplanes_list[i],
                                       kernel_size=3, stride=2, padding=1)]
    def forward(self, x):
        out = [x]
        for i in range(len(self.mods)):
            out += [self.mods[i](out[i])]
        return out[1:]

    
class MsdLayer(nn.Module):
    def __init__(self, nplanes_tab, btneck_widths):
        super().__init__()
        in_scales, out_scales = nplanes_tab.astype(bool).sum(axis=0)
        assert in_scales - out_scales <= 1
        
        if not btneck_widths:
            btneck_widths = [None] * len(nplanes_tab)
    
        self.mods = nn.ModuleList()
        for i, (n_in, n_out) in enumerate(nplanes_tab):
            n_in_prev = nplanes_tab[i-1, 0] if i else 0
            btneck_width_prev = btneck_widths[i-1] if i else None
            self.mods += [MsdJoinConv(n_in, n_in_prev, n_out - n_in,
                                      btneck_widths[i], btneck_width_prev)
                          if n_out else None]
            
    def forward(self, x):
        out = []
        for i, m in enumerate(self.mods):
            x_down = None if i == 0 else x[i-1]
            out += [m(x[i], x[i], x_down) if m else None]
        return out


class MsdTransition(nn.Module):
    def __init__(self, nplanes_tab):
        super().__init__()
        self.conv = nn.ModuleList([ConvBnRelu2d(n_in, n_out, kernel_size=1)
                                   if n_in else None
                                   for n_in, n_out in nplanes_tab])
    def forward(self, inputs):
        return [m(x) if m else None for m, x in zip(self.conv, inputs)]

    
class MsdNet(_TraceInForward):
    def __init__(self, in_shape, out_dim, n_scales, n_exits, nlayers_to_exit,
                 nlayers_between_exits, nplanes_mulv:List[int],
                 nplanes_addh:int, nplanes_init=32, prune=None,
                 plane_reduction=0.0, exit_width=None, btneck_widths=(),
                 execute_exits=None):

        super().__init__()
        assert nlayers_to_exit >= nlayers_between_exits
        self.n_exits = n_exits
        self.out_dim = out_dim
        self.exit_width = exit_width
        self._execute_exits = (execute_exits if execute_exits is not None
                               else range(n_exits))
        
        block_nlayers = [nlayers_to_exit] + [nlayers_between_exits]*(n_exits-1)
        n_layers = 1 + sum(block_nlayers)
        nplanes_tab = self.nplanes_tab(n_scales, n_layers, nplanes_init,
                                       nplanes_mulv, nplanes_addh, prune,
                                       plane_reduction)
        self._nplanes_tab = nplanes_tab
        self._block_sep = block_sep = self.block_sep(block_nlayers, n_scales,
                                                    prune, plane_reduction)
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()
        for i in range(n_exits):
            self.blocks[i] = self.Block(
                nplanes_tab[:, block_sep[i]-1:block_sep[i+1]],
                in_shape if i == 0 else 0, btneck_widths)
            self.exits[i] = self.Exit(nplanes_tab[-1,block_sep[i+1]-1],
                                      out_dim, exit_width)
        self.init_weights()

    def Block(self, nplanes_tab, layer0_size, btneck_widths):
        block = []
        if layer0_size:
            block = [MsdLayer0(nplanes_tab[:,0], layer0_size)]
        for i in range(1, nplanes_tab.shape[1]):
            block += [MsdLayer(nplanes_tab[:,i-1:i+1], btneck_widths)
                      if nplanes_tab[-1,i-1] < nplanes_tab[-1,i] else
                      MsdTransition(nplanes_tab[:,i-1:i+1])]
        return nn.Sequential(*block)

    def Exit(self, n_channels, out_dim, inner_channels=None):
        inner_channels = inner_channels or n_channels
        return nn.Sequential(
            ConvBnRelu2d(n_channels, inner_channels, kernel_size=3,
                         stride=2, padding=1),
            ConvBnRelu2d(inner_channels, inner_channels, kernel_size=3,
                         stride=2, padding=1),
            nn.AvgPool2d(kernel_size=2),
            View(-1, inner_channels),
            nn.Linear(inner_channels, out_dim),
        )

    def block_sep(self, block_nlayers, n_scales, prune, plane_reduction):
        n_layers = 1 + sum(block_nlayers)
        reduce_layers = self._reduce_layers(n_scales, n_layers, prune,
                                           plane_reduction)
        sep = np.cumsum([1] + block_nlayers)
        shift = np.zeros_like(sep)
        for i in reduce_layers:
            shift += (sep >= i)
        return sep + shift

    def nplanes_tab(self, n_scales, n_layers, nplanes_init, nplanes_mulv,
                    nplanes_addh, prune, plane_reduction):
        
        reduce_layers = self._reduce_layers(n_scales, n_layers, prune,
                                           plane_reduction)
        nprune_per_layer = self._nprune_per_layer(n_scales, n_layers, prune)
        hbase, nprune = [nplanes_init], [0]
        for i in range(1, n_layers):
            hbase += [hbase[-1] + nplanes_addh]
            nprune += [nprune_per_layer[i]]
            if i in reduce_layers:
                hbase += [math.floor(hbase[-1] * plane_reduction)]
                nprune += [nprune_per_layer[i]]
                
        planes_tab = np.outer(nplanes_mulv, hbase)
        for i in range(len(hbase)):
            planes_tab[:nprune[i], i] = 0
        return planes_tab

    def _reduce_layers(self, n_scales, n_layers, prune, plane_reduction):
        if not plane_reduction:
            return []
        elif prune == 'min':
            return [math.floor((n_layers-1)*1/3),
                    math.floor((n_layers-1)*2/3)]
        elif prune == 'max':
            interval = math.ceil((n_layers-1) / n_scales)
            return list(range(interval+1, n_layers, interval))

    def _nprune_per_layer(self, n_scales, n_layers, prune):
        if prune == 'min':
            nprune = min(n_scales, n_layers) - np.arange(n_layers, 0, -1)
            return list(np.maximum(0, nprune))
        elif prune == 'max':
            interval = math.ceil((n_layers-1) / n_scales)
            return [0] + [math.floor(i/interval) for i in range(n_layers-1)]
        else:
            return [0] * n_layers
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0.0, math.sqrt(2/n))
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

    def forward(self, x, keep_layers=()):
        max_block = max(self._execute_exits)
        logits = []
        x = TransientDict(x=x, _keep=keep_layers)
        for i in range(max_block+1):
            h = self.blocks[i](x[-1])
            x[f'block{i}'] = h
            if i in self._execute_exits:
                logits += [self.exits[i](h[-1])]
            else:
                logits += [()]
        x[-1]
        x['logits'] = logits
        self._trace = x
        return x['logits']

