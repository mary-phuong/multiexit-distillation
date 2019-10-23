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

def attribution_by_occlusion(X, net, gpu, occl_size, occl_val=0.0):
    "X: single image"
    c, w, h = X.shape
    net.to(device(gpu))

    Xs = [X]
    for i in range(0, w, occl_size):
        for j in range(0, h, occl_size):
            x = X.clone()
            x[:, i:i+occl_size, j:j+occl_size] = occl_val
            Xs += [x]

    Xs = torch.stack(Xs).to(device(gpu))
    logits = net.train(False)(Xs).data
    pred = logits[0].max(dim=0)[1][0]

    out_vec = logits[1:,pred]
    out = torch.zeros(w, h)
    k = 0
    for i in range(0, w, occl_size):
        for j in range(0, h, occl_size):
            out[i:i+occl_size, j:j+occl_size] = out_vec[k]
            k += 1
    return out


def integrated_gradients(X, net, gpu, n_grid=51):
    "X: single image"
    net.to(device(gpu))
    
    X_interp = [lam*X for lam in np.linspace(0.0, 1.0, n_grid)]
    X = torch.stack(X_interp).to(device(gpu)).requires_grad_()

    logits = net.train(False)(X)
    pred = logits[-1].max(dim=0)[1][0]

    logits[:,pred].sum().backward()
    return X.grad.mean(dim=0)


#########################################################

class GuidedBpRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0.0)

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_variables
        return (x >= 0).float() * grad_y.clamp(min=0.0)

guided_bp_relu = GuidedBpRelu.apply


class ZeilerFergusRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clamp(min=0.0)

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y.clamp(min=0.0)

zeiler_fergus_relu = ZeilerFergusRelu.apply
    
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

class GaussianMixtureDataset(Dataset):
    def __init__(self, n_per_class, means:List[np.array],
                 covs:List[np.array]=None, seed=0, target_transform=None):
        
        self.N_CLASSES = len(means)
        self.n_per_class = n_per_class
        self.target_transform = target_transform or (lambda y: y)

        dim = len(means[0])
        covs = covs or [np.eye(dim) for k in range(self.N_CLASSES)]

        self.TARGETS = np.repeat(range(self.N_CLASSES), n_per_class)
        rs = np.random.RandomState(seed)
        self.X = [rs.multivariate_normal(
                      means[i], covs[i], size=n_per_class).astype(np.float32)
                  for i in range(self.N_CLASSES)]

    def __len__(self):
        return len(self.TARGETS)

    def __getitem__(self, i):
        k = i // self.n_per_class
        j = i % self.n_per_class
        return self.X[k][j], self.target_transform(k)


class BinaryGmDataset(GaussianMixtureDataset):
    def __init__(self, n_per_class, means:List[np.array or str], cov='',
                 shift=0.0, boundary=False, dir_vector='',
                 scale=1.0, seed=0, target_transform=None):
        
        assert len(means) == 2
        if isinstance(means[0], str):
            means[0] = np.load(DATA_DIR/means[0])
            means[1] = np.load(DATA_DIR/means[1])
        if isinstance(shift, str):
            shift = np.load(DATA_DIR/shift)
        means = [np.array(m) + shift for m in means]
        
        if cov:
            cov = np.load(DATA_DIR/cov)
            covs = [cov, cov]
        else:
            covs = None
        super().__init__(n_per_class, means, covs, seed, target_transform)
        
        if dir_vector:
            self.dir_vector = np.load(DATA_DIR/dir_vector)
        else:
            self.dir_vector = means[1] - means[0]
        self.dir_vector /= np.linalg.norm(self.dir_vector)
        
        self.boundary = self._project(sum(means)/2.0) if boundary else None
        self.scale = scale

    def _project(self, x):
        return np.array([(x*self.dir_vector).sum()], dtype=np.float32)

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        if self.boundary is None:
            return x, y
        else:
            return x, y, (self._project(x)-self.boundary)*self.scale


class IrisDataset(Dataset):
    def __init__(self, features:List[str]=None, norm=True,
                 target_transform=None):
        self.N_CLASSES = 3
        self.d = pd.read_csv(DATA_DIR/'iris.csv', sep='\t')
        self.TARGETS = self.d.y
        self.target_transform = target_transform or (lambda y: y)
        
        if features is not None:
            self.d = self.d[features + ['y']]
        if norm:
            for col in self.d.columns[:-1]:
                self.d[col] -= self.d[col].min()
                self.d[col] = self.d[col] / self.d[col].max() * 2.0 - 1.0

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        *x, y = self.d.iloc[i]
        return np.array(x, dtype=np.float32), self.target_transform(int(y))


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


class Cifar10LabelInceptionDataset(Dataset):
    def __init__(self, snapshot, gpu, n_iter=5000, lr=0.01,
                 target_transform=None):
        self.N_CLASSES = 10
        self.TARGETS = range(10)
        self.target_transform = target_transform or (lambda y: y)
        
        net = load_snapshot(snapshot).to(device(gpu))

        x = torch.zeros(10, 3, 32, 32)
        x = x.to(device(gpu)).requires_grad_()
        opt = torch.optim.SGD([x], lr=lr)

        for j in range(n_iter):
            opt.zero_grad()
            logits = net.train(False)(x)
            loss = -sum(logits[i][i] for i in range(10))
            loss.backward()
            opt.step()

        self.X = x.data.cpu()

    def __len__(self):
        return 10

    def __getitem__(self, i):
        return self.X[i], self.target_transform(i)

    
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

class GaussianMixture(DataLoader):
    def __init__(self, split, batch_size, gpu, n_per_class, means, covs=None,
                 seed=0):
        dataset = GaussianMixtureDataset(n_per_class, means, covs, seed)
        super().__init__(dataset, batch_size, shuffle=True, drop_last=True)


class BinaryGaussianMixture(DataLoader):
    def __init__(self, split, batch_size, gpu, n_per_class, means, cov='',
                 shift=0.0, boundary=False, dir_vector='', scale=1.0, seed=0):
        dataset = BinaryGmDataset(n_per_class, means, cov, shift, boundary,
                                  dir_vector, scale, seed)
        super().__init__(dataset, batch_size, shuffle=True, drop_last=True)
    

class Iris(DataLoader):
    def __init__(self, split, batch_size, gpu, nval_per_class=None,
                 features=None):
        
        assert split in ['train', 'val']
        dataset = IrisDataset(features)
        sampler = SubsetSampler(split, dataset, nval_per_class=nval_per_class)
        super().__init__(dataset, batch_size, sampler=sampler, drop_last=True)


class Cifar10LabelInception(DataLoader):
    def __init__(self, split, batch_size, gpu, snapshot, n_iter=5000, lr=0.01):
        dataset = Cifar10LabelInceptionDataset(snapshot, gpu, n_iter, lr)
        super().__init__(dataset, batch_size, shuffle=True, drop_last=True)


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

            
class RandomConvData(DataLoader):
    def __init__(self, split, batch_size, gpu, n_classes, dim_z, shape,
                 net_size):
        self.batch_size = batch_size
        self.n_batches = 500 if split == 'train' else 1
        self.n_classes = n_classes
        self.gpu = gpu
        self.dim_z = dim_z
        self.conv = GConvNet(dim_z, shape, net_size).to(device(gpu))
        for par in self.conv.parameters():
            par.requires_grad = False
        
    def __iter__(self):
        for i in range(self.n_batches):
            z = torch.rand(self.batch_size, self.dim_z).to(device(self.gpu))
            X = self.conv.train(False)(z).tanh().data.cpu()
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


class BinaryCrossEntropyLoss(_Loss):
    def __call__(self, net, X, y, *args):
        self._logit = net(X)
        self._y = y
        return F.binary_cross_entropy(
            self._logit.view(-1).sigmoid(), y.float())

    def metrics(self, net, X, y, *args):
        logit = net.train(False)(X)
        return [binary_accuracy(logit, y)]

    def trn_metrics(self):
        return [binary_accuracy(self._logit, self._y)]

    metric_names = ['acc']


class ConditionalBinaryCrossEntropyLoss(_Loss):
    def __call__(self, net, X, y_target, y_cond, *args):
        self._logit = net(X, y_cond)
        self._y_target = y_target
        return F.binary_cross_entropy(
            self._logit.view(-1).sigmoid(), y_target.float())

    def metrics(self, net, X, y_target, y_cond, *args):
        logit = net.train(False)(X, y_cond)
        return [binary_accuracy(logit, y_target)]

    def trn_metrics(self):
        return [binary_accuracy(self._logit, self._y_target)]
    
    
class _TopKAccuracy(_Loss):
    def __init__(self, acc_tops=(1,)):
        self._cache = dict()
        self._acc_tops = acc_tops
        self.metric_names = ['acc{}'.format(i if i != 1 else '')
                             for i in acc_tops]

    def __call__(self, net, *args):
        raise NotImplementedError

    def metrics(self, net, X, y, *args):
        logits = net.train(False)(X)
        return multiclass_accuracies(logits, y, self._acc_tops)

    def trn_metrics(self):
        return multiclass_accuracies(
            self._cache['logits'], self._cache['y'], self._acc_tops)

    
class CrossEntropyLoss(_TopKAccuracy):
    def __call__(self, net, X, y, *args):
        self._cache['logits'] = net(X)
        self._cache['y'] = y
        return F.cross_entropy(self._cache['logits'], y)


class FakeCrossEntropyLoss(_TopKAccuracy):
    def __call__(self, net, X, y, *args):
        self._cache['logits'] = logits = net(X)
        self._cache['y'] = y
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(dim=1, index=y[:,None], value=1.0)

        sum_exp = torch.exp(logits).sum(dim=1, keepdim=True)
        eta = torch.log(sum_exp - torch.exp(logits))
        p = (logits - eta.data).sigmoid()
        
        losses = F.binary_cross_entropy(p, y_onehot, reduce=False)
        return losses.sum(dim=1).mean()


class SigmoidCrossEntropyLoss(_TopKAccuracy):
    def __init__(self, n_classes, acc_tops=(1,)):
        super().__init__(acc_tops)
        self.metric_names += [f'ovr_acc{i}' for i in range(n_classes)]
    
    def __call__(self, net, X, y, *args):
        self._cache['logits'] = logits = net(X)
        self._cache['y'] = y
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(dim=1, index=y[:,None], value=1.0)

        p = logits.sigmoid()
        losses = F.binary_cross_entropy(p, y_onehot, reduce=False)
        return losses.sum(dim=1).mean()

    def metrics(self, net, X, y, *args):
        logits = net.train(False)(X)
        out = multiclass_accuracies(logits, y, self._acc_tops)
        out += [binary_accuracy(logits[:,i], (y==i).long())
                for i in range(logits.shape[1])]
        return out

    def trn_metrics(self):
        logits = self._cache['logits']
        y = self._cache['y']
        out = multiclass_accuracies(logits, y, self._acc_tops)
        out += [binary_accuracy(logits[:,i], (y==i).long())
                for i in range(logits.shape[1])]
        return out
    

class LabelSmoothingLoss(_TopKAccuracy):
    def __init__(self, T, C, conf, acc_tops=(1,)):
        super().__init__(acc_tops)
        self.T = T
        self.C = C
        self.conf = conf
    
    def __call__(self, net, X, y, *args):
        logits_s = net(X)
        n, k = logits_s.shape
        self._cache['logits'] = logits_s
        self._cache['y'] = y

        logprob_s = F.log_softmax(logits_s/self.T, dim=1)
        prob_t = torch.zeros_like(logits_s) + (1-self.conf)/(k-1)
        prob_t.scatter_(dim=1, index=y[:,None], value=self.conf)

        dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
        cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
        return ((1.0-self.C) * cross_ent + (self.T**2) * self.C * dist_loss)


class SigmoidLabelSmoothingLoss(SigmoidCrossEntropyLoss):
    def __init__(self, n_classes, conf, acc_tops=(1,)):
        super().__init__(n_classes, acc_tops)
        self.conf = conf

    def __call__(self, net, X, y, *args):
        self._cache['logits'] = logits = net(X)
        self._cache['y'] = y

        n, k = logits.shape
        prob_t = torch.zeros_like(logits) + (1.0-self.conf)/(k-1)
        prob_t.scatter_(dim=1, index=y[:,None], value=self.conf)

        p = logits.sigmoid()
        losses = F.binary_cross_entropy(p, prob_t, reduce=False)
        return losses.sum(dim=1).mean()
    
    
class MixupCrossEntropyLoss(_TopKAccuracy):
    def __init__(self, alpha, n_classes, acc_tops=(1,)):
        super().__init__(acc_tops)
        self.alpha = alpha
        self.n_classes = n_classes
    
    def __call__(self, net, X, y, *args):
        self._cache = dict(net=net, X=X, y=y)
        perm = torch.randperm(len(y)).to(X.device)
        
        lam = np.random.beta(self.alpha, self.alpha)
        X_mixup = lam*X + (1.0-lam)*X[perm]

        y_onehot = torch.eye(self.n_classes)[y.cpu()].to(X.device)
        y_mixup = lam*y_onehot + (1.0-lam)*y_onehot[perm]

        p_hat = F.log_softmax(net(X_mixup), dim=1)
        return -(y_mixup*p_hat).sum(dim=1).mean()

    def trn_metrics(self):
        logits = self._cache['net'](self._cache['X'])
        return multiclass_accuracies(logits, self._cache['y'], self._acc_tops)

    
class DistillationLoss(_Loss):
    def __init__(self, T, C, acc_tops=(1,)):
        self._cache = dict()
        self._acc_tops = acc_tops
        self.metric_names = ['acc{}'.format(i if i != 1 else '')
                             for i in acc_tops] + ['teacher_match']
        self.T = T
        self.C = C

    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y
        
        logprob_s = F.log_softmax(logits_s / self.T, dim=1)
        prob_t = F.softmax(logits_t / self.T, dim=1)

        dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
        cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
        return ((1.0 - self.C) * cross_ent + (self.T ** 2) * self.C * dist_loss)

    def metrics(self, net, X, y, logits_t, *args):
        logits = net.train(False)(X)
        acc = multiclass_accuracies(logits, y, self._acc_tops)
        match = (logits.max(dim=1)[1] == logits_t.max(dim=1)[1]).float().mean()
        return acc + [match]
    
    def trn_metrics(self):
        acc = multiclass_accuracies(
            self._cache['logits'], self._cache['y'], self._acc_tops)
        _, pred_s = self._cache['logits'].max(dim=1)
        _, pred_t = self._cache['logits_t'].max(dim=1)
        match = (pred_s == pred_t).float().mean()
        return acc + [match]


class ControlledDistillationLoss(DistillationLoss):
    def __init__(self, T, C, acc_tops=(1,), target='dist', conf=1.0):
        super().__init__(T, C, acc_tops)
        self.target = target
        self.conf = conf

    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y
        n, k = logits_s.shape
        
        logprob_s = F.log_softmax(logits_s / self.T, dim=1)
        if self.target == 'dist':
            prob_t = F.softmax(logits_t / self.T, dim=1)
        elif self.target == 'y':
            prob_t = torch.eye(k, device=y.device)[y]
        elif self.target == 'y_smooth':
            prob_t = torch.zeros_like(logits_s) + (1.0-self.conf)/(k-1)
            prob_t.scatter_(dim=1, index=y[:,None], value=self.conf)
        elif self.target == 'dist_mode':
            prob_t = F.softmax(logits_t / self.T, dim=1)
            maxprob_t, pred_t = prob_t.max(dim=1, keepdim=True)
            prob_t = torch.zeros_like(logits_s) + (1.0-maxprob_t)/(k-1)
            prob_t.scatter_(dim=1, index=pred_t, src=maxprob_t)

        dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
        cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
        return ((1.0 - self.C) * cross_ent + (self.T ** 2) * self.C * dist_loss)


class BinaryDistillationLoss(_Loss):
    metric_names = ['acc', 'teacher_match']
    
    def __init__(self, T, C):
        self._cache = dict()
        self.T = T
        self.C = C
    
    def __call__(self, net, X, y, logits_t, *args):
        self._cache['logits_s'] = logits_s = net(X)
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y

        prob_t = F.sigmoid(logits_t.view(-1)/self.T)
        dist_loss = (bce_w_logits(logits_s.view(-1)/self.T, prob_t) -
                     bce_w_logits(logits_t.view(-1)/self.T, prob_t))
        cross_ent = bce_w_logits(logits_s.view(-1), y.float())
        cross_ent = 0.0 if self.C == 1.0 else cross_ent
        return ((1.0-self.C)*cross_ent + (self.T**2)*self.C*dist_loss)

    def metrics(self, net, X, y, logits_t, *args):
        return self._metrics(y, net.train(False)(X), logits_t)

    def trn_metrics(self):
        return self._metrics(self._cache['y'], self._cache['logits_s'],
                             self._cache['logits_t'])

    def _metrics(self, y, logits_s, logits_t):
        acc = binary_accuracy(logits_s, y)
        pred_t = logits_t.sigmoid().round().long()
        pred_s = logits_s.sigmoid().round().long()
        teacher_match = (pred_t == pred_s).float().mean()
        return [acc, teacher_match]
    

class ControlledBinaryDistillationLoss(BinaryDistillationLoss):
    def __init__(self, T, C, target='dist', conf=1.0):
        self._cache = dict()
        self.T = T
        self.C = C
        self.target = target
        self.conf = conf

    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits_s'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y

        if self.target == 'dist':
            prob_t = F.sigmoid(logits_t.view(-1)/self.T)
        elif self.target == 'dist_bin':
            prob_t = F.sigmoid(logits_t.view(-1)/self.T).round()
        elif self.target == 'y':
            prob_t = y.float()
        elif self.target == 'y_smooth':
            prob_t = y.float() * (2.0*self.conf-1.0) + (1.0-self.conf)

        dist_loss = bce_w_logits(logits_s.view(-1)/self.T, prob_t)
        cross_ent = bce_w_logits(logits_s.view(-1), y.float())
        cross_ent = 0.0 if self.C == 1.0 else cross_ent
        return ((1.0-self.C) * cross_ent + (self.T**2)*self.C * dist_loss)


class DoubleTempBinaryDistillationLoss(BinaryDistillationLoss):
    def __init__(self, T_teacher, T_student, C):
        self._cache = dict()
        self.T_teacher = T_teacher
        self.T_student = T_student
        self.C = C

    def __call__(self, net, X, y, logits_t, *args):
        self._cache['logits_s'] = logits_s = net(X)
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y

        prob_t = F.sigmoid(logits_t.view(-1)/self.T_teacher)
        dist_loss = (bce_w_logits(logits_s.view(-1)/self.T_student, prob_t) -
                     bce_w_logits(logits_t.view(-1)/self.T_teacher, prob_t))
        cross_ent = bce_w_logits(logits_s.view(-1), y.float())
        cross_ent = 0.0 if self.C == 1.0 else cross_ent
        return ((1.0-self.C)*cross_ent + (self.T_student**2)*self.C*dist_loss)
    

class BinaryL1DistillationLoss(BinaryDistillationLoss):
    def __call__(self, net, X, y, t, *args):
        self._cache['logits_s'] = net(X)
        self._cache['logits_t'] = t
        self._cache['y'] = y
        return (self._cache['logits_s']-t).abs().mean()


class BinaryL2DistillationLoss(BinaryDistillationLoss):
    def __call__(self, net, X, y, t, *args):
        self._cache['logits_s'] = net(X)
        self._cache['logits_t'] = t
        self._cache['y'] = y
        return ((self._cache['logits_s']-t)**2).mean()


class AgnosticDistillationLoss(DistillationLoss):
    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y

        logprob_s = F.log_softmax(logits_s/self.T, dim=1)
        prob_t = F.softmax(logits_s/self.T, dim=1).data
        _, pred_t = logits_t.max(dim=1)
        prob_t.scatter_(dim=1, index=pred_t[:,None], value=1.0)

        dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
        cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
        return ((1.0 - self.C) * cross_ent + (self.T ** 2) * self.C * dist_loss)
        
    
class OverconfidentDistillationLoss(DistillationLoss):
    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y

        logprob_s = F.log_softmax(logits_s/self.T, dim=1)
        prob_t = F.softmax(logits_t/self.T, dim=1)
        _, pred_t = logits_t.max(dim=1)
        prob_t.scatter_(dim=1, index=pred_t[:,None], value=1.0)

        dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
        cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
        return ((1.0 - self.C) * cross_ent + (self.T ** 2) * self.C * dist_loss)


class ModeDistillationLoss(DistillationLoss):
    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y
        n_classes = logits_s.shape[1]
        
        logprob_s = F.log_softmax(logits_s/self.T, dim=1)
        maxprob_t, pred_t = F.softmax(logits_t/self.T, dim=1).max(dim=1)
        fill_prob = (1.0-maxprob_t[:,None]) / (n_classes-1)
        target = torch.zeros_like(logits_t) + fill_prob
        target.scatter_(dim=1, index=pred_t[:,None], src=maxprob_t[:,None])

        dist_loss = -(target * logprob_s).sum(dim=1).mean()
        cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
        return (1.0-self.C)*cross_ent + (self.T**2)*self.C*dist_loss


class SigmoidDistillationLoss(DistillationLoss):
    def __init__(self, n_classes, T, acc_tops=(1,), target='dist', conf=1.0):
        super().__init__(T, None, acc_tops)
        self.metric_names += [f'ovr_acc{i}' for i in range(n_classes)]
        self.target = target
        self.conf = conf
        
    def __call__(self, net, X, y, logits_t, *args):
        logits_s = net(X)
        self._cache['logits'] = logits_s
        self._cache['logits_t'] = logits_t
        self._cache['y'] = y
        n, k = logits_s.shape

        if self.target == 'dist':
            prob_t = F.softmax(logits_t / self.T, dim=1)
        elif self.target == 'y':
            prob_t = torch.eye(k)[y.cpu()].to(y.device)
        elif self.target == 'y_smooth':
            prob_t = torch.zeros_like(logits_s) + (1-self.conf)/(k-1)
            prob_t.scatter_(dim=1, index=y[:,None], value=self.conf)
        elif self.target == 'dist_mode':
            prob_t = F.softmax(logits_t / self.T, dim=1)
            maxprob_t, pred_t = prob_t.max(dim=1, keepdim=True)
            prob_t = torch.zeros_like(logits_s) + (1.0-maxprob_t)/(k-1)
            prob_t.scatter_(dim=1, index=pred_t, src=maxprob_t)
            
        prob_s = (logits_s / self.T).sigmoid()
        losses = F.binary_cross_entropy(prob_s, prob_t, reduce=False)
        return losses.sum(dim=1).mean()

    def metrics(self, net, X, y, logits_t, *args):
        logits = net.train(False)(X)
        acc = multiclass_accuracies(logits, y, self._acc_tops)
        match = (logits.max(dim=1)[1] == logits_t.max(dim=1)[1]).float().mean()
        out = acc + [match]
        out += [binary_accuracy(logits[:,i], (y==i).long())
                for i in range(logits.shape[1])]
        return out

    def trn_metrics(self):
        out = super().trn_metrics()
        logits = self._cache['logits']
        y = self._cache['y']
        out += [binary_accuracy(logits[:,i], (y==i).long())
                for i in range(logits.shape[1])]
        return out
    

class L2DistillationLoss(_TopKAccuracy):
    def __call__(self, net, X, y, logits_t, *args):
        self._cache['logits'] = net(X)
        self._cache['y'] = y
        return ((self._cache['logits']-logits_t)**2).mean()
    
###################
    
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
    
    
class MultipleCrossEntropyLoss(_MultiExitAccuracy):
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = net(X)
        self._cache['y'] = y
        return sum(F.cross_entropy(logits, y)
                   for logits in self._cache['logits_list'])


class EnsembleCrossEntropyLoss(_MultiExitAccuracy):
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y
        logit_sum = 0.0
        ensemble_logits = []
        for i in range(self.n_exits):
            logit_sum += logits_list[i]
            ensemble_logits += [logit_sum/(i+1)]
        return sum(F.cross_entropy(logits, y) for logits in ensemble_logits)
            

class MultipleBinaryCrossEntropyLoss(_MultiExitAccuracy):
    def __init__(self, n_exits):
        super().__init__(n_exits, _binary_clf=True)

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = net(X)
        self._cache['y'] = y
        return sum(bce_w_logits(logits.view(-1), y.float())
                   for logits in self._cache['logits_list'])


class MultipleDistillationLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C
    
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        for logits in logits_list[:-1]:
            logprob_s = F.log_softmax(logits / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss
    

class MaxprobMultiDistLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.1,
                 ep_start=0, unif_weights=True):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.ep_start = ep_start
        self.unif_weights = unif_weights

    def __call__(self, net, X, y, ep, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        if ep < self.ep_start:
            return sum(F.cross_entropy(logits, y)
                       for logits in self._cache['logits_list'])
        else:
            cum_loss = F.cross_entropy(logits_list[-1], y)
            if not self.unif_weights:
                cum_loss *= (1.0 - self.C)
            prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
            for logits in logits_list[:-1]:
                logprob_s = F.log_softmax(logits / self.T, dim=1)
                dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
                cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
                cum_loss += (1.0 - self.C) * cross_ent
                cum_loss += (self.T ** 2) * self.C * dist_loss

            if prob_t.max(dim=1)[0].mean() > self.maxprob:
                self.T *= self.Tmult
            if not self.unif_weights:
                cum_loss /= (1.0 - self.C)
            return cum_loss


class MaxprobMultiFrzDistLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.05,
                 weight_last=False, global_scale=1.0, freeze=True):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.weight_last = weight_last
        self.global_scale = global_scale
        self.freeze = freeze
        
        self.metric_names += ['adj_maxprob', 'temperature'] 

    def __call__(self, net, X, y, *args):
        logits_list, logits_list_frz = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        if self.weight_last:
            cum_loss *= (1.0 - self.C)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        for logits, logits_frz in zip(logits_list[:-1], logits_list_frz[:-1]):
            if not self.freeze:
                logits_frz = logits
            logprob_s = F.log_softmax(logits_frz / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        self._cache['adj_maxprob'] = adj_maxprob = prob_t.max(dim=1)[0].mean()
        if adj_maxprob > self.maxprob:
            self.T *= self.Tmult
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list, logits_list_frz = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob, self.T]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        return out + [self._cache['adj_maxprob'], self.T]


class MaxprobMultiFrzDistNextLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.05,
                 weight_last=False, global_scale=1.0, freeze=True):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.weight_last = weight_last
        self.global_scale = global_scale
        self.freeze = freeze
        
        self.metric_names += ['adj_maxprob', 'temperature'] 

    def __call__(self, net, X, y, *args):
        logits_list, logits_list_frz = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        if self.weight_last:
            cum_loss *= (1.0 - self.C)
        prob_list = [F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list]
        for i, (logits, logits_frz) in enumerate(
                zip(logits_list[:-1], logits_list_frz[:-1])):
            if not self.freeze:
                logits_frz = logits
            logprob_s = F.log_softmax(logits_frz / self.T, dim=1)
            prob_t = prob_list[i+1]
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        self._cache['adj_maxprob'] = adj_maxprob = prob_t.max(dim=1)[0].mean()
        if adj_maxprob > self.maxprob:
            self.T *= self.Tmult
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list, logits_list_frz = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob, self.T]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        return out + [self._cache['adj_maxprob'], self.T]
    

class MaxprobMultiFrzDistLaterLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.05,
                 weight_last=False, global_scale=1.0, freeze=True):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.weight_last = weight_last
        self.global_scale = global_scale
        self.freeze = freeze
        
        self.metric_names += ['adj_maxprob', 'temperature'] 

    def __call__(self, net, X, y, *args):
        logits_list, logits_list_frz = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        if self.weight_last:
            cum_loss *= (1.0 - self.C)
        prob_list = [F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list]
        for i, (logits, logits_frz) in enumerate(
                zip(logits_list[:-1], logits_list_frz[:-1])):
            if not self.freeze:
                logits_frz = logits
            logprob_s = F.log_softmax(logits_frz / self.T, dim=1)
            prob_t = sum(prob_list[i+1:]) / (self.n_exits-i-1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        self._cache['adj_maxprob'] = adj_maxprob = prob_t.max(dim=1)[0].mean()
        if adj_maxprob > self.maxprob:
            self.T *= self.Tmult
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list, logits_list_frz = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob, self.T]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        return out + [self._cache['adj_maxprob'], self.T]
    

class MultiFrzDistLossConstTemp(_MultiExitAccuracy):
    def __init__(self, C, T, n_exits, acc_tops=(1,),
                 weight_last=False, global_scale=1.0, freeze=True):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.T = T
        self.weight_last = weight_last
        self.global_scale = global_scale
        self.freeze = freeze
        self.metric_names += ['adj_maxprob']

    def __call__(self, net, X, y, *args):
        logits_list, logits_list_frz = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        if self.weight_last:
            cum_loss *= (1.0 - self.C)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        self._cache['prob_t'] = prob_t
        for logits, logits_frz in zip(logits_list[:-1], logits_list_frz[:-1]):
            if not self.freeze:
                logits_frz = logits
            logprob_s = F.log_softmax(logits_frz / self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list, logits_list_frz = net.train(False)(X)
        prob_t = F.softmax(logits_list[-1].data / self.T, dim=1)
        adj_maxprob = prob_t.max(dim=1)[0].mean()
        return self._metrics(logits_list, y) + [adj_maxprob]

    def trn_metrics(self):
        out = self._metrics(self._cache['logits_list'], self._cache['y'])
        adj_maxprob = self._cache['prob_t'].max(dim=1)[0].mean()
        return out + [adj_maxprob]
    

class MaxprobBestFrzDistLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.05,
                 weight_last=False, global_scale=1.0):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0
        self.weight_last = weight_last
        self.global_scale = global_scale

    def __call__(self, net, X, y, *args):
        logits_list, logits_list_frz = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        cum_loss = sum(losses) * (1.0 - self.C)
        t = np.argmin(losses)
        prob_t = F.softmax(logits_list[t].data / self.T, dim=1)
        for s, logits_frz in enumerate(logits_list_frz):
            logprob_s = F.log_softmax(logits_frz / self.T, dim=1)
            if s != t:
                dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
                cum_loss += (self.T ** 2) * self.C * dist_loss

        if prob_t.max(dim=1)[0].mean() > self.maxprob:
            self.T *= self.Tmult
        return cum_loss * self.global_scale

    def metrics(self, net, X, y, *args):
        logits_list, logits_list_frz = net.train(False)(X)
        return self._metrics(logits_list, y)
    

class HydraLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.1):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0

    def __call__(self, net, X, y, *args):
        logits_list, teacher_list = net(X)
        self._cache['logits_list'] = logits_list
        self._cache['y'] = y

        sum_prob_t = 0.0
        cum_loss = F.cross_entropy(logits_list[-1], y)
        for logits_s, logits_t in zip(logits_list[:-1], teacher_list):
            cum_loss += F.cross_entropy(logits_t, y)
            logprob_s = F.log_softmax(logits_s / self.T, dim=1)
            prob_t = F.softmax(logits_t.data / self.T, dim=1)
            sum_prob_t += prob_t
            
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits_s, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        avg_prob_t = sum_prob_t / len(teacher_list)
        if avg_prob_t.max(dim=1)[0].mean() > self.maxprob:
            self.T *= self.Tmult

        return cum_loss

    def metrics(self, net, X, y, *args):
        logits_list, teacher_list = net.train(False)(X)
        return self._metrics(logits_list, y)

                                      
class EnsembleDistillationLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,), n_distill=99,
                 n_teach=None):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C
        self.n_distill = n_distill
        self.n_teach = n_teach or n_exits

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        cum_loss = 0.0
        prob_t = sum(F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list[-self.n_teach:]) / self.n_teach
        for i, logits in enumerate(logits_list):
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            if i < self.n_distill:
                logprob_s = F.log_softmax(logits/self.T, dim=1)
                dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
                cum_loss += (1.0 - self.C) * cross_ent
                cum_loss += (self.T ** 2) * self.C * dist_loss
            else:
                cum_loss += cross_ent
        return cum_loss


class MaxprobEnsembleDistLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.1):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        cum_loss = 0.0
        prob_t = sum(F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list) / self.n_exits
        for i, logits in enumerate(logits_list):
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        if prob_t.max(dim=1)[0].mean() > self.maxprob:
            self.T *= self.Tmult
            
        return cum_loss


class LaterExitDistillationLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        prob_list = [F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list]
        for i, logits in enumerate(logits_list[:-1]):
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            prob_t = sum(prob_list[i+1:]) / (self.n_exits-i-1)
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss


class MaxprobLaterDistLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.1):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        cum_loss = F.cross_entropy(logits_list[-1], y)
        prob_list = [F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list]
        sum_prob_t = 0.0
        for i, logits in enumerate(logits_list[:-1]):
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            prob_t = sum(prob_list[i+1:]) / (self.n_exits-i-1)
            sum_prob_t += prob_t
            
            dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss

        avg_prob_t = sum_prob_t / (self.n_exits - 1)
        if avg_prob_t.max(dim=1)[0].mean() > self.maxprob:
            self.T *= self.Tmult
            
        return cum_loss


class SelectiveDistillationLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        cum_loss = sum(losses)
        probs = [F.softmax(logits.data/self.T, dim=1) for logits in logits_list]
        
        for s, logits in enumerate(logits_list[:-1]):
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            for t in range(s+1, self.n_exits):
                if losses[s] > losses[t]:
                    dist_loss = -(probs[t]*logprob_s).sum(dim=1).mean()
                    cum_loss += (self.T**2) * self.C * dist_loss
        return cum_loss

    
class ReweightedDistillationLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        probs = [F.softmax(logits.data/self.T, dim=1) for logits in logits_list]
        cum_loss = 0.0
        
        for s, logits in enumerate(logits_list):
            loss_s = losses[s]
            weight = 1.0
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            for t in range(s+1, self.n_exits):
                if losses[s] > losses[t]:
                    dist_loss = -(probs[t]*logprob_s).sum(dim=1).mean()
                    loss_s += (self.T**2) * self.C * dist_loss
                    weight += self.C
            cum_loss += loss_s / weight
        return cum_loss


class DistillFromBetterLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        probs = [F.softmax(logits.data/self.T, dim=1) for logits in logits_list]
        cum_loss = 0.0
        
        for s, logits in enumerate(logits_list):
            loss_s = losses[s]
            weight = 1.0
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            for t in range(self.n_exits):
                if s != t and losses[s] > losses[t]:
                    dist_loss = -(probs[t]*logprob_s).sum(dim=1).mean()
                    loss_s += (self.T**2) * self.C * dist_loss
                    weight += self.C
            cum_loss += loss_s / weight
        return cum_loss


class MaxprobBetterDistLoss(_MultiExitAccuracy):
    def __init__(self, C, maxprob, n_exits, acc_tops=(1,), Tmult=1.1):
        super().__init__(n_exits, acc_tops)
        self.C = C
        self.maxprob = maxprob
        self.Tmult = Tmult
        self.T = 1.0

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        probs = [F.softmax(logits.data/self.T, dim=1) for logits in logits_list]
        cum_loss = 0.0
        
        for s, logits in enumerate(logits_list):
            loss_s = losses[s]
            weight = 1.0
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            for t in range(self.n_exits):
                if s != t and losses[s] > losses[t]:
                    dist_loss = -(probs[t]*logprob_s).sum(dim=1).mean()
                    loss_s += (self.T**2) * self.C * dist_loss
                    weight += self.C
            cum_loss += loss_s / weight

        avg_prob_t = sum(probs) / len(probs)
        if avg_prob_t.max(dim=1)[0].mean() > self.maxprob:
            self.T *= self.Tmult
            
        return cum_loss


class DistillFromDeeperLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        probs = [F.softmax(logits.data/self.T, dim=1) for logits in logits_list]
        cum_loss = 0.0
        
        for s, logits in enumerate(logits_list):
            loss_s = F.cross_entropy(logits, y)
            weight = 1.0
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            for t in range(s+1, self.n_exits):
                dist_loss = -(probs[t]*logprob_s).sum(dim=1).mean()
                loss_s += (self.T**2) * self.C * dist_loss
                weight += self.C
            cum_loss += loss_s / weight
        return cum_loss


class DistillFromCorrectLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,), acc_thr=0.99):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C
        self.acc_thr = acc_thr

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        teachers = [i for i, logits in enumerate(logits_list)
                    if multiclass_accuracy(logits, y) > self.acc_thr]
        cum_loss = 0.0
        if teachers:
            prob_t = sum(F.softmax(logits_list[i].data/self.T, dim=1)
                         for i in teachers) / len(teachers)
        for i, logits in enumerate(logits_list):
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logits, y)
            if i in teachers or not teachers:
                cum_loss += cross_ent
            else:
                logprob_s = F.log_softmax(logits/self.T, dim=1)
                dist_loss = -(prob_t * logprob_s).sum(dim=1).mean()
                cum_loss += (1.0 - self.C) * cross_ent
                cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss


class AttractiveRepulsiveLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        losses = [F.cross_entropy(logits, y) for logits in logits_list]
        cum_loss = sum(losses)
        probs = [F.softmax(logits.data/self.T, dim=1) for logits in logits_list]

        for s, logits in enumerate(logits_list):
            logprob_s = F.log_softmax(logits/self.T, dim=1)
            for t in range(self.n_exits):
                if s == t:
                    continue
                dist_loss = -(probs[t]*logprob_s).sum(dim=1).mean()
                if losses[s] > losses[t]:
                    cum_loss += (self.T**2) * self.C * dist_loss
                elif losses[s] < losses[t]:
                    cum_loss -= (self.T**2) * self.C * dist_loss
        return cum_loss

    
class AlphaBoostingLoss(_MultiExitAccuracy):
    def __init__(self, n_exits, acc_tops=(1,), alpha=None):
        super().__init__(n_exits, acc_tops)
        assert alpha > 0.0
        self.alpha = alpha
    
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        d = torch.ones_like(y, dtype=torch.float32) / len(y)
        cum_loss = 0.0
        for logits in logits_list:
            cross_ent = F.cross_entropy(logits, y, reduce=False)
            cum_loss += (cross_ent * d).sum()

            hit = multiclass_accuracy(logits, y, reduce=False)
            d = d * torch.exp(- self.alpha * hit)
            d = d / d.sum()
        return cum_loss
    
    
class BoostingLoss(_MultiExitAccuracy):
    def __init__(self, n_exits, C=1.0):
        super().__init__(n_exits, _binary_clf=True)
        self.C = C
    
    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        d = torch.ones_like(y, dtype=torch.float32) / len(y)
        cum_loss = 0.0
        for logits in logits_list:
            cross_ent = bce_w_logits(logits.view(-1), y.float(), reduce=False)
            cum_loss += (1.0-self.C) * cross_ent.mean() 
            cum_loss += self.C * (cross_ent*d).sum()

            hit = binary_accuracy(logits, y, reduce=False)
            acc = (d*hit).sum()
            if 1.0 > acc > 0.5:
                alpha2 = torch.log(acc/(1.0-acc))
                d = d * torch.exp(- alpha2 * hit)
                d = d / d.sum()
        return cum_loss


class Ens2EnsDistillationLoss(_MultiExitAccuracy):
    def __init__(self, T, C, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)
        self.T = T
        self.C = C

    def __call__(self, net, X, y, *args):
        self._cache['logits_list'] = logits_list = net(X)
        self._cache['y'] = y

        logprobsT = torch.stack([F.log_softmax(logits/self.T, dim=1)
                                      for logits in logits_list])
        logprobs1 = torch.stack([F.log_softmax(logits, dim=1)
                                      for logits in logits_list])
        logits_ens = logsumexp(logprobs1, dim=0) - np.log(self.n_exits)
        cum_loss = F.cross_entropy(logits_ens, y)
        prob_t = sum(F.softmax(logits.data/self.T, dim=1)
                     for logits in logits_list) / self.n_exits
        
        for i in range(1, self.n_exits):
            logprob_sT = logsumexp(logprobsT[:i], dim=0) - np.log(i)
            logprob_s1 = logsumexp(logprobs1[:i], dim=0) - np.log(i)
            dist_loss = -(prob_t * logprob_sT).sum(dim=1).mean()
            cross_ent = 0.0 if self.C == 1.0 else F.cross_entropy(logprob_s1, y)
            cum_loss += (1.0 - self.C) * cross_ent
            cum_loss += (self.T ** 2) * self.C * dist_loss
        return cum_loss
    
############    
    
class ConvVaeDistillationLoss(_Loss):
    def __init__(self, match_layer, C, beta):
        self.match_layer = match_layer
        self.C = C
        self.beta = beta
    
    def __call__(self, net, X, y, tensor_t, *args):
        z, logqz_x = net.enc.samplez_x(X)
        logpx_z = net.dec.logpx_z(X, z)
        logpz = net.prior.logp(z)

        tensor_s = net.dec.conv.trace(
            z, keep_layers=[self.match_layer])[self.match_layer]
        dist_loss = F.mse_loss(tensor_s, tensor_t)

        self._X = X
        self._recon_loss = -logpx_z.mean()
        self._kl = (logqz_x - logpz).mean()
        self._elbo_loss = self._recon_loss + self.beta * self._kl
        return (1.0 - self.C) * self._elbo_loss + self.C * dist_loss

    def metrics(self, net, X, *args):
        net.train(False)
        z, logqz_x = net.enc.samplez_x(X)
        logpx_z = net.dec.logpx_z(X, z)
        logpz = net.prior.logp(z)

        x_dim = float(np.product(X.size()[1:]))
        recon_loss = -logpx_z.data.mean()
        kl = (logqz_x - logpz).data.mean()
        elbo = -recon_loss - self.beta * kl
        return [elbo, recon_loss / x_dim, kl]
        
    def trn_metrics(self):
        x_dim = float(np.product(self._X.size()[1:]))
        return [-self._elbo_loss.data,
                self._recon_loss.data / x_dim, self._kl.data]

    metric_names = ['elbo', 'recon', 'kl']
    

class ElboLoss(_Loss):
    def __init__(self, beta=1.0):
        self.beta = beta
    
    def __call__(self, net, X, *args):
        z, logqz_x = net.enc.samplez_x(X)
        logpx_z = net.dec.logpx_z(X, z)
        logpz = net.prior.logp(z)

        self._X = X
        self._recon_loss = -logpx_z.mean()
        self._kl = (logqz_x - logpz).mean()
        self._loss = self._recon_loss + self.beta * self._kl
        return self._loss

    def metrics(self, net, X, *args):
        x_dim = float(np.product(X.size()[1:]))
        self.__call__(net.train(False), X)
        return [-self._loss.data, self._recon_loss.data / x_dim,
                self._kl.data]

    def trn_metrics(self):
        x_dim = float(np.product(self._X.size()[1:]))
        return [-self._loss.data, self._recon_loss.data / x_dim,
                self._kl.data]

    metric_names = ['elbo', 'recon', 'kl']
    
#########################################################

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


class ConvNet(_TraceInForward):
    def __init__(self, in_shape, out_dim, conv_spec, pool_size=1, 
                 nonlin='ReLU'):
        """
        in_shape:  tuple (n_channels, width, height)
        conv_spec: list of tuples (n_kernels, kernel_size, stride, padding)
        """
        super().__init__()
        self.conv = nn.ModuleList()
        in_channels, h, w = in_shape
        
        for args in conv_spec:
            n_kernels, kernel_size, stride, padding = args
            self.conv.append(nn.Conv2d(in_channels, *args))
            in_channels = n_kernels
            h = int((h + 2*padding - kernel_size) / stride) + 1
            w = int((w + 2*padding - kernel_size) / stride) + 1

        h = int(h/pool_size)
        w = int(w/pool_size)
        
        in_features = conv_spec[-1][0] * h * w
        self.lin = nn.Linear(in_features, out_dim)

        self.nonlin = getattr(nn, nonlin)()
        self.pool_size = pool_size
        
    def forward(self, x, keep_layers=()):
        n = x.size(0)
        x = TransientDict(x=x, _keep=keep_layers)
        for i, conv in enumerate(self.conv):
            x[f'conv{i}'] = conv(x[-1])
            x[f'conv{i}_nonlin'] = self.nonlin(x[-1])
            
        x['pool'] = F.avg_pool2d(x[-1], kernel_size=self.pool_size)
        x['logits'] = self.lin(x[-1].view(n, -1))
        self._trace = x
        return x['logits']


class GConvNet(_TraceInForward):
    def __init__(self, in_dim, out_shape, size):
        "out_shape = (C, W, W) with W a power of two."
        assert out_shape[1] == out_shape[2]
        super().__init__()
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        n_channels = size * out_shape[1] // 8
        self.conv.append(nn.ConvTranspose2d(
            in_dim, n_channels, 4, 1, 0, bias=False))  # 4x4
        self.bn.append(nn.BatchNorm2d(n_channels))
        
        while n_channels > size:
            n_channels //= 2
            self.conv.append(nn.ConvTranspose2d(
                n_channels*2, n_channels, 4, 2, 1, bias=False))
            self.bn.append(nn.BatchNorm2d(n_channels))

        assert n_channels == size
        self.conv.append(nn.ConvTranspose2d(
            size, out_shape[0], 4, 2, 1, bias=False))

    def forward(self, z, keep_layers=()):
        z = TransientDict(z=z[:,:,None,None], _keep=keep_layers)
        for i in range(len(self.bn)):
            z[f'conv{i}'] = self.conv[i](z[-1])
            z[f'conv{i}_bn'] = self.bn[i](z[-1])
            z[f'conv{i}_relu'] = F.relu(z[-1])
        z[f'conv{i+1}'] = self.conv[-1](z[-1])
        self._trace = z
        return z[-1]


class DConvNet(_TraceInForward):
    def __init__(self, in_shape, out_dim, size):
        super().__init__()
        self.conv0 = nn.Conv2d(in_shape[0], size, 4, 2, 1, bias=False)
        img_dim = in_shape[1] // 2
        
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        while img_dim > 4:
            size *= 2
            self.conv.append(nn.Conv2d(size//2, size, 4, 2, 1, bias=False))
            self.bn.append(nn.BatchNorm2d(size))
            img_dim //= 2

        assert img_dim == 4
        self.n_layers = 1 + len(self.bn)
        self.shape_last = (size, 4, 4)
        self.conv.append(nn.Conv2d(size, out_dim, 4, 1, 0, bias=False))

    def forward(self, x, keep_layers=()):
        n = x.size(0)
        x = TransientDict(x=x, _keep=keep_layers)
        x['conv0'] = self.conv0(x[-1])
        x['conv0_lrelu'] = F.leaky_relu(x[-1], 0.2)
        
        for i in range(self.n_layers - 1):
            x[f'conv{i+1}'] = self.conv[i](x[-1])
            x[f'conv{i+1}_bn'] = self.bn[i](x[-1])
            x[f'conv{i+1}_lrelu'] = F.leaky_relu(x[-1], 0.2)
            
        x['logits'] = self.conv[-1](x[-1]).view(n, -1)
        self._trace = x
        return x['logits']

    
class Mlp(_TraceInForward):
    def __init__(self, in_dim:int, out_dim:int, hidden_dims:List[int],
                 nonlin='ReLU', dropout_p=0.5, dropout_init=0.2):
        super().__init__()
        hidden_dims = [in_dim] + hidden_dims + [out_dim]
        self.lin = nn.ModuleList(
            [nn.Linear(hidden_dims[i-1], hidden_dims[i])
             for i in range(1, len(hidden_dims))]
        )
        self.nonlin = getattr(nn, nonlin)()
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_init = nn.Dropout(dropout_init)

    def forward(self, x, keep_layers=()):
        x = TransientDict(x=x.view(x.size(0), -1), _keep=keep_layers)
        x['x_dropout'] = self.dropout_init(x[-1])
        for i in range(len(self.lin)-1):
            x[f'lin{i}'] = self.lin[i](x[-1])
            x[f'lin{i}_nonlin'] = self.nonlin(x[-1])
            x[f'lin{i}_dropout'] = self.dropout(x[-1])
        x['logits'] = self.lin[-1](x[-1])
        self._trace = x
        return x['logits']


class MlpDecoupled(_TraceInForward):
    def __init__(self, in_dim:int, out_dim:int, hidden_dims:List[int],
                 nonlin='ReLU', dropout_p=0.5, dropout_init=0.2):
        super().__init__()
        hidden_dims = [in_dim] + hidden_dims + [1]
        lins = []
        for j in range(out_dim):
            lins += [nn.ModuleList(
                [nn.Linear(hidden_dims[i-1], hidden_dims[i])
                 for i in range(1, len(hidden_dims))]
            )]
        self.lin = nn.ModuleList(lins)
        self.nonlin = getattr(nn, nonlin)()
        self.dropout = nn.Dropout(dropout_init)
        self.dropout_init = nn.Dropout(dropout_init)
        self._out_dim = out_dim

    def forward(self, x, keep_layers=()):
        x = TransientDict(x=x.view(x.size(0),-1), _keep=keep_layers)
        x_input = x[-1]
        for j in range(self._out_dim):
            x[f'x_dropout{j}'] = self.dropout_init(x_input)
            for i in range(len(self.lin[j])-1):
                x[f'lin{j}{i}'] = self.lin[j][i](x[-1])
                x[f'lin{j}{i}_nonlin'] = self.nonlin(x[-1])
                x[f'lin{j}{i}_dropout'] = self.dropout(x[-1])
            x[f'logits{j}'] = self.lin[j][-1](x[-1])
        logits = [x[f'logits{j}'] for j in range(self._out_dim)]
        x['logits'] = torch.cat(logits, dim=1)
        self._trace = x
        return x['logits']

############################    

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


class MsdNetFrz(MsdNet):
    def forward(self, x, keep_layers=()):
        logits = []
        logits_frz = []
        x = TransientDict(x=x, _keep=keep_layers)
        for i in range(len(self.blocks)):
            h = self.blocks[i](x[-1])
            x[f'block{i}'] = h
            logits += [self.exits[i](h[-1])]
            logits_frz += [self.exits[i](h[-1].data)]
        x[-1]
        x['logits'] = logits
        x['logits_frz'] = logits_frz
        self._trace = x
        return x['logits'], x['logits_frz']
    

class HydraMsdNet(MsdNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teachers = nn.ModuleList()
        for i in range(self.n_exits-1):
            self.teachers[i] = self.Exit(
                self._nplanes_tab[-1, self._block_sep[self.n_exits]-1],
                self.out_dim, self.exit_width)

    def forward(self, x, keep_layers=()):
        logits = []
        x = TransientDict(x=x, _keep=keep_layers)
        for i in range(len(self.blocks)):
            h = self.blocks[i](x[-1])
            x[f'block{i}'] = h
            logits += [self.exits[i](h[-1])]
        x[-1]
        x['logits'] = logits
        x['teachers'] = [self.teachers[i](h[-1]) for i in range(self.n_exits-1)]
        self._trace = x
        return x['logits'], x['teachers']
    
############################
    
class ResBlock(nn.Module):
    def __init__(self, n_in, n_out, stride, nonlin='ReLU'):
        super().__init__()
        self.nonlin = getattr(nn, nonlin)()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride == 1 and n_in == n_out:
            self.residual = nn.Sequential()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(n_out)
            )

    def forward(self, x):
        h = self.bn1(self.conv1(x))
        h = self.nonlin(h)
        h = self.bn2(self.conv2(h)) + self.residual(x)
        return self.nonlin(h)


def ResGroup(n_in, n_out, n_blocks, stride, nonlin='ReLU'):
    blocks = [ResBlock(n_in, n_out, stride, nonlin)]
    for i in range(n_blocks-1):
        blocks += [ResBlock(n_in=n_out, n_out=n_out, stride=1, nonlin=nonlin)]
    return nn.Sequential(*blocks)

    
class Resnet(_TraceInForward):
    def __init__(self, in_shape, out_dim, n_blocks: List[int], nonlin='ReLU'):
        super().__init__()
        self.nonlin = getattr(nn, nonlin)()
        self.conv = nn.Conv2d(in_shape[0], out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)

        blocks = []
        blocks += [ResGroup( 64,  64, n_blocks[0], stride=1, nonlin=nonlin)]
        blocks += [ResGroup( 64, 128, n_blocks[1], stride=2, nonlin=nonlin)]
        blocks += [ResGroup(128, 256, n_blocks[2], stride=2, nonlin=nonlin)]
        blocks += [ResGroup(256, 512, n_blocks[3], stride=2, nonlin=nonlin)]
        self.blocks = nn.ModuleList(blocks)

        assert in_shape[1] % 8 == 0
        assert in_shape[2] % 8 == 0
        self._pool_size = (in_shape[1]//8, in_shape[2]//8)
        self.lin = nn.Linear(512, out_dim)

    def forward(self, x, keep_layers=()):
        n = x.size(0)
        x = TransientDict(x=x, _keep=keep_layers)
        x['conv_nonlin'] = self.nonlin(self.bn(self.conv(x[-1])))
        for i in range(4):
            x[f'block{i}'] = self.blocks[i](x[-1])
        x['pool'] = F.avg_pool2d(x[-1], self._pool_size)
        x['logits'] = self.lin(x[-1].view(n, -1))
        self._trace = x
        return x['logits']


class ResnetB(_TraceInForward):
    def __init__(self, in_shape, out_dim, n_blocks: List[int], nonlin='ReLU'):
        super().__init__()
        self.nonlin = getattr(nn, nonlin)()
        self.conv0 = nn.Conv2d(in_shape[0], out_channels=64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)

        blocks = []
        blocks += [ResGroup( 64,  64, n_blocks[0], stride=1, nonlin=nonlin)]
        blocks += [ResGroup( 64, 128, n_blocks[1], stride=2, nonlin=nonlin)]
        blocks += [ResGroup(128, 256, n_blocks[2], stride=2, nonlin=nonlin)]
        blocks += [ResGroup(256, 512, n_blocks[3], stride=1, nonlin=nonlin)]
        self.blocks = nn.ModuleList(blocks)

        self.conv1 = nn.Conv2d(512, out_dim, kernel_size=1)
        assert in_shape[1] % 4 == 0
        assert in_shape[2] % 4 == 0
        self._pool_size = (in_shape[1]//4, in_shape[2]//4)

    def forward(self, x, keep_layers=()):
        n = x.size(0)
        x = TransientDict(x=x, _keep=keep_layers)
        x['conv0_nonlin'] = self.nonlin(self.bn(self.conv0(x[-1])))
        for i in range(4):
            x[f'block{i}'] = self.blocks[i](x[-1])
        x['conv1'] = self.conv1(x[-1])
        x['pool'] = F.avg_pool2d(x[-1], self._pool_size)
        x['logits'] = x[-1].view(n, -1)
        self._trace = x
        return x['logits']
    
###########################

class TeacherGan(nn.Module):
    def __init__(self, gen_snapshot, teacher_snapshot, student):
        super().__init__()
        cf_gen = utils.snapshot_cf(gen_snapshot)
        cf_teacher = utils.snapshot_cf(teacher_snapshot)
        
        xshape_gen = cf_gen['cf_net']['shape_x'][1]
        xshape_teacher = cf_teacher['cf_net']['in_shape'][1]
        pool = xshape_gen // xshape_teacher
        
        self.gen = nn.Sequential(
            load_snapshot(gen_snapshot).gen,
            nn.AvgPool2d(kernel_size=pool, stride=pool)
        )
        self.teacher = load_net(cf_teacher)
        for par in self.teacher.parameters():
            par.requires_grad = False
        
        Student = globals()[student['call']]
        self.student = Student(in_shape=cf_teacher['cf_net']['in_shape'],
                               out_dim=cf_teacher['cf_net']['out_dim'],
                               **dict_drop(student, 'call'))
        
        
class ConditionalGan(nn.Module):
    def __init__(self, dim_z:int, shape_x:Tuple[int,int,int], dim_y:int,
                 gen_size:int, disc_size:int):
        super().__init__()
        self.gen = self.Generator(dim_z, shape_x, dim_y, gen_size)
        self.disc = self.Discriminator(shape_x, dim_y, disc_size)

    class Generator(_TraceInForward):
        def __init__(self, dim_z, shape_x, dim_y, size):
            super().__init__()
            self.gconvnet = GConvNet(dim_z, shape_x, size)
            self.lin = nn.Linear(dim_y, dim_z)

        def forward(self, z, y, keep_layers=()):
            x = self.gconvnet(z + self.lin(y), keep_layers)
            self._trace = self.gconvnet._trace
            return x

    class Discriminator(_TraceInForward):
        def __init__(self, shape_x, dim_y, size):
            super().__init__()
            self.dconvnet = DConvNet(shape_x, 1, size)
            i = self.dconvnet.n_layers - 1
            self.last_conv = f'conv{i}_lrelu'
            self.lin = nn.Linear(dim_y, int(np.prod(self.dconvnet.shape_last)))

        def forward(self, x, y, keep_layers=()):
            n = x.size(0)
            x = self.dconvnet.trace(
                x, keep_layers=(self.last_conv, *keep_layers))
            last_conv = x[self.last_conv]
            x['xy'] = last_conv + self.lin(y).view(*last_conv.shape)
            x['logits'] = self.dconvnet.conv[-1](x['xy']).view(n, -1)
            self._trace = x
            return x['logits']
        

class Gan(nn.Module):
    def __init__(self, dim_z, shape_x, gen, disc):
        super().__init__()
        Gen = globals()[gen['call']]
        Disc = globals()[disc['call']]
        self.gen = Gen(dim_z, shape_x, **dict_drop(gen, 'call'))
        self.disc = Disc(shape_x, 1, **dict_drop(disc, 'call'))

        
class Vae(nn.Module):
    def __init__(self, dim_z, shape_x, prior, enc, dec):
        super().__init__()
        Enc = globals()[enc['call']]
        Dec = globals()[dec['call']]
        self.enc = Enc(shape_x, dim_z, **dict_drop(enc, 'call'))
        self.dec = Dec(dim_z, shape_x, **dict_drop(dec, 'call'))
        self.prior = globals()[prior['call']](dim_z, **dict_drop(prior, 'call'))
    
####################################################

def bernoulli_logp(x, logits):
    logp = Bernoulli(logits=logits).log_prob(x)
    return logp.view(x.size(0), -1).sum(dim=1)
    
def bernoulli_sample(logits):
    return torch.bernoulli(logits.sigmoid())

def gaussian_logp(x, mean, logvar):
    logp = Normal(mean, torch.exp(logvar/2.0)).log_prob(x)
    return logp.view(x.size(0), -1).sum(dim=1)

def gaussian_sample(mean, logvar):
    return Normal(mean, torch.exp(logvar/2.0)).rsample()

##########################

class _Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self._dummy.device
        
    def logp(self, z):
        raise NotImplementedError

    def sample(self, n):
        raise NotImplementedError

    def Hz(self):
        raise NotImplementedError


class NormalPrior(_Prior):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def logp(self, z):
        return Normal(0, 1).log_prob(z).sum(dim=1)

    def sample(self, n):
        return torch.randn(n, self.dim).to(self.device)

    def Hz(self):
        return (Normal(0, 1).entropy() * self.dim).to(self.device)

##########################

class _Encoder(nn.Module):
    def logqz_x(self, z, x):
        raise NotImplementedError
        return logq

    def samplez_x(self, x):
        """
        Returns:
            z:    use reparametrisation trick
            logq: for low variance, function of z but not of q params;
                  otherwise, function of z and of q params.
        """
        raise NotImplementedError


class ConvGaussianEncoder(_Encoder):
    def __init__(self, shape_x, dim_z, conv_spec, dim_lin, nonlin='ReLU'):
        super().__init__()
        self.conv = ConvNet(shape_x, dim_lin, conv_spec, nonlin=nonlin)
        self.lin_m = nn.Linear(dim_lin, dim_z)
        self.lin_s = nn.Linear(dim_lin, dim_z)
        self.nonlin = getattr(nn, nonlin)()

    def mean_logvar(self, x):
        lin = self.nonlin(self.conv(x))
        return self.lin_m(lin), self.lin_s(lin)
        
    def logqz_x(self, z, x):
        mean, logvar = self.mean_logvar(x)
        return gaussian_logp(z, mean, logvar)

    def samplez_x(self, x):
        mean, logvar = self.mean_logvar(x)
        z = gaussian_sample(mean, logvar)
        return z, gaussian_logp(z, mean.detach(), logvar.detach())

###########################

class _Decoder(nn.Module):
    def logpx_z(self, x, z):
        raise NotImplementedError
        return logp

    def samplex_z(self, z):
        raise NotImplementedError
        return x, logp


class ConvBernoulliDecoder(_Decoder):
    def __init__(self, dim_z, shape_x, size):
        super().__init__()
        self.conv = GConvNet(dim_z, shape_x, size)
    
    def logpx_z(self, x, z):
        return bernoulli_logp(x, self.conv(z))

    def samplex_z(self, z):
        x = bernoulli_sample(self.conv(z))
        return x, bernoulli_logp(x, self.conv(z))


class ConvPseudoBernoulliDecoder(_Decoder):
    def __init__(self, dim_z, shape_x, size):
        super().__init__()
        self.conv = GConvNet(dim_z, shape_x, size)
    
    def logpx_z(self, x, z):
        return bernoulli_logp((x+1.0)/2.0, self.conv(z))

    def samplex_z(self, z):
        logits = self.conv(z)
        x = logits.sigmoid()
        return x, bernoulli_logp(x, logits)
    
    
class ConvGaussianDecoder(_Decoder):
    def __init__(self, dim_z, shape_x, size, var_x=1.0):
        super().__init__()
        self.conv = GConvNet(dim_z, shape_x, size)
        self.logvar_x = np.log(var_x)
    
    def logpx_z(self, x, z):
        mean = self.conv(z).tanh()
        logvar = torch.zeros_like(mean) + self.logvar_x
        return gaussian_logp(x, mean, logvar)

    def samplex_z(self, z):
        mean = self.conv(z).tanh()
        logvar = torch.zeros_like(mean) + self.logvar_x
        x = gaussian_sample(mean, logvar)
        logpx_z = gaussian_logp(x, mean, logvar)
        return x, logpx_z
    
    
