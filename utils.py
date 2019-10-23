import collections
import hue
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
import re
import sys
import time
from typing import Dict, List, Tuple
import torch


PROJ_DIR = Path(os.path.realpath(__file__)).parent
DATA_DIR = PROJ_DIR / 'data'
SNAPSHOT_DIR = PROJ_DIR / 'snapshots'
RUNS_DB_DIR = PROJ_DIR / 'runs_db'
TABLES_DIR = PROJ_DIR / 'tables'
ERRORS_LOG = RUNS_DB_DIR / '_stderr.txt'


def device(gpu):
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')


def remove_file(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass

    
def unpickle(fname):
    with open(fname, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def tab_str(*args):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    strings = (f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in args)
    return '\t'.join(strings)
    

class T:
    def __init__(self):
        self.times = []
        self.t = None

    def _start(self):
        self.t = time.time()

    def _finish(self, chunk_id):
        self.times[chunk_id] += (time.time() - self.t)
        self.t = None

    def __call__(self, finish_id=None):
        if self.t is None:
            assert finish_id is None
            self._start()
        elif finish_id is None:
            self.times += [0.0]
            self._finish(-1)
            self._start()
        else:
            assert finish_id <= len(self.times)
            if finish_id == len(self.times):
                self.times += [0.0]
            self._finish(finish_id)
            self._start()

    def __str__(self):
        return ', '.join([f'{t:.1f}' for t in self.times])
        
##############################################
    
class IntervalSaver:
    def __init__(self, snapshot_name, ep, interval):
        self.ep = ep
        self._snap_ep = str(SNAPSHOT_DIR/snapshot_name) + '_ep'
        self.interval = interval

    def save(self, perf, net, gpu, ep=None):
        self.ep = ep or self.ep+1
        if self.ep % self.interval == 0:
            torch.save(net.cpu().state_dict(), self._snap_ep+str(self.ep))
            net.to(device(gpu))

    
class RecordSaver:
    def __init__(self, snapshot_name, ep):
        self.ep = ep
        self.best_perf = -9999.0
        self.best_ep = -1
        self._snap_ep = str(SNAPSHOT_DIR/snapshot_name) + '_ep'
        self.last_save_time = time.time()

    def save(self, perf, net, gpu, ep=None):
        self.ep = ep or self.ep+1
        if perf > self.best_perf:
            torch.save(net.cpu().state_dict(), self._snap_ep+str(self.ep))
            net.to(device(gpu))
            
            if self.best_ep >= 0 and time.time() - self.last_save_time < 3600:
                remove_file(self._snap_ep + str(self.best_ep))
            elif time.time() - self.last_save_time > 3600:
                self.last_save_time = time.time()

            self.best_perf = perf
            self.best_ep = ep

##############################################

def dict_drop(dic, *keys):
    new_dic = dic.copy()
    for key in keys:
        if key in new_dic:
            del new_dic[key]
    return new_dic


class TransientDict:
    def __init__(self, _keep=(), **kw):
        self._dic = collections.OrderedDict(**kw)
        self._keep = _keep

    def __iter__(self):
        return self._dic.__iter__()

    def __repr__(self):
        items_str = '\n'.join(str(i) for i in self._dic.items())
        return 'TransientDict([\n{}\n])'.format(items_str)

    def __delitem__(self, key):
        del self._dic[key]

    def __setitem__(self, key, val):
        self._dic[key] = val

    def __getitem__(self, key):
        if key == -1:
            k, v = self._dic.popitem()
            self._dic[k] = v if k in self._keep else None
            return v
        else:
            val = self._dic[key]
            if key not in self._keep: self._dic[key] = None
            return val

    def keys(self):
        return self._dic.keys()

##############################################

def load_json(path):
    text = path.read_text()
    while text.strip() == '':
        time.sleep(1e-4)
        text = path.read_text()
    return json.loads(text)


def load_csv(fname, **kw):
    while True:
        try:
            return pd.read_csv(fname, sep='\t', **kw)
        except pd.errors.EmptyDataError:
            time.sleep(1e-4)

###########################################################
## Snapshotting

def get_snapname_ep(snapshot='', prompt='Snapshot to load:'):
    # snapshot may or may not end with ep
    if snapshot == '': snapshot = input(hue.que(prompt + ' '))
    if re.search(u'_ep[0-9]+$', snapshot): return snapshot
    eps = _snapshot_eps(snapshot)
    assert len(eps), 'No such snapshot.'
    return snapshot + '_ep' + str(max(eps))


def _snapshot_eps(snapshot):
    return [int(f.split('_ep')[-1]) for f in os.listdir(SNAPSHOT_DIR)
            if re.search(snapshot + u'_ep[0-9]+$', f)]


def _snapshotname_free(name):
    for run in os.listdir(RUNS_DB_DIR):
        if run[0] == '_': continue
        cf = load_json(RUNS_DB_DIR / run / 'config.json')
        if cf['snapshot_name'] == name: return False
    return True


def _snapshot_names():
    return set(runs_df().run.apply(lambda run: run.cf['snapshot_name']))
    

def numbered_snapshot(name):
    i_taken = [int(s.split(':')[1])
               for s in _snapshot_names() if s.startswith(name+':')]
    i = max(i_taken)+1 if i_taken else 0
    return f'{name}:{i}'


def snapshot_cf(snapshot):
    # snapshot can but doesn't have to end with '_ep'
    snapshot_split = snapshot.split('_')
    if snapshot_split[-1][:2] == 'ep':
        snapshot = '_'.join(snapshot_split[:-1])
    for rundir in os.listdir(RUNS_DB_DIR):
        if rundir[0] == '_': continue
        cf = load_json(RUNS_DB_DIR / rundir / 'config.json')
        if cf['snapshot_name'] == snapshot and 'cf_net' in cf:
            return cf
    raise FileNotFoundError(f'Could not find config for snapshot: {snapshot}.')

        
def snapshot_config_resume(snapshot_ep):
    cf = snapshot_cf(snapshot_ep)
    return (cf['cf_trn'], cf['cf_val'], cf['cf_net'], cf['cf_loss'], 
            cf['cf_opt'], cf['batch_size'], cf['cf_scheduler'],
            cf['snapshot_name'])

def snapshot_config_fork(snapshot_ep):
    cf = snapshot_cf(snapshot_ep)
    return (cf['cf_trn'], cf['cf_val'], cf['cf_net'])


