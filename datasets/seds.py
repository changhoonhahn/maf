import os
import numpy as np
import matplotlib.pyplot as plt

import datasets
import util


class SED:

    class Data:

        def __init__(self, data_x, data_y):

            self.x = data_x.astype(np.float32)
            self.y = data_y.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        
        # load in training + validation + test data 
        x_trn, y_trn, x_val, y_val, x_tst, y_tst = load_data()

        self.trn = self.Data(x_trn, y_trn)
        self.val = self.Data(x_val, y_val)
        self.tst = self.Data(x_tst, y_tst)

        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]


def load_data():
    version = '0.0'
    dat_dir = "/tigress/chhahn/arcoiris/sedflow/"
    if not os.path.isdir(dat_dir): 
        dat_dir = "/Users/chahah/data/arcoiris/sedflow/"

    props, mags, sigs, zreds = [], [], [], []
    for seed in range(11): 
        _prop = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.props.prune_cnf.npy' % (version, seed)))
        _mags = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _sigs = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.sigma_mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _zred = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.redshifts.prune_cnf.npy' % (version, seed)))

        props.append(_prop)
        mags.append(_mags)
        sigs.append(_sigs)
        zreds.append(_zred)
    
    data_x = np.concatenate(props)
    data_y = np.concatenate([
        np.concatenate(mags), 
        np.concatenate(sigs), 
        np.atleast_2d(np.concatenate(zreds)).T], 
        axis=1)
    N_validate = int(0.1 * data_x.shape[0]) 

    data_x_train = data_x[:-N_validate]
    data_y_train = data_y[:-N_validate]

    data_x_valid = data_x[-N_validate:]
    data_y_valid = data_y[-N_validate:]

    # load test data

    for seed in [101]: 
        _prop = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.props.prune_cnf.npy' % (version, seed)))
        _mags = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _sigs = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.sigma_mags.noise_cnf_flux.prune_cnf.npy' % (version, seed)))
        _zred = np.load(os.path.join(dat_dir, 
            'train.v%s.%i.redshifts.prune_cnf.npy' % (version, seed)))
    data_x_test = _prop
    data_y_test = np.concatenate([_mags, _sigs, np.atleast_2d(_zred).T], axis=1)

    return data_x_train, data_y_train, data_x_valid, data_y_valid, data_x_test, data_y_test
