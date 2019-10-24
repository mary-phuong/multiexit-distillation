from train import ex
import utils


@ex.config
def config_new():
    gpu = 0                 # single GPU ordinal or list, or -1 for CPU
    snapshot_name = 'test:0'
    parent_snapshot = ''    # If empty, trains from scratch. Otherwise loads the
                            # specified snapshot and continues training.

                          
    cf_net = dict(          # MSDNet architecture parameters
        call = 'MsdNet',
        in_shape = 32,
        out_dim = 100,
        n_scales = 3,
        n_exits = 11,
        nlayers_to_exit = 4,
        nlayers_between_exits = 2,
        nplanes_mulv = [6, 12, 24],
        nplanes_addh = 1,
        nplanes_init = 1,
        prune = 'min',
        plane_reduction = 0.5,
        exit_width = 128,
        btneck_widths = [4, 4, 4],
    )

    
    cf_loss = dict(         # distillation-based training with temperature
                            # annealing
        call = 'DistillationBasedLoss',
        n_exits = cf_net['n_exits'],
        acc_tops = [1, 5],
        
        C = 0.5,
        maxprob = 0.5,
        global_scale = 2.0 * 5/cf_net['n_exits'],
    )

    # cf_loss = dict(       # distillation-based training with constant
    #                       # temperature
    #     call = 'DistillationLossConstTemp',
    #     n_exits = cf_net['n_exits'],
    #     acc_tops = [1, 5],
        
    #     C = 0.5,
    #     T = 4.0,
    #     global_scale = 2.0 * 5/cf_net['n_exits'],
    # )

    # cf_loss = dict(       # train with classification loss only
    #     call = 'ClassificationOnlyLoss',
    #     n_exits = cf_net['n_exits'],
    #     acc_tops = [1, 5],
    # )

    
    cf_trn = dict(          # training set parameters
        call = 'Cifar100',
        n_per_class = 150,  # number of images per class (including validation)
        nval_per_class = 50,
        augment = True,     # data augmentation
        seed = 0,
    )
    cf_val = cf_trn.copy()
    cf_val['augment'] = False

    
    cf_opt = dict(          # optimization method
        call = 'SGD',
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 1e-4,
        nesterov = True,
    )
    
    batch_size = 64
    val_batch_size = 250
    n_epochs = 300
    
    cf_scheduler = dict(   # learning rate schedule
        call = 'MultiStepLR',
        milestones = [150, 225],
        gamma = 0.1
    )
    
    save_interval = 0      # if nonzero, save a snapshot every X epochs,
                           # otherwise save a snapshot when it's the best so far

                           

if __name__ == '__main__':
    ex.run_commandline()
