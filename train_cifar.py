from train import ex
import utils


@ex.config
def config_new():
    gpu = 0  # single GPU ordinal or list, or -1 for CPU
    autoname = f'msd11_cif100cls80_ce'
    ####
    snapshot_name = ''
    parent_snapshot = ''
    ####
    
    cf_net = dict(
        call = 'MsdNet', #!
        in_shape = 32,
        out_dim = 100,
        n_scales = 3,
        n_exits = 11, #!
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
    cf_loss = dict(
        n_exits = cf_net['n_exits'],
        acc_tops = [1, 5],
        
        call = 'MultipleCrossEntropyLoss',
        
        # call = 'MaxprobMultiFrzDistLaterLoss',
        # call = 'MaxprobMultiFrzDistLoss',
        # C = 0.5,
        # maxprob = 0.5,
        # weight_last = True,
        # global_scale = 2.0 * 5/cf_net['n_exits'],
        # freeze = False, #!

        # call = 'MultiFrzDistLossConstTemp',
        # C = 0.5,
        # T = 4.0,
        # weight_last = True,
        # global_scale = 2.0 * 5/cf_net['n_exits'],
        # freeze = False,
    )
    cf_trn = dict(
        call = 'Cifar100',
        n_per_class = 80,
        nval_per_class = 50,
        # n_unl_per_class = 400,
        # lab_unl_ratio = (1, 1),
        augment = True,
        seed = 0,
    )
    cf_val = cf_trn.copy()
    cf_val['augment'] = False
    
    cf_opt = dict(
        call = 'SGD',
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 1e-4,
        nesterov = True,
    )
    batch_size = 64 #!
    val_batch_size = 320
    time_budget = 0  # in hours
    n_epochs = 300
    cf_scheduler = dict(
        call = 'MultiStepLR',
        milestones = [150, 225],
        gamma = 0.1
    )
    
    log_trn_full = False
    save_interval = 0

    #### rarely used params #######################
    cf_grad_transforms = []
    cf_init = dict(
        call = 'ScaleParamInit',
        scale = 1.0,
    )
    
    #### autofill #################################
    if snapshot_name:
        autoname = snapshot_name.split(':')[0]
    elif autoname:
        snapshot_name = utils.numbered_snapshot(autoname)


if __name__ == '__main__':
    ex.run_commandline()
