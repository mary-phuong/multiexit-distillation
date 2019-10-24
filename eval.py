from evaluate import ex


@ex.config
def config():
    gpu = 0          # GPU ordinal or -1 for CPU
    snapshot_name = 'test:0'
    
    cf_test = dict(  # test dataset
        call = 'Cifar100',
        seed = 0,
    )
    
    cf_loss = dict(  # evaluation metric
        call = '_MultiExitAccuracy',
        n_exits = 11,
        acc_tops = (1, 5),
    )
    
    batch_size = 250


if __name__ == '__main__':
    ex.run_commandline()
