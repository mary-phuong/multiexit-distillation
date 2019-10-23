import sacred
from sacred.observers import FileStorageObserver
import time
import torch
import main, utils
from utils import device, dict_drop, RUNS_DB_DIR


sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
sacred.SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False
ex = sacred.Experiment()
ex.observers.append(FileStorageObserver.create(str(RUNS_DB_DIR)))

#########################################################

@ex.capture
def get_net(cf_net, parent_snapshot, gpu):
    if parent_snapshot:
        net = main.load_snapshot(parent_snapshot)
    else:
        net = getattr(main, cf_net['call'])(**dict_drop(cf_net, 'call'))

    if isinstance(gpu, int):
        net.to(device(gpu))
    else:
        net = torch.nn.DataParallel(net, gpu)
        net.to(device(gpu[0]))
    
    return net

#########################################################

@ex.main
def train(cf_init, cf_trn, cf_val, cf_opt, cf_loss, cf_grad_transforms,
          cf_scheduler, batch_size, val_batch_size, gpu, time_budget, n_epochs,
          parent_snapshot, snapshot_name, log_trn_full, save_interval):

    torch.backends.cudnn.benchmark = True
    main_gpu = gpu if isinstance(gpu, int) else gpu[0]
    dvc = device(main_gpu)
    
    net = get_net()
    loss_f = getattr(main, cf_loss['call'])(**dict_drop(cf_loss, 'call'))

    init = getattr(main, cf_init['call'])(**dict_drop(cf_init, 'call'))
    for p in net.parameters():
        init(p)
    
    Opt = getattr(torch.optim, cf_opt['call'])
    opt = Opt(net.parameters(), **dict_drop(cf_opt, 'call'))

    ep = int(parent_snapshot.split('_')[-1][2:]) if parent_snapshot else 0
    Scheduler = getattr(torch.optim.lr_scheduler, cf_scheduler['call'])
    scheduler = Scheduler(opt, last_epoch=-1,
                          **dict_drop(cf_scheduler, 'call'))
    for i in range(ep): scheduler.step()

    Data = getattr(main, cf_trn['call'])
    trn_iter = Data(split='train', batch_size=batch_size, gpu=main_gpu,
                    **dict_drop(cf_trn, 'call'))
    Data = getattr(main, cf_val['call'])
    val_iter = Data(split='val', batch_size=val_batch_size, gpu=main_gpu,
                    **dict_drop(cf_val, 'call'))

    grad_tforms = []
    for cf in cf_grad_transforms:
        grad_tforms += [getattr(main, cf['call'])(**dict_drop(cf, 'call'))]
    
    if save_interval:
        saver = utils.IntervalSaver(snapshot_name, ep, save_interval)
    else:
        saver = utils.RecordSaver(snapshot_name, ep)
    start_time = time.time()
    print('\t'.join(['ep', 'loss'] + loss_f.metric_names))
    
    while time.time() - start_time < 3600 * time_budget or ep < n_epochs:
        scheduler.step()
        for trn_tuple in trn_iter:
            trn_tuple = [t.to(dvc) for t in trn_tuple]
            opt.zero_grad()
            loss = loss_f(net.train(True), *trn_tuple, ep)
            loss.backward()
            for tform in grad_tforms:
                for par in net.parameters():
                    tform(par.grad.data)
            opt.step()
            
        ep += 1
        trn_metrics = (main.validate(loss_f, net, trn_iter, main_gpu)
                       if log_trn_full else loss_f.trn_metrics())
        val_metrics = main.validate(loss_f, net, val_iter, main_gpu)
        print(utils.tab_str(ep, loss, *trn_metrics))
        print(utils.tab_str('', 0.0, *val_metrics))

        saver.save(val_metrics[0], net, main_gpu, ep)

    del net, opt, trn_iter, val_iter
