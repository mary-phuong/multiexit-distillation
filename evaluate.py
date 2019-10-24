import sacred
from sacred.observers import FileStorageObserver
import main, utils
from utils import device, dict_drop, RUNS_DB_DIR


sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
sacred.SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False
ex = sacred.Experiment()
ex.observers.append(FileStorageObserver.create(str(RUNS_DB_DIR)))


@ex.main
def evaluate(cf_test, cf_loss, batch_size, snapshot_name, gpu):
    net = main.load_snapshot(snapshot_name).to(device(gpu))
    loss_f = getattr(main, cf_loss['call'])(**dict_drop(cf_loss, 'call'))

    Data = getattr(main, cf_test['call'])
    test_iter = Data(split='test', batch_size=batch_size, gpu=gpu,
                     **dict_drop(cf_test, 'call'))

    print('\t'.join(loss_f.metric_names))
    val_metrics = main.validate(loss_f, net, test_iter, gpu)
    print(utils.tab_str(*val_metrics))

