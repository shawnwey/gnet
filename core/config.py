import logging
import time
from pathlib import Path


def create_log(output_dir, exp_id, phase='train'):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    # dataset / model /
    final_output_dir = root_output_dir / exp_id

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # weights
    weights_dir = root_output_dir / 'weights'
    print('=> creating {}'.format(weights_dir))
    weights_dir.mkdir(parents=True, exist_ok=True)

    # log
    log_dir = final_output_dir / 'log'
    print('=> creating {}'.format(log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(exp_id, time_str, phase)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(weights_dir), str(log_dir)