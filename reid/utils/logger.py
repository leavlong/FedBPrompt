import logging
import os
import sys
import os.path as osp
from datetime import datetime
def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 获取当前时间并格式化为字符串（例如：2023-10-25_14-30-00）
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 构造日志文件名，如：train_log_2023-10-25_14-30-00.txt
    trainlog_filename = f"train_log_{current_time}.txt"
    testlog_filename = f"test_log_{current_time}.txt"
    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, trainlog_filename), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, testlog_filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger