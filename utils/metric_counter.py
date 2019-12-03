import os
import logging
import time
import datetime
import yaml

import numpy as np
from tensorboardX import SummaryWriter
from collections import defaultdict


class Collector:
    def __init__(self):
        self.data = defaultdict(list)

    def add(self, key, value):
        self.data[key].append(value)

    def get(self, item):
        return self.data[item]

    def keys(self):
        return self.data.keys()

    __getitem__ = get
    __setitem__ = add


class SubMetricCounter:
    def __init__(self, writer, metric_name, loss_name, counter_name):
        self.writer = writer
        self.metric_name = metric_name
        self.loss_name = loss_name
        self.name = counter_name

    def clear(self):
        self.loss = []
        self.metric = []

    def add_losses(self, loss):
        self.loss.append(loss)

    def add_metrics(self, metric):
        self.metric.append(metric)

    def add_batch_losses(self, loss_iterable):
        loss_iterable = list(loss_iterable)
        self.loss.extend(loss_iterable)
        return len(loss_iterable)

    def add_batch_metrics(self, metric_iterable):
        metric_iterable = list(metric_iterable)
        self.metric.extend(metric_iterable)
        return len(metric_iterable)

    def last_n_losses(self, n):
        return self.loss[-n:]

    def last_n_metrics(self, n):
        return self.metric[-n:]

    def get_loss(self):
        return np.mean(self.loss)

    def tqdm_message(self, last_n=None):
        # `last_n_loss` should be always 1,
        # or the whole epoch wanted
        if last_n is None:
            last_n = max(len(self.loss), len(self.metric))
            last_n_loss = last_n
        else:
            last_n_loss = 1
        mean_loss = np.mean(self.last_n_losses(last_n_loss))
        mean_metric = np.mean(self.last_n_metrics(last_n))
        return {"loss_%s" % self.loss_name: mean_loss,
                "metr_%s" % self.metric_name: mean_metric}

    def write_epoch(self, epoch_num):
        print(np.mean(self.metric))
        self.writer.add_scalars(
            'epoch/loss_{}'.format(self.loss_name),
            {self.name: np.mean(self.loss)},
            epoch_num)
        self.writer.add_scalars(
            'epoch/metr_{}'.format(self.metric_name),
            {self.name: np.mean(self.metric)},
            epoch_num)


class MetricCounter:
    def __init__(self, config, stage, metric_name="Def. metr.", loss_name="Def. loss"):
        stage = str(stage)
        print(config['experiment_path'], "runs-", stage)
        path = os.path.join(config['experiment_path'], "runs-{}".format(stage))
        if not os.path.exists(path):
            os.makedirs(path)

        # Setup logging
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
        logging.basicConfig(filename=os.path.join(path, '{}.log'.format(st)), level=logging.DEBUG)

        with open(os.path.join(config['experiment_path'], 'config.yaml'), 'w') as f:
            yaml.dump(config, f)


        self.writer = SummaryWriter(path)

        self.metric_name = metric_name
        self.loss_name = loss_name

        self.best_metric = 0
        self.train = SubMetricCounter(self.writer, metric_name, loss_name, "train")
        self.val = SubMetricCounter(self.writer, metric_name, loss_name, "val")

    def write_batch(self, last_n, reducer, global_step):
        """
        Writes train batch
        :param last_n: number of last values to reduce
        :param reducer:
        :param step:
        :return:
        """
        mc = self.train
        # loss is always 1
        mc.writer.add_scalar(
            'train_batch/loss_{}'.format(self.loss_name),
            reducer(mc.last_n_losses(1)),
            global_step)
        mc.writer.add_scalar(
            'train_batch/metr_{}'.format(self.metric_name),
            reducer(mc.last_n_metrics(last_n)),
            global_step)

    def update_best_model(self):
        cur_metric = np.mean(self.val.metric)
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False
