###################################################################################
# trainertorch.py: a simple trainer                                               #
###################################################################################
# University of Applied Sciences Munich                                           #
# Dept of Electrical Enineering and Information Technology                        #
# Institute for Applications of Machine Learning and Intelligent Systems (IAMLIS) #
#                                                        (c) Alfred Schöttl 2022  #
###################################################################################
#                                                                                 #
# These classes give some basic functionality to train and evaluate               #
# networks.                                                                       #
#                                                                                 #
###################################################################################

import numpy as np
import torch
from tqdm.notebook import trange
import matplotlib.pyplot as plt
import os


class HistoryManager:

    """The HistoryManager provides basic functions to manage metric data which are computed
    during the training. The current training step is stored in overall_steps. The metric
    data consists of a <name> and a <val>, the name of test data should be test_<name>.
    - store data of current training step:  self._add_to_hist(<name>, <val>)
    - query data of current training step:  self.<name>
    - query current averaged data:          self.avg_<name>
    - query data for all steps:             self.get_hist(<name>) or self.get_avg_hist(<name>)
    - plot data for all steps:              self.plot_val(<name>)
    """

    SAVE_FOLDER_VAL = os.path.join(os.getcwd(), "evaluation_data")

    def __init__(self, config, steps_per_epoch_train, steps_per_epoch_test):
        self.hist = None
        self.overall_steps = 0
        self.avg_steps_train = config.get("avg_steps_train", steps_per_epoch_train)
        self.avg_steps_test = config.get("avg_steps_test", steps_per_epoch_test)

    def _reset_hist(self):
        self.hist = {}
        self.overall_steps = 0

    def _add_to_hist(self, name, val):
        """Adds a value to the history. The history is a dictionary. Each entry is itself a dictionary
        with the global step number as key."""
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        if name not in self.hist:
            self.hist[name] = {}
        if self.overall_steps not in self.hist[name]:
            self.hist[name][self.overall_steps] = val
        else:
            self.hist[name][self.overall_steps] += val

    def _divide_hist_val_by(self, name, c):
        self.hist[name][self.overall_steps] /= c

    def get_curr(self, name):
        """Returns the current value of the history."""
        if name in self.hist:
            return self.hist[name][self.overall_steps]
        else:
            raise NotImplementedError(f"Name {name} is not available.")

    def get_hist(self, name):
        """Returns the pair T, V of the metric name from the history. T, V are two vectors with time points and the values."""
        if name in self.hist:
            return np.array(list(self.hist[name].keys())), np.array(
                list(self.hist[name].values())
            )
        else:
            raise NotImplementedError(f"Name {name} is not available.")

    def get_avg(self, name):
        """Returns the current averaged value of the history. The average is computed
        as moving average with a configurable window."""
        if name in self.hist:
            n = self.avg_steps_test if name[:5] == "test_" else self.avg_steps_train
            h = self.hist[name]
            h = [
                h[k]
                for k in range(
                    self.overall_steps - n + 1 if self.overall_steps - n + 1 > 0 else 0,
                    self.overall_steps + 1,
                )
            ]
            return np.array(h).mean()
        else:
            raise NotImplementedError(f"Name {name} is not available.")

    def get_avg_hist(self, name):
        """Returns the pair T, V of the averaged metric name from the history. T, V are two vectors with time points and the
        values. The average is computed as moving average with a configurable window."""
        if name in self.hist:
            T, val = np.array(list(self.hist[name].keys())), np.array(
                list(self.hist[name].values())
            )
            n = min(
                self.avg_steps_test if name[:5] == "test_" else self.avg_steps_train,
                len(val),
            )
            cs = np.cumsum(val)
            smoothed_val_part1 = cs[:n] / np.arange(1, n + 1)
            smoothed_val_part2 = ((cs[n:] - cs[:-n]) / n) if len(cs) > n else []
            return T, np.concatenate([smoothed_val_part1, smoothed_val_part2])
        else:
            raise NotImplementedError(f"Name {name} is not available.")

    def plot_val(self, name):
        """Plot a value from the metrics. The metric names may be prepended by "avg_" to obtain averaged versions.
        The average is computed as moving average with a configurable window."""
        plt.gca().set_title(name)
        if name[:4] == "avg_":  # compute the moving average of the history value
            T, val = self.get_avg_hist(name[4:])
            name_without_avg = name[4:]
        else:
            T, val = self.get_hist(name)
            name_without_avg = name
        plt.plot(T, val)
        if (
            "test_" + name_without_avg in self.hist
            and len(self.hist["test_" + name_without_avg]) > 0
        ):
            if name[:4] == "avg_":  # compute the moving average of the history value
                T, val = self.get_avg_hist("test_" + name[4:])
            else:
                T, val = self.get_hist("test_" + name)
            plt.plot(T, val, "r")
        plt.grid()

    def save_val(self, name):
        T, val = self.get_hist(name)
        name_file = name + ".npz"
        path = os.path.join(self.SAVE_FOLDER_VAL, name_file)
        np.savez(path, steps=T, val=val)

    def __getattr__(self, name):
        """This function allows access to the metrics' data by the syntax "self.metric_name", which is useful in the
        report function. The metric names may be prepended by "avg_" to obtain averaged versions.
        The average is computed as moving average with a configurable window."""
        if name in dir(self):
            return getattrib(self, name)
        elif name in self.hist:
            return self.get_curr(name)
        elif name[:4] == "avg_":
            if name[4:] in self.hist:
                return self.get_avg(name[4:])
            else:
                raise NotImplementedError(f"Name {name[4:]} is not available.")
        else:
            raise NotImplementedError(f"Name {name} is not available.")


class Trainer(HistoryManager):
    def __init__(self, mdl, config={}):
        self.mdl = mdl
        self.n_epochs = config.get("n_epochs")  # number of epochs
        self.batch_size = config.get("batch_size")  # batch size used
        self.opt = config.get("opt")  # optimizer
        self.train_dl = config.get("train_dl")  # data loader
        self.test_dl = config.get(
            "test_dl"
        )  # the test parts are implemented in the derived class
        self.report_period = config.get(
            "report_period", 100
        )  # output a report every report_period steps
        self.n_save_weights = config.get("n_save_weights", self.n_epochs)
        self.dir_save_weights = config.get("dir_save_weights", self.n_epochs)

        self.device = config.get("device")
        self.mdl = self.mdl.to(self.device)
        if self.train_dl is not None:
            self.test_period = config.get(
                "test_period", len(self.train_dl.dataset) // self.batch_size
            )
            steps_per_epoch_train = len(self.train_dl.dataset) // self.batch_size
            steps_per_epoch_test = (
                len(self.train_dl.dataset) // (self.batch_size * self.test_period)
                if self.test_dl is not None
                else None
            )
        else:
            self.test_period = steps_per_epoch_train = steps_per_epoch_test = None
        super().__init__(config, steps_per_epoch_train, steps_per_epoch_test)

    def train_step(self, x, y, label, contact_points):
        """Executes one training step."""
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        label = label.permute(0, 2, 1)
        contact_points = contact_points.permute(0, 2, 1)

        x = x.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)
        contact_points = contact_points.to(device=self.device, non_blocking=True)
        self.opt.zero_grad()
        label = label.to(device=self.device)
        X = x, y
        y_pred_cont, y_pred_pts = self.mdl.forward(X)
        loss = self.mdl.loss_fn(label, y_pred_cont, contact_points, y_pred_pts)
        loss = loss.to(device=self.device)
        loss.backward()
        loss = loss.to(device=self.device)
        self.opt.step()

        return loss, (y_pred_pts, y_pred_cont)

    def train(self):
        """Starts the traing run. It also calls tests regularly if the test functions are available
        (inherited by the TrainerWithTest class)
        """
        self._reset_hist()
        self.mdl.to(device=self.device)
        self.mdl.train(True)
        for epoch in trange(self.n_epochs):
            print(f"epoch {epoch}:")
            for step, (x, y, label, contact_points) in enumerate(self.train_dl):
                loss, y_pred = self.train_step(x, y, label, contact_points)
                label = label.to(device=self.device)
                contact_points = contact_points.to(device=self.device)
                Y_true = label, contact_points
                m = self.metrics(L=loss, X=x, Y_true=Y_true, Y_pred=y_pred)
                for name, val in m.items():
                    self._add_to_hist(name, val)
                if step % self.report_period == 0:
                    self.train_report(epoch, step)
                if self.test_dl is not None and (step + 1) % self.test_period == 0:
                    self.mdl.train(False)
                    self.test(epoch, step)
                    self.mdl.train(True)
                self.overall_steps += 1
            if epoch > 1:
                if epoch % self.n_save_weights == 0:
                    print("--------------saving weights ----------------")
                    self._save_weights_cp(epoch)

        return self.hist

    def _save_weights_cp(self, epoch):
        savepath = self.dir_save_weights + "_epoch_" + str(epoch) + ".pth"
        torch.save(self.mdl.state_dict(), savepath)

    def summary(self):
        """Outputs a summary of the implemented model."""
        try:
            import torchsummaryX
        except:
            print(
                "No summary available: install torchsummaryx (don't forget the x, you may use pip)"
            )
        test = torch.zeros(
            *self.mdl.in_shape, dtype=torch.float64, device=self.device
        ), torch.zeros((10, 3, 3000), dtype=torch.float64, device=self.device)
        # torchsummaryX.summary(
        #     self.mdl,
        #     test,
        # )

    # --- default implementations of the metrics and the report methods, may be overwritten

    def metrics(self, L, X, Y_pred, Y_true):
        """Computes all the metrics which shall be reported during training. The metrics are returned
        as dictionary. May be overwritten in a derived class.
        """
        return {"loss": L}

    def train_report(self, epoch, step):
        """Outputs the report. The frequency of reports can be configured. May be overwritten in a
        derived class.
        """
        print(f"    {epoch:03d}/{step:05d}:  loss {self.loss:6.4f}")


class TrainerWithTest(Trainer):
    def __init__(self, mdl, config):
        self.test_metrics = self.metrics
        super().__init__(mdl, config)

    def test_step(self, x, y, label):
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        label = label.permute(0, 2, 1)
        x = x.to(device=self.device, non_blocking=True)
        y = y.to(device=self.device, non_blocking=True)
        X = x, y
        y_pred_pts, y_pred_cont = self.mdl.forward(X)
        loss = self.mdl.loss_fn(label, label, y_pred_cont)
        return loss, (y_pred_pts, y_pred_cont)

    def test_report(self, epoch, step):
        print("-" * 50)
        print(f"> {epoch:03d}/{step+1:05d}:  loss {self.test_loss:6.4f}")
        print("-" * 50)

    def test(self, epoch, train_step):
        """The test is performed over all batches of the test set. We sum up all metrics."""
        for step, (x, y, label) in enumerate(self.train_dl):
            loss, y_pred = self.test_step(x, y, label)
            label = label.to(device=self.device)
            m = self.test_metrics(L=loss, X=x, Y_true=label, Y_pred=y_pred)
            for name, val in m.items():
                self._add_to_hist("test_" + name, val)
        for name, val in m.items():
            self._divide_hist_val_by("test_" + name, step + 1)
        self.test_report(epoch, train_step)

    def visualize_test_predictions(self, x, y_pred):
        x = x.cpu().numpy().squeeze()
        y_pred = y_pred.detach().cpu().numpy().squeeze()
        # print(y_pred.max())
        # print(y_pred.min())
        # show_voxel_texture(x, y_pred)
