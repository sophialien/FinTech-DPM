
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .tradertrainer import TraderTrainer
import logging
# import tflearn


class RollingTrainer(TraderTrainer):
    def __init__(self, config, restore_dir=None, save_path=None, agent=None, device="cpu"):
        config["training"]["buffer_biased"] = config["trading"]["buffer_biased"]
        config["training"]["learning_rate"] = config["trading"]["learning_rate"]
        TraderTrainer.__init__(self, config, restore_dir=restore_dir, save_path=save_path,
                               agent=agent, device=device)

    @property
    def agent(self):
        return self._agent

    @property
    def coin_list(self):
        return self._matrix.coin_list

    @property
    def data_matrices(self):
        return self._matrix

    @property
    def rolling_training_steps(self):
        return self.config["trading"]["rolling_training_steps"]

    def __rolling_logging(self):
        fast_train = self.train_config["fast_train"]
        if not fast_train:
            v_pv, v_log_mean = self._evaluate("validation",
                                              'portfolio_value',
                                              'log_mean')
            t_pv, t_log_mean = self._evaluate("test", 'portfolio_value', 'log_mean')
            loss_value = self._evaluate("training", 'loss')
            print(t_pv)

            logging.info('training loss is %s\n' % loss_value)
            logging.info('the portfolio value on validation asset is %s\nlog_mean is %s\n' %
                         (v_pv,v_log_mean))
            logging.info('the portfolio value on test asset is %s\n mean is %s' % (t_pv,t_log_mean))

    def decide_by_history(self, history, last_w):
        result = self._agent.decide_by_history(history, last_w)
        return result

    def rolling_train(self, online_w=None):
        steps = self.rolling_training_steps
        if steps > 0:
            self._matrix.append_experience(online_w)
            for i in range(steps):
                x, y, last_w, w, y_cont = self.next_batch()
                self._agent.train(x, y, last_w, w, y_cont)
            self.__rolling_logging()
