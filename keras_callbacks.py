from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, TensorBoard
from math import pi, cos
import os


def generate_learning_rate_schedule(min_lr, max_lr, period, decay_start=10):
    def learn_rate_func(idx):
        idx = max(0, idx-decay_start)
        idx = float(idx % period)
        diff_lr = max_lr - min_lr
        multiplier_lr = cos(pi/2 * idx / (period-1))
        return min_lr + diff_lr * multiplier_lr
    return LearningRateScheduler(learn_rate_func, verbose=1)


def generate_check_point(model_name, target='all'):
    model_dir = os.path.join(os.path.join('../checkpoints', model_name), target)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_path = model_dir+'_{epoch:02d}_loss_{val_loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
    return checkpoint


def generate_early_stopping():
    early_stopping = EarlyStopping(patience=5)
    return early_stopping


def generate_tensorboard(model_name, target):
    dir = '../logs/' + model_name + '_' + target
    if not os.path.exists(dir):
        os.makedirs(dir)
    tb = TensorBoard(log_dir=dir)
    return tb