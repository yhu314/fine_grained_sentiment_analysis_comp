from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from math import pi, cos


def generate_learning_rate_schedule(min_lr, max_lr, period, decay_start=10):
    def learn_rate_func(idx):
        idx = max(0, idx-decay_start)
        idx = float(idx % period)
        diff_lr = max_lr - min_lr
        multiplier_lr = cos(pi/2 * idx / (period-1))
        return min_lr + diff_lr * multiplier_lr
    return LearningRateScheduler(learn_rate_func, verbose=1)


def generate_check_point(model_name):
    file_path = model_name+'_{epoch:02d}_loss_{val_loss:.4f}_acc_{acc:0.4f}_f1_{f1:0.4f}.hdf5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
    return checkpoint


def generate_early_stopping():
    early_stopping = EarlyStopping(patience=5)
    return early_stopping