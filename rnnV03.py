from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.callbacks import LearningRateScheduler
import dataManger
from keras import backend as K
import tensorflow as tf


# 定义学习率调度函数
def lr_schedule(epoch):
    # 在这里定义您的学习率调整策略
    if epoch < 5:
        return learn_rate
    else:
        new_lr = learn_rate * tf.math.exp(0.1 * (5 - epoch))
        K.set_value(model.optimizer.lr, new_lr)
        print("Learning rate changed to {}".format(new_lr))
    return K.get_value(model.optimizer.lr)

X_train = dataManger.X_train
Y_train = dataManger.Y_train
X_test = dataManger.X_test
Y_test = dataManger.Y_test

learn_rates = [0.01]
in_units = [24]
num_layers = [2]
time_steps = [12]
results = []

for learn_rate in learn_rates:
    for in_unit in in_units:
        for num_layer in num_layers:
            for time_step in time_steps:
                # 在此处使用当前的超参数进行模型训练，并评估性能
                # 初始化标准化器，只标准化x，滑动取值
                startTime = datetime.now()
                #标准化
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                # 设置训练集的特征列表和对应标签列表
                x_train = []
                y_train = []
                for i in np.arange(time_step, len(X_train)):
                    x_train.append(X_train[i - time_step:i, :])
                    y_train.append(Y_train[i - 1])
                # 将训练集由list格式变为array格式
                x_train = np.array(x_train)
                y_train = np.array(y_train).reshape(-1, 1)

                # 调整测试集
                x_test = []
                y_test = []
                # 按照上述规律不断滑动取值
                for i in np.arange(time_step, len(X_test)):
                    x_test.append(X_test[i - time_step:i, :])
                    y_test.append(Y_test[i - 1])

                # 将训练集由list格式变为array格式
                x_test = np.array(x_test)
                y_test = np.array(y_test).reshape(-1, 1)
                # 创建Sequential模型
                model = Sequential()

                # 添加第一个隐藏层
                if num_layer == 1:
                    model.add(SimpleRNN(units=in_unit, activation='relu', return_sequences=False,
                                        input_shape=(time_step, X_train.shape[1])))
                else:
                    model.add(SimpleRNN(units=in_unit, activation='relu', return_sequences=True,
                                        input_shape=(time_step, X_train.shape[1])))
                    for i in range(num_layer - 2):
                        model.add(SimpleRNN(units=in_unit, activation='relu', return_sequences=True))
                    model.add(SimpleRNN(units=in_unit, activation='relu', return_sequences=False))

                # 添加输出层
                model.add(Dense(units=1, activation='linear'))

                # 设置学习率
                custom_optimizer = Adam(learning_rate=learn_rate)
                model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

                # 查看模型结构
                model.summary()
                # 定义LearningRateScheduler回调
                lr_scheduler = LearningRateScheduler(lr_schedule)
                # 训练模型
                # 定义 EarlyStopping 回调
                early_stopping = EarlyStopping(monitor='loss',  # 监控验证集上的损失
                                               min_delta=0.1,
                                               patience=5,  # 如果连续 5 个 epoch 没有改善，则提前停止
                                               restore_best_weights=True)  # 恢复在损失最小时的模型参数
                # 假设 model 是你的 Keras 模型
                model.fit(x_train, y_train, batch_size=128, epochs=100,
                          callbacks=[early_stopping, lr_scheduler]
                          )

                # 评估RNN模型
                y_pred = model.predict(x_test)
                r2 = r2_score(y_test, y_pred)
                endTime = datetime.now()
                time = endTime - startTime
                results.append([learn_rate, in_unit, num_layer, time_step, r2, time])
results = pd.DataFrame(results, columns=['learn_rate', 'in_unit', 'num_layer', 'time_step', 'r2', 'time'])
excel_filename = 'results1.xlsx'
results.to_excel(excel_filename, index=False)
