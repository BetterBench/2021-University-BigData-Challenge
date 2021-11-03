from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import openpyxl
from sklearn.metrics import mean_absolute_error, mean_squared_error



def get_batch(train_x, train_y, TIME_STEPS):
    data_len = len(train_x) - TIME_STEPS
    seq = []
    res = []
    for i in range(data_len):
        seq.append(train_x[i:i + TIME_STEPS])
        res.append(train_y[i:i + TIME_STEPS])  # 取后5组数据
    seq, res = np.array(seq), np.array(res)
    return seq, res

def MAPE_score(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)

def build_model(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE):
    model = Sequential()
    model.add(LSTM(units=128,  activation='relu',return_sequences=True, input_shape=(TIME_STEPS, INPUT_SIZE)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    #全连接，输出， add output layer
    model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
    model.summary()
    model.compile(metrics=['mape'], loss=['mae'], optimizer='adam')
    return model
def train_model(load_first=True,test_tag=False):
    # 时间步长
    TIME_STEPS = 58
    # 预测大小，前3天数据预测预测后3天的数据，每次预测1个小时，，就是一个batch的数据是两个小时的。那需要从9.23 00:00开始预测
    PRED_SIZE = 3*24*58 
    # 输出大小
    OUTPUT_SIZE = 4
    BATCH_START = 58*4
    # 输入大小
    INPUT_SIZE = 67
    # 异常值处理
    # clear_path = 'data/Q3_clear_data.csv'
    # if os.path.exists(clear_path):
    #     data_all = pd.read_csv(
    #         'data/Q3_clear_data.csv')
    # else:
    #     # load_dataset
    #     # 第一轮训练
    #     raw_data = pd.read_csv('data/Or_detection_data.csv')
    #     # 为了后面打乱顺序后还能根据id重新排序回来
    #     raw_data['id'] = raw_data.index
    #     data_one = raw_data.drop(columns=['time'])
    #     data_one = data_one.copy()
    #     # 把异常值换成缺失值
    #     error_index = data_one[data_one.label == 1].index
    #     data_one.loc[error_index,'avg_user'] = np.nan
    #     data_one.loc[error_index,'upward_throughput'] = np.nan
    #     data_one.loc[error_index,'downward_throughput'] = np.nan
    #     data_one.loc[error_index,'avg_activate_users'] = np.nan
    #     # 以小区分组，数据以时间顺序排列
    #     all_df = []
    #     for id in raw_data['community_id'].value_counts().keys():
    #         com_train = raw_data.loc[raw_data['community_id'] == int(id)]
    #         com_train = com_train.drop(columns=['time','label'])
    #         all_df.append(com_train)
    #     inter_all_df = []
    #     # 线性插值
    #     for j in range(len(all_df)):
    #         com_one = all_df[j]
    #         for indexs in ['avg_user', 'upward_throughput', 'downward_throughput', 'avg_activate_users']:
    #             com_one[indexs] = com_one[indexs].interpolate()
    #         inter_all_df.append(com_one)
    #     inter_all_df = pd.concat(inter_all_df)
    #     inter_all_df.set_index(["id"], inplace=True)
    #     inter_all_df.sort_index(inplace=True)
    #     # 给列别特征，one-hot编码
    #     data_all = pd.get_dummies(inter_all_df, prefix=['baseId', 'communityId'], columns=['base_id', 'community_id'])
    #     data_all.to_csv('data/Q3_clear_data.csv', index=False)
    # # 不异常值处理
    raw_data = pd.read_csv('/data/detection_data.csv')
    data_all = raw_data.drop(columns=['time', 'avg_user_label', 'upward_throughput_label',
                             'downward_throughput_label', 'avg_activate_users_label'])
    data_all = pd.get_dummies(data_all, prefix=['baseId', 'communityId'], columns=[
                   'base_id', 'community_id'])
    # 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    train = scaler_X.fit_transform(data_all.values.astype('float32'))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    label_data = data_all[['avg_user', 'upward_throughput','downward_throughput', 'avg_activate_users']]
    train_label = scaler_Y.fit_transform(label_data.values.astype('float32'))
    data_X, data_Y = get_batch(train[:-PRED_SIZE], train_label[PRED_SIZE:], TIME_STEPS)  
    testX, testY = data_X[-1:, :, :], data_Y[-1:, :, :]
    # trainX, trainY = data_X[:-1,:, :], data_Y[:-1 :, :]
    trainX, trainY = data_X, data_Y
    # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
    model = build_model(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE)
    save_path = './weight/Q3_weight.tf'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, save_format='tf', monitor='val_mape',verbose=0, save_best_only=True, save_weights_only=True)
    
    k = trainX.shape[0] % BATCH_START
    trainX, trainY = trainX[k:], trainY[k:]
    # model.load_weights(save_path)
    history = model.fit(trainX, trainY, batch_size=BATCH_START, epochs=8,
                        validation_split=0.1,
                        callbacks=[checkpoint],
                        verbose=1)
    plt.figure(1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.show()
    print()
    y_out = model.predict(testX)
    #预测数据逆缩放 invert scaling for forecast
    y_p = []
    y_t = []
    for i in range(y_out.shape[0]):
        y_pre = scaler_Y.inverse_transform(y_out[i, :, :])
        y_p.append(y_pre)
        #真实数据逆缩放 invert scaling for actual
        y_true = scaler_Y.inverse_transform(testY[i, :, :])
        y_t.append(y_true)
    #画出真实数据和预测数据
    y_pre = np.array(y_p).reshape(-1, OUTPUT_SIZE)
    y_true = np.array(y_t).reshape(-1, OUTPUT_SIZE)
    plt.figure(2)
    for i, index in enumerate(label_data.columns):
        ax1 = plt.subplot(4, 1, i+1)
        mape_score = MAPE_score(y_true[:, i], y_pre[:, i])
        print('{} 指标MAPE ={} '.format(str(index), mape_score))
        mae_score = MAPE_score(y_true[:, i], y_pre[:, i])
        print('{} 指标MAE ={} '.format(str(index), mae_score))
        rmse_socre = np.sqrt(mean_squared_error(y_true[:, i], y_pre[:, i]))
        print('{} 指标RMSE ={} '.format(str(index), rmse_socre))
        plt.sca(ax1)
        plt.plot(y_pre[:, i], label="Pre-"+str(index))
        plt.plot(y_true[:, i], label="True-"+str(index))
        plt.legend()
    plt.show()
    if test_tag ==True:
        model.load_weights(save_path)
        dataX,dataY = get_batch(train, train_label, TIME_STEPS)
        test_data = dataX[-int(4176/TIME_STEPS):,:,:]
        prediction_3day = model.predict(test_data)
        pre_data = np.array([])
        for i in range(prediction_3day.shape[0]):
            tmp = scaler_Y.inverse_transform(prediction_3day[i, :, :])
            if i ==0:
                pre_data = tmp
            else:
                pre_data = np.vstack((pre_data,tmp))
            print()
        result_arr = pre_data
        result_df = pd.read_excel('data/附件2：赛题A预测表格.xlsx', sheet_name='预测数据')
        result_df['小区内的平均用户数'] = list(result_arr[:, 0])
        result_df['小区PDCP层所接收到的上行数据的总吞吐量比特'] = list(result_arr[:,1])
        result_df['小区PDCP层所发送的下行数据的总吞吐量比特'] = list(result_arr[:, 2])
        result_df['平均激活用户数'] = list(result_arr[:,3])
        result_df.to_excel('/data/prediction_table.xlsx',sheet_name='预测数据',index =False)
if __name__ == '__main__':
    train_model(True,True)

