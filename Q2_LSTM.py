from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


def get_batch(train_x, train_y, TIME_STEPS):
    data_len = len(train_x) - TIME_STEPS
    seq = []
    res = []
    for i in range(data_len):
        seq.append(train_x[i:i + TIME_STEPS])
        res.append(train_y[i:i + TIME_STEPS])  # 取后5组数据
    seq, res = np.array(seq), np.array(res)
    return seq, res

def build_model(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE):
    model = Sequential()
    model.add(LSTM(units=128,  activation='relu',
              return_sequences=True, input_shape=(TIME_STEPS, INPUT_SIZE)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    #全连接，输出， add output layer
    model.add(TimeDistributed(Dense(OUTPUT_SIZE,activation='sigmoid')))
    model.summary()
    model.compile(metrics=['accuracy'], loss=['binary_crossentropy'], optimizer='adam')
    return model

def train_model():
    TIME_STEPS = 58*4   # 自定义
    PRED_SIZE = 1*24*58  # 自定义
    OUTPUT_SIZE = 1
    BATCH_START = 24*10 
    INPUT_SIZE = 67
    raw_data = pd.read_csv(
        '/data/Or_detection_data.csv')
    data_one = raw_data.drop(columns=['time'])
    a = data_one['avg_user_label'].tolist()
    b = data_one['upward_throughput_label'].tolist()
    c = data_one['downward_throughput_label'].tolist()
    d = data_one['avg_activate_users_label'].tolist()
    label_data = list(map(lambda x: (x[0] or x[1] or x[2] or x[3]), zip(a, b, c, d)))
    label_data = np.array(label_data)
    label_data = label_data[:,np.newaxis]
    # # 给列别特征，one-hot编码
    data_one = pd.get_dummies(data_one, prefix=['baseId', 'communityId'], columns=['base_id', 'community_id'])

    # 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    train = scaler_X.fit_transform(data_one.astype('float32'))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    train_label = scaler_Y.fit_transform(label_data)
    data_X, data_Y = get_batch(train[:-PRED_SIZE], train_label[PRED_SIZE:], TIME_STEPS)
    testX, testY = data_X[-1000:, :, :], data_Y[-1000:,:,:]
    # testX, testY = data_X, data_Y
    # 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
    model = build_model(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE)
    save_path = './weight/Q2_weight_all.tf'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path, save_format='tf', monitor='val_accuracy',verbose=0, save_best_only=True, save_weights_only=True)
    k = data_X.shape[0] % BATCH_START
    trainX, trainY = data_X[k:], data_Y[k:]
    # model.load_weights(save_path)
    history = model.fit(trainX, trainY, batch_size=BATCH_START, epochs=7,
                        validation_split=0.1,
                        callbacks=[checkpoint],
                        verbose=1)
    plt.figure(1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    plt.show()
    y_p = model.predict(testX)
    y_t = testY
    y_pre = np.array(y_p).reshape(-1, OUTPUT_SIZE)
    y_true = np.array(y_t).reshape(-1, OUTPUT_SIZE)
    score = f1_score(y_true, np.argmax(y_pre, axis=1), average='macro')
    print('模型F1得分 ={} '.format(str(score)))
    print()



if __name__ == '__main__':
    train_model()
