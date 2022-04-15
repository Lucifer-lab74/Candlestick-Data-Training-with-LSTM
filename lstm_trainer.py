
import numpy as np
import pandas as pd
from keras.layers import Dense, CuDNNLSTM, Dropout, Bidirectional
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, SGD

model_dir = "/content/drive/MyDrive/BTCLSTM_DATA/TestModels"
def read_dataset():
    file_location = "/content/drive/MyDrive/BTCLSTM_DATA/1L_data.csv"
    csv_data = pd.read_csv(file_location)
    csv_data = csv_data.values
    all_columns = []
    for column_value in csv_data.tolist():
        all_columns.append(column_value)
    return all_columns[-50000:]


def pre_processing(data_to_process, win_size):
    all_close_values = []
    all_col_values = []
    scalar = MinMaxScaler(feature_range=(0, 1))

    for row in data_to_process:
        all_close_values.append([row[4]])
        all_col_values.append([row[1], row[2], row[3], row[4]])

    x_train, y_train = [], []
    scalar.fit(all_col_values)
    # all_close_values = np.reshape(all_close_values, (-1, 4))
    for cut_index in range(win_size, len(all_col_values)):
        temp = scalar.transform(all_col_values[cut_index - win_size:cut_index + 1])
        x_train.append(temp[:len(temp) - 1])
        y_train.append(temp[len(temp) - 1])

    x_train = np.stack(x_train, axis=0)
    y_train = np.array(y_train, dtype=object)

    return x_train, y_train, scalar


def get_vanilla_model(win_size, n_features, adam):
    model = Sequential()
    model.add(CuDNNLSTM(units=150, input_shape=(win_size, n_features)))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model


def get_bidirectional_model(win_size, n_features, adam):
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(units=100, input_shape=(win_size, n_features), return_sequences=True)))
    model.add(Dropout(rate=0.25))
    model.add(Bidirectional(CuDNNLSTM(units=50, return_sequences=True)))
    model.add(Dropout(rate=0.25))
    model.add(Bidirectional(CuDNNLSTM(units=25)))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model

def get_stack_model(win_size, n_features, adam):
    model = Sequential()
    model.add(CuDNNLSTM(units=150, input_shape=(win_size, n_features), return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(CuDNNLSTM(units=100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(CuDNNLSTM(units=50, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(CuDNNLSTM(units=25))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model


def train_vanilla_model(x_train, y_train, window_size, n_features, eph, bs, adam):
    _model = get_vanilla_model(window_size, n_features, adam)
    _model.fit(x_train, y_train, epochs=eph, batch_size=bs, verbose=1)
    _model.save(model_dir + "/VanillaNew{}.h5".format(window_size), save_format='h5')


def train_bidirectional_model(x_train, y_train, window_size, n_features, eph, bs, adam):
    _model = get_bidirectional_model(window_size, n_features, adam)
    _model.fit(x_train, y_train, epochs=eph, batch_size=bs, verbose=1)
    _model.save(model_dir + "/Bidirectional{}.h5".format(window_size), save_format='h5')


def train_stack_model(x_train, y_train, window_size, n_features, eph, bs, adam):
    _model = get_stack_model(window_size, n_features, adam)
    _model.fit(x_train, y_train, epochs=eph, batch_size=bs)
    _model.save(model_dir + "/Stack{}.h5".format(window_size), save_format='h5')


def train_all_models():
    all_data = read_dataset()
    lrs = [0.001, 0.001, 0.05]
    btch = [1024, 2048, 1024]
    c = 0   
    for ws in [50, 150, 30]:
        adam = Adam(learning_rate=lrs[c])
        c += 1
        window_size = ws
        x_train, y_train, _ = pre_processing(all_data, window_size)
        x_train = np.asarray(x_train).astype('float32')
        y_train = np.asarray(y_train).astype('float32')
        eph = 5000
        if ws == 50:
          train_stack_model(x_train, y_train, window_size, 4, eph, btch[c], adam)
          train_bidirectional_model(x_train, y_train, window_size, 4, eph, btch[c], adam)
        else:
          train_vanilla_model(x_train, y_train, window_size, 4, eph, btch[c], adam)
          train_stack_model(x_train, y_train, window_size, 4, eph, btch[c], adam)
          train_bidirectional_model(x_train, y_train, window_size, 4, eph, btch[c], adam)


if __name__ == '__main__':
    train_all_models()