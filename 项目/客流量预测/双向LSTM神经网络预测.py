import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout , LSTM , Bidirectional 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 加载数据
data = pd.read_csv("./train.csv")  # 替换为你的数据文件路径
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
flow_data = data['Flow'].values.reshape(-1, 1)
print('加载数据成功')
# 数据归一化
scaler = MinMaxScaler()
scaled_flow_data = scaler.fit_transform(flow_data)
print("数据归一化成功")
# 构建时间窗口
n_steps = 30  # 时间窗口大小
X = []
y = []
for i in range(len(scaled_flow_data) - n_steps):
    X.append(scaled_flow_data[i:i+n_steps])
    y.append(scaled_flow_data[i+n_steps])
X = np.array(X)
y = np.array(y)

# 分割训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print('训练集、测试集分割成功')
# 构建LSTM模型
# model = Sequential()
# model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
model = Sequential()
model.add(Bidirectional(LSTM(50,input_shape=(n_steps, 1))))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer="adam",loss="mse")
print('开始训练')
# 模型训练

history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0,validation_split=0.1, shuffle=True)
loss_values = history.history['loss']
epochs = range(1, len(loss_values) + 1)
# 使用seaborn库绘制损失值曲线
sns.set(style="darkgrid")
plt.plot(epochs, loss_values, 'b', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# 对未来100天进行预测
future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=100)
future_dates = pd.DataFrame(future_dates, columns=['Date'])
future_dates['Date'] = pd.to_datetime(future_dates['Date'])
future_dates = future_dates.set_index('Date')

# 构建预测数据集
X_future = scaled_flow_data[-n_steps:].reshape(1, n_steps, 1)
predictions = []
for i in range(100):
    y_future = model.predict(X_future)
    predictions.append(y_future[0][0])
    X_future = np.append(X_future[:, 1:, :], y_future.reshape(1, 1, 1), axis=1)

# 逆归一化
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 构建预测结果DataFrame
future_dates['Flow'] = predictions.astype(int)

# 打印预测结果
print(future_dates)