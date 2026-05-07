import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------
# 1. 定义 PyTorch 神经网络模型
# ---------------------------------------------------------
class DemandPredictorNN(nn.Module):
    def __init__(self, input_dim):
        super(DemandPredictorNN, self).__init__()
        # 定义一个简单的三层全连接神经网络
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出层，预测一个连续的数值（需求量）
        )

    def forward(self, x):
        return self.net(x)


def build_and_train_models(df):
    print("\n--- 开始执行 M3 预测模型 ---")
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 2. 数据聚合（将订单级数据转换为需求量级数据）
    # ---------------------------------------------------------
    import pandas as pd
    import numpy as np

    print("正在聚合数据，生成需求量标签 (Demand)...")

    # 1. 创建完整的时间-空间网格 (所有日期 x 所有小时 x 所有区域)
    unique_dates = df['tpep_pickup_datetime'].dt.date.unique()
    unique_hours = range(24)
    unique_locations = df['PULocationID'].unique()

    # 使用 pd.MultiIndex.from_product 生成全排列组合
    full_idx = pd.MultiIndex.from_product(
        [unique_dates, unique_hours, unique_locations],
        names=['pickup_date', 'pickup_hour', 'PULocationID']
    ).to_frame(index=False)

    # 2. 按照老方法计算实际有订单的需求量
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    actual_demand = df.groupby(['pickup_date', 'pickup_hour', 'PULocationID']).size().reset_index(name='demand')

    # 3. 将全网格与实际需求量左连接 (Left Join)
    # 这样原本没有订单的时段就会出现 NaN
    agg_df = pd.merge(full_idx, actual_demand, on=['pickup_date', 'pickup_hour', 'PULocationID'], how='left')

    # 4. 把 NaN 填充为 0 (这一步就是灵魂所在！)
    agg_df['demand'] = agg_df['demand'].fillna(0)

    # 5. 重新生成特征列 (因为全网格里缺少星期几和是否高峰，需要补上)
    # 将 pickup_date 转为 datetime 以提取星期几
    agg_df['temp_datetime'] = pd.to_datetime(agg_df['pickup_date'])
    agg_df['day_of_week'] = agg_df['temp_datetime'].dt.dayofweek

    # 重新计算是否高峰
    is_weekday = agg_df['day_of_week'] < 5
    is_morning_peak = agg_df['pickup_hour'].isin([7, 8, 9])
    is_evening_peak = agg_df['pickup_hour'].isin([17, 18, 19])
    agg_df['is_peak_hour'] = np.where(is_weekday & (is_morning_peak | is_evening_peak), 1, 0)

    # 6. 提取 X 和 y
    features = ['PULocationID', 'pickup_hour', 'day_of_week', 'is_peak_hour']
    X = agg_df[features].values
    y = agg_df['demand'].values

    print(f"聚合完成！包含了 0 需求的完整数据集形状为: {agg_df.shape}")

    # ---------------------------------------------------------
    # 3. 数据预处理与 8:2 划分
    # ---------------------------------------------------------
    print("正在划分训练集和测试集 (8:2)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化 (神经网络对特征尺度很敏感，必须做标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # 4. PyTorch 神经网络训练与评估
    # ---------------------------------------------------------
    print("正在构建并训练 PyTorch 神经网络...")
    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 初始化模型、损失函数(MSE)和优化器(Adam)
    model = DemandPredictorNN(input_dim=X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 20
    train_losses = []

    # 训练循环
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            batch_losses.append(loss.item())

        # 记录每个 epoch 的平均 loss
        epoch_loss = np.mean(batch_losses)
        train_losses.append(epoch_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', label='Train Loss (MSE)')
    plt.title('Neural Network Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '5_nn_loss_curve.png'), bbox_inches='tight')
    plt.close()
    print(f"Loss 曲线已保存至 {output_dir}/5_nn_loss_curve.png")

    # 神经网络评估
    model.eval()
    with torch.no_grad():
        nn_predictions = model(X_test_tensor).numpy()

    nn_mae = mean_absolute_error(y_test, nn_predictions)
    nn_rmse = mean_squared_error(y_test, nn_predictions, squared=False)

    # ---------------------------------------------------------
    # 5. 随机森林对比实验
    # ---------------------------------------------------------
    print("正在训练 Random Forest 随机森林模型进行对比...")
    # 随机森林不需要特征标准化，但为了公平对比输入特征，直接使用未缩放或缩放后的均可，这里使用缩放前的 X_train
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)

    # ---------------------------------------------------------
    # 6. 输出对比报告
    # ---------------------------------------------------------
    print("\n================ 预测模型对比报告 ================")
    print(f"神经网络 (PyTorch) - MAE: {nn_mae:.4f}, RMSE: {nn_rmse:.4f}")
    print(f"随机森林 (RF)      - MAE: {rf_mae:.4f}, RMSE: {rf_rmse:.4f}")
    print("==================================================")


    # 返回训练好的模型和 scaler，供 M4 问答模块调用
    return model, rf_model, scaler


# --- 联调测试代码 ---
if __name__ == "__main__":
    # 假设 clean_data 是 M1 返回的数据
    # nn_model, rf_model, scaler = build_and_train_models(clean_data)
    pass