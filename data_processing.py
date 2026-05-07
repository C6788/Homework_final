import pandas as pd
import numpy as np


def load_and_process_data(file_path):
    print("正在加载数据，请稍候...")
    # 1. 加载数据
    df = pd.read_parquet(file_path)
    print(f"数据加载完成！原始数据共有 {df.shape[0]} 行，{df.shape[1]} 列。")

    # 2. 生成数据质量报告
    print("\n--- 数据质量报告 ---")
    # 统计缺失率
    missing_rate = df.isnull().mean() * 100
    print("\n1. 各字段缺失率 (%)：")
    print(missing_rate[missing_rate > 0].round(2))  # 只打印有缺失值的字段

    # 统计异常值（查看数值型特征的分布）
    print("\n2. 关键字段异常值统计 (描述性统计)：")
    print(df[['trip_distance', 'fare_amount', 'passenger_count']].describe().round(2))

    # 3. 清洗数据（注释中说明策略理由）
    print("\n--- 正在清洗数据 ---")
    initial_count = len(df)

    # 策略A：清洗时间异常的数据。理由：下车时间必须晚于上车时间。
    df = df[df['tpep_dropoff_datetime'] > df['tpep_pickup_datetime']]

    # 策略B：清洗距离异常的数据。理由：行程距离为0或负数通常是设备故障或取消的订单，不具备分析价值。
    df = df[df['trip_distance'] > 0]

    # 策略C：清洗车费异常的数据。理由：车费必须大于0，负数通常是退款或系统错误。
    df = df[df['fare_amount'] > 0]

    # 策略D：清洗乘客数异常的数据。理由：空车（乘客为0）的行程记录不符合正常打车逻辑（针对存在该字段的记录处理）。
    df = df[df['passenger_count'].fillna(1) > 0]

    # 策略E：清理不在1月份的数据。理由：数据集中可能混入因为系统时钟错误导致的非2023年1月的脏数据。
    df = df[(df['tpep_pickup_datetime'].dt.year == 2023) & (df['tpep_pickup_datetime'].dt.month == 1)]

    print(f"数据清洗完成！清洗掉了 {initial_count - len(df)} 条异常记录。当前剩余 {len(df)} 条记录。")

    # 4. 特征提取与衍生
    print("\n--- 正在提取特征 ---")
    # 提取时间特征
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0代表周一，6代表周日

    # 提取是否为高峰时段 (假设工作日 7-9点 和 17-19点 为高峰)
    is_weekday = df['day_of_week'] < 5
    is_morning_peak = df['pickup_hour'].isin([7, 8, 9])
    is_evening_peak = df['pickup_hour'].isin([17, 18, 19])
    df['is_peak_hour'] = np.where(is_weekday & (is_morning_peak | is_evening_peak), 1, 0)

    # 衍生特征 1：行程时长 (trip_duration_min)
    # 理由：时长是分析拥堵、车费的核心因素。
    df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0

    # 清理因计算时长产生的极端离谱数据（比如时长超过24小时的，或者不足1分钟的）
    df = df[(df['trip_duration_min'] >= 1) & (df['trip_duration_min'] <= 1440)]

    # 衍生特征 2：平均行驶速度 (average_speed_mph)
    # 理由：速度能很好地反映交通拥堵情况，对挖掘区域热度和出行规律很有帮助。
    df['average_speed_mph'] = df['trip_distance'] / (df['trip_duration_min'] / 60.0)

    # 清理速度异常（比如速度超过100英里/小时的，在纽约市区基本不可能）
    df = df[df['average_speed_mph'] <= 100]
    # 衍生特征 3：付款方式文本标签 (payment_type_label)
    # 理由：原始数据的 payment_type 是数字，映射成文字标签更直观，方便 M2 的可视化。
    payment_mapping = {
        1: 'Credit Card',  # 信用卡
        2: 'Cash',  # 现金
        3: 'No Charge',  # 免费
        4: 'Dispute',  # 争议
        5: 'Unknown',  # 未知
        6: 'Voided Trip'  # 作废
    }
    df['payment_type_label'] = df['payment_type'].map(payment_mapping)


    # 策略F：清洗付款方式异常的数据。
    # 理由：对于大多数需求分析和预测来说，'未知'或'作废'的订单是噪音数据。
    # 我们这里保守一点，只保留信用卡(1)和现金(2)这两种最主流的有效支付方式。
    df = df[df['payment_type'].isin([1, 2])]

    print("特征提取完成！")
    return df
#测试入口
if __name__ == "__main__":
    file_path = "data/yellow_tripdata_2023-01.parquet"
    try:
        clean_df = load_and_process_data(file_path)
        print("\n处理后的数据预览：")
        print(clean_df[['tpep_pickup_datetime', 'trip_distance', 'fare_amount', 'payment_type_label']].head())
    except Exception as e:
        print(f"哎呀，出错了：{e}")