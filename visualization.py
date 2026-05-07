import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def analyze_and_visualize(df):
    print("\n--- 开始执行 M2 分析与可视化 ---")

    # 确保 outputs 目录存在，如果不存在则自动创建
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # 设置统一的图表风格和字体大小
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # ---------------------------------------------------------
    # 1. 出行需求时间规律（工作日 vs 周末的分小时订单量）
    # ---------------------------------------------------------
    print("正在绘制：1. 出行需求时间规律图...")
    plt.figure(figsize=(12, 6))

    # 新增一个 'day_type' 字段用于区分工作日和周末
    df_plot = df.copy()
    df_plot['day_type'] = df_plot['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

    # 按小时和日期类型分组计算订单量
    hourly_trend = df_plot.groupby(['pickup_hour', 'day_type']).size().reset_index(name='trip_count')

    # 绘制折线图
    sns.lineplot(data=hourly_trend, x='pickup_hour', y='trip_count', hue='day_type', marker='o')
    plt.title('NYC Taxi Trip Demand by Hour: Weekday vs Weekend', fontsize=16)
    plt.xlabel('Hour of Day (0-23)')
    plt.ylabel('Total Number of Trips')
    plt.xticks(range(0, 24))

    # 保存图表
    plt.savefig(os.path.join(output_dir, '1_time_pattern.png'), bbox_inches='tight')
    plt.close()  # 关闭画布，释放内存

    # ---------------------------------------------------------
    # 2. 区域热度分析（TOP 10 上客区域及高峰时段热力图）
    # ---------------------------------------------------------
    print("正在绘制：2. TOP10区域热度分布图...")
    plt.figure(figsize=(12, 8))

    # 找出上客量最高的 Top 10 区域 ID
    top_10_locations = df['PULocationID'].value_counts().nlargest(10).index

    # 过滤出这 10 个区域的数据
    df_top10 = df[df['PULocationID'].isin(top_10_locations)]

    # 创建数据透视表，行是区域ID，列是小时，值是订单数量
    heatmap_data = df_top10.pivot_table(index='PULocationID',
                                        columns='pickup_hour',
                                        values='tpep_pickup_datetime',
                                        aggfunc='count',
                                        fill_value=0)

    # 绘制热力图 (使用 YlOrRd 颜色映射，颜色越深代表单量越多)
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=.5)
    plt.title('Heatmap of Top 10 Pickup Locations by Hour', fontsize=16)
    plt.xlabel('Hour of Day')
    plt.ylabel('Pickup Location ID')

    plt.savefig(os.path.join(output_dir, '2_regional_heatmap.png'), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 3. 车费影响因素分析（行程距离与车费的散点图）
    # ---------------------------------------------------------
    print("正在绘制：3. 行程距离与车费关系图...")
    plt.figure(figsize=(10, 6))

    # 策略：为了防止百万级数据导致图表过载（Overplotting），随机抽取 10,000 个样本进行绘图
    sample_df = df.sample(n=10000, random_state=42) if len(df) > 10000 else df

    # 绘制散点图，alpha=0.4 增加透明度，便于观察密集区域
    sns.scatterplot(data=sample_df, x='trip_distance', y='fare_amount', alpha=0.4, color='teal')
    plt.title('Fare Amount vs. Trip Distance (Sampled 10,000 trips)', fontsize=16)
    plt.xlabel('Trip Distance (miles)')
    plt.ylabel('Fare Amount ($)')

    plt.savefig(os.path.join(output_dir, '3_fare_distance_scatter.png'), bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 4. 自选洞察分析：不同付款方式的车费分布对比
    # ---------------------------------------------------------
    print("正在绘制：4. 支付方式与车费分布图...")
    plt.figure(figsize=(8, 6))

    # 绘制箱线图
    sns.boxplot(data=df, x='payment_type_label', y='fare_amount', palette='Set2')
    plt.title('Fare Distribution by Payment Type', fontsize=16)
    plt.xlabel('Payment Type')
    plt.ylabel('Fare Amount ($)')

    # 限制 y 轴范围，过滤掉极端的异常高价单，让箱体看起来更清晰
    plt.ylim(0, 80)

    plt.savefig(os.path.join(output_dir, '4_payment_insight.png'), bbox_inches='tight')
    plt.close()

    print(f"M2 可视化完成！所有图表已成功保存至当前目录下的 '{output_dir}/' 文件夹中。")


# --- 联调测试代码 ---
# 在 main.py 或实际执行的地方，你可以这样将 M1 和 M2 串联起来：
if __name__ == "__main__":
    # 假设你已经定义了 M1 的 load_and_process_data 函数
    # from data_processing import load_and_process_data

    file_path = "data/yellow_tripdata_2023-01.parquet"
    # 1. 运行 M1 获取清洗后的数据
    # clean_data = load_and_process_data(file_path)

    # 2. 运行 M2 进行可视化 (这里假设 clean_data 已经准备好)
    # analyze_and_visualize(clean_data)
    pass