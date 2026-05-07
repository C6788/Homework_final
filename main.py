import os
import sys

# 导入我们之前编写的四个模块中的核心函数
from data_processing import load_and_process_data
from visualization import analyze_and_visualize
from prediction_model import build_and_train_models
from m4_qa_system import run_qa_system

def main():
    print("="*60)
    print("🚕 城市出租车出行数据分析与智能问答系统 - 正在启动")
    print("="*60)

    # 要求使用相对路径
    data_path = "data/yellow_tripdata_2023-01.parquet"

    # 友好提示：检查数据文件是否放置正确
    if not os.path.exists(data_path):
        print(f"\n❌ 严重错误: 找不到数据文件 '{data_path}'")
        print("💡 请确保:")
        print("1. 你在项目根目录下创建了 'data' 文件夹。")
        print("2. 已经将下载的 'yellow_tripdata_2023-01.parquet' 文件放进去了。")
        sys.exit(1)

    try:
        # ==========================================
        # 阶段 1: 数据处理 (M1)
        # ==========================================
        print("\n>>> [1/4] 正在执行 M1: 数据加载与清洗...")
        clean_df = load_and_process_data(data_path)

        # ==========================================
        # 阶段 2: 分析与可视化 (M2)
        # ==========================================
        print("\n>>> [2/4] 正在执行 M2: 数据分析与可视化图表生成...")
        analyze_and_visualize(clean_df)

        # ==========================================
        # 阶段 3: 预测模型 (M3)
        # ==========================================
        print("\n>>> [3/4] 正在执行 M3: 训练深度学习与机器学习模型...")
        # 注意：这里的模型训练可能需要一点时间，请耐心等待
        nn_model, rf_model, scaler = build_and_train_models(clean_df)

        # ==========================================
        # 阶段 4: 问答接口 (M4)
        # ==========================================
        print("\n>>> [4/4] 正在执行 M4: 启动智能交互问答系统...")
        # 启动命令行问答循环，交由用户控制
        run_qa_system()

    except Exception as e:
        print(f"\n❌ 程序执行过程中发生未捕获的错误: {e}")
        print("💡 请检查相关的模块代码是否正确，或缺少必要的依赖包。")

if __name__ == "__main__":
    main()