import os
import re
from openai import OpenAI

# ==========================================
# 0. DeepSeek API 配置
# ==========================================
API_KEY = "功能已展示，api密钥为了安全不上传"
BASE_URL = "https://api.deepseek.com"  # DeepSeek 官方 API 地址

# 初始化客户端
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
except Exception as e:
    print("API 客户端初始化失败，请检查是否安装了 openai 库。")

# 定义系统提示词 (System Prompt)
SYSTEM_PROMPT = """
你是一个专业的纽约市出租车出行数据分析助手。
我们的系统目前专注于2023年1月的黄色出租车数据。
当用户询问系统不支持的规则外问题（如宽泛的交通建议、纽约天气、政策等）时，
请运用你的通用知识给予友好、专业的解答。
注意：不要伪造具体的百万级统计数字，明确告知用户你的回答基于通用常识。
保持回答简洁、专业、热情。
"""


# ==========================================
# 1. 定义 5 种支持的规则问题处理函数 (模拟调用 M1-M3)
# ==========================================
def handle_time_query():
    # 实际应用中，这里会接收 M2 的统计结果
    return "【数字结论】根据数据统计，工作日的晚高峰 (17:00-19:00) 是出行需求最大的时段。\n【图表路径】参考图表：outputs/1_time_pattern.png"


def handle_region_query():
    return "【数字结论】上客量排名前三的区域分别是：JFK Airport, Upper East Side South, Midtown Center。\n【图表路径】参考图表：outputs/2_regional_heatmap.png"


def handle_fare_query():
    return "【数字结论】车费与行程距离呈显著正相关，平均每英里车费约为 $3.5。\n【图表路径】参考图表：outputs/3_fare_distance_scatter.png"


def handle_payment_query():
    return "【数字结论】超过 75% 的乘客偏好使用信用卡支付，且信用卡支付的单均车费略高于现金支付。\n【图表路径】参考图表：outputs/4_payment_insight.png"


def handle_prediction_query(location_id, hour):
    # 实际应用中，这里会调用 M3 的 PyTorch 模型进行 predict
    mock_demand = 156  # 模拟预测结果
    return f"【模型预测】经神经网络模型预测，区域ID {location_id} 在 {hour}:00 的预计出行需求量为 {mock_demand} 单。\n【图表路径】模型Loss曲线：outputs/5_nn_loss_curve.png"


# ==========================================
# 2. 调用 DeepSeek 大模型的兜底函数
# ==========================================
def ask_deepseek(user_question):
    print("🧠 [系统提示] 未匹配到预设图表，正在请教 DeepSeek 大模型...")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return "【DeepSeek 智能回复】\n" + response.choices[0].message.content
    except Exception as e:
        return f"【API 报错】调用大模型失败，请检查 API Key 或网络连接。错误信息: {e}"


# ==========================================
# 3. 核心交互循环 (主干逻辑)
# ==========================================
def run_qa_system():
    print("=" * 50)
    print("🚕 欢迎使用纽约市出租车出行数据智能问答系统！")
    print("💡 您可以问我关于：时间规律、热门区域、车费估算、支付方式、或特定区域的需求预测。")
    print("💡 遇到我算不出的问题，我会寻求 DeepSeek 大脑的帮助哦！")
    print("输入 'quit' 或 'exit' 退出系统。")
    print("=" * 50)

    while True:
        user_input = input("\n🙋 用户提问: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 感谢使用")
            break

        if not user_input:
            continue

        print("\n🤖 系统分析中...")

        # 规则 1：时间规律查询
        if "预测" in user_input or "需求量" in user_input:
            # 用正则尝试提取用户输入的数字作为区域和小时，如果没有则给默认值
            numbers = re.findall(r'\d+', user_input)
            loc_id = numbers[0] if len(numbers) > 0 else "237"
            hour = numbers[1] if len(numbers) > 1 else "18"
            print(handle_prediction_query(loc_id, hour))

            # 规则 2：时间规律查询
        elif any(keyword in user_input for keyword in ["时间", "时段", "高峰", "什么时候"]):
            print(handle_time_query())

            # 规则 3：区域热度查询 (此时再匹配“区域”就不会和“预测”冲突了)
        elif any(keyword in user_input for keyword in ["区域", "地点", "哪里", "热点", "最多"]):
            print(handle_region_query())

            # 规则 4：车费相关查询
        elif any(keyword in user_input for keyword in ["车费", "多少钱", "价格", "贵"]):
            print(handle_fare_query())

            # 规则 5：支付方式查询
        elif any(keyword in user_input for keyword in ["支付", "付款", "现金", "信用卡"]):
            print(handle_payment_query())

            # 规则 6：兜底 - 调用大模型
        else:
            answer = ask_deepseek(user_input)
            print(answer)

# ==========================================
# 4. 一键运行入口
# ==========================================
if __name__ == "__main__":
    run_qa_system()
