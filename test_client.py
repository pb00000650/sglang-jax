from openai import Client
import sys

def main():
    # 服务器配置需与启动参数匹配
    base_url = "http://localhost:30000/v1"  # 默认端口30000，可根据实际情况修改
    api_key = "dummy"  # SGL-JAX无需实际API密钥
    model = "Qwen/Qwen3-32B"  # 与服务器--model-path保持一致

    # 初始化客户端
    client = Client(api_key=api_key, base_url=base_url)

    # 固定问题：明天天气怎么样？
    user_question = "明天天气怎么样？"

    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个天气查询助手，会简洁准确地回答天气相关问题。"},
        {"role": "user", "content": user_question}
    ]

    try:
        # 发送流式请求（匹配服务器的单请求限制--max-running-requests=1）
        print(f"正在查询: {user_question}\n")
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # 与服务器随机种子配合使用
            max_tokens=200,   # 限制生成长度
            stream=True
        )

        # 接收并打印流式响应
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

    except Exception as e:
        print(f"\n请求错误: {str(e)}")

if __name__ == "__main__":
    main()
