import requests
url = f"http://localhost:30000/v1/chat/completions"
data = {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "图片是什么?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/home/wanggangqiang/img-test/1280-720.png"
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}
response = requests.post(url, json=data)
print(response.text)

