import requests
import json

# 请求地址，ollama默认运行在本地的11434端口的API接口
url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2",
    "prompt": "tell me a short story and make it funny.",
}

response = requests.post(
    url, json=data, stream=True
)  # 发送Post请求，自动将python dict转为JSON，true表示用流式读取响应，避免一次性加载完整返回

# check the response status
if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    # 逐行读取服务器返回的文本
    for line in response.iter_lines():
        if line:
            # 字节转字符串
            decoded_line = line.decode("utf-8") 
            # 每行的JSON字符串反序列化为dict
            result = json.loads(decoded_line)
            # 拿到LLM当前生成的一段文本
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)