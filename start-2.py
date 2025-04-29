import ollama

# 获取本地所有 Ollama 支持的模型及其元信息（如 llama3、mistral、phi3 等）。
response = ollama.list()

# print(response)

# == Chat example ==
# res = ollama.chat(
#     model="llama3.2",
#     messages=[
#         {"role": "user", "content": "why is the sky blue?"},
#     ],
# )
# print(res["message"]["content"])


# res = ollama.chat(
#     model="llama3.2",
#     messages=[
#         {
#             "role": "user",
#             "content": "从来没去过海边但是害怕大海和海里的黑色巨物?",
#         },
#     ],
#     stream=True # 开启流式响应，每次返回一小段内容
# )
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)


# ==================================================================================
# ==== The Ollama Python library's API is designed around the Ollama REST API ====
# ==================================================================================

# == Generate example ==
# res = ollama.generate(
#     model="llama3.2",
#     prompt="why is the sky blue?",
# )

# # show
# print(ollama.show("llama3.2"))


# 创建模型配置脚本，可以用来构建新的模型行为
modelfile = """
FROM llama3.2
SYSTEM You are very smart assistant who knows everything about oceans. You are very succinct and informative.
PARAMETER temperature 0.1
"""

# 创建自定义模型
ollama.create(model="DarkSeed", modelfile=modelfile)
# 使用自定义模型生成内容
res = ollama.generate(model="DarkSeed", prompt="伊隆·马斯克的大儿子为什么要和他断绝关系?")
print(res["response"])


# # 删除自定义模型，释放磁盘空间
# ollama.delete("DarkSeed")