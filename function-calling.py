import os
import json
import asyncio
import random
from ollama import AsyncClient


# 从本地文件中获取商品分类
def load_grocery_list(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return []
    with open(file_path, "r") as file:
        items = [line.strip() for line in file if line.strip()]
    return items

# 模拟从API获取指定获取商品价格和营养信息
# item表示想要查询的商品信息
async def fetch_price_and_nutrition(item):
    print(f"Fetching price and nutrition data for '{item}'...")
    await asyncio.sleep(0.1)  # 模拟网络延迟(0.1秒)
    return {
        "item": item, # 随机生成1-10美元之间的价格
        "price": f"${random.uniform(1, 10):.2f}",
        "calories": f"{random.randint(50, 500)} kcal",
        "fat": f"{random.randint(1, 20)} g",
        "protein": f"{random.randint(1, 30)} g",
    }


# 模拟从API获取指定类别的食谱信息
# category表示想要查询的食谱类别
async def fetch_recipe(category):
    print(f"Fetching a recipe for the '{category}' category...")
    await asyncio.sleep(0.1) # 模拟网络延迟(0.1秒)
    return {
        "category": category,
        "recipe": f"Delicious {category} dish",
        "ingredients": ["Ingredient 1", "Ingredient 2", "Ingredient 3"],
        "instructions": "Mix ingredients and cook.",
    }

"""
这段代码实现了一个完整的异步购物清单处理系统，主要功能包括：
加载购物清单、分类商品、获取商品价格营养信息以及获取食谱建议
"""
async def main():
    # 加载购物清单
    grocery_items = load_grocery_list("./data/grocery_list.txt")
    if not grocery_items:
        print("Grocery list is empty or file not found.")
        return

    # 创建异步客户端实例，用于与Ollama模型交互
    client = AsyncClient()

    # 定义工具函数供模型调用
    tools = [
        # 价格营养信息获取工具
        {
            "type": "function",
            "function": {
                "name": "fetch_price_and_nutrition",
                "description": "Fetch price and nutrition data for a grocery item",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {
                            "type": "string",
                            "description": "The name of the grocery item",
                        },
                    },
                    "required": ["item"],
                },
            },
        },
        # 食谱获取工具
        {
            "type": "function",
            "function": {
                "name": "fetch_recipe",
                "description": "Fetch a recipe based on a category",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "The category of food (e.g., Produce, Dairy)",
                        },
                    },
                    "required": ["category"],
                },
            },
        },
    ]

    # 使用模型给商品分类，构造分类提示词
    categorize_prompt = f"""
You are an assistant that categorizes grocery items.

**Instructions:**

- Return the result **only** as a valid JSON object.
- Do **not** include any explanations, greetings, or additional text.
- Use double quotes (`"`) for all strings.
- Ensure the JSON is properly formatted.
- The JSON should have categories as keys and lists of items as values.

**Example Format:**

{{
  "Produce": ["Apples", "Bananas"],
  "Dairy": ["Milk", "Cheese"]
}}

**Grocery Items:**

{', '.join(grocery_items)}
"""

    messages = [{"role": "user", "content": categorize_prompt}]
    
    response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,  )

    messages.append(response["message"])
    print(response["message"]["content"])

    assistant_message = response["message"]["content"]

    try:
        categorized_items = json.loads(assistant_message)
        print("Categorized items:")
        print(categorized_items)

    except json.JSONDecodeError:
        print("Failed to parse the model's response as JSON.")
        print("Model's response:")
        print(assistant_message)
        return
    # 模拟API调用外部函数获取商品价格和营养信息
    fetch_prompt = """
    For each item in the grocery list, use the 'fetch_price_and_nutrition' function to get its price and nutrition data.
    """

    messages.append({"role": "user", "content": fetch_prompt})

    response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,
    )
    messages.append(response["message"])
    # 检查tool_calls是否存在,如果存在说明模型要求调用外部工具（函数）。
    if response["message"].get("tool_calls"):
        print("Function calls made by the model:")
        # 定义一个字典，映射函数名(字符串)到实际的Python函数
        available_functions = {
            "fetch_price_and_nutrition": fetch_price_and_nutrition,
        }
        # 初始化列表用于存储所有工具调用的结果（商品价格和营养信息）
        item_details = []
        # 获取模型返回的所有工具调用请求（可能同时调用多个函数）
        # for函数遍历处理每个请求
        for tool_call in response["message"]["tool_calls"]:
            # 从工具调用请求中提取函数名（如 "fetch_price_and_nutrition"）
            function_name = tool_call["function"]["name"]
            # 从工具调用请求中提取函数参数（如 {"item": "Apples"}）
            arguments = tool_call["function"]["arguments"]
            # 根据函数名从字典中查找对应的Python函数
            function_to_call = available_functions.get(function_name)
            if function_to_call:
                # 执行对应的函数
                result = await function_to_call(**arguments)
                # 将结果返回给模型,message表示对话历史列表,用于记录所有消息
                messages.append(
                    {
                        "role": "tool",# 表示这条消息是工具的执行结果
                        "content": json.dumps(result),# 结果JSON化
                    }
                )
                item_details.append(result)

                print(item_details)
    else:
        print(
            "The model didn't make any function calls for fetching price and nutrition data."
        )
        return

    # 随机选择一个商品类别,构造提示词要求获取该类别食谱
    random_category = random.choice(list(categorized_items.keys()))
    recipe_prompt = f"""
    Fetch a recipe for the '{random_category}' category using the 'fetch_recipe' function.
    """
    messages.append({"role": "user", "content": recipe_prompt})

    response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,
    )

    messages.append(response["message"])
    if response["message"].get("tool_calls"):
        available_functions = {
            "fetch_recipe": fetch_recipe,
        }
        for tool_call in response["message"]["tool_calls"]:
            function_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            function_to_call = available_functions.get(function_name)
            if function_to_call:
                result = await function_to_call(**arguments)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )
    else:
        print("The model didn't make any function calls for fetching a recipe.")
        return

    final_response = await client.chat(
        model="llama3.2",
        messages=messages,
        tools=tools,
    )

    print("\nAssistant's Final Response:")
    print(final_response["message"]["content"])

asyncio.run(main())