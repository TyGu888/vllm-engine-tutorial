#!/usr/bin/env python3
"""
vLLM API服务器使用演示
展示如何启动API服务器和客户端调用
"""

import requests
import json
import time
import asyncio
from openai import OpenAI

def demo_openai_completions():
    """演示OpenAI Completions API"""
    print("=== OpenAI Completions API 演示 ===")
    
    # 创建客户端 (需要先启动vLLM服务器)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"  # vLLM不验证token,可以是任意值
    )
    
    try:
        # 单个completion请求
        completion = client.completions.create(
            model="microsoft/DialoGPT-medium",  # 使用启动服务器时的模型名
            prompt="The future of AI is",
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
        )
        
        print("单个请求结果:")
        print(f"Generated: {completion.choices[0].text}")
        print()
        
        # 多个prompts
        prompts = [
            "Once upon a time,",
            "In the year 2050,", 
            "The secret to happiness is",
        ]
        
        print("批量请求结果:")
        for i, prompt in enumerate(prompts):
            completion = client.completions.create(
                model="microsoft/DialoGPT-medium",
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
            )
            print(f"{i+1}. Prompt: {prompt}")
            print(f"   Generated: {completion.choices[0].text.strip()}")
        print()
            
    except Exception as e:
        print(f"请求失败: {e}")
        print("请确保vLLM服务器正在运行:")
        print("python -m vllm.entrypoints.openai.api_server --model microsoft/DialoGPT-medium --port 8000")

def demo_openai_chat():
    """演示OpenAI Chat API"""
    print("=== OpenAI Chat API 演示 ===")
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    try:
        # 聊天对话
        response = client.chat.completions.create(
            model="microsoft/DialoGPT-medium",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you?"},
            ],
            max_tokens=100,
            temperature=0.7,
        )
        
        print("聊天对话:")
        print(f"User: Hello! How are you?")
        print(f"Assistant: {response.choices[0].message.content}")
        print()
        
        # 多轮对话
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
        ]
        
        response = client.chat.completions.create(
            model="microsoft/DialoGPT-medium",
            messages=messages,
            max_tokens=150,
            temperature=0.5,
        )
        
        print("多轮对话:")
        print(f"User: What is machine learning?")
        print(f"Assistant: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"聊天请求失败: {e}")

def demo_streaming():
    """演示流式输出"""
    print("=== 流式输出演示 ===")
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    try:
        print("流式生成 (逐字输出):")
        print("Prompt: Write a short story about space exploration")
        print("Generated: ", end="", flush=True)
        
        stream = client.completions.create(
            model="microsoft/DialoGPT-medium",
            prompt="Write a short story about space exploration",
            max_tokens=200,
            temperature=0.8,
            stream=True,  # 启用流式输出
        )
        
        for chunk in stream:
            if chunk.choices[0].text:
                print(chunk.choices[0].text, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"流式请求失败: {e}")

def demo_raw_http():
    """演示原始HTTP请求"""
    print("=== 原始HTTP请求演示 ===")
    
    url = "http://localhost:8000/v1/completions"
    
    payload = {
        "model": "microsoft/DialoGPT-medium",
        "prompt": "The key to success is",
        "max_tokens": 80,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer token-abc123"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("原始HTTP请求成功:")
            print(f"Prompt: {payload['prompt']}")
            print(f"Generated: {result['choices'][0]['text']}")
            print(f"Status: {response.status_code}")
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"HTTP请求失败: {e}")
    
    print()

def demo_performance_test():
    """演示性能测试"""
    print("=== 性能测试演示 ===")
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123"
    )
    
    # 并发请求测试
    async def async_request(prompt, request_id):
        """异步请求函数"""
        try:
            # 注意: OpenAI客户端不直接支持async,这里演示概念
            # 实际应用中可能需要使用aiohttp等库
            completion = client.completions.create(
                model="microsoft/DialoGPT-medium",
                prompt=f"{prompt} (request {request_id})",
                max_tokens=50,
                temperature=0.7,
            )
            return completion.choices[0].text
        except Exception as e:
            return f"Error: {e}"
    
    try:
        # 顺序请求性能测试
        print("顺序请求性能测试:")
        prompts = [f"Question {i}: What is" for i in range(5)]
        
        start_time = time.time()
        results = []
        for prompt in prompts:
            completion = client.completions.create(
                model="microsoft/DialoGPT-medium",
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
            )
            results.append(completion.choices[0].text)
        
        sequential_time = time.time() - start_time
        print(f"处理 {len(prompts)} 个请求耗时: {sequential_time:.2f}秒")
        print(f"平均每个请求: {sequential_time/len(prompts):.2f}秒")
        print()
        
    except Exception as e:
        print(f"性能测试失败: {e}")

def demo_model_info():
    """演示获取模型信息"""
    print("=== 模型信息演示 ===")
    
    try:
        # 获取可用模型列表
        response = requests.get("http://localhost:8000/v1/models")
        
        if response.status_code == 200:
            models = response.json()
            print("可用模型:")
            for model in models.get('data', []):
                print(f"- {model['id']}")
        else:
            print(f"获取模型列表失败: {response.status_code}")
        
        print()
        
    except Exception as e:
        print(f"获取模型信息失败: {e}")

def print_server_instructions():
    """打印启动服务器的说明"""
    print("=" * 60)
    print("vLLM API服务器演示")
    print("=" * 60)
    print()
    print("在运行此演示之前，请先启动vLLM服务器:")
    print()
    print("基本启动命令:")
    print("python -m vllm.entrypoints.openai.api_server \\")
    print("    --model microsoft/DialoGPT-medium \\")
    print("    --host 0.0.0.0 \\")
    print("    --port 8000")
    print()
    print("带更多参数的启动命令:")
    print("python -m vllm.entrypoints.openai.api_server \\")
    print("    --model microsoft/DialoGPT-medium \\")
    print("    --host 0.0.0.0 \\")
    print("    --port 8000 \\")
    print("    --gpu-memory-utilization 0.8 \\")
    print("    --max-model-len 1024")
    print()
    print("服务器启动后，访问 http://localhost:8000/docs 查看API文档")
    print("=" * 60)
    print()

def check_server_health():
    """检查服务器是否运行"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ vLLM服务器运行正常")
            return True
        else:
            print(f"❌ 服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ 无法连接到vLLM服务器 (http://localhost:8000)")
        return False

def main():
    """运行所有演示"""
    print_server_instructions()
    
    # 检查服务器状态
    if not check_server_health():
        print("\n请先按照上述说明启动vLLM服务器，然后重新运行此演示。")
        return
    
    print()
    
    try:
        # 运行各种演示
        demo_model_info()
        demo_openai_completions()
        demo_openai_chat()
        demo_streaming()
        demo_raw_http()
        demo_performance_test()
        
        print("🎉 所有API演示完成!")
        print()
        print("进一步探索:")
        print("1. 查看API文档: http://localhost:8000/docs")
        print("2. 尝试不同的模型参数")
        print("3. 测试更大的模型和批次")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")

if __name__ == "__main__":
    main() 