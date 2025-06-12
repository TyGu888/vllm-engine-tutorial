#!/usr/bin/env python3
"""
vLLM 基本使用示例
演示如何使用vLLM进行离线推理
"""

from vllm import LLM, SamplingParams
import time

def example_1_basic_generation():
    """示例1: 基本文本生成"""
    print("=== 示例1: 基本文本生成 ===")
    
    # 创建LLM实例 (使用小模型便于测试)
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        gpu_memory_utilization=0.7,
        max_model_len=512,
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )
    
    # 单个prompt
    prompt = "The future of artificial intelligence is"
    outputs = llm.generate([prompt], sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()

def example_2_batch_generation():
    """示例2: 批量生成"""
    print("=== 示例2: 批量生成 ===")
    
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        gpu_memory_utilization=0.7,
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=64,
    )
    
    # 多个prompts批量处理
    prompts = [
        "Once upon a time,",
        "In a galaxy far far away,",
        "The secret to happiness is",
        "Technology will change the world by",
    ]
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    print(f"批量处理 {len(prompts)} 个请求耗时: {end_time - start_time:.2f}秒")
    print()
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 50)

def example_3_different_sampling():
    """示例3: 不同采样策略对比"""
    print("=== 示例3: 不同采样策略对比 ===")
    
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        gpu_memory_utilization=0.7,
    )
    
    prompt = "Write a short story about a robot:"
    
    # 创造性设置 (高随机性)
    creative_params = SamplingParams(
        temperature=1.2,
        top_p=0.9,
        max_tokens=100,
    )
    
    # 保守设置 (低随机性)
    conservative_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=100,
    )
    
    # 生成两个版本
    creative_output = llm.generate([prompt], creative_params)[0]
    conservative_output = llm.generate([prompt], conservative_params)[0]
    
    print("创造性输出 (temperature=1.2):")
    print(creative_output.outputs[0].text)
    print()
    
    print("保守输出 (temperature=0.3):")
    print(conservative_output.outputs[0].text)
    print()

def example_4_chat_format():
    """示例4: 对话格式处理"""
    print("=== 示例4: 对话格式处理 ===")
    
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        gpu_memory_utilization=0.7,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
    )
    
    # 模拟对话历史
    conversation_history = [
        "Human: Hello, how are you today?",
        "Assistant: I'm doing well, thank you for asking! How can I help you?",
        "Human: Can you explain what machine learning is?",
        "Assistant: "
    ]
    
    # 将对话历史合并为单个prompt
    chat_prompt = "\n".join(conversation_history)
    
    outputs = llm.generate([chat_prompt], sampling_params)
    
    print("对话上下文:")
    print(chat_prompt)
    print()
    print("AI回复:")
    print(outputs[0].outputs[0].text)
    print()

def example_5_performance_comparison():
    """示例5: 性能对比测试"""
    print("=== 示例5: 性能对比测试 ===")
    
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        gpu_memory_utilization=0.7,
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
    )
    
    # 测试不同批次大小的性能
    batch_sizes = [1, 4, 8, 16]
    base_prompt = "The weather today is"
    
    for batch_size in batch_sizes:
        prompts = [f"{base_prompt} (request {i})" for i in range(batch_size)]
        
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(outputs) / total_time
        
        print(f"批次大小: {batch_size:2d}, "
              f"总时间: {total_time:.2f}s, "
              f"吞吐量: {throughput:.2f} req/s")
    
    print()

def example_6_memory_monitoring():
    """示例6: 内存使用监控"""
    print("=== 示例6: 内存使用监控 ===")
    
    import torch
    
    def print_memory_usage(stage):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"{stage}: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
        else:
            print(f"{stage}: 未检测到CUDA设备")
    
    print_memory_usage("初始状态")
    
    # 创建LLM实例
    llm = LLM(
        model="microsoft/DialoGPT-medium",
        gpu_memory_utilization=0.7,
    )
    
    print_memory_usage("模型加载后")
    
    # 执行推理
    sampling_params = SamplingParams(max_tokens=64)
    outputs = llm.generate(["Hello world"], sampling_params)
    
    print_memory_usage("推理完成后")
    
    # 删除模型释放内存
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_memory_usage("清理后")
    print()

def main():
    """运行所有示例"""
    print("vLLM 基本使用示例集合")
    print("=" * 50)
    
    try:
        example_1_basic_generation()
        example_2_batch_generation()
        example_3_different_sampling()
        example_4_chat_format()
        example_5_performance_comparison()
        example_6_memory_monitoring()
        
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保:")
        print("1. 已正确安装vLLM")
        print("2. 有可用的GPU设备")
        print("3. 有足够的显存运行模型")

if __name__ == "__main__":
    main() 