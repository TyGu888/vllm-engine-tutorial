#!/usr/bin/env python3
"""
vLLM Engine Step 方法使用示例

这个示例展示了如何使用LLMEngine的step()方法进行推理，
帮助理解step的工作原理和使用模式。
"""

import time
from typing import List, Optional
from vllm import LLMEngine, EngineArgs, SamplingParams


def basic_step_example():
    """基础的step使用示例"""
    print("=== 基础Step使用示例 ===")
    
    # 初始化引擎
    engine_args = EngineArgs(
        model="facebook/opt-125m",  # 使用小模型便于测试
        max_num_seqs=4,
        max_model_len=512,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 添加请求
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
        stop=["\n"]
    )
    
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "The benefits of exercise include"
    ]
    
    # 添加所有请求
    for i, prompt in enumerate(prompts):
        engine.add_request(
            request_id=str(i),
            prompt=prompt,
            params=sampling_params
        )
    
    print(f"添加了 {len(prompts)} 个请求")
    
    # 主要的step循环
    step_count = 0
    completed_requests = 0
    
    while engine.has_unfinished_requests():
        step_count += 1
        print(f"\n--- Step {step_count} ---")
        
        # 执行一次step
        start_time = time.time()
        request_outputs = engine.step()
        step_time = time.time() - start_time
        
        print(f"Step执行时间: {step_time:.3f}s")
        print(f"返回了 {len(request_outputs)} 个输出")
        
        # 处理输出
        for output in request_outputs:
            if output.finished:
                completed_requests += 1
                print(f"请求 {output.request_id} 完成:")
                print(f"  输出: {output.outputs[0].text}")
                print(f"  总tokens: {len(output.outputs[0].token_ids)}")
            else:
                print(f"请求 {output.request_id} 继续中...")
                if output.outputs:
                    print(f"  当前输出: ...{output.outputs[0].text[-50:]}")
    
    print(f"\n总共执行了 {step_count} 次step")
    print(f"完成了 {completed_requests} 个请求")


def streaming_step_example():
    """流式输出的step使用示例"""
    print("\n=== 流式输出Step示例 ===")
    
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_num_seqs=2,
        max_model_len=256,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 添加一个长文本生成请求
    sampling_params = SamplingParams(
        temperature=0.9,
        max_tokens=100,
        stop=["<|endoftext|>"]
    )
    
    prompt = "Once upon a time, in a magical forest"
    engine.add_request(
        request_id="story",
        prompt=prompt,
        params=sampling_params
    )
    
    print(f"Prompt: {prompt}")
    print("Generated text: ", end="", flush=True)
    
    last_output_len = 0
    
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        
        for output in request_outputs:
            if output.request_id == "story":
                current_text = output.outputs[0].text
                # 只打印新生成的部分
                new_text = current_text[last_output_len:]
                print(new_text, end="", flush=True)
                last_output_len = len(current_text)
                
                if output.finished:
                    print(f"\n\n生成完成! 总长度: {len(output.outputs[0].token_ids)} tokens")
                    break


def batch_processing_example():
    """批处理的step使用示例"""
    print("\n=== 批处理Step示例 ===")
    
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_num_seqs=8,  # 支持更大的batch
        max_model_len=512,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 创建多个不同长度的请求
    requests = [
        ("short1", "Hello", SamplingParams(max_tokens=5)),
        ("short2", "Goodbye", SamplingParams(max_tokens=5)),
        ("medium1", "The quick brown fox", SamplingParams(max_tokens=20)),
        ("medium2", "In the beginning", SamplingParams(max_tokens=20)),
        ("long1", "Write a story about", SamplingParams(max_tokens=50)),
        ("long2", "Explain machine learning", SamplingParams(max_tokens=50)),
    ]
    
    # 分批添加请求（模拟动态到达）
    for i, (req_id, prompt, params) in enumerate(requests):
        engine.add_request(req_id, prompt, params)
        print(f"添加请求 {req_id}: {prompt}")
        
        # 模拟请求间隔
        if i % 2 == 1:
            # 执行几次step来处理之前的请求
            for _ in range(2):
                if engine.has_unfinished_requests():
                    outputs = engine.step()
                    for output in outputs:
                        if output.finished:
                            print(f"  --> {output.request_id} 完成")
    
    # 处理剩余请求
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for output in outputs:
            if output.finished:
                print(f"  --> {output.request_id} 完成: {output.outputs[0].text[:30]}...")


def performance_monitoring_example():
    """性能监控的step使用示例"""
    print("\n=== 性能监控Step示例 ===")
    
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_num_seqs=4,
        max_model_len=512,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 添加测试请求
    for i in range(4):
        engine.add_request(
            request_id=f"perf_test_{i}",
            prompt=f"Test prompt {i}: Generate some text",
            params=SamplingParams(max_tokens=30)
        )
    
    step_times = []
    scheduler_times = []
    execution_times = []
    
    class StepProfiler:
        def __init__(self):
            self.step_start = None
            self.schedule_time = 0
            self.execution_time = 0
    
    profiler = StepProfiler()
    
    while engine.has_unfinished_requests():
        step_start = time.time()
        
        # 执行step并记录时间
        outputs = engine.step()
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        print(f"Step时间: {step_time:.3f}s, 输出: {len(outputs)}个")
        
        # 显示引擎内部状态
        if hasattr(engine, 'scheduler'):
            scheduler = engine.scheduler[0]  # virtual_engine=0
            print(f"  队列状态 - WAITING: {len(scheduler.waiting)}, "
                  f"RUNNING: {len(scheduler.running)}, "
                  f"SWAPPED: {len(scheduler.swapped)}")
    
    # 性能统计
    print(f"\n性能统计:")
    print(f"  平均step时间: {sum(step_times)/len(step_times):.3f}s")
    print(f"  最小step时间: {min(step_times):.3f}s")
    print(f"  最大step时间: {max(step_times):.3f}s")
    print(f"  总step数: {len(step_times)}")


def error_handling_example():
    """错误处理的step使用示例"""
    print("\n=== 错误处理Step示例 ===")
    
    engine_args = EngineArgs(
        model="facebook/opt-125m",
        max_num_seqs=2,
        max_model_len=512,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # 添加正常请求
    engine.add_request(
        request_id="normal",
        prompt="This is a normal request",
        params=SamplingParams(max_tokens=20)
    )
    
    # 添加可能导致问题的请求（超长）
    try:
        engine.add_request(
            request_id="problematic",
            prompt="x" * 1000,  # 可能过长
            params=SamplingParams(max_tokens=50)
        )
    except Exception as e:
        print(f"添加请求时出错: {e}")
    
    # 执行step并处理可能的错误
    while engine.has_unfinished_requests():
        try:
            outputs = engine.step()
            
            for output in outputs:
                if output.finished:
                    print(f"请求 {output.request_id} 成功完成")
                else:
                    print(f"请求 {output.request_id} 继续处理中...")
                    
        except Exception as e:
            print(f"Step执行出错: {e}")
            # 在实际应用中，这里应该进行适当的错误恢复
            break


if __name__ == "__main__":
    print("vLLM Engine Step 方法使用示例")
    print("================================")
    
    try:
        basic_step_example()
        streaming_step_example()
        batch_processing_example()
        performance_monitoring_example()
        error_handling_example()
        
        print("\n所有示例执行完成!")
        
    except Exception as e:
        print(f"示例执行出错: {e}")
        print("请确保已正确安装vLLM并有足够的GPU内存") 