# vLLM 快速开始指南

## 环境搭建

### 1. 系统要求
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (V100, T4, RTX 20/30/40系列等)
- **CUDA**: 11.8 或 12.1+
- **Python**: 3.8+
- **内存**: 至少8GB GPU显存 (7B模型)

### 2. 安装vLLM

```bash
# 方法1: 从PyPI安装 (推荐)
pip install vllm

# 方法2: 从源码安装 (开发用)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 3. 验证安装

```python
import vllm
print(f"vLLM version: {vllm.__version__}")
```

## 第一个示例

### 1. 离线推理 (简单模式)

```python
from vllm import LLM, SamplingParams

# 创建LLM实例
llm = LLM(
    model="microsoft/DialoGPT-medium",  # 小模型，适合测试
    gpu_memory_utilization=0.7,        # 使用70%显存
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,    # 控制随机性
    top_p=0.95,        # nucleus sampling
    max_tokens=128,    # 最大生成长度
)

# 批量推理
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is",
]

# 生成文本
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
```

### 2. 在线服务 (API模式)

```bash
# 启动OpenAI兼容的API服务器
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/DialoGPT-medium \
    --host 0.0.0.0 \
    --port 8000
```

客户端调用:
```python
from openai import OpenAI

# 连接到vLLM服务器
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # 可以是任意值
)

# 调用API
completion = client.completions.create(
    model="microsoft/DialoGPT-medium",
    prompt="San Francisco is a",
    max_tokens=64,
    temperature=0.7
)

print(completion.choices[0].text)
```

## 进阶示例

### 1. 大模型推理

```python
from vllm import LLM, SamplingParams

# 使用更大的模型 (需要更多显存)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,      # 使用2个GPU
    gpu_memory_utilization=0.9,  # 使用90%显存
    max_model_len=4096,          # 最大序列长度
)

# 对话格式
prompt = """[INST] You are a helpful assistant. Answer the following question: 
What is the capital of Japan? [/INST]"""

sampling_params = SamplingParams(
    temperature=0.1,  # 降低随机性，更准确的回答
    max_tokens=256,
)

output = llm.generate([prompt], sampling_params)[0]
print(output.outputs[0].text)
```

### 2. 流式输出

```python
from vllm import LLM, SamplingParams

llm = LLM(model="microsoft/DialoGPT-medium")

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=128,
    stream=True  # 启用流式输出
)

prompt = "Once upon a time, in a distant galaxy"

# 流式生成
for output in llm.generate([prompt], sampling_params):
    for stream_output in output.outputs:
        if stream_output.text:
            print(stream_output.text, end='', flush=True)
```

### 3. 自定义采样参数

```python
from vllm import LLM, SamplingParams

llm = LLM(model="microsoft/DialoGPT-medium")

# 创造性写作设置
creative_params = SamplingParams(
    temperature=1.2,      # 高创造性
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    max_tokens=200,
)

# 准确问答设置
precise_params = SamplingParams(
    temperature=0.1,      # 低随机性
    top_p=0.95,
    max_tokens=100,
)

prompts = [
    "Write a creative story about a robot:",
    "What is 2+2?",
]

# 使用不同参数
creative_output = llm.generate([prompts[0]], creative_params)
precise_output = llm.generate([prompts[1]], precise_params)

print("Creative:", creative_output[0].outputs[0].text)
print("Precise:", precise_output[0].outputs[0].text)
```

## 性能优化技巧

### 1. 内存优化

```python
# 调整显存使用率
llm = LLM(
    model="your-model",
    gpu_memory_utilization=0.85,  # 根据显存大小调整
    swap_space=4,                 # CPU交换空间 (GB)
)
```

### 2. 批处理优化

```python
# 较大的批处理可以提高吞吐量
sampling_params = SamplingParams(max_tokens=100)

# 一次处理多个请求
large_batch = [f"Question {i}: " for i in range(32)]
outputs = llm.generate(large_batch, sampling_params)
```

### 3. 分布式推理

```bash
# 多GPU推理
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-hf \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

## 常见问题排查

### 1. 显存不足 (CUDA OOM)

**解决方案**:
```python
# 降低显存使用率
llm = LLM(model="your-model", gpu_memory_utilization=0.7)

# 或减少最大序列长度
llm = LLM(model="your-model", max_model_len=2048)
```

### 2. 模型加载失败

**检查**:
```bash
# 验证模型路径
ls /path/to/model/

# 检查Hugging Face token (私有模型)
export HF_TOKEN=your_token_here
```

### 3. 性能较慢

**优化**:
```python
# 启用CUDA图优化
llm = LLM(
    model="your-model", 
    enforce_eager=False,  # 启用CUDA图
)

# 或调整批处理大小
sampling_params = SamplingParams(
    max_tokens=64,  # 减少生成长度
)
```

## 下一步

1. 📖 阅读 [主教程](../README.md) 了解架构详情
2. 🔧 查看 [开发指南](development-guide.md) 学习贡献代码
3. 📊 运行 [性能测试](benchmarks.md) 评估效果
4. 🚀 部署到生产环境使用

---

**提示**: 如果遇到问题，请查看 [vLLM文档](https://docs.vllm.ai/) 或在GitHub提交issue。 