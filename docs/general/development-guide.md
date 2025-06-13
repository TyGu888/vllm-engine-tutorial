# vLLM 开发指南

## 准备开发环境

### 1. Fork和克隆仓库

```bash
# 1. 在GitHub上fork vllm仓库
# 2. 克隆你的fork
git clone https://github.com/YOUR_USERNAME/vllm.git
cd vllm

# 3. 添加upstream remote
git remote add upstream https://github.com/vllm-project/vllm.git
```

### 2. 设置开发环境

```bash
# 创建虚拟环境
conda create -n vllm-dev python=3.9
conda activate vllm-dev

# 安装开发依赖
pip install -e .
pip install -r requirements-dev.txt

# 安装pre-commit hooks
pre-commit install
```

### 3. 验证环境

```bash
# 运行基础测试
pytest tests/test_basic.py -v

# 检查代码格式
yapf --diff --recursive vllm/
mypy vllm/
```

## 代码贡献流程

### 1. 选择要解决的问题

```bash
# 查看good first issue标签
# https://github.com/vllm-project/vllm/labels/good%20first%20issue

# 创建新分支
git checkout -b feature/my-awesome-feature
```

### 2. 开发和测试

```python
# 示例: 添加新的采样方法
# 文件: vllm/sampling_params.py

@dataclass
class SamplingParams:
    # 现有参数...
    my_new_sampling: bool = False
    my_new_param: float = 1.0
```

```python
# 文件: vllm/model_executor/layers/sampler.py

def _apply_my_new_sampling(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """实现新的采样算法"""
    # 新采样逻辑
    modified_logits = logits * my_new_param
    return modified_logits

# 在主采样函数中集成
def forward(self, logits: torch.Tensor, ...) -> SamplerOutput:
    # 现有代码...
    
    if sampling_params.my_new_sampling:
        logits = _apply_my_new_sampling(logits, sampling_metadata)
    
    # 继续现有流程...
```

### 3. 添加测试

```python
# 文件: tests/samplers/test_my_new_sampling.py

import pytest
import torch
from vllm import LLM, SamplingParams

class TestMyNewSampling:
    
    @pytest.fixture
    def llm(self):
        return LLM(model="facebook/opt-125m", max_model_len=32)
    
    def test_my_new_sampling_basic(self, llm):
        """测试基本功能"""
        sampling_params = SamplingParams(
            my_new_sampling=True,
            my_new_param=2.0,
            max_tokens=10
        )
        
        outputs = llm.generate(["Hello"], sampling_params)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].text) > 0
    
    def test_my_new_sampling_parameter_validation(self):
        """测试参数验证"""
        with pytest.raises(ValueError):
            SamplingParams(my_new_param=-1.0)  # 无效参数
```

### 4. 运行测试

```bash
# 运行特定测试
pytest tests/samplers/test_my_new_sampling.py -v

# 运行相关的回归测试
pytest tests/samplers/ -v

# 运行完整测试套件 (CI会运行)
pytest tests/ -x  # -x 表示遇到第一个失败就停止
```

## 核心组件开发

### 1. 调度器扩展

```python
# 示例: 添加基于优先级的调度策略
# 文件: vllm/core/scheduler.py

class PriorityScheduler(Scheduler):
    
    def _schedule_prefills(
        self, 
        budget: SchedulingBudget,
        curr_loras: Set[int],
        enable_chunking: bool = False
    ) -> SchedulerPrefillOutputs:
        """基于优先级的prefill调度"""
        
        # 按优先级排序 (高优先级优先)
        sorted_waiting = sorted(
            self.waiting, 
            key=lambda seq_group: seq_group.priority,
            reverse=True
        )
        
        scheduled_seq_groups = []
        ignored_seq_groups = []
        
        for seq_group in sorted_waiting:
            # 检查资源可用性
            num_new_tokens = self._get_num_new_tokens(...)
            
            if budget.can_schedule(
                num_new_tokens=num_new_tokens,
                num_new_seqs=seq_group.get_max_num_running_seqs()
            ):
                # 调度这个序列组
                self._allocate_and_set_running(seq_group)
                scheduled_seq_groups.append(...)
                budget.add_num_batched_tokens(...)
            else:
                ignored_seq_groups.append(seq_group)
                
        return SchedulerPrefillOutputs(
            seq_groups=scheduled_seq_groups,
            ignored_seq_groups=ignored_seq_groups
        )
```

### 2. 内存管理优化

```python
# 示例: 实现智能内存预分配
# 文件: vllm/core/block_manager.py

class SmartBlockAllocator:
    
    def __init__(self, ..., prefetch_ratio: float = 0.1):
        self.prefetch_ratio = prefetch_ratio
        self.usage_history = []
        
    def allocate_with_prediction(self, seq_group: SequenceGroup):
        """基于历史使用模式预分配内存"""
        
        # 分析历史使用模式
        predicted_length = self._predict_sequence_length(seq_group)
        
        # 预分配额外的blocks
        extra_blocks = int(predicted_length * self.prefetch_ratio)
        
        # 执行分配
        allocated_blocks = self._allocate_blocks(
            required_blocks + extra_blocks
        )
        
        return allocated_blocks
    
    def _predict_sequence_length(self, seq_group: SequenceGroup) -> int:
        """基于prompt特征预测序列长度"""
        prompt_length = len(seq_group.get_seqs()[0].get_prompt_token_ids())
        
        # 简单的启发式预测
        if "question" in seq_group.get_seqs()[0].prompt.lower():
            return prompt_length + 50  # 问答通常较短
        else:
            return prompt_length + 200  # 生成任务通常较长
```

### 3. Attention机制扩展

```python
# 示例: 实现稀疏attention
# 文件: vllm/attention/backends/sparse_attention.py

class SparseAttentionBackend(AttentionBackend):
    
    def __init__(self, sparsity_pattern: str = "local"):
        self.sparsity_pattern = sparsity_pattern
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """稀疏attention计算"""
        
        # 生成稀疏mask
        sparse_mask = self._generate_sparse_mask(
            query.shape, self.sparsity_pattern
        )
        
        # 合并attention mask
        combined_mask = attn_mask & sparse_mask
        
        # 执行稀疏attention
        return self._sparse_attention(query, key, value, combined_mask)
    
    def _generate_sparse_mask(self, shape, pattern):
        """生成稀疏attention模式"""
        if pattern == "local":
            return self._local_attention_mask(shape)
        elif pattern == "strided":
            return self._strided_attention_mask(shape)
        else:
            raise ValueError(f"Unknown sparsity pattern: {pattern}")
```

## 调试技巧

### 1. 日志和监控

```python
# 在关键位置添加日志
import logging
logger = logging.getLogger(__name__)

def critical_function(...):
    logger.debug(f"Processing sequence group: {seq_group.request_id}")
    logger.info(f"Allocated {num_blocks} blocks")
    
    if error_condition:
        logger.error(f"Failed to allocate memory: {error_details}")
```

```bash
# 设置详细日志级别
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1  # 函数调用跟踪
```

### 2. 性能分析

```python
# 使用PyTorch profiler
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # 你的代码
    output = model_executor.execute_model(request)

# 导出性能数据
prof.export_chrome_trace("trace.json")
```

### 3. 内存调试

```python
# 监控GPU内存使用
import torch

def log_memory_usage(stage: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"{stage}: GPU memory allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

# 在关键点调用
log_memory_usage("Before model loading")
model = load_model(...)
log_memory_usage("After model loading")
```

### 4. 单元测试调试

```python
# 使用pytest的调试功能
pytest tests/test_scheduler.py::TestScheduler::test_priority_scheduling -v -s --pdb

# 添加断点调试
import pdb; pdb.set_trace()
```

## 性能优化最佳实践

### 1. CUDA内核优化

```python
# 尽量使用融合内核
def fused_attention_kernel(...):
    """融合的attention + norm + activation"""
    # 减少内存传输，提高性能
    pass

# 避免频繁的CPU-GPU数据传输
# 坏例子:
for i in range(batch_size):
    result = gpu_function(cpu_data[i].cuda())
    cpu_results.append(result.cpu())

# 好例子:
batch_gpu_data = torch.stack(cpu_data).cuda()
batch_results = gpu_function(batch_gpu_data)
cpu_results = batch_results.cpu()
```

### 2. 内存管理优化

```python
# 及时释放不需要的张量
def process_batch(batch_data):
    intermediate = heavy_computation(batch_data)
    result = final_computation(intermediate)
    
    # 明确释放中间结果
    del intermediate
    torch.cuda.empty_cache()  # 谨慎使用
    
    return result
```

### 3. 批处理优化

```python
# 动态批处理大小
def get_optimal_batch_size(available_memory: int, sequence_length: int):
    """根据可用内存动态调整批处理大小"""
    memory_per_sequence = estimate_memory_usage(sequence_length)
    optimal_batch_size = available_memory // memory_per_sequence
    return max(1, min(optimal_batch_size, MAX_BATCH_SIZE))
```

## 贡献代码检查清单

### 提交前检查

- [ ] 代码通过所有测试: `pytest tests/ -v`
- [ ] 代码格式正确: `yapf --diff --recursive .`
- [ ] 类型检查通过: `mypy vllm/`
- [ ] 添加了必要的测试用例
- [ ] 更新了相关文档
- [ ] 添加了适当的日志记录

### Pull Request准备

```bash
# 1. 同步upstream最新代码
git fetch upstream
git rebase upstream/main

# 2. 确保提交信息清晰
git commit -m "feat: add sparse attention backend

- Implement local and strided sparsity patterns
- Add comprehensive test cases  
- Update documentation with usage examples"

# 3. 推送到你的fork
git push origin feature/sparse-attention

# 4. 创建Pull Request
```

### PR描述模板

```markdown
## 描述
简要说明这个PR解决了什么问题或添加了什么功能。

## 类型
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## 测试
- [ ] 添加了新的测试用例
- [ ] 所有现有测试仍然通过
- [ ] 手工测试了主要功能

## 检查清单
- [ ] 代码遵循项目的编码标准
- [ ] 自我审查了代码变更
- [ ] 代码有适当的注释
- [ ] 更新了相关文档
```

## 常见陷阱和解决方案

### 1. 内存泄漏

```python
# 问题: 循环引用导致内存泄漏
class Component:
    def __init__(self):
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        child.parent = self  # 循环引用!
        self.children.append(child)

# 解决: 使用weakref
import weakref

class Component:
    def __init__(self):
        self.children = []
        self._parent_ref = None
    
    @property
    def parent(self):
        return self._parent_ref() if self._parent_ref else None
    
    def add_child(self, child):
        child._parent_ref = weakref.ref(self)
        self.children.append(child)
```

### 2. 竞态条件

```python
# 问题: 多线程访问共享状态
class UnsafeCounter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1  # 不是原子操作!

# 解决: 使用锁
import threading

class SafeCounter:
    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self.count += 1
```

### 3. CUDA同步问题

```python
# 问题: 忘记同步CUDA操作
def wrong_timing():
    start = time.time()
    result = gpu_computation(data)
    end = time.time()  # 错误！GPU操作可能还没完成
    print(f"Time: {end - start}")

# 解决: 正确同步
def correct_timing():
    torch.cuda.synchronize()  # 确保之前的操作完成
    start = time.time()
    result = gpu_computation(data)
    torch.cuda.synchronize()  # 等待GPU操作完成
    end = time.time()
    print(f"Time: {end - start}")
```

---

通过遵循这个开发指南，你就能有效地参与vLLM的开发，贡献高质量的代码！记住：从小的改进开始，逐步深入核心功能。 