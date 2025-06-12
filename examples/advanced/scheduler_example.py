#!/usr/bin/env python3
"""
Scheduler 工作机制示例

本示例模拟vLLM Scheduler的核心工作流程，包括：
- 请求队列管理 (waiting, running, swapped)
- KV Cache内存分配和回收
- 抢占和swap机制
- Prefix caching优化
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running" 
    SWAPPED = "swapped"
    FINISHED = "finished"


class AllocStatus(Enum):
    OK = "ok"
    LATER = "later"
    NEVER = "never"


@dataclass
class Request:
    """模拟请求"""
    request_id: str
    prompt_tokens: List[int]
    status: RequestStatus = RequestStatus.WAITING
    allocated_blocks: List[int] = None
    generated_tokens: int = 0
    priority: int = 0
    
    def __post_init__(self):
        if self.allocated_blocks is None:
            self.allocated_blocks = []


@dataclass
class SchedulingBudget:
    """调度预算"""
    token_budget: int
    max_num_seqs: int
    current_tokens: int = 0
    current_seqs: int = 0
    
    def can_schedule(self, num_tokens: int, num_seqs: int) -> bool:
        return (self.current_tokens + num_tokens <= self.token_budget and
                self.current_seqs + num_seqs <= self.max_num_seqs)
    
    def add_request(self, num_tokens: int, num_seqs: int = 1):
        self.current_tokens += num_tokens
        self.current_seqs += num_seqs


class BlockManager:
    """模拟Block Manager"""
    
    def __init__(self, block_size: int = 16, num_gpu_blocks: int = 100, 
                 num_cpu_blocks: int = 200):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        
        self.free_gpu_blocks = list(range(num_gpu_blocks))
        self.free_cpu_blocks = list(range(num_cpu_blocks))
        self.allocated_gpu_blocks = set()
        self.allocated_cpu_blocks = set()
        
        # 模拟block映射表
        self.block_tables: Dict[str, List[int]] = {}
        self.swapped_blocks: Dict[str, List[int]] = {}
        
        # Prefix cache
        self.prefix_cache: Dict[str, List[int]] = {}
        
    def get_required_blocks(self, num_tokens: int) -> int:
        """计算所需的block数量"""
        return (num_tokens + self.block_size - 1) // self.block_size
    
    def can_allocate(self, request: Request) -> AllocStatus:
        """检查是否可以分配内存"""
        required_blocks = self.get_required_blocks(len(request.prompt_tokens))
        
        # 检查prefix cache
        prefix_key = str(request.prompt_tokens[:10])  # 简化的prefix
        if prefix_key in self.prefix_cache:
            cached_blocks = len(self.prefix_cache[prefix_key])
            required_blocks = max(0, required_blocks - cached_blocks)
            print(f"🎯 Request {request.request_id}: Prefix cache hit! "
                  f"Saved {cached_blocks} blocks")
        
        if len(self.free_gpu_blocks) >= required_blocks:
            return AllocStatus.OK
        elif required_blocks <= self.num_gpu_blocks:
            return AllocStatus.LATER
        else:
            return AllocStatus.NEVER
    
    def allocate(self, request: Request) -> bool:
        """分配内存"""
        required_blocks = self.get_required_blocks(len(request.prompt_tokens))
        
        # 检查prefix cache
        prefix_key = str(request.prompt_tokens[:10])
        cached_blocks = []
        if prefix_key in self.prefix_cache:
            cached_blocks = self.prefix_cache[prefix_key].copy()
            required_blocks = max(0, required_blocks - len(cached_blocks))
        
        if len(self.free_gpu_blocks) < required_blocks:
            return False
        
        # 分配新blocks
        new_blocks = []
        for _ in range(required_blocks):
            if self.free_gpu_blocks:
                block_id = self.free_gpu_blocks.pop(0)
                new_blocks.append(block_id)
                self.allocated_gpu_blocks.add(block_id)
        
        # 组合cached blocks和新blocks
        all_blocks = cached_blocks + new_blocks
        self.block_tables[request.request_id] = all_blocks
        request.allocated_blocks = all_blocks
        
        # 更新prefix cache
        if not cached_blocks and len(request.prompt_tokens) >= 10:
            self.prefix_cache[prefix_key] = new_blocks[:len(new_blocks)//2]
        
        print(f"📦 Allocated {len(all_blocks)} blocks for {request.request_id} "
              f"(cached: {len(cached_blocks)}, new: {len(new_blocks)})")
        return True
    
    def can_append(self, request: Request) -> bool:
        """检查是否可以继续生成"""
        # 简单检查：每个新token可能需要新block
        return len(self.free_gpu_blocks) > 0
    
    def append_slot(self, request: Request):
        """为新token分配slot"""
        if not self.can_append(request):
            return False
        
        # 可能需要新block
        if request.generated_tokens % self.block_size == 0:
            if self.free_gpu_blocks:
                new_block = self.free_gpu_blocks.pop(0)
                self.block_tables[request.request_id].append(new_block)
                self.allocated_gpu_blocks.add(new_block)
                print(f"📦 Added new block {new_block} for {request.request_id}")
        
        request.generated_tokens += 1
        return True
    
    def swap_out(self, request: Request) -> bool:
        """Swap out到CPU"""
        if request.request_id not in self.block_tables:
            return False
        
        gpu_blocks = self.block_tables[request.request_id]
        
        # 检查CPU空间
        if len(self.free_cpu_blocks) < len(gpu_blocks):
            return False
        
        # 分配CPU blocks
        cpu_blocks = []
        for _ in range(len(gpu_blocks)):
            cpu_block = self.free_cpu_blocks.pop(0)
            cpu_blocks.append(cpu_block)
            self.allocated_cpu_blocks.add(cpu_block)
        
        # 释放GPU blocks
        for block in gpu_blocks:
            self.allocated_gpu_blocks.remove(block)
            self.free_gpu_blocks.append(block)
        
        # 更新映射
        self.swapped_blocks[request.request_id] = cpu_blocks
        del self.block_tables[request.request_id]
        
        print(f"💾 Swapped out {request.request_id}: "
              f"GPU blocks {gpu_blocks} → CPU blocks {cpu_blocks}")
        return True
    
    def swap_in(self, request: Request) -> bool:
        """Swap in到GPU"""
        if request.request_id not in self.swapped_blocks:
            return False
        
        cpu_blocks = self.swapped_blocks[request.request_id]
        
        # 检查GPU空间
        if len(self.free_gpu_blocks) < len(cpu_blocks):
            return False
        
        # 分配GPU blocks
        gpu_blocks = []
        for _ in range(len(cpu_blocks)):
            gpu_block = self.free_gpu_blocks.pop(0)
            gpu_blocks.append(gpu_block)
            self.allocated_gpu_blocks.add(gpu_block)
        
        # 释放CPU blocks
        for block in cpu_blocks:
            self.allocated_cpu_blocks.remove(block)
            self.free_cpu_blocks.append(block)
        
        # 更新映射
        self.block_tables[request.request_id] = gpu_blocks
        del self.swapped_blocks[request.request_id]
        
        print(f"📤 Swapped in {request.request_id}: "
              f"CPU blocks {cpu_blocks} → GPU blocks {gpu_blocks}")
        return True
    
    def free(self, request: Request):
        """释放内存"""
        if request.request_id in self.block_tables:
            blocks = self.block_tables[request.request_id]
            for block in blocks:
                self.allocated_gpu_blocks.remove(block)
                self.free_gpu_blocks.append(block)
            del self.block_tables[request.request_id]
            print(f"🗑️  Freed {len(blocks)} GPU blocks for {request.request_id}")
        
        if request.request_id in self.swapped_blocks:
            blocks = self.swapped_blocks[request.request_id]
            for block in blocks:
                self.allocated_cpu_blocks.remove(block)
                self.free_cpu_blocks.append(block)
            del self.swapped_blocks[request.request_id]
            print(f"🗑️  Freed {len(blocks)} CPU blocks for {request.request_id}")
    
    def get_memory_stats(self) -> Dict:
        """获取内存统计"""
        return {
            "free_gpu_blocks": len(self.free_gpu_blocks),
            "allocated_gpu_blocks": len(self.allocated_gpu_blocks),
            "free_cpu_blocks": len(self.free_cpu_blocks),
            "allocated_cpu_blocks": len(self.allocated_cpu_blocks),
            "gpu_usage": len(self.allocated_gpu_blocks) / self.num_gpu_blocks,
            "cpu_usage": len(self.allocated_cpu_blocks) / self.num_cpu_blocks,
        }


class SimpleScheduler:
    """简化的Scheduler实现"""
    
    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        
        # 三个队列
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.swapped: List[Request] = []
        
        # 调度配置
        self.max_num_batched_tokens = 512
        self.max_num_seqs = 8
        
    def add_request(self, request: Request):
        """添加新请求"""
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        print(f"➕ Added request {request.request_id} to waiting queue")
    
    def schedule(self) -> Dict:
        """主调度逻辑"""
        print("\n" + "="*60)
        print("🔄 Scheduler 开始调度")
        
        budget = SchedulingBudget(
            token_budget=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs
        )
        
        scheduled_requests = []
        blocks_to_swap_in = []
        blocks_to_swap_out = []
        preempted_requests = []
        
        # 1. 调度running队列
        print("\n1️⃣ 调度Running队列")
        running_to_remove = []
        for request in self.running:
            if request.generated_tokens >= 20:  # 模拟完成条件
                request.status = RequestStatus.FINISHED
                running_to_remove.append(request)
                self.block_manager.free(request)
                print(f"✅ Request {request.request_id} finished")
                continue
            
            if not self.block_manager.can_append(request):
                # 需要抢占
                print(f"⚠️  Request {request.request_id} 需要抢占")
                if self.block_manager.swap_out(request):
                    request.status = RequestStatus.SWAPPED
                    self.swapped.append(request)
                    running_to_remove.append(request)
                    preempted_requests.append(request.request_id)
                    print(f"💾 Request {request.request_id} 被swap out")
                continue
            
            if budget.can_schedule(1, 0):  # decode需要1个token
                self.block_manager.append_slot(request)
                scheduled_requests.append(request)
                budget.add_request(1, 0)
                print(f"🎯 Request {request.request_id} 继续decode "
                      f"(generated: {request.generated_tokens})")
        
        # 移除已处理的请求
        for request in running_to_remove:
            self.running.remove(request)
        
        # 2. 调度swapped队列
        print("\n2️⃣ 调度Swapped队列") 
        swapped_to_remove = []
        for request in self.swapped:
            if budget.can_schedule(1, 1):
                if self.block_manager.swap_in(request):
                    request.status = RequestStatus.RUNNING
                    self.running.append(request)
                    swapped_to_remove.append(request)
                    scheduled_requests.append(request)
                    budget.add_request(1, 1)
                    blocks_to_swap_in.append(request.request_id)
                    print(f"📤 Request {request.request_id} 被swap in")
        
        for request in swapped_to_remove:
            self.swapped.remove(request)
        
        # 3. 调度waiting队列
        print("\n3️⃣ 调度Waiting队列")
        waiting_to_remove = []
        for request in self.waiting:
            prompt_tokens = len(request.prompt_tokens)
            
            if not budget.can_schedule(prompt_tokens, 1):
                print(f"⏸️  Request {request.request_id} 超出预算，稍后调度")
                continue
            
            alloc_status = self.block_manager.can_allocate(request)
            if alloc_status == AllocStatus.OK:
                if self.block_manager.allocate(request):
                    request.status = RequestStatus.RUNNING
                    self.running.append(request)
                    waiting_to_remove.append(request)
                    scheduled_requests.append(request)
                    budget.add_request(prompt_tokens, 1)
                    print(f"🚀 Request {request.request_id} 开始prefill")
            elif alloc_status == AllocStatus.LATER:
                print(f"⏳ Request {request.request_id} 内存不足，稍后重试")
            else:
                print(f"❌ Request {request.request_id} 永远无法分配")
        
        for request in waiting_to_remove:
            self.waiting.remove(request)
        
        # 构建调度结果
        result = {
            "scheduled_requests": [r.request_id for r in scheduled_requests],
            "num_scheduled_tokens": budget.current_tokens,
            "blocks_to_swap_in": blocks_to_swap_in,
            "blocks_to_swap_out": blocks_to_swap_out,
            "preempted_requests": preempted_requests,
            "queue_lengths": {
                "waiting": len(self.waiting),
                "running": len(self.running),
                "swapped": len(self.swapped)
            }
        }
        
        print(f"\n📊 调度结果: {result['num_scheduled_tokens']} tokens, "
              f"{len(scheduled_requests)} requests")
        print(f"📋 队列状态: Waiting({len(self.waiting)}), "
              f"Running({len(self.running)}), Swapped({len(self.swapped)})")
        
        return result
    
    def get_status(self) -> Dict:
        """获取调度器状态"""
        memory_stats = self.block_manager.get_memory_stats()
        return {
            "queue_lengths": {
                "waiting": len(self.waiting),
                "running": len(self.running),
                "swapped": len(self.swapped)
            },
            "memory_stats": memory_stats
        }


def main():
    """主演示程序"""
    print("🎭 vLLM Scheduler 工作机制演示")
    print("=" * 60)
    
    # 创建组件
    block_manager = BlockManager(
        block_size=16,
        num_gpu_blocks=20,  # 故意设置小一点以演示抢占
        num_cpu_blocks=40
    )
    
    scheduler = SimpleScheduler(block_manager)
    
    # 创建测试请求
    requests = [
        Request("req_1", list(range(50)), priority=1),   # 长请求
        Request("req_2", list(range(30)), priority=2),   # 中等请求
        Request("req_3", list(range(20)), priority=3),   # 短请求
        Request("req_4", list(range(60)), priority=1),   # 长请求
        Request("req_5", list(range(10)), priority=4),   # 很短请求
        Request("req_6", list(range(10)), priority=4),   # 相同prefix，测试缓存
    ]
    
    # 逐个添加请求
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        
        # 每添加一个请求就调度一次
        print(f"\n🔄 第{i+1}轮调度")
        schedule_result = scheduler.schedule()
        
        # 显示内存状态
        status = scheduler.get_status()
        memory = status["memory_stats"]
        print(f"💾 GPU使用率: {memory['gpu_usage']:.1%}, "
              f"空闲blocks: {memory['free_gpu_blocks']}")
        
        time.sleep(0.5)  # 模拟处理延迟
    
    # 继续调度直到所有请求完成
    print("\n" + "="*60)
    print("🔄 继续调度直到所有请求完成...")
    
    for step in range(10):
        if (len(scheduler.waiting) == 0 and 
            len(scheduler.running) == 0 and 
            len(scheduler.swapped) == 0):
            print("✅ 所有请求已完成!")
            break
        
        print(f"\n🔄 第{step+7}轮调度")
        schedule_result = scheduler.schedule()
        
        status = scheduler.get_status()
        memory = status["memory_stats"]
        print(f"💾 GPU使用率: {memory['gpu_usage']:.1%}")
        
        time.sleep(0.3)
    
    # 最终统计
    print("\n" + "="*60)
    print("📈 最终统计")
    final_memory = block_manager.get_memory_stats()
    print(f"GPU blocks: {final_memory['free_gpu_blocks']}/{block_manager.num_gpu_blocks} 空闲")
    print(f"CPU blocks: {final_memory['free_cpu_blocks']}/{block_manager.num_cpu_blocks} 空闲")
    print(f"Prefix cache 大小: {len(block_manager.prefix_cache)}")
    
    print("\n🎯 演示要点:")
    print("1. ✅ 请求队列管理: waiting → running → finished")
    print("2. ✅ 内存分配: 动态分配GPU blocks给新请求")
    print("3. ✅ 抢占机制: GPU内存不足时swap out到CPU")
    print("4. ✅ Swap恢复: GPU空间释放后swap in回来")
    print("5. ✅ Prefix缓存: 相同前缀的请求共享blocks")
    print("6. ✅ 资源预算: 限制每轮调度的token数和请求数")


if __name__ == "__main__":
    main() 