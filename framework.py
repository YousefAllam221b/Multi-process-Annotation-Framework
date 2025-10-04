import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from dataclasses import dataclass
from typing import  Tuple
from threading import BrokenBarrierError

@dataclass
class Config:
    """Configuration for the annotation framework"""
    dataset_size: int = 64
    batch_size: int = 32
    num_processes: int = 2
    n_annotation_rounds: int = 3
    m_sampling_rounds: int = 2
    buffer_size: int = 500
    overwrite_policy: str = "age"  # "age" or "annotation"
    device: str = "cpu"

class DummyDataset(Dataset):
    """Simple dataset for demonstration"""
    def __init__(self, size: int):
        self.data = torch.randn(size, 10)
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], idx

# Works
class DummyModel:
    """Dummy model that assigns random annotations"""
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def annotate(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.rand(batch.size(0))

# Todo: consider combining the annotations with the data in the dataset
class CentralBuffer:
    """Process-safe central buffer with prioritized sampling using Manager"""
    def __init__(self, manager, max_size: int, overwrite_policy: str = "age"):
        self.max_size = max_size
        self.overwrite_policy = overwrite_policy
        # Use Manager lists and lock
        self.buffer = manager.list()
        self.annotations = manager.list()
        self.timestamps = manager.list()
        self.lock = manager.Lock()
        self.current_time = manager.Value('i', 0)
        
    # Works
    # def add(self, data: torch.Tensor, annotations: torch.Tensor):
    #     """Add annotated data to buffer"""
    #     batch = [(data[i].numpy(), annotations[i].item()) for i in range(data.size(0))]

    #     with self.lock:
    #         # Todo: should i be adding one by one or in batch?
    #         for item, annotation in batch:
    #             if len(self.buffer) < self.max_size:
    #                 # Buffer not full, just append
    #                 self.buffer.append(item)
    #                 self.annotations.append(annotation)
    #                 self.timestamps.append(self.current_time.value)
    #             else:
    #                 # Buffer full, need to overwrite
    #                 if self.overwrite_policy == "age":
    #                     # Overwrite oldest item
    #                     idx = int(np.argmin(list(self.timestamps)))
    #                 else:  # annotation
    #                     # Overwrite item with lowest annotation
    #                     idx = int(np.argmin(list(self.annotations)))
                    
    #                 self.buffer[idx] = item
    #                 self.annotations[idx] = annotation
    #                 self.timestamps[idx] = self.current_time.value
                
    #             self.current_time.value += 1
    
    def add(self, data: torch.Tensor, annotations: torch.Tensor):
        """Add annotated data to buffer in batch"""
        with self.lock:
            remaining_space = self.max_size - len(self.buffer)
            if remaining_space > 0:
                # Add what fits
                n_add = min(remaining_space, data.size(0))
                self.buffer.extend([d.numpy() for d in data[:n_add]])
                self.annotations.extend([a.item() for a in annotations[:n_add]])
                self.timestamps.extend([self.current_time.value + i for i in range(n_add)])
                self.current_time.value += n_add

            # Handle if more items remain
            n_overflow = data.size(0) - remaining_space
            if n_overflow > 0:
                # Pick indices to overwrite
                if self.overwrite_policy == "age":
                    overwrite_idxs = np.argsort(list(self.timestamps))[:n_overflow]
                else: # annotation
                    overwrite_idxs = np.argsort(list(self.annotations))[:n_overflow]
                
                for i, idx in enumerate(overwrite_idxs):
                    self.buffer[idx] = data[-n_overflow + i].numpy()
                    self.annotations[idx] = annotations[-n_overflow + i].item()
                    self.timestamps[idx] = self.current_time.value
                    self.current_time.value += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from buffer based on annotation probability"""
        with self.lock:
            if len(self.buffer) == 0:
                return None, None
            
            # Todo: should i assign probabilities based on annotations like this?
            # Convert annotations to probabilities
            annotations_array = np.array(list(self.annotations))
            
            # Small epsilon is to avoid zero probabilities
            min_prob = 1e-6
            probs = annotations_array + min_prob
            probs = probs / probs.sum()

            # Sample indices based on probabilities
            n_samples = min(batch_size, len(self.buffer))
            indices = np.random.choice(len(self.buffer), size=n_samples, p=probs, replace=False)
            
            # Convert back to torch tensors
            sampled_data = torch.stack([torch.from_numpy(np.array(self.buffer[i])) for i in indices])
            sampled_annotations = torch.tensor([self.annotations[i] for i in indices])
            
            return sampled_data, sampled_annotations
    
    def size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)


def worker_process(process_id: int, config: Config, batch_queue: mp.Queue, buffer: CentralBuffer, 
                   barrier: mp.Barrier, stats_queue: mp.Queue): # type: ignore
    """Worker process for annotation and sampling"""
    # Create model for process
    model = DummyModel(device=config.device)
    
    process_stats = {
        'process_id': process_id,
        'annotation_time': 0,
        'sampling_time': 0,
        'total_time': 0
    }
    
    start_time = time.time()

    processed_batches = 0
    finished = False
    
    # Repeat until dataset is processed (all batches consumed)
    while True:
        # Phase 1: 
        batch = None

        # Get next batch from shared queue
        try:
            batch = batch_queue.get(timeout=0.1)
        except:
            print(f"Process {process_id} - no more batches available")
            batch = None

        if batch is None:
            finished = True

        # Annotation (repeat n times)
        if not finished:
            for annotation_round in range(config.n_annotation_rounds):
                
                ann_start = time.time()
                
                # Annotate batch
                batch = batch.to(config.device)
                annotations = model.annotate(batch)
                
                # Add to central buffer
                buffer.add(batch.cpu(), annotations.cpu())
                
                process_stats['annotation_time'] += time.time() - ann_start

        # Synchronize all processes before sampling
        # I am assuming that all processes will need to reach this point before we can proceed to sampling
        # This is to ensure that all processes have annotated their batches before any process starts sampling
        # Because if one process starts sampling before others have annotated, then the buffer might be empty or not fully populated
        # and the sampling will not be effective
        try:
            barrier.wait()
        except BrokenBarrierError:
            print(f"Process {process_id} - barrier broken after annotation")
        
        # Phase 2: Sampling (repeat m times)
        if not finished:
            for sampling_round in range(config.m_sampling_rounds):
                sampling_start = time.time()
                
                # Sample from buffer
                sampled_data, _ = buffer.sample(config.batch_size)
                
                if sampled_data is not None:
                    # In real scenario, would train model here
                    pass
                
                process_stats['sampling_time'] += time.time() - sampling_start
        
        if not finished:
            processed_batches += 1
        
        # Synchronize before next annotation round
        try:
            barrier.wait()
        except BrokenBarrierError:
            print(f"Process {process_id} - barrier broken after sampling")
    
        if batch_queue.empty():
            break
        
    print(f"Process {process_id} done. Processed {processed_batches} batches.")
    process_stats['total_time'] = time.time() - start_time
    stats_queue.put(process_stats)
    barrier.abort()


def run_framework(config: Config) -> dict:
    """Run the complete framework"""
    print(f"\n{'='*60}")
    print(f"Running with {config.num_processes} processes")
    print(f"{'='*60}")
    
    # Create manager for shared objects
    manager = mp.Manager()
    
    # Create shared buffer
    buffer = CentralBuffer(manager, config.buffer_size, config.overwrite_policy)
    
    # Create barrier for synchronization
    barrier = mp.Barrier(config.num_processes)
    
    # Queue for collecting statistics
    stats_queue = mp.Queue()
    
    # Create dataset
    dataset = DummyDataset(config.dataset_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create shared queue for batches
    batch_queue = mp.Queue()
    
    # Populate the batch queue with all batches
    print(f"Loading {len(dataloader)} batches into queue...")
    for batch, _ in dataloader:
        batch_queue.put(batch)
    
    # Add sentinel values to signal end of data
    for _ in range(len(dataloader) % config.num_processes):
        batch_queue.put(None)

    # Start worker processes
    processes = []
    start_time = time.time()
    
    for i in range(config.num_processes):
        p = mp.Process(target=worker_process, 
                      args=(i, config, batch_queue, buffer, barrier, stats_queue))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Collect statistics
    process_stats = []
    while not stats_queue.empty():
        process_stats.append(stats_queue.get())
    
    # Compute aggregate statistics
    avg_annotation_time = np.mean([s['annotation_time'] for s in process_stats])
    avg_sampling_time = np.mean([s['sampling_time'] for s in process_stats])
    
    results = {
        'num_processes': config.num_processes,
        'total_time': total_time,
        'avg_annotation_time': avg_annotation_time,
        'avg_sampling_time': avg_sampling_time,
    }
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg annotation time per process: {avg_annotation_time:.2f}s")
    print(f"Avg sampling time per process: {avg_sampling_time:.2f}s")
    
    return results


def benchmark_scaling():
    """Benchmark the framework with different numbers of processes"""
    print("\n" + "="*60)
    print("MULTI-PROCESS ANNOTATION FRAMEWORK BENCHMARK")
    print("="*60)
    
    base_config = Config(
        dataset_size=20000,
        batch_size=64,
        num_processes=1,
        n_annotation_rounds=5,
        m_sampling_rounds=3,
        buffer_size=500,
        overwrite_policy="age"
    )
    
    # process_counts = [1, 2, 4, 8]
    process_counts = [2, 4]
    results = []
    
    for num_processes in process_counts:
        config = Config(
            dataset_size=base_config.dataset_size,
            batch_size=base_config.batch_size,
            num_processes=num_processes,
            n_annotation_rounds=base_config.n_annotation_rounds,
            m_sampling_rounds=base_config.m_sampling_rounds,
            buffer_size=base_config.buffer_size,
            overwrite_policy=base_config.overwrite_policy
        )
        
        result = run_framework(config)
        results.append(result)
    
    # Print summary table
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Processes':<12} {'Total Time (s)':<18} ")
    print("-"*60)
    
    for r in results:
        print(f"{r['num_processes']:<12} {r['total_time']:<18.2f}")
    
    return results


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Run benchmark
    results = benchmark_scaling()