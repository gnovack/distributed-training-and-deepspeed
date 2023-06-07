import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from util import get_device_count

def create_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

def profile_collective_operations(rank, world_size):
    create_process_group(rank, world_size)

    # with torch.profiler.profile() as p:
    #     tensor = torch.rand(10_000_000).to(rank)
    #     dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # if rank == 0:
    #     p.export_chrome_trace("all-reduce-10M.json")

    
    with torch.profiler.profile() as p:
        reduced_t1 = torch.zeros(10_000_000).to(rank)
        reduced_t2 = torch.zeros(10_000_000).to(rank)
        
        if rank == 0:
            t1 = torch.ones(10_000_000).to(rank) #* (rank+1)
            dist.reduce_scatter(reduced_t1, [t1 for _ in range(world_size)], op=dist.ReduceOp.SUM)
        else:
            t2 = torch.rand(10_000_000).to(rank) #* (rank+1)
            dist.reduce_scatter(reduced_t2, [t2 for _ in range(world_size)], op=dist.ReduceOp.SUM)

        print(f"Rank: {rank}")
        print(f"T1: {reduced_t1}")
        print(f"T2: {reduced_t2}")


        gathered_t1 = [torch.zeros(10_000_000).to(rank) for _ in range(world_size)]
        # gathered_t2 = [torch.zeros(10_000_000).to(rank) for _ in range(world_size)]
        dist.all_gather(gathered_t1, reduced_t1)

        print(f"Rank: {rank}; Gathered T1: {gathered_t1}")
        # print(len(gathered))

    if rank == 0:
        p.export_chrome_trace("reduce-scatter-all-gather.json")


if __name__ == "__main__":
    device_count = get_device_count()
    mp.spawn(profile_collective_operations, args=(device_count,), nprocs=device_count)