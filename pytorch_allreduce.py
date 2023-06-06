import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def create_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )

def all_reduce_example(rank, world_size):
    create_process_group(rank, world_size)

    # create a different tensor on each device
    if rank == 0:
        tensor = torch.tensor([1, 2, 3])
    elif rank == 1:
        tensor = torch.tensor([10, 20, 30])
    elif rank == 2:
        tensor = torch.tensor([4, 5, 6])

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


if __name__ == "__main__":
    device_count = 3
    mp.spawn(all_reduce_example, args=(device_count,), nprocs=device_count)