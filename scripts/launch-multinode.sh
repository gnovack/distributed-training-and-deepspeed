# ssh into first worker node and launch training
MASTER_ADDR=$(ssh -G worker-1 | awk '$1 == "hostname" { print $2 }')

scp hostfile worker-1:/home/ubuntu/distributed-training-and-deepspeed/hostfile
ssh worker-1 'cd distributed-training-and-deepspeed && PATH="/home/ubuntu/.local/bin:$PATH" deepspeed --master_addr=$MASTER_ADDR --hostfile=./hostfile zero_dp_training.py --stage=2 --model_name=facebook/opt-125m'
