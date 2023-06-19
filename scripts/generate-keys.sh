
mkdir -p multi-node-training-keys
ssh-keygen -t rsa -N "" -f ./multi-node-training-keys/id_rsa

setup_worker () {
    # copy SSH keys to worker node
    scp ./multi-node-training-keys/id_rsa.pub worker-$1:/home/ubuntu/.ssh/id_rsa.pub
    scp ./multi-node-training-keys/id_rsa worker-$1:/home/ubuntu/.ssh/id_rsa

    # add SSH keys to authorized_keys on worker node
    ssh worker-$1 'cat /home/ubuntu/.ssh/id_rsa.pub >> /home/ubuntu/.ssh/authorized_keys'

    # install requirements on worker node
    ssh worker-$1 'rm -rf distributed-training-and-deepspeed && git clone https://github.com/gnovack/distributed-training-and-deepspeed.git && pip install -r distributed-training-and-deepspeed/requirements.txt && distributed-training-and-deepspeed/scripts/worker-prereqs.sh'
}

update_ssh_config () {
    worker_ip_1=$(ssh -G worker-1 | awk '$1 == "hostname" { print $2 }')
    worker_ip_2=$(ssh -G worker-2 | awk '$1 == "hostname" { print $2 }')
    worker_ip_3=$(ssh -G worker-3 | awk '$1 == "hostname" { print $2 }')

    ssh worker-$1 'cat > /home/ubuntu/.ssh/config' << EOF
Host worker-1
    HostName $worker_ip_1
    StrictHostKeyChecking no

Host worker-2
    HostName $worker_ip_2
    StrictHostKeyChecking no

Host worker-3
    HostName $worker_ip_3
    StrictHostKeyChecking no
EOF
}

setup_worker 1 &
setup_worker 2 &
setup_worker 3 &
wait

update_ssh_config 1
update_ssh_config 2
update_ssh_config 3
