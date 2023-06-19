from aws_cdk import (
    Stack,
    Environment,
    CfnOutput,
    aws_ec2 as ec2,
)
from constructs import Construct

class MultiNodeTrainingStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env: Environment, **kwargs) -> None:
        super().__init__(scope, construct_id, env=env)

        ssh_key_name = kwargs.get("ssh_key_name")

        training_instance_count = 3
        gpus_per_instance = 1
        training_instance_type = ec2.InstanceType("g4dn.xlarge")

        training_vpc = ec2.Vpc(self, "TrainingVpc", 
                               max_azs=1,
                               subnet_configuration=[ec2.SubnetConfiguration(name="TrainingSubnet", subnet_type=ec2.SubnetType.PUBLIC)])
        
        security_group = ec2.SecurityGroup(self, f"TrainingInstanceSecurityGroup", vpc=training_vpc)
        training_instance_dns_names = []
        for i in range(training_instance_count):
            training_instance: ec2.Instance = ec2.Instance(self, f"TrainingInstance-{i}", 
                                            vpc=training_vpc, 
                                            instance_type=training_instance_type,
                                            security_group=security_group,
                                            machine_image=ec2.MachineImage.lookup(name="Deep Learning AMI GPU PyTorch 2.0.*"),
                                            key_name=ssh_key_name)
            training_instance_dns_names.append(training_instance.instance_public_dns_name)
            
            # allow SSH access
            training_instance.connections.allow_from_any_ipv4(ec2.Port.tcp(22), "Allow SSH")
            training_instance.connections.allow_internally(ec2.Port.all_traffic(), "Allow all traffic within VPC")

        
        hostfile_content = "\n" + "\n".join([f"{name} slots={gpus_per_instance}" for name in training_instance_dns_names])
        CfnOutput(self, "Hostfile Content", value=hostfile_content) 


        ssh_config_content = "\n" + "\n".join([f"""Host worker-{i+1}
    HostName {training_instance_dns_names[i]}
    User ubuntu
    IdentityFile {{YOUR_SSH_KEY_FILE_PATH}}
    StrictHostKeyChecking no""" for i in range(training_instance_count)])
        CfnOutput(self, "SSH Config Content", value=ssh_config_content)

