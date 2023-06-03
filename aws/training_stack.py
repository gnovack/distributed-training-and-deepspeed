from aws_cdk import (
    Stack,
    Environment,
    CfnOutput,
    aws_ec2 as ec2,
)
from constructs import Construct

class TrainingStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env: Environment, **kwargs) -> None:
        super().__init__(scope, construct_id, env=env)

        ssh_key_name = kwargs.get("ssh_key_name")

        training_vpc = ec2.Vpc(self, "TrainingVpc", 
                               max_azs=1,
                               subnet_configuration=[ec2.SubnetConfiguration(name="TrainingSubnet", subnet_type=ec2.SubnetType.PUBLIC)])
        training_instance: ec2.Instance = ec2.Instance(self, "TrainingInstance", 
                                         vpc=training_vpc, 
                                         instance_type=ec2.InstanceType("g4dn.12xlarge"),
                                         machine_image=ec2.MachineImage.lookup(name="Deep Learning AMI GPU PyTorch 2.0.*"),
                                         key_name=ssh_key_name)
        
        # allow SSH access
        training_instance.connections.allow_from_any_ipv4(ec2.Port.tcp(22), "Allow SSH")

        ssh_command = "ssh -i {}.pem ubuntu@{}".format(ssh_key_name, training_instance.instance_public_dns_name)
        CfnOutput(self, "Connect Command", value=ssh_command) 
