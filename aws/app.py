#!/usr/bin/env python3

import aws_cdk as cdk
import os

from training_stack import TrainingStack


app = cdk.App()
TrainingStack(app, "TrainingStack",

    # AWS account and region to deploy to
    env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # SSH key name to use for EC2 instance
    ssh_key_name = os.getenv('EC2_SSH_KEY_NAME')
)

app.synth()
