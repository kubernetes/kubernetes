import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as glue from '../lib';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-glue-connection');

const vpc = new ec2.Vpc(stack, 'Vpc', { restrictDefaultSecurityGroup: false });

const sg = new ec2.SecurityGroup(stack, 'SecurityGroup', {
  vpc,
});

// Network connection with an explicit subnet
new glue.Connection(stack, 'NetworkConnection', {
  type: glue.ConnectionType.NETWORK,
  subnet: vpc.privateSubnets[0],
  securityGroups: [sg],
});

// Network connection that lets the CDK select a subnet from the VPC
new glue.Connection(stack, 'NetworkConnectionFromVpc', {
  type: glue.ConnectionType.NETWORK,
  vpc,
  vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
  securityGroups: [sg],
});
