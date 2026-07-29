#!/usr/bin/env node
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as cdk from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-autoscaling-instance-refresh-update-policy');

const vpc = new ec2.Vpc(stack, 'VPC', {
  maxAzs: 2,
  restrictDefaultSecurityGroup: false,
});

new autoscaling.AutoScalingGroup(stack, 'ASG', {
  vpc,
  instanceType: new ec2.InstanceType('t2.micro'),
  machineImage: new ec2.AmazonLinuxImage({ generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2 }),
  updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    minHealthyPercentage: 90,
    maxHealthyPercentage: 100,
    instanceWarmup: cdk.Duration.seconds(300),
    skipMatching: true,
    checkpointPercentages: [50, 100],
    checkpointDelay: cdk.Duration.minutes(10),
  }),
});

new integ.IntegTest(app, 'AutoScalingInstanceRefreshUpdatePolicyTest', {
  testCases: [stack],
});

app.synth();
