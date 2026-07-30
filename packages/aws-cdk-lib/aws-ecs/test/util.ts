import type { IConstruct } from 'constructs';
import * as autoscaling from '../../aws-autoscaling';
import * as ec2 from '../../aws-ec2';
import * as cdk from '../../core';
import * as ecs from '../lib';

/**
 * Acknowledge validation rules that fire on aws-ecs test infrastructure patterns.
 * Call on the stack (or any ancestor scope) to suppress known false positives.
 */
export function acknowledgeTestValidationRules(scope: IConstruct) {
  cdk.Validations.of(scope).acknowledge(
    { id: 'CloudFormation-Validate::W3697', reason: 'LaunchConfiguration used intentionally in tests' },
    { id: 'CloudFormation-Validate::W9013', reason: 'Duplicate subnets are intentional in tests' },
    { id: 'CloudFormation-Validate::W2001', reason: 'Unreferenced parameters are expected in test stacks' },
    { id: 'CloudFormation-Validate::W2531', reason: 'Hardcoded ARNs are expected in tests' },
    { id: 'CloudFormation-Validate::W9009', reason: 'Placeholder security group IDs used in tests' },
    { id: 'CloudFormation-Validate::W9002', reason: 'Deprecated properties used intentionally in tests' },
    { id: 'CloudFormation-Validate::W9007', reason: 'Deprecated runtimes used intentionally in tests' },
    { id: 'CloudFormation-Validate::E1150', reason: 'Hardcoded account IDs are expected in tests' },
    { id: 'CloudFormation-Validate::F3004', reason: 'Circular dependencies are intentional in some tests' },
  );
}

export function addDefaultCapacityProvider(cluster: ecs.Cluster,
  stack: cdk.Stack,
  vpc: ec2.Vpc,
  props?: Omit<ecs.AsgCapacityProviderProps, 'autoScalingGroup'>) {
  acknowledgeTestValidationRules(stack);
  const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'DefaultAutoScalingGroup', {
    vpc,
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    instanceType: new ec2.InstanceType('t2.micro'),
  });
  const provider = new ecs.AsgCapacityProvider(stack, 'AsgCapacityProvider', {
    ...props,
    autoScalingGroup,
  });
  cluster.addAsgCapacityProvider(provider);
  cluster.connections.addSecurityGroup(...autoScalingGroup.connections.securityGroups);
}
