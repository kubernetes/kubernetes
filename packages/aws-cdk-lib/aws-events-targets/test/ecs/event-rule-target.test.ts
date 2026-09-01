
import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Match, Template } from '../../../assertions';
import * as autoscaling from '../../../aws-autoscaling';
import * as ec2 from '../../../aws-ec2';
import * as ecs from '../../../aws-ecs';
import * as events from '../../../aws-events';
import * as iam from '../../../aws-iam';
import * as sqs from '../../../aws-sqs';
import * as cdk from '../../../core';
import * as targets from '../../lib';
import { EcsTask } from '../../lib';

let stack: cdk.Stack;
let vpc: ec2.Vpc;
let cluster: ecs.Cluster;

beforeEach(() => {
  stack = new cdk.Stack();
  vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 1 });
  cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
  const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'DefaultAutoScalingGroup', {
    vpc,
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    instanceType: new ec2.InstanceType('t2.micro'),
  });
  const provider = new ecs.AsgCapacityProvider(stack, 'AsgCapacityProvider', { autoScalingGroup });
  cluster.addAsgCapacityProvider(provider);
});

test('Can use EC2 taskdef as EventRule target', () => {
  const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
    memoryLimitMiB: 256,
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: { Ref: 'TaskDef54694570' },
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

test('Can use EC2 taskdef as EventRule target with dead letter queue', () => {
  // GIVEN
  const deadLetterQueue = new sqs.Queue(stack, 'MyDeadLetterQueue');
  const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
    memoryLimitMiB: 256,
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    deadLetterQueue,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: { Ref: 'TaskDef54694570' },
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
        DeadLetterConfig: {
          Arn: {
            'Fn::GetAtt': [
              'MyDeadLetterQueueD997968A',
              'Arn',
            ],
          },
        },
      },
    ],
  });
});

test('Throws error for lacking of taskRole ' +
    'when importing from an EC2 task definition just from a task definition arn as EventRule target', () => {
  // GIVEN
  const taskDefinition = ecs.Ec2TaskDefinition.fromEc2TaskDefinitionArn(stack, 'TaskDef', 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef');

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', events.EventField.fromPath('$.detail.event')],
      }],
    }));
  }).toThrow('This operation requires the taskRole in ImportedTaskDefinition to be defined. ' +
    'Add the \'taskRole\' in ImportedTaskDefinitionProps to instantiate ImportedTaskDefinition');
});

test('Can import an EC2 task definition from task definition attributes as EventRule target', () => {
  // GIVEN
  const taskDefinition = ecs.Ec2TaskDefinition.fromEc2TaskDefinitionAttributes(stack, 'TaskDef', {
    taskDefinitionArn: 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef',
    networkMode: ecs.NetworkMode.BRIDGE,
    taskRole: new iam.Role(stack, 'TaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    }),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef',
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

test('Throws error for lacking of taskRole ' +
  'when importing from a Fargate task definition just from a task definition arn as EventRule target', () => {
  // GIVEN
  const taskDefinition = ecs.FargateTaskDefinition.fromFargateTaskDefinitionArn(stack, 'TaskDef', 'ImportedTaskDefArn');

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', events.EventField.fromPath('$.detail.event')],
      }],
    }));
  }).toThrow('This operation requires the taskRole in ImportedTaskDefinition to be defined. ' +
    'Add the \'taskRole\' in ImportedTaskDefinitionProps to instantiate ImportedTaskDefinition');
});

test('Can import a Fargate task definition from task definition attributes as EventRule target', () => {
  // GIVEN
  const taskDefinition = ecs.FargateTaskDefinition.fromFargateTaskDefinitionAttributes(stack, 'TaskDef', {
    taskDefinitionArn: 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef',
    networkMode: ecs.NetworkMode.AWS_VPC,
    taskRole: new iam.Role(stack, 'TaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    }),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef',
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

test('Throws error for lacking of taskRole ' +
  'when importing from a task definition just from a task definition arn as EventRule target', () => {
  // GIVEN
  const taskDefinition = ecs.TaskDefinition.fromTaskDefinitionArn(stack, 'TaskDef', 'ImportedTaskDefArn');

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', events.EventField.fromPath('$.detail.event')],
      }],
    }));
  }).toThrow('This operation requires the taskRole in ImportedTaskDefinition to be defined. ' +
    'Add the \'taskRole\' in ImportedTaskDefinitionProps to instantiate ImportedTaskDefinition');
});

test('Can import a Task definition from task definition attributes as EventRule target', () => {
  // GIVEN
  const taskDefinition = ecs.FargateTaskDefinition.fromFargateTaskDefinitionAttributes(stack, 'TaskDef', {
    taskDefinitionArn: 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef',
    networkMode: ecs.NetworkMode.AWS_VPC,
    taskRole: new iam.Role(stack, 'TaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    }),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: 'arn:aws:ecs:us-east-1:123456789012:task-definition/TaskDef',
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

test('Can use Fargate taskdef as EventRule target', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  const target = new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
  });
  rule.addTarget(target);

  // THEN
  expect(target.securityGroups?.length).toBeGreaterThan(0); // Generated security groups should be accessible.
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: { Ref: 'TaskDef54694570' },
          LaunchType: 'FARGATE',
          NetworkConfiguration: {
            AwsVpcConfiguration: {
              Subnets: [
                {
                  Ref: 'VpcPrivateSubnet1Subnet536B997A',
                },
              ],
              AssignPublicIp: 'DISABLED',
              SecurityGroups: [
                {
                  'Fn::GetAtt': [
                    'TaskDefSecurityGroupD50E7CF0',
                    'GroupId',
                  ],
                },
              ],
            },
          },
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

test('Can use Fargate taskdef as EventRule target with dead letter queue', () => {
  // GIVEN
  const deadLetterQueue = new sqs.Queue(stack, 'MyDeadLetterQueue');
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  const target = new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    deadLetterQueue,
  });
  rule.addTarget(target);

  // THEN
  expect(target.securityGroups?.length).toBeGreaterThan(0); // Generated security groups should be accessible.
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: { Ref: 'TaskDef54694570' },
          LaunchType: 'FARGATE',
          NetworkConfiguration: {
            AwsVpcConfiguration: {
              Subnets: [
                {
                  Ref: 'VpcPrivateSubnet1Subnet536B997A',
                },
              ],
              AssignPublicIp: 'DISABLED',
              SecurityGroups: [
                {
                  'Fn::GetAtt': [
                    'TaskDefSecurityGroupD50E7CF0',
                    'GroupId',
                  ],
                },
              ],
            },
          },
        },
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}]}',
        },
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
        DeadLetterConfig: {
          Arn: {
            'Fn::GetAtt': [
              'MyDeadLetterQueueD997968A',
              'Arn',
            ],
          },
        },
      },
    ],
  });
});

test('Can use same fargate taskdef with multiple rules', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const scheduledRule = new events.Rule(stack, 'ScheduleRule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  const patternRule = new events.Rule(stack, 'PatternRule', {
    eventPattern: {
      detail: ['test'],
    },
  });

  scheduledRule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
  }));

  expect(() => patternRule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
  }))).not.toThrow();
});

test('Can use same fargate taskdef multiple times in a rule', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'ScheduleRule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.a')],
    }],
  }));

  expect(() => rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.b')],
    }],
  }))).not.toThrow();
});

test('Isolated subnet does not have AssignPublicIp=true', () => {
  // GIVEN
  vpc = new ec2.Vpc(stack, 'Vpc2', {
    maxAzs: 1,
    subnetConfiguration: [{
      subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
      name: 'Isolated',
    }],
  });
  cluster = new ecs.Cluster(stack, 'EcsCluster2', { vpc });

  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    subnetSelection: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', 'yay'],
    }],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster2F191ADEC', 'Arn'] },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: { Ref: 'TaskDef54694570' },
          LaunchType: 'FARGATE',
          NetworkConfiguration: {
            AwsVpcConfiguration: {
              Subnets: [
                {
                  Ref: 'Vpc2IsolatedSubnet1SubnetB1A200D6',
                },
              ],
              AssignPublicIp: 'DISABLED',
              SecurityGroups: [
                {
                  'Fn::GetAtt': [
                    'TaskDefSecurityGroupD50E7CF0',
                    'GroupId',
                  ],
                },
              ],
            },
          },
        },
        Input: '{"containerOverrides":[{"name":"TheContainer","command":["echo","yay"]}]}',
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

testDeprecated('throws an error if both securityGroup and securityGroups is specified', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });
  const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      securityGroup,
      securityGroups: [securityGroup],
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', 'yay'],
      }],
    }));
  }).toThrow(/Only one of SecurityGroup or SecurityGroups can be populated./);
});

test('uses multiple security groups', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });
  const securityGroups = [
    new ec2.SecurityGroup(stack, 'SecurityGroupA', { vpc }),
    new ec2.SecurityGroup(stack, 'SecurityGroupB', { vpc }),
  ];

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    securityGroups,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', 'yay'],
    }],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          LaunchType: 'FARGATE',
          NetworkConfiguration: {
            AwsVpcConfiguration: {
              AssignPublicIp: 'DISABLED',
              SecurityGroups: [
                { 'Fn::GetAtt': ['SecurityGroupAED40ADC5', 'GroupId'] },
                { 'Fn::GetAtt': ['SecurityGroupB04591F90', 'GroupId'] },
              ],
              Subnets: [{ Ref: 'VpcPrivateSubnet1Subnet536B997A' }],
            },
          },
          TaskCount: 1,
          TaskDefinitionArn: {
            Ref: 'TaskDef54694570',
          },
        },
        Id: 'Target0',
        Input: '{"containerOverrides":[{"name":"TheContainer","command":["echo","yay"]}]}',
        RoleArn: { 'Fn::GetAtt': ['TaskDefEventsRoleFB3B67B8', 'Arn'] },
      },
    ],
  });
});

test('uses existing IAM role', () => {
  // GIVEN
  const role = new iam.Role(stack, 'CustomIamRole', {
    assumedBy: new iam.ServicePrincipal('events.amazonaws.com'),
  });

  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    role,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          LaunchType: 'FARGATE',
          TaskCount: 1,
          TaskDefinitionArn: {
            Ref: 'TaskDef54694570',
          },
        },
        RoleArn: { 'Fn::GetAtt': ['CustomIamRoleE653F2D1', 'Arn'] },
        Id: 'Target0',
      },
    ],
  });
});

test('uses the specific fargate platform version', () => {
  // GIVEN
  const platformVersion = ecs.FargatePlatformVersion.VERSION1_4;

  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    platformVersion,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: { 'Fn::GetAtt': ['EcsCluster97242B84', 'Arn'] },
        EcsParameters: {
          LaunchType: 'FARGATE',
          TaskCount: 1,
          TaskDefinitionArn: {
            Ref: 'TaskDef54694570',
          },
          PlatformVersion: '1.4.0',
        },
        Id: 'Target0',
      },
    ],
  });
});

test('sets the propagate tags flag', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    propagateTags: ecs.PropagatedTagSource.TASK_DEFINITION,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      Match.objectLike({
        EcsParameters: Match.objectLike({
          PropagateTags: ecs.PropagatedTagSource.TASK_DEFINITION,
        }),
      }),
    ],
  });
});

test('throws an error when trying to pass a disallowed value for propagateTags', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', events.EventField.fromPath('$.detail.event')],
      }],
      propagateTags: ecs.PropagatedTagSource.SERVICE, // propagateTags must be TASK_DEFINITION or NONE
    }));
  }).toThrow('When propagateTags is passed, it must be set to TASK_DEFINITION or NONE.');
});

test('set enableExecuteCommand', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    enableExecuteCommand: true,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      Match.objectLike({
        EcsParameters: Match.objectLike({
          EnableExecuteCommand: true,
        }),
      }),
    ],
  });
});

test('sets tag lists', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    tags: [
      {
        key: 'my-tag',
        value: 'my-tag-value',
      },
    ],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      Match.objectLike({
        EcsParameters: Match.objectLike({
          TagList: [
            {
              Key: 'my-tag',
              Value: 'my-tag-value',
            },
          ],
        }),
      }),
    ],
  });
});

test('enable assignPublicIp for public subnet', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    subnetSelection: { subnetType: ec2.SubnetType.PUBLIC },
    assignPublicIp: true,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        EcsParameters: {
          NetworkConfiguration: {
            AwsVpcConfiguration: {
              AssignPublicIp: 'ENABLED',
            },
          },
        },
      },
    ],
  });
});

test('DISABLE is set when disable assignPublicIp in a public subnet', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    subnetSelection: { subnetType: ec2.SubnetType.PUBLIC },
    assignPublicIp: false,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        EcsParameters: {
          NetworkConfiguration: {
            AwsVpcConfiguration: {
              AssignPublicIp: 'DISABLED',
            },
          },
        },
      },
    ],
  });
});

test('throw error when enable assignPublicIp for non-Fargate task', () => {
  // GIVEN
  const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TaskDef', {});
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', events.EventField.fromPath('$.detail.event')],
      }],
      subnetSelection: { subnetType: ec2.SubnetType.PUBLIC },
      assignPublicIp: true,
    }));
  }).toThrow('assignPublicIp is only supported for FARGATE tasks');
});

test('throw an error when assignPublicIp is set to true for private subnets', () => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // THEN
  expect(() => {
    rule.addTarget(new targets.EcsTask({
      cluster,
      taskDefinition,
      taskCount: 1,
      containerOverrides: [{
        containerName: 'TheContainer',
        command: ['echo', events.EventField.fromPath('$.detail.event')],
      }],
      subnetSelection: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      assignPublicIp: true,
    }));
  }).toThrow('assignPublicIp should be set to true only for PUBLIC subnets');
});

test.each([
  ['EC2', ecs.LaunchType.EC2, ecs.Compatibility.EC2_AND_FARGATE],
  ['FARGATE', ecs.LaunchType.FARGATE, ecs.Compatibility.EC2_AND_FARGATE],
  ['EXTERNAL', ecs.LaunchType.EXTERNAL, ecs.Compatibility.EC2_AND_FARGATE],
  ['EC2', undefined, ecs.Compatibility.EC2_AND_FARGATE],
  ['FARGATE', undefined, ecs.Compatibility.FARGATE],
])('launch type %s is set by %s type specified', (expected, launchType, compatibility) => {
  // GIVEN
  const taskDefinition = new ecs.TaskDefinition(stack, 'TaskDef', {
    networkMode: ecs.NetworkMode.AWS_VPC,
    compatibility,
    cpu: '256',
    memoryMiB: '512',
  });
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    launchType,
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        EcsParameters: {
          LaunchType: expected,
        },
      },
    ],
  });
});

test('When using non-imported TaskDefinition, the IAM policy `Resource` should use `Ref` to reference the TaskDefinition', () => {
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.rate(cdk.Duration.hours(1)),
  });

  rule.addTarget(
    new EcsTask({
      cluster: cluster,
      taskDefinition: taskDefinition,
    }),
  );

  const policyMatch = Match.objectLike({
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'ecs:RunTask',
          Resource: {
            Ref: 'TaskDef54694570',
          },
        }),
      ]),
    },
  });
  const template = Template.fromStack(stack);
  template.hasResource('AWS::IAM::Policy', { Properties: policyMatch });
});

test('Imported task definition without revision adds wildcard to policy resource', () => {
  const taskDefinition = ecs.FargateTaskDefinition.fromFargateTaskDefinitionAttributes(stack, 'TaskDefImport', {
    taskDefinitionArn: 'arn:aws:ecs:us-west-2:012345678901:task-definition/MyTask',
    taskRole: iam.Role.fromRoleArn(stack, 'RoleImport', 'arn:aws:iam::012345678901:role/MyTaskRole'),
    networkMode: ecs.NetworkMode.AWS_VPC,
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.rate(cdk.Duration.hours(1)),
  });

  rule.addTarget(
    new EcsTask({
      cluster: cluster,
      taskDefinition: taskDefinition,
    }),
  );

  const policyMatch = Match.objectLike({
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'ecs:RunTask',
          Resource: `${taskDefinition.taskDefinitionArn}:*`,
        }),
      ]),
    },
  });
  const template = Template.fromStack(stack);
  template.hasResource('AWS::IAM::Policy', { Properties: policyMatch });
});

test('Imported task definition with revision uses original arn for policy resource', () => {
  const taskDefinition = ecs.FargateTaskDefinition.fromFargateTaskDefinitionAttributes(stack, 'TaskDefImport', {
    taskDefinitionArn: 'arn:aws:ecs:us-west-2:012345678901:task-definition/MyTask:1',
    taskRole: iam.Role.fromRoleArn(stack, 'RoleImport', 'arn:aws:iam::012345678901:role/MyTaskRole'),
    networkMode: ecs.NetworkMode.AWS_VPC,
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.rate(cdk.Duration.hours(1)),
  });

  rule.addTarget(
    new EcsTask({
      cluster: cluster,
      taskDefinition: taskDefinition,
    }),
  );

  const policyMatch = Match.objectLike({
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'ecs:RunTask',
          Resource: taskDefinition.taskDefinitionArn,
        }),
      ]),
    },
  });
  const template = Template.fromStack(stack);
  template.hasResource('AWS::IAM::Policy', { Properties: policyMatch });
});

test.each([
  [10, 'less than minimum'],
  [201, 'greater than maximum'],
])('throws an error when ephemeral storage size is %s 20 GiB (%s)', (sizeInGiB: number) => {
  // GIVEN
  const taskDefinition = new ecs.FargateTaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });
  const target = new targets.EcsTask({
    cluster,
    taskDefinition,
    ephemeralStorage: {
      sizeInGiB,
    },
  });

  // WHEN
  // THEN
  expect(() => {
    rule.addTarget(target);
  }).toThrow(/Ephemeral storage size must be between 20 GiB and 200 GiB/);
});

test('it can override all possible ecs task properties via the input property', () => {
  const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TaskDef');
  taskDefinition.addContainer('TheContainer', {
    image: ecs.ContainerImage.fromRegistry('henk'),
    memoryLimitMiB: 256,
  });

  const rule = new events.Rule(stack, 'Rule', {
    schedule: events.Schedule.expression('rate(1 min)'),
  });

  // WHEN
  rule.addTarget(new targets.EcsTask({
    cluster,
    taskDefinition,
    taskCount: 1,

    // All possible overrides
    // see https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_TaskOverride.html
    containerOverrides: [{
      containerName: 'TheContainer',
      command: ['echo', events.EventField.fromPath('$.detail.event')],
    }],
    cpu: '1024',
    ephemeralStorage: {
      sizeInGiB: 100,
    },
    executionRole: new iam.Role(stack, 'ExecutionRoleOverride', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    }),
    inferenceAcceleratorOverrides: [{
      deviceName: 'device-name',
      deviceType: 'device-type',
    }],
    memory: '2048',
    taskRole: new iam.Role(stack, 'TaskRoleOverride', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    }),
  }));

  // THEN
  Template.fromStack(stack);
  debugger;
  Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
    Targets: [
      {
        Arn: {
          'Fn::GetAtt': [
            'EcsCluster97242B84',
            'Arn',
          ],
        },
        EcsParameters: {
          TaskCount: 1,
          TaskDefinitionArn: {
            Ref: 'TaskDef54694570',
          },
        },
        Id: 'Target0',
        InputTransformer: {
          InputPathsMap: {
            'detail-event': '$.detail.event',
          },
          InputTemplate: {
            'Fn::Join': [
              '',
              [
                '{"containerOverrides":[{"name":"TheContainer","command":["echo",<detail-event>]}],"cpu":"1024","ephemeralStorage":{"sizeInGiB":100},"executionRole":"',
                {
                  'Fn::GetAtt': [
                    'ExecutionRoleOverrideE576BCB8',
                    'Arn',
                  ],
                },
                '","inferenceAcceleratorOverrides":[{"deviceName":"device-name","deviceType":"device-type"}],"memory":"2048","taskRole":"',
                {
                  'Fn::GetAtt': [
                    'TaskRoleOverride9910DE20',
                    'Arn',
                  ],
                },
                '"}',
              ],
            ],
          },
        },
        RoleArn: {
          'Fn::GetAtt': [
            'TaskDefEventsRoleFB3B67B8',
            'Arn',
          ],
        },
      },
    ],
  });
});
