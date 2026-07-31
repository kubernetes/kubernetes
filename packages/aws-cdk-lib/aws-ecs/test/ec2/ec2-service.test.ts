import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Annotations, Match, Template } from '../../../assertions';
import * as autoscaling from '../../../aws-autoscaling';
import * as cloudwatch from '../../../aws-cloudwatch';
import * as ec2 from '../../../aws-ec2';
import * as elb from '../../../aws-elasticloadbalancing';
import * as elbv2 from '../../../aws-elasticloadbalancingv2';
import * as kms from '../../../aws-kms';
import * as logs from '../../../aws-logs';
import * as s3 from '../../../aws-s3';
import * as cloudmap from '../../../aws-servicediscovery';
import * as cdk from '../../../core';
import { App } from '../../../core';
import { flattenMeta } from '../../../core/test/util';
import { ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME } from '../../../cx-api';
import * as ecs from '../../lib';
import {
  AlarmBehavior,
  DeploymentControllerType,
  LaunchType,
  PropagatedTagSource,
} from '../../lib/base/base-service';
import { PlacementConstraint, PlacementStrategy } from '../../lib/placement';
import { acknowledgeTestValidationRules, addDefaultCapacityProvider } from '../util';

describe('ec2 service', () => {
  describe('When creating an EC2 Service', () => {
    test('with only required properties set, it correctly sets default properties', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'Ec2TaskDef0226F28C',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        LaunchType: LaunchType.EC2,
        SchedulingStrategy: 'REPLICA',
        EnableECSManagedTags: false,
        AvailabilityZoneRebalancing: Match.absent(),
      });

      expect(service.node.defaultChild).toBeDefined();
    });

    test.each([false, true])('suggests using circuitBreaker if %p set', (circuitBreakerSet) => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'Stack');
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        circuitBreaker: circuitBreakerSet ? { } : undefined,
      });

      // THEN
      const warnings = flattenMeta(app.synth().getStackByName('Stack').metadata)['/Stack/Ec2Service']['aws:cdk:warning'];

      if (circuitBreakerSet) {
        expect(warnings).not.toContainEqual(expect.stringContaining('Enable the \'circuitBreaker\' property'));
      } else {
        expect(warnings).toContainEqual(expect.stringContaining('Enable the \'circuitBreaker\' property'));
      }
    });

    [false, undefined].forEach((value) => {
      test('set cloudwatch permissions based on falsy feature flag when no cloudwatch log configured', () => {
        // GIVEN
        const app = new App(
          {
            context: {
              '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions': value,
            },
          },
        );
        const stack = new cdk.Stack(app);
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          enableExecuteCommand: true,
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
          PolicyDocument: {
            Statement: [
              {
                Action: [
                  'ssmmessages:CreateControlChannel',
                  'ssmmessages:CreateDataChannel',
                  'ssmmessages:OpenControlChannel',
                  'ssmmessages:OpenDataChannel',
                ],
                Effect: 'Allow',
                Resource: '*',
              },
              {
                Action: 'logs:DescribeLogGroups',
                Effect: 'Allow',
                Resource: '*',
              },
              {
                Action: [
                  'logs:CreateLogStream',
                  'logs:DescribeLogStreams',
                  'logs:PutLogEvents',
                ],
                Effect: 'Allow',
                Resource: '*',
              },
            ],
            Version: '2012-10-17',
          },
          PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
          Roles: [
            {
              Ref: 'Ec2TaskDefTaskRole400FA349',
            },
          ],
        });
      });
    });

    test('set cloudwatch permissions based on true feature flag when no cloudwatch log configured', () => {
      // GIVEN
      const app = new App(
        {
          context: {
            '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions': true,
          },
        },
      );
      const stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('set cloudwatch permissions based on true feature flag when cloudwatch log is configured', () => {
      // GIVEN
      const app = new App(
        {
          context: {
            '@aws-cdk/aws-ecs:reduceEc2FargateCloudWatchPermissions': true,
          },
        },
      );
      const stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          logConfiguration: {
            cloudWatchLogGroup: new logs.LogGroup(stack, 'LogGroup'),
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('allows setting enable execute command', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'Ec2TaskDef0226F28C',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        LaunchType: LaunchType.EC2,
        SchedulingStrategy: 'REPLICA',
        EnableECSManagedTags: false,
        EnableExecuteCommand: true,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('no logging enabled when logging field is set to NONE', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          logging: ecs.ExecuteCommandLogging.NONE,
        },
      });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        logging: ecs.LogDrivers.awsLogs({
          logGroup,
          streamPrefix: 'log-group',
        }),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('enables execute command logging when logging field is set to OVERRIDE', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      const execBucket = new s3.Bucket(stack, 'ExecBucket');

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
            s3Bucket: execBucket,
            s3KeyPrefix: 'exec-output',
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetBucketLocation',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'ExecBucket29559356',
                    },
                    '/*',
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });
    });

    test('enables only execute command session encryption', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      const kmsKey = new kms.Key(stack, 'KmsKey');

      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      const execBucket = new s3.Bucket(stack, 'EcsExecBucket');

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          kmsKey,
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
            s3Bucket: execBucket,
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });

      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'kms:Decrypt',
                'kms:GenerateDataKey',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'KmsKey46693ADD',
                  'Arn',
                ],
              },
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetBucketLocation',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'EcsExecBucket4F468651',
                    },
                    '/*',
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
        KeyPolicy: {
          Statement: [
            {
              Action: 'kms:*',
              Effect: 'Allow',
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':iam::',
                      {
                        Ref: 'AWS::AccountId',
                      },
                      ':root',
                    ],
                  ],
                },
              },
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    test('enables encryption for execute command logging', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      const kmsKey = new kms.Key(stack, 'KmsKey');

      const logGroup = new logs.LogGroup(stack, 'LogGroup', {
        encryptionKey: kmsKey,
      });

      const execBucket = new s3.Bucket(stack, 'EcsExecBucket', {
        encryptionKey: kmsKey,
      });

      // WHEN
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        executeCommandConfiguration: {
          kmsKey,
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
            cloudWatchEncryptionEnabled: true,
            s3Bucket: execBucket,
            s3EncryptionEnabled: true,
            s3KeyPrefix: 'exec-output',
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });

      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        enableExecuteCommand: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ssmmessages:CreateControlChannel',
                'ssmmessages:CreateDataChannel',
                'ssmmessages:OpenControlChannel',
                'ssmmessages:OpenDataChannel',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'kms:Decrypt',
                'kms:GenerateDataKey',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'KmsKey46693ADD',
                  'Arn',
                ],
              },
            },
            {
              Action: 'logs:DescribeLogGroups',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: [
                'logs:CreateLogStream',
                'logs:DescribeLogStreams',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':logs:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':log-group:',
                    {
                      Ref: 'LogGroupF5B46931',
                    },
                    ':*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetBucketLocation',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'EcsExecBucket4F468651',
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: 's3:GetEncryptionConfiguration',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':s3:::',
                    {
                      Ref: 'EcsExecBucket4F468651',
                    },
                  ],
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'Ec2TaskDefTaskRoleDefaultPolicyA24FB970',
        Roles: [
          {
            Ref: 'Ec2TaskDefTaskRole400FA349',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
        KeyPolicy: {
          Statement: [
            {
              Action: 'kms:*',
              Effect: 'Allow',
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':iam::',
                      {
                        Ref: 'AWS::AccountId',
                      },
                      ':root',
                    ],
                  ],
                },
              },
              Resource: '*',
            },
            {
              Action: [
                'kms:Encrypt*',
                'kms:Decrypt*',
                'kms:ReEncrypt*',
                'kms:GenerateDataKey*',
                'kms:Describe*',
              ],
              Condition: {
                ArnLike: {
                  'kms:EncryptionContext:aws:logs:arn': {
                    'Fn::Join': [
                      '',
                      [
                        'arn:',
                        {
                          Ref: 'AWS::Partition',
                        },
                        ':logs:',
                        {
                          Ref: 'AWS::Region',
                        },
                        ':',
                        {
                          Ref: 'AWS::AccountId',
                        },
                        ':*',
                      ],
                    ],
                  },
                },
              },
              Effect: 'Allow',
              Principal: {
                Service: {
                  'Fn::Join': [
                    '',
                    [
                      'logs.',
                      {
                        Ref: 'AWS::Region',
                      },
                      '.amazonaws.com',
                    ],
                  ],
                },
              },
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    test('with custom cloudmap namespace', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      const cloudMapNamespace = new cloudmap.PrivateDnsNamespace(stack, 'TestCloudMapNamespace', {
        name: 'scorekeep.com',
        vpc,
      });

      cdk.Validations.of(stack).acknowledge(
        { id: 'CloudFormation-Validate::F3034', reason: 'failureThreshold intentionally exceeds the CloudFormation maximum of 10; the test asserts it is passed through to HealthCheckCustomConfig unmodified' },
      );

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
          failureThreshold: 20,
          cloudMapNamespace,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'SRV',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'TestCloudMapNamespace1FB9B446',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 20,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'TestCloudMapNamespace1FB9B446',
            'Id',
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::PrivateDnsNamespace', {
        Name: 'scorekeep.com',
        Vpc: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });
    });

    test('with all properties set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.AWS_VPC,
      });

      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      // WHEN
      cdk.Validations.of(stack).acknowledge(
        { id: 'CloudFormation-Validate::F3034', reason: 'failureThreshold intentionally exceeds the CloudFormation maximum of 10; this test exercises every property, not valid value ranges' },
      );

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        desiredCount: 2,
        assignPublicIp: true,
        cloudMapOptions: {
          name: 'myapp',
          dnsRecordType: cloudmap.DnsRecordType.A,
          dnsTtl: cdk.Duration.seconds(50),
          failureThreshold: 20,
        },
        daemon: false,
        healthCheckGracePeriod: cdk.Duration.seconds(60),
        maxHealthyPercent: 150,
        minHealthyPercent: 55,
        deploymentController: {
          type: ecs.DeploymentControllerType.ECS,
        },
        securityGroups: [new ec2.SecurityGroup(stack, 'SecurityGroup1', {
          allowAllOutbound: true,
          description: 'Example',
          securityGroupName: 'Bob',
          vpc,
        })],
        serviceName: 'bonjour',
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      service.addPlacementConstraints(PlacementConstraint.memberOf('attribute:ecs.instance-type =~ t2.*'));
      service.addPlacementStrategies(PlacementStrategy.spreadAcross(ecs.BuiltInAttributes.AVAILABILITY_ZONE));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'Ec2TaskDef0226F28C',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 150,
          MinimumHealthyPercent: 55,
        },
        DeploymentController: {
          Type: ecs.DeploymentControllerType.ECS,
        },
        DesiredCount: 2,
        LaunchType: LaunchType.EC2,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'ENABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'SecurityGroup1F554B36F',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPublicSubnet1SubnetF6608456',
              },
              {
                Ref: 'MyVpcPublicSubnet2Subnet492B6BFB',
              },
            ],
          },
        },
        PlacementConstraints: [
          {
            Expression: 'attribute:ecs.instance-type =~ t2.*',
            Type: 'memberOf',
          },
        ],
        PlacementStrategies: [
          {
            Field: 'attribute:ecs.availability-zone',
            Type: 'spread',
          },
        ],
        SchedulingStrategy: 'REPLICA',
        ServiceName: 'bonjour',
        ServiceRegistries: [
          {
            RegistryArn: {
              'Fn::GetAtt': [
                'Ec2ServiceCloudmapService45B52C0F',
                'Arn',
              ],
            },
          },
        ],
        AvailabilityZoneRebalancing: 'ENABLED',
      });
    });

    test('with autoscaling group capacity provider', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
        vpc,
        instanceType: new ec2.InstanceType('bogus'),
        machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
      });
      acknowledgeTestValidationRules(stack);

      // WHEN
      const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provider', {
        autoScalingGroup,
        enableManagedTerminationProtection: false,
      });
      cluster.addAsgCapacityProvider(capacityProvider);

      const taskDefinition = new ecs.TaskDefinition(stack, 'ServerTask', {
        compatibility: ecs.Compatibility.EC2,
      });
      taskDefinition.addContainer('app', {
        image: new ecs.RepositoryImage('bogus'),
        cpu: 1024,
        memoryReservationMiB: 900,
        portMappings: [{
          containerPort: 80,
        }],
      });
      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        desiredCount: 0,
        capacityProviderStrategies: [{
          capacityProvider: capacityProvider.capacityProviderName,
        }],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        CapacityProviderStrategy: [
          {
            CapacityProvider: {
              Ref: 'providerD3FF4D3A',
            },
          },
        ],
      });
    });

    test('with multiple security groups, it correctly updates the cfn template', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.AWS_VPC,
      });
      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      const securityGroup1 = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
        allowAllOutbound: true,
        description: 'Example',
        securityGroupName: 'Bingo',
        vpc,
      });
      const securityGroup2 = new ec2.SecurityGroup(stack, 'SecurityGroup2', {
        allowAllOutbound: false,
        description: 'Example',
        securityGroupName: 'Rolly',
        vpc,
      });

      // WHEN
      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        desiredCount: 2,
        assignPublicIp: true,
        daemon: false,
        securityGroups: [securityGroup1, securityGroup2],
        serviceName: 'bonjour',
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        TaskDefinition: {
          Ref: 'Ec2TaskDef0226F28C',
        },
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DesiredCount: 2,
        LaunchType: LaunchType.EC2,
        NetworkConfiguration: {
          AwsvpcConfiguration: {
            AssignPublicIp: 'ENABLED',
            SecurityGroups: [
              {
                'Fn::GetAtt': [
                  'SecurityGroup1F554B36F',
                  'GroupId',
                ],
              },
              {
                'Fn::GetAtt': [
                  'SecurityGroup23BE86BB7',
                  'GroupId',
                ],
              },
            ],
            Subnets: [
              {
                Ref: 'MyVpcPublicSubnet1SubnetF6608456',
              },
              {
                Ref: 'MyVpcPublicSubnet2Subnet492B6BFB',
              },
            ],
          },
        },
        SchedulingStrategy: 'REPLICA',
        ServiceName: 'bonjour',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Example',
        GroupName: 'Bingo',
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Example',
        GroupName: 'Rolly',
        SecurityGroupEgress: [
          {
            CidrIp: '255.255.255.255/32',
            Description: 'Disallow all traffic',
            FromPort: 252,
            IpProtocol: 'icmp',
            ToPort: 86,
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });
    });

    test('throws when availability zone rebalancing is enabled and maxHealthyPercent is 100', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
          maxHealthyPercent: 100,
        });
      }).toThrow(/requires maxHealthyPercent > 100/);
    });

    test('throws when availability zone rebalancing is enabled and daemon is true', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
          daemon: true,
        });
      }).toThrow(/cannot be used with daemon mode/);
    });

    test('sets task definition to family when CODE_DEPLOY deployment controller is specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: ecs.DeploymentControllerType.CODE_DEPLOY,
        },
      });

      // THEN
      Template.fromStack(stack).hasResource('AWS::ECS::Service', {
        Properties: {
          TaskDefinition: 'Ec2TaskDef',
          DeploymentController: {
            Type: 'CODE_DEPLOY',
          },
        },
        DependsOn: [
          'Ec2TaskDef0226F28C',
          'Ec2TaskDefTaskRole400FA349',
        ],
      });
    });

    testDeprecated('throws when both securityGroup and securityGroups are supplied', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.AWS_VPC,
      });
      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });
      const securityGroup1 = new ec2.SecurityGroup(stack, 'SecurityGroup1', {
        allowAllOutbound: true,
        description: 'Example',
        securityGroupName: 'Bingo',
        vpc,
      });
      const securityGroup2 = new ec2.SecurityGroup(stack, 'SecurityGroup2', {
        allowAllOutbound: false,
        description: 'Example',
        securityGroupName: 'Rolly',
        vpc,
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          desiredCount: 2,
          assignPublicIp: true,
          maxHealthyPercent: 150,
          minHealthyPercent: 55,
          securityGroup: securityGroup1,
          securityGroups: [securityGroup2],
          serviceName: 'bonjour',
          vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        });
      }).toThrow(/Only one of SecurityGroup or SecurityGroups can be populated./);
    });

    test('throws when task definition is not EC2 compatible', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.TaskDefinition(stack, 'FargateTaskDef', {
        compatibility: ecs.Compatibility.FARGATE,
        cpu: '256',
        memoryMiB: '512',
      });
      taskDefinition.addContainer('BaseContainer', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryReservationMiB: 10,
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
        });
      }).toThrow(/Supplied TaskDefinition is not configured for compatibility with EC2/);
    });

    test('ignore task definition and launch type if deployment controller is set to be EXTERNAL', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: DeploymentControllerType.EXTERNAL,
        },
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Ec2Service', 'taskDefinition and launchType are blanked out when using external deployment controller. [ack: @aws-cdk/aws-ecs:externalDeploymentController]');
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        Cluster: {
          Ref: 'EcsCluster97242B84',
        },
        DeploymentConfiguration: {
          MaximumPercent: 200,
          MinimumHealthyPercent: 50,
        },
        SchedulingStrategy: 'REPLICA',
        EnableECSManagedTags: false,
      });
    });

    test('add warning to annotations if circuitBreaker is specified with a non-ECS DeploymentControllerType', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        deploymentController: {
          type: DeploymentControllerType.EXTERNAL,
        },
        circuitBreaker: { rollback: true },
      });
      app.synth();

      // THEN
      expect(service.node.metadata[0].data).toEqual('taskDefinition and launchType are blanked out when using external deployment controller. [ack: @aws-cdk/aws-ecs:externalDeploymentController]');
      expect(service.node.metadata[1].data).toEqual('Deployment circuit breaker requires the ECS deployment controller.');
    });

    test('errors if daemon and desiredCount both specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      taskDefinition.addContainer('BaseContainer', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryReservationMiB: 10,
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          daemon: true,
          desiredCount: 2,
        });
      }).toThrow(/Don't supply desiredCount/);
    });

    test('errors if daemon and maximumPercent not 100', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      taskDefinition.addContainer('BaseContainer', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryReservationMiB: 10,
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          daemon: true,
          maxHealthyPercent: 300,
        });
      }).toThrow(/Maximum percent must be 100 for daemon mode./);
    });

    test('errors if minimum not less than maximum', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      taskDefinition.addContainer('BaseContainer', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryReservationMiB: 10,
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          daemon: true,
          minHealthyPercent: 100,
          maxHealthyPercent: 100,
        });
      }).toThrow(/Minimum healthy percent must be less than maximum healthy percent./);
    });

    test('errors if no container definitions', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      // Errors on validation, not on construction.
      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      expect(() => {
        Template.fromStack(stack);
      }).toThrow(/one essential container/);
    });

    test('allows adding the default container after creating the service', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      new ecs.Ec2Service(stack, 'FargateService', {
        cluster,
        taskDefinition,
      });

      // Add the container *after* creating the service
      taskDefinition.addContainer('main', {
        image: ecs.ContainerImage.fromRegistry('somecontainer'),
        memoryReservationMiB: 10,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::TaskDefinition', {
        ContainerDefinitions: [
          Match.objectLike({
            Name: 'main',
          }),
        ],
      });
    });

    test('sets daemon scheduling strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        SchedulingStrategy: 'DAEMON',
        DeploymentConfiguration: {
          MaximumPercent: 100,
          MinimumHealthyPercent: 0,
        },
      });
    });

    test('warning if minHealthyPercent not set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Ec2Service', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercent]');
    });

    test('no warning if minHealthyPercent set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        minHealthyPercent: 50,
      });

      // THEN
      Annotations.fromStack(stack).hasNoWarning('/Default/Ec2Service', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercent]');
    });

    test('warning if minHealthyPercent not set for a daemon service', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Ec2Service', 'minHealthyPercent has not been configured so the default value of 0% for a daemon service is used. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercentDaemon]');
      Annotations.fromStack(stack).hasNoWarning('/Default/Ec2Service', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercent]');
    });

    test('no warning if minHealthyPercent set for a daemon service', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        minHealthyPercent: 50,
        daemon: true,
      });

      // THEN
      Annotations.fromStack(stack).hasNoWarning('/Default/Ec2Service', 'minHealthyPercent has not been configured so the default value of 0% for a daemon service is used. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercentDaemon]');
      Annotations.fromStack(stack).hasNoWarning('/Default/Ec2Service', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705 [ack: @aws-cdk/aws-ecs:minHealthyPercent]');
    });

    describe('with a TaskDefinition with Bridge network mode', () => {
      test('it errors if vpcSubnets is specified', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.BRIDGE,
        });

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        // THEN
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            vpcSubnets: {
              subnetType: ec2.SubnetType.PUBLIC,
            },
          });
        });

        // THEN
      });

      test('it errors if assignPublicIp is true', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.BRIDGE,
        });

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        // THEN
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            assignPublicIp: true,
          });
        }).toThrow(/vpcSubnets, securityGroup\(s\) and assignPublicIp can only be used in AwsVpc networking mode/);

        // THEN
      });

      test('it errors if vpc subnets is provided', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const subnet = new ec2.Subnet(stack, 'MySubnet', {
          vpcId: vpc.vpcId,
          availabilityZone: 'eu-central-1a',
          cidrBlock: '10.10.0.0/20',
        });
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.BRIDGE,
        });
        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        // THEN
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            vpcSubnets: {
              subnets: [subnet],
            },
          });
        }).toThrow(/vpcSubnets, securityGroup\(s\) and assignPublicIp can only be used in AwsVpc networking mode/);

        // THEN
      });

      test('it errors if security group is provided', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const securityGroup = new ec2.SecurityGroup(stack, 'MySG', { vpc });
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.BRIDGE,
        });
        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        // THEN
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            securityGroups: [securityGroup],
          });
        }).toThrow(/vpcSubnets, securityGroup\(s\) and assignPublicIp can only be used in AwsVpc networking mode/);

        // THEN
      });

      test('it errors if multiple security groups is provided', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const securityGroups = [
          new ec2.SecurityGroup(stack, 'MyFirstSG', { vpc }),
          new ec2.SecurityGroup(stack, 'MySecondSG', { vpc }),
        ];
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.BRIDGE,
        });
        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        // THEN
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            securityGroups,
          });
        }).toThrow(/vpcSubnets, securityGroup\(s\) and assignPublicIp can only be used in AwsVpc networking mode/);

        // THEN
      });
    });

    describe('with a TaskDefinition with AwsVpc network mode', () => {
      test('it creates a security group for the service', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.AWS_VPC,
        });

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          NetworkConfiguration: {
            AwsvpcConfiguration: {
              AssignPublicIp: 'DISABLED',
              SecurityGroups: [
                {
                  'Fn::GetAtt': [
                    'Ec2ServiceSecurityGroupAEC30825',
                    'GroupId',
                  ],
                },
              ],
              Subnets: [
                {
                  Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
                },
                {
                  Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
                },
              ],
            },
          },
        });
      });

      test('it allows vpcSubnets', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
          networkMode: ecs.NetworkMode.AWS_VPC,
        });

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });

        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          vpcSubnets: {
            subnetType: ec2.SubnetType.PUBLIC,
          },
        });

        // THEN
      });
    });

    test('with distinctInstance placement constraint', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        placementConstraints: [ecs.PlacementConstraint.distinctInstances()],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementConstraints: [{
          Type: 'distinctInstance',
        }],
      });
    });

    test('with memberOf placement constraints', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      service.addPlacementConstraints(PlacementConstraint.memberOf('attribute:ecs.instance-type =~ t2.*'));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementConstraints: [{
          Expression: 'attribute:ecs.instance-type =~ t2.*',
          Type: 'memberOf',
        }],
      });
    });

    test('throws with AvailabilityZoneBalancing.ENABLED and placement constraint uses memberOf attribute:ecs.availability-zone', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      // THEN
      expect(() => {
        service.addPlacementConstraints(
          PlacementConstraint.memberOf(`${ecs.BuiltInAttributes.AVAILABILITY_ZONE} =~ us-east-1a`),
        );
      }).toThrow(/AvailabilityZoneBalancing.ENABLED disallows usage of "attribute:ecs.availability-zone"/);
    });

    test('does not throw with AvailabilityZoneBalancing.ENABLED and placement constraints that don\'t use memberOf attribute:ecs.availability-zone', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      // WHEN
      service.addPlacementConstraints(PlacementConstraint.memberOf('attribute:ecs.instance-type =~ t2.*'));

      // THEN
      // did not throw
    });

    test('with spreadAcross container instances strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      service.addPlacementStrategies(PlacementStrategy.spreadAcrossInstances());

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: [{
          Field: 'instanceId',
          Type: 'spread',
        }],
      });
    });

    test('with spreadAcross placement strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      service.addPlacementStrategies(PlacementStrategy.spreadAcross(ecs.BuiltInAttributes.AVAILABILITY_ZONE));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: [{
          Field: 'attribute:ecs.availability-zone',
          Type: 'spread',
        }],
      });
    });

    test('throws with AvailabilityZoneBalancing.ENABLED and first placement strategy is not spread-across-AZ', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      // THEN
      expect(() => {
        service.addPlacementStrategies(PlacementStrategy.spreadAcrossInstances());
      }).toThrow(/requires that the first placement strategy, if any, be 'spread across "attribute:ecs.availability-zone"'/);
    });

    test('does not throw with AvailabilityZoneBalancing.ENABLED and first placement strategy is spread-across-AZ', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      // WHEN
      service.addPlacementStrategies(PlacementStrategy.spreadAcross(ecs.BuiltInAttributes.AVAILABILITY_ZONE));
      service.addPlacementStrategies(PlacementStrategy.spreadAcrossInstances());

      // THEN
      // did not throw
    });

    test('can turn PlacementStrategy into json format', () => {
      // THEN
      expect(PlacementStrategy.spreadAcross(ecs.BuiltInAttributes.AVAILABILITY_ZONE).toJson()).toEqual([{
        type: 'spread',
        field: 'attribute:ecs.availability-zone',
      }]);
    });

    test('can turn PlacementConstraints into json format', () => {
      // THEN
      expect(PlacementConstraint.distinctInstances().toJson()).toEqual([{
        type: 'distinctInstance',
      }]);
    });

    test('errors when spreadAcross with no input', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      expect(() => {
        service.addPlacementStrategies(PlacementStrategy.spreadAcross());
      }).toThrow('spreadAcross: give at least one field to spread by');
    });

    test('errors with spreadAcross placement strategy if daemon specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      // THEN
      expect(() => {
        service.addPlacementStrategies(PlacementStrategy.spreadAcross(ecs.BuiltInAttributes.AVAILABILITY_ZONE));
      });
    });

    test('with no placement constraints', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementConstraints: Match.absent(),
      });
    });

    test('with empty [] placement constraints', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        placementConstraints: [],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementConstraints: Match.arrayEquals([]),
      });
    });

    test('with empty [] placement strategies', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        placementStrategies: [],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: Match.arrayEquals([]),
      });
    });

    testDeprecated('with both propagateTags and propagateTaskTagsFrom defined', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      expect(() => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          propagateTags: PropagatedTagSource.SERVICE,
          propagateTaskTagsFrom: PropagatedTagSource.SERVICE,
        });
      }).toThrow(/You can only specify either propagateTags or propagateTaskTagsFrom. Alternatively, you can leave both blank/);
    });

    test('with no placement strategy if daemon specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: Match.absent(),
      });
    });

    test('with random placement strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      service.addPlacementStrategies(PlacementStrategy.randomly());

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: [{
          Type: 'random',
        }],
      });
    });

    test('errors with random placement strategy if daemon specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      // THEN
      expect(() => {
        service.addPlacementStrategies(PlacementStrategy.randomly());
      }).toThrow();
    });

    test('with packedbyCpu placement strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      service.addPlacementStrategies(PlacementStrategy.packedByCpu());

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: [{
          Field: 'CPU',
          Type: 'binpack',
        }],
      });
    });

    test('with packedbyMemory placement strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      service.addPlacementStrategies(PlacementStrategy.packedByMemory());

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: [{
          Field: 'MEMORY',
          Type: 'binpack',
        }],
      });
    });

    test('with packedBy placement strategy', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      service.addPlacementStrategies(PlacementStrategy.packedBy(ecs.BinPackResource.MEMORY));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        PlacementStrategies: [{
          Field: 'MEMORY',
          Type: 'binpack',
        }],
      });
    });

    test('errors with packedBy placement strategy if daemon specified', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      const service = new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      // THEN
      expect(() => {
        service.addPlacementStrategies(PlacementStrategy.packedBy(ecs.BinPackResource.MEMORY));
      }).toThrow();
    });

    test('throws an exception if non-DAEMON service is added but no EC2 capacity is associated with the cluster', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
      });

      expect(() => {
        Template.fromStack(stack);
      }).toThrow(/Cluster for this service needs Ec2 capacity/);
    });

    test('does not throw an exception if DAEMON service is added but no EC2 capacity is associated with the cluster', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
        memoryLimitMiB: 512,
      });

      new ecs.Ec2Service(stack, 'Ec2Service', {
        cluster,
        taskDefinition,
        daemon: true,
      });

      expect(() => {
        Template.fromStack(stack);
      }).not.toThrow();
    });

    describe('with deployment alarms', () => {
      let stack: cdk.Stack;
      let cluster: ecs.Cluster;
      let taskDefinition: ecs.TaskDefinition;

      beforeEach(() => {
        stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');

        taskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });
      });

      test('minimum configuration', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');

        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          deploymentAlarms: {
            alarmNames: [myAlarm.alarmName],
          },
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: true,
              Rollback: true,
              AlarmNames: [myAlarm.alarmName],
            },
          },
        });
      });

      test('and explicitly set behavior to ROLLBACK_ON_ALARM', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');

        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          deploymentAlarms: {
            alarmNames: [myAlarm.alarmName],
            behavior: AlarmBehavior.ROLLBACK_ON_ALARM,
          },
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: true,
              Rollback: true,
              AlarmNames: [myAlarm.alarmName],
            },
          },
        });
      });

      test('and explicitly set behavior to FAIL_ON_ALARM', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          deploymentAlarms: {
            alarmNames: [myAlarm.alarmName],
            behavior: AlarmBehavior.FAIL_ON_ALARM,
          },
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: true,
              Rollback: false,
              AlarmNames: [myAlarm.alarmName],
            },
          },
        });
      });

      test('use enableDeploymentAlarms()', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');

        const service = new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
        });
        service.enableDeploymentAlarms([myAlarm.alarmName]);

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: true,
              Rollback: true,
              AlarmNames: [myAlarm.alarmName],
            },
          },
        });
      });

      test('use enableDeploymentAlarms() and explicitly set behavior to ROLLBACK_ON_ALARM', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');

        const service = new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
        });
        service.enableDeploymentAlarms([myAlarm.alarmName], {
          behavior: AlarmBehavior.ROLLBACK_ON_ALARM,
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: true,
              Rollback: true,
              AlarmNames: [myAlarm.alarmName],
            },
          },
        });
      });

      test('use enableDeploymentAlarms() and explicitly set behavior to FAIL_ON_ALARM', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');
        const service = new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
        });
        service.enableDeploymentAlarms([myAlarm.alarmName], {
          behavior: AlarmBehavior.FAIL_ON_ALARM,
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: true,
              Rollback: false,
              AlarmNames: [myAlarm.alarmName],
            },
          },
        });
      });

      test('throw error if deploymentAlarms is specified with a non-ECS DeploymentControllerType', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            deploymentController: {
              type: DeploymentControllerType.EXTERNAL,
            },
            deploymentAlarms: {
              alarmNames: [myAlarm.alarmName],
            },
          });
        }).toThrow('Deployment alarms requires the ECS deployment controller.');
      });

      test('mixing alarm behaviors throws errors', () => {
        const myAlarm = cloudwatch.Alarm.fromAlarmArn(stack, 'myAlarm', 'arn:aws:cloudwatch:us-east-1:1234567890:alarm:alarm1');
        const service = new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
          deploymentAlarms: {
            alarmNames: [myAlarm.alarmName],
            behavior: AlarmBehavior.ROLLBACK_ON_ALARM,
          },
        });
        expect(() => {
          service.enableDeploymentAlarms([myAlarm.alarmName], { behavior: AlarmBehavior.FAIL_ON_ALARM });
        }).toThrow('all deployment alarms on an ECS service must have the same AlarmBehavior. Attempted to enable deployment alarms with FAIL_ON_ALARM, but alarms were previously enabled with ROLLBACK_ON_ALARM');
        const anotherService = new ecs.Ec2Service(stack, 'Ec2Service2', {
          cluster,
          taskDefinition,
          deploymentAlarms: {
            alarmNames: [myAlarm.alarmName],
            behavior: AlarmBehavior.FAIL_ON_ALARM,
          },
        });
        expect(() => {
          anotherService.enableDeploymentAlarms([myAlarm.alarmName], { behavior: AlarmBehavior.ROLLBACK_ON_ALARM });
        }).toThrow('all deployment alarms on an ECS service must have the same AlarmBehavior. Attempted to enable deployment alarms with ROLLBACK_ON_ALARM, but alarms were previously enabled with FAIL_ON_ALARM');
      });

      test('empty array of alarm names is not allowed', () => {
        expect(() => {
          new ecs.Ec2Service(stack, 'Ec2Service', {
            cluster,
            taskDefinition,
            deploymentAlarms: {
              alarmNames: [],
            },
          });
        }).toThrow('at least one alarm name is required when specifying deploymentAlarms, received empty array');

        const service = new ecs.Ec2Service(stack, 'AnotherEc2Service', {
          cluster,
          taskDefinition,
        });
        expect(() => service.enableDeploymentAlarms([])).toThrow('at least one alarm name is required when calling enableDeploymentAlarms(), received empty array');
      });

      test('no deployment alarms configured', () => {
        new ecs.Ec2Service(stack, 'Ec2Service', {
          cluster,
          taskDefinition,
        });

        Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: {
              Enable: false,
              Rollback: false,
              AlarmNames: [],
            },
          },
        });
      });

      test('no deployment alarms configured in gov cloud', () => {
        const app = new cdk.App();
        const govCloudStack = new cdk.Stack(app, 'GovStack', {
          env: { region: 'us-gov-east-1' },
        });
        const vpc = new ec2.Vpc(govCloudStack, 'MyVpc', {});
        const gcCluster = new ecs.Cluster(govCloudStack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(gcCluster, govCloudStack, vpc);
        const gcTaskDefinition = new ecs.Ec2TaskDefinition(govCloudStack, 'Ec2TaskDef');

        gcTaskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });
        new ecs.Ec2Service(govCloudStack, 'Ec2Service', {
          cluster: gcCluster,
          taskDefinition: gcTaskDefinition,
        });

        Template.fromStack(govCloudStack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: Match.absent(),
          },
        });
      });

      test('no deployment alarms in isolated partitions', () => {
        const app = new cdk.App();
        const govCloudStack = new cdk.Stack(app, 'IsoStack', {
          env: { region: 'us-isob-east-1' },
        });
        const vpc = new ec2.Vpc(govCloudStack, 'MyVpc', {});
        const gcCluster = new ecs.Cluster(govCloudStack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(gcCluster, govCloudStack, vpc);
        const gcTaskDefinition = new ecs.Ec2TaskDefinition(govCloudStack, 'Ec2TaskDef');

        gcTaskDefinition.addContainer('web', {
          image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
          memoryLimitMiB: 512,
        });
        new ecs.Ec2Service(govCloudStack, 'Ec2Service', {
          cluster: gcCluster,
          taskDefinition: gcTaskDefinition,
        });

        Template.fromStack(govCloudStack).hasResourceProperties('AWS::ECS::Service', {
          DeploymentConfiguration: {
            Alarms: Match.absent(),
          },
        });
      });

      /**
       * This section of tests test all combinations of the following possible
       * alarm names and metrics. Most combinations work just fine, some
       * combinations could cause a circular dependency and will have an info
       * annotation for the user.
       * NAME:
       *   - name is undefined, so it is a token referencing its own logical id
       *   - contains a token referencing the service
       *   - contains a token referencing another resource
       *   - hardcoded
       * METRIC:
       *   - contains a token referencing the service
       *   - contains a token referencing another resource
       *   - hardcoded
       *
       * The tests might seem repetitive because the implementation is not fully
       * able to detect the alarm <-> service circular dependency. Keeping these
       * tests in place to make it easier to validate a future implementation that
       * does have proper errors for the alarm <-> service cycle.
       */
      describe('circular dependency tests', () => {
        let service: ecs.Ec2Service;
        function infoMessage(alarmName: string, serviceId: string): string {
          return `Deployment alarm (${alarmName}) enabled on ${serviceId} may cause a circular dependency error when this stack deploys. The alarm name references the alarm's logical id, or another resource. See the 'Deployment alarms' section in the module README for more details.`;
        }
        beforeEach(() => {
          service = new ecs.Ec2Service(stack, 'EC2Service', {
            cluster,
            taskDefinition,
          });
        });
        test ('alarm name is undefined and alarm metric references service', () => {
          // This configuration will fail deployment
          const metric = service.metricCpuUtilization();
          const alarm = new cloudwatch.Alarm(stack, 'MyAlarm', {
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarm.alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarm.alarmName)), service.node.id)));
        });
        test ('alarm name is undefined and alarm metric references other resource', () => {
          // This will succeed deployment, but we still have an info message because we can't tell it apart from scenarios that will fail deployment
          const metric = cluster.metricMemoryUtilization();
          const alarm = new cloudwatch.Alarm(stack, 'MyAlarm', {
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarm.alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarm.alarmName)), service.node.id)));
        });
        test ('alarm name is undefined and alarm metric is hardcoded', () => {
          // This will succeed deployment, but we still have an info message because we can't tell it apart from scenarios that will fail deployment
          const metric = new cloudwatch.Metric({ namespace: 'AWS/ECS', metricName: 'CustomMetric' });
          const alarm = new cloudwatch.Alarm(stack, 'MyAlarm', {
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarm.alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarm.alarmName)), service.node.id)));
        });
        test ('alarm name references the service', () => {
          // This configuration will fail deployment
          const alarmName = `${service.serviceName}Alarm`;
          const metric = new cloudwatch.Metric({ namespace: 'CustomNamespace', metricName: 'CustomMetric' });
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarmName)), service.node.id)));
        });
        test ('alarm name references other resource and alarm metric references service', () => {
          // This will succeed deployment, but we still have an info message because we can't tell it apart from scenarios that will fail deployment
          const alarmName = `${cluster.clusterName}ServiceCpuAlarm`;
          const metric = service.metricCpuUtilization();
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarmName)), service.node.id)));
        });
        test ('alarm name and metric reference other resource', () => {
          // This will succeed deployment, but we still have an info message because we can't tell it apart from scenarios that will fail deployment
          const alarmName = `${cluster.clusterName}Alarm`;
          const metric = cluster.metricMemoryUtilization();
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarmName)), service.node.id)));
        });
        test ('alarm name references other resource and alarm metric is hardcoded', () => {
          // This will succeed deployment, but we still have an info message because we can't tell it apart from scenarios that will fail deployment
          const alarmName = `${cluster.clusterName}Alarm`;
          const metric = new cloudwatch.Metric({ namespace: 'CustomNamespace', metricName: 'CustomMetric' });
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasInfo(`/${service.node.path}`, Match.exact(infoMessage(JSON.stringify(stack.resolve(alarmName)), service.node.id)));
        });
        test ('alarm name is hardcoded and alarm metric references service', () => {
          // This will succeed deployment, and we know this during synthesis, so there is no info Annotation about circular dependency errors
          const alarmName = 'MyAlarm';
          const metric = service.metricCpuUtilization();
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasNoInfo(`/${service.node.path}`, Match.stringLikeRegexp('circular dependency error'));
        });
        test ('alarm name is hardcoded and alarm metric references other resource', () => {
          // This will succeed deployment, and we know this during synthesis, so there is no info Annotation about circular dependency errors
          const alarmName = 'MyAlarm';
          const metric = cluster.metricMemoryUtilization();
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasNoInfo(`/${service.node.path}`, Match.stringLikeRegexp('circular dependency error'));
        });
        test ('alarm name and metric are hardcoded', () => {
          // This will succeed deployment, and we know this during synthesis, so there is no info Annotation about circular dependency errors
          const alarmName = 'MyAlarm';
          const metric = new cloudwatch.Metric( { namespace: 'CustomNamespace', metricName: 'CustomMetric' } );
          new cloudwatch.Alarm(stack, 'MyAlarm', {
            alarmName,
            metric,
            evaluationPeriods: 5,
            threshold: 2,
          });
          service.enableDeploymentAlarms([alarmName]);
          Annotations.fromStack(stack).hasNoInfo(`/${service.node.path}`, Match.stringLikeRegexp('circular dependency error'));
        });
      });
    });
  });

  describe('attachToClassicLB', () => {
    test('allows network mode of task definition to be host', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const cluster = new ecs.Cluster(stack, 'Cluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TD', { networkMode: ecs.NetworkMode.HOST });
      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryLimitMiB: 1024,
      });
      container.addPortMappings({ containerPort: 808 });
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });
      service.attachToClassicLB(lb);
    });

    test('allows network mode of task definition to be bridge', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const cluster = new ecs.Cluster(stack, 'Cluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TD', { networkMode: ecs.NetworkMode.BRIDGE });
      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryLimitMiB: 1024,
      });
      container.addPortMappings({ containerPort: 808 });
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });
      service.attachToClassicLB(lb);
    });

    test('throws when network mode of task definition is AwsVpc', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const cluster = new ecs.Cluster(stack, 'Cluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TD', { networkMode: ecs.NetworkMode.AWS_VPC });
      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryLimitMiB: 1024,
      });
      container.addPortMappings({ containerPort: 808 });
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });
      expect(() => {
        service.attachToClassicLB(lb);
      }).toThrow(/Cannot use a Classic Load Balancer if NetworkMode is AwsVpc. Use Host or Bridge instead./);
    });

    test('throws when network mode of task definition is none', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const cluster = new ecs.Cluster(stack, 'Cluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TD', { networkMode: ecs.NetworkMode.NONE });
      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryLimitMiB: 1024,
      });
      container.addPortMappings({ containerPort: 808 });
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // THEN
      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });
      expect(() => {
        service.attachToClassicLB(lb);
      }).toThrow(/Cannot use a Classic Load Balancer if NetworkMode is None. Use Host or Bridge instead./);
    });

    test('throws when AvailabilityZoneRebalancing.ENABLED', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        availabilityZoneRebalancing: ecs.AvailabilityZoneRebalancing.ENABLED,
      });

      taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('amazon/amazon-ecs-sample'),
      });

      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });

      // THEN
      expect(() => {
        lb.addTarget(service);
      }).toThrow('AvailabilityZoneRebalancing.ENABLED disallows using the service as a target of a Classic Load Balancer');
    });
  });

  describe('attachToApplicationTargetGroup', () => {
    test('allows network mode of task definition to be other than none', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode: ecs.NetworkMode.AWS_VPC });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
      });

      // THEN
      service.attachToApplicationTargetGroup(targetGroup);
    });

    test('throws when network mode of task definition is none', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode: ecs.NetworkMode.NONE });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
      });

      // THEN
      expect(() => {
        service.attachToApplicationTargetGroup(targetGroup);
      }).toThrow(/Cannot use a load balancer if NetworkMode is None. Use Bridge, Host or AwsVpc instead./);
    });

    test('throws when the first port mapping added to the container does not expose a single port', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({
        containerPort: ecs.ContainerDefinition.CONTAINER_PORT_USE_RANGE,
        containerPortRange: '8000-8001',
      });

      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
      });

      // THEN
      expect(() => {
        service.attachToApplicationTargetGroup(targetGroup);
      }).toThrow(/The first port mapping of the container MainContainer must expose a single port./);
    });

    describe('correctly setting ingress and egress port', () => {
      test('with bridge/NAT network mode and 0 host port', () => {
        [ecs.NetworkMode.BRIDGE, ecs.NetworkMode.NAT].forEach((networkMode: ecs.NetworkMode) => {
          // GIVEN
          const stack = new cdk.Stack();
          acknowledgeTestValidationRules(stack);
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          addDefaultCapacityProvider(cluster, stack, vpc);
          cluster.connections.addSecurityGroup();
          const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode });
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
            memoryLimitMiB: 512,
          });
          container.addPortMappings({ containerPort: 8000 });
          container.addPortMappings({ containerPort: 8001 });

          const service = new ecs.Ec2Service(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });
          listener.addTargets('target', {
            port: 80,
            targets: [service.loadBalancerTarget({
              containerName: 'MainContainer',
              containerPort: 8001,
            })],
          });

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
            Description: 'Load balancer to target',
            FromPort: 32768,
            ToPort: 65535,
          });

          Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
            Description: 'Load balancer to target',
            FromPort: 32768,
            ToPort: 65535,
          });
        });
      });

      test('with bridge/NAT network mode and host port other than 0', () => {
        [ecs.NetworkMode.BRIDGE, ecs.NetworkMode.NAT].forEach((networkMode: ecs.NetworkMode) => {
          // GIVEN
          const stack = new cdk.Stack();
          const vpc = new ec2.Vpc(stack, 'MyVpc', {});
          const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
          addDefaultCapacityProvider(cluster, stack, vpc);
          const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode });
          const container = taskDefinition.addContainer('MainContainer', {
            image: ecs.ContainerImage.fromRegistry('hello'),
            memoryLimitMiB: 512,
          });
          container.addPortMappings({ containerPort: 8000 });
          container.addPortMappings({ containerPort: 8001, hostPort: 80 });

          const service = new ecs.Ec2Service(stack, 'Service', {
            cluster,
            taskDefinition,
          });

          // WHEN
          const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
          const listener = lb.addListener('listener', { port: 80 });
          listener.addTargets('target', {
            port: 80,
            targets: [service.loadBalancerTarget({
              containerName: 'MainContainer',
              containerPort: 8001,
            })],
          });

          // THEN
          Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
            Description: 'Load balancer to target',
            FromPort: 80,
            ToPort: 80,
          });

          Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
            Description: 'Load balancer to target',
            FromPort: 80,
            ToPort: 80,
          });
        });
      });

      test('with host network mode', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode: ecs.NetworkMode.HOST });
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
          memoryLimitMiB: 512,
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001 });

        const service = new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });
        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
            containerPort: 8001,
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
          Description: 'Load balancer to target',
          FromPort: 8001,
          ToPort: 8001,
        });

        Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
          Description: 'Load balancer to target',
          FromPort: 8001,
          ToPort: 8001,
        });
      });

      test('with aws_vpc network mode', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const vpc = new ec2.Vpc(stack, 'MyVpc', {});
        const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
        addDefaultCapacityProvider(cluster, stack, vpc);
        const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode: ecs.NetworkMode.AWS_VPC });
        const container = taskDefinition.addContainer('MainContainer', {
          image: ecs.ContainerImage.fromRegistry('hello'),
          memoryLimitMiB: 512,
        });
        container.addPortMappings({ containerPort: 8000 });
        container.addPortMappings({ containerPort: 8001 });

        const service = new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
        });

        // WHEN
        const lb = new elbv2.ApplicationLoadBalancer(stack, 'lb', { vpc });
        const listener = lb.addListener('listener', { port: 80 });
        listener.addTargets('target', {
          port: 80,
          targets: [service.loadBalancerTarget({
            containerName: 'MainContainer',
            containerPort: 8001,
          })],
        });

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
          Description: 'Load balancer to target',
          FromPort: 8001,
          ToPort: 8001,
        });

        Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
          Description: 'Load balancer to target',
          FromPort: 8001,
          ToPort: 8001,
        });
      });
    });
  });

  describe('attachToNetworkTargetGroup', () => {
    test('allows network mode of task definition to be other than none', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode: ecs.NetworkMode.AWS_VPC });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.NetworkLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
      });

      // THEN
      service.attachToNetworkTargetGroup(targetGroup);
    });

    test('throws when network mode of task definition is none', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', { networkMode: ecs.NetworkMode.NONE });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({ containerPort: 8000 });

      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.NetworkLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
      });

      // THEN
      expect(() => {
        service.attachToNetworkTargetGroup(targetGroup);
      }).toThrow(/Cannot use a load balancer if NetworkMode is None. Use Bridge, Host or AwsVpc instead./);
    });

    test('throws when the first port mapping added to the container does not expose a single port', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
      });
      container.addPortMappings({
        containerPort: ecs.ContainerDefinition.CONTAINER_PORT_USE_RANGE,
        containerPortRange: '8000-8001',
      });

      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      const lb = new elbv2.NetworkLoadBalancer(stack, 'lb', { vpc });
      const listener = lb.addListener('listener', { port: 80 });
      const targetGroup = listener.addTargets('target', {
        port: 80,
      });

      // THEN
      expect(() => {
        service.attachToNetworkTargetGroup(targetGroup);
      }).toThrow(/The first port mapping of the container MainContainer must expose a single port./);
    });
  });

  describe('classic ELB', () => {
    test('can attach to classic ELB', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const cluster = new ecs.Cluster(stack, 'Cluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TD', { networkMode: ecs.NetworkMode.HOST });
      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryLimitMiB: 1024,
      });
      container.addPortMappings({ containerPort: 808 });
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });
      lb.addTarget(service);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        LoadBalancers: [
          {
            ContainerName: 'web',
            ContainerPort: 808,
            LoadBalancerName: { Ref: 'LB8A12904C' },
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        // if any load balancer is configured and healthCheckGracePeriodSeconds is not
        // set, then it should default to 60 seconds.
        HealthCheckGracePeriodSeconds: 60,
      });
    });

    test('can attach any container and port as a target', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'VPC');
      const cluster = new ecs.Cluster(stack, 'Cluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'TD', { networkMode: ecs.NetworkMode.HOST });
      const container = taskDefinition.addContainer('web', {
        image: ecs.ContainerImage.fromRegistry('test'),
        memoryLimitMiB: 1024,
      });
      container.addPortMappings({ containerPort: 808 });
      container.addPortMappings({ containerPort: 8080 });
      const service = new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
      });

      // WHEN
      const lb = new elb.LoadBalancer(stack, 'LB', { vpc });
      lb.addTarget(service.loadBalancerTarget({
        containerName: 'web',
        containerPort: 8080,
      }));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        LoadBalancers: [
          {
            ContainerName: 'web',
            ContainerPort: 8080,
            LoadBalancerName: { Ref: 'LB8A12904C' },
          },
        ],
      });
    });
  });

  describe('When enabling service discovery', () => {
    test('throws if namespace has not been added to cluster', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      // default network mode is bridge
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            name: 'myApp',
          },
        });
      }).toThrow(/Cannot enable service discovery if a Cloudmap Namespace has not been created in the cluster./);
    });

    test('fails to enable Service Discovery with HTTP defaultCloudmapNamespace', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.NONE,
      });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      cluster.addDefaultCloudMapNamespace({ name: 'foo.com', type: cloudmap.NamespaceType.HTTP });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            name: 'myApp',
          },
        });
      }).toThrow(/Cannot enable DNS service discovery for HTTP Cloudmap Namespace./);
    });

    test('throws if network mode is none', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.NONE,
      });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      cluster.addDefaultCloudMapNamespace({ name: 'foo.com' });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            name: 'myApp',
          },
        });
      }).toThrow(/Cannot use a service discovery if NetworkMode is None. Use Bridge, Host or AwsVpc instead./);
    });

    test('creates AWS Cloud Map service for Private DNS namespace with bridge network mode', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      // default network mode is bridge
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            ContainerName: 'MainContainer',
            ContainerPort: 8000,
            RegistryArn: {
              'Fn::GetAtt': [
                'ServiceCloudmapService046058A4',
                'Arn',
              ],
            },
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'SRV',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('creates AWS Cloud Map service for Private DNS namespace with host network mode', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.HOST,
      });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            ContainerName: 'MainContainer',
            ContainerPort: 8000,
            RegistryArn: {
              'Fn::GetAtt': [
                'ServiceCloudmapService046058A4',
                'Arn',
              ],
            },
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'SRV',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('throws if wrong DNS record type specified with bridge network mode', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      // default network mode is bridge
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef');
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
      });

      // THEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            name: 'myApp',
            dnsRecordType: cloudmap.DnsRecordType.A,
          },
        });
      }).toThrow(/SRV records must be used when network mode is Bridge or Host./);
    });

    test('creates AWS Cloud Map service for Private DNS namespace with AwsVpc network mode', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.AWS_VPC,
      });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            RegistryArn: {
              'Fn::GetAtt': [
                'ServiceCloudmapService046058A4',
                'Arn',
              ],
            },
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'A',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('creates AWS Cloud Map service for Private DNS namespace with AwsVpc network mode with SRV records', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);

      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Ec2TaskDef', {
        networkMode: ecs.NetworkMode.AWS_VPC,
      });
      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      container.addPortMappings({ containerPort: 8000 });

      // WHEN
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });

      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          name: 'myApp',
          dnsRecordType: cloudmap.DnsRecordType.SRV,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            ContainerName: 'MainContainer',
            ContainerPort: 8000,
            RegistryArn: {
              'Fn::GetAtt': [
                'ServiceCloudmapService046058A4',
                'Arn',
              ],
            },
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::Service', {
        DnsConfig: {
          DnsRecords: [
            {
              TTL: 60,
              Type: 'SRV',
            },
          ],
          NamespaceId: {
            'Fn::GetAtt': [
              'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
              'Id',
            ],
          },
          RoutingPolicy: 'MULTIVALUE',
        },
        HealthCheckCustomConfig: {
          FailureThreshold: 1,
        },
        Name: 'myApp',
        NamespaceId: {
          'Fn::GetAtt': [
            'EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F',
            'Id',
          ],
        },
      });
    });

    test('user can select any container and port', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'FargateTaskDef', {
        networkMode: ecs.NetworkMode.BRIDGE,
      });

      const mainContainer = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      mainContainer.addPortMappings({ containerPort: 8000 });

      const otherContainer = taskDefinition.addContainer('OtherContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });
      otherContainer.addPortMappings({ containerPort: 8001 });

      // WHEN
      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          dnsRecordType: cloudmap.DnsRecordType.SRV,
          container: otherContainer,
          containerPort: 8001,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [
          {
            RegistryArn: { 'Fn::GetAtt': ['ServiceCloudmapService046058A4', 'Arn'] },
            ContainerName: 'OtherContainer',
            ContainerPort: 8001,
          },
        ],
      });
    });

    test('By default, the container name is the default', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Task', {
        networkMode: ecs.NetworkMode.BRIDGE,
      });

      taskDefinition.addContainer('main', {
        image: ecs.ContainerImage.fromRegistry('some'),
        memoryLimitMiB: 512,
      }).addPortMappings({ containerPort: 1234 });

      taskDefinition.addContainer('second', {
        image: ecs.ContainerImage.fromRegistry('some'),
        memoryLimitMiB: 512,
      }).addPortMappings({ containerPort: 4321 });

      // WHEN
      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {},
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [Match.objectLike({
          ContainerName: 'main',
          ContainerPort: Match.anyValue(),
        })],
      });
    });

    test('For SRV, by default, container name is default container and port is the default container port', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Task', {
        networkMode: ecs.NetworkMode.BRIDGE,
      });

      taskDefinition.addContainer('main', {
        image: ecs.ContainerImage.fromRegistry('some'),
        memoryLimitMiB: 512,
      }).addPortMappings({ containerPort: 1234 });

      taskDefinition.addContainer('second', {
        image: ecs.ContainerImage.fromRegistry('some'),
        memoryLimitMiB: 512,
      }).addPortMappings({ containerPort: 4321 });

      // WHEN
      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          dnsRecordType: cloudmap.DnsRecordType.SRV,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [Match.objectLike({
          ContainerName: 'main',
          ContainerPort: 1234,
        })],
      });
    });

    test('allows SRV service discovery to select the container and port', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Task', {
        networkMode: ecs.NetworkMode.BRIDGE,
      });

      taskDefinition.addContainer('main', {
        image: ecs.ContainerImage.fromRegistry('some'),
        memoryLimitMiB: 512,
      }).addPortMappings({ containerPort: 1234 });

      const secondContainer = taskDefinition.addContainer('second', {
        image: ecs.ContainerImage.fromRegistry('some'),
        memoryLimitMiB: 512,
      });
      secondContainer.addPortMappings({ containerPort: 4321 });

      // WHEN
      new ecs.Ec2Service(stack, 'Service', {
        cluster,
        taskDefinition,
        cloudMapOptions: {
          dnsRecordType: cloudmap.DnsRecordType.SRV,
          container: secondContainer,
          containerPort: 4321,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::Service', {
        ServiceRegistries: [Match.objectLike({
          ContainerName: 'second',
          ContainerPort: 4321,
        })],
      });
    });

    test('throws if SRV and container is not part of task definition', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Task', {
        networkMode: ecs.NetworkMode.BRIDGE,
      });

      // The right container
      taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });

      const wrongTaskDefinition = new ecs.Ec2TaskDefinition(stack, 'WrongTaskDef');
      // The wrong container
      const wrongContainer = wrongTaskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });

      // WHEN
      expect(() => {
        new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            dnsRecordType: cloudmap.DnsRecordType.SRV,
            container: wrongContainer,
            containerPort: 4321,
          },
        });
      }).toThrow(/another task definition/i);
    });

    test('throws if SRV and the container port is not mapped', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
      addDefaultCapacityProvider(cluster, stack, vpc);
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
        type: cloudmap.NamespaceType.DNS_PRIVATE,
      });
      const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'Task', {
        networkMode: ecs.NetworkMode.BRIDGE,
      });

      const container = taskDefinition.addContainer('MainContainer', {
        image: ecs.ContainerImage.fromRegistry('hello'),
        memoryLimitMiB: 512,
      });

      container.addPortMappings({ containerPort: 8000 });

      expect(() => {
        new ecs.Ec2Service(stack, 'Service', {
          cluster,
          taskDefinition,
          cloudMapOptions: {
            dnsRecordType: cloudmap.DnsRecordType.SRV,
            container: container,
            containerPort: 4321,
          },
        });
      }).toThrow(/container port.*not.*mapped/i);
    });
  });

  test('Metric', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});
    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    addDefaultCapacityProvider(cluster, stack, vpc);
    const taskDefinition = new ecs.Ec2TaskDefinition(stack, 'FargateTaskDef');
    taskDefinition.addContainer('Container', {
      image: ecs.ContainerImage.fromRegistry('hello'),
    });

    // WHEN
    const service = new ecs.Ec2Service(stack, 'Service', {
      cluster,
      taskDefinition,
    });

    // THEN
    expect(stack.resolve(service.metricMemoryUtilization())).toEqual({
      dimensions: {
        ClusterName: { Ref: 'EcsCluster97242B84' },
        ServiceName: { 'Fn::GetAtt': ['ServiceD69D759B', 'Name'] },
      },
      namespace: 'AWS/ECS',
      metricName: 'MemoryUtilization',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });

    expect(stack.resolve(service.metricCpuUtilization())).toEqual({
      dimensions: {
        ClusterName: { Ref: 'EcsCluster97242B84' },
        ServiceName: { 'Fn::GetAtt': ['ServiceD69D759B', 'Name'] },
      },
      namespace: 'AWS/ECS',
      metricName: 'CPUUtilization',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });
  });

  describe('When import an EC2 Service', () => {
    test('fromEc2ServiceArn old format', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const service = ecs.Ec2Service.fromEc2ServiceArn(stack, 'EcsService', 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service');

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');
    });

    test('fromEc2ServiceArn new format', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const service = ecs.Ec2Service.fromEc2ServiceArn(stack, 'EcsService', 'arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service');

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');
    });

    describe('fromEc2ServiceArn tokenized ARN', () => {
      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is disabled, use old ARN format', () => {
        // GIVEN
        const stack = new cdk.Stack();

        // WHEN
        const service = ecs.Ec2Service.fromEc2ServiceArn(stack, 'EcsService', new cdk.CfnParameter(stack, 'ARN').valueAsString);

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            1,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });
      });

      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is enabled, use new ARN format', () => {
        // GIVEN
        const app = new App({
          context: {
            [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: true,
          },
        });

        const stack = new cdk.Stack(app);

        // WHEN
        const service = ecs.Ec2Service.fromEc2ServiceArn(stack, 'EcsService', new cdk.CfnParameter(stack, 'ARN').valueAsString);

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            2,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });
      });
    });

    test('with serviceArn old format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN
      const service = ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
        serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service',
        cluster,
      });

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');

      expect(service.env.account).toEqual('123456789012');
      expect(service.env.region).toEqual('us-west-2');
    });

    test('with serviceArn new format', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN
      const service = ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
        serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service',
        cluster,
      });

      // THEN
      expect(service.serviceArn).toEqual('arn:aws:ecs:us-west-2:123456789012:service/my-cluster-name/my-http-service');
      expect(service.serviceName).toEqual('my-http-service');

      expect(service.env.account).toEqual('123456789012');
      expect(service.env.region).toEqual('us-west-2');
    });

    describe('with serviceArn tokenized ARN', () => {
      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is disabled, use old ARN format', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
          serviceArn: new cdk.CfnParameter(stack, 'ARN').valueAsString,
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            1,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });

        expect(stack.resolve(service.env.account)).toEqual({
          'Fn::Select': [
            4,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
        expect(stack.resolve(service.env.region)).toEqual({
          'Fn::Select': [
            3,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
      });

      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is enabled, use new ARN format', () => {
        // GIVEN
        const app = new App({
          context: {
            [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: true,
          },
        });
        const stack = new cdk.Stack(app);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
          serviceArn: new cdk.CfnParameter(stack, 'ARN').valueAsString,
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual({ Ref: 'ARN' });
        expect(stack.resolve(service.serviceName)).toEqual({
          'Fn::Select': [
            2,
            {
              'Fn::Split': [
                '/',
                {
                  'Fn::Select': [
                    5,
                    {
                      'Fn::Split': [
                        ':',
                        { Ref: 'ARN' },
                      ],
                    },
                  ],
                },
              ],
            },
          ],
        });

        expect(stack.resolve(service.env.account)).toEqual({
          'Fn::Select': [
            4,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
        expect(stack.resolve(service.env.region)).toEqual({
          'Fn::Select': [
            3,
            {
              'Fn::Split': [
                ':',
                { Ref: 'ARN' },
              ],
            },
          ],
        });
      });
    });

    describe('with serviceName', () => {
      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is disabled, use old ARN format', () => {
        // GIVEN
        const stack = new cdk.Stack();
        const pseudo = new cdk.ScopedAws(stack);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
          serviceName: 'my-http-service',
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual(stack.resolve(`arn:${pseudo.partition}:ecs:${pseudo.region}:${pseudo.accountId}:service/my-http-service`));
        expect(service.serviceName).toEqual('my-http-service');
      });

      test('when @aws-cdk/aws-ecs:arnFormatIncludesClusterName is enabled, use new ARN format', () => {
        // GIVEN
        const app = new App({
          context: {
            [ECS_ARN_FORMAT_INCLUDES_CLUSTER_NAME]: true,
          },
        });
        const stack = new cdk.Stack(app);
        const pseudo = new cdk.ScopedAws(stack);
        const cluster = new ecs.Cluster(stack, 'EcsCluster');

        // WHEN
        const service = ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
          serviceName: 'my-http-service',
          cluster,
        });

        // THEN
        expect(stack.resolve(service.serviceArn)).toEqual(stack.resolve(`arn:${pseudo.partition}:ecs:${pseudo.region}:${pseudo.accountId}:service/${cluster.clusterName}/my-http-service`));
        expect(service.serviceName).toEqual('my-http-service');
      });
    });

    test('throws an exception if both serviceArn and serviceName were provided for fromEc2ServiceAttributes', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      expect(() => {
        ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
          serviceArn: 'arn:aws:ecs:us-west-2:123456789012:service/my-http-service',
          serviceName: 'my-http-service',
          cluster,
        });
      }).toThrow(/only specify either serviceArn or serviceName/);
    });

    test('throws an exception if neither serviceArn nor serviceName were provided for fromEc2ServiceAttributes', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      expect(() => {
        ecs.Ec2Service.fromEc2ServiceAttributes(stack, 'EcsService', {
          cluster,
        });
      }).toThrow(/only specify either serviceArn or serviceName/);
    });
  });
});
