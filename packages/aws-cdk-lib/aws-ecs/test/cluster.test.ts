
import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Annotations, Match, Template } from '../../assertions';
import * as autoscaling from '../../aws-autoscaling';
import * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import * as logs from '../../aws-logs';
import * as s3 from '../../aws-s3';
import * as cloudmap from '../../aws-servicediscovery';
import * as cdk from '../../core';
import * as cxapi from '../../cx-api';
import * as ecs from '../lib';
import { acknowledgeTestValidationRules } from './util';

describe('cluster', () => {
  describe('isCluster() returns', () => {
    test('true if given cluster instance', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      // WHEN
      const createdCluster = new ecs.Cluster(stack, 'EcsCluster');
      // THEN
      expect(ecs.Cluster.isCluster(createdCluster)).toBe(true);
    });

    test('false if given imported cluster instance', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const importedSg = ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG1', 'sg-1', { allowAllOutbound: false });
      // WHEN
      const importedCluster = ecs.Cluster.fromClusterAttributes(stack, 'Cluster', {
        clusterName: 'cluster-name',
        securityGroups: [importedSg],
        vpc,
      });
      // THEN
      expect(ecs.Cluster.isCluster(importedCluster)).toBe(false);
    });

    test('false if given undefined', () => {
      // THEN
      expect(ecs.Cluster.isCluster(undefined)).toBe(false);
    });
  });

  describe('When creating an ECS Cluster', () => {
    testDeprecated('with no properties set, it correctly sets default properties', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      cluster.addCapacity('DefaultAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
      });

      Template.fromStack(stack).resourceCountIs('AWS::ECS::Cluster', 1);

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
        CidrBlock: '10.0.0.0/16',
        EnableDnsHostnames: true,
        EnableDnsSupport: true,
        InstanceTenancy: ec2.DefaultInstanceTenancy.DEFAULT,
        Tags: [
          {
            Key: 'Name',
            Value: 'Default/EcsCluster/Vpc',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
        ImageId: {
          Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2recommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
        },
        InstanceType: 't2.micro',
        IamInstanceProfile: {
          Ref: 'EcsClusterDefaultAutoScalingGroupInstanceProfile2CE606B3',
        },
        SecurityGroups: [
          {
            'Fn::GetAtt': [
              'EcsClusterDefaultAutoScalingGroupInstanceSecurityGroup912E1231',
              'GroupId',
            ],
          },
        ],
        UserData: {
          'Fn::Base64': {
            'Fn::Join': [
              '',
              [
                '#!/bin/bash\necho ECS_CLUSTER=',
                {
                  Ref: 'EcsCluster97242B84',
                },

                ' >> /etc/ecs/ecs.config',
              ],
            ],
          },
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
        MaxSize: '1',
        MinSize: '1',
        LaunchConfigurationName: {
          Ref: 'EcsClusterDefaultAutoScalingGroupLaunchConfigB7E376C1',
        },
        Tags: [
          {
            Key: 'Name',
            PropagateAtLaunch: true,
            Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
          },
        ],
        VPCZoneIdentifier: [
          {
            Ref: 'EcsClusterVpcPrivateSubnet1SubnetFAB0E487',
          },
          {
            Ref: 'EcsClusterVpcPrivateSubnet2SubnetC2B7B1BA',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Default/EcsCluster/DefaultAutoScalingGroup/InstanceSecurityGroup',
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        Tags: [
          {
            Key: 'Name',
            Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
          },
        ],
        VpcId: {
          Ref: 'EcsClusterVpc779914AB',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ec2.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ecs:DeregisterContainerInstance',
                'ecs:RegisterContainerInstance',
                'ecs:Submit*',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'EcsCluster97242B84',
                  'Arn',
                ],
              },
            },
            {
              Action: [
                'ecs:Poll',
                'ecs:StartTelemetrySession',
              ],
              Effect: 'Allow',
              Resource: '*',
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
            },
            {
              Action: [
                'ecs:DiscoverPollEndpoint',
                'ecr:GetAuthorizationToken',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    testDeprecated('with only vpc set, it correctly sets default properties', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
      });

      cluster.addCapacity('DefaultAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
      });

      Template.fromStack(stack).resourceCountIs('AWS::ECS::Cluster', 1);

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
        CidrBlock: '10.0.0.0/16',
        EnableDnsHostnames: true,
        EnableDnsSupport: true,
        InstanceTenancy: ec2.DefaultInstanceTenancy.DEFAULT,
        Tags: [
          {
            Key: 'Name',
            Value: 'Default/MyVpc',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
        ImageId: {
          Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2recommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
        },
        InstanceType: 't2.micro',
        IamInstanceProfile: {
          Ref: 'EcsClusterDefaultAutoScalingGroupInstanceProfile2CE606B3',
        },
        SecurityGroups: [
          {
            'Fn::GetAtt': [
              'EcsClusterDefaultAutoScalingGroupInstanceSecurityGroup912E1231',
              'GroupId',
            ],
          },
        ],
        UserData: {
          'Fn::Base64': {
            'Fn::Join': [
              '',
              [
                '#!/bin/bash\necho ECS_CLUSTER=',
                {
                  Ref: 'EcsCluster97242B84',
                },

                ' >> /etc/ecs/ecs.config',
              ],
            ],
          },
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
        MaxSize: '1',
        MinSize: '1',
        LaunchConfigurationName: {
          Ref: 'EcsClusterDefaultAutoScalingGroupLaunchConfigB7E376C1',
        },
        Tags: [
          {
            Key: 'Name',
            PropagateAtLaunch: true,
            Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
          },
        ],
        VPCZoneIdentifier: [
          {
            Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
          },
          {
            Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Default/EcsCluster/DefaultAutoScalingGroup/InstanceSecurityGroup',
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        Tags: [
          {
            Key: 'Name',
            Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ec2.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ecs:DeregisterContainerInstance',
                'ecs:RegisterContainerInstance',
                'ecs:Submit*',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'EcsCluster97242B84',
                  'Arn',
                ],
              },
            },
            {
              Action: [
                'ecs:Poll',
                'ecs:StartTelemetrySession',
              ],
              Effect: 'Allow',
              Resource: '*',
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
            },
            {
              Action: [
                'ecs:DiscoverPollEndpoint',
                'ecr:GetAuthorizationToken',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });

    testDeprecated('multiple clusters with default capacity', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});

      // WHEN
      for (let i = 0; i < 2; i++) {
        const cluster = new ecs.Cluster(stack, `EcsCluster${i}`, { vpc });
        cluster.addCapacity('MyCapacity', {
          instanceType: new ec2.InstanceType('m3.medium'),
        });
      }
    });

    testDeprecated('lifecycle hook is automatically added @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy enabled', () => {
      // GIVEN
      const app = new cdk.App({
        context: {
          [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: true,
        },
      });
      const stack = new cdk.Stack(app);
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
      });

      // WHEN
      cluster.addCapacity('DefaultAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LifecycleHook', {
        AutoScalingGroupName: { Ref: 'EcsClusterDefaultAutoScalingGroupASGC1A785DB' },
        LifecycleTransition: 'autoscaling:EC2_INSTANCE_TERMINATING',
        DefaultResult: 'CONTINUE',
        HeartbeatTimeout: 300,
        NotificationTargetARN: { Ref: 'EcsClusterDefaultAutoScalingGroupLifecycleHookDrainHookTopicACD2D4A4' },
        RoleARN: { 'Fn::GetAtt': ['EcsClusterDefaultAutoScalingGroupLifecycleHookDrainHookRoleA38EC83B', 'Arn'] },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Timeout: 310,
        Environment: {
          Variables: {
            CLUSTER: {
              Ref: 'EcsCluster97242B84',
            },
          },
        },
        Handler: 'index.lambda_handler',
        Runtime: 'python3.13',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ec2:DescribeInstances',
                'ec2:DescribeInstanceAttribute',
                'ec2:DescribeInstanceStatus',
                'ec2:DescribeHosts',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'EcsClusterDefaultAutoScalingGroupDrainECSHookFunctioninlinePolicyAddedToExecutionRole075025F00',
        Roles: [
          {
            Ref: 'EcsClusterDefaultAutoScalingGroupDrainECSHookFunctionServiceRole94543EDA',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: 'autoscaling:CompleteLifecycleAction',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':autoscaling:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':autoScalingGroup:*:autoScalingGroupName/',
                    {
                      Ref: 'EcsClusterDefaultAutoScalingGroupASGC1A785DB',
                    },
                  ],
                ],
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ecs:DescribeContainerInstances',
                'ecs:DescribeTasks',
              ],
              Effect: 'Allow',
              Resource: '*',
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ecs:ListContainerInstances',
                'ecs:SubmitContainerStateChange',
                'ecs:SubmitTaskStateChange',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'EcsCluster97242B84',
                  'Arn',
                ],
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ecs:UpdateContainerInstancesState',
                'ecs:ListTasks',
              ],
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
              Effect: 'Allow',
              Resource: '*',
            },
          ],
        },
      });
    });

    testDeprecated('lifecycle hook is automatically added @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy disabled', () => {
      // GIVEN
      const app = new cdk.App({
        context: {
          [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: false,
        },
      });
      const stack = new cdk.Stack(app);
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
      });

      // WHEN
      cluster.addCapacity('DefaultAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LifecycleHook', {
        AutoScalingGroupName: { Ref: 'EcsClusterDefaultAutoScalingGroupASGC1A785DB' },
        LifecycleTransition: 'autoscaling:EC2_INSTANCE_TERMINATING',
        DefaultResult: 'CONTINUE',
        HeartbeatTimeout: 300,
        NotificationTargetARN: { Ref: 'EcsClusterDefaultAutoScalingGroupLifecycleHookDrainHookTopicACD2D4A4' },
        RoleARN: { 'Fn::GetAtt': ['EcsClusterDefaultAutoScalingGroupLifecycleHookDrainHookRoleA38EC83B', 'Arn'] },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Timeout: 310,
        Environment: {
          Variables: {
            CLUSTER: {
              Ref: 'EcsCluster97242B84',
            },
          },
        },
        Handler: 'index.lambda_handler',
        Runtime: 'python3.13',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ec2:DescribeInstances',
                'ec2:DescribeInstanceAttribute',
                'ec2:DescribeInstanceStatus',
                'ec2:DescribeHosts',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: 'autoscaling:CompleteLifecycleAction',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':autoscaling:',
                    {
                      Ref: 'AWS::Region',
                    },
                    ':',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':autoScalingGroup:*:autoScalingGroupName/',
                    {
                      Ref: 'EcsClusterDefaultAutoScalingGroupASGC1A785DB',
                    },
                  ],
                ],
              },
            },
            {
              Action: [
                'ecs:DescribeContainerInstances',
                'ecs:DescribeTasks',
              ],
              Effect: 'Allow',
              Resource: '*',
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
            },
            {
              Action: [
                'ecs:ListContainerInstances',
                'ecs:SubmitContainerStateChange',
                'ecs:SubmitTaskStateChange',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'EcsCluster97242B84',
                  'Arn',
                ],
              },
            },
            {
              Action: [
                'ecs:UpdateContainerInstancesState',
                'ecs:ListTasks',
              ],
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
              Effect: 'Allow',
              Resource: '*',
            },
          ],
        },
      });
    });

    testDeprecated('lifecycle hook with encrypted SNS is added correctly', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
      });
      const key = new kms.Key(stack, 'Key');

      // WHEN
      cluster.addCapacity('DefaultAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
        topicEncryptionKey: key,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::SNS::Topic', {
        KmsMasterKeyId: {
          'Fn::GetAtt': [
            'Key961B73FD',
            'Arn',
          ],
        },
      });
    });

    testDeprecated('with capacity and cloudmap namespace properties set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'MyVpc', {});
      new ecs.Cluster(stack, 'EcsCluster', {
        vpc,
        capacity: {
          instanceType: new ec2.InstanceType('t2.micro'),
        },
        defaultCloudMapNamespace: {
          name: 'foo.com',
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::PrivateDnsNamespace', {
        Name: 'foo.com',
        Vpc: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });

      Template.fromStack(stack).resourceCountIs('AWS::ECS::Cluster', 1);

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
        CidrBlock: '10.0.0.0/16',
        EnableDnsHostnames: true,
        EnableDnsSupport: true,
        InstanceTenancy: ec2.DefaultInstanceTenancy.DEFAULT,
        Tags: [
          {
            Key: 'Name',
            Value: 'Default/MyVpc',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
        ImageId: {
          Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2recommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
        },
        InstanceType: 't2.micro',
        IamInstanceProfile: {
          Ref: 'EcsClusterDefaultAutoScalingGroupInstanceProfile2CE606B3',
        },
        SecurityGroups: [
          {
            'Fn::GetAtt': [
              'EcsClusterDefaultAutoScalingGroupInstanceSecurityGroup912E1231',
              'GroupId',
            ],
          },
        ],
        UserData: {
          'Fn::Base64': {
            'Fn::Join': [
              '',
              [
                '#!/bin/bash\necho ECS_CLUSTER=',
                {
                  Ref: 'EcsCluster97242B84',
                },

                ' >> /etc/ecs/ecs.config',
              ],
            ],
          },
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
        MaxSize: '1',
        MinSize: '1',
        LaunchConfigurationName: {
          Ref: 'EcsClusterDefaultAutoScalingGroupLaunchConfigB7E376C1',
        },
        Tags: [
          {
            Key: 'Name',
            PropagateAtLaunch: true,
            Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
          },
        ],
        VPCZoneIdentifier: [
          {
            Ref: 'MyVpcPrivateSubnet1Subnet5057CF7E',
          },
          {
            Ref: 'MyVpcPrivateSubnet2Subnet0040C983',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        GroupDescription: 'Default/EcsCluster/DefaultAutoScalingGroup/InstanceSecurityGroup',
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
        ],
        Tags: [
          {
            Key: 'Name',
            Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
          },
        ],
        VpcId: {
          Ref: 'MyVpcF9F0CA6F',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ec2.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'ecs:DeregisterContainerInstance',
                'ecs:RegisterContainerInstance',
                'ecs:Submit*',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'EcsCluster97242B84',
                  'Arn',
                ],
              },
            },
            {
              Action: [
                'ecs:Poll',
                'ecs:StartTelemetrySession',
              ],
              Effect: 'Allow',
              Resource: '*',
              Condition: {
                ArnEquals: {
                  'ecs:cluster': {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              },
            },
            {
              Action: [
                'ecs:DiscoverPollEndpoint',
                'ecr:GetAuthorizationToken',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
      });
    });
  });

  testDeprecated('allows specifying instance type', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('m3.large'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      InstanceType: 'm3.large',
    });
  });

  testDeprecated('allows specifying cluster size', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      desiredCapacity: 3,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MaxSize: '3',
    });
  });

  testDeprecated('configures userdata with powershell if windows machine image is specified', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('WindowsAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: new ecs.EcsOptimizedAmi({
        windowsVersion: ecs.WindowsOptimizedVersion.SERVER_2019,
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsserviceamiwindowslatestWindowsServer2019EnglishFullECSOptimizedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
      InstanceType: 't2.micro',
      IamInstanceProfile: {
        Ref: 'EcsClusterWindowsAutoScalingGroupInstanceProfile65DFA6BB',
      },
      SecurityGroups: [
        {
          'Fn::GetAtt': [
            'EcsClusterWindowsAutoScalingGroupInstanceSecurityGroupDA468DF1',
            'GroupId',
          ],
        },
      ],
      UserData: {
        'Fn::Base64': {
          'Fn::Join': [
            '',
            [
              '<powershell>Remove-Item -Recurse C:\\ProgramData\\Amazon\\ECS\\Cache\nImport-Module ECSTools\n[Environment]::SetEnvironmentVariable("ECS_CLUSTER", "',
              {
                Ref: 'EcsCluster97242B84',
              },
              "\", \"Machine\")\n[Environment]::SetEnvironmentVariable(\"ECS_ENABLE_AWSLOGS_EXECUTIONROLE_OVERRIDE\", \"true\", \"Machine\")\n[Environment]::SetEnvironmentVariable(\"ECS_AVAILABLE_LOGGING_DRIVERS\", '[\"json-file\",\"awslogs\"]', \"Machine\")\nInitialize-ECSAgent -Cluster '",
              {
                Ref: 'EcsCluster97242B84',
              },
              "'</powershell>",
            ],
          ],
        },
      },
    });
  });

  /*
   * TODO:v2.0.0 BEGINNING OF OBSOLETE BLOCK
   */
  testDeprecated('allows specifying special HW AMI Type', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('GpuAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: new ecs.EcsOptimizedAmi({
        hardwareType: ecs.AmiHardwareType.GPU,
      }),
    });

    // THEN
    const assembly = app.synth();
    const template = assembly.getStackByName(stack.stackName).template;
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2gpurecommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
    });

    expect(template.Parameters).toEqual({
      SsmParameterValueawsserviceecsoptimizedamiamazonlinux2gpurecommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter: {
        Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
        Default: '/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id',
      },
    });
  });

  testDeprecated('errors if amazon linux given with special HW type', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // THEN
    expect(() => {
      cluster.addCapacity('GpuAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: new ecs.EcsOptimizedAmi({
          generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX,
          hardwareType: ecs.AmiHardwareType.GPU,
        }),
      });
    }).toThrow(/Amazon Linux does not support special hardware type/);
  });

  testDeprecated('allows specifying windows image', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('WindowsAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: new ecs.EcsOptimizedAmi({
        windowsVersion: ecs.WindowsOptimizedVersion.SERVER_2019,
      }),
    });

    // THEN
    const assembly = app.synth();
    const template = assembly.getStackByName(stack.stackName).template;
    expect(template.Parameters).toEqual({
      SsmParameterValueawsserviceamiwindowslatestWindowsServer2019EnglishFullECSOptimizedimageidC96584B6F00A464EAD1953AFF4B05118Parameter: {
        Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
        Default: '/aws/service/ami-windows-latest/Windows_Server-2019-English-Full-ECS_Optimized/image_id',
      },
    });
  });

  testDeprecated('errors if windows given with special HW type', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // THEN
    expect(() => {
      cluster.addCapacity('WindowsGpuAutoScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: new ecs.EcsOptimizedAmi({
          windowsVersion: ecs.WindowsOptimizedVersion.SERVER_2019,
          hardwareType: ecs.AmiHardwareType.GPU,
        }),
      });
    }).toThrow(/Windows Server does not support special hardware type/);
  });

  testDeprecated('errors if windowsVersion and linux generation are set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // THEN
    expect(() => {
      cluster.addCapacity('WindowsScalingGroup', {
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: new ecs.EcsOptimizedAmi({
          windowsVersion: ecs.WindowsOptimizedVersion.SERVER_2019,
          generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX,
        }),
      });
    }).toThrow(/"windowsVersion" and Linux image "generation" cannot be both set/);
  });

  testDeprecated('allows returning the correct image for windows for EcsOptimizedAmi', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const ami = new ecs.EcsOptimizedAmi({
      windowsVersion: ecs.WindowsOptimizedVersion.SERVER_2019,
    });

    expect(ami.getImage(stack).osType).toEqual(ec2.OperatingSystemType.WINDOWS);
  });

  testDeprecated('allows returning the correct image for linux for EcsOptimizedAmi', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const ami = new ecs.EcsOptimizedAmi({
      generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX,
    });

    expect(ami.getImage(stack).osType).toEqual(ec2.OperatingSystemType.LINUX);
  });

  testDeprecated('allows returning the correct image for linux 2 for EcsOptimizedAmi', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const ami = new ecs.EcsOptimizedAmi({
      generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
    });

    expect(ami.getImage(stack).osType).toEqual(ec2.OperatingSystemType.LINUX);
  });

  testDeprecated('allows returning the correct image for linux 2023 for EcsOptimizedAmi', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const ami = new ecs.EcsOptimizedAmi({
      generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2023,
    });

    expect(ami.getImage(stack).osType).toEqual(ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for linux for EcsOptimizedImage', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.amazonLinux().getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for linux 2 for EcsOptimizedImage', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.amazonLinux2().getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for linux 2 for EcsOptimizedImage with ARM hardware', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.ARM).getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for linux 2 for EcsOptimizedImage with Neuron hardware', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.NEURON).getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for linux 2023 for EcsOptimizedImage', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.amazonLinux2023().getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for linux 2023 for EcsOptimizedImage with ARM hardware', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.amazonLinux2023(ecs.AmiHardwareType.ARM).getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.LINUX);
  });

  test('allows returning the correct image for windows for EcsOptimizedImage', () => {
    // GIVEN
    const stack = new cdk.Stack();

    expect(ecs.EcsOptimizedImage.windows(ecs.WindowsOptimizedVersion.SERVER_2019).getImage(stack).osType).toEqual(
      ec2.OperatingSystemType.WINDOWS);
  });

  test('correct SSM parameter is set for amazon linux 2 Neuron AMI', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    // WHEN
    cluster.addCapacity('amazonlinux2-neuron-asg', {
      instanceType: new ec2.InstanceType('inf1.xlarge'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.NEURON),
    });

    // THEN
    Template.fromStack(stack).hasParameter('*', {
      Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
      Default: '/aws/service/ecs/optimized-ami/amazon-linux-2/inf/recommended/image_id',
    });
  });

  test('allows setting cluster ServiceConnectDefaults.Namespace property when useAsServiceConnectDefault is true', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addDefaultCloudMapNamespace({
      name: 'foo.com',
      useForServiceConnect: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ServiceConnectDefaults: {
        Namespace: {
          'Fn::GetAtt': ['EcsClusterDefaultServiceDiscoveryNamespaceB0971B2F', 'Arn'],
        },
      },
    });
  });

  test('allows setting cluster _defaultCloudMapNamespace for HTTP namespace', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});
    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    // WHEN
    cluster.addDefaultCloudMapNamespace({
      name: 'foo',
      type: cloudmap.NamespaceType.HTTP,
    });
    expect(cluster.defaultCloudMapNamespace).not.toBe(undefined);
    expect(cluster.defaultCloudMapNamespace!.namespaceName).toBe('foo');
  });

  test('arnForTasks returns a task arn from key pattern', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});
    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    const taskIdPattern = '*';

    // WHEN
    const policyStatement = new iam.PolicyStatement({
      resources: [cluster.arnForTasks(taskIdPattern)],
      actions: ['ecs:RunTask'],
      principals: [new iam.ServicePrincipal('ecs.amazonaws.com')],
    });

    // THEN
    expect(stack.resolve(policyStatement.toStatementJson())).toEqual({
      Action: 'ecs:RunTask',
      Effect: 'Allow',
      Principal: { Service: 'ecs.amazonaws.com' },
      Resource: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':ecs:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':task/',
            { Ref: 'EcsCluster97242B84' },
            `/${taskIdPattern}`,
          ],
        ],
      },
    });
  });

  test('grantTaskProtection grants ecs:UpdateTaskProtection permission', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});
    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    const role = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
    });

    // WHEN
    cluster.grantTaskProtection(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'ecs:UpdateTaskProtection',
            Effect: 'Allow',
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':ecs:',
                  { Ref: 'AWS::Region' },
                  ':',
                  { Ref: 'AWS::AccountId' },
                  ':task/',
                  { Ref: 'EcsCluster97242B84' },
                  '/*',
                ],
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  /*
   * TODO:v2.0.0 END OF OBSOLETE BLOCK
   */

  testDeprecated('allows specifying special HW AMI Type v2', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('GpuAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.GPU),
    });

    // THEN
    const assembly = app.synth();
    const template = assembly.getStackByName(stack.stackName).template;
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2gpurecommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
    });

    expect(template.Parameters).toEqual({
      SsmParameterValueawsserviceecsoptimizedamiamazonlinux2gpurecommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter: {
        Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
        Default: '/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id',
      },
    });
  });

  testDeprecated('allows specifying Amazon Linux v1 AMI', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('GpuAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux(),
    });

    // THEN
    const assembly = app.synth();
    const template = assembly.getStackByName(stack.stackName).template;
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinuxrecommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
    });

    expect(template.Parameters).toEqual({
      SsmParameterValueawsserviceecsoptimizedamiamazonlinuxrecommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter: {
        Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
        Default: '/aws/service/ecs/optimized-ami/amazon-linux/recommended/image_id',
      },
    });
  });

  testDeprecated('allows specifying windows image v2', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('WindowsAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ecs.EcsOptimizedImage.windows(ecs.WindowsOptimizedVersion.SERVER_2019),
    });

    // THEN
    const assembly = app.synth();
    const template = assembly.getStackByName(stack.stackName).template;
    expect(template.Parameters).toEqual({
      SsmParameterValueawsserviceamiwindowslatestWindowsServer2019EnglishFullECSOptimizedimageidC96584B6F00A464EAD1953AFF4B05118Parameter: {
        Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
        Default: '/aws/service/ami-windows-latest/Windows_Server-2019-English-Full-ECS_Optimized/image_id',
      },
    });
  });

  testDeprecated('allows specifying spot fleet', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      spotPrice: '0.31',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      SpotPrice: '0.31',
    });
  });

  testDeprecated('allows specifying drain time', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      taskDrainTime: cdk.Duration.minutes(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LifecycleHook', {
      HeartbeatTimeout: 60,
    });
  });

  testDeprecated('allows specifying automated spot draining', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('c5.xlarge'),
      spotPrice: '0.0735',
      spotInstanceDraining: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      UserData: {
        'Fn::Base64': {
          'Fn::Join': [
            '',
            [
              '#!/bin/bash\necho ECS_CLUSTER=',
              {
                Ref: 'EcsCluster97242B84',
              },
              ' >> /etc/ecs/ecs.config\necho ECS_ENABLE_SPOT_INSTANCE_DRAINING=true >> /etc/ecs/ecs.config',
            ],
          ],
        },
      },
    });
  });

  testDeprecated('allows containers access to instance metadata service', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      UserData: {
        'Fn::Base64': {
          'Fn::Join': [
            '',
            [
              '#!/bin/bash\necho ECS_CLUSTER=',
              {
                Ref: 'EcsCluster97242B84',
              },
              ' >> /etc/ecs/ecs.config',
            ],
          ],
        },
      },
    });
  });

  testDeprecated('allows adding default service discovery namespace', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
    });

    // WHEN
    cluster.addDefaultCloudMapNamespace({
      name: 'foo.com',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::PrivateDnsNamespace', {
      Name: 'foo.com',
      Vpc: {
        Ref: 'MyVpcF9F0CA6F',
      },
    });
  });

  testDeprecated('allows adding public service discovery namespace', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
    });

    // WHEN
    cluster.addDefaultCloudMapNamespace({
      name: 'foo.com',
      type: cloudmap.NamespaceType.DNS_PUBLIC,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ServiceDiscovery::PublicDnsNamespace', {
      Name: 'foo.com',
    });

    expect(cluster.defaultCloudMapNamespace!.type).toEqual(cloudmap.NamespaceType.DNS_PUBLIC);
  });

  testDeprecated('throws if default service discovery namespace added more than once', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
    });

    // WHEN
    cluster.addDefaultCloudMapNamespace({
      name: 'foo.com',
    });

    // THEN
    expect(() => {
      cluster.addDefaultCloudMapNamespace({
        name: 'foo.com',
      });
    }).toThrow(/Can only add default namespace once./);
  });

  test('allows using an existing PrivateDnsNamespace as default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.PrivateDnsNamespace(stack, 'ExistingNamespace', {
      name: 'existing.local',
      vpc,
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    const namespace = cluster.addExistingDefaultCloudMapNamespace({
      namespace: existingNamespace,
    });

    // THEN
    expect(namespace).toBe(existingNamespace);
    expect(cluster.defaultCloudMapNamespace).toBe(existingNamespace);

    // Should not create a new namespace
    Template.fromStack(stack).resourceCountIs('AWS::ServiceDiscovery::PrivateDnsNamespace', 1);
  });

  test('allows using an existing PublicDnsNamespace as default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.PublicDnsNamespace(stack, 'ExistingNamespace', {
      name: 'existing.com',
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    const namespace = cluster.addExistingDefaultCloudMapNamespace({
      namespace: existingNamespace,
    });

    // THEN
    expect(namespace).toBe(existingNamespace);
    expect(cluster.defaultCloudMapNamespace).toBe(existingNamespace);

    // Should not create a new namespace
    Template.fromStack(stack).resourceCountIs('AWS::ServiceDiscovery::PublicDnsNamespace', 1);
  });

  test('allows using an existing HttpNamespace as default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.HttpNamespace(stack, 'ExistingNamespace', {
      name: 'existing',
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    const namespace = cluster.addExistingDefaultCloudMapNamespace({
      namespace: existingNamespace,
    });

    // THEN
    expect(namespace).toBe(existingNamespace);
    expect(cluster.defaultCloudMapNamespace).toBe(existingNamespace);

    // Should not create a new namespace
    Template.fromStack(stack).resourceCountIs('AWS::ServiceDiscovery::HttpNamespace', 1);
  });

  test('allows using an imported namespace as default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const importedNamespace = cloudmap.PrivateDnsNamespace.fromPrivateDnsNamespaceAttributes(stack, 'ImportedNamespace', {
      namespaceId: 'ns-xxxxxxxxxxxxx',
      namespaceArn: 'arn:aws:servicediscovery:us-east-1:123456789012:namespace/ns-xxxxxxxxxxxxx',
      namespaceName: 'imported.local',
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    const namespace = cluster.addExistingDefaultCloudMapNamespace({
      namespace: importedNamespace,
    });

    // THEN
    expect(namespace).toBe(importedNamespace);
    expect(cluster.defaultCloudMapNamespace).toBe(importedNamespace);

    // Should not create any namespace
    Template.fromStack(stack).resourceCountIs('AWS::ServiceDiscovery::PrivateDnsNamespace', 0);
  });

  test('existing namespace can be used for Service Connect', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.PrivateDnsNamespace(stack, 'ExistingNamespace', {
      name: 'existing.local',
      vpc,
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addExistingDefaultCloudMapNamespace({
      namespace: existingNamespace,
      useForServiceConnect: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ServiceConnectDefaults: {
        Namespace: {
          'Fn::GetAtt': ['ExistingNamespaceE824D60B', 'Arn'],
        },
      },
    });
  });

  test('fails when addExistingDefaultCloudMapNamespace is called twice', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.PrivateDnsNamespace(stack, 'ExistingNamespace', {
      name: 'existing.local',
      vpc,
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addExistingDefaultCloudMapNamespace({ namespace: existingNamespace });

    // THEN
    expect(() => {
      cluster.addExistingDefaultCloudMapNamespace({ namespace: existingNamespace });
    }).toThrow(/Can only add default namespace once./);
  });

  test('fails when addExistingDefaultCloudMapNamespace is called after addDefaultCloudMapNamespace', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.PrivateDnsNamespace(stack, 'ExistingNamespace', {
      name: 'existing.local',
      vpc,
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });
    cluster.addDefaultCloudMapNamespace({ name: 'created.local' });

    // THEN
    expect(() => {
      cluster.addExistingDefaultCloudMapNamespace({ namespace: existingNamespace });
    }).toThrow(/Can only add default namespace once./);
  });

  test('fails when the existing namespace is in a different region', () => {
    // GIVEN
    const app = new cdk.App();
    const namespaceStack = new cdk.Stack(app, 'NamespaceStack', { env: { account: '111111111111', region: 'us-east-1' } });
    const clusterStack = new cdk.Stack(app, 'ClusterStack', { env: { account: '111111111111', region: 'us-west-2' } });

    const importedNamespace = cloudmap.PrivateDnsNamespace.fromPrivateDnsNamespaceAttributes(namespaceStack, 'ImportedNamespace', {
      namespaceId: 'ns-xxxxxxxxxxxxx',
      namespaceArn: 'arn:aws:servicediscovery:us-east-1:111111111111:namespace/ns-xxxxxxxxxxxxx',
      namespaceName: 'other-region.local',
    });

    const vpc = new ec2.Vpc(clusterStack, 'MyVpc', {});
    const cluster = new ecs.Cluster(clusterStack, 'EcsCluster', { vpc });

    // THEN
    expect(() => {
      cluster.addExistingDefaultCloudMapNamespace({ namespace: importedNamespace });
    }).toThrow(/Cloud Map namespace must be in the same region as the ECS cluster/);
  });

  test('warns when the existing namespace belongs to a different account', () => {
    // GIVEN
    const app = new cdk.App();
    const namespaceStack = new cdk.Stack(app, 'NamespaceStack', { env: { account: '111111111111', region: 'us-east-1' } });
    const clusterStack = new cdk.Stack(app, 'ClusterStack', { env: { account: '222222222222', region: 'us-east-1' } });
    acknowledgeTestValidationRules(namespaceStack);
    acknowledgeTestValidationRules(clusterStack);

    const importedNamespace = cloudmap.HttpNamespace.fromHttpNamespaceAttributes(namespaceStack, 'ImportedNamespace', {
      namespaceId: 'ns-xxxxxxxxxxxxx',
      namespaceArn: 'arn:aws:servicediscovery:us-east-1:111111111111:namespace/ns-xxxxxxxxxxxxx',
      namespaceName: 'other-account',
    });

    const vpc = new ec2.Vpc(clusterStack, 'MyVpc', {});
    const cluster = new ecs.Cluster(clusterStack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addExistingDefaultCloudMapNamespace({ namespace: importedNamespace });

    // THEN
    Annotations.fromStack(clusterStack).hasWarning('/ClusterStack/EcsCluster', Match.stringLikeRegexp('belongs to a different account'));
  });

  test('warns about VPC association when using an existing PrivateDnsNamespace', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.PrivateDnsNamespace(stack, 'ExistingNamespace', {
      name: 'existing.local',
      vpc,
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addExistingDefaultCloudMapNamespace({ namespace: existingNamespace });

    // THEN
    Annotations.fromStack(stack).hasWarning('/Default/EcsCluster', Match.stringLikeRegexp('ensure it is associated with the same VPC'));
  });

  test('does not warn about VPC association for an existing HttpNamespace', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const existingNamespace = new cloudmap.HttpNamespace(stack, 'ExistingNamespace', {
      name: 'existing',
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addExistingDefaultCloudMapNamespace({ namespace: existingNamespace });

    // THEN
    Annotations.fromStack(stack).hasNoWarning('/Default/EcsCluster', Match.stringLikeRegexp('ensure it is associated with the same VPC'));
  });

  test('fails when an imported namespace without a valid ARN is used for Service Connect', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const importedNamespace = cloudmap.HttpNamespace.fromHttpNamespaceAttributes(stack, 'ImportedNamespace', {
      namespaceId: 'ns-xxxxxxxxxxxxx',
      namespaceArn: 'not-an-arn',
      namespaceName: 'imported',
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // THEN
    expect(() => {
      cluster.addExistingDefaultCloudMapNamespace({
        namespace: importedNamespace,
        useForServiceConnect: true,
      });
    }).toThrow(/The imported namespace does not have a valid ARN/);
  });

  test('imported namespace ARN is used for Service Connect defaults', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const importedNamespace = cloudmap.HttpNamespace.fromHttpNamespaceAttributes(stack, 'ImportedNamespace', {
      namespaceId: 'ns-xxxxxxxxxxxxx',
      namespaceArn: 'arn:aws:servicediscovery:us-east-1:123456789012:namespace/ns-xxxxxxxxxxxxx',
      namespaceName: 'imported',
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addExistingDefaultCloudMapNamespace({
      namespace: importedNamespace,
      useForServiceConnect: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ServiceConnectDefaults: {
        Namespace: 'arn:aws:servicediscovery:us-east-1:123456789012:namespace/ns-xxxxxxxxxxxxx',
      },
    });
  });

  test('export/import of a cluster with a namespace', () => {
    // GIVEN
    const stack1 = new cdk.Stack();
    const vpc1 = new ec2.Vpc(stack1, 'Vpc');
    const cluster1 = new ecs.Cluster(stack1, 'Cluster', { vpc: vpc1 });
    cluster1.addDefaultCloudMapNamespace({
      name: 'hello.com',
    });

    const stack2 = new cdk.Stack();

    // WHEN
    const cluster2 = ecs.Cluster.fromClusterAttributes(stack2, 'Cluster', {
      vpc: vpc1,
      securityGroups: cluster1.connections.securityGroups,
      defaultCloudMapNamespace: cloudmap.PrivateDnsNamespace.fromPrivateDnsNamespaceAttributes(stack2, 'ns', {
        namespaceId: 'import-namespace-id',
        namespaceArn: 'import-namespace-arn',
        namespaceName: 'import-namespace-name',
      }),
      clusterName: 'cluster-name',
    });

    // THEN
    expect(cluster2.defaultCloudMapNamespace!.type).toEqual(cloudmap.NamespaceType.DNS_PRIVATE);
    expect(stack2.resolve(cluster2.defaultCloudMapNamespace!.namespaceId)).toEqual('import-namespace-id');

    // Can retrieve subnets from VPC - will throw 'There are no 'Private' subnets in this VPC. Use a different VPC subnet selection.' if broken.
    cluster2.vpc.selectSubnets();
  });

  test('imported cluster with imported security groups honors allowAllOutbound', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');

    const importedSg1 = ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG1', 'sg-1', { allowAllOutbound: false });
    const importedSg2 = ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG2', 'sg-2');

    const cluster = ecs.Cluster.fromClusterAttributes(stack, 'Cluster', {
      clusterName: 'cluster-name',
      securityGroups: [importedSg1, importedSg2],
      vpc,
    });

    // WHEN
    cluster.connections.allowToAnyIpv4(ec2.Port.tcp(443));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
      GroupId: 'sg-1',
    });

    Template.fromStack(stack).resourceCountIs('AWS::EC2::SecurityGroupEgress', 1);
  });

  test('Security groups are optonal for imported clusters', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');

    const cluster = ecs.Cluster.fromClusterAttributes(stack, 'Cluster', {
      clusterName: 'cluster-name',
      vpc,
    });

    // THEN
    expect(cluster.connections.securityGroups).toEqual([]);
  });

  test('Can import autoscaling groups', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoscalingGroup = new autoscaling.AutoScalingGroup(stack, 'asgal2', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    const cluster = ecs.Cluster.fromClusterAttributes(stack, 'Cluster', {
      clusterName: 'cluster-name',
      vpc,
      autoscalingGroup,
    });

    // THEN
    expect(cluster.autoscalingGroup).toEqual(autoscalingGroup);
  });

  test('Metric', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyVpc', {});

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // THEN
    expect(stack.resolve(cluster.metricCpuReservation())).toEqual({
      dimensions: {
        ClusterName: { Ref: 'EcsCluster97242B84' },
      },
      namespace: 'AWS/ECS',
      metricName: 'CPUReservation',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });

    expect(stack.resolve(cluster.metricMemoryReservation())).toEqual({
      dimensions: {
        ClusterName: { Ref: 'EcsCluster97242B84' },
      },
      namespace: 'AWS/ECS',
      metricName: 'MemoryReservation',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });

    expect(stack.resolve(cluster.metric('myMetric'))).toEqual({
      dimensions: {
        ClusterName: { Ref: 'EcsCluster97242B84' },
      },
      namespace: 'AWS/ECS',
      metricName: 'myMetric',
      period: cdk.Duration.minutes(5),
      statistic: 'Average',
    });
  });

  testDeprecated('ASG with a public VPC without NAT Gateways', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'MyPublicVpc', {
      natGateways: 0,
      subnetConfiguration: [
        { cidrMask: 24, name: 'ingress', subnetType: ec2.SubnetType.PUBLIC },
      ],
    });

    const cluster = new ecs.Cluster(stack, 'EcsCluster', { vpc });

    // WHEN
    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType('t2.micro'),
      associatePublicIpAddress: true,
      vpcSubnets: {
        onePerAz: true,
        subnetType: ec2.SubnetType.PUBLIC,
      },
    });

    Template.fromStack(stack).resourceCountIs('AWS::ECS::Cluster', 1);

    Template.fromStack(stack).hasResourceProperties('AWS::EC2::VPC', {
      CidrBlock: '10.0.0.0/16',
      EnableDnsHostnames: true,
      EnableDnsSupport: true,
      InstanceTenancy: ec2.DefaultInstanceTenancy.DEFAULT,
      Tags: [
        {
          Key: 'Name',
          Value: 'Default/MyPublicVpc',
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2recommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
      InstanceType: 't2.micro',
      AssociatePublicIpAddress: true,
      IamInstanceProfile: {
        Ref: 'EcsClusterDefaultAutoScalingGroupInstanceProfile2CE606B3',
      },
      SecurityGroups: [
        {
          'Fn::GetAtt': [
            'EcsClusterDefaultAutoScalingGroupInstanceSecurityGroup912E1231',
            'GroupId',
          ],
        },
      ],
      UserData: {
        'Fn::Base64': {
          'Fn::Join': [
            '',
            [
              '#!/bin/bash\necho ECS_CLUSTER=',
              {
                Ref: 'EcsCluster97242B84',
              },

              ' >> /etc/ecs/ecs.config',
            ],
          ],
        },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MaxSize: '1',
      MinSize: '1',
      LaunchConfigurationName: {
        Ref: 'EcsClusterDefaultAutoScalingGroupLaunchConfigB7E376C1',
      },
      Tags: [
        {
          Key: 'Name',
          PropagateAtLaunch: true,
          Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
        },
      ],
      VPCZoneIdentifier: [
        {
          Ref: 'MyPublicVpcingressSubnet1Subnet9191044C',
        },
        {
          Ref: 'MyPublicVpcingressSubnet2SubnetD2F2E034',
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
      GroupDescription: 'Default/EcsCluster/DefaultAutoScalingGroup/InstanceSecurityGroup',
      SecurityGroupEgress: [
        {
          CidrIp: '0.0.0.0/0',
          Description: 'Allow all outbound traffic by default',
          IpProtocol: '-1',
        },
      ],
      Tags: [
        {
          Key: 'Name',
          Value: 'Default/EcsCluster/DefaultAutoScalingGroup',
        },
      ],
      VpcId: {
        Ref: 'MyPublicVpcA2BF6CDA',
      },
    });

    // THEN
  });

  test('enable container insights', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    new ecs.Cluster(stack, 'EcsCluster', { containerInsights: true });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ClusterSettings: [
        {
          Name: 'containerInsights',
          Value: 'enabled',
        },
      ],
    });
  });

  test('disable container insights', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    new ecs.Cluster(stack, 'EcsCluster', { containerInsights: false });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ClusterSettings: [
        {
          Name: 'containerInsights',
          Value: 'disabled',
        },
      ],
    });
  });

  test('disabled container insights', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    new ecs.Cluster(stack, 'EcsCluster', { containerInsightsV2: ecs.ContainerInsights.DISABLED });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ClusterSettings: [
        {
          Name: 'containerInsights',
          Value: 'disabled',
        },
      ],
    });
  });

  test('enabled container insights', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    new ecs.Cluster(stack, 'EcsCluster', { containerInsightsV2: ecs.ContainerInsights.ENABLED });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ClusterSettings: [
        {
          Name: 'containerInsights',
          Value: 'enabled',
        },
      ],
    });
  });

  test('enhanced container insights', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    new ecs.Cluster(stack, 'EcsCluster', { containerInsightsV2: ecs.ContainerInsights.ENHANCED });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      ClusterSettings: [
        {
          Name: 'containerInsights',
          Value: 'enhanced',
        },
      ],
    });
  });

  test('should throw an error if containerInsights and containerInsightsLevel are both set', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster',
        {
          containerInsights: true,
          containerInsightsV2: ecs.ContainerInsights.ENHANCED,
        });
    }).toThrow('You cannot set both containerInsights and containerInsightsV2');
  });

  test('should throw an error if containerInsights and containerInsightsLevel are both set, even if containerInsights is false', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster',
        {
          containerInsights: true,
          containerInsightsV2: ecs.ContainerInsights.ENHANCED,
        });
    }).toThrow('You cannot set both containerInsights and containerInsightsV2');
  });

  test('default container insights is undefined', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    new ecs.Cluster(stack, 'EcsCluster');

    // THEN
    const assembly = app.synth();
    const stackAssembly = assembly.getStackByName(stack.stackName);
    const template = stackAssembly.template;

    expect(
      template.Resources.EcsCluster97242B84.Properties === undefined ||
      template.Resources.EcsCluster97242B84.Properties.ClusterSettings === undefined,
    ).toEqual(true);
  });

  test('enable fargate ephemeral storage encryption on cluster with random name', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const key = new kms.Key(stack, 'key', { policy: new iam.PolicyDocument() });
    new ecs.Cluster(stack, 'EcsCluster', { managedStorageConfiguration: { fargateEphemeralStorageKmsKey: key } });

    // THEN
    const output = Template.fromStack(stack);
    output.hasResourceProperties('AWS::ECS::Cluster', {
      Configuration: {
        ManagedStorageConfiguration: {
          FargateEphemeralStorageKmsKeyId: {
            Ref: 'keyFEDD6EC0',
          },
        },
      },
    });
    output.hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Statement: [
          {
            Resource: '*',
            Effect: 'Allow',
            Action: 'kms:GenerateDataKeyWithoutPlaintext',
            Principal: { Service: 'fargate.amazonaws.com' },
            Condition: {
              StringEquals: {
                'kms:EncryptionContext:aws:ecs:clusterAccount': [{ Ref: 'AWS::AccountId' }],
              },
            },
          },
          {
            Resource: '*',
            Effect: 'Allow',
            Action: 'kms:CreateGrant',
            Principal: { Service: 'fargate.amazonaws.com' },
            Condition: {
              'StringEquals': {
                'kms:EncryptionContext:aws:ecs:clusterAccount': [{ Ref: 'AWS::AccountId' }],
              },
              'ForAllValues:StringEquals': {
                'kms:GrantOperations': ['Decrypt'],
              },
            },
          },
        ],
      },
    });
  });

  test('enable fargate ephemeral storage encryption on cluster with defined name', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const key = new kms.Key(stack, 'key', { policy: new iam.PolicyDocument() });
    new ecs.Cluster(stack, 'EcsCluster', { clusterName: 'cluster-name', managedStorageConfiguration: { fargateEphemeralStorageKmsKey: key } });

    // THEN
    const output = Template.fromStack(stack);
    output.hasResourceProperties('AWS::ECS::Cluster', {
      Configuration: {
        ManagedStorageConfiguration: {
          FargateEphemeralStorageKmsKeyId: {
            Ref: 'keyFEDD6EC0',
          },
        },
      },
    });
    output.hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Statement: [
          {
            Resource: '*',
            Effect: 'Allow',
            Action: 'kms:GenerateDataKeyWithoutPlaintext',
            Principal: { Service: 'fargate.amazonaws.com' },
            Condition: {
              StringEquals: {
                'kms:EncryptionContext:aws:ecs:clusterAccount': [{ Ref: 'AWS::AccountId' }],
                'kms:EncryptionContext:aws:ecs:clusterName': ['cluster-name'],
              },
            },
          },
          {
            Resource: '*',
            Effect: 'Allow',
            Action: 'kms:CreateGrant',
            Principal: { Service: 'fargate.amazonaws.com' },
            Condition: {
              'StringEquals': {
                'kms:EncryptionContext:aws:ecs:clusterAccount': [{ Ref: 'AWS::AccountId' }],
                'kms:EncryptionContext:aws:ecs:clusterName': ['cluster-name'],
              },
              'ForAllValues:StringEquals': {
                'kms:GrantOperations': ['Decrypt'],
              },
            },
          },
        ],
      },
    });
  });

  test('enable managed storage encryption on cluster', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const key = new kms.Key(stack, 'key', { policy: new iam.PolicyDocument() });
    new ecs.Cluster(stack, 'EcsCluster', { managedStorageConfiguration: { kmsKey: key } });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      Configuration: {
        ManagedStorageConfiguration: {
          KmsKeyId: {
            Ref: 'keyFEDD6EC0',
          },
        },
      },
    });
  });

  test('BottleRocketImage() returns correct AMI', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // WHEN
    new ecs.BottleRocketImage().getImage(stack);

    // THEN
    const assembly = app.synth();
    const parameters = assembly.getStackByName(stack.stackName).template.Parameters;
    expect(Object.entries(parameters).some(
      ([k, v]) => k.startsWith('SsmParameterValueawsservicebottlerocketawsecs') &&
        (v as any).Default.includes('/bottlerocket/'),
    )).toEqual(true);
    expect(Object.entries(parameters).some(
      ([k, v]) => k.startsWith('SsmParameterValueawsservicebottlerocketawsecs') &&
        (v as any).Default.includes('/aws-ecs-1/'),
    )).toEqual(true);
  });

  describe('isBottleRocketImage() returns', () => {
    test('true if given bottleRocketImage instance', () => {
      // WHEN
      const bottleRockectImage = new ecs.BottleRocketImage();
      // THEN
      expect(ecs.BottleRocketImage.isBottleRocketImage(bottleRockectImage)).toBe(true);
    });

    test('false if given amazonLinux instance', () => {
      // GIVEN
      const wrongImage = ec2.MachineImage.latestAmazonLinux();
      // THEN
      expect(ecs.BottleRocketImage.isBottleRocketImage(wrongImage)).toBe(false);
    });

    test('false if given undefined', () => {
      // THEN
      expect(ecs.BottleRocketImage.isBottleRocketImage(undefined)).toBe(false);
    });
  });

  testDeprecated('cluster capacity with bottlerocket AMI, by setting machineImageType', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImageType: ecs.MachineImageType.BOTTLEROCKET,
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::ECS::Cluster', 1);
    Template.fromStack(stack).resourceCountIs('AWS::AutoScaling::AutoScalingGroup', 1);
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsservicebottlerocketawsecs1x8664latestimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
      UserData: {
        'Fn::Base64': {
          'Fn::Join': [
            '',
            [
              '\n[settings.ecs]\ncluster = "',
              {
                Ref: 'EcsCluster97242B84',
              },
              '"',
            ],
          ],
        },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'ec2.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
      ManagedPolicyArns: [
        {
          'Fn::Join': [
            '',
            [
              'arn:',
              {
                Ref: 'AWS::Partition',
              },
              ':iam::aws:policy/AmazonSSMManagedInstanceCore',
            ],
          ],
        },
        {
          'Fn::Join': [
            '',
            [
              'arn:',
              {
                Ref: 'AWS::Partition',
              },
              ':iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role',
            ],
          ],
        },
      ],
      Tags: [
        {
          Key: 'Name',
          Value: 'test/EcsCluster/bottlerocket-asg',
        },
      ],
    });
  });

  testDeprecated('correct bottlerocket AMI for ARM64 architecture', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('m6g.large'),
      machineImageType: ecs.MachineImageType.BOTTLEROCKET,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      ImageId: {
        Ref: 'SsmParameterValueawsservicebottlerocketawsecs1arm64latestimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
      },
    });

    Template.fromStack(stack).hasParameter('SsmParameterValueawsservicebottlerocketawsecs1arm64latestimageidC96584B6F00A464EAD1953AFF4B05118Parameter', {
      Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
      Default: '/aws/service/bottlerocket/aws-ecs-1/arm64/latest/image_id',
    });
  });

  testDeprecated('throws when machineImage and machineImageType both specified', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImage: new ecs.BottleRocketImage(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      UserData: {
        'Fn::Base64': {
          'Fn::Join': [
            '',
            [
              '\n[settings.ecs]\ncluster = "',
              {
                Ref: 'EcsCluster97242B84',
              },
              '"',
            ],
          ],
        },
      },
    });
  });

  testDeprecated('updatePolicy set when passed without updateType', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImage: new ecs.BottleRocketImage(),
      updatePolicy: autoscaling.UpdatePolicy.replacingUpdate(),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        AutoScalingReplacingUpdate: {
          WillReplace: true,
        },
      },
    });
  });

  testDeprecated('undefined updateType & updatePolicy replaced by default updatePolicy', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImage: new ecs.BottleRocketImage(),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        AutoScalingReplacingUpdate: {
          WillReplace: true,
        },
      },
    });
  });

  testDeprecated('updateType.NONE replaced by updatePolicy equivalent', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImage: new ecs.BottleRocketImage(),
      updateType: autoscaling.UpdateType.NONE,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        AutoScalingScheduledAction: {
          IgnoreUnmodifiedGroupSizeProperties: true,
        },
      },
    });
  });

  testDeprecated('updateType.REPLACING_UPDATE replaced by updatePolicy equivalent', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImage: new ecs.BottleRocketImage(),
      updateType: autoscaling.UpdateType.REPLACING_UPDATE,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        AutoScalingReplacingUpdate: {
          WillReplace: true,
        },
      },
    });
  });

  testDeprecated('updateType.ROLLING_UPDATE replaced by updatePolicy equivalent', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.addCapacity('bottlerocket-asg', {
      instanceType: new ec2.InstanceType('c5.large'),
      machineImage: new ecs.BottleRocketImage(),
      updateType: autoscaling.UpdateType.ROLLING_UPDATE,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        AutoScalingRollingUpdate: {
          WaitOnResourceSignals: false,
          PauseTime: 'PT0S',
          SuspendProcesses: [
            'HealthCheck',
            'ReplaceUnhealthy',
            'AZRebalance',
            'AlarmNotification',
            'ScheduledActions',
            'InstanceRefresh',
          ],
        },
        AutoScalingScheduledAction: {
          IgnoreUnmodifiedGroupSizeProperties: true,
        },
      },
    });
  });

  testDeprecated('throws when updatePolicy and updateType both specified', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    expect(() => {
      cluster.addCapacity('bottlerocket-asg', {
        instanceType: new ec2.InstanceType('c5.large'),
        machineImage: new ecs.BottleRocketImage(),
        updatePolicy: autoscaling.UpdatePolicy.replacingUpdate(),
        updateType: autoscaling.UpdateType.REPLACING_UPDATE,
      });
    }).toThrow("Cannot set 'signals'/'updatePolicy' and 'updateType' together. Prefer 'signals'/'updatePolicy'");
  });

  testDeprecated('allows specifying capacityProviders (deprecated)', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // WHEN
    new ecs.Cluster(stack, 'EcsCluster', { capacityProviders: ['FARGATE_SPOT'] });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      CapacityProviders: Match.absent(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      CapacityProviders: ['FARGATE_SPOT'],
    });
  });

  test('allows specifying Fargate capacityProviders', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // WHEN
    new ecs.Cluster(stack, 'EcsCluster', {
      enableFargateCapacityProviders: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      CapacityProviders: Match.absent(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      CapacityProviders: ['FARGATE', 'FARGATE_SPOT'],
    });
  });

  test('allows specifying capacityProviders (alternate method)', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // WHEN
    const cluster = new ecs.Cluster(stack, 'EcsCluster');
    cluster.enableFargateCapacityProviders();

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      CapacityProviders: Match.absent(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      CapacityProviders: ['FARGATE', 'FARGATE_SPOT'],
    });
  });

  testDeprecated('allows adding capacityProviders post-construction (deprecated)', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    // WHEN
    cluster.addCapacityProvider('FARGATE');
    cluster.addCapacityProvider('FARGATE'); // does not add twice

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      CapacityProviders: Match.absent(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      CapacityProviders: ['FARGATE'],
    });
  });

  testDeprecated('allows adding capacityProviders post-construction', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    // WHEN
    cluster.addCapacityProvider('FARGATE');
    cluster.addCapacityProvider('FARGATE'); // does not add twice

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      CapacityProviders: Match.absent(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      CapacityProviders: ['FARGATE'],
    });
  });

  testDeprecated('throws for unsupported capacity providers', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    // THEN
    expect(() => {
      cluster.addCapacityProvider('HONK');
    }).toThrow(/CapacityProvider not supported/);
  });

  describe('creates ASG capacity providers ', () => {
    test('with expected defaults', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
        vpc,
        instanceType: new ec2.InstanceType('bogus'),
        machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
      });

      // WHEN
      new ecs.AsgCapacityProvider(stack, 'provider', {
        autoScalingGroup,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        AutoScalingGroupProvider: {
          AutoScalingGroupArn: {
            Ref: 'asgASG4D014670',
          },
          ManagedScaling: {
            Status: 'ENABLED',
            TargetCapacity: 100,
          },
          ManagedTerminationProtection: 'ENABLED',
        },
      });
    });

    test('with IAutoScalingGroup should throw an error if Managed Termination Protection is enabled.', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const autoScalingGroup = autoscaling.AutoScalingGroup.fromAutoScalingGroupName(stack, 'ASG', 'my-asg');

      // THEN
      expect(() => {
        new ecs.AsgCapacityProvider(stack, 'provider', {
          autoScalingGroup,
        });
      }).toThrow('Cannot enable Managed Termination Protection on a Capacity Provider when providing an imported AutoScalingGroup.');
    });

    test('with IAutoScalingGroup should not throw an error if Managed Termination Protection is disabled.', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const autoScalingGroup = autoscaling.AutoScalingGroup.fromAutoScalingGroupName(stack, 'ASG', 'my-asg');

      // WHEN
      new ecs.AsgCapacityProvider(stack, 'provider', {
        autoScalingGroup,
        enableManagedTerminationProtection: false,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        AutoScalingGroupProvider: {
          AutoScalingGroupArn: 'my-asg',
          ManagedScaling: {
            Status: 'ENABLED',
            TargetCapacity: 100,
          },
          ManagedTerminationProtection: 'DISABLED',
        },
      });
    });
  });

  describe('creates Managed Instances capacity providers', () => {
    test('with expected defaults', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
        },
      });
    });

    test('with custom capacity provider name', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        capacityProviderName: 'my-managed-instances-cp',
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        Name: 'my-managed-instances-cp',
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
        },
      });
    });

    test('with security groups', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      // WHEN
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
              SecurityGroups: [
                { 'Fn::GetAtt': ['SecurityGroupDD263621', 'GroupId'] },
              ],
            },
          },
        },
      });
    });

    test('with task volume storage', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        taskVolumeStorage: cdk.Size.gibibytes(100),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
            StorageConfiguration: {
              StorageSizeGiB: 100,
            },
          },
        },
      });
    });

    test('with monitoring configuration', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        monitoring: ecs.InstanceMonitoring.DETAILED,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
            Monitoring: 'DETAILED',
          },
        },
      });
    });

    test('with capacity option type ON_DEMAND', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      // WHEN
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        capacityOptionType: ecs.CapacityOptionType.ON_DEMAND,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            CapacityOptionType: 'ON_DEMAND',
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
        },
      });
    });

    test('with capacity option type SPOT', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      // WHEN
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        capacityOptionType: ecs.CapacityOptionType.SPOT,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            CapacityOptionType: 'SPOT',
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
        },
      });
    });

    test('without capacity option type does not include property in template', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
            CapacityOptionType: Match.absent(),
          },
        },
      });
    });

    test('with instance requirements', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        instanceRequirements: {
          vCpuCountMin: 2,
          vCpuCountMax: 8,
          memoryMin: cdk.Size.gibibytes(4),
          memoryMax: cdk.Size.gibibytes(16),
          cpuManufacturers: [ec2.CpuManufacturer.INTEL, ec2.CpuManufacturer.AMD],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
            InstanceRequirements: {
              VCpuCount: {
                Min: 2,
                Max: 8,
              },
              MemoryMiB: {
                Min: 4096,
                Max: 16384,
              },
              CpuManufacturers: ['intel', 'amd'],
            },
          },
        },
      });
    });

    test('with propagate tags', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        propagateTags: ecs.PropagateManagedInstancesTags.CAPACITY_PROVIDER,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('InfrastructureRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
          PropagateTags: 'CAPACITY_PROVIDER',
        },
      });
    });

    test('throws when subnets are not provided', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      // THEN
      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: [],
          securityGroups: [securityGroup],
        });
      }).toThrow('Subnets are required and should be non-empty.');
    });

    test('throws when securityGroups is an empty array', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // THEN
      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: vpc.privateSubnets,
          securityGroups: [],
        });
      }).toThrow('Security groups cannot be an empty array. Provide at least one security group.');
    });

    test('throws when both allowedInstanceTypes and excludedInstanceTypes are specified in instanceRequirements', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // THEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: vpc.privateSubnets,
          securityGroups: [securityGroup],
          instanceRequirements: {
            vCpuCountMin: 2,
            memoryMin: cdk.Size.gibibytes(4),
            allowedInstanceTypes: ['m5.large', 'c5.xlarge'],
            excludedInstanceTypes: ['t2.micro', 't3.nano'],
          },
        });
      }).toThrow('Cannot specify both allowedInstanceTypes and excludedInstanceTypes. Use one or the other.');
    });

    test('throws when both spotMaxPricePercentageOverLowestPrice and maxSpotPriceAsPercentageOfOptimalOnDemandPrice are specified in instanceRequirements', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // THEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: vpc.privateSubnets,
          securityGroups: [securityGroup],
          instanceRequirements: {
            vCpuCountMin: 2,
            memoryMin: cdk.Size.gibibytes(4),
            spotMaxPricePercentageOverLowestPrice: 30,
            onDemandMaxPricePercentageOverLowestPrice: 50,
          },
        });
      }).toThrow('Cannot specify both spotMaxPricePercentageOverLowestPrice and onDemandMaxPricePercentageOverLowestPrice. Use one or the other.');
    });

    test('throws when capacity provider name starts with aws, ecs or fargate', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      // THEN
      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
          capacityProviderName: 'awscp',
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: vpc.privateSubnets,
          securityGroups: [securityGroup],
        });
      }).toThrow(/Invalid Capacity Provider Name: awscp, If a name is specified, it cannot start with aws, ecs, or fargate./);

      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider2', {
          capacityProviderName: 'ecscp',
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: vpc.privateSubnets,
          securityGroups: [securityGroup],
        });
      }).toThrow(/Invalid Capacity Provider Name: ecscp, If a name is specified, it cannot start with aws, ecs, or fargate./);

      expect(() => {
        new ecs.ManagedInstancesCapacityProvider(stack, 'provider3', {
          capacityProviderName: 'fargatecp',
          infrastructureRole,
          ec2InstanceProfile: instanceProfile,
          subnets: vpc.privateSubnets,
          securityGroups: [securityGroup],
        });
      }).toThrow(/Invalid Capacity Provider Name: fargatecp, If a name is specified, it cannot start with aws, ecs, or fargate./);
    });

    test('allows modifying security groups via IConnectable interface', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonECSInfrastructureRolePolicyForManagedInstances'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonECSInstanceRolePolicyForManagedInstances'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      // WHEN
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      // Use connections API to allow inbound traffic
      capacityProvider.connections.allowFrom(ec2.Peer.anyIpv4(), ec2.Port.tcp(80));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupIngress: [
          {
            IpProtocol: 'tcp',
            FromPort: 80,
            ToPort: 80,
            CidrIp: '0.0.0.0/0',
          },
        ],
      });
    });

    test('creates default instance profile when ec2InstanceProfile is not provided', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('provider.*Role.*'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('provider.*InstanceProfile.*'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
        },
      });

      // Verify default infrastructure role is created
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ecs.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
        ManagedPolicyArns: [
          {
            'Fn::Join': [
              '',
              [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':iam::aws:policy/AmazonECSInfrastructureRolePolicyForManagedInstances',
              ],
            ],
          },
        ],
      });

      // Verify default instance profile is created with ecsInstanceRole prefix
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::InstanceProfile', {
        InstanceProfileName: Match.stringLikeRegexp('^ecsInstanceRole.*'),
        Roles: [
          { Ref: Match.stringLikeRegexp('provider.*InstanceRole.*') },
        ],
      });

      // Verify default instance role is created
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ec2.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
        Policies: [
          {
            PolicyName: 'ECSInstancePolicy',
            PolicyDocument: {
              Statement: [
                {
                  Sid: 'ECSAgentDiscoverPollEndpointPermissions',
                  Effect: 'Allow',
                  Action: 'ecs:DiscoverPollEndpoint',
                  Resource: '*',
                },
                {
                  Sid: 'ECSAgentRegisterPermissions',
                  Effect: 'Allow',
                  Action: 'ecs:RegisterContainerInstance',
                  Resource: {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
                {
                  Sid: 'ECSAgentPollPermissions',
                  Effect: 'Allow',
                  Action: 'ecs:Poll',
                  Resource: {
                    'Fn::Join': [
                      '',
                      [
                        'arn:',
                        { Ref: 'AWS::Partition' },
                        ':ecs:',
                        { Ref: 'AWS::Region' },
                        ':',
                        { Ref: 'AWS::AccountId' },
                        ':container-instance/*',
                      ],
                    ],
                  },
                },
                {
                  Sid: 'ECSAgentTelemetryPermissions',
                  Effect: 'Allow',
                  Action: ['ecs:StartTelemetrySession', 'ecs:PutSystemLogEvents'],
                  Resource: {
                    'Fn::Join': [
                      '',
                      [
                        'arn:',
                        { Ref: 'AWS::Partition' },
                        ':ecs:',
                        { Ref: 'AWS::Region' },
                        ':',
                        { Ref: 'AWS::AccountId' },
                        ':container-instance/*',
                      ],
                    ],
                  },
                },
                {
                  Sid: 'ECSAgentStateChangePermissions',
                  Effect: 'Allow',
                  Action: ['ecs:SubmitAttachmentStateChanges', 'ecs:SubmitTaskStateChange'],
                  Resource: {
                    'Fn::GetAtt': [
                      'EcsCluster97242B84',
                      'Arn',
                    ],
                  },
                },
              ],
              Version: '2012-10-17',
            },
          },
        ],
      });

      // Verify public properties are accessible
      expect(capacityProvider.infrastructureRole).toBeDefined();
      expect(capacityProvider.ec2InstanceProfile).toBeDefined();
    });

    test('uses provided instance profile when ec2InstanceProfile is specified', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const instanceRole = new iam.Role(stack, 'CustomInstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonECSInstanceRolePolicyForManagedInstances'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'CustomInstanceProfile', {
        role: instanceRole,
        instanceProfileName: 'customInstanceProfile',
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('CustomInstanceProfile.*'),
                'Arn',
              ],
            },
          },
        },
      });

      // Verify the provided instance profile is used
      expect(capacityProvider.ec2InstanceProfile).toBe(instanceProfile);
    });

    test('creates default infrastructure role when not provided', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      // Verify default infrastructure role is created with correct managed policy
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ecs.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
        ManagedPolicyArns: [
          {
            'Fn::Join': [
              '',
              [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':iam::aws:policy/AmazonECSInfrastructureRolePolicyForManagedInstances',
              ],
            ],
          },
        ],
      });

      // Verify the infrastructure role is accessible
      expect(capacityProvider.infrastructureRole).toBeDefined();
    });

    test('uses provided infrastructure role when specified', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      const customInfrastructureRole = new iam.Role(stack, 'CustomInfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonECSInfrastructureRolePolicyForManagedInstances'),
        ],
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole: customInfrastructureRole,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('CustomInfrastructureRole.*'),
              'Arn',
            ],
          },
        },
      });

      // Verify the provided infrastructure role is used
      expect(capacityProvider.infrastructureRole).toBe(customInfrastructureRole);
    });

    test('default instance profile name has ecsInstanceRole prefix', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      // Verify the instance profile name starts with 'ecsInstanceRole'
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::InstanceProfile', {
        InstanceProfileName: Match.stringLikeRegexp('^ecsInstanceRole.*'),
      });
    });

    test('default instance role name has ecsInstanceRole prefix', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      // Verify the instance role name starts with 'ecsInstanceRole'
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'ec2.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
        RoleName: Match.stringLikeRegexp('^ecsInstanceRole.*'),
      });
    });

    test('can add Managed Instances capacity via Capacity Provider', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.enableFargateCapacityProviders();

      // Ensure not added twice
      cluster.addManagedInstancesCapacityProvider(capacityProvider);
      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InfrastructureRoleArn: {
            'Fn::GetAtt': [
              Match.stringLikeRegexp('providerRole'),
              'Arn',
            ],
          },
          InstanceLaunchTemplate: {
            Ec2InstanceProfileArn: {
              'Fn::GetAtt': [
                Match.stringLikeRegexp('InstanceProfile'),
                'Arn',
              ],
            },
            NetworkConfiguration: {
              Subnets: [
                { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
                { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
              ],
            },
          },
        },
      });
    });

    test('does not create CfnClusterCapacityProviderAssociations when using managed instances capacity provider', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const cluster = new ecs.Cluster(stack, 'EcsCluster');

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonECSInstanceRolePolicyForManagedInstances'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      const capacityProvider = new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
      });

      cluster.addManagedInstancesCapacityProvider(capacityProvider);

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::ECS::ClusterCapacityProviderAssociations', 0);
    });

    test('minimal configuration with required fields only', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const config: ec2.InstanceRequirementsConfig = {
        memoryMin: cdk.Size.gibibytes(4),
        vCpuCountMin: 2,
      };

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        instanceRequirements: config,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InstanceLaunchTemplate: {
            InstanceRequirements: {
              VCpuCount: {
                Min: 2,
                Max: Match.absent(),
              },
              MemoryMiB: {
                Min: 4096, // 4 GiB in MiB
                Max: Match.absent(),
              },
              AcceleratorCount: Match.absent(),
              AcceleratorManufacturers: Match.absent(),
              AcceleratorNames: Match.absent(),
              AcceleratorTotalMemoryMiB: Match.absent(),
              AcceleratorTypes: Match.absent(),
              AllowedInstanceTypes: Match.absent(),
              BareMetal: Match.absent(),
              BaselineEbsBandwidthMbps: Match.absent(),
              BurstablePerformance: Match.absent(),
              CpuManufacturers: Match.absent(),
              ExcludedInstanceTypes: Match.absent(),
              InstanceGenerations: Match.absent(),
              LocalStorage: Match.absent(),
              LocalStorageTypes: Match.absent(),
              MaxSpotPriceAsPercentageOfOptimalOnDemandPrice: Match.absent(),
              MemoryGiBPerVCpu: Match.absent(),
              NetworkBandwidthGbps: Match.absent(),
              NetworkInterfaceCount: Match.absent(),
              OnDemandMaxPricePercentageOverLowestPrice: Match.absent(),
              RequireHibernateSupport: Match.absent(),
              SpotMaxPricePercentageOverLowestPrice: Match.absent(),
              TotalLocalStorageGB: Match.absent(),
            },
          },
        },
      });
    });

    test('minimal configuration with required fields only', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const config: ec2.InstanceRequirementsConfig = {
        memoryMin: cdk.Size.gibibytes(4),
        vCpuCountMin: 2,
      };

      // WHEN

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        instanceRequirements: config,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InstanceLaunchTemplate: {
            InstanceRequirements: {
              VCpuCount: {
                Min: 2,
                Max: Match.absent(),
              },
              MemoryMiB: {
                Min: 4096, // 4 GiB in MiB
                Max: Match.absent(),
              },
              AcceleratorCount: Match.absent(),
              AcceleratorManufacturers: Match.absent(),
              AcceleratorNames: Match.absent(),
              AcceleratorTotalMemoryMiB: Match.absent(),
              AcceleratorTypes: Match.absent(),
              AllowedInstanceTypes: Match.absent(),
              BareMetal: Match.absent(),
              BaselineEbsBandwidthMbps: Match.absent(),
              BurstablePerformance: Match.absent(),
              CpuManufacturers: Match.absent(),
              ExcludedInstanceTypes: Match.absent(),
              InstanceGenerations: Match.absent(),
              LocalStorage: Match.absent(),
              LocalStorageTypes: Match.absent(),
              MaxSpotPriceAsPercentageOfOptimalOnDemandPrice: Match.absent(),
              MemoryGiBPerVCpu: Match.absent(),
              NetworkBandwidthGbps: Match.absent(),
              NetworkInterfaceCount: Match.absent(),
              OnDemandMaxPricePercentageOverLowestPrice: Match.absent(),
              RequireHibernateSupport: Match.absent(),
              SpotMaxPricePercentageOverLowestPrice: Match.absent(),
              TotalLocalStorageGB: Match.absent(),
            },
          },
        },
      });
    });

    test('full configuration with all fields', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'test');
      acknowledgeTestValidationRules(stack);
      const vpc = new ec2.Vpc(stack, 'Vpc');

      const infrastructureRole = new iam.Role(stack, 'InfrastructureRole', {
        assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceRole = new iam.Role(stack, 'InstanceRole', {
        assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
        managedPolicies: [
          iam.ManagedPolicy.fromAwsManagedPolicyName('AdministratorAccess'),
        ],
      });

      const instanceProfile = new iam.InstanceProfile(stack, 'InstanceProfile', {
        role: instanceRole,
      });

      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
        vpc,
        description: 'Test security group',
      });

      const config: ec2.InstanceRequirementsConfig = {
        acceleratorCountMin: 1,
        acceleratorCountMax: 4,
        acceleratorManufacturers: [ec2.AcceleratorManufacturer.NVIDIA, ec2.AcceleratorManufacturer.AMD],
        acceleratorNames: [ec2.AcceleratorName.A100, ec2.AcceleratorName.V100],
        acceleratorTotalMemoryMin: cdk.Size.gibibytes(8),
        acceleratorTotalMemoryMax: cdk.Size.gibibytes(32),
        acceleratorTypes: [ec2.AcceleratorType.GPU],
        allowedInstanceTypes: ['m5.large', 'c5.xlarge'],
        bareMetal: ec2.BareMetal.EXCLUDED,
        baselineEbsBandwidthMbpsMin: 1000,
        baselineEbsBandwidthMbpsMax: 5000,
        burstablePerformance: ec2.BurstablePerformance.INCLUDED,
        cpuManufacturers: [ec2.CpuManufacturer.INTEL, ec2.CpuManufacturer.AMD],
        instanceGenerations: [ec2.InstanceGeneration.CURRENT],
        localStorage: ec2.LocalStorage.REQUIRED,
        localStorageTypes: [ec2.LocalStorageType.SSD],
        maxSpotPriceAsPercentageOfOptimalOnDemandPrice: 50,
        memoryPerVCpuMin: cdk.Size.gibibytes(2),
        memoryPerVCpuMax: cdk.Size.gibibytes(8),
        memoryMin: cdk.Size.gibibytes(4),
        memoryMax: cdk.Size.gibibytes(64),
        networkBandwidthGbpsMin: 1,
        networkBandwidthGbpsMax: 10,
        networkInterfaceCountMin: 1,
        networkInterfaceCountMax: 4,
        requireHibernateSupport: true,
        spotMaxPricePercentageOverLowestPrice: 30,
        totalLocalStorageGBMin: 100,
        totalLocalStorageGBMax: 1000,
        vCpuCountMin: 2,
        vCpuCountMax: 16,
      };

      // WHEN
      new ecs.ManagedInstancesCapacityProvider(stack, 'provider', {
        infrastructureRole,
        ec2InstanceProfile: instanceProfile,
        subnets: vpc.privateSubnets,
        securityGroups: [securityGroup],
        instanceRequirements: config,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
        ManagedInstancesProvider: {
          InstanceLaunchTemplate: {
            InstanceRequirements: {
              VCpuCount: {
                Min: 2,
                Max: 16,
              },
              MemoryMiB: {
                Min: 4096, // 4 GiB in MiB
                Max: 65536, // 64 GiB in MiB
              },
              AcceleratorCount: {
                Min: 1,
                Max: 4,
              },
              AcceleratorManufacturers: ['nvidia', 'amd'],
              AcceleratorNames: ['a100', 'v100'],
              AcceleratorTotalMemoryMiB: {
                Min: 8192, // 8 GiB in MiB
                Max: 32768, // 32 GiB in MiB
              },
              AcceleratorTypes: ['gpu'],
              AllowedInstanceTypes: ['m5.large', 'c5.xlarge'],
              BareMetal: 'excluded',
              BaselineEbsBandwidthMbps: {
                Min: 1000,
                Max: 5000,
              },
              BurstablePerformance: 'included',
              CpuManufacturers: ['intel', 'amd'],
              InstanceGenerations: ['current'],
              LocalStorage: 'required',
              LocalStorageTypes: ['ssd'],
              MaxSpotPriceAsPercentageOfOptimalOnDemandPrice: 50,
              MemoryGiBPerVCpu: {
                Min: 2, // 2 GiB
                Max: 8, // 8 GiB
              },
              NetworkBandwidthGbps: {
                Min: 1,
                Max: 10,
              },
              NetworkInterfaceCount: {
                Min: 1,
                Max: 4,
              },
              RequireHibernateSupport: true,
              SpotMaxPricePercentageOverLowestPrice: 30,
              TotalLocalStorageGB: {
                Min: 100,
                Max: 1000,
              },
            },
          },
        },
      });
    });
  });

  test('can disable Managed Scaling and Managed Termination Protection for ASG capacity provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedScaling: false,
      enableManagedTerminationProtection: false,
      enableManagedDraining: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
      AutoScalingGroupProvider: {
        AutoScalingGroupArn: {
          Ref: 'asgASG4D014670',
        },
        ManagedScaling: Match.absent(),
        ManagedTerminationProtection: 'DISABLED',
        ManagedDraining: 'DISABLED',
      },
    });
  });

  test('can disable Managed Termination Protection for ASG capacity provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedTerminationProtection: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
      AutoScalingGroupProvider: {
        AutoScalingGroupArn: {
          Ref: 'asgASG4D014670',
        },
        ManagedScaling: {
          Status: 'ENABLED',
          TargetCapacity: 100,
        },
        ManagedTerminationProtection: 'DISABLED',
      },
    });
  });

  test('can disable Managed Draining for ASG capacity provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedDraining: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
      AutoScalingGroupProvider: {
        AutoScalingGroupArn: {
          Ref: 'asgASG4D014670',
        },
        ManagedDraining: 'DISABLED',
        ManagedScaling: {
          Status: 'ENABLED',
          TargetCapacity: 100,
        },
      },
    });
  });

  test('can enable Managed Draining for ASG capacity provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedDraining: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::CapacityProvider', {
      AutoScalingGroupProvider: {
        AutoScalingGroupArn: {
          Ref: 'asgASG4D014670',
        },
        ManagedDraining: 'ENABLED',
        ManagedScaling: {
          Status: 'ENABLED',
          TargetCapacity: 100,
        },
      },
    });
  });

  test('throws error, when ASG capacity provider has Managed Scaling disabled and Managed Termination Protection is undefined (defaults to true)', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // THEN
    expect(() => {
      new ecs.AsgCapacityProvider(stack, 'provider', {
        autoScalingGroup,
        enableManagedScaling: false,
      });
    }).toThrow('Cannot enable Managed Termination Protection on a Capacity Provider when Managed Scaling is disabled. Either enable Managed Scaling or disable Managed Termination Protection.');
  });

  test('throws error, when Managed Scaling is disabled and Managed Termination Protection is enabled.', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // THEN
    expect(() => {
      new ecs.AsgCapacityProvider(stack, 'provider', {
        autoScalingGroup,
        enableManagedScaling: false,
        enableManagedTerminationProtection: true,
      });
    }).toThrow('Cannot enable Managed Termination Protection on a Capacity Provider when Managed Scaling is disabled. Either enable Managed Scaling or disable Managed Termination Protection.');
  });

  test('capacity provider enables ASG new instance scale-in protection by default', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NewInstancesProtectedFromScaleIn: true,
    });
  });

  test('capacity provider disables ASG new instance scale-in protection', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedTerminationProtection: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NewInstancesProtectedFromScaleIn: Match.absent(),
    });
  });

  test('can add ASG capacity via Capacity Provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedTerminationProtection: false,
    });

    cluster.enableFargateCapacityProviders();

    // Ensure not added twice
    cluster.addAsgCapacityProvider(capacityProvider);
    cluster.addAsgCapacityProvider(capacityProvider);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      Cluster: {
        Ref: 'EcsCluster97242B84',
      },
      CapacityProviders: [
        'FARGATE',
        'FARGATE_SPOT',
        {
          Ref: 'providerD3FF4D3A',
        },
      ],
      DefaultCapacityProviderStrategy: [],
    });
  });

  test('throws when calling Cluster.addAsgCapacityProvider with an AsgCapacityProvider created with an imported ASG', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const importedAsg = autoscaling.AutoScalingGroup.fromAutoScalingGroupName(stack, 'ASG', 'my-asg');
    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup: importedAsg,
      enableManagedTerminationProtection: false,
    });
    // THEN
    expect(() => {
      cluster.addAsgCapacityProvider(capacityProvider);
    }).toThrow('Cannot configure the AutoScalingGroup because it is an imported resource.');
  });

  test('should throw an error if capacity provider with default strategy is not present in capacity providers', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        enableFargateCapacityProviders: true,
      }).addDefaultCapacityProviderStrategy([
        { capacityProvider: 'test capacityProvider', base: 10, weight: 50 },
      ]);
    }).toThrow('Capacity provider test capacityProvider must be added to the cluster with addAsgCapacityProvider() or addManagedInstancesCapacityProvider() before it can be used in a default capacity provider strategy.');
  });

  test('should throw an error when capacity providers is length 0 and default capacity provider startegy specified', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        enableFargateCapacityProviders: false,
      }).addDefaultCapacityProviderStrategy([
        { capacityProvider: 'test capacityProvider', base: 10, weight: 50 },
      ]);
    }).toThrow('Capacity provider test capacityProvider must be added to the cluster with addAsgCapacityProvider() or addManagedInstancesCapacityProvider() before it can be used in a default capacity provider strategy.');
  });

  test('should throw an error when more than 1 default capacity provider have base specified', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        enableFargateCapacityProviders: true,
      }).addDefaultCapacityProviderStrategy([
        { capacityProvider: 'FARGATE', base: 10, weight: 50 },
        { capacityProvider: 'FARGATE_SPOT', base: 10, weight: 50 },
      ]);
    }).toThrow(/Only 1 capacity provider in a capacity provider strategy can have a nonzero base./);
  });

  test('should throw an error when a capacity provider strategy contains a mix of Auto Scaling groups and Fargate providers', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });
    const cluster = new ecs.Cluster(stack, 'EcsCluster', {
      enableFargateCapacityProviders: true,
    });
    const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedTerminationProtection: false,
    });
    cluster.addAsgCapacityProvider(capacityProvider);

    // THEN
    expect(() => {
      cluster.addDefaultCapacityProviderStrategy([
        { capacityProvider: 'FARGATE', base: 10, weight: 50 },
        { capacityProvider: 'FARGATE_SPOT' },
        { capacityProvider: capacityProvider.capacityProviderName },
      ]);
    }).toThrow(/A capacity provider strategy cannot contain a mix of capacity providers using Auto Scaling groups and Fargate providers. Specify one or the other and try again./);
  });

  test('should throw an error if addDefaultCapacityProviderStrategy is called more than once', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      const cluster = new ecs.Cluster(stack, 'EcsCluster', {
        enableFargateCapacityProviders: true,
      });
      cluster.addDefaultCapacityProviderStrategy([
        { capacityProvider: 'FARGATE', base: 10, weight: 50 },
        { capacityProvider: 'FARGATE_SPOT' },
      ]);
      cluster.addDefaultCapacityProviderStrategy([
        { capacityProvider: 'FARGATE', base: 10, weight: 50 },
        { capacityProvider: 'FARGATE_SPOT' },
      ]);
    }).toThrow(/Cluster default capacity provider strategy is already set./);
  });

  test('can add ASG capacity via Capacity Provider with default capacity provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const cluster = new ecs.Cluster(stack, 'EcsCluster', {
      enableFargateCapacityProviders: true,
    });

    cluster.addDefaultCapacityProviderStrategy([
      { capacityProvider: 'FARGATE', base: 10, weight: 50 },
      { capacityProvider: 'FARGATE_SPOT' },
    ]);

    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedTerminationProtection: false,
    });

    cluster.addAsgCapacityProvider(capacityProvider);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      Cluster: {
        Ref: 'EcsCluster97242B84',
      },
      CapacityProviders: [
        'FARGATE',
        'FARGATE_SPOT',
        {
          Ref: 'providerD3FF4D3A',
        },
      ],
      DefaultCapacityProviderStrategy: [
        { CapacityProvider: 'FARGATE', Base: 10, Weight: 50 },
        { CapacityProvider: 'FARGATE_SPOT' },
      ],
    });
  });

  test('can add ASG default capacity provider', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const cluster = new ecs.Cluster(stack, 'EcsCluster');

    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'asg', {
      vpc,
      instanceType: new ec2.InstanceType('bogus'),
      machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
    });

    // WHEN
    const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provider', {
      autoScalingGroup,
      enableManagedTerminationProtection: false,
    });

    cluster.addAsgCapacityProvider(capacityProvider);

    cluster.addDefaultCapacityProviderStrategy([
      { capacityProvider: capacityProvider.capacityProviderName },
    ]);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
      Cluster: {
        Ref: 'EcsCluster97242B84',
      },
      CapacityProviders: [
        {
          Ref: 'providerD3FF4D3A',
        },
      ],
      DefaultCapacityProviderStrategy: [
        {
          CapacityProvider: {
            Ref: 'providerD3FF4D3A',
          },
        },
      ],
    });
  });

  test('correctly sets log configuration for execute command', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const kmsKey = new kms.Key(stack, 'KmsKey');

    const logGroup = new logs.LogGroup(stack, 'LogGroup', {
      encryptionKey: kmsKey,
    });

    const execBucket = new s3.Bucket(stack, 'EcsExecBucket', {
      encryptionKey: kmsKey,
    });

    // WHEN
    new ecs.Cluster(stack, 'EcsCluster', {
      executeCommandConfiguration: {
        kmsKey: kmsKey,
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

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ECS::Cluster', {
      Configuration: {
        ExecuteCommandConfiguration: {
          KmsKeyId: {
            'Fn::GetAtt': [
              'KmsKey46693ADD',
              'Arn',
            ],
          },
          LogConfiguration: {
            CloudWatchEncryptionEnabled: true,
            CloudWatchLogGroupName: {
              Ref: 'LogGroupF5B46931',
            },
            S3BucketName: {
              Ref: 'EcsExecBucket4F468651',
            },
            S3EncryptionEnabled: true,
            S3KeyPrefix: 'exec-output',
          },
          Logging: 'OVERRIDE',
        },
      },
    });
  });

  test('throws when no log configuration is provided when logging is set to OVERRIDE', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        executeCommandConfiguration: {
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
    }).toThrow(/Execute command log configuration must only be specified when logging is OVERRIDE./);
  });

  test('throws when log configuration provided but logging is set to DEFAULT', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    const logGroup = new logs.LogGroup(stack, 'LogGroup');

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        executeCommandConfiguration: {
          logConfiguration: {
            cloudWatchLogGroup: logGroup,
          },
          logging: ecs.ExecuteCommandLogging.DEFAULT,
        },
      });
    }).toThrow(/Execute command log configuration must only be specified when logging is OVERRIDE./);
  });

  test('throws when CloudWatchEncryptionEnabled without providing CloudWatch Logs log group name', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        executeCommandConfiguration: {
          logConfiguration: {
            cloudWatchEncryptionEnabled: true,
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
    }).toThrow(/You must specify a CloudWatch log group in the execute command log configuration to enable CloudWatch encryption./);
  });

  test('throws when S3EncryptionEnabled without providing S3 Bucket name', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test');
    acknowledgeTestValidationRules(stack);

    // THEN
    expect(() => {
      new ecs.Cluster(stack, 'EcsCluster', {
        executeCommandConfiguration: {
          logConfiguration: {
            s3EncryptionEnabled: true,
          },
          logging: ecs.ExecuteCommandLogging.OVERRIDE,
        },
      });
    }).toThrow(/You must specify an S3 bucket name in the execute command log configuration to enable S3 encryption./);
  });

  test('When importing ECS Cluster via Arn', () => {
    // GIVEN
    const stack = new cdk.Stack();
    acknowledgeTestValidationRules(stack);
    const clusterName = 'my-cluster';
    const region = 'service-region';
    const account = 'service-account';
    const cluster = ecs.Cluster.fromClusterArn(stack, 'Cluster', `arn:aws:ecs:${region}:${account}:cluster/${clusterName}`);

    // THEN
    expect(cluster.clusterName).toEqual(clusterName);
    expect(cluster.env.region).toEqual(region);
    expect(cluster.env.account).toEqual(account);
  });

  test('throws error when import ECS Cluster without resource name in arn', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // THEN
    expect(() => {
      ecs.Cluster.fromClusterArn(stack, 'Cluster', 'arn:aws:ecs:service-region:service-account:cluster');
    }).toThrow(/Missing required Cluster Name from Cluster ARN: /);
  });
});

test('can add ASG capacity via Capacity Provider by not specifying machineImageType', () => {
  // GIVEN
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'test');
  acknowledgeTestValidationRules(stack);
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const cluster = new ecs.Cluster(stack, 'EcsCluster');

  const autoScalingGroupAl2 = new autoscaling.AutoScalingGroup(stack, 'asgal2', {
    vpc,
    instanceType: new ec2.InstanceType('bogus'),
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  });

  const autoScalingGroupBottlerocket = new autoscaling.AutoScalingGroup(stack, 'asgBottlerocket', {
    vpc,
    instanceType: new ec2.InstanceType('bogus'),
    machineImage: new ecs.BottleRocketImage(),
  });

  // WHEN
  const capacityProviderAl2 = new ecs.AsgCapacityProvider(stack, 'provideral2', {
    autoScalingGroup: autoScalingGroupAl2,
    enableManagedTerminationProtection: false,
  });

  const capacityProviderBottlerocket = new ecs.AsgCapacityProvider(stack, 'providerBottlerocket', {
    autoScalingGroup: autoScalingGroupBottlerocket,
    enableManagedTerminationProtection: false,
    machineImageType: ecs.MachineImageType.BOTTLEROCKET,
  });

  cluster.enableFargateCapacityProviders();

  // Ensure not added twice
  cluster.addAsgCapacityProvider(capacityProviderAl2);
  cluster.addAsgCapacityProvider(capacityProviderAl2);

  // Add Bottlerocket ASG Capacity Provider
  cluster.addAsgCapacityProvider(capacityProviderBottlerocket);

  // THEN Bottlerocket LaunchConfiguration
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
    ImageId: {
      Ref: 'SsmParameterValueawsservicebottlerocketawsecs1x8664latestimageidC96584B6F00A464EAD1953AFF4B05118Parameter',

    },
    UserData: {
      'Fn::Base64': {
        'Fn::Join': [
          '',
          [
            '\n[settings.ecs]\ncluster = \"',
            {
              Ref: 'EcsCluster97242B84',
            },
            '\"',
          ],
        ],
      },
    },
  });

  // THEN AmazonLinux2 LaunchConfiguration
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
    ImageId: {
      Ref: 'SsmParameterValueawsserviceecsoptimizedamiamazonlinux2recommendedimageidC96584B6F00A464EAD1953AFF4B05118Parameter',
    },
    UserData: {
      'Fn::Base64': {
        'Fn::Join': [
          '',
          [
            '#!/bin/bash\necho ECS_CLUSTER=',
            {
              Ref: 'EcsCluster97242B84',

            },
            ' >> /etc/ecs/ecs.config',
          ],
        ],
      },
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::ECS::ClusterCapacityProviderAssociations', {
    CapacityProviders: [
      'FARGATE',
      'FARGATE_SPOT',
      {
        Ref: 'provideral2A427CBC0',
      },
      {
        Ref: 'providerBottlerocket90C039FA',
      },
    ],
    Cluster: {
      Ref: 'EcsCluster97242B84',
    },
    DefaultCapacityProviderStrategy: [],
  });
});

test('throws when ASG Capacity Provider with capacityProviderName starting with aws, ecs or fargate', () => {
  // GIVEN
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'test');
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const cluster = new ecs.Cluster(stack, 'EcsCluster');

  const autoScalingGroupAl2 = new autoscaling.AutoScalingGroup(stack, 'asgal2', {
    vpc,
    instanceType: new ec2.InstanceType('bogus'),
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  });

  // THEN
  expect(() => {
    // WHEN Capacity Provider define capacityProviderName start with aws.
    const capacityProviderAl2 = new ecs.AsgCapacityProvider(stack, 'provideral2', {
      autoScalingGroup: autoScalingGroupAl2,
      enableManagedTerminationProtection: false,
      capacityProviderName: 'awscp',
    });

    cluster.addAsgCapacityProvider(capacityProviderAl2);
  }).toThrow(/Invalid Capacity Provider Name: awscp, If a name is specified, it cannot start with aws, ecs, or fargate./);

  expect(() => {
    // WHEN Capacity Provider define capacityProviderName start with ecs.
    const capacityProviderAl2 = new ecs.AsgCapacityProvider(stack, 'provideral2-2', {
      autoScalingGroup: autoScalingGroupAl2,
      enableManagedTerminationProtection: false,
      capacityProviderName: 'ecscp',
    });

    cluster.addAsgCapacityProvider(capacityProviderAl2);
  }).toThrow(/Invalid Capacity Provider Name: ecscp, If a name is specified, it cannot start with aws, ecs, or fargate./);
});

test('throws when ASG Capacity Provider with no capacityProviderName but stack name starting with aws, ecs or fargate', () => {
  // GIVEN
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'ecscp');
  acknowledgeTestValidationRules(stack);
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const cluster = new ecs.Cluster(stack, 'EcsCluster');

  const autoScalingGroupAl2 = new autoscaling.AutoScalingGroup(stack, 'asgal2', {
    vpc,
    instanceType: new ec2.InstanceType('bogus'),
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  });

  expect(() => {
    // WHEN Capacity Provider when stack name starts with ecs.
    const capacityProvider = new ecs.AsgCapacityProvider(stack, 'provideral2-2', {
      autoScalingGroup: autoScalingGroupAl2,
      enableManagedTerminationProtection: false,
    });

    cluster.addAsgCapacityProvider(capacityProvider);
  }).not.toThrow();
});

test('throws when InstanceWarmupPeriod is less than 0', () => {
  // GIVEN
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'test');
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const cluster = new ecs.Cluster(stack, 'EcsCluster');

  const autoScalingGroupAl2 = new autoscaling.AutoScalingGroup(stack, 'asgal2', {
    vpc,
    instanceType: new ec2.InstanceType('t2.micro'),
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  });

  // THEN
  expect(() => {
    const capacityProviderAl2 = new ecs.AsgCapacityProvider(stack, 'provideral2', {
      autoScalingGroup: autoScalingGroupAl2,
      instanceWarmupPeriod: -1,
    });

    cluster.addAsgCapacityProvider(capacityProviderAl2);
  }).toThrow(/InstanceWarmupPeriod must be between 0 and 10000 inclusive, got: -1./);
});

test('throws when InstanceWarmupPeriod is greater than 10000', () => {
  // GIVEN
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'test');
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const cluster = new ecs.Cluster(stack, 'EcsCluster');

  const autoScalingGroupAl2 = new autoscaling.AutoScalingGroup(stack, 'asgal2', {
    vpc,
    instanceType: new ec2.InstanceType('t2.micro'),
    machineImage: ecs.EcsOptimizedImage.amazonLinux2(),
  });

  // THEN
  expect(() => {
    const capacityProviderAl2 = new ecs.AsgCapacityProvider(stack, 'provideral2', {
      autoScalingGroup: autoScalingGroupAl2,
      instanceWarmupPeriod: 99999,
    });

    cluster.addAsgCapacityProvider(capacityProviderAl2);
  }).toThrow(/InstanceWarmupPeriod must be between 0 and 10000 inclusive, got: 99999./);
});
