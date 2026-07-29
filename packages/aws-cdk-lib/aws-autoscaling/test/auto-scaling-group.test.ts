import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Annotations, Match, Template } from '../../assertions';
import * as cloudwatch from '../../aws-cloudwatch';
import * as ec2 from '../../aws-ec2';
import { AmazonLinuxCpuType, AmazonLinuxGeneration, AmazonLinuxImage, InstanceType, LaunchTemplate } from '../../aws-ec2';
import type { ApplicationListener } from '../../aws-elasticloadbalancingv2';
import { ApplicationLoadBalancer, ApplicationTargetGroup } from '../../aws-elasticloadbalancingv2';
import * as iam from '../../aws-iam';
import * as sns from '../../aws-sns';
import * as ssm from '../../aws-ssm';
import * as cdk from '../../core';
import { AUTOSCALING_GENERATE_LAUNCH_TEMPLATE } from '../../cx-api';
import * as autoscaling from '../lib';
import { OnDemandAllocationStrategy, SpotAllocationStrategy } from '../lib';

/* eslint-disable @stylistic/quote-props */

describe('auto scaling group', () => {
  test('default fleet', () => {
    const stack = getTestStack();
    const vpc = mockVpc(stack);

    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });

    Template.fromStack(stack).templateMatches({
      'Parameters': {
        'SsmParameterValueawsserviceamiamazonlinuxlatestamznamihvmx8664gp2C96584B6F00A464EAD1953AFF4B05118Parameter': {
          'Type': 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
          'Default': '/aws/service/ami-amazon-linux-latest/amzn-ami-hvm-x86_64-gp2',
        },
      },
      'Resources': {
        'MyFleetInstanceSecurityGroup774E8234': {
          'Type': 'AWS::EC2::SecurityGroup',
          'Properties': {
            'GroupDescription': 'TestStack/MyFleet/InstanceSecurityGroup',
            'SecurityGroupEgress': [
              {
                'CidrIp': '0.0.0.0/0',
                'Description': 'Allow all outbound traffic by default',
                'IpProtocol': '-1',
              },
            ],
            'Tags': [
              {
                'Key': 'Name',
                'Value': 'TestStack/MyFleet',
              },
            ],

            'VpcId': 'my-vpc',
          },
        },
        'MyFleetInstanceRole25A84AB8': {
          'Type': 'AWS::IAM::Role',
          'Properties': {
            'AssumeRolePolicyDocument': {
              'Statement': [
                {
                  'Action': 'sts:AssumeRole',
                  'Effect': 'Allow',
                  'Principal': {
                    'Service': 'ec2.amazonaws.com',
                  },
                },
              ],
              'Version': '2012-10-17',
            },
            'Tags': [
              {
                'Key': 'Name',
                'Value': 'TestStack/MyFleet',
              },
            ],
          },
        },
        'MyFleetInstanceProfile70A58496': {
          'Type': 'AWS::IAM::InstanceProfile',
          'Properties': {
            'Roles': [
              {
                'Ref': 'MyFleetInstanceRole25A84AB8',
              },
            ],
          },
        },
        'MyFleetLaunchConfig5D7F9801': {
          'Type': 'AWS::AutoScaling::LaunchConfiguration',
          'Properties': {
            'IamInstanceProfile': {
              'Ref': 'MyFleetInstanceProfile70A58496',
            },
            'ImageId': { 'Ref': 'SsmParameterValueawsserviceamiamazonlinuxlatestamznamihvmx8664gp2C96584B6F00A464EAD1953AFF4B05118Parameter' },
            'InstanceType': 'm4.micro',
            'SecurityGroups': [
              {
                'Fn::GetAtt': [
                  'MyFleetInstanceSecurityGroup774E8234',
                  'GroupId',
                ],
              },
            ],
            'UserData': {
              'Fn::Base64': '#!/bin/bash',
            },
          },
          'DependsOn': [
            'MyFleetInstanceRole25A84AB8',
          ],
        },
        'MyFleetASG88E55886': {
          'Type': 'AWS::AutoScaling::AutoScalingGroup',
          'UpdatePolicy': {
            'AutoScalingScheduledAction': {
              'IgnoreUnmodifiedGroupSizeProperties': true,
            },
          },
          'Properties': {
            'LaunchConfigurationName': {
              'Ref': 'MyFleetLaunchConfig5D7F9801',
            },
            'Tags': [
              {
                'Key': 'Name',
                'PropagateAtLaunch': true,
                'Value': 'TestStack/MyFleet',
              },
            ],

            'MaxSize': '1',
            'MinSize': '1',
            'VPCZoneIdentifier': [
              'pri1',
            ],
          },
        },
      },
    });
  });

  test('can create launch template from launch config props when @aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig is set', () => {
    // GIVEN
    const stack = getTestStack();
    stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
    const vpc = mockVpc(stack);
    const userData = ec2.UserData.forLinux();
    userData.addCommands('it me!');
    const blockDevices = [{
      deviceName: 'ebs',
      mappingEnabled: true,
      volume: autoscaling.BlockDeviceVolume.ebs(15, {
        deleteOnTermination: true,
        encrypted: true,
        volumeType: autoscaling.EbsDeviceVolumeType.IO1,
        iops: 5000,
      }),
    }, {
      deviceName: 'ebs-snapshot',
      volume: autoscaling.BlockDeviceVolume.ebsFromSnapshot('snapshot-id', {
        volumeSize: 500,
        deleteOnTermination: false,
        volumeType: autoscaling.EbsDeviceVolumeType.SC1,
      }),
    }];

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      machineImage: new ec2.AmazonLinuxImage(),
      keyName: 'key-name',
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      instanceMonitoring: autoscaling.Monitoring.DETAILED,
      securityGroup: mockSecurityGroup(stack),
      role: iam.Role.fromRoleArn(stack, 'ImportedRole', 'arn:aws:iam::123456789012:role/MockRole'),
      userData,
      associatePublicIpAddress: true,
      spotPrice: '0.05',
      blockDevices,
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
    });

    // THEN
    Template.fromStack(stack).templateMatches({
      Resources: {
        'MyFleetInstanceProfile70A58496': {
          Type: 'AWS::IAM::InstanceProfile',
          Properties: {
            Roles: [
              'MockRole',
            ],
          },
        },
        'MyFleetLaunchTemplate165E2717': {
          Type: 'AWS::EC2::LaunchTemplate',
          Properties: {
            LaunchTemplateData: {
              BlockDeviceMappings: [
                {
                  DeviceName: 'ebs',
                  Ebs: {
                    DeleteOnTermination: true,
                    Encrypted: true,
                    Iops: 5000,
                    VolumeSize: 15,
                    VolumeType: 'io1',
                  },
                },
                {
                  DeviceName: 'ebs-snapshot',
                  Ebs: {
                    DeleteOnTermination: false,
                    SnapshotId: 'snapshot-id',
                    VolumeSize: 500,
                    VolumeType: 'sc1',
                  },
                },
              ],
              IamInstanceProfile: {
                Arn: {
                  'Fn::GetAtt': [
                    'MyFleetInstanceProfile70A58496',
                    'Arn',
                  ],
                },
              },
              ImageId: {
                Ref: 'SsmParameterValueawsserviceamiamazonlinuxlatestamznamihvmx8664gp2C96584B6F00A464EAD1953AFF4B05118Parameter',
              },
              InstanceMarketOptions: {
                MarketType: 'spot',
                SpotOptions: {
                  MaxPrice: '0.05',
                },
              },
              InstanceType: 'm4.micro',
              KeyName: 'key-name',
              Monitoring: {
                Enabled: true,
              },
              NetworkInterfaces: [
                {
                  'AssociatePublicIpAddress': true,
                  'DeviceIndex': 0,
                  'Groups': [
                    'most-secure',
                  ],
                },
              ],
              TagSpecifications: [
                {
                  ResourceType: 'instance',
                  Tags: [
                    {
                      Key: 'Name',
                      Value: 'TestStack/MyFleet/LaunchTemplate',
                    },
                  ],
                },
                {
                  ResourceType: 'volume',
                  Tags: [
                    {
                      Key: 'Name',
                      Value: 'TestStack/MyFleet/LaunchTemplate',
                    },
                  ],
                },
              ],
              UserData: {
                'Fn::Base64': '#!/bin/bash\nit me!',
              },
            },
            TagSpecifications: [
              {
                ResourceType: 'launch-template',
                Tags: [
                  {
                    Key: 'Name',
                    Value: 'TestStack/MyFleet/LaunchTemplate',
                  },
                ],
              },
            ],
          },
        },
        'MyFleetASG88E55886': {
          Type: 'AWS::AutoScaling::AutoScalingGroup',
          Properties: {
            MaxSize: '1',
            MinSize: '1',
            LaunchTemplate: {
              LaunchTemplateId: {
                Ref: 'MyFleetLaunchTemplate165E2717',
              },
              Version: {
                'Fn::GetAtt': [
                  'MyFleetLaunchTemplate165E2717',
                  'LatestVersionNumber',
                ],
              },
            },
            VPCZoneIdentifier: [
              'pub1',
            ],
          },
          UpdatePolicy: {
            AutoScalingScheduledAction: {
              IgnoreUnmodifiedGroupSizeProperties: true,
            },
          },
        },
      },
      Parameters: {
        'SsmParameterValueawsserviceamiamazonlinuxlatestamznamihvmx8664gp2C96584B6F00A464EAD1953AFF4B05118Parameter': {
          Type: 'AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>',
          Default: '/aws/service/ami-amazon-linux-latest/amzn-ami-hvm-x86_64-gp2',
        },
      },
    });
  });

  test('can add security group to a launch template when @aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig is set', () => {
    // GIVEN
    const stack = getTestStack();
    stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
    const vpc = mockVpc(stack);

    // WHEN
    const autoScalingGroup = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      machineImage: new ec2.AmazonLinuxImage(),
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      securityGroup: mockSecurityGroup(stack),
      vpc,
    });
    autoScalingGroup.addSecurityGroup(new ec2.SecurityGroup(stack, 'AddedSG', { vpc }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        SecurityGroupIds: [
          'most-secure',
          {
            'Fn::GetAtt': [
              'AddedSG710A1221',
              'GroupId',
            ],
          },
        ],
      },
    });
  });

  test('can set minCapacity, maxCapacity, desiredCapacity to 0', () => {
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: 0,
      maxCapacity: 0,
      desiredCapacity: 0,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MinSize: '0',
      MaxSize: '0',
      DesiredCapacity: '0',
    });
  });

  test('validation is not performed when using Tokens', () => {
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    cdk.Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::E3706',
      reason: 'Post-synth validation would otherwise still catch this',
    });
    const vpc = mockVpc(stack);

    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: cdk.Lazy.number({ produce: () => 5 }),
      maxCapacity: cdk.Lazy.number({ produce: () => 1 }),
      desiredCapacity: cdk.Lazy.number({ produce: () => 20 }),
    });

    // THEN: no exception
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MinSize: '5',
      MaxSize: '1',
      DesiredCapacity: '20',
    });
  });

  test('maxCapacity defaults to minCapacity when using Token', () => {
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: cdk.Token.asNumber(ssm.StringParameter.valueForStringParameter(stack, '/Min')),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MinSize: { Ref: 'SsmParameterValueMinC96584B6F00A464EAD1953AFF4B05118Parameter' },
      MaxSize: { Ref: 'SsmParameterValueMinC96584B6F00A464EAD1953AFF4B05118Parameter' },
    });
  });

  test('userdata can be overridden by image', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    const ud = ec2.UserData.forLinux();
    ud.addCommands('it me!');

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage({
        userData: ud,
      }),
      vpc,
    });

    // THEN
    expect(asg.userData.render()).toEqual('#!/bin/bash\nit me!');
  });

  test('userdata can be overridden at ASG directly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    const ud1 = ec2.UserData.forLinux();
    ud1.addCommands('it me!');

    const ud2 = ec2.UserData.forLinux();
    ud2.addCommands('no me!');

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage({
        userData: ud1,
      }),
      vpc,
      userData: ud2,
    });

    // THEN
    expect(asg.userData.render()).toEqual('#!/bin/bash\nno me!');
  });

  test('can specify only min capacity', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: 10,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MinSize: '10',
      MaxSize: '10',
    });
  });

  test('can specify only max capacity', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      maxCapacity: 10,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MinSize: '1',
      MaxSize: '10',
    });
  });

  test('can specify only desiredCount', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      desiredCapacity: 10,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MinSize: '1',
      MaxSize: '10',
      DesiredCapacity: '10',
    });
  });

  test('can specify only defaultInstanceWarmup', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      defaultInstanceWarmup: cdk.Duration.seconds(5),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      DefaultInstanceWarmup: 5,
    });
  });

  test('addToRolePolicy can be used to add statements to the role policy', () => {
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    const fleet = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });

    fleet.addToRolePolicy(new iam.PolicyStatement({
      actions: ['test:SpecialName'],
      resources: ['*'],
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 'test:SpecialName',
            Effect: 'Allow',
            Resource: '*',
          },
        ],
      },
    });
  });

  testDeprecated('can configure replacing update', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      updateType: autoscaling.UpdateType.REPLACING_UPDATE,
      replacingUpdateMinSuccessfulInstancesPercent: 50,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        AutoScalingReplacingUpdate: {
          WillReplace: true,
        },
      },
      CreationPolicy: {
        AutoScalingCreationPolicy: {
          MinSuccessfulInstancesPercent: 50,
        },
      },
    });
  });

  testDeprecated('can configure rolling update', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      updateType: autoscaling.UpdateType.ROLLING_UPDATE,
      rollingUpdateConfiguration: {
        minSuccessfulInstancesPercent: 50,
        pauseTime: cdk.Duration.seconds(345),
      },
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      UpdatePolicy: {
        'AutoScalingRollingUpdate': {
          'MinSuccessfulInstancesPercent': 50,
          'WaitOnResourceSignals': true,
          'PauseTime': 'PT5M45S',
          'SuspendProcesses': ['HealthCheck', 'ReplaceUnhealthy', 'AZRebalance', 'AlarmNotification', 'ScheduledActions', 'InstanceRefresh'],
        },
      },
    });
  });

  testDeprecated('can configure resource signals', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      resourceSignalCount: 5,
      resourceSignalTimeout: cdk.Duration.seconds(666),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
      CreationPolicy: {
        ResourceSignal: {
          Count: 5,
          Timeout: 'PT11M6S',
        },
      },
    });
  });

  testDeprecated('can configure EC2 health check', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      healthCheck: autoscaling.HealthCheck.ec2(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      HealthCheckType: 'EC2',
    });
  });

  testDeprecated('can configure ELB health check', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      healthCheck: autoscaling.HealthCheck.elb({ grace: cdk.Duration.minutes(15) }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      HealthCheckType: 'ELB',
      HealthCheckGracePeriod: 900,
    });
  });

  test.each([
    cdk.Duration.seconds(100),
    undefined,
  ])('can configure EC2 health checks with gracePeriod is %s', (gracePeriod) => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      healthChecks: autoscaling.HealthChecks.ec2({
        gracePeriod,
      }),
    });

    const expectedGrace = gracePeriod ? gracePeriod.toSeconds() : Match.absent();

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      HealthCheckType: 'EC2',
      HealthCheckGracePeriod: expectedGrace,
    });
  });

  test.each([
    cdk.Duration.seconds(100),
    undefined,
  ])('can configure additional health checks with gracePeriod is %s', (gracePeriod) => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      healthChecks: autoscaling.HealthChecks.withAdditionalChecks({
        gracePeriod,
        additionalTypes: [
          autoscaling.AdditionalHealthCheckType.EBS,
          autoscaling.AdditionalHealthCheckType.ELB,
          autoscaling.AdditionalHealthCheckType.VPC_LATTICE,
        ],
      }),
    });

    const expectedGrace = gracePeriod ? gracePeriod.toSeconds() : Match.absent();

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      HealthCheckType: 'EBS,ELB,VPC_LATTICE',
      HealthCheckGracePeriod: expectedGrace,
    });
  });

  test('throws if both healthCheck and healthChecks are specified.', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        healthCheck: autoscaling.HealthCheck.ec2(),
        healthChecks: autoscaling.HealthChecks.withAdditionalChecks({
          gracePeriod: cdk.Duration.seconds(100),
          additionalTypes: [
            autoscaling.AdditionalHealthCheckType.EBS,
          ],
        }),
      });
    }).toThrow(/Cannot specify both 'healthCheck' and 'healthChecks'. Please use 'healthChecks' only./);
  });

  test('throws when additionalTypes array for additional health checks is empty', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        healthChecks: autoscaling.HealthChecks.withAdditionalChecks({
          gracePeriod: cdk.Duration.seconds(100),
          additionalTypes: [],
        }),
      });
    }).toThrow(/At least one health check type must be specified in 'additionalTypes' for 'healthChecks'/);
  });

  test('can add Security Group to Fleet', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
    const vpc = mockVpc(stack);

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });
    asg.addSecurityGroup(mockSecurityGroup(stack));
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      SecurityGroups: [
        {
          'Fn::GetAtt': [
            'MyFleetInstanceSecurityGroup774E8234',
            'GroupId',
          ],
        },
        'most-secure',
      ],
    });
  });

  test('can set tags', () => {
    // GIVEN
    const stack = getTestStack();
    const vpc = mockVpc(stack);

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      updatePolicy: autoscaling.UpdatePolicy.rollingUpdate(),
    });

    cdk.Tags.of(asg).add('superfood', 'acai');
    cdk.Tags.of(asg).add('notsuper', 'caramel', { applyToLaunchedInstances: false });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      Tags: [
        {
          Key: 'Name',
          PropagateAtLaunch: true,
          Value: 'TestStack/MyFleet',
        },
        {
          Key: 'notsuper',
          PropagateAtLaunch: false,
          Value: 'caramel',
        },
        {
          Key: 'superfood',
          PropagateAtLaunch: true,
          Value: 'acai',
        },
      ],
    });
  });

  test('allows setting spot price', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      spotPrice: '0.05',
    });

    // THEN
    expect(asg.spotPrice).toEqual('0.05');
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      SpotPrice: '0.05',
    });
  });

  test('allows association of public IP address', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: 0,
      maxCapacity: 0,
      desiredCapacity: 0,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      associatePublicIpAddress: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      AssociatePublicIpAddress: true,
    });
  });

  test('association of public IP address requires public subnet', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        minCapacity: 0,
        maxCapacity: 0,
        desiredCapacity: 0,
        associatePublicIpAddress: true,
      });
    }).toThrow();
  });

  test('allows disassociation of public IP address', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: 0,
      maxCapacity: 0,
      desiredCapacity: 0,
      associatePublicIpAddress: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      AssociatePublicIpAddress: false,
    });
  });

  test('does not specify public IP address association by default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      minCapacity: 0,
      maxCapacity: 0,
      desiredCapacity: 0,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      AssociatePublicIpAddress: Match.absent(),
    });
  });

  test('an existing security group can be specified instead of auto-created', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const securityGroup = mockSecurityGroup(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      securityGroup,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      SecurityGroups: ['most-secure'],
    });
  });

  test('an existing role can be specified instead of auto-created', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const importedRole = iam.Role.fromRoleArn(stack, 'ImportedRole', 'arn:aws:iam::123456789012:role/HelloDude');

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      role: importedRole,
    });

    // THEN
    expect(asg.role).toEqual(importedRole);
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::InstanceProfile', {
      'Roles': ['HelloDude'],
    });
  });

  test('defaultChild is available on an ASG', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });

    // THEN
    expect(asg.node.defaultChild instanceof autoscaling.CfnAutoScalingGroup).toEqual(true);
  });

  testDeprecated('can set blockDeviceMappings', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      blockDevices: [{
        deviceName: 'ebs',
        mappingEnabled: true,
        volume: autoscaling.BlockDeviceVolume.ebs(15, {
          deleteOnTermination: true,
          encrypted: true,
          volumeType: autoscaling.EbsDeviceVolumeType.IO1,
          iops: 5000,
        }),
      }, {
        deviceName: 'ebs-snapshot',
        volume: autoscaling.BlockDeviceVolume.ebsFromSnapshot('snapshot-id', {
          volumeSize: 500,
          deleteOnTermination: false,
          volumeType: autoscaling.EbsDeviceVolumeType.SC1,
        }),
      }, {
        deviceName: 'ephemeral',
        volume: autoscaling.BlockDeviceVolume.ephemeral(0),
      }, {
        deviceName: 'disabled',
        volume: autoscaling.BlockDeviceVolume.ephemeral(1),
        mappingEnabled: false,
      }, {
        deviceName: 'none',
        volume: autoscaling.BlockDeviceVolume.noDevice(),
      }, {
        deviceName: 'gp3-with-throughput',
        volume: autoscaling.BlockDeviceVolume.ebs(15, {
          volumeType: autoscaling.EbsDeviceVolumeType.GP3,
          throughput: 350,
        }),
      }],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      BlockDeviceMappings: [
        {
          DeviceName: 'ebs',
          Ebs: {
            DeleteOnTermination: true,
            Encrypted: true,
            Iops: 5000,
            VolumeSize: 15,
            VolumeType: 'io1',
          },
          NoDevice: Match.absent(),
        },
        {
          DeviceName: 'ebs-snapshot',
          Ebs: {
            DeleteOnTermination: false,
            SnapshotId: 'snapshot-id',
            VolumeSize: 500,
            VolumeType: 'sc1',
          },
          NoDevice: Match.absent(),
        },
        {
          DeviceName: 'ephemeral',
          VirtualName: 'ephemeral0',
          NoDevice: Match.absent(),
        },
        {
          DeviceName: 'disabled',
          NoDevice: true,
        },
        {
          DeviceName: 'none',
          NoDevice: true,
        },
        {
          DeviceName: 'gp3-with-throughput',
          Ebs: {
            VolumeSize: 15,
            VolumeType: 'gp3',
            Throughput: 350,
          },
        },
      ],
    });
  });

  test('can configure maxInstanceLifetime', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      maxInstanceLifetime: cdk.Duration.days(7),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      'MaxInstanceLifetime': 604800,
    });
  });

  test('can configure maxInstanceLifetime with 0', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      maxInstanceLifetime: cdk.Duration.days(0),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      'MaxInstanceLifetime': 0,
    });
  });

  test('throws if maxInstanceLifetime < 1 day', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        maxInstanceLifetime: cdk.Duration.hours(23),
      });
    }).toThrow(/maxInstanceLifetime must be between 1 and 365 days \(inclusive\)/);
  });

  test('throws if maxInstanceLifetime > 365 days', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        maxInstanceLifetime: cdk.Duration.days(366),
      });
    }).toThrow(/maxInstanceLifetime must be between 1 and 365 days \(inclusive\)/);
  });

  test.each([124, 2001])('throws if throughput is set less than 125 or more than 2000', (throughput) => {
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        machineImage: new ec2.AmazonLinuxImage(),
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        vpc,
        maxInstanceLifetime: cdk.Duration.days(0),
        blockDevices: [{
          deviceName: 'ebs',
          volume: autoscaling.BlockDeviceVolume.ebs(15, {
            volumeType: autoscaling.EbsDeviceVolumeType.GP3,
            throughput,
          }),
        }],
      });
    }).toThrow(/throughput property takes a minimum of 125 and a maximum of 2000/);
  });

  test.each([
    ...Object.values(autoscaling.EbsDeviceVolumeType).filter((v) => v !== 'gp3'),
  ])('throws if throughput is set on any volume type other than GP3', (volumeType) => {
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        machineImage: new ec2.AmazonLinuxImage(),
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        vpc,
        maxInstanceLifetime: cdk.Duration.days(0),
        blockDevices: [{
          deviceName: 'ebs',
          volume: autoscaling.BlockDeviceVolume.ebs(15, {
            volumeType: volumeType,
            throughput: 150,
          }),
        }],
      });
    }).toThrow(/throughput property requires volumeType: EbsDeviceVolumeType.GP3/);
  });

  test('throws if throughput / iops ratio is greater than 0.25', () => {
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        machineImage: new ec2.AmazonLinuxImage(),
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        vpc,
        maxInstanceLifetime: cdk.Duration.days(0),
        blockDevices: [{
          deviceName: 'ebs',
          volume: autoscaling.BlockDeviceVolume.ebs(15, {
            volumeType: autoscaling.EbsDeviceVolumeType.GP3,
            throughput: 751,
            iops: 3000,
          }),
        }],
      });
    }).toThrow('Throughput (MiBps) to iops ratio of 0.25033333333333335 is too high; maximum is 0.25 MiBps per iops');
  });

  test('can configure instance monitoring', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      instanceMonitoring: autoscaling.Monitoring.BASIC,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      InstanceMonitoring: false,
    });
  });

  test('instance monitoring defaults to absent', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      InstanceMonitoring: Match.absent(),
    });
  });

  test('throws if ephemeral volumeIndex < 0', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        blockDevices: [{
          deviceName: 'ephemeral',
          volume: autoscaling.BlockDeviceVolume.ephemeral(-1),
        }],
      });
    }).toThrow(/volumeIndex must be a number starting from 0/);
  });

  test('throws if volumeType === IO1 without iops', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyStack', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        blockDevices: [{
          deviceName: 'ebs',
          volume: autoscaling.BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: autoscaling.EbsDeviceVolumeType.IO1,
          }),
        }],
      });
    }).toThrow(/ops property is required with volumeType: EbsDeviceVolumeType.IO1/);
  });

  test('warning if iops without volumeType', () => {
    // GIVEN
    const stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::W3671',
      reason: 'We have our own warning',
    });
    const vpc = mockVpc(stack);

    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      blockDevices: [{
        deviceName: 'ebs',
        volume: autoscaling.BlockDeviceVolume.ebs(15, {
          deleteOnTermination: true,
          encrypted: true,
          iops: 5000,
        }),
      }],
    });

    // THEN
    Annotations.fromStack(stack).hasWarning('/Default/MyStack', 'iops will be ignored without volumeType: EbsDeviceVolumeType.IO1 [ack: @aws-cdk/aws-autoscaling:iopsIgnored]');
  });

  test('warning if iops and volumeType !== IO1', () => {
    // GIVEN
    const stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::W3671',
      reason: 'We have our own warning',
    });
    const vpc = mockVpc(stack);

    new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      blockDevices: [{
        deviceName: 'ebs',
        volume: autoscaling.BlockDeviceVolume.ebs(15, {
          deleteOnTermination: true,
          encrypted: true,
          volumeType: autoscaling.EbsDeviceVolumeType.GP2,
          iops: 5000,
        }),
      }],
    });

    // THEN
    Annotations.fromStack(stack).hasWarning('/Default/MyStack', 'iops will be ignored without volumeType: EbsDeviceVolumeType.IO1 [ack: @aws-cdk/aws-autoscaling:iopsIgnored]');
  });

  test('step scaling on metric', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });

    // WHEN
    asg.scaleOnMetric('Metric', {
      metric: new cloudwatch.Metric({
        namespace: 'Test',
        metricName: 'Metric',
      }),
      adjustmentType: autoscaling.AdjustmentType.CHANGE_IN_CAPACITY,
      scalingSteps: [
        { change: -1, lower: 0, upper: 49 },
        { change: 0, lower: 50, upper: 99 },
        { change: 1, lower: 100 },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
      ComparisonOperator: 'LessThanOrEqualToThreshold',
      EvaluationPeriods: 1,
      MetricName: 'Metric',
      Namespace: 'Test',
      Period: 300,
    });
  });

  test('step scaling on MathExpression', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyStack', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });

    // WHEN
    asg.scaleOnMetric('Metric', {
      metric: new cloudwatch.MathExpression({
        expression: 'a',
        usingMetrics: {
          a: new cloudwatch.Metric({
            namespace: 'Test',
            metricName: 'Metric',
          }),
        },
      }),
      adjustmentType: autoscaling.AdjustmentType.CHANGE_IN_CAPACITY,
      scalingSteps: [
        { change: -1, lower: 0, upper: 49 },
        { change: 0, lower: 50, upper: 99 },
        { change: 1, lower: 100 },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', Match.not({
      Period: 60,
    }));

    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
      'ComparisonOperator': 'LessThanOrEqualToThreshold',
      'EvaluationPeriods': 1,
      'Metrics': [
        {
          'Expression': 'a',
          'Id': 'expr_1',
        },
        {
          'Id': 'a',
          'MetricStat': {
            'Metric': {
              'MetricName': 'Metric',
              'Namespace': 'Test',
            },
            'Period': 300,
            'Stat': 'Average',
          },
          'ReturnData': false,
        },
      ],
      'Threshold': 49,
    });
  });

  test('test GroupMetrics.all(), adds a single MetricsCollection with no Metrics specified', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    // When
    new autoscaling.AutoScalingGroup(stack, 'ASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      groupMetrics: [autoscaling.GroupMetrics.all()],
    });

    // Then
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MetricsCollection: [
        {
          Granularity: '1Minute',
          Metrics: Match.absent(),
        },
      ],
    });
  });

  test('test can specify a subset of group metrics', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'ASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      groupMetrics: [
        new autoscaling.GroupMetrics(autoscaling.GroupMetric.MIN_SIZE,
          autoscaling.GroupMetric.MAX_SIZE,
          autoscaling.GroupMetric.DESIRED_CAPACITY,
          autoscaling.GroupMetric.IN_SERVICE_INSTANCES),
        new autoscaling.GroupMetrics(autoscaling.GroupMetric.PENDING_INSTANCES,
          autoscaling.GroupMetric.STANDBY_INSTANCES,
          autoscaling.GroupMetric.TOTAL_INSTANCES,
          autoscaling.GroupMetric.TERMINATING_INSTANCES,
        ),
      ],
      vpc,
    });

    // Then
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MetricsCollection: [
        {
          Granularity: '1Minute',
          Metrics: ['GroupMinSize', 'GroupMaxSize', 'GroupDesiredCapacity', 'GroupInServiceInstances'],
        }, {
          Granularity: '1Minute',
          Metrics: ['GroupPendingInstances', 'GroupStandbyInstances', 'GroupTotalInstances', 'GroupTerminatingInstances'],
        },
      ],
    });
  });

  test('test deduplication of group metrics ', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    new autoscaling.AutoScalingGroup(stack, 'ASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      groupMetrics: [new autoscaling.GroupMetrics(autoscaling.GroupMetric.MIN_SIZE,
        autoscaling.GroupMetric.MAX_SIZE,
        autoscaling.GroupMetric.MAX_SIZE,
        autoscaling.GroupMetric.MIN_SIZE,
      )],
    });

    // Then
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MetricsCollection: [
        {
          Granularity: '1Minute',
          Metrics: ['GroupMinSize', 'GroupMaxSize'],
        },
      ],
    });
  });

  test('allow configuring notifications', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const topic = new sns.Topic(stack, 'MyTopic');

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      notifications: [
        {
          topic,
          scalingEvents: autoscaling.ScalingEvents.ERRORS,
        },
        {
          topic,
          scalingEvents: new autoscaling.ScalingEvents(autoscaling.ScalingEvent.INSTANCE_TERMINATE),
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NotificationConfigurations: [
        {
          TopicARN: { Ref: 'MyTopic86869434' },
          NotificationTypes: [
            'autoscaling:EC2_INSTANCE_LAUNCH_ERROR',
            'autoscaling:EC2_INSTANCE_TERMINATE_ERROR',
          ],
        },
        {
          TopicARN: { Ref: 'MyTopic86869434' },
          NotificationTypes: [
            'autoscaling:EC2_INSTANCE_TERMINATE',
          ],
        },
      ],
    });
  });

  testDeprecated('throw if notification and notificationsTopics are both configured', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const topic = new sns.Topic(stack, 'MyTopic');

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyASG', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
        notificationsTopic: topic,
        notifications: [{
          topic,
        }],
      });
    }).toThrow('Cannot set \'notificationsTopic\' and \'notifications\', \'notificationsTopic\' is deprecated use \'notifications\' instead');
  });

  test('notificationTypes default includes all non test NotificationType', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const topic = new sns.Topic(stack, 'MyTopic');

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      notifications: [
        {
          topic,
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NotificationConfigurations: [
        {
          TopicARN: { Ref: 'MyTopic86869434' },
          NotificationTypes: [
            'autoscaling:EC2_INSTANCE_LAUNCH',
            'autoscaling:EC2_INSTANCE_LAUNCH_ERROR',
            'autoscaling:EC2_INSTANCE_TERMINATE',
            'autoscaling:EC2_INSTANCE_TERMINATE_ERROR',
          ],
        },
      ],
    });
  });

  testDeprecated('setting notificationTopic configures all non test NotificationType', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const topic = new sns.Topic(stack, 'MyTopic');

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      notificationsTopic: topic,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NotificationConfigurations: [
        {
          TopicARN: { Ref: 'MyTopic86869434' },
          NotificationTypes: [
            'autoscaling:EC2_INSTANCE_LAUNCH',
            'autoscaling:EC2_INSTANCE_LAUNCH_ERROR',
            'autoscaling:EC2_INSTANCE_TERMINATE',
            'autoscaling:EC2_INSTANCE_TERMINATE_ERROR',
          ],
        },
      ],
    });
  });

  test('NotificationTypes.ALL includes all non test NotificationType', () => {
    expect(Object.values(autoscaling.ScalingEvent).length - 1).toEqual(autoscaling.ScalingEvents.ALL._types.length);
  });

  test('Can set Capacity Rebalancing via constructor property', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      capacityRebalance: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      CapacityRebalance: true,
    });
  });

  test('Can protect new instances from scale-in via constructor property', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      newInstancesProtectedFromScaleIn: true,
    });

    // THEN
    expect(asg.areNewInstancesProtectedFromScaleIn()).toEqual(true);
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NewInstancesProtectedFromScaleIn: true,
    });
  });

  test('Can protect new instances from scale-in via setter', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
    });
    asg.protectNewInstancesFromScaleIn();

    // THEN
    expect(asg.areNewInstancesProtectedFromScaleIn()).toEqual(true);
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      NewInstancesProtectedFromScaleIn: true,
    });
  });

  testDeprecated('requires imdsv2', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      vpc,
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ec2.MachineImage.latestAmazonLinux(),
      requireImdsv2: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::LaunchConfiguration', {
      MetadataOptions: {
        HttpTokens: 'required',
      },
    });
  });

  testDeprecated('supports termination policies', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      vpc,
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ec2.MachineImage.latestAmazonLinux(),
      terminationPolicies: [
        autoscaling.TerminationPolicy.OLDEST_INSTANCE,
        autoscaling.TerminationPolicy.DEFAULT,
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      TerminationPolicies: [
        'OldestInstance',
        'Default',
      ],
    });
  });

  test('supports custom termination policy with lambda function arn specified', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const arn = stack.formatArn({
      service: 'lambda',
      resource: 'function',
      account: '123456789012',
      region: 'us-east-1',
      partition: 'aws',
      resourceName: 'CustomTerminationPolicyLambda:1',
    });

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyASG', {
      vpc,
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: new ec2.AmazonLinuxImage(),
      terminationPolicies: [
        autoscaling.TerminationPolicy.CUSTOM_LAMBDA_FUNCTION,
        autoscaling.TerminationPolicy.OLDEST_INSTANCE,
        autoscaling.TerminationPolicy.DEFAULT,
      ],
      terminationPolicyCustomLambdaFunctionArn: arn,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      TerminationPolicies: [
        arn,
        'OldestInstance',
        'Default',
      ],
    });
  });

  test('Should specify TerminationPolicy.CUSTOM_LAMBDA_FUNCTION in first', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    const terminationPolicies = [
      autoscaling.TerminationPolicy.DEFAULT,
      autoscaling.TerminationPolicy.CUSTOM_LAMBDA_FUNCTION,
    ];

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: new ec2.AmazonLinuxImage(),
        terminationPolicies: terminationPolicies,
      });
    }).toThrow('TerminationPolicy.CUSTOM_LAMBDA_FUNCTION must be specified first in the termination policies');
  });

  test('Should specify terminationPolicyCustomLambdaFunctionArn property if TerminationPolicy.CUSTOM_LAMBDA_FUNCTION is used', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    const terminationPolicies = [
      autoscaling.TerminationPolicy.CUSTOM_LAMBDA_FUNCTION,
    ];

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: new ec2.AmazonLinuxImage(),
        terminationPolicies: terminationPolicies,
      });
    }).toThrow('terminationPolicyCustomLambdaFunctionArn property must be specified if the TerminationPolicy.CUSTOM_LAMBDA_FUNCTION is used');
  });

  test('Can use imported Launch Template with ID', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
      launchTemplate: LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
        launchTemplateId: 'test-lt-id',
        versionNumber: '0',
      }),
      vpc: mockVpc(stack),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      LaunchTemplate: {
        LaunchTemplateId: 'test-lt-id',
        Version: '0',
      },
    });
  });

  test('Can use imported Launch Template with name', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
      launchTemplate: LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
        launchTemplateName: 'test-lt',
        versionNumber: '0',
      }),
      vpc: mockVpc(stack),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      LaunchTemplate: {
        LaunchTemplateName: 'test-lt',
        Version: '0',
      },
    });
  });

  test('Can use in-stack Launch Template reference', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = new LaunchTemplate(stack, 'lt', {
      instanceType: new InstanceType('t3.micro'),
      machineImage: new AmazonLinuxImage({
        generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
        cpuType: AmazonLinuxCpuType.X86_64,
      }),
    });

    new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
      launchTemplate: lt,
      vpc: mockVpc(stack),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      LaunchTemplate: {
        LaunchTemplateId: {
          'Ref': 'ltB6511CF5',
        },
        Version: {
          'Fn::GetAtt': [
            'ltB6511CF5',
            'LatestVersionNumber',
          ],
        },
      },
    });
  });

  test('Can use mixed instance policy', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });

    new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
      mixedInstancesPolicy: {
        launchTemplate: lt,
        launchTemplateOverrides: [{
          instanceType: new InstanceType('t4g.micro'),
          launchTemplate: lt,
          weightedCapacity: 9,
        }],
        instancesDistribution: {
          onDemandAllocationStrategy: OnDemandAllocationStrategy.PRIORITIZED,
          onDemandBaseCapacity: 1,
          onDemandPercentageAboveBaseCapacity: 2,
          spotAllocationStrategy: SpotAllocationStrategy.CAPACITY_OPTIMIZED_PRIORITIZED,
          spotInstancePools: 3,
          spotMaxPrice: '4',
        },
      },
      vpc: mockVpc(stack),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MixedInstancesPolicy: {
        LaunchTemplate: {
          LaunchTemplateSpecification: {
            LaunchTemplateId: 'test-lt-id',
            Version: '0',
          },
          Overrides: [
            {
              InstanceType: 't4g.micro',
              LaunchTemplateSpecification: {
                LaunchTemplateId: 'test-lt-id',
                Version: '0',
              },
              WeightedCapacity: '9',
            },
          ],
        },
        InstancesDistribution: {
          OnDemandAllocationStrategy: 'prioritized',
          OnDemandBaseCapacity: 1,
          OnDemandPercentageAboveBaseCapacity: 2,
          SpotAllocationStrategy: 'capacity-optimized-prioritized',
          SpotInstancePools: 3,
          SpotMaxPrice: '4',
        },
      },
    });
  });

  test('Can use mixed instance policy without instances distribution', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });

    new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
      mixedInstancesPolicy: {
        launchTemplate: lt,
        launchTemplateOverrides: [{
          instanceType: new InstanceType('t4g.micro'),
          launchTemplate: lt,
          weightedCapacity: 9,
        }],
      },
      vpc: mockVpc(stack),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MixedInstancesPolicy: {
        LaunchTemplate: {
          LaunchTemplateSpecification: {
            LaunchTemplateId: 'test-lt-id',
            Version: '0',
          },
          Overrides: [
            {
              InstanceType: 't4g.micro',
              LaunchTemplateSpecification: {
                LaunchTemplateId: 'test-lt-id',
                Version: '0',
              },
              WeightedCapacity: '9',
            },
          ],
        },
      },
    });
  });

  test('Can specify InstanceRequirements', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });

    new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
      mixedInstancesPolicy: {
        launchTemplate: lt,
        launchTemplateOverrides: [{
          instanceRequirements: {
            vCpuCount: { min: 4, max: 8 },
            memoryMiB: { min: 16384 },
            cpuManufacturers: ['intel'],
          },
          launchTemplate: lt,
          weightedCapacity: 9,
        }],
      },
      vpc: mockVpc(stack),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      MixedInstancesPolicy: {
        LaunchTemplate: {
          LaunchTemplateSpecification: {
            LaunchTemplateId: 'test-lt-id',
            Version: '0',
          },
          Overrides: [
            {
              InstanceRequirements: {
                VCpuCount: {
                  Min: 4,
                  Max: 8,
                },
                MemoryMiB: {
                  Min: 16384,
                },
                CpuManufacturers: ['intel'],
              },
              LaunchTemplateSpecification: {
                LaunchTemplateId: 'test-lt-id',
                Version: '0',
              },
              WeightedCapacity: '9',
            },
          ],
        },
      },
    });
  });

  test('Cannot specify InstanceRequirements and InstanceType at the same time', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
        mixedInstancesPolicy: {
          launchTemplate: lt,
          launchTemplateOverrides: [{
            instanceRequirements: {
              vCpuCount: { min: 4, max: 8 },
              memoryMiB: { min: 16384 },
              cpuManufacturers: ['intel'],
            },
            instanceType: new InstanceType('t4g.micro'),
            launchTemplate: lt,
            weightedCapacity: 9,
          }],
        },
        vpc: mockVpc(stack),
      });
    }).toThrow('You can specify either \'instanceRequirements\' or \'instanceType\', not both.');
  });

  test('Should specify either InstanceRequirements or InstanceType', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
        mixedInstancesPolicy: {
          launchTemplate: lt,
          launchTemplateOverrides: [{
            launchTemplate: lt,
            weightedCapacity: 9,
          }],
        },
        vpc: mockVpc(stack),
      });
    }).toThrow('You must specify either \'instanceRequirements\' or \'instanceType\'.');
  });

  test('Cannot specify both Launch Template and Launch Config', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });
    const vpc = mockVpc(stack);

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
        launchTemplate: lt,
        instanceType: new InstanceType('t3.micro'),
        machineImage: new AmazonLinuxImage({
          generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
          cpuType: AmazonLinuxCpuType.X86_64,
        }),
        vpc,
      });
    }).toThrow('Setting \'machineImage\' must not be set when \'launchTemplate\' or \'mixedInstancesPolicy\' is set');
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg-2', {
        launchTemplate: lt,
        associatePublicIpAddress: true,
        vpc,
      });
    }).toThrow('Setting \'associatePublicIpAddress\' must not be set when \'launchTemplate\' or \'mixedInstancesPolicy\' is set');
  });

  test('Cannot specify Launch Template without instance type', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = new LaunchTemplate(stack, 'lt', {
      machineImage: new AmazonLinuxImage({
        generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
        cpuType: AmazonLinuxCpuType.X86_64,
      }),
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
        launchTemplate: lt,
        vpc: mockVpc(stack),
      });
    }).toThrow('Setting \'launchTemplate\' requires its \'instanceType\' to be set');
  });

  test('Cannot specify Launch Template without machine image', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = new LaunchTemplate(stack, 'lt', {
      instanceType: new InstanceType('t3.micro'),
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
        launchTemplate: lt,
        vpc: mockVpc(stack),
      });
    }).toThrow('Setting \'launchTemplate\' requires its \'machineImage\' to be set');
  });

  test('Cannot specify mixed instance policy without machine image', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const lt = new LaunchTemplate(stack, 'lt', {
      instanceType: new InstanceType('t3.micro'),
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
        mixedInstancesPolicy: {
          launchTemplate: lt,
          launchTemplateOverrides: [{
            instanceType: new InstanceType('t3.micro'),
          }],
        },
        vpc: mockVpc(stack),
      });
    }).toThrow('Setting \'mixedInstancesPolicy.launchTemplate\' requires its \'machineImage\' to be set');
  });

  test('Cannot be created with launch configuration without machine image', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
        instanceType: new InstanceType('t3.micro'),
        vpc: mockVpc(stack),
      });
    }).toThrow('Setting \'machineImage\' is required when \'launchTemplate\' and \'mixedInstancesPolicy\' is not set');
  });

  test('Cannot be created with launch configuration without instance type', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
        machineImage: new AmazonLinuxImage({
          generation: AmazonLinuxGeneration.AMAZON_LINUX_2,
          cpuType: AmazonLinuxCpuType.X86_64,
        }),
        vpc: mockVpc(stack),
      });
    }).toThrow('Setting \'instanceType\' is required when \'launchTemplate\' and \'mixedInstancesPolicy\' is not set');
  });

  test('Should throw when accessing inferred fields with imported Launch Template', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
      launchTemplate: LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
        launchTemplateId: 'test-lt-id',
        versionNumber: '0',
      }),
      vpc: mockVpc(stack),
    });

    // THEN
    expect(() => {
      asg.userData;
    }).toThrow('The provided launch template does not expose its user data.');

    expect(() => {
      asg.connections;
    }).toThrow('AutoScalingGroup can only be used as IConnectable if it is not created from an imported Launch Template.');

    expect(() => {
      asg.role;
    }).toThrow('The provided launch template does not expose or does not define its role.');

    expect(() => {
      asg.addSecurityGroup(mockSecurityGroup(stack));
    }).toThrow('You cannot add security groups when the Auto Scaling Group is created from a Launch Template.');
  });

  test('Should throw when accessing inferred fields with in-stack Launch Template not having corresponding properties', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
      launchTemplate: new LaunchTemplate(stack, 'in-stack-lt', {
        instanceType: new ec2.InstanceType('t3.micro'),
        machineImage: new ec2.AmazonLinuxImage({
          generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
          cpuType: ec2.AmazonLinuxCpuType.X86_64,
        }),
      }),
      vpc: mockVpc(stack),
    });

    // THEN
    expect(() => {
      asg.userData;
    }).toThrow('The provided launch template does not expose its user data.');

    expect(() => {
      asg.connections;
    }).toThrow('LaunchTemplate can only be used as IConnectable if a securityGroup is provided when constructing it.');

    expect(() => {
      asg.role;
    }).toThrow('The provided launch template does not expose or does not define its role.');

    expect(() => {
      asg.addSecurityGroup(mockSecurityGroup(stack));
    }).toThrow('You cannot add security groups when the Auto Scaling Group is created from a Launch Template.');
  });

  test('Should not throw when accessing inferred fields with in-stack Launch Template having corresponding properties', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const asg = new autoscaling.AutoScalingGroup(stack, 'imported-lt-asg', {
      launchTemplate: new LaunchTemplate(stack, 'in-stack-lt', {
        instanceType: new ec2.InstanceType('t3.micro'),
        machineImage: new ec2.AmazonLinuxImage({
          generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
          cpuType: ec2.AmazonLinuxCpuType.X86_64,
        }),
        userData: ec2.UserData.forLinux(),
        securityGroup: mockSecurityGroup(stack),
        role: iam.Role.fromRoleArn(stack, 'ImportedRole', 'arn:aws:iam::123456789012:role/HelloDude'),
      }),
      vpc: mockVpc(stack),
    });

    // THEN
    expect(() => {
      asg.userData;
    }).not.toThrow();

    expect(() => {
      asg.connections;
    }).not.toThrow();

    expect(() => {
      asg.role;
    }).not.toThrow();

    expect(() => {
      asg.addSecurityGroup(mockSecurityGroup(stack));
    }).toThrow('You cannot add security groups when the Auto Scaling Group is created from a Launch Template.');
  });

  test('Should not throw when LaunchTemplate is used with CloudformationInit', () => {
    const stack = new cdk.Stack();

    // WHEN
    const lt = new LaunchTemplate(stack, 'LaunchTemplate', {
      machineImage: new AmazonLinuxImage(),
      instanceType: new ec2.InstanceType('t3.micro'),
      userData: ec2.UserData.forLinux(),
      securityGroup: ec2.SecurityGroup.fromSecurityGroupId(stack, 'ImportedSg', 'securityGroupId'),
      role: iam.Role.fromRoleArn(stack, 'ImportedRole', 'arn:aws:iam::123456789012:role/MockRole'),
    });

    const cfInit = ec2.CloudFormationInit.fromElements(
      ec2.InitCommand.shellCommand('/bash'),
    );

    // THEN
    expect(() => new autoscaling.AutoScalingGroup(stack, 'Asg', {
      launchTemplate: lt,
      init: cfInit,
      vpc: mockVpc(stack),
      signals: autoscaling.Signals.waitForAll(),
    })).not.toThrow();
  });

  describe('multiple target groups', () => {
    let asg: autoscaling.AutoScalingGroup;
    let stack: cdk.Stack;
    let vpc: ec2.IVpc;
    let alb: ApplicationLoadBalancer;
    let listener: ApplicationListener;

    beforeEach(() => {
      stack = new cdk.Stack(undefined, 'MyStack', { env: { region: 'us-east-1', account: '1234' } });
      vpc = mockVpc(stack);
      alb = new ApplicationLoadBalancer(stack, 'alb', {
        vpc,
        internetFacing: true,
      });

      listener = alb.addListener('Listener', {
        port: 80,
        open: true,
      });

      asg = new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        vpc,
      });
    });

    test('Adding two application target groups should succeed validation', () => {
      const atg1 = new ApplicationTargetGroup(stack, 'ATG1', { port: 443 });
      const atg2 = new ApplicationTargetGroup(stack, 'ATG2', { port: 443 });

      listener.addTargetGroups('tgs', { targetGroups: [atg1, atg2] });

      asg.attachToApplicationTargetGroup(atg1);
      asg.attachToApplicationTargetGroup(atg2);

      expect(asg.node.validate()).toEqual([]);
    });

    test('Adding two application target groups should fail validation validate if `scaleOnRequestCount()` has been called', () => {
      const atg1 = new ApplicationTargetGroup(stack, 'ATG1', { port: 443 });
      const atg2 = new ApplicationTargetGroup(stack, 'ATG2', { port: 443 });

      listener.addTargetGroups('tgs', { targetGroups: [atg1, atg2] });

      asg.attachToApplicationTargetGroup(atg1);
      asg.attachToApplicationTargetGroup(atg2);

      asg.scaleOnRequestCount('requests-per-minute', { targetRequestsPerMinute: 60 });

      expect(asg.node.validate()).toContainEqual('Cannon use multiple target groups if `scaleOnRequestCount()` is being used.');
    });
  });

  test('can configure keyPair', () => {
    // GIVE
    const stack = new cdk.Stack();
    stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
    const vpc = mockVpc(stack);
    const keyPair = new ec2.KeyPair(stack, 'MyKeyPair', { keyPairName: 'MyKeyPair' });

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyAutoScalingGroup', {
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      keyPair: keyPair,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        KeyName: {
          Ref: 'MyKeyPair659209CA',
        },
      },
    });
  });

  test('keyPair should throw if @aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig is false', () => {
    // WHEN
    const stack = new cdk.Stack();
    stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, false);
    const vpc = mockVpc(stack);
    const keyPair = new ec2.KeyPair(stack, 'MyKeyPair', { keyPairName: 'MyKeyPair' });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyASG', {
        vpc,
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        keyPair: keyPair,
      });
    }).toThrow("Can only use \'keyPair\' when feature flag \'AUTOSCALING_GENERATE_LAUNCH_TEMPLATE\' is set");
  });

  test('keyPair and keyName cannot be defined together', () => {
    // WHEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const keyPair = new ec2.KeyPair(stack, 'MyKeyPair', { keyPairName: 'MyKeyPair' });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyASG', {
        vpc,
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
        machineImage: new ec2.AmazonLinuxImage(),
        keyName: 'MyKeyPair',
        keyPair: keyPair,
      });
    }).toThrow("Cannot specify both of \'keyName\' and \'keyPair\'; prefer \'keyPair\'");
  });

  test('keyPair and launchTemplate cannot be defined together', () => {
    // WHEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const keyPair = new ec2.KeyPair(stack, 'MyKeyPair', { keyPairName: 'MyKeyPair' });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyASG', {
        vpc,
        launchTemplate: new LaunchTemplate(stack, 'in-stack-lt', {
          instanceType: new ec2.InstanceType('t3.micro'),
          machineImage: new ec2.AmazonLinuxImage({
            generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
            cpuType: ec2.AmazonLinuxCpuType.X86_64,
          }),
        }),
        keyPair: keyPair,
      });
    }).toThrow("Setting \'keyPair\' must not be set when \'launchTemplate\' or \'mixedInstancesPolicy\' is set");
  });

  test('keyName and mixedInstancesPolicy cannot be defined together', () => {
    // WHEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    const keyPair = new ec2.KeyPair(stack, 'MyKeyPair', { keyPairName: 'MyKeyPair' });
    const lt = new LaunchTemplate(stack, 'in-stack-lt', {
      instanceType: new ec2.InstanceType('t3.micro'),
      machineImage: new ec2.AmazonLinuxImage({
        generation: ec2.AmazonLinuxGeneration.AMAZON_LINUX_2,
        cpuType: ec2.AmazonLinuxCpuType.X86_64,
      }),
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
        mixedInstancesPolicy: {
          launchTemplate: lt,
          launchTemplateOverrides: [{
            instanceType: new InstanceType('t4g.micro'),
            launchTemplate: lt,
            weightedCapacity: 9,
          }],
          instancesDistribution: {
            onDemandAllocationStrategy: OnDemandAllocationStrategy.PRIORITIZED,
            onDemandBaseCapacity: 1,
            onDemandPercentageAboveBaseCapacity: 2,
            spotAllocationStrategy: SpotAllocationStrategy.CAPACITY_OPTIMIZED_PRIORITIZED,
            spotInstancePools: 3,
            spotMaxPrice: '4',
          },
        },
        keyPair: keyPair,
        vpc,
      });
    }).toThrow("Setting \'keyPair\' must not be set when \'launchTemplate\' or \'mixedInstancesPolicy\' is set");
  });
});

function mockVpc(stack: cdk.Stack) {
  return ec2.Vpc.fromVpcAttributes(stack, 'MyVpc', {
    vpcId: 'my-vpc',
    availabilityZones: ['az1'],
    publicSubnetIds: ['pub1'],
    privateSubnetIds: ['pri1'],
    isolatedSubnetIds: [],
  });
}

test('Can set autoScalingGroupName', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyASG', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    autoScalingGroupName: 'MyAsg',
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
    AutoScalingGroupName: 'MyAsg',
  });
});

test('can use Vpc imported from unparseable list tokens', () => {
  // GIVEN
  const stack = new cdk.Stack();

  const vpcId = cdk.Fn.importValue('myVpcId');
  const availabilityZones = cdk.Fn.split(',', cdk.Fn.importValue('myAvailabilityZones'));
  const publicSubnetIds = cdk.Fn.split(',', cdk.Fn.importValue('myPublicSubnetIds'));
  const privateSubnetIds = cdk.Fn.split(',', cdk.Fn.importValue('myPrivateSubnetIds'));
  const isolatedSubnetIds = cdk.Fn.split(',', cdk.Fn.importValue('myIsolatedSubnetIds'));

  const vpc = ec2.Vpc.fromVpcAttributes(stack, 'importedVpc', {
    vpcId,
    availabilityZones,
    publicSubnetIds,
    privateSubnetIds,
    isolatedSubnetIds,
  });

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'ecs-ec2-asg', {
    instanceType: new ec2.InstanceType('t2.micro'),
    machineImage: new ec2.AmazonLinuxImage(),
    minCapacity: 1,
    maxCapacity: 1,
    desiredCapacity: 1,
    vpc,
    allowAllOutbound: false,
    associatePublicIpAddress: false,
    vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
    VPCZoneIdentifier: {
      'Fn::Split': [',', { 'Fn::ImportValue': 'myPrivateSubnetIds' }],
    },
  });
});

test('add price-capacity-optimized', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
    launchTemplateId: 'test-lt-id',
    versionNumber: '0',
  });

  new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
    mixedInstancesPolicy: {
      launchTemplate: lt,
      launchTemplateOverrides: [{
        instanceType: new InstanceType('t4g.micro'),
        launchTemplate: lt,
        weightedCapacity: 9,
      }],
      instancesDistribution: {
        onDemandAllocationStrategy: OnDemandAllocationStrategy.PRIORITIZED,
        onDemandBaseCapacity: 1,
        onDemandPercentageAboveBaseCapacity: 2,
        spotAllocationStrategy: SpotAllocationStrategy.PRICE_CAPACITY_OPTIMIZED,
        spotInstancePools: 3,
        spotMaxPrice: '4',
      },
    },
    vpc: mockVpc(stack),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
    MixedInstancesPolicy: {
      InstancesDistribution: {
        SpotAllocationStrategy: 'price-capacity-optimized',
      },
    },
  });
});

test('add on-demand lowest-price allocation strategy', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
    launchTemplateId: 'test-lt-id',
    versionNumber: '0',
  });

  new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
    mixedInstancesPolicy: {
      launchTemplate: lt,
      launchTemplateOverrides: [{
        instanceType: new InstanceType('t4g.micro'),
        launchTemplate: lt,
        weightedCapacity: 9,
      }],
      instancesDistribution: {
        onDemandAllocationStrategy: OnDemandAllocationStrategy.LOWEST_PRICE,
        onDemandBaseCapacity: 1,
        onDemandPercentageAboveBaseCapacity: 100,
      },
    },
    vpc: mockVpc(stack),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
    MixedInstancesPolicy: {
      InstancesDistribution: {
        OnDemandAllocationStrategy: 'lowest-price',
      },
    },
  });
});

test('ssm permissions adds right managed policy', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
    vpc: mockVpc(stack),
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE),
    ssmSessionPermissions: true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
    ManagedPolicyArns: [
      {
        'Fn::Join': ['', [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':iam::aws:policy/AmazonSSMManagedInstanceCore',
        ]],
      },
    ],
  });
});

test('ssm permissions adds right managed policy with launch template', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  const role = new iam.Role(stack, 'role', {
    assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
  });

  const lt = new LaunchTemplate(stack, 'launch-template', {
    machineImage: ec2.MachineImage.latestAmazonLinux2(),
    instanceType: InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE),
    role: role,
  });

  new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
    vpc: mockVpc(stack),
    launchTemplate: lt,
    ssmSessionPermissions: true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
    ManagedPolicyArns: [
      {
        'Fn::Join': ['', [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':iam::aws:policy/AmazonSSMManagedInstanceCore',
        ]],
      },
    ],
  });
});

test('ssm permissions adds right managed policy with mixed instance policy', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  const role = new iam.Role(stack, 'role', {
    assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
  });

  const lt = new LaunchTemplate(stack, 'launch-template', {
    machineImage: ec2.MachineImage.latestAmazonLinux2(),
    instanceType: InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE),
    role: role,
  });

  new autoscaling.AutoScalingGroup(stack, 'mip-asg', {
    vpc: mockVpc(stack),
    mixedInstancesPolicy: {
      instancesDistribution: {
        onDemandPercentageAboveBaseCapacity: 50,
      },
      launchTemplate: lt,
      launchTemplateOverrides: [
        { instanceType: new ec2.InstanceType('t3.micro') },
        { instanceType: new ec2.InstanceType('t3a.micro') },
      ],
    },
    ssmSessionPermissions: true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
    ManagedPolicyArns: [
      {
        'Fn::Join': ['', [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':iam::aws:policy/AmazonSSMManagedInstanceCore',
        ]],
      },
    ],
  });
});

test('requires imdsv2 when @aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig is set', () => {
  // GIVEN
  const stack = new cdk.Stack();
  stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    vpc,
    instanceType: new ec2.InstanceType('t2.micro'),
    machineImage: ec2.MachineImage.latestAmazonLinux2(),
    requireImdsv2: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
    LaunchTemplateData: {
      MetadataOptions: {
        HttpTokens: 'required',
      },
    },
  });
});

describe('InstanceMaintenancePolicy', () => {
  test('maxHealthyPercentage and minHealthyPercentage can be specified', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    new autoscaling.AutoScalingGroup(stack, 'ASG', {
      vpc,
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ec2.MachineImage.latestAmazonLinux2(),
      maxHealthyPercentage: 200,
      minHealthyPercentage: 100,
    });

    // Then
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      InstanceMaintenancePolicy: {
        MaxHealthyPercentage: 200,
        MinHealthyPercentage: 100,
      },
    });
  });

  test('maxHealthyPercentage and minHealthyPercentage can be set to -1', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);
    new autoscaling.AutoScalingGroup(stack, 'ASG', {
      vpc,
      instanceType: new ec2.InstanceType('t2.micro'),
      machineImage: ec2.MachineImage.latestAmazonLinux2(),
      maxHealthyPercentage: -1,
      minHealthyPercentage: -1,
    });

    // Then
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      InstanceMaintenancePolicy: {
        MaxHealthyPercentage: -1,
        MinHealthyPercentage: -1,
      },
    });
  });

  test('can specify capacityDistributionStrategy', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // WHEN
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
      machineImage: new ec2.AmazonLinuxImage(),
      vpc,
      azCapacityDistributionStrategy: autoscaling.CapacityDistributionStrategy.BALANCED_ONLY,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
      AvailabilityZoneDistribution: {
        CapacityDistributionStrategy: 'balanced-only',
      },
    });
  });

  test('throws if maxHealthyPercentage is greater than 200', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 250,
        minHealthyPercentage: 100,
      });
    }).toThrow(/maxHealthyPercentage must be between 100 and 200, or -1 to clear the previously set value, got 250/);
  });

  test('throws if maxHealthyPercentage is less than 100', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 50,
        minHealthyPercentage: 100,
      });
    }).toThrow(/maxHealthyPercentage must be between 100 and 200, or -1 to clear the previously set value, got 50/);
  });

  test('throws if minHealthyPercentage is greater than 100', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 200,
        minHealthyPercentage: 150,
      });
    }).toThrow(/minHealthyPercentage must be between 0 and 100, or -1 to clear the previously set value, got 150/);
  });

  test('throws if minHealthyPercentage is less than 0', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 200,
        minHealthyPercentage: -100,
      });
    }).toThrow(/minHealthyPercentage must be between 0 and 100, or -1 to clear the previously set value, got -100/);
  });

  test('throws if only minHealthyPercentage is set to -1', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 200,
        minHealthyPercentage: -1,
      });
    }).toThrow(/Both minHealthyPercentage and maxHealthyPercentage must be -1 to clear the previously set value, got minHealthyPercentage: -1 and maxHealthyPercentage: 200/);
  });

  test('throws if only maxHealthyPercentage is set to -1', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: -1,
        minHealthyPercentage: 100,
      });
    }).toThrow(/Both minHealthyPercentage and maxHealthyPercentage must be -1 to clear the previously set value, got minHealthyPercentage: 100 and maxHealthyPercentage: -1/);
  });

  test('throws if only minHealthyPercentage is specified', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        minHealthyPercentage: 100,
      });
    }).toThrow(/Both or neither of minHealthyPercentage and maxHealthyPercentage must be specified, got minHealthyPercentage: 100 and maxHealthyPercentage: undefined/);
  });

  test('throws if only maxHealthyPercentage is specified', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 200,
      });
    }).toThrow(/Both or neither of minHealthyPercentage and maxHealthyPercentage must be specified, got minHealthyPercentage: undefined and maxHealthyPercentage: 200/);
  });

  test('throws if a difference between minHealthyPercentage and maxHealthyPercentage is greater than 100', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = mockVpc(stack);

    // Then
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'ASG', {
        vpc,
        instanceType: new ec2.InstanceType('t2.micro'),
        machineImage: ec2.MachineImage.latestAmazonLinux2(),
        maxHealthyPercentage: 200,
        minHealthyPercentage: 0,
      });
    }).toThrow(/The difference between minHealthyPercentage and maxHealthyPercentage cannot be greater than 100, got 200/);
  });

  test('throws if requireImdsv2 set when launchTemplate is set', () => {
    // GIVEN
    const stack = new cdk.Stack();
    stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
    const vpc = mockVpc(stack);
    const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
      launchTemplateId: 'test-lt-id',
      versionNumber: '0',
    });

    // THEN
    expect(() => {
      new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
        vpc,
        launchTemplate: lt,
        requireImdsv2: true,
      });
    }).toThrow(/Setting \'requireImdsv2\' must not be set when \'launchTemplate\' or \'mixedInstancesPolicy\' is set/);
  });
});

test('throws if updatePolicy is not set when migrateToLaunchTemplate is true', () => {
  // GIVEN
  const stack = new cdk.Stack();
  stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
  const vpc = mockVpc(stack);
  const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
    launchTemplateId: 'test-lt-id',
    versionNumber: '0',
  });

  // THEN
  expect(() => {
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      vpc,
      launchTemplate: lt,
      migrateToLaunchTemplate: true,
    });
  }).toThrow(/When migrateToLaunchTemplate is true, you must use AutoScalingRollingUpdate to ensure instances are properly replaced during migration. This prevents instances from referencing a deleted IAM instance profile./);
});

test('throws if updatePolicy is set with AutoScalingReplacingUpdate when migrateToLaunchTemplate is true', () => {
  // GIVEN
  const stack = new cdk.Stack();
  stack.node.setContext(AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
  const vpc = mockVpc(stack);
  const lt = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'imported-lt', {
    launchTemplateId: 'test-lt-id',
    versionNumber: '0',
  });

  // THEN
  expect(() => {
    new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
      vpc,
      launchTemplate: lt,
      migrateToLaunchTemplate: true,
      updatePolicy: autoscaling.UpdatePolicy.replacingUpdate(),
    });
  }).toThrow(/When migrateToLaunchTemplate is true, you must use AutoScalingRollingUpdate to ensure instances are properly replaced during migration. This prevents instances from referencing a deleted IAM instance profile./);
});

function mockSecurityGroup(stack: cdk.Stack) {
  const existing = stack.node.tryFindChild('MySG');
  if (existing) {
    return existing as ec2.ISecurityGroup;
  }
  cdk.Validations.of(stack).acknowledge({
    id: 'CloudFormation-Validate::E1150',
    reason: 'Using a bogus Security Group Id',
  });
  return ec2.SecurityGroup.fromSecurityGroupId(stack, 'MySG', 'most-secure');
}

function getTestStack(): cdk.Stack {
  return new cdk.Stack(undefined, 'TestStack', { env: { account: '1234', region: 'us-east-1' } });
}

test.each([
  [autoscaling.TerminateHookAbandonAction.RETAIN, 'retain'],
  [autoscaling.TerminateHookAbandonAction.TERMINATE, 'terminate'],
  [undefined, Match.absent()],
])('can configure instanceLifecyclePolicy with %s', (terminateHookAbandon, expectedValue) => {
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  new autoscaling.AutoScalingGroup(stack, `MyASG-${expectedValue}`, {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    instanceLifecyclePolicy: {
      retentionTriggers: {
        terminateHookAbandon,
      },
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
    InstanceLifecyclePolicy: {
      RetentionTriggers: {
        TerminateHookAbandon: expectedValue,
      },
    },
  });
});

test.each([
  [autoscaling.DeletionProtection.NONE, 'none'],
  [autoscaling.DeletionProtection.PREVENT_FORCE_DELETION, 'prevent-force-deletion'],
  [autoscaling.DeletionProtection.PREVENT_ALL_DELETION, 'prevent-all-deletion'],
])('can configure deletion protection with %s', (deletionProtection, expectedValue) => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    deletionProtection,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::AutoScaling::AutoScalingGroup', {
    DeletionProtection: expectedValue,
  });
});

test('can configure instance refresh update policy with only strategy (Rolling)', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    }),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
    UpdatePolicy: {
      AutoScalingInstanceRefresh: {
        Strategy: 'Rolling',
      },
    },
  });
});

test('can configure instance refresh update policy with all options', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  const metric = new cloudwatch.Metric({ namespace: 'MyService', metricName: 'MyMetric' });
  const alarm1 = new cloudwatch.Alarm(stack, 'Alarm1', { metric, threshold: 100, evaluationPeriods: 1 });
  const alarm2 = new cloudwatch.Alarm(stack, 'Alarm2', { metric, threshold: 100, evaluationPeriods: 1 });
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
      minHealthyPercentage: 90,
      maxHealthyPercentage: 110,
      instanceWarmup: cdk.Duration.seconds(300),
      skipMatching: true,
      checkpointPercentages: [20, 50, 100],
      checkpointDelay: cdk.Duration.seconds(3600),
      bakeTime: cdk.Duration.seconds(600),
      alarms: [alarm1, alarm2],
      scaleInProtectedInstances: autoscaling.ScaleInProtectedInstances.WAIT,
      standbyInstances: autoscaling.StandbyInstances.IGNORE,
    }),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
    UpdatePolicy: {
      AutoScalingInstanceRefresh: {
        Strategy: 'Rolling',
        Preferences: {
          MinHealthyPercentage: 90,
          MaxHealthyPercentage: 110,
          InstanceWarmup: 300,
          SkipMatching: true,
          CheckpointPercentages: [20, 50, 100],
          CheckpointDelay: 3600,
          BakeTime: 600,
          AlarmSpecification: {
            Alarms: [
              { Ref: 'Alarm1F9009D71' },
              { Ref: 'Alarm2A7122E13' },
            ],
          },
          ScaleInProtectedInstances: 'Wait',
          StandbyInstances: 'Ignore',
        },
      },
    },
  });
});

test('can configure instance refresh with only strategy (ReplaceRootVolume)', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.REPLACE_ROOT_VOLUME,
    }),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
    UpdatePolicy: {
      AutoScalingInstanceRefresh: {
        Strategy: 'ReplaceRootVolume',
      },
    },
  });
});

test('instance refresh allows minHealthyPercentage on its own', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
      minHealthyPercentage: 50,
    }),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
    UpdatePolicy: {
      AutoScalingInstanceRefresh: {
        Strategy: 'Rolling',
        Preferences: {
          MinHealthyPercentage: 50,
        },
      },
    },
  });
});

test.each([-1, 101])('instance refresh fails when minHealthyPercentage is out of range (%d)', (value) => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({ strategy: autoscaling.InstanceRefreshStrategy.ROLLING, minHealthyPercentage: value }))
    .toThrow(/minHealthyPercentage must be between 0 and 100/);
});

test.each([99, 201])('instance refresh fails when maxHealthyPercentage is out of range (%d)', (value) => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    minHealthyPercentage: 50,
    maxHealthyPercentage: value,
  })).toThrow(/maxHealthyPercentage must be between 100 and 200/);
});

test('instance refresh fails when maxHealthyPercentage is set without minHealthyPercentage', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({ strategy: autoscaling.InstanceRefreshStrategy.ROLLING, maxHealthyPercentage: 150 }))
    .toThrow(/maxHealthyPercentage requires minHealthyPercentage to be specified/);
});

test('instance refresh fails when the difference between min and max healthy percentage exceeds 100', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    minHealthyPercentage: 0,
    maxHealthyPercentage: 200,
  })).toThrow(/the difference between minHealthyPercentage and maxHealthyPercentage cannot be greater than 100/);
});

test('instance refresh fails when bakeTime exceeds the maximum', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    bakeTime: cdk.Duration.seconds(172801),
  })).toThrow(/bakeTime must be between 0 and 172800 seconds/);
});

test('instance refresh fails when checkpointDelay is out of range', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    checkpointPercentages: [100],
    checkpointDelay: cdk.Duration.seconds(172801),
  })).toThrow(/checkpointDelay must be between 0 and 172800 seconds/);
});

test('instance refresh fails when checkpointDelay is set without checkpointPercentages', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    checkpointDelay: cdk.Duration.minutes(10),
  })).toThrow(/checkpointDelay requires checkpointPercentages to be specified/);
});

test.each([[[0, 100]], [[50, 101]]])('instance refresh fails when checkpointPercentages are out of range (%j)', (percentages) => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    checkpointPercentages: percentages,
  })).toThrow(/each value in checkpointPercentages must be between 1 and 100/);
});

test('instance refresh fails when checkpointPercentages contains duplicates', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    checkpointPercentages: [50, 50, 100],
  })).toThrow(/each value in checkpointPercentages must be unique/);
});

test('instance refresh fails when checkpointPercentages are not in ascending order', () => {
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    checkpointPercentages: [100, 50],
  })).toThrow(/checkpointPercentages must be in ascending order/);
});

test('instance refresh fails when more than 10 alarms are specified', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const metric = new cloudwatch.Metric({ namespace: 'MyService', metricName: 'MyMetric' });
  const alarms = Array.from({ length: 11 }, (_, i) =>
    new cloudwatch.Alarm(stack, `Alarm${i}`, { metric, threshold: 100, evaluationPeriods: 1 }));

  // THEN
  expect(() => autoscaling.UpdatePolicy.instanceRefresh({
    strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    alarms,
  })).toThrow(/a maximum of 10 alarms can be specified, got 11/);
});

test('instance refresh omits AlarmSpecification when no alarms are provided', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
      alarms: [],
    }),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::AutoScaling::AutoScalingGroup', {
    UpdatePolicy: {
      AutoScalingInstanceRefresh: {
        Strategy: 'Rolling',
        Preferences: Match.absent(),
      },
    },
  });
});

test('warns when signals are combined with instance refresh update policy', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    signals: autoscaling.Signals.waitForAll(),
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    }),
  });

  // THEN
  Annotations.fromStack(stack).hasWarning('/Default/MyFleet', 'Instance refresh doesn\'t support the cfn-signal helper script. For information about how to verify instance readiness during an instance refresh, see Verify instance readiness during an instance refresh. https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-attribute-updatepolicy.html#cfn-attributes-updatepolicy-instancerefresh-readiness [ack: @aws-cdk/aws-autoscaling:signalsNotUsedByInstanceRefresh]');
});

test('does not warn about signals when instance refresh is used without signals', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = mockVpc(stack);

  // WHEN
  new autoscaling.AutoScalingGroup(stack, 'MyFleet', {
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.M4, ec2.InstanceSize.MICRO),
    machineImage: new ec2.AmazonLinuxImage(),
    vpc,
    updatePolicy: autoscaling.UpdatePolicy.instanceRefresh({
      strategy: autoscaling.InstanceRefreshStrategy.ROLLING,
    }),
  });

  // THEN
  Annotations.fromStack(stack).hasNoWarning('/Default/MyFleet', Match.stringLikeRegexp('.*signalsNotUsedByInstanceRefresh.*'));
});
