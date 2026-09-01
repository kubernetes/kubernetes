import * as path from 'path';
import { Annotations, Match, Template } from '../../assertions';
import { InstanceProfile, Role, ServicePrincipal } from '../../aws-iam';
import { Key } from '../../aws-kms';
import { Asset } from '../../aws-s3-assets';
import { StringParameter } from '../../aws-ssm';
import { App, Stack, Duration, Validations, Size } from '../../core';
import * as cxapi from '../../cx-api';
import type { LaunchTemplate } from '../lib';
import {
  AmazonLinuxImage,
  BlockDeviceVolume,
  CloudFormationInit,
  EbsDeviceVolumeType,
  InitCommand,
  Instance,
  InstanceArchitecture,
  InstanceClass,
  InstanceSize,
  InstanceType,
  HttpTokens,
  UserData,
  Vpc,
  SubnetType,
  SecurityGroup,
  WindowsImage,
  WindowsVersion,
  KeyPair,
  KeyPairType,
  CpuCredits,
  InstanceInitiatedShutdownBehavior,
  PlacementGroup,
} from '../lib';

let stack: Stack;
let vpc: Vpc;
beforeEach(() => {
  stack = new Stack();
  vpc = new Vpc(stack, 'VPC');
});

describe('instance', () => {
  test('instance is created with source/dest check switched off', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      sourceDestCheck: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      InstanceType: 't3.large',
      SourceDestCheck: false,
    });
  });
  test('instance is grantable', () => {
    // GIVEN
    const param = new StringParameter(stack, 'Param', { stringValue: 'Foobar' });
    const instance = new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    });

    // WHEN
    param.grantRead(instance);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'ssm:DescribeParameters',
              'ssm:GetParameters',
              'ssm:GetParameter',
              'ssm:GetParameterHistory',
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
                  ':ssm:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':parameter/',
                  {
                    Ref: 'Param165332EC',
                  },
                ],
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });
  test('instance architecture is correctly discerned for arm instances', () => {
    // GIVEN
    const sampleInstanceClasses = [
      'a1', 't4g', 'c6g', 'c7g', 'c6gd', 'c6gn', 'c7g', 'c7gd', 'm6g', 'm6gd', 'm7g', 'm7gd', 'r6g', 'r6gd', 'r7g', 'r7gd', 'g5g', 'im4gn', 'is4gen', // current Graviton-based instance classes
      'a13', 't11g', 'y10ng', 'z11ngd', // theoretical future Graviton-based instance classes
    ];

    for (const instanceClass of sampleInstanceClasses) {
      // WHEN
      const instanceType = InstanceType.of(instanceClass as InstanceClass, InstanceSize.XLARGE18);

      // THEN
      expect(instanceType.architecture).toBe(InstanceArchitecture.ARM_64);
    }
  });
  test('instance architecture is correctly discerned for x86-64 instance', () => {
    // GIVEN
    const sampleInstanceClasses = ['c5', 'm5ad', 'r5n', 'm6', 't3a', 'r6i', 'r6a', 'g6', 'p4de', 'p5', 'm7i-flex']; // A sample of x86-64 instance classes

    for (const instanceClass of sampleInstanceClasses) {
      // WHEN
      const instanceType = InstanceType.of(instanceClass as InstanceClass, InstanceSize.XLARGE18);

      // THEN
      expect(instanceType.architecture).toBe(InstanceArchitecture.X86_64);
    }
  });

  test('sameInstanceClassAs compares InstanceTypes contains dashes', () => {
    // GIVEN
    const comparitor = InstanceType.of(InstanceClass.M7I_FLEX, InstanceSize.LARGE);
    // WHEN
    const largerInstanceType = InstanceType.of(InstanceClass.M7I_FLEX, InstanceSize.XLARGE);
    // THEN
    expect(largerInstanceType.sameInstanceClassAs(comparitor)).toBeTruthy();
  });

  test('sameInstanceClassAs compares InstanceSize contains dashes', () => {
    // GIVEN
    const comparitor = new InstanceType('c7a.metal-48xl');
    // WHEN
    const largerInstanceType = new InstanceType('c7a.xlarge');
    // THEN
    expect(largerInstanceType.sameInstanceClassAs(comparitor)).toBeTruthy();
  });

  test('instances with local NVME drive are correctly named', () => {
    // GIVEN
    const sampleInstanceClassKeys = [{
      key: InstanceClass.R5D,
      value: 'r5d',
    }, {
      key: InstanceClass.MEMORY5_NVME_DRIVE,
      value: 'r5d',
    }, {
      key: InstanceClass.R5AD,
      value: 'r5ad',
    }, {
      key: InstanceClass.MEMORY5_AMD_NVME_DRIVE,
      value: 'r5ad',
    }, {
      key: InstanceClass.M5AD,
      value: 'm5ad',
    }, {
      key: InstanceClass.STANDARD5_AMD_NVME_DRIVE,
      value: 'm5ad',
    }]; // A sample of instances with NVME drives

    for (const instanceClass of sampleInstanceClassKeys) {
      // WHEN
      const instanceType = InstanceType.of(instanceClass.key, InstanceSize.LARGE);
      // THEN
      expect(instanceType.toString().split('.')[0]).toBe(instanceClass.value);
    }
  });
  test('instance architecture throws an error when instance type is invalid', () => {
    // GIVEN
    const malformedInstanceTypes = ['t4', 't4g.nano.', 't4gnano', ''];

    for (const malformedInstanceType of malformedInstanceTypes) {
      // WHEN
      const instanceType = new InstanceType(malformedInstanceType);

      // THEN
      expect(() => instanceType.architecture).toThrow('Malformed instance type identifier');
    }
  });
  test('can propagate EBS volume tags', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      propagateTagsToVolumeOnCreation: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      PropagateTagsToVolumeOnCreation: true,
    });
  });
  // placementGroup
  describe('placementGroup', () => {
    test('can set placementGroup', () => {
      // WHEN
      // create a new placementgroup
      const pg1 = new PlacementGroup(stack, 'myPlacementGroup1');
      new PlacementGroup(stack, 'myPlacementGroup2');
      new Instance(stack, 'Instance1', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        placementGroup: pg1,
      });
      new Instance(stack, 'Instance2', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        placementGroup: PlacementGroup.fromPlacementGroupName(stack, 'importedPlacementGroup', 'myPlacementGroup2'),
      });

      const t = Template.fromStack(stack);
      // THEN
      t.hasResourceProperties('AWS::EC2::Instance', {
        PlacementGroupName: {
          'Fn::GetAtt': ['myPlacementGroup180969E8B', 'GroupName'],
        },
      });
      t.hasResourceProperties('AWS::EC2::Instance', {
        PlacementGroupName: 'myPlacementGroup2',
      });
    });
  });
  describe('blockDeviceMappings', () => {
    test('can set blockDeviceMappings', () => {
      Validations.of(stack).acknowledge({
        id: 'CloudFormation-Validate::F3014',
        reason: 'test violates this',
      });

      // WHEN
      const kmsKey = new Key(stack, 'EbsKey');
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        blockDevices: [{
          deviceName: 'ebs',
          mappingEnabled: true,
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: EbsDeviceVolumeType.IO1,
            iops: 5000,
          }),
        }, {
          deviceName: 'ebs-gp3',
          mappingEnabled: true,
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: EbsDeviceVolumeType.GP3,
            iops: 5000,
          }),
        }, {
          deviceName: 'ebs-cmk',
          mappingEnabled: true,
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            kmsKey: kmsKey,
            volumeType: EbsDeviceVolumeType.IO1,
            iops: 5000,
          }),
        }, {
          deviceName: 'ebs-snapshot',
          mappingEnabled: false,
          volume: BlockDeviceVolume.ebsFromSnapshot('snapshot-id', {
            volumeSize: 500,
            deleteOnTermination: false,
            volumeType: EbsDeviceVolumeType.SC1,
          }),
        }, {
          deviceName: 'ephemeral',
          volume: BlockDeviceVolume.ephemeral(0),
        }],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
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
            DeviceName: 'ebs-gp3',
            Ebs: {
              DeleteOnTermination: true,
              Encrypted: true,
              Iops: 5000,
              VolumeSize: 15,
              VolumeType: 'gp3',
            },
          },
          {
            DeviceName: 'ebs-cmk',
            Ebs: {
              DeleteOnTermination: true,
              Encrypted: true,
              KmsKeyId: {
                'Fn::GetAtt': [
                  'EbsKeyD3FEE551',
                  'Arn',
                ],
              },
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
            NoDevice: {},
          },
          {
            DeviceName: 'ephemeral',
            VirtualName: 'ephemeral0',
          },
        ],
      });
    });

    test('warns if throughput is specified for an EBS volume', () => {
      // WHEN
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: EbsDeviceVolumeType.GP3,
            throughput: 125,
          }),
        }],
      });
      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Instance', Match.stringLikeRegexp('The throughput property is not supported on EC2 instances. Use a Launch Template instead'));
    });

    test('warns if volumeInitializationRate is specified for an EBS volume on an EC2 instance', () => {
      // WHEN
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: EbsDeviceVolumeType.GP3,
            volumeInitializationRate: Size.mebibytes(300),
          }),
        }],
      });
      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Instance', Match.stringLikeRegexp('The volumeInitializationRate is not supported on EC2 instances. Use a Launch Template instead.'));

      // Assert VolumeInitializationRate is absent from the synthesized template
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        BlockDeviceMappings: Match.arrayWith([
          Match.objectLike({
            Ebs: Match.not(Match.objectLike({
              VolumeInitializationRate: Match.anyValue(),
            })),
          }),
        ]),
      });
    });

    test('throws if ephemeral volumeIndex < 0', () => {
      // THEN
      expect(() => {
        new Instance(stack, 'Instance', {
          vpc,
          machineImage: new AmazonLinuxImage(),
          instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
          blockDevices: [{
            deviceName: 'ephemeral',
            volume: BlockDeviceVolume.ephemeral(-1),
          }],
        });
      }).toThrow(/volumeIndex must be a number starting from 0/);
    });

    test('throws if volumeType === IO1 without iops', () => {
      // THEN
      expect(() => {
        new Instance(stack, 'Instance', {
          vpc,
          machineImage: new AmazonLinuxImage(),
          instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
          blockDevices: [{
            deviceName: 'ebs',
            volume: BlockDeviceVolume.ebs(15, {
              deleteOnTermination: true,
              encrypted: true,
              volumeType: EbsDeviceVolumeType.IO1,
            }),
          }],
        });
      }).toThrow(/ops property is required with volumeType: EbsDeviceVolumeType.IO1/);
    });

    test('throws if volumeType === IO2 without iops', () => {
      // THEN
      expect(() => {
        new Instance(stack, 'Instance', {
          vpc,
          machineImage: new AmazonLinuxImage(),
          instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
          blockDevices: [{
            deviceName: 'ebs',
            volume: BlockDeviceVolume.ebs(15, {
              deleteOnTermination: true,
              encrypted: true,
              volumeType: EbsDeviceVolumeType.IO2,
            }),
          }],
        });
      }).toThrow(/ops property is required with volumeType: EbsDeviceVolumeType.IO1 and EbsDeviceVolumeType.IO2/);
    });

    test('warning if iops without volumeType', () => {
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            iops: 5000,
          }),
        }],
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Instance', 'iops will be ignored without volumeType: IO1, IO2, or GP3 [ack: @aws-cdk/aws-ec2:iopsIgnored]');
    });

    test('warning if iops and invalid volumeType', () => {
      Validations.of(stack).acknowledge({
        id: 'CloudFormation-Validate::W3671',
        reason: 'We already have a warning here',
      });
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            deleteOnTermination: true,
            encrypted: true,
            volumeType: EbsDeviceVolumeType.GP2,
            iops: 5000,
          }),
        }],
      });

      // THEN
      Annotations.fromStack(stack).hasWarning('/Default/Instance', 'iops will be ignored without volumeType: IO1, IO2, or GP3 [ack: @aws-cdk/aws-ec2:iopsIgnored]');
    });
  });

  describe('instanceProfile', () => {
    let instanceProfile: InstanceProfile;
    let role: Role;

    beforeEach(() => {
      role = new Role(stack, 'MyRole', {
        assumedBy: new ServicePrincipal('ec2.amazonaws.com'),
      });
      instanceProfile = new InstanceProfile(stack, 'MyInstanceProfile', {
        role,
      });
    });

    test('can specify instanceProfile', () => {
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        instanceProfile,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::InstanceProfile', {
        Roles: [{ Ref: 'MyRoleF48FFE04' }],
      });
    });

    test('throws if used with role', () => {
      expect(() => {
        new Instance(stack, 'Instance', {
          vpc,
          machineImage: new AmazonLinuxImage(),
          instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
          instanceProfile,
          role,
        });
      }).toThrow(/You cannot provide both instanceProfile and role/);
    });
  });

  test('instance can be created with Private IP Address', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      privateIpAddress: '10.0.0.2',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      InstanceType: 't3.large',
      PrivateIpAddress: '10.0.0.2',
    });
  });

  test('instance can be created with Private IP Address AND Associate Public IP Address', () => {
    const privateIpAddress = '10.0.0.2';
    // GIVEN
    const securityGroup = new SecurityGroup(stack, 'SecurityGroup', { vpc });

    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      vpcSubnets: { subnetType: SubnetType.PUBLIC },
      securityGroup,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      privateIpAddress: privateIpAddress,
      associatePublicIpAddress: true,
    });

    // THEN
    // PrivateIpAddress AND NetworkInterfaces cannot both be present
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      PrivateIpAddress: Match.absent(),
    });
    Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
      Properties: {
        NetworkInterfaces: [{
          PrivateIpAddress: privateIpAddress,
          AssociatePublicIpAddress: true,
          DeviceIndex: '0',
          GroupSet: [
            {
              'Fn::GetAtt': [
                'SecurityGroupDD263621',
                'GroupId',
              ],
            },
          ],
          SubnetId: {
            Ref: 'VPCPublicSubnet1SubnetB4246D30',
          },
        }],
      },
      DependsOn: [
        'InstanceInstanceRoleE9785DE5',
        'VPCPublicSubnet1DefaultRoute91CEF279',
        'VPCPublicSubnet1RouteTableAssociation0B0896DC',
        'VPCPublicSubnet2DefaultRouteB7481BBA',
        'VPCPublicSubnet2RouteTableAssociation5A808732',
      ],
    });
  });

  test('instance requires IMDSv2', () => {
    // WHEN
    const instance = new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: new InstanceType('t2.micro'),
      requireImdsv2: true,
    });

    // Force stack synth so the InstanceRequireImdsv2Aspect is applied
    Template.fromStack(stack);

    // THEN
    const launchTemplate = instance.node.tryFindChild('LaunchTemplate') as LaunchTemplate;
    expect(launchTemplate).toBeDefined();
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateName: stack.resolve(launchTemplate.launchTemplateName),
      LaunchTemplateData: {
        MetadataOptions: {
          HttpTokens: 'required',
        },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      LaunchTemplate: {
        LaunchTemplateName: stack.resolve(launchTemplate.launchTemplateName),
      },
    });
  });

  it('throws an error on incompatible Key Pair for operating system', () => {
    // GIVEN
    const keyPair = new KeyPair(stack, 'KeyPair', {
      type: KeyPairType.ED25519,
    });

    // THEN
    expect(() => new Instance(stack, 'Instance', {
      vpc,
      machineImage: new WindowsImage(WindowsVersion.WINDOWS_SERVER_2022_ENGLISH_CORE_BASE),
      instanceType: new InstanceType('t2.micro'),
      keyPair,
    })).toThrow('ed25519 keys are not compatible with the chosen AMI');
  });

  it('throws an error if keyName and keyPair both provided', () => {
    // GIVEN
    const keyPair = new KeyPair(stack, 'KeyPair');

    // THEN
    expect(() => new Instance(stack, 'Instance', {
      vpc,
      instanceType: new InstanceType('t2.micro'),
      machineImage: new AmazonLinuxImage(),
      keyName: 'test-key-pair',
      keyPair,
    })).toThrow('Cannot specify both of \'keyName\' and \'keyPair\'; prefer \'keyPair\'');
  });

  it('correctly associates a key pair', () => {
    // GIVEN
    const keyPair = new KeyPair(stack, 'KeyPair', {
      keyPairName: 'test-key-pair',
    });

    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      instanceType: new InstanceType('t2.micro'),
      machineImage: new AmazonLinuxImage(),
      keyPair,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      KeyName: stack.resolve(keyPair.keyPairName),
    });
  });

  describe('Detailed Monitoring', () => {
    test('instance with Detailed Monitoring enabled', () => {
      // WHEN
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: new InstanceType('t2.micro'),
        detailedMonitoring: true,
      });

      // Force stack synth so the Instance is applied
      Template.fromStack(stack);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        Monitoring: true,
      });
    });

    test('instance with Detailed Monitoring disabled', () => {
      // WHEN
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: new InstanceType('t2.micro'),
        detailedMonitoring: false,
      });

      // Force stack synth so the Instance is applied
      Template.fromStack(stack);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        Monitoring: false,
      });
    });

    test('instance with Detailed Monitoring unset falls back to disabled', () => {
      // WHEN
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: new InstanceType('t2.micro'),
      });

      // Force stack synth so the Instance is applied
      Template.fromStack(stack);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
        Monitoring: Match.absent(),
      });
    });
  });

  test('burstable instance with explicit credit specification', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      creditSpecification: CpuCredits.STANDARD,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      InstanceType: 't3.large',
      CreditSpecification: {
        CPUCredits: 'standard',
      },
    });
  });

  test('throw if creditSpecification is defined for a non-burstable instance type', () => {
    // THEN
    expect(() => {
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.M5, InstanceSize.LARGE),
        creditSpecification: CpuCredits.STANDARD,
      });
    }).toThrow('creditSpecification is supported only for T4g, T3a, T3, T2 instance type, got: m5.large');
  });

  test('set instanceInitiatedShutdownBehavior', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: new InstanceType('t2.micro'),
      instanceInitiatedShutdownBehavior: InstanceInitiatedShutdownBehavior.TERMINATE,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      InstanceType: 't2.micro',
      InstanceInitiatedShutdownBehavior: 'terminate',
    });
  });
});

test('add CloudFormation Init to instance', () => {
  // GIVEN
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    init: CloudFormationInit.fromElements(
      InitCommand.shellCommand('echo hello'),
    ),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    UserData: {
      'Fn::Base64': {
        'Fn::Join': ['', [
          '#!/bin/bash\n# fingerprint: 85ac432b1de1144f\n(\n  set +e\n  /opt/aws/bin/cfn-init -v --region ',
          { Ref: 'AWS::Region' },
          ' --stack ',
          { Ref: 'AWS::StackName' },
          ' --resource InstanceC1063A87 -c default\n  /opt/aws/bin/cfn-signal -e $? --region ',
          { Ref: 'AWS::Region' },
          ' --stack ',
          { Ref: 'AWS::StackName' },
          ' --resource InstanceC1063A87\n  cat /var/log/cfn-init.log >&2\n)',
        ]],
      },
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['cloudformation:DescribeStackResource', 'cloudformation:SignalResource'],
        Effect: 'Allow',
        Resource: { Ref: 'AWS::StackId' },
      }]),
      Version: '2012-10-17',
    },
  });
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    CreationPolicy: {
      ResourceSignal: {
        Count: 1,
        Timeout: 'PT5M',
      },
    },
  });
});

test('cause replacement from s3 asset in userdata', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false,
    },
  });
  stack = new Stack(app);
  vpc = new Vpc(stack, 'Vpc)');
  const userData1 = UserData.forLinux();
  const asset1 = new Asset(stack, 'UserDataAssets1', {
    path: path.join(__dirname, 'asset-fixture', 'data.txt'),
  });
  userData1.addS3DownloadCommand({ bucket: asset1.bucket, bucketKey: asset1.s3ObjectKey });

  const userData2 = UserData.forLinux();
  const asset2 = new Asset(stack, 'UserDataAssets2', {
    path: path.join(__dirname, 'asset-fixture', 'data.txt'),
  });
  userData2.addS3DownloadCommand({ bucket: asset2.bucket, bucketKey: asset2.s3ObjectKey });

  // WHEN
  new Instance(stack, 'InstanceOne', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    userData: userData1,
    userDataCausesReplacement: true,
  });
  new Instance(stack, 'InstanceTwo', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    userData: userData2,
    userDataCausesReplacement: true,
  });

  // THEN -- both instances have the same userData hash, telling us the hash is based
  // on the actual asset hash and not accidentally on the token stringification of them.
  // (which would base the hash on '${Token[1234.bla]}'
  const hash = 'f88eace39faf39d7';
  Template.fromStack(stack).templateMatches(Match.objectLike({
    Resources: Match.objectLike({
      [`InstanceOne5B821005${hash}`]: Match.objectLike({
        Type: 'AWS::EC2::Instance',
        Properties: Match.anyValue(),
      }),
      [`InstanceTwoDC29A7A7${hash}`]: Match.objectLike({
        Type: 'AWS::EC2::Instance',
        Properties: Match.anyValue(),
      }),
    }),
  }));
});

test('ssm permissions adds right managed policy', () => {
  // WHEN
  new Instance(stack, 'InstanceOne', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
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

test('sameInstanceClassAs compares identical InstanceTypes correctly', () => {
  // GIVEN
  const comparitor = InstanceType.of(InstanceClass.T3, InstanceSize.LARGE);
  // WHEN
  const sameInstanceType = InstanceType.of(InstanceClass.T3, InstanceSize.LARGE);
  // THEN
  expect(sameInstanceType.sameInstanceClassAs(comparitor)).toBeTruthy();
});

test('sameInstanceClassAs compares InstanceTypes correctly regardless of size', () => {
  // GIVEN
  const comparitor = InstanceType.of(InstanceClass.T3, InstanceSize.LARGE);
  // WHEN
  const largerInstanceType = InstanceType.of(InstanceClass.T3, InstanceSize.XLARGE);
  // THEN
  expect(largerInstanceType.sameInstanceClassAs(comparitor)).toBeTruthy();
});

test('sameInstanceClassAs compares different InstanceTypes correctly', () => {
  // GIVEN
  const comparitor = InstanceType.of(InstanceClass.C4, InstanceSize.LARGE);
  // WHEN
  const instanceType = new InstanceType('t3.large');
  // THEN
  expect(instanceType.sameInstanceClassAs(comparitor)).toBeFalsy();
});

test('associate public IP address with instance', () => {
  // GIVEN
  const securityGroup = new SecurityGroup(stack, 'SecurityGroup', { vpc });

  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    vpcSubnets: { subnetType: SubnetType.PUBLIC },
    securityGroup,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    sourceDestCheck: false,
    associatePublicIpAddress: true,
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    Properties: {
      NetworkInterfaces: [{
        AssociatePublicIpAddress: true,
        DeviceIndex: '0',
        GroupSet: [
          {
            'Fn::GetAtt': [
              'SecurityGroupDD263621',
              'GroupId',
            ],
          },
        ],
        SubnetId: {
          Ref: 'VPCPublicSubnet1SubnetB4246D30',
        },
      }],
    },
    DependsOn: [
      'InstanceInstanceRoleE9785DE5',
      'VPCPublicSubnet1DefaultRoute91CEF279',
      'VPCPublicSubnet1RouteTableAssociation0B0896DC',
      'VPCPublicSubnet2DefaultRouteB7481BBA',
      'VPCPublicSubnet2RouteTableAssociation5A808732',
    ],
  });
});

test('do not associate public IP address with instance', () => {
  // GIVEN
  const securityGroup = new SecurityGroup(stack, 'SecurityGroup', { vpc });

  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    vpcSubnets: { subnetType: SubnetType.PUBLIC },
    securityGroup,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    sourceDestCheck: false,
    associatePublicIpAddress: false,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    NetworkInterfaces: [{
      AssociatePublicIpAddress: false,
      DeviceIndex: '0',
      GroupSet: [
        {
          'Fn::GetAtt': [
            'SecurityGroupDD263621',
            'GroupId',
          ],
        },
      ],
      SubnetId: {
        Ref: 'VPCPublicSubnet1SubnetB4246D30',
      },
    }],
  });
});

test('associate public IP address with instance and no public subnet', () => {
  // WHEN/THEN
  expect(() => {
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      sourceDestCheck: false,
      associatePublicIpAddress: true,
    });
  }).toThrow("To set 'associatePublicIpAddress: true' you must select Public subnets (vpcSubnets: { subnetType: SubnetType.PUBLIC })");
});

test('specify ebs optimized instance', () => {
  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: new InstanceType('t3.large'),
    ebsOptimized: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    InstanceType: 't3.large',
    EbsOptimized: true,
  });
});

test('specify disable api termination', () => {
  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: new InstanceType('t3.large'),
    disableApiTermination: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    InstanceType: 't3.large',
    DisableApiTermination: true,
  });
});

test.each([
  [true, true],
  [false, false],
])('given enclaveEnabled %p', (given: boolean, expected: boolean) => {
  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.M5, InstanceSize.XLARGE),
    enclaveEnabled: given,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    EnclaveOptions: {
      Enabled: expected,
    },
  });
});

test.each([
  [true, true],
  [false, false],
])('given hibernationEnabled %p', (given: boolean, expected: boolean) => {
  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.M5, InstanceSize.XLARGE),
    hibernationEnabled: given,
    blockDevices: [{
      deviceName: '/dev/xvda',
      volume: BlockDeviceVolume.ebs(30, {
        volumeType: EbsDeviceVolumeType.GP3,
        encrypted: true,
        deleteOnTermination: true,
      }),
    }],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    HibernationOptions: {
      Configured: expected,
    },
  });
});

test('throw if both enclaveEnabled and hibernationEnabled are set to true', () => {
  // WHEN/THEN
  expect(() => {
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.M5, InstanceSize.LARGE),
      enclaveEnabled: true,
      hibernationEnabled: true,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: BlockDeviceVolume.ebs(30, {
          volumeType: EbsDeviceVolumeType.GP3,
          encrypted: true,
          deleteOnTermination: true,
        }),
      }],
    });
  }).toThrow('You can\'t set both `enclaveEnabled` and `hibernationEnabled` to true on the same instance');
});

test('instance with ipv6 address count', () => {
  // WHEN
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: new InstanceType('t2.micro'),
    ipv6AddressCount: 2,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
    InstanceType: 't2.micro',
    Ipv6AddressCount: 2,
  });
});

test.each([-1, 0.1, 1.1])('throws if ipv6AddressCount is not a positive integer', (ipv6AddressCount: number) => {
  // THEN
  expect(() => {
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: new InstanceType('t2.micro'),
      ipv6AddressCount: ipv6AddressCount,
    });
  }).toThrow(`\'ipv6AddressCount\' must be a non-negative integer, got: ${ipv6AddressCount}`);
});

test.each([true, false])('throw error for specifying ipv6AddressCount with associatePublicIpAddress', (associatePublicIpAddress) => {
  // THEN
  expect(() => {
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: new InstanceType('t2.micro'),
      ipv6AddressCount: 2,
      associatePublicIpAddress,
    });
  }).toThrow('You can\'t set both \'ipv6AddressCount\' and \'associatePublicIpAddress\'');
});

test('resourceSignalTimeout overwrites initOptions.timeout when feature flag turned off', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.EC2_SUM_TIMEOUT_ENABLED]: false,
    },
  });
  stack = new Stack(app);
  vpc = new Vpc(stack, 'Vpc)');
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    init: CloudFormationInit.fromElements(
      InitCommand.shellCommand('echo hello'),
    ),
    initOptions: {
      timeout: Duration.minutes(30),
    },
    resourceSignalTimeout: Duration.minutes(10),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    CreationPolicy: {
      ResourceSignal: {
        Count: 1,
        Timeout: 'PT10M',
      },
    },
  });
});

test('initOptions.timeout and resourceSignalTimeout are both not set. Timeout is set to default of 5 min', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.EC2_SUM_TIMEOUT_ENABLED]: true,
    },
  });
  stack = new Stack(app);
  vpc = new Vpc(stack, 'Vpc)');
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    init: CloudFormationInit.fromElements(
      InitCommand.shellCommand('echo hello'),
    ),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    CreationPolicy: {
      ResourceSignal: {
        Timeout: 'PT5M',
      },
    },
  });
});

test('initOptions.timeout is set and not resourceSignalTimeout. Timeout is set to initOptions.timeout value', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.EC2_SUM_TIMEOUT_ENABLED]: true,
    },
  });
  stack = new Stack(app);
  vpc = new Vpc(stack, 'Vpc)');
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    init: CloudFormationInit.fromElements(
      InitCommand.shellCommand('echo hello'),
    ),
    initOptions: {
      timeout: Duration.minutes(10),
    },
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    CreationPolicy: {
      ResourceSignal: {
        Count: 1,
        Timeout: 'PT10M',
      },
    },
  });
});

test('resourceSignalTimeout is set and not initOptions.timeout. Timeout is set to resourceSignalTimeout value', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.EC2_SUM_TIMEOUT_ENABLED]: true,
    },
  });
  stack = new Stack(app);
  vpc = new Vpc(stack, 'Vpc)');
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    init: CloudFormationInit.fromElements(
      InitCommand.shellCommand('echo hello'),
    ),
    resourceSignalTimeout: Duration.minutes(10),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    CreationPolicy: {
      ResourceSignal: {
        Timeout: 'PT15M',
      },
    },
  });
});

test('resourceSignalTimeout and initOptions.timeout are both set, sum timeout and log warning', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.EC2_SUM_TIMEOUT_ENABLED]: true,
    },
  });
  stack = new Stack(app);
  vpc = new Vpc(stack, 'Vpc)');
  new Instance(stack, 'Instance', {
    vpc,
    machineImage: new AmazonLinuxImage(),
    instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    init: CloudFormationInit.fromElements(
      InitCommand.shellCommand('echo hello'),
    ),
    initOptions: {
      timeout: Duration.minutes(10),
    },
    resourceSignalTimeout: Duration.minutes(10),
  });

  // THEN
  Template.fromStack(stack).hasResource('AWS::EC2::Instance', {
    CreationPolicy: {
      ResourceSignal: {
        Count: 1,
        Timeout: 'PT20M',
      },
    },
  });
});

describe('metadataOptions', () => {
  test('can set all metadata options', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpEndpoint: true,
      httpProtocolIpv6: false,
      httpPutResponseHopLimit: 2,
      httpTokens: HttpTokens.REQUIRED,
      instanceMetadataTags: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpEndpoint: 'enabled',
        HttpProtocolIpv6: 'disabled',
        HttpPutResponseHopLimit: 2,
        HttpTokens: 'required',
        InstanceMetadataTags: 'enabled',
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('httpEndpoint %s maps to %s', (input, expected) => {
    // WHEN
    new Instance(stack, `Instance${input}`, {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpEndpoint: input,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpEndpoint: expected,
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('httpProtocolIpv6 %s maps to %s', (input, expected) => {
    // WHEN
    new Instance(stack, `Instance${input}`, {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpProtocolIpv6: input,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpProtocolIpv6: expected,
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('instanceMetadataTags %s maps to %s', (input, expected) => {
    // WHEN
    new Instance(stack, `Instance${input}`, {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      instanceMetadataTags: input,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        InstanceMetadataTags: expected,
      },
    });
  });

  test.each([
    [HttpTokens.OPTIONAL, 'optional'],
    [HttpTokens.REQUIRED, 'required'],
  ])('httpTokens %s maps to %s', (input, expected) => {
    // WHEN
    new Instance(stack, `Instance${expected}`, {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpTokens: input,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpTokens: expected,
      },
    });
  });

  test.each([1, 32, 64])('httpPutResponseHopLimit %d is valid', (hopLimit) => {
    // WHEN
    new Instance(stack, `Instance${hopLimit}`, {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpPutResponseHopLimit: hopLimit,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpPutResponseHopLimit: hopLimit,
      },
    });
  });

  test.each([0, 65, -1, 100])('httpPutResponseHopLimit %d throws validation error', (hopLimit) => {
    // WHEN/THEN
    expect(() => {
      new Instance(stack, `Instance${hopLimit}`, {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        httpPutResponseHopLimit: hopLimit,
      });
    }).toThrow('httpPutResponseHopLimit must be between 1 and 64');
  });

  test('no metadata options renders undefined', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      InstanceType: 't3.large',
    });

    // Ensure MetadataOptions is not present
    const template = Template.fromStack(stack);
    const instances = template.findResources('AWS::EC2::Instance');
    const instanceProps = Object.values(instances)[0].Properties;
    expect(instanceProps.MetadataOptions).toBeUndefined();
  });

  test('partial metadata options only render specified properties', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpTokens: HttpTokens.REQUIRED,
      httpPutResponseHopLimit: 10,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpTokens: 'required',
        HttpPutResponseHopLimit: 10,
      },
    });

    // Ensure other properties are not present
    const template = Template.fromStack(stack);
    const instances = template.findResources('AWS::EC2::Instance');
    const metadataOptions = Object.values(instances)[0].Properties.MetadataOptions;
    expect(metadataOptions.HttpEndpoint).toBeUndefined();
    expect(metadataOptions.HttpProtocolIpv6).toBeUndefined();
    expect(metadataOptions.InstanceMetadataTags).toBeUndefined();
  });

  test('metadata options conflict with requireImdsv2', () => {
    // WHEN/THEN - Should throw validation error when both are used together
    expect(() => {
      new Instance(stack, 'Instance', {
        vpc,
        machineImage: new AmazonLinuxImage(),
        instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
        requireImdsv2: true,
        httpEndpoint: true,
        instanceMetadataTags: true,
      });
    }).toThrow('Cannot use both requireImdsv2 and metadata options');
  });

  test('requireImdsv2 works without metadata options', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      requireImdsv2: true,
    });

    // Force stack synth so the InstanceRequireImdsv2Aspect is applied
    Template.fromStack(stack);

    // THEN - Should create LaunchTemplate with IMDSv2 required
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        MetadataOptions: {
          HttpTokens: 'required',
        },
      },
    });

    // Instance should not have direct MetadataOptions
    const template = Template.fromStack(stack);
    const instances = template.findResources('AWS::EC2::Instance');
    const instanceProps = Object.values(instances)[0].Properties;
    expect(instanceProps.MetadataOptions).toBeUndefined();
  });

  test('metadata options with only undefined values does not render MetadataOptions', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpEndpoint: undefined,
      httpProtocolIpv6: undefined,
      httpPutResponseHopLimit: undefined,
      httpTokens: undefined,
      instanceMetadataTags: undefined,
    });

    // THEN - Should not render MetadataOptions property (matches LaunchTemplate behavior)
    const template = Template.fromStack(stack);
    const instances = template.findResources('AWS::EC2::Instance');
    const instanceProps = Object.values(instances)[0].Properties;

    // MetadataOptions should not be present when all properties are undefined
    expect(instanceProps.MetadataOptions).toBeUndefined();
  });

  test('metadata options with partial configuration leaves unspecified properties undefined', () => {
    // WHEN
    new Instance(stack, 'Instance', {
      vpc,
      machineImage: new AmazonLinuxImage(),
      instanceType: InstanceType.of(InstanceClass.T3, InstanceSize.LARGE),
      httpEndpoint: false,
      httpTokens: HttpTokens.REQUIRED,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::Instance', {
      MetadataOptions: {
        HttpEndpoint: 'disabled',
        HttpTokens: 'required',
        // Other properties should be undefined (CloudFormation will use its defaults)
      },
    });

    // Verify unspecified properties are undefined (CloudFormation handles defaults)
    const template = Template.fromStack(stack);
    const instances = template.findResources('AWS::EC2::Instance');
    const metadataOptions = Object.values(instances)[0].Properties.MetadataOptions;
    expect(metadataOptions.HttpProtocolIpv6).toBeUndefined();
    expect(metadataOptions.HttpPutResponseHopLimit).toBeUndefined();
    expect(metadataOptions.InstanceMetadataTags).toBeUndefined();
  });
});
