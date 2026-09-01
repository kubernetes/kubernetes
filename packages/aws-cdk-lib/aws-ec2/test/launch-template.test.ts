import { stringLike } from './util';
import { Annotations, Template, Match } from '../../assertions';
import type { CfnInstanceProfile } from '../../aws-iam';
import {
  Role,
  ServicePrincipal,
  InstanceProfile,
} from '../../aws-iam';
import { Key } from '../../aws-kms';
import {
  App,
  Duration,
  Expiration,
  Size,
  Stack,
  Tags,
  Validations,
} from '../../core';
import * as cxapi from '../../cx-api';
import type { BlockDevice } from '../lib';
import {
  AmazonLinuxImage,
  BlockDeviceVolume,
  CpuCredits,
  EbsDeviceVolumeType,
  InstanceInitiatedShutdownBehavior,
  InstanceType,
  KeyPair,
  KeyPairType,
  LaunchTemplate,
  LaunchTemplateHttpTokens,
  OperatingSystemType,
  PlacementGroup,
  PlacementGroupStrategy,
  SecurityGroup,
  SpotInstanceInterruption,
  SpotRequestType,
  UserData,
  Vpc,
  WindowsImage,
  WindowsVersion,
} from '../lib';

describe('LaunchTemplate', () => {
  let app: App;
  let stack: Stack;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app);
  });

  test('Empty props', () => {
    // WHEN
    const template = new LaunchTemplate(stack, 'Template');

    // THEN
    // Note: The following is intentionally a haveResource instead of haveResourceLike
    // to ensure that only the bare minimum of properties have values when no properties
    // are given to a LaunchTemplate.
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        TagSpecifications: [
          {
            ResourceType: 'instance',
            Tags: [
              {
                Key: 'Name',
                Value: 'Default/Template',
              },
            ],
          },
          {
            ResourceType: 'volume',
            Tags: [
              {
                Key: 'Name',
                Value: 'Default/Template',
              },
            ],
          },
        ],
      },
      TagSpecifications: [
        {
          ResourceType: 'launch-template',
          Tags: [
            {
              Key: 'Name',
              Value: 'Default/Template',
            },
          ],
        },
      ],
    });
    Template.fromStack(stack).resourceCountIs('AWS::IAM::InstanceProfile', 0);
    expect(() => { template.grantPrincipal; }).toThrow();
    expect(() => { template.connections; }).toThrow();
    expect(template.osType).toBeUndefined();
    expect(template.role).toBeUndefined();
    expect(template.userData).toBeUndefined();
  });

  test('Import from attributes with name', () => {
    // WHEN
    const template = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'Template', {
      launchTemplateName: 'TestName',
      versionNumber: 'TestVersion',
    });

    // THEN
    expect(template.launchTemplateId).toBeUndefined();
    expect(template.launchTemplateName).toBe('TestName');
    expect(template.versionNumber).toBe('TestVersion');
  });

  test('Import from attributes with id', () => {
    // WHEN
    const template = LaunchTemplate.fromLaunchTemplateAttributes(stack, 'Template', {
      launchTemplateId: 'TestId',
      versionNumber: 'TestVersion',
    });

    // THEN
    expect(template.launchTemplateId).toBe('TestId');
    expect(template.launchTemplateName).toBeUndefined();
    expect(template.versionNumber).toBe('TestVersion');
  });

  test('Import from attributes fails with name and id', () => {
    expect(() => {
      LaunchTemplate.fromLaunchTemplateAttributes(stack, 'Template', {
        launchTemplateName: 'TestName',
        launchTemplateId: 'TestId',
        versionNumber: 'TestVersion',
      });
    }).toThrow();
  });

  test('Given name', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      launchTemplateName: 'LTName',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateName: 'LTName',
    });
  });

  test('Given versionDescription', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      versionDescription: 'test template',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      VersionDescription: 'test template',
    });
  });

  test('throw error when versionDescription is too long', () => {
    const tooLongDescription = 'a'.repeat(256);
    // WHEN / THEN
    expect(() => {
      new LaunchTemplate(stack, 'TemplateWithTooLongDescription', {
        versionDescription: tooLongDescription,
      });
    }).toThrow('versionDescription must be less than or equal to 255 characters, got 256');
  });

  test('Given instanceType', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      instanceType: new InstanceType('tt.test'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceType: 'tt.test',
      },
    });
  });

  test('Given machineImage (Linux)', () => {
    // WHEN
    const template = new LaunchTemplate(stack, 'Template', {
      machineImage: new AmazonLinuxImage(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        ImageId: {
          Ref: stringLike('SsmParameterValueawsserviceamiamazonlinuxlatestamznami.*Parameter'),
        },
      },
    });
    expect(template.osType).toBe(OperatingSystemType.LINUX);
    expect(template.userData).toBeUndefined();
  });

  test('Given machineImage (Windows)', () => {
    // WHEN
    const template = new LaunchTemplate(stack, 'Template', {
      machineImage: new WindowsImage(WindowsVersion.WINDOWS_SERVER_2019_ENGLISH_FULL_BASE),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        ImageId: {
          Ref: stringLike('SsmParameterValueawsserviceamiwindowslatestWindowsServer2019EnglishFullBase.*Parameter'),
        },
      },
    });
    expect(template.osType).toBe(OperatingSystemType.WINDOWS);
    expect(template.userData).toBeUndefined();
  });

  test('Given userData', () => {
    // GIVEN
    const userData = UserData.forLinux();
    userData.addCommands('echo Test');

    // WHEN
    const template = new LaunchTemplate(stack, 'Template', {
      userData,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        UserData: {
          'Fn::Base64': '#!/bin/bash\necho Test',
        },
      },
    });
    expect(template.userData).toBeDefined();
  });

  test('Given role', () => {
    // GIVEN
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('ec2.amazonaws.com'),
    });

    // WHEN
    const template = new LaunchTemplate(stack, 'Template', {
      role,
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::InstanceProfile', {
      Roles: [
        {
          Ref: 'TestRole6C9272DF',
        },
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        IamInstanceProfile: {
          Arn: stack.resolve((template.node.findChild('Profile') as CfnInstanceProfile).getAtt('Arn')),
        },
        TagSpecifications: [
          {
            ResourceType: 'instance',
            Tags: [
              {
                Key: 'Name',
                Value: 'Default/Template',
              },
            ],
          },
          {
            ResourceType: 'volume',
            Tags: [
              {
                Key: 'Name',
                Value: 'Default/Template',
              },
            ],
          },
        ],
      },
      TagSpecifications: [
        {
          ResourceType: 'launch-template',
          Tags: [
            {
              Key: 'Name',
              Value: 'Default/Template',
            },
          ],
        },
      ],
    });
    expect(template.role).toBeDefined();
    expect(template.grantPrincipal).toBeDefined();
  });

  test('Given blockDeviceMapping', () => {
    // GIVEN
    Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::F3014',
      reason: 'test violates this',
    });
    const kmsKey = new Key(stack, 'EbsKey');
    const blockDevices: BlockDevice[] = [
      {
        deviceName: 'ebs',
        mappingEnabled: true,
        volume: BlockDeviceVolume.ebs(15, {
          deleteOnTermination: true,
          encrypted: true,
          volumeType: EbsDeviceVolumeType.IO1,
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
      }, {
        deviceName: 'gp3-with-throughput',
        volume: BlockDeviceVolume.ebs(15, {
          volumeType: EbsDeviceVolumeType.GP3,
          throughput: 350,
        }),
      }, {
        deviceName: 'volumeInitializationRate',
        volume: BlockDeviceVolume.ebsFromSnapshot('snapshot-id', {
          volumeInitializationRate: Size.mebibytes(300),
          volumeSize: 15,
        }),
      },
    ];

    // WHEN
    new LaunchTemplate(stack, 'Template', {
      blockDevices,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
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
            NoDevice: '',
          },
          {
            DeviceName: 'ephemeral',
            VirtualName: 'ephemeral0',
          },
          {
            DeviceName: 'gp3-with-throughput',
            Ebs: {
              VolumeSize: 15,
              VolumeType: 'gp3',
              Throughput: 350,
            },
          },
          {
            DeviceName: 'volumeInitializationRate',
            Ebs: {
              VolumeSize: 15,
              VolumeInitializationRate: 300,
              SnapshotId: 'snapshot-id',
            },
          },
        ],
      },
    });
  });
  test.each([124, 2001])('throws if throughput is set less than 125 or more than 2000', (throughput) => {
    expect(() => {
      new LaunchTemplate(stack, 'LaunchTemplate', {
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            volumeType: EbsDeviceVolumeType.GP3,
            throughput,
          }),
        }],
      });
    }).toThrow(/'throughput' must be between 125 and 2000, got/);
  });
  test('throws if throughput is not an integer', () => {
    expect(() => {
      new LaunchTemplate(stack, 'LaunchTemplate', {
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            volumeType: EbsDeviceVolumeType.GP3,
            throughput: 234.56,
          }),
        }],
      });
    }).toThrow("'throughput' must be an integer, got: 234.56.");
  });
  test.each([
    ...Object.values(EbsDeviceVolumeType).filter((v) => v !== 'gp3'),
  ])('throws if throughput is set on any volume type other than GP3', (volumeType) => {
    expect(() => {
      new LaunchTemplate(stack, 'LaunchTemplate', {
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            volumeType: volumeType,
            throughput: 150,
          }),
        }],
      });
    }).toThrow(/'throughput' requires 'volumeType': gp3, got/);
  });
  test('throws if throughput / iops ratio is greater than 0.25', () => {
    expect(() => {
      new LaunchTemplate(stack, 'LaunchTemplate', {
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            volumeType: EbsDeviceVolumeType.GP3,
            throughput: 751,
            iops: 3000,
          }),
        }],
      });
    }).toThrow('Throughput (MiBps) to iops ratio of 0.25033333333333335 is too high; maximum is 0.25 MiBps per iops');
  });

  test.each([Size.mebibytes(99), Size.mebibytes(301)])('throws if volumeInitializationRate is set less than 100 or more than 300', (volumeInitializationRate) => {
    expect(() => {
      new LaunchTemplate(stack, 'LaunchTemplate', {
        blockDevices: [{
          deviceName: 'ebs',
          volume: BlockDeviceVolume.ebs(15, {
            volumeType: EbsDeviceVolumeType.GP3,
            volumeInitializationRate,
          }),
        }],
      });
    }).toThrow(`volumeInitializationRate must be between 100 and 300 MiB/s, got: ${volumeInitializationRate.toMebibytes()} MiB/s`);
  });

  test('Given instance profile', () => {
    // GIVEN
    const role = new Role(stack, 'TestRole', {
      assumedBy: new ServicePrincipal('ec2.amazonaws.com'),
    });
    const instanceProfile = new InstanceProfile(stack, 'InstanceProfile', {
      role,
    });

    // WHEN
    const template = new LaunchTemplate(stack, 'Template', {
      instanceProfile,
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::InstanceProfile', {
      Roles: [
        {
          Ref: 'TestRole6C9272DF',
        },
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        IamInstanceProfile: {
          Arn: {
            'Fn::GetAtt': [
              'InstanceProfile9F2F41CB',
              'Arn',
            ],
          },
        },
      },
    });
    expect(template.role).toBeDefined();
    expect(template.grantPrincipal).toBeDefined();
  });

  describe('feature flag @aws-cdk/aws-ec2:launchTemplateDefaultUserData', () => {
    test('Given machineImage (Linux)', () => {
      // WHEN
      stack.node.setContext(cxapi.EC2_LAUNCH_TEMPLATE_DEFAULT_USER_DATA, true);
      const template = new LaunchTemplate(stack, 'Template', {
        machineImage: new AmazonLinuxImage(),
      });
      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
        LaunchTemplateData: {
          ImageId: {
            Ref: stringLike('SsmParameterValueawsserviceamiamazonlinuxlatestamznami.*Parameter'),
          },
        },
      });
      expect(template.osType).toBe(OperatingSystemType.LINUX);
      expect(template.userData).toBeDefined();
    });

    test('Given machineImage (Windows)', () => {
      // WHEN
      stack.node.setContext(cxapi.EC2_LAUNCH_TEMPLATE_DEFAULT_USER_DATA, true);
      const template = new LaunchTemplate(stack, 'Template', {
        machineImage: new WindowsImage(WindowsVersion.WINDOWS_SERVER_2019_ENGLISH_FULL_BASE),
      });
      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
        LaunchTemplateData: {
          ImageId: {
            Ref: stringLike('SsmParameterValueawsserviceamiwindowslatestWindowsServer2019EnglishFullBase.*Parameter'),
          },
        },
      });
      expect(template.osType).toBe(OperatingSystemType.WINDOWS);
      expect(template.userData).toBeDefined();
    });
  });

  describe('feature flag @aws-cdk/aws-autoscaling:generateLaunchTemplateInsteadOfLaunchConfig', () => {
    test('Given machineImage (Linux)', () => {
      // WHEN
      stack.node.setContext(cxapi.AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
      const template = new LaunchTemplate(stack, 'Template', {
        machineImage: new AmazonLinuxImage(),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
        LaunchTemplateData: {
          ImageId: {
            Ref: stringLike('SsmParameterValueawsserviceamiamazonlinuxlatestamznami.*Parameter'),
          },
        },
      });
      expect(template.osType).toBe(OperatingSystemType.LINUX);
      expect(template.userData).toBeDefined();
    });

    test('Given machineImage (Windows)', () => {
      // WHEN
      stack.node.setContext(cxapi.AUTOSCALING_GENERATE_LAUNCH_TEMPLATE, true);
      const template = new LaunchTemplate(stack, 'Template', {
        machineImage: new WindowsImage(WindowsVersion.WINDOWS_SERVER_2019_ENGLISH_FULL_BASE),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
        LaunchTemplateData: {
          ImageId: {
            Ref: stringLike('SsmParameterValueawsserviceamiwindowslatestWindowsServer2019EnglishFullBase.*Parameter'),
          },
        },
      });
      expect(template.osType).toBe(OperatingSystemType.WINDOWS);
      expect(template.userData).toBeDefined();
    });
  });

  test.each([
    [CpuCredits.STANDARD, 'standard'],
    [CpuCredits.UNLIMITED, 'unlimited'],
  ])('Given cpuCredits %p', (given: CpuCredits, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      cpuCredits: given,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        CreditSpecification: {
          CpuCredits: expected,
        },
      },
    });
  });

  test('Given cpuOptions', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      cpuOptions: {
        amdSevSnp: true,
        coreCount: 4,
        nestedVirtualization: true,
        threadsPerCore: 1,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        CpuOptions: {
          AmdSevSnp: 'enabled',
          CoreCount: 4,
          NestedVirtualization: 'enabled',
          ThreadsPerCore: 1,
        },
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('Given cpuOptions.nestedVirtualization %p', (given: boolean, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      cpuOptions: {
        nestedVirtualization: given,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        CpuOptions: {
          NestedVirtualization: expected,
        },
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('Given cpuOptions.amdSevSnp %p', (given: boolean, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      cpuOptions: {
        amdSevSnp: given,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        CpuOptions: {
          AmdSevSnp: expected,
        },
      },
    });
  });

  test('Given partial cpuOptions only renders specified fields', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      cpuOptions: {
        coreCount: 2,
        threadsPerCore: 2,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        CpuOptions: {
          CoreCount: 2,
          ThreadsPerCore: 2,
          AmdSevSnp: Match.absent(),
          NestedVirtualization: Match.absent(),
        },
      },
    });
  });

  test('Given empty cpuOptions does not render CpuOptions', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      cpuOptions: {},
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        CpuOptions: Match.absent(),
      },
    });
  });

  test.each([
    [true, true],
    [false, false],
  ])('Given disableApiTermination %p', (given: boolean, expected: boolean) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      disableApiTermination: given,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        DisableApiTermination: expected,
      },
    });
  });

  test.each([
    [true, true],
    [false, false],
  ])('Given ebsOptimized %p', (given: boolean, expected: boolean) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      ebsOptimized: given,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        EbsOptimized: expected,
      },
    });
  });

  test.each([
    [true, true],
    [false, false],
  ])('Given nitroEnclaveEnabled %p', (given: boolean, expected: boolean) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      nitroEnclaveEnabled: given,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        EnclaveOptions: {
          Enabled: expected,
        },
      },
    });
  });

  test.each([
    [InstanceInitiatedShutdownBehavior.STOP, 'stop'],
    [InstanceInitiatedShutdownBehavior.TERMINATE, 'terminate'],
  ])('Given instanceInitiatedShutdownBehavior %p', (given: InstanceInitiatedShutdownBehavior, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      instanceInitiatedShutdownBehavior: given,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceInitiatedShutdownBehavior: expected,
      },
    });
  });

  test('Given keyName', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      keyName: 'TestKeyname',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        KeyName: 'TestKeyname',
      },
    });
  });

  it('throws an error on incompatible Key Pair for operating system', () => {
    // GIVEN
    const keyPair = new KeyPair(stack, 'KeyPair', {
      type: KeyPairType.ED25519,
    });

    // THEN
    expect(() => new LaunchTemplate(stack, 'Instance', {
      machineImage: new WindowsImage(WindowsVersion.WINDOWS_SERVER_2022_ENGLISH_CORE_BASE),
      keyPair,
    })).toThrow('ed25519 keys are not compatible with the chosen AMI');
  });

  it('throws when keyName and keyPair are provided', () => {
    // GIVEN
    const keyPair = new KeyPair(stack, 'KeyPair');

    // THEN
    expect(() => new LaunchTemplate(stack, 'Instance', {
      keyName: 'test-key-pair',
      keyPair,
    })).toThrow('Cannot specify both of \'keyName\' and \'keyPair\'; prefer \'keyPair\'');
  });

  test.each([
    [true, true],
    [false, false],
  ])('Given detailedMonitoring %p', (given: boolean, expected: boolean) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      detailedMonitoring: given,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        Monitoring: {
          Enabled: expected,
        },
      },
    });
  });

  test('Given securityGroup', () => {
    // GIVEN
    const vpc = new Vpc(stack, 'VPC');
    const sg = new SecurityGroup(stack, 'SG', { vpc });

    // WHEN
    const template = new LaunchTemplate(stack, 'Template', {
      securityGroup: sg,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        SecurityGroupIds: [
          {
            'Fn::GetAtt': [
              'SGADB53937',
              'GroupId',
            ],
          },
        ],
      },
    });
    expect(template.connections).toBeDefined();
    expect(template.connections.securityGroups).toHaveLength(1);
    expect(template.connections.securityGroups[0]).toBe(sg);
  });

  test('Adding tags', () => {
    // GIVEN
    const template = new LaunchTemplate(stack, 'Template');

    // WHEN
    Tags.of(template).add('TestKey', 'TestValue');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        TagSpecifications: [
          {
            ResourceType: 'instance',
            Tags: [
              {
                Key: 'Name',
                Value: 'Default/Template',
              },
              {
                Key: 'TestKey',
                Value: 'TestValue',
              },
            ],
          },
          {
            ResourceType: 'volume',
            Tags: [
              {
                Key: 'Name',
                Value: 'Default/Template',
              },
              {
                Key: 'TestKey',
                Value: 'TestValue',
              },
            ],
          },
        ],
      },
      TagSpecifications: [
        {
          ResourceType: 'launch-template',
          Tags: [
            {
              Key: 'Name',
              Value: 'Default/Template',
            },
            {
              Key: 'TestKey',
              Value: 'TestValue',
            },
          ],
        },
      ],
    });
  });

  test('Requires IMDSv2', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
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

  test('Associate public IP address', () => {
    // GIVEN
    const vpc = new Vpc(stack, 'VPC');
    const sg = new SecurityGroup(stack, 'SG', { vpc });

    // WHEN
    new LaunchTemplate(stack, 'Template', {
      associatePublicIpAddress: true,
      securityGroup: sg,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        NetworkInterfaces: [
          {
            DeviceIndex: 0,
            AssociatePublicIpAddress: true,
            Groups: [
              {
                'Fn::GetAtt': [
                  'SGADB53937',
                  'GroupId',
                ],
              },
            ],
          },
        ],
      },
    });
  });

  test('Dissociate public IP address', () => {
    // GIVEN
    const vpc = new Vpc(stack, 'VPC');
    const sg = new SecurityGroup(stack, 'SG', { vpc });

    // WHEN
    new LaunchTemplate(stack, 'Template', {
      associatePublicIpAddress: false,
      securityGroup: sg,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        NetworkInterfaces: [
          {
            DeviceIndex: 0,
            AssociatePublicIpAddress: false,
            Groups: [
              {
                'Fn::GetAtt': [
                  'SGADB53937',
                  'GroupId',
                ],
              },
            ],
          },
        ],
      },
    });
  });

  test('Given placementGroup', () => {
    // GIVEN
    const pg = new PlacementGroup(stack, 'PlacementGroup', {
      strategy: PlacementGroupStrategy.CLUSTER,
    });

    // WHEN
    new LaunchTemplate(stack, 'Template', {
      placementGroup: pg,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        Placement: {
          GroupName: {
            'Fn::GetAtt': [
              'PlacementGroup68E91A0F',
              'GroupName',
            ],
          },
        },
      },
    });
  });

  test('Given imported placementGroup', () => {
    // GIVEN
    const importedPg = PlacementGroup.fromPlacementGroupName(stack, 'ImportedPlacementGroup', 'my-placement-group');

    // WHEN
    new LaunchTemplate(stack, 'Template', {
      placementGroup: importedPg,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        Placement: {
          GroupName: 'my-placement-group',
        },
      },
    });
  });
});

describe('LaunchTemplate marketOptions', () => {
  let app: App;
  let stack: Stack;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app);
  });

  test('given spotOptions', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {},
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceMarketOptions: {
          MarketType: 'spot',
        },
      },
    });
  });

  test.each([
    [0, 1],
    [1, 0],
    [6, 0],
    [7, 1],
  ])('for range duration errors: %p', (duration: number, expectedErrors: number) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {
        blockDuration: Duration.hours(duration),
      },
    });

    // THEN
    const errors = Annotations.fromStack(stack).findError('/Default/Template', Match.anyValue());
    expect(errors).toHaveLength(expectedErrors);
  });

  test('for bad duration', () => {
    expect(() => {
      new LaunchTemplate(stack, 'Template', {
        spotOptions: {
          // Duration must be an integral number of hours.
          blockDuration: Duration.minutes(61),
        },
      });
    }).toThrow();
  });

  test('given blockDuration', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {
        blockDuration: Duration.hours(1),
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceMarketOptions: {
          MarketType: 'spot',
          SpotOptions: {
            BlockDurationMinutes: 60,
          },
        },
      },
    });
  });

  test.each([
    [SpotInstanceInterruption.STOP, 'stop'],
    [SpotInstanceInterruption.TERMINATE, 'terminate'],
    [SpotInstanceInterruption.HIBERNATE, 'hibernate'],
  ])('given interruptionBehavior %p', (given: SpotInstanceInterruption, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {
        interruptionBehavior: given,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceMarketOptions: {
          MarketType: 'spot',
          SpotOptions: {
            InstanceInterruptionBehavior: expected,
          },
        },
      },
    });
  });

  test.each([
    [0.001, '0.001'],
    [1, '1'],
    [2.5, '2.5'],
  ])('given maxPrice %p', (given: number, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {
        maxPrice: given,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceMarketOptions: {
          MarketType: 'spot',
          SpotOptions: {
            MaxPrice: expected,
          },
        },
      },
    });
  });

  test.each([
    [SpotRequestType.ONE_TIME, 'one-time'],
    [SpotRequestType.PERSISTENT, 'persistent'],
  ])('given requestType %p', (given: SpotRequestType, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {
        requestType: given,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceMarketOptions: {
          MarketType: 'spot',
          SpotOptions: {
            SpotInstanceType: expected,
          },
        },
      },
    });
  });

  test('given validUntil', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      spotOptions: {
        validUntil: Expiration.atTimestamp(0),
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        InstanceMarketOptions: {
          MarketType: 'spot',
          SpotOptions: {
            ValidUntil: 'Thu, 01 Jan 1970 00:00:00 GMT',
          },
        },
      },
    });
  });
});

describe('LaunchTemplate metadataOptions', () => {
  let app: App;
  let stack: Stack;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app);
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('given httpEndpoint %p', (given: boolean, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      httpEndpoint: given,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        MetadataOptions: {
          HttpEndpoint: expected,
        },
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('given httpProtocolIpv6 %p', (given: boolean, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      httpProtocolIpv6: given,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        MetadataOptions: {
          HttpProtocolIpv6: expected,
        },
      },
    });
  });

  test.each([
    [1, 1],
    [2, 2],
  ])('given httpPutResponseHopLimit %p', (given: number, expected: number) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      httpPutResponseHopLimit: given,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        MetadataOptions: {
          HttpPutResponseHopLimit: expected,
        },
      },
    });
  });

  test.each([
    [LaunchTemplateHttpTokens.OPTIONAL, 'optional'],
    [LaunchTemplateHttpTokens.REQUIRED, 'required'],
  ])('given httpTokens %p', (given: LaunchTemplateHttpTokens, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      httpTokens: given,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        MetadataOptions: {
          HttpTokens: expected,
        },
      },
    });
  });

  test.each([
    [true, 'enabled'],
    [false, 'disabled'],
  ])('given instanceMetadataTags %p', (given: boolean, expected: string) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      instanceMetadataTags: given,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::LaunchTemplate', {
      LaunchTemplateData: {
        MetadataOptions: {
          InstanceMetadataTags: expected,
        },
      },
    });
  });

  test.each([
    [0, 1],
    [-1, 1],
    [1, 0],
    [64, 0],
    [65, 1],
  ])('given instanceMetadataTags %p', (given: number, expected: number) => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      httpPutResponseHopLimit: given,
    });
    // THEN
    const errors = Annotations.fromStack(stack).findError('/Default/Template', Match.anyValue());
    expect(errors).toHaveLength(expected);
  });

  test('throw when requireImdsv2 is true and httpTokens is OPTIONAL', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      requireImdsv2: true,
      httpTokens: LaunchTemplateHttpTokens.OPTIONAL,
    });
    // THEN
    const errors = Annotations.fromStack(stack).findError('/Default/Template', Match.anyValue());
    expect(errors[0].entry.data).toMatch(/httpTokens must be required when requireImdsv2 is true/);
  });
  test('httpTokens REQUIRED is allowed when requireImdsv2 is true', () => {
    // WHEN
    new LaunchTemplate(stack, 'Template', {
      requireImdsv2: true,
      httpTokens: LaunchTemplateHttpTokens.REQUIRED,
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
});
