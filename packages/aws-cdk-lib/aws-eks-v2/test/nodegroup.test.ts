import { testFixture } from './util';
import { Annotations, Match, Template } from '../../assertions';
import * as ec2 from '../../aws-ec2';
import * as cdk from '../../core';
import * as cxapi from '../../cx-api';
import type { CfnNodegroup } from '../lib';
import * as eks from '../lib';
import { NodegroupAmiType, TaintEffect } from '../lib';
import { isGpuInstanceType } from '../lib/private/nodegroup';

const CLUSTER_VERSION = eks.KubernetesVersion.V1_31;

const commonProps = {
  version: CLUSTER_VERSION,
  defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
  defaultCapacity: 0,
};

describe('node group', () => {
  test('default ami type is not applied when launch template is configured', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    const launchTemplate = new ec2.CfnLaunchTemplate(stack, 'LaunchTemplate', {
      launchTemplateData: {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.MEDIUM).toString(),
      },
    });

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.LARGE)],
      launchTemplateSpec: {
        id: launchTemplate.ref,
        version: launchTemplate.attrLatestVersionNumber,
      },
    });

    // THEN
    const root = stack.node.root as cdk.App;
    const stackArtifact = root.synth().getStackByName(stack.stackName);
    expect(stackArtifact.template.Resources.Nodegroup62B4B2C1.Properties.AmiType).toBeUndefined();
  });

  test('explicit ami type is applied even when launch template is configured', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    const launchTemplate = new ec2.CfnLaunchTemplate(stack, 'LaunchTemplate', {
      launchTemplateData: {
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.MEDIUM).toString(),
      },
    });

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: eks.NodegroupAmiType.AL2_X86_64,
      launchTemplateSpec: {
        id: launchTemplate.ref,
        version: launchTemplate.attrLatestVersionNumber,
      },
    });

    // THEN
    const root = stack.node.root as cdk.App;
    const stackArtifact = root.synth().getStackByName(stack.stackName);
    expect(stackArtifact.template.Resources.Nodegroup62B4B2C1.Properties.AmiType).toEqual('AL2_x86_64');
  });

  test('ami type is taken as is when no instance types are configured', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: eks.NodegroupAmiType.AL2_X86_64_GPU,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2_x86_64_GPU',
    });
  });

  test('create a default nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', { cluster });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
    });
  });

  test('create a x86_64 AL2023 nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.AL2023_X86_64_STANDARD,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'AL2023_x86_64_STANDARD',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
    });
  });

  test('create a ARM64 AL2023 nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.AL2023_ARM_64_STANDARD,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'AL2023_ARM_64_STANDARD',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
    });
  });

  test('create a x86_64 bottlerocket nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.BOTTLEROCKET_X86_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'BOTTLEROCKET_x86_64',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
    });
  });

  test('create a ARM_64 bottlerocket nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.BOTTLEROCKET_ARM_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'BOTTLEROCKET_ARM_64',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
    });
  });

  test('create a x86_64 Windows Core 2019 nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.WINDOWS_CORE_2019_X86_64,
      taints: [
        {
          effect: TaintEffect.NO_SCHEDULE,
          key: 'os',
          value: 'windows',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'WINDOWS_CORE_2019_x86_64',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
      Taints: [
        {
          Effect: 'NO_SCHEDULE',
          Key: 'os',
          Value: 'windows',
        },
      ],
    });
  });

  test('create a x86_64 Windows Core 2022 nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.WINDOWS_CORE_2022_X86_64,
      taints: [
        {
          effect: TaintEffect.NO_SCHEDULE,
          key: 'os',
          value: 'windows',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'WINDOWS_CORE_2022_x86_64',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
      Taints: [
        {
          Effect: 'NO_SCHEDULE',
          Key: 'os',
          Value: 'windows',
        },
      ],
    });
  });

  test('create a x86_64 Windows Full 2019 nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.WINDOWS_FULL_2019_X86_64,
      taints: [
        {
          effect: TaintEffect.NO_SCHEDULE,
          key: 'os',
          value: 'windows',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'WINDOWS_FULL_2019_x86_64',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
      Taints: [
        {
          Effect: 'NO_SCHEDULE',
          Key: 'os',
          Value: 'windows',
        },
      ],
    });
  });

  test('create a x86_64 Windows Full 2022 nodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      amiType: NodegroupAmiType.WINDOWS_FULL_2022_X86_64,
      taints: [
        {
          effect: TaintEffect.NO_SCHEDULE,
          key: 'os',
          value: 'windows',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'NodegroupNodeGroupRole038A128B',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      AmiType: 'WINDOWS_FULL_2022_x86_64',
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
      Taints: [
        {
          Effect: 'NO_SCHEDULE',
          Key: 'os',
          Value: 'windows',
        },
      ],
    });
  });

  /**
   * When LaunchTemplate and amiType are undefined and instanceTypes are x86_64 instances,
   * the amiType should be implicitly set as AL2_x86_64.
   */
  test('amiType should be AL2_x86_64 with LaunchTemplate and amiType undefined and instanceTypes is x86_64', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('m5.large'),
        new ec2.InstanceType('c5.large'),
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2_x86_64',
    });
  });

  /**
   * When LaunchTemplate and amiType are both undefined and instanceTypes are ARM64 instances,
   * the amiType should be implicitly set as AL2_ARM_64.
   */
  test('amiType should be AL2_ARM_64 with LaunchTemplate and amiType undefined and instanceTypes is ARM_64', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('c6g.large'),
        new ec2.InstanceType('t4g.large'),
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2_ARM_64',
    });
  });

  /**
   * When LaunchTemplate and amiType are both undefined and instanceTypes are GPU instances,
   * the amiType should be implicitly set as AL2_x86_64_GPU.
   */
  test('amiType should be AL2_x86_64_GPU with LaunchTemplate and amiType undefined and instanceTypes is GPU', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('p3.large'),
        new ec2.InstanceType('g3.large'),
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2_x86_64_GPU',
    });
  });

  /**
   * When EKS_DEFAULT_AL2023 feature flag is disabled, the amiType should remain AL2.
   */
  test('amiType defaults to AL2_x86_64 when EKS_DEFAULT_AL2023 flag is disabled', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    stack.node.setContext(cxapi.EKS_DEFAULT_AL2023, false);

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [new ec2.InstanceType('c6g.large')],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2_ARM_64',
    });
  });

  /**
   * When EKS_DEFAULT_AL2023 feature flag is enabled and instanceTypes are x86_64,
   * the amiType should be implicitly set as AL2023_x86_64_STANDARD.
   */
  test('amiType should be AL2023_x86_64_STANDARD when EKS_DEFAULT_AL2023 flag is enabled and instanceTypes is x86_64', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    stack.node.setContext(cxapi.EKS_DEFAULT_AL2023, true);

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('m5.large'),
        new ec2.InstanceType('c5.large'),
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2023_x86_64_STANDARD',
    });
  });

  /**
   * When EKS_DEFAULT_AL2023 feature flag is enabled and instanceTypes are ARM64,
   * the amiType should be implicitly set as AL2023_ARM_64_STANDARD.
   */
  test('amiType should be AL2023_ARM_64_STANDARD when EKS_DEFAULT_AL2023 flag is enabled and instanceTypes is ARM_64', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    stack.node.setContext(cxapi.EKS_DEFAULT_AL2023, true);

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('c6g.large'),
        new ec2.InstanceType('t4g.large'),
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2023_ARM_64_STANDARD',
    });
  });

  /**
   * When EKS_DEFAULT_AL2023 feature flag is enabled and instanceTypes are GPU,
   * the amiType should remain AL2_x86_64_GPU (GPU instances are excluded from the flag).
   */
  test('amiType should remain AL2_x86_64_GPU when EKS_DEFAULT_AL2023 flag is enabled and instanceTypes is GPU', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    stack.node.setContext(cxapi.EKS_DEFAULT_AL2023, true);

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('p3.large'),
        new ec2.InstanceType('g3.large'),
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'AL2_x86_64_GPU',
    });

    Annotations.fromStack(stack).hasWarning('/Stack/Nodegroup',
      Match.stringLikeRegexp('GPU instance types will continue to use AL2'));
  });

  /**
   * When EKS_DEFAULT_AL2023 flag is enabled and the user explicitly sets amiType, no
   * warning should fire (explicit user choice opts out of both warnings).
   */
  test('no warning when EKS_DEFAULT_AL2023 flag is enabled and amiType is set explicitly', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    stack.node.setContext(cxapi.EKS_DEFAULT_AL2023, true);

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [new ec2.InstanceType('m5.large')],
      amiType: NodegroupAmiType.AL2_X86_64,
    });

    // THEN
    Annotations.fromStack(stack).hasNoWarning('/Stack/Nodegroup',
      Match.stringLikeRegexp('GPU instance types will continue to use AL2'));
  });

  /**
   * When LaunchTemplate is undefined, amiType is AL2_x86_64 and instanceTypes are not x86_64,
   * we should throw an error.
   */
  test('throws when LaunchTemplate is undefined, amiType is AL2_x86_64 and instanceTypes are not x86_64', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });

    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.AL2_X86_64,
      instanceTypes: [
        new ec2.InstanceType('p3.large'),
        new ec2.InstanceType('g3.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_X86_64_GPU, AL2023_X86_64_NEURON, AL2023_X86_64_NVIDIA, AL2023_ARM_64_NVIDIA, BOTTLEROCKET_X86_64_NVIDIA, BOTTLEROCKET_ARM_64_NVIDIA or don't specify any/);
  });

  /**
   * When LaunchTemplate is undefined, amiType is AL2023_X86_64_STANDARD and instanceTypes are not x86_64,
   * we should throw an error.
   */
  test('throws when LaunchTemplate is undefined, amiType is AL2023_X86_64_STANDARD and instanceTypes are not x86_64', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.AL2023_X86_64_STANDARD,
      instanceTypes: [
        new ec2.InstanceType('c6g.large'),
        new ec2.InstanceType('t4g.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_ARM_64, AL2023_ARM_64_STANDARD, BOTTLEROCKET_ARM_64 or don't specify any/);
  });

  /**
   * When LaunchTemplate is undefined, amiType is AL2023_ARM_64_STANDARD and instanceTypes are not ARM_64,
   * we should throw an error.
   */
  test('throws when LaunchTemplate is undefined, amiType is AL2023_ARM_64_STANDARD and instanceTypes are not ARM_64', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.AL2023_ARM_64_STANDARD,
      instanceTypes: [
        new ec2.InstanceType('m5.large'),
        new ec2.InstanceType('c5.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_X86_64, AL2023_X86_64_STANDARD, BOTTLEROCKET_X86_64, WINDOWS_CORE_2019_X86_64, WINDOWS_CORE_2022_X86_64, WINDOWS_FULL_2019_X86_64, WINDOWS_FULL_2022_X86_64 or don't specify any/);
  });

  /**
   * When LaunchTemplate is undefined, amiType is AL2_ARM_64 and instanceTypes are not ARM_64,
   * we should throw an error.
   */
  test('throws when LaunchTemplate is undefined, amiType is AL2_ARM_64 and instanceTypes are not ARM_64', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.AL2_ARM_64,
      instanceTypes: [
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('m5.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_X86_64, AL2023_X86_64_STANDARD, BOTTLEROCKET_X86_64, WINDOWS_CORE_2019_X86_64, WINDOWS_CORE_2022_X86_64, WINDOWS_FULL_2019_X86_64, WINDOWS_FULL_2022_X86_64 or don't specify any/);
  });

  test('throws when AmiType is Windows and forbidden instanceType is selected', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.WINDOWS_FULL_2022_X86_64,
      instanceTypes: [
        new ec2.InstanceType('c3.large'),
      ],
    })).toThrow(/The specified instanceType does not support Windows workloads. Amazon EC2 instance types C3, C4, D2, I2, M4 \(excluding m4.16xlarge\), M6a.x, and R3 instances aren't supported for Windows workloads./);
  });

  /**
   * When LaunchTemplate is undefined, amiType is AL2_x86_64_GPU and instanceTypes are not GPU instances,
   * we should throw an error.
   */
  test('throws when LaunchTemplate is undefined, amiType is AL2_X86_64_GPU and instanceTypes are not X86_64_GPU', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.AL2_X86_64_GPU,
      instanceTypes: [
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('m5.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_X86_64, AL2023_X86_64_STANDARD, BOTTLEROCKET_X86_64, WINDOWS_CORE_2019_X86_64, WINDOWS_CORE_2022_X86_64, WINDOWS_FULL_2019_X86_64, WINDOWS_FULL_2022_X86_64 or don't specify any/);
  });

  test('throws when LaunchTemplate is undefined, amiType is BOTTLEROCKET_ARM_64_NVIDIA and instanceTypes are not GPU', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.BOTTLEROCKET_ARM_64_NVIDIA,
      instanceTypes: [
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('m5.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_X86_64, AL2023_X86_64_STANDARD, BOTTLEROCKET_X86_64, WINDOWS_CORE_2019_X86_64, WINDOWS_CORE_2022_X86_64, WINDOWS_FULL_2019_X86_64, WINDOWS_FULL_2022_X86_64 or don't specify any/);
  });

  test('throws when LaunchTemplate is undefined, amiType is BOTTLEROCKET_X86_64_NVIDIA and instanceTypes are not GPU', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      amiType: NodegroupAmiType.BOTTLEROCKET_X86_64_NVIDIA,
      instanceTypes: [
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('m5.large'),
      ],
    })).toThrow(/The specified AMI does not match the instance types architecture, either specify one of AL2_X86_64, AL2023_X86_64_STANDARD, BOTTLEROCKET_X86_64, WINDOWS_CORE_2019_X86_64, WINDOWS_CORE_2022_X86_64, WINDOWS_FULL_2019_X86_64, WINDOWS_FULL_2022_X86_64 or don't specify any/);
  });

  /**
   * When LaunchTemplate is defined, amiType is undefined and instanceTypes are GPU instances,
   * we should deploy correctly.
   */
  test('deploy correctly with defined LaunchTemplate and instanceTypes(GPU) and amiType undefined.', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    const ng = new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('p3.large'),
        new ec2.InstanceType('g3.large'),
      ],
      launchTemplateSpec: {
        id: 'mock',
      },
    });

    // THEN
    expect(ng).not.toHaveProperty('AmiType');
  });

  /**
   * When LaunchTemplate is defined, amiType is undefined and instanceTypes are x86_64 instances,
   * we should deploy correctly.
   */
  test('deploy correctly with defined LaunchTemplate and instanceTypes(x86_64) and amiType undefined.', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    const ng = new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('m5.large'),
      ],
      launchTemplateSpec: {
        id: 'mock',
      },
    });

    // THEN
    expect(ng).not.toHaveProperty('AmiType');
  });

  /**
   * When LaunchTemplate is defined, amiType is undefined and instanceTypes are ARM_64 instances,
   * we should deploy correctly.
   */
  test('deploy correctly with defined LaunchTemplate and instanceTypes(ARM_64) and amiType undefined.', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    const ng = new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('c6g.large'),
        new ec2.InstanceType('t4g.large'),
      ],
      launchTemplateSpec: {
        id: 'mock',
      },
    });

    // THEN
    expect(ng).not.toHaveProperty('AmiType');
  });

  /**
   * BOTTLEROCKET_X86_64 with defined instance types w/o launchTemplateSpec should deploy correctly.
   */
  test('BOTTLEROCKET_X86_64 with defined instance types w/o launchTemplateSpec should deploy correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    cluster.addNodegroupCapacity('bottlerocket', {
      instanceTypes: [new ec2.InstanceType('m5a.xlarge')],
      amiType: NodegroupAmiType.BOTTLEROCKET_X86_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'BOTTLEROCKET_x86_64',
    });
  });

  /**
   * BOTTLEROCKET_ARM_64 with defined instance types w/o launchTemplateSpec should deploy correctly.
   */
  test('BOTTLEROCKET_ARM_64 with defined instance types w/o launchTemplateSpec should deploy correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
      // THEN
    cluster.addNodegroupCapacity('bottlerocket', {
      instanceTypes: [new ec2.InstanceType('c6g.xlarge')],
      amiType: NodegroupAmiType.BOTTLEROCKET_ARM_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'BOTTLEROCKET_ARM_64',
    });
  });

  /**
   * WINDOWS_CORE_2019_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly.
   */
  test('WINDOWS_CORE_2019_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    cluster.addNodegroupCapacity('windows', {
      instanceTypes: [new ec2.InstanceType('m5a.xlarge')],
      amiType: NodegroupAmiType.WINDOWS_CORE_2019_X86_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'WINDOWS_CORE_2019_x86_64',
    });
  });

  /**
   * WINDOWS_CORE_2022_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly.
   */
  test('WINDOWS_CORE_2019_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    cluster.addNodegroupCapacity('windows', {
      instanceTypes: [new ec2.InstanceType('m5a.xlarge')],
      amiType: NodegroupAmiType.WINDOWS_CORE_2022_X86_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'WINDOWS_CORE_2022_x86_64',
    });
  });

  /**
   * WINDOWS_FULL_2019_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly.
   */
  test('WINDOWS_FULL_2019_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    cluster.addNodegroupCapacity('windows', {
      instanceTypes: [new ec2.InstanceType('m5a.xlarge')],
      amiType: NodegroupAmiType.WINDOWS_FULL_2019_X86_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'WINDOWS_FULL_2019_x86_64',
    });
  });

  /**
   * WINDOWS_FULL_2022_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly.
   */
  test('WINDOWS_FULL_2022_x86_64 with defined instance types w/o launchTemplateSpec should deploy correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    cluster.addNodegroupCapacity('windows', {
      instanceTypes: [new ec2.InstanceType('m5a.xlarge')],
      amiType: NodegroupAmiType.WINDOWS_FULL_2022_X86_64,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      AmiType: 'WINDOWS_FULL_2022_x86_64',
    });
  });

  test('create nodegroup correctly with security groups provided', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      remoteAccess: {
        sshKeyName: 'foo',
        sourceSecurityGroups: [new ec2.SecurityGroup(stack, 'SG', { vpc })],
      },
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      RemoteAccess: {
        Ec2SshKey: 'foo',
        SourceSecurityGroups: [
          {
            'Fn::GetAtt': [
              'SGADB53937',
              'GroupId',
            ],
          },
        ],
      },
    });
  });

  test('create nodegroup correctly with enableNodeAutoRepair provided', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      enableNodeAutoRepair: true,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      NodeRepairConfig: {
        Enabled: true,
      },
    });
  });

  test('create nodegroup with forceUpdate disabled', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', { cluster, forceUpdate: false });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ForceUpdateEnabled: false,
    });
  });

  test('create nodegroup with instanceTypes provided', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [new ec2.InstanceType('m5.large')],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      InstanceTypes: [
        'm5.large',
      ],
    });
  });

  test('create nodegroup with on-demand capacity type', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [new ec2.InstanceType('m5.large')],
      capacityType: eks.CapacityType.ON_DEMAND,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      InstanceTypes: [
        'm5.large',
      ],
      CapacityType: 'ON_DEMAND',
    });
  });

  test('create nodegroup with spot capacity type', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('m5.large'),
        new ec2.InstanceType('t3.large'),
        new ec2.InstanceType('c5.large'),
      ],
      capacityType: eks.CapacityType.SPOT,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      InstanceTypes: [
        'm5.large',
        't3.large',
        'c5.large',
      ],
      CapacityType: 'SPOT',
    });
  });

  test('create nodegroup with on-demand capacity type and multiple instance types', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      instanceTypes: [
        new ec2.InstanceType('m5.large'),
        new ec2.InstanceType('t3.large'),
        new ec2.InstanceType('c5.large'),
      ],
      capacityType: eks.CapacityType.ON_DEMAND,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      InstanceTypes: [
        'm5.large',
        't3.large',
        'c5.large',
      ],
      CapacityType: 'ON_DEMAND',
    });
  });

  test('create nodegroup with neither instanceTypes nor instanceType defined', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      version: CLUSTER_VERSION,
      defaultCapacityType: eks.DefaultCapacityType.NODEGROUP,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      capacityType: eks.CapacityType.SPOT,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      CapacityType: 'SPOT',
    });
  });

  test('throws when instanceTypes provided with different CPU architrcture', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      instanceTypes: [
        // X86
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('c5a.large'),
        // ARM64
        new ec2.InstanceType('m6g.large'),
      ],
    })).toThrow(/instanceTypes of different architectures is not allowed/);
  });

  test('throws when amiType provided is incorrect', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', {
      instanceTypes: [
        new ec2.InstanceType('c5.large'),
        new ec2.InstanceType('c5a.large'),
        new ec2.InstanceType('c5d.large'),
      ],
      // incorrect amiType
      amiType: eks.NodegroupAmiType.AL2_ARM_64,
    })).toThrow(/The specified AMI does not match the instance types architecture/);
  });

  test('remoteAccess without security group provided', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      remoteAccess: {
        sshKeyName: 'foo',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      RemoteAccess: {
        Ec2SshKey: 'foo',
      },
    });
  });

  test('import nodegroup correctly', () => {
    // GIVEN
    const { stack: stack1, vpc, app } = testFixture();
    const stack2 = new cdk.Stack(app, 'stack2', { env: { region: 'us-east-1' } });
    const cluster = new eks.Cluster(stack1, 'Cluster', {
      vpc,
      ...commonProps,
    });

    // WHEN
    // const cluster = new eks.Cluster(stack, 'Cluster', { vpc, kubectlEnabled: true, defaultCapacity: 0 });
    const ng = new eks.Nodegroup(stack1, 'Nodegroup', { cluster });
    const imported = eks.Nodegroup.fromNodegroupName(stack2, 'ImportedNg', ng.nodegroupName);
    new cdk.CfnOutput(stack2, 'NodegroupName', { value: imported.nodegroupName });

    // THEN
    Template.fromStack(stack2).templateMatches({
      Outputs: {
        NodegroupName: {
          Value: {
            'Fn::ImportValue': 'Stack:ExportsOutputRefNodegroup62B4B2C1EF8AB7C1',
          },
        },
      },
    });
  });

  test('addNodegroup correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });

    // WHEN
    cluster.addNodegroupCapacity('ng');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      NodeRole: {
        'Fn::GetAtt': [
          'ClusterNodegroupngNodeGroupRoleDA0D35DA',
          'Arn',
        ],
      },
      Subnets: [
        {
          Ref: 'VPCPrivateSubnet1Subnet8BCA10E0',
        },
        {
          Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A',
        },
      ],
      ForceUpdateEnabled: true,
      ScalingConfig: {
        DesiredSize: 2,
        MaxSize: 2,
        MinSize: 1,
      },
    });
  });

  test('add node group with taints', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });

    // WHEN
    cluster.addNodegroupCapacity('ng', {
      taints: [
        {
          effect: eks.TaintEffect.NO_SCHEDULE,
          key: 'foo',
          value: 'bar',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ClusterName: {
        Ref: 'ClusterEB0386A7',
      },
      Taints: [
        {
          Effect: 'NO_SCHEDULE',
          Key: 'foo',
          Value: 'bar',
        },
      ],
    });
  });

  test('throws when desiredSize > maxSize', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { desiredSize: 3, maxSize: 2 })).toThrow(/Desired capacity 3 can't be greater than max size 2/);
  });

  test('throws when desiredSize < minSize', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { desiredSize: 2, minSize: 3 })).toThrow(/Minimum capacity 3 can't be greater than desired size 2/);
  });

  test('can set minSize , maxSize and DesiredSize', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // WHEN
    new eks.Nodegroup(stack, 'NodeGroup', {
      cluster: cluster,
      minSize: 2,
      maxSize: 6,
      desiredSize: 4,
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ScalingConfig: {
        MinSize: 2,
        MaxSize: 6,
        DesiredSize: 4,
      },
    });
  });

  test('validation is not performed when using Tokens', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // WHEN
    new eks.Nodegroup(stack, 'NodeGroup', {
      cluster: cluster,
      minSize: cdk.Lazy.number({ produce: () => 5 }),
      maxSize: cdk.Lazy.number({ produce: () => 1 }),
      desiredSize: cdk.Lazy.number({ produce: () => 20 }),
    });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      ScalingConfig: {
        MinSize: 5,
        MaxSize: 1,
        DesiredSize: 20,
      },
    });
  });

  test('create nodegroup correctly with launch template', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'set -o xtrace',
      `/etc/eks/bootstrap.sh ${cluster.clusterName}`,
    );
    const lt = new ec2.CfnLaunchTemplate(stack, 'LaunchTemplate', {
      launchTemplateData: {
        imageId: new eks.EksOptimizedImage().getImage(stack).imageId,
        instanceType: new ec2.InstanceType('t3.small').toString(),
        userData: cdk.Fn.base64(userData.render()),
      },
    });
    cluster.addNodegroupCapacity('ng-lt', {
      launchTemplateSpec: {
        id: lt.ref,
        version: lt.attrDefaultVersionNumber,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      LaunchTemplate: {
        Id: {
          Ref: 'LaunchTemplate',
        },
        Version: {
          'Fn::GetAtt': [
            'LaunchTemplate',
            'DefaultVersionNumber',
          ],
        },
      },
    });
  });

  test('throws when both diskSize and launch template specified', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      'set -o xtrace',
      `/etc/eks/bootstrap.sh ${cluster.clusterName}`,
    );
    const lt = new ec2.CfnLaunchTemplate(stack, 'LaunchTemplate', {
      launchTemplateData: {
        imageId: new eks.EksOptimizedImage().getImage(stack).imageId,
        instanceType: new ec2.InstanceType('t3.small').toString(),
        userData: cdk.Fn.base64(userData.render()),
      },
    });
    // THEN
    expect(() =>
      cluster.addNodegroupCapacity('ng-lt', {
        diskSize: 100,
        launchTemplateSpec: {
          id: lt.ref,
          version: lt.attrDefaultVersionNumber,
        },
      })).toThrow(/diskSize must be specified within the launch template/);
  });

  test('create updateConfig for maxUnavailable correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      maxUnavailable: 3,
      maxSize: 5,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      UpdateConfig: {
        MaxUnavailable: 3,
      },
    });
  });

  test('create updateConfig for maxUnavailablePercentage correctly', () => {
    // GIVEN
    const { stack, vpc } = testFixture();

    // WHEN
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      maxUnavailablePercentage: 33,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EKS::Nodegroup', {
      UpdateConfig: {
        MaxUnavailablePercentage: 33,
      },
    });
  });

  test('EKS_NODEGROUP_NAME feature flag should return correct nodegroupName', () => {
    // GIVEN
    const app = new cdk.App();
    const stackWithFlag = new cdk.Stack(app, 'StackWithFlag', {
      env: { account: '1234', region: 'testregion' },
    });

    // WHEN
    stackWithFlag.node.setContext(cxapi.EKS_NODEGROUP_NAME, true);
    const cluster = new eks.Cluster(stackWithFlag, 'Cluster', {
      ...commonProps,
    });
    const ng = new eks.Nodegroup(stackWithFlag, 'Nodegroup', {
      cluster,
    });

    // THEN
    expect(ng.nodegroupName).not.toEqual((ng.node.defaultChild as CfnNodegroup).ref);
  });

  test('throws when maxUnavailable and maxUnavailablePercentage are set', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { maxUnavailable: 3, maxUnavailablePercentage: 2 })).toThrow(/maxUnavailable and maxUnavailablePercentage are not allowed to be defined together/);
  });

  test('throws when maxUnavailable is greater than maxSize', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { maxUnavailable: 5, maxSize: 4 })).toThrow(/maxUnavailable must be lower than maxSize/);
  });

  test('throws when maxUnavailable is less than 1', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { maxUnavailable: -3, maxSize: 10 })).toThrow(/maxUnavailable must be between 1 and 100/);
  });

  test('throws when maxUnavailable is greater than 100', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { maxUnavailable: 101, maxSize: 200 })).toThrow(/maxUnavailable must be between 1 and 100/);
  });

  test('throws when maxUnavailablePercentage is less than 1', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { maxUnavailablePercentage: -3, maxSize: 10 })).toThrow(/maxUnavailablePercentage must be between 1 and 100/);
  });

  test('throws when maxUnavailablePercentage is greater than 100', () => {
    // GIVEN
    const { stack, vpc } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', {
      vpc,
      ...commonProps,
    });
    // THEN
    expect(() => cluster.addNodegroupCapacity('ng', { maxUnavailablePercentage: 101 })).toThrow(/maxUnavailablePercentage must be between 1 and 100/);
  });
});

describe('isGpuInstanceType', () => {
  it('should return true for known GPU instance types', () => {
    const gpuInstanceTypes = [
      ec2.InstanceType.of(ec2.InstanceClass.P2, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G3, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.P4D, ec2.InstanceSize.LARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.MEDIUM),
      ec2.InstanceType.of(ec2.InstanceClass.G6E, ec2.InstanceSize.XLARGE2),
      ec2.InstanceType.of(ec2.InstanceClass.INF1, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.INF2, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.P3, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.P3DN, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.P4DE, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G4AD, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G4DN, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G3S, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G5, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G5G, ec2.InstanceSize.XLARGE),
    ];
    gpuInstanceTypes.forEach(instanceType => {
      expect(isGpuInstanceType(instanceType)).toBe(true);
    });
  });
  it('should return false for non-GPU instance types', () => {
    const nonGpuInstanceTypes = [
      ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
      ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE),
      ec2.InstanceType.of(ec2.InstanceClass.C5, ec2.InstanceSize.XLARGE),
    ];
    nonGpuInstanceTypes.forEach(instanceType => {
      expect(isGpuInstanceType(instanceType)).toBe(false);
    });
  });
  it('should return true for different sizes of GPU instance types', () => {
    const gpuInstanceTypes = [
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.XLARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.XLARGE16),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.XLARGE48),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.LARGE),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.MEDIUM),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.SMALL),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.NANO),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.MICRO),
      ec2.InstanceType.of(ec2.InstanceClass.G6, ec2.InstanceSize.METAL),
    ];
    gpuInstanceTypes.forEach(instanceType => {
      expect(isGpuInstanceType(instanceType)).toBe(true);
    });
  });

  test('applies removal policy', () => {
    const { stack } = testFixture();
    const cluster = new eks.Cluster(stack, 'Cluster', { version: CLUSTER_VERSION });

    new eks.Nodegroup(stack, 'Nodegroup', {
      cluster,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    Template.fromStack(stack).hasResource('AWS::EKS::Nodegroup', {
      DeletionPolicy: 'Retain',
    });
  });
});
