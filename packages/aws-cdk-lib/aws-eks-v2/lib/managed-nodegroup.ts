import type { Construct } from 'constructs';
import { Node } from 'constructs';
import type { ICluster } from './cluster';
import { Cluster, IpFamily } from './cluster';
import { CfnNodegroup } from './eks.generated';
import { InstanceType, InstanceArchitecture, InstanceClass, InstanceSize } from '../../aws-ec2';
import type { ISecurityGroup, SubnetSelection } from '../../aws-ec2';
import type { IRole } from '../../aws-iam';
import { ManagedPolicy, PolicyStatement, Role, ServicePrincipal } from '../../aws-iam';
import type { IResource, RemovalPolicy } from '../../core';
import { Resource, Annotations, withResolved, FeatureFlags, ValidationError, RemovalPolicies, UnscopedValidationError } from '../../core';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import * as cxapi from '../../cx-api';
import { isGpuInstanceType } from './private/nodegroup';
import { memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import type { INodegroupRef, NodegroupReference } from '../../interfaces/generated/aws-eks-interfaces.generated';

/**
 * NodeGroup interface
 */
export interface INodegroup extends IResource, INodegroupRef {
  /**
   * Name of the nodegroup
   * @attribute
   */
  readonly nodegroupName: string;
}

/**
 * The AMI type for your node group.
 *
 * GPU instance types should use the `AL2_x86_64_GPU` AMI type, which uses the
 * Amazon EKS-optimized Linux AMI with GPU support or the `BOTTLEROCKET_ARM_64_NVIDIA` or `BOTTLEROCKET_X86_64_NVIDIA`
 * AMI types, which uses the Amazon EKS-optimized Linux AMI with Nvidia-GPU support.
 *
 * Non-GPU instances should use the `AL2_x86_64` AMI type, which uses the Amazon EKS-optimized Linux AMI.
 */
export enum NodegroupAmiType {
  /**
   * Amazon Linux 2 (x86-64)
   */
  AL2_X86_64 = 'AL2_x86_64',
  /**
   * Amazon Linux 2 with GPU support
   */
  AL2_X86_64_GPU = 'AL2_x86_64_GPU',
  /**
   * Amazon Linux 2 (ARM-64)
   */
  AL2_ARM_64 = 'AL2_ARM_64',
  /**
   *  Bottlerocket Linux (ARM-64)
   */
  BOTTLEROCKET_ARM_64 = 'BOTTLEROCKET_ARM_64',
  /**
   * Bottlerocket (x86-64)
   */
  BOTTLEROCKET_X86_64 = 'BOTTLEROCKET_x86_64',
  /**
   *  Bottlerocket Linux with Nvidia-GPU support (ARM-64)
   */
  BOTTLEROCKET_ARM_64_NVIDIA = 'BOTTLEROCKET_ARM_64_NVIDIA',
  /**
   * Bottlerocket with Nvidia-GPU support (x86-64)
   */
  BOTTLEROCKET_X86_64_NVIDIA = 'BOTTLEROCKET_x86_64_NVIDIA',
  /**
   * Bottlerocket Linux (ARM-64) with FIPS enabled
   */
  BOTTLEROCKET_ARM_64_FIPS = 'BOTTLEROCKET_ARM_64_FIPS',
  /**
   * Bottlerocket (x86-64) with FIPS enabled
   */
  BOTTLEROCKET_X86_64_FIPS = 'BOTTLEROCKET_x86_64_FIPS',
  /**
   * Windows Core 2019 (x86-64)
   */
  WINDOWS_CORE_2019_X86_64 = 'WINDOWS_CORE_2019_x86_64',
  /**
   * Windows Core 2022 (x86-64)
   */
  WINDOWS_CORE_2022_X86_64 = 'WINDOWS_CORE_2022_x86_64',
  /**
   * Windows Full 2019 (x86-64)
   */
  WINDOWS_FULL_2019_X86_64 = 'WINDOWS_FULL_2019_x86_64',
  /**
   * Windows Full 2022 (x86-64)
   */
  WINDOWS_FULL_2022_X86_64 = 'WINDOWS_FULL_2022_x86_64',
  /**
   * Amazon Linux 2023 (x86-64)
   */
  AL2023_X86_64_STANDARD = 'AL2023_x86_64_STANDARD',
  /**
   * Amazon Linux 2023 with AWS Neuron drivers (x86-64)
   */
  AL2023_X86_64_NEURON = 'AL2023_x86_64_NEURON',
  /**
   * Amazon Linux 2023 with NVIDIA drivers (x86-64)
   */
  AL2023_X86_64_NVIDIA = 'AL2023_x86_64_NVIDIA',
  /**
   * Amazon Linux 2023 with NVIDIA drivers (ARM-64)
   */
  AL2023_ARM_64_NVIDIA = 'AL2023_ARM_64_NVIDIA',
  /**
   * Amazon Linux 2023 (ARM-64)
   */
  AL2023_ARM_64_STANDARD = 'AL2023_ARM_64_STANDARD',
}

/**
 * Capacity type of the managed node group
 */
export enum CapacityType {
  /**
   * spot instances
   */
  SPOT = 'SPOT',
  /**
   * on-demand instances
   */
  ON_DEMAND = 'ON_DEMAND',
  /**
   * capacity block instances
   */
  CAPACITY_BLOCK = 'CAPACITY_BLOCK',
}

/**
 * The remote access (SSH) configuration to use with your node group.
 *
 * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-eks-nodegroup-remoteaccess.html
 */
export interface NodegroupRemoteAccess {
  /**
   * The Amazon EC2 SSH key that provides access for SSH communication with the worker nodes in the managed node group.
   */
  readonly sshKeyName: string;
  /**
   * The security groups that are allowed SSH access (port 22) to the worker nodes. If you specify an Amazon EC2 SSH
   * key but do not specify a source security group when you create a managed node group, then port 22 on the worker
   * nodes is opened to the internet (0.0.0.0/0).
   *
   * @default - port 22 on the worker nodes is opened to the internet (0.0.0.0/0)
   */
  readonly sourceSecurityGroups?: ISecurityGroup[];
}

/**
 * Launch template property specification
 */
export interface LaunchTemplateSpec {
  /**
   * The Launch template ID
   */
  readonly id: string;
  /**
   * The launch template version to be used (optional).
   *
   * @default - the default version of the launch template
   */
  readonly version?: string;
}

/**
 * Effect types of kubernetes node taint.
 *
 * Note: These values are specifically for AWS EKS NodeGroups and use the AWS API format.
 * When using AWS CLI or API, taint effects must be NO_SCHEDULE, PREFER_NO_SCHEDULE, or NO_EXECUTE.
 * When using Kubernetes directly or kubectl, taint effects must be NoSchedule, PreferNoSchedule, or NoExecute.
 *
 * For Kubernetes manifests (like Karpenter NodePools), use string literals with PascalCase format:
 * - 'NoSchedule' instead of TaintEffect.NO_SCHEDULE
 * - 'PreferNoSchedule' instead of TaintEffect.PREFER_NO_SCHEDULE
 * - 'NoExecute' instead of TaintEffect.NO_EXECUTE
 *
 * @see https://docs.aws.amazon.com/eks/latest/userguide/node-taints-managed-node-groups.html
 */
export enum TaintEffect {
  /**
   * NoSchedule
   */
  NO_SCHEDULE = 'NO_SCHEDULE',
  /**
   * PreferNoSchedule
   */
  PREFER_NO_SCHEDULE = 'PREFER_NO_SCHEDULE',
  /**
   * NoExecute
   */
  NO_EXECUTE = 'NO_EXECUTE',
}

/**
 * Taint interface
 */
export interface TaintSpec {
  /**
   * Effect type
   *
   * @default - None
   */
  readonly effect?: TaintEffect;
  /**
   * Taint key
   *
   * @default - None
   */
  readonly key?: string;
  /**
   * Taint value
   *
   * @default - None
   */
  readonly value?: string;
}

/**
 * The Nodegroup Options for addNodeGroup() method
 */
export interface NodegroupOptions {
  /**
   * Name of the Nodegroup
   *
   * @default - resource ID
   */
  readonly nodegroupName?: string;
  /**
   * The subnets to use for the Auto Scaling group that is created for your node group. By specifying the
   * SubnetSelection, the selected subnets will automatically apply required tags i.e.
   * `kubernetes.io/cluster/CLUSTER_NAME` with a value of `shared`, where `CLUSTER_NAME` is replaced with
   * the name of your cluster.
   *
   * @default - private subnets
   */
  readonly subnets?: SubnetSelection;
  /**
   * The AMI type for your node group. If you explicitly specify the launchTemplate with custom AMI, do not specify this property, or
   * the node group deployment will fail. In other cases, you will need to specify correct amiType for the nodegroup.
   *
   * @default - auto-determined from the instanceTypes property when launchTemplateSpec property is not specified
   */
  readonly amiType?: NodegroupAmiType;
  /**
   * The root device disk size (in GiB) for your node group instances.
   *
   * @default 20
   */
  readonly diskSize?: number;
  /**
   * The current number of worker nodes that the managed node group should maintain. If not specified,
   * the nodewgroup will initially create `minSize` instances.
   *
   * @default 2
   */
  readonly desiredSize?: number;
  /**
   * The maximum number of worker nodes that the managed node group can scale out to. Managed node groups can support up to 100 nodes by default.
   *
   * @default - same as desiredSize property
   */
  readonly maxSize?: number;
  /**
   * The minimum number of worker nodes that the managed node group can scale in to. This number must be greater than or equal to zero.
   *
   * @default 1
   */
  readonly minSize?: number;
  /**
   * Force the update if the existing node group's pods are unable to be drained due to a pod disruption budget issue.
   * If an update fails because pods could not be drained, you can force the update after it fails to terminate the old
   * node whether or not any pods are
   * running on the node.
   *
   * @default true
   */
  readonly forceUpdate?: boolean;
  /**
   * The instance types to use for your node group.
   * @default t3.medium will be used according to the cloudformation document.
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-eks-nodegroup.html#cfn-eks-nodegroup-instancetypes
   */
  readonly instanceTypes?: InstanceType[];
  /**
   * The Kubernetes labels to be applied to the nodes in the node group when they are created.
   *
   * @default - None
   */
  readonly labels?: { [name: string]: string };
  /**
   * The Kubernetes taints to be applied to the nodes in the node group when they are created.
   *
   * @default - None
   */
  readonly taints?: TaintSpec[];
  /**
   * The IAM role to associate with your node group. The Amazon EKS worker node kubelet daemon
   * makes calls to AWS APIs on your behalf. Worker nodes receive permissions for these API calls through
   * an IAM instance profile and associated policies. Before you can launch worker nodes and register them
   * into a cluster, you must create an IAM role for those worker nodes to use when they are launched.
   *
   * @default - None. Auto-generated if not specified.
   */
  readonly nodeRole?: IRole;
  /**
   * The AMI version of the Amazon EKS-optimized AMI to use with your node group (for example, `1.14.7-YYYYMMDD`).
   *
   * @default - The latest available AMI version for the node group's current Kubernetes version is used.
   */
  readonly releaseVersion?: string;
  /**
   * The remote access (SSH) configuration to use with your node group. Disabled by default, however, if you
   * specify an Amazon EC2 SSH key but do not specify a source security group when you create a managed node group,
   * then port 22 on the worker nodes is opened to the internet (0.0.0.0/0)
   *
   * @default - disabled
   */
  readonly remoteAccess?: NodegroupRemoteAccess;
  /**
   * The metadata to apply to the node group to assist with categorization and organization. Each tag consists of
   * a key and an optional value, both of which you define. Node group tags do not propagate to any other resources
   * associated with the node group, such as the Amazon EC2 instances or subnets.
   *
   * @default None
   */
  readonly tags?: { [name: string]: string };
  /**
   * Launch template specification used for the nodegroup
   * @see https://docs.aws.amazon.com/eks/latest/userguide/launch-templates.html
   * @default - no launch template
   */
  readonly launchTemplateSpec?: LaunchTemplateSpec;
  /**
   * The capacity type of the nodegroup.
   *
   * @default CapacityType.ON_DEMAND
   */
  readonly capacityType?: CapacityType;

  /**
   * The maximum number of nodes unavailable at once during a version update.
   * Nodes will be updated in parallel. The maximum number is 100.
   *
   * This value or `maxUnavailablePercentage` is required to have a value for custom update configurations to be applied.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-eks-nodegroup-updateconfig.html#cfn-eks-nodegroup-updateconfig-maxunavailable
   * @default 1
   */
  readonly maxUnavailable?: number;

  /**
   * The maximum percentage of nodes unavailable during a version update.
   * This percentage of nodes will be updated in parallel, up to 100 nodes at once.
   *
   * This value or `maxUnavailable` is required to have a value for custom update configurations to be applied.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-eks-nodegroup-updateconfig.html#cfn-eks-nodegroup-updateconfig-maxunavailablepercentage
   * @default undefined - node groups will update instances one at a time
   */
  readonly maxUnavailablePercentage?: number;

  /**
   * Specifies whether to enable node auto repair for the node group. Node auto repair is disabled by default.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/node-health.html#node-auto-repair
   * @default false
   */
  readonly enableNodeAutoRepair?: boolean;

  /**
   * The removal policy applied to the managed node group resources.
   *
   * The removal policy controls what happens to the resource if it stops being managed by CloudFormation.
   * This can happen in one of three situations:
   *
   * - The resource is removed from the template, so CloudFormation stops managing it
   * - A change to the resource is made that requires it to be replaced, so CloudFormation stops managing it
   * - The stack is deleted, so CloudFormation stops managing all resources in it
   *
   * @default RemovalPolicy.DESTROY
   */
  readonly removalPolicy?: RemovalPolicy;
}

/**
 * NodeGroup properties interface
 */
export interface NodegroupProps extends NodegroupOptions {
  /**
   * Cluster resource
   */
  readonly cluster: ICluster;
}

/**
 * The Nodegroup resource class
 * @resource AWS::EKS::Nodegroup
 */
@propertyInjectable
export class Nodegroup extends Resource implements INodegroup {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-eks-v2.Nodegroup';

  /**
   * Import the Nodegroup from attributes
   */
  public static fromNodegroupName(scope: Construct, id: string, nodegroupName: string): INodegroup {
    class Import extends Resource implements INodegroup {
      public readonly nodegroupName = nodegroupName;

      public get nodegroupRef(): NodegroupReference {
        // eslint-disable-next-line @cdklabs/no-throw-default-error
        throw new Error('Cannot use Nodegroup.fromNodegroupName() in this API');
      }
    }
    return new Import(scope, id);
  }
  /**
   * the Amazon EKS cluster resource
   *
   * @attribute ClusterName
   */
  public readonly cluster: ICluster;
  /**
   * IAM role of the instance profile for the nodegroup
   */
  public readonly role: IRole;

  private readonly resource: CfnNodegroup;

  private readonly desiredSize: number;
  private readonly maxSize: number;
  private readonly minSize: number;

  constructor(scope: Construct, id: string, props: NodegroupProps) {
    super(scope, id, {
      physicalName: props.nodegroupName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.cluster = props.cluster;

    this.desiredSize = props.desiredSize ?? props.minSize ?? 2;
    this.maxSize = props.maxSize ?? this.desiredSize;
    this.minSize = props.minSize ?? 1;

    withResolved(this.desiredSize, this.maxSize, (desired, max) => {
      if (desired === undefined) {return ;}
      if (desired > max) {
        throw new ValidationError(lit`DesiredCapacityCannotBeGreaterThanMaxSize`, `Desired capacity ${desired} can't be greater than max size ${max}`, this);
      }
    });

    withResolved(this.desiredSize, this.minSize, (desired, min) => {
      if (desired === undefined) {return ;}
      if (desired < min) {
        throw new ValidationError(lit`MinimumCapacityCannotBeGreaterThanDesiredSize`, `Minimum capacity ${min} can't be greater than desired size ${desired}`, this);
      }
    });

    if (props.launchTemplateSpec && props.diskSize) {
      // see - https://docs.aws.amazon.com/eks/latest/userguide/launch-templates.html
      // and https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-eks-nodegroup.html#cfn-eks-nodegroup-disksize
      throw new ValidationError(lit`DiskSizeMustBeSpecifiedInLaunchTemplate`, 'diskSize must be specified within the launch template', this);
    }

    let possibleAmiTypes: NodegroupAmiType[] = [];

    if (props.instanceTypes && props.instanceTypes.length > 0) {
      /**
       * if the user explicitly configured instance types, we can't caculate the expected ami type as we support
       * Amazon Linux 2, Bottlerocket, and Windows now. However we can check:
       *
       * 1. instance types of different CPU architectures are not mixed(e.g. X86 with ARM).
       * 2. user-specified amiType should be included in `possibleAmiTypes`.
       */
      possibleAmiTypes = getPossibleAmiTypes(this, props.instanceTypes);

      // if the user explicitly configured an ami type, make sure it's included in the possibleAmiTypes
      if (props.amiType && !possibleAmiTypes.includes(props.amiType)) {
        throw new ValidationError(lit`AmiTypeDoesNotMatchInstanceArchitecture`, `The specified AMI does not match the instance types architecture, either specify one of ${possibleAmiTypes.join(', ').toUpperCase()} or don't specify any`, this);
      }

      // if the user explicitly configured a Windows ami type, make sure the instanceType is allowed
      if (props.amiType && windowsAmiTypes.includes(props.amiType) &&
      props.instanceTypes.filter(isWindowsSupportedInstanceType).length < props.instanceTypes.length) {
        throw new ValidationError(lit`InstanceTypeDoesNotSupportWindows`, 'The specified instanceType does not support Windows workloads. '
        + 'Amazon EC2 instance types C3, C4, D2, I2, M4 (excluding m4.16xlarge), M6a.x, and '
        + 'R3 instances aren\'t supported for Windows workloads.', this);
      }

      // Warn users when the EKS_DEFAULT_AL2023 flag is enabled and amiType is not set,
      // then GPU instance types are intentionally NOT migrated to AL2023.
      const useAL2023 = FeatureFlags.of(this).isEnabled(cxapi.EKS_DEFAULT_AL2023) ?? false;
      const isGpuNodegroup = props.instanceTypes.some(isGpuInstanceType);
      if (!props.amiType && useAL2023 && isGpuNodegroup) {
        Annotations.of(this).addWarningV2('@aws-cdk/aws-eks:gpuInstancesUseAL2',
          'GPU instance types will continue to use AL2 even with the @aws-cdk/aws-eks:defaultToAL2023 feature flag enabled because '
          + 'AL2023 splits GPU support into AL2023_X86_64_NVIDIA and AL2023_X86_64_NEURON variants. To use AL2023, explicitly set amiType to the corresponding variant.');
      }
    }

    if (!props.nodeRole) {
      const ngRole = new Role(this, 'NodeGroupRole', {
        assumedBy: new ServicePrincipal('ec2.amazonaws.com'),
      });

      ngRole.addManagedPolicy(ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'));
      ngRole.addManagedPolicy(ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'));
      ngRole.addManagedPolicy(ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'));

      // Grant additional IPv6 networking permissions if running in IPv6
      // https://docs.aws.amazon.com/eks/latest/userguide/cni-iam-role.html
      if (props.cluster.ipFamily == IpFamily.IP_V6) {
        ngRole.addToPrincipalPolicy(new PolicyStatement({
          // eslint-disable-next-line @cdklabs/no-literal-partition
          resources: ['arn:aws:ec2:*:*:network-interface/*'],
          actions: [
            'ec2:AssignIpv6Addresses',
            'ec2:UnassignIpv6Addresses',
          ],
        }));
      }
      this.role = ngRole;
    } else {
      this.role = props.nodeRole;
    }

    this.validateUpdateConfig(props.maxUnavailable, props.maxUnavailablePercentage);

    this.resource = new CfnNodegroup(this, 'Resource', {
      clusterName: this.cluster.clusterName,
      nodegroupName: props.nodegroupName,
      nodeRole: this.role.roleArn,
      subnets: this.cluster.vpc.selectSubnets(props.subnets).subnetIds,
      /**
       * Case 1: If launchTemplate is explicitly specified with custom AMI, we cannot specify amiType, or the node group deployment will fail.
       * As we don't know if the custom AMI is specified in the lauchTemplate, we just use props.amiType.
       *
       * Case 2: If launchTemplate is not specified, we try to determine amiType from the instanceTypes and it could be either AL2, AL2023, or Bottlerocket.
       * When `amiType` is undefined we fall back to `possibleAmiTypes[0]`. The first element
       * depends on the `@aws-cdk/aws-eks:defaultToAL2023` feature flag: AL2 when the flag is
       * off (default, for backwards compatibility), AL2023 when the flag is on. GPU instance types
       * continue to use `AL2_X86_64_GPU` irrespective of the feature flag.
       *
       * That being said, users now either have to explicitly specify correct amiType or just leave it undefined.
       */
      amiType: props.launchTemplateSpec ? props.amiType : (props.amiType ?? possibleAmiTypes[0]),
      capacityType: props.capacityType ? props.capacityType.valueOf() : undefined,
      diskSize: props.diskSize,
      forceUpdateEnabled: props.forceUpdate ?? true,

      // note that we don't check if a launch template is configured here (even though it might configure instance types as well)
      // because this doesn't have a default value, meaning the user had to explicitly configure this.
      instanceTypes: props.instanceTypes?.map(t => t.toString()),
      labels: props.labels,
      taints: props.taints,
      launchTemplate: props.launchTemplateSpec,
      releaseVersion: props.releaseVersion,
      remoteAccess: props.remoteAccess ? {
        ec2SshKey: props.remoteAccess.sshKeyName,
        sourceSecurityGroups: props.remoteAccess.sourceSecurityGroups ?
          props.remoteAccess.sourceSecurityGroups.map(m => m.securityGroupId) : undefined,
      } : undefined,
      scalingConfig: {
        desiredSize: this.desiredSize,
        maxSize: this.maxSize,
        minSize: this.minSize,
      },
      tags: props.tags,
      updateConfig: props.maxUnavailable || props.maxUnavailablePercentage ? {
        maxUnavailable: props.maxUnavailable,
        maxUnavailablePercentage: props.maxUnavailablePercentage,
      } : undefined,
      nodeRepairConfig: props.enableNodeAutoRepair ? {
        enabled: props.enableNodeAutoRepair,
      } : undefined,
    });

    if (this.cluster instanceof Cluster) {
      // the controller runs on the worker nodes so they cannot
      // be deleted before the controller.
      if (this.cluster.albController) {
        Node.of(this.cluster.albController).addDependency(this);
      }
    }

    if (props.removalPolicy) {
      RemovalPolicies.of(this).apply(props.removalPolicy);
    }
  }

  /**
   * ARN of the nodegroup
   *
   * @attribute
   */
  @memoizedGetter
  public get nodegroupArn(): string {
    return this.getResourceArnAttribute(this.resource.attrArn, {
      service: 'eks',
      resource: 'nodegroup',
      resourceName: this.physicalName,
    });
  }

  /**
   * Nodegroup name
   *
   * @attribute
   */
  @memoizedGetter
  public get nodegroupName(): string {
    if (FeatureFlags.of(this).isEnabled(cxapi.EKS_NODEGROUP_NAME)) {
      return this.getResourceNameAttribute(this.resource.attrNodegroupName);
    } else {
      return this.getResourceNameAttribute(this.resource.ref);
    }
  }

  public get nodegroupRef(): NodegroupReference {
    return {
      nodegroupArn: this.nodegroupArn,
      get nodegroupId(): string {
        // eslint-disable-next-line @cdklabs/no-throw-default-error
        throw new Error('Cannot get nodegroupId from this NodeGroup');
      },
    };
  }

  private validateUpdateConfig(maxUnavailable?: number, maxUnavailablePercentage?: number) {
    if (!maxUnavailable && !maxUnavailablePercentage) return;
    if (maxUnavailable && maxUnavailablePercentage) {
      throw new ValidationError(lit`MaxUnavailableAndPercentageMutuallyExclusive`, 'maxUnavailable and maxUnavailablePercentage are not allowed to be defined together', this);
    }
    if (maxUnavailablePercentage && (maxUnavailablePercentage < 1 || maxUnavailablePercentage > 100)) {
      throw new ValidationError(lit`MaxUnavailablePercentageOutOfRange`, `maxUnavailablePercentage must be between 1 and 100, got ${maxUnavailablePercentage}`, this);
    }
    if (maxUnavailable) {
      if (maxUnavailable > this.maxSize) {
        throw new ValidationError(lit`MaxUnavailableExceedsMaxSize`, `maxUnavailable must be lower than maxSize (${this.maxSize}), got ${maxUnavailable}`, this);
      }
      if (maxUnavailable < 1 || maxUnavailable > 100) {
        throw new ValidationError(lit`MaxUnavailableOutOfRange`, `maxUnavailable must be between 1 and 100, got ${maxUnavailable}`, this);
      }
    }
  }
}

/**
 * AMI types of different architectures. The first element is the default.
 * AmiType if amiType and launchTemplateSpec are both undefined.
 */
const arm64AmiTypes = (useAL2023: boolean): NodegroupAmiType[] =>
  [
    ...(useAL2023 ? [
      NodegroupAmiType.AL2023_ARM_64_STANDARD,
      NodegroupAmiType.AL2_ARM_64,
    ] : [
      NodegroupAmiType.AL2_ARM_64,
      NodegroupAmiType.AL2023_ARM_64_STANDARD,
    ]),
    NodegroupAmiType.BOTTLEROCKET_ARM_64,
  ];
const x8664AmiTypes = (useAL2023: boolean): NodegroupAmiType[] =>
  [
    ...(useAL2023 ? [
      NodegroupAmiType.AL2023_X86_64_STANDARD,
      NodegroupAmiType.AL2_X86_64,
    ] : [
      NodegroupAmiType.AL2_X86_64,
      NodegroupAmiType.AL2023_X86_64_STANDARD,
    ]),
    NodegroupAmiType.BOTTLEROCKET_X86_64,
    NodegroupAmiType.WINDOWS_CORE_2019_X86_64,
    NodegroupAmiType.WINDOWS_CORE_2022_X86_64,
    NodegroupAmiType.WINDOWS_FULL_2019_X86_64,
    NodegroupAmiType.WINDOWS_FULL_2022_X86_64,
  ];
const windowsAmiTypes: NodegroupAmiType[] = [
  NodegroupAmiType.WINDOWS_CORE_2019_X86_64,
  NodegroupAmiType.WINDOWS_CORE_2022_X86_64,
  NodegroupAmiType.WINDOWS_FULL_2019_X86_64,
  NodegroupAmiType.WINDOWS_FULL_2022_X86_64,
];
const gpuAmiTypes: NodegroupAmiType[] = [
  NodegroupAmiType.AL2_X86_64_GPU,
  NodegroupAmiType.AL2023_X86_64_NEURON,
  NodegroupAmiType.AL2023_X86_64_NVIDIA,
  NodegroupAmiType.AL2023_ARM_64_NVIDIA,
  NodegroupAmiType.BOTTLEROCKET_X86_64_NVIDIA,
  NodegroupAmiType.BOTTLEROCKET_ARM_64_NVIDIA,
];

/**
 * This function check if the instanceType is supported by Windows AMI.
 * https://docs.aws.amazon.com/eks/latest/userguide/windows-support.html
 * @param instanceType The EC2 instance type
 */
function isWindowsSupportedInstanceType(instanceType: InstanceType): boolean {
  // compare instanceType to forbidden InstanceTypes for Windows. Add exception for m6a.16xlarge.
  // NOTE: i2 instance class is not present in the InstanceClass enum.
  const forbiddenInstanceClasses: InstanceClass[] = [InstanceClass.C3, InstanceClass.C4, InstanceClass.D2, InstanceClass.M4,
    InstanceClass.M6A, InstanceClass.R3];
  return instanceType.toString() === InstanceType.of(InstanceClass.M4, InstanceSize.XLARGE16).toString() ||
    forbiddenInstanceClasses.every((c) => !instanceType.sameInstanceClassAs(InstanceType.of(c, InstanceSize.LARGE)) && !instanceType.toString().match(/^i2/));
}

type AmiArchitecture = InstanceArchitecture | 'GPU';
/**
 * This function examines the CPU architecture of every instance type and determines
 * what AMI types are compatible for all of them. it either throws or produces an array of possible AMI types because
 * instance types of different CPU architectures are not supported.
 * @param instanceTypes The instance types
 * @returns NodegroupAmiType[]
 */
function getPossibleAmiTypes(scope: Construct, instanceTypes: InstanceType[]): NodegroupAmiType[] {
  function typeToArch(instanceType: InstanceType): AmiArchitecture {
    return isGpuInstanceType(instanceType) ? 'GPU' : instanceType.architecture;
  }

  const useAL2023 = FeatureFlags.of(scope).isEnabled(cxapi.EKS_DEFAULT_AL2023) ?? false;
  const archAmiMap = new Map<AmiArchitecture, NodegroupAmiType[]>([
    [InstanceArchitecture.ARM_64, arm64AmiTypes(useAL2023)],
    [InstanceArchitecture.X86_64, x8664AmiTypes(useAL2023)],
    ['GPU', gpuAmiTypes],
  ]);
  const architectures: Set<AmiArchitecture> = new Set(instanceTypes.map(typeToArch));

  if (architectures.size === 0) { // protective code, the current implementation will never result in this.
    throw new UnscopedValidationError(lit`CannotDetermineCompatibleAmiType`, `Cannot determine any ami type compatible with instance types: ${instanceTypes.map(i => i.toString()).join(', ')}`);
  }

  if (architectures.size > 1) {
    throw new UnscopedValidationError(lit`InstanceTypesDifferentArchitecturesNotAllowed`, 'instanceTypes of different architectures is not allowed');
  }

  return archAmiMap.get(Array.from(architectures)[0])!;
}
