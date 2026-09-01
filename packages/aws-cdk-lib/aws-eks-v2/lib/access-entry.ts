import type { Construct } from 'constructs';
import type { ICluster } from './cluster';
import { CfnAccessEntry } from './eks.generated';
import type { IResource, RemovalPolicy } from '../../core';
import { Resource, Aws, ValidationError, Token } from '../../core';
import type { IArrayBox } from '../../core/lib/helpers-internal';
import { Box, memoizedGetter } from '../../core/lib/helpers-internal';
import { MethodMetadata, addConstructMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { AccessEntryReference, IAccessEntryRef } from '../../interfaces/generated/aws-eks-interfaces.generated';

/**
 * Represents an access entry in an Amazon EKS cluster.
 *
 * An access entry defines the permissions and scope for a user or role to access an Amazon EKS cluster.
 *
 * @interface IAccessEntry
 * @extends {IResource}
 * @property {string} accessEntryName - The name of the access entry.
 * @property {string} accessEntryArn - The Amazon Resource Name (ARN) of the access entry.
 */
export interface IAccessEntry extends IResource, IAccessEntryRef {
  /**
   * The name of the access entry.
   * @attribute
   */
  readonly accessEntryName: string;
  /**
   * The Amazon Resource Name (ARN) of the access entry.
   * @attribute
   */
  readonly accessEntryArn: string;
}

/**
 * Represents the attributes of an access entry.
 */
export interface AccessEntryAttributes {
  /**
   * The name of the access entry.
   */
  readonly accessEntryName: string;
  /**
   * The Amazon Resource Name (ARN) of the access entry.
   */
  readonly accessEntryArn: string;
}

/**
 * Represents the scope type of an access policy.
 *
 * The scope type determines the level of access granted by the policy.
 *
 * @export
 * @enum {string}
 */
export enum AccessScopeType {
  /**
   * The policy applies to a specific namespace within the cluster.
   */
  NAMESPACE = 'namespace',
  /**
   * The policy applies to the entire cluster.
   */
  CLUSTER = 'cluster',
}

/**
 * Represents the scope of an access policy.
 *
 * The scope defines the namespaces or cluster-level access granted by the policy.
 *
 * @interface AccessScope
 * @property {string[]} [namespaces] - The namespaces to which the policy applies, if the scope type is 'namespace'.
 * @property {AccessScopeType} type - The scope type of the policy, either 'namespace' or 'cluster'.
 */
export interface AccessScope {
  /**
   * A Kubernetes namespace that an access policy is scoped to. A value is required if you specified
   * namespace for Type.
   *
   * @default - no specific namespaces for this scope.
   */
  readonly namespaces?: string[];
  /**
   * The scope type of the policy, either 'namespace' or 'cluster'.
   */
  readonly type: AccessScopeType;
}

/**
 * Represents an Amazon EKS Access Policy ARN.
 *
 * Amazon EKS Access Policies are used to control access to Amazon EKS clusters.
 *
 * @see https://docs.aws.amazon.com/eks/latest/userguide/access-policies.html
 */
export class AccessPolicyArn {
  /**
   * The Amazon EKS Admin Policy. This access policy includes permissions that grant an IAM principal
   * most permissions to resources. When associated to an access entry, its access scope is typically
   * one or more Kubernetes namespaces.
   */
  public static readonly AMAZON_EKS_ADMIN_POLICY = AccessPolicyArn.of('AmazonEKSAdminPolicy');

  /**
   * The Amazon EKS Cluster Admin Policy. This access policy includes permissions that grant an IAM
   * principal administrator access to a cluster. When associated to an access entry, its access scope
   * is typically the cluster, rather than a Kubernetes namespace.
   */
  public static readonly AMAZON_EKS_CLUSTER_ADMIN_POLICY = AccessPolicyArn.of('AmazonEKSClusterAdminPolicy');

  /**
   * The Amazon EKS Admin View Policy. This access policy includes permissions that grant an IAM principal
   * access to list/view all resources in a cluster.
   */
  public static readonly AMAZON_EKS_ADMIN_VIEW_POLICY = AccessPolicyArn.of('AmazonEKSAdminViewPolicy');

  /**
   * The Amazon EKS Edit Policy. This access policy includes permissions that allow an IAM principal
   * to edit most Kubernetes resources.
   */
  public static readonly AMAZON_EKS_EDIT_POLICY = AccessPolicyArn.of('AmazonEKSEditPolicy');

  /**
   * The Amazon EKS View Policy. This access policy includes permissions that grant an IAM principal
   * access to list/view all resources in a cluster.
   */
  public static readonly AMAZON_EKS_VIEW_POLICY = AccessPolicyArn.of('AmazonEKSViewPolicy');

  /**
   * Creates a new instance of the AccessPolicy class with the specified policy name.
   * @param policyName The name of the access policy.
   * @returns A new instance of the AccessPolicy class.
   */
  public static of(policyName: string) { return new AccessPolicyArn(policyName); }

  /**
   * The Amazon Resource Name (ARN) of the access policy.
   */
  public readonly policyArn: string;
  /**
   * Constructs a new instance of the `AccessEntry` class.
   *
   * @param policyName - The name of the Amazon EKS access policy. This is used to construct the policy ARN.
   */
  constructor(public readonly policyName: string) {
    this.policyArn = `arn:${Aws.PARTITION}:eks::aws:cluster-access-policy/${policyName}`;
  }
}

/**
 * Represents an access policy that defines the permissions and scope for a user or role to access an Amazon EKS cluster.
 *
 * @interface IAccessPolicy
 */
export interface IAccessPolicy {
  /**
   * The scope of the access policy, which determines the level of access granted.
   */
  readonly accessScope: AccessScope;
  /**
   * The access policy itself, which defines the specific permissions.
   */
  readonly policy: string;
}

/**
 * Properties for configuring an Amazon EKS Access Policy.
 */
export interface AccessPolicyProps {
  /**
   * The scope of the access policy, which determines the level of access granted.
   */
  readonly accessScope: AccessScope;
  /**
   * The access policy itself, which defines the specific permissions.
   */
  readonly policy: AccessPolicyArn;
}

/**
 * Represents the options required to create an Amazon EKS Access Policy using the `fromAccessPolicyName()` method.
 */
export interface AccessPolicyNameOptions {
  /**
   * The scope of the access policy. This determines the level of access granted by the policy.
   */
  readonly accessScopeType: AccessScopeType;
  /**
   * An optional array of Kubernetes namespaces to which the access policy applies.
   * @default - no specific namespaces for this scope
   */
  readonly namespaces?: string[];
}

/**
 * Represents an Amazon EKS Access Policy that implements the IAccessPolicy interface.
 *
 * @implements {IAccessPolicy}
 */
export class AccessPolicy implements IAccessPolicy {
  /**
   * Import AccessPolicy by name.
   */
  public static fromAccessPolicyName(policyName: string, options: AccessPolicyNameOptions): IAccessPolicy {
    class Import implements IAccessPolicy {
      public readonly policy = `arn:${Aws.PARTITION}:eks::aws:cluster-access-policy/${policyName}`;
      public readonly accessScope: AccessScope = {
        type: options.accessScopeType,
        namespaces: options.namespaces,
      };
    }
    return new Import();
  }
  /**
   * The scope of the access policy, which determines the level of access granted.
   */
  public readonly accessScope: AccessScope;

  /**
   * The access policy itself, which defines the specific permissions.
   */
  public readonly policy: string;

  /**
   * Constructs a new instance of the AccessPolicy class.
   *
   * @param {AccessPolicyProps} props - The properties for configuring the access policy.
   */
  constructor(props: AccessPolicyProps) {
    this.accessScope = props.accessScope;
    this.policy = props.policy.policyArn;
  }
}

/**
 * Represents the different types of access entries that can be used in an Amazon EKS cluster.
 *
 * @enum {string}
 */
export enum AccessEntryType {
  /**
   * Represents a standard access entry.
   * Use this type for standard IAM principals that need cluster access with policies.
   */
  STANDARD = 'STANDARD',

  /**
   * Represents a Fargate Linux access entry.
   * Use this type for AWS Fargate profiles running Linux containers.
   */
  FARGATE_LINUX = 'FARGATE_LINUX',

  /**
   * Represents an EC2 Linux access entry.
   * Use this type for self-managed EC2 instances running Linux that join the cluster as worker nodes.
   */
  EC2_LINUX = 'EC2_LINUX',

  /**
   * Represents an EC2 Windows access entry.
   * Use this type for self-managed EC2 instances running Windows that join the cluster as worker nodes.
   */
  EC2_WINDOWS = 'EC2_WINDOWS',

  /**
   * Represents an EC2 access entry for EKS Auto Mode.
   * Use this type for node roles in EKS Auto Mode clusters where AWS automatically manages
   * the compute infrastructure. This type cannot have access policies attached.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/eks-auto-mode.html
   */
  EC2 = 'EC2',

  /**
   * Represents a Hybrid Linux access entry for EKS Hybrid Nodes.
   * Use this type for on-premises or edge infrastructure running Linux that connects
   * to your EKS cluster. This type cannot have access policies attached.
   *
   * @see https://docs.aws.amazon.com/eks/latest/userguide/hybrid-nodes.html
   */
  HYBRID_LINUX = 'HYBRID_LINUX',

  /**
   * Represents a HyperPod Linux access entry for Amazon SageMaker HyperPod.
   * Use this type for SageMaker HyperPod clusters that need access to your EKS cluster
   * for distributed machine learning workloads. This type cannot have access policies attached.
   *
   * @see https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod.html
   */
  HYPERPOD_LINUX = 'HYPERPOD_LINUX',
}

/**
 * Represents the properties required to create an Amazon EKS access entry.
 */
export interface AccessEntryProps {
  /**
   * The name of the AccessEntry.
   *
   * @default - No access entry name is provided
   */
  readonly accessEntryName?: string;
  /**
   * The type of the AccessEntry.
   *
   * @default STANDARD
   */
  readonly accessEntryType?: AccessEntryType;
  /**
   * The Amazon EKS cluster to which the access entry applies.
   */
  readonly cluster: ICluster;
  /**
   * The access policies that define the permissions and scope for the access entry.
   */
  readonly accessPolicies: IAccessPolicy[];
  /**
   * The Amazon Resource Name (ARN) of the principal (user or role) to associate the access entry with.
   */
  readonly principal: string;
  /**
   * The removal policy applied to the access entry.
   *
   * The removal policy controls what happens to the resources if they stop being managed by CloudFormation.
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
 * Represents an access entry in an Amazon EKS cluster.
 *
 * An access entry defines the permissions and scope for a user or role to access an Amazon EKS cluster.
 *
 * @implements {IAccessEntry}
 * @resource AWS::EKS::AccessEntry
 */
@propertyInjectable
@noBoxStackTraces
export class AccessEntry extends Resource implements IAccessEntry {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-eks-v2.AccessEntry';

  /**
   * Imports an `AccessEntry` from its attributes.
   *
   * @param scope - The parent construct.
   * @param id - The ID of the imported construct.
   * @param attrs - The attributes of the access entry to import.
   * @returns The imported access entry.
   */
  public static fromAccessEntryAttributes(scope: Construct, id: string, attrs: AccessEntryAttributes): IAccessEntry {
    class Import extends Resource implements IAccessEntry {
      public readonly accessEntryName = attrs.accessEntryName;
      public readonly accessEntryArn = attrs.accessEntryArn;

      public get accessEntryRef(): AccessEntryReference {
        return {
          accessEntryArn: this.accessEntryArn,
          // eslint-disable-next-line @cdklabs/no-throw-default-error
          get clusterName(): string { throw new Error('Cannot access clusterName from this AccessEntry; it has been created without knowledge of it'); },
          // eslint-disable-next-line @cdklabs/no-throw-default-error
          get principalArn(): string { throw new Error('Cannot access principalArn from this AccessEntry; it has been created without knowledge of it'); },
        };
      }
    }
    return new Import(scope, id);
  }
  private resource: CfnAccessEntry;
  private cluster: ICluster;
  private principal: string;
  private _accessPolicies: IArrayBox<IAccessPolicy>;
  private readonly accessEntryType?: AccessEntryType;

  constructor(scope: Construct, id: string, props: AccessEntryProps ) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.cluster = props.cluster;
    this.principal = props.principal;
    this._accessPolicies = Box.fromArray(props.accessPolicies, { omitEmpty: false });
    this.accessEntryType = props.accessEntryType;

    // Validate that certain access entry types cannot have access policies
    this.validateAccessPoliciesForRestrictedTypes(props.accessPolicies, props.accessEntryType);

    this.resource = new CfnAccessEntry(this, 'Resource', {
      clusterName: this.cluster.clusterName,
      principalArn: this.principal,
      type: props.accessEntryType,
      accessPolicies: this._accessPolicies.map(p => ({
        accessScope: {
          type: p.accessScope.type,
          namespaces: p.accessScope.namespaces,
        },
        policyArn: p.policy,
      })),
    });

    if (props.removalPolicy) {
      this.resource.applyRemovalPolicy(props.removalPolicy);
    }
  }

  @memoizedGetter
  public get accessEntryName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  @memoizedGetter
  public get accessEntryArn(): string {
    return this.getResourceArnAttribute(this.resource.attrAccessEntryArn, {
      service: 'eks',
      resource: 'accessentry',
      resourceName: this.physicalName,
    });
  }

  /**
   * Add the access policies for this entry.
   * @param newAccessPolicies - The new access policies to add.
   */
  @MethodMetadata()
  public addAccessPolicies(newAccessPolicies: IAccessPolicy[]): void {
    // Validate that restricted access entry types cannot have access policies
    this.validateAccessPoliciesForRestrictedTypes(newAccessPolicies, this.accessEntryType);
    // add newAccessPolicies to this.accessPolicies
    this._accessPolicies.push(...newAccessPolicies);
  }

  public get accessEntryRef(): AccessEntryReference {
    return {
      accessEntryArn: this.accessEntryArn,
      clusterName: this.cluster.clusterName,
      principalArn: this.principal,
    };
  }

  /**
   * Validates that restricted access entry types cannot have access policies attached.
   *
   * @param accessPolicies - The access policies to validate
   * @param accessEntryType - The access entry type to check
   * @throws {ValidationError} If a restricted access entry type has access policies
   * @private
   */
  private validateAccessPoliciesForRestrictedTypes(accessPolicies: IAccessPolicy[], accessEntryType?: AccessEntryType): void {
    const restrictedTypes = [AccessEntryType.EC2, AccessEntryType.HYBRID_LINUX, AccessEntryType.HYPERPOD_LINUX];
    if (accessEntryType && restrictedTypes.includes(accessEntryType) &&
        !Token.isUnresolved(accessPolicies) && accessPolicies.length > 0) {
      throw new ValidationError(lit`AccessEntryTypeCannot`, `Access entry type '${accessEntryType}' cannot have access policies attached. Use AccessEntryType.STANDARD for access entries that require policies.`, this);
    }
  }
}
