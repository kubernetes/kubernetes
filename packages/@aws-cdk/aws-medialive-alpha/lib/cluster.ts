import type { IResource } from 'aws-cdk-lib';
import { Lazy, Names, Resource } from 'aws-cdk-lib';
import type { IRole } from 'aws-cdk-lib/aws-iam';
import type { ClusterReference, IClusterRef } from 'aws-cdk-lib/aws-medialive';
import { CfnCluster } from 'aws-cdk-lib/aws-medialive';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import { extractResourceId } from './shared';

/**
 * The hardware type for the cluster.
 */
export class ClusterType {
  /** On-premises cluster */
  public static readonly ON_PREMISES = new ClusterType('ON_PREMISES');
  /** AWS Outposts rack */
  public static readonly OUTPOSTS_RACK = new ClusterType('OUTPOSTS_RACK');
  /** AWS Outposts server */
  public static readonly OUTPOSTS_SERVER = new ClusterType('OUTPOSTS_SERVER');
  /** Amazon EC2 */
  public static readonly EC2 = new ClusterType('EC2');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ClusterType {
    return new ClusterType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * A mapping between a logical interface name and a network ID.
 */
export interface InterfaceMapping {
  /**
   * The logical interface name.
   */
  readonly logicalInterfaceName: string;
  /**
   * The network ID to map to.
   */
  readonly networkId: string;
}

/**
 * Network settings for a MediaLive Cluster.
 */
export interface ClusterNetworkSettings {
  /**
   * The default route for the cluster.
   * @default - no default route
   */
  readonly defaultRoute?: string;
  /**
   * The interface mappings for the cluster.
   * @default - no interface mappings
   */
  readonly interfaceMappings?: InterfaceMapping[];
}

/**
 * Represents a MediaLive Cluster.
 */
export interface ICluster extends IResource, IClusterRef {
  /**
   * The ARN of the cluster.
   * @attribute
   */
  readonly clusterArn: string;
  /**
   * The ID of the cluster.
   * @attribute
   */
  readonly clusterId: string;
  /**
   * The IDs of channels running on this cluster.
   * @attribute
   */
  readonly clusterChannelIds?: string[];
  /**
   * The current state of the cluster.
   * @attribute
   */
  readonly clusterState?: string;
}

/**
 * Properties for creating a MediaLive Cluster.
 */
export interface ClusterProps {
  /**
   * The name of the cluster.
   * @default - auto-generated name
   */
  readonly clusterName?: string;
  /**
   * The hardware type for the cluster.
   * @default ClusterType.ON_PREMISES
   */
  readonly clusterType?: ClusterType;
  /**
   * The IAM role for nodes in the cluster.
   *
   * [disable-awslint:prefer-ref-interface]
   */
  readonly instanceRole: IRole;
  /**
   * Network settings for the cluster - only required if your networking setup requires it.
   * @see https://docs.aws.amazon.com/medialive/latest/ug/emla-deploy-identify-network-requirements.html
   *
   * @default - no network settings
   */
  readonly networkSettings?: ClusterNetworkSettings;
  /**
   * Tags to add to the cluster.
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Defines an AWS Elemental MediaLive Cluster.
 *
 * A cluster represents a group of on-premises hardware nodes used by MediaLive Anywhere.
 */
@propertyInjectable
export class Cluster extends Resource implements ICluster {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.Cluster';

  /** Import an existing cluster by its ARN. The id is parsed out of the ARN. */
  public static fromClusterArn(scope: Construct, id: string, clusterArn: string): ICluster {
    const clusterId = extractResourceId(clusterArn, 'Cluster');

    class Import extends Resource implements ICluster {
      public readonly clusterArn = clusterArn;
      public readonly clusterId = clusterId;
      public readonly clusterChannelIds = undefined;
      public readonly clusterState = undefined;
      public get clusterRef(): ClusterReference {
        return {
          clusterId: this.clusterId,
          clusterArn: this.clusterArn,
        };
      }
    }
    return new Import(scope, id);
  }

  public readonly clusterArn: string;
  public readonly clusterId: string;
  public readonly clusterChannelIds?: string[];
  public readonly clusterState?: string;

  /** A reference to this Cluster resource. */
  public get clusterRef(): ClusterReference {
    return {
      clusterId: this.clusterId,
      clusterArn: this.clusterArn,
    };
  }

  constructor(scope: Construct, id: string, props: ClusterProps) {
    super(scope, id, {
      physicalName: props.clusterName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 256 }) }),
    });

    addConstructMetadata(this, props);

    const resource = new CfnCluster(this, 'Resource', {
      name: this.physicalName,
      clusterType: (props.clusterType ?? ClusterType.ON_PREMISES).value,
      instanceRoleArn: props.instanceRole.roleArn,
      networkSettings: props.networkSettings ? {
        defaultRoute: props.networkSettings.defaultRoute,
        interfaceMappings: props.networkSettings.interfaceMappings?.map(mapping => ({
          logicalInterfaceName: mapping.logicalInterfaceName,
          networkId: mapping.networkId,
        })),
      } : undefined,
      tags: props.tags ? Object.entries(props.tags).map(([key, value]) => ({ key, value })) : undefined,
    });

    this.clusterArn = resource.attrArn;
    this.clusterId = resource.ref;
    this.clusterChannelIds = resource.attrChannelIds;
    this.clusterState = resource.attrState;
  }
}
