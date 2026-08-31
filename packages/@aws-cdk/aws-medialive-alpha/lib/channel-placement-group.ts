import type { IResource } from 'aws-cdk-lib';
import { Lazy, Names, Resource } from 'aws-cdk-lib';
import type { ChannelPlacementGroupReference, IChannelPlacementGroupRef, IClusterRef } from 'aws-cdk-lib/aws-medialive';
import { CfnChannelPlacementGroup } from 'aws-cdk-lib/aws-medialive';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';

/**
 * Represents a MediaLive Channel Placement Group.
 */
export interface IChannelPlacementGroup extends IResource, IChannelPlacementGroupRef {
  /**
   * The ARN of the channel placement group.
   * @attribute
   */
  readonly channelPlacementGroupArn: string;
  /**
   * The ID of the channel placement group.
   * @attribute
   */
  readonly channelPlacementGroupId: string;
}

/**
 * Properties for creating a MediaLive Channel Placement Group.
 */
export interface ChannelPlacementGroupProps {
  /**
   * The name of the channel placement group.
   * @default - auto-generated name
   */
  readonly channelPlacementGroupName?: string;
  /**
   * The cluster this channel placement group belongs to.
   */
  readonly cluster: IClusterRef;
  /**
   * List of node IDs for the channel placement group.
   * @default - no nodes
   */
  readonly nodes?: string[];
  /**
   * Tags to add to the channel placement group.
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Attributes for importing an existing Channel Placement Group.
 */
export interface ChannelPlacementGroupAttributes {
  /** The ARN of the channel placement group. */
  readonly channelPlacementGroupArn: string;
  /** The ID of the channel placement group. */
  readonly channelPlacementGroupId: string;
  /** The ID of the cluster this group belongs to. */
  readonly clusterId: string;
}

/**
 * Defines an AWS Elemental MediaLive Channel Placement Group.
 *
 * A channel placement group assigns channels to specific nodes within a cluster.
 */
@propertyInjectable
export class ChannelPlacementGroup extends Resource implements IChannelPlacementGroup {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.ChannelPlacementGroup';

  /** Import an existing channel placement group by its attributes. */
  public static fromChannelPlacementGroupAttributes(scope: Construct, id: string, attrs: ChannelPlacementGroupAttributes): IChannelPlacementGroup {
    class Import extends Resource implements IChannelPlacementGroup {
      public readonly channelPlacementGroupArn = attrs.channelPlacementGroupArn;
      public readonly channelPlacementGroupId = attrs.channelPlacementGroupId;
      public get channelPlacementGroupRef(): ChannelPlacementGroupReference {
        return {
          channelPlacementGroupId: this.channelPlacementGroupId,
          clusterId: attrs.clusterId,
          channelPlacementGroupArn: this.channelPlacementGroupArn,
        };
      }
    }
    return new Import(scope, id);
  }

  public readonly channelPlacementGroupArn: string;
  public readonly channelPlacementGroupId: string;
  private readonly clusterId: string;

  /** A reference to this Channel Placement Group resource. */
  public get channelPlacementGroupRef(): ChannelPlacementGroupReference {
    return {
      channelPlacementGroupId: this.channelPlacementGroupId,
      clusterId: this.clusterId,
      channelPlacementGroupArn: this.channelPlacementGroupArn,
    };
  }

  constructor(scope: Construct, id: string, props: ChannelPlacementGroupProps) {
    super(scope, id, {
      physicalName: props.channelPlacementGroupName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 256 }) }),
    });

    addConstructMetadata(this, props);

    this.clusterId = props.cluster.clusterRef.clusterId;

    const resource = new CfnChannelPlacementGroup(this, 'Resource', {
      name: this.physicalName,
      clusterId: props.cluster.clusterRef.clusterId,
      nodes: props.nodes,
      tags: props.tags ? Object.entries(props.tags).map(([key, value]) => ({ key, value })) : undefined,
    });

    this.channelPlacementGroupArn = resource.attrArn;
    this.channelPlacementGroupId = resource.attrId;
  }
}
