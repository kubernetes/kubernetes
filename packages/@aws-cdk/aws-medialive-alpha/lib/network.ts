import type { IResource } from 'aws-cdk-lib';
import { Lazy, Names, Resource, ValidationError } from 'aws-cdk-lib';
import type { INetworkRef, NetworkReference } from 'aws-cdk-lib/aws-medialive';
import { CfnNetwork } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';

/**
 * A route for a MediaLive Network.
 */
export interface NetworkRoute {
  /**
   * The CIDR block for the route.
   */
  readonly cidr: string;
  /**
   * The gateway for the route.
   */
  readonly gateway: string;
}

/**
 * Represents a MediaLive Network.
 */
export interface INetwork extends IResource, INetworkRef {
  /**
   * The ARN of the network.
   * @attribute
   */
  readonly networkArn: string;
  /**
   * The ID of the network.
   * @attribute
   */
  readonly networkId: string;
}

/**
 * Properties for creating a MediaLive Network.
 */
export interface NetworkProps {
  /**
   * The name of the network.
   * @default - auto-generated name
   */
  readonly networkName?: string;
  /**
   * The list of IP address CIDR pools for the network.
   */
  readonly ipPools: string[];
  /**
   * The routes for the network.
   * @default - no routes
   */
  readonly routes?: NetworkRoute[];
  /**
   * Tags to add to the network.
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Attributes for importing an existing Network.
 */
export interface NetworkAttributes {
  /** The ARN of the network. */
  readonly networkArn: string;
  /** The ID of the network. */
  readonly networkId: string;
}

/**
 * Defines an AWS Elemental MediaLive Network.
 *
 * A network represents a collection of IP address pools and routes used by MediaLive Anywhere resources.
 */
@propertyInjectable
export class Network extends Resource implements INetwork {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.Network';

  /** Import an existing network by its attributes. */
  public static fromNetworkAttributes(scope: Construct, id: string, attrs: NetworkAttributes): INetwork {
    class Import extends Resource implements INetwork {
      public readonly networkArn = attrs.networkArn;
      public readonly networkId = attrs.networkId;
      public get networkRef(): NetworkReference {
        return {
          networkId: this.networkId,
          networkArn: this.networkArn,
        };
      }
    }
    return new Import(scope, id);
  }

  public readonly networkArn: string;
  public readonly networkId: string;

  /** A reference to this Network resource. */
  public get networkRef(): NetworkReference {
    return {
      networkId: this.networkId,
      networkArn: this.networkArn,
    };
  }

  constructor(scope: Construct, id: string, props: NetworkProps) {
    super(scope, id, {
      physicalName: props.networkName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 256 }) }),
    });

    if (props.ipPools.length === 0) {
      throw new ValidationError(lit`NetworkIpPoolsEmpty`, 'ipPools must contain at least one CIDR block', this);
    }

    addConstructMetadata(this, props);

    const resource = new CfnNetwork(this, 'Resource', {
      name: this.physicalName,
      ipPools: props.ipPools.map(cidr => ({ cidr })),
      routes: props.routes?.map(route => ({ cidr: route.cidr, gateway: route.gateway })),
      tags: props.tags ? Object.entries(props.tags).map(([key, value]) => ({ key, value })) : undefined,
    });

    this.networkArn = resource.attrArn;
    this.networkId = resource.ref;
  }
}
