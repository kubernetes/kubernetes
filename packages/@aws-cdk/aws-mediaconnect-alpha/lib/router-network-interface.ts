import type { IResource } from 'aws-cdk-lib';
import { Annotations, Lazy, Names, Resource, Token, ValidationError } from 'aws-cdk-lib';
import type { ISecurityGroup, ISubnet } from 'aws-cdk-lib/aws-ec2';
import { CfnRouterNetworkInterface } from 'aws-cdk-lib/aws-mediaconnect';
import type { IRouterNetworkInterfaceRef, RouterNetworkInterfaceReference } from 'aws-cdk-lib/aws-mediaconnect';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import { isOpenCidr, renderTags } from './shared';

/**
 * Interface for Router Network Interface
 */
export interface IRouterNetworkInterface extends IResource, IRouterNetworkInterfaceRef {
  /**
   * The Amazon Resource Name (ARN) of the router network interface.
   *
   * @attribute
   */
  readonly routerNetworkInterfaceArn: string;

  /**
   * The unique identifier of the router network interface.
   *
   * @attribute
   */
  readonly routerNetworkInterfaceId: string;

  /**
   * The date and time the router network interface was created.
   *
   * @attribute
   */
  readonly createdAt?: string;

  /**
   * The date and time the router network interface was last updated.
   *
   * @attribute
   */
  readonly updatedAt?: string;
}

/**
 * Properties for Router Network Interface
 */
export interface RouterNetworkInterfaceProps {
  /**
   * The name of the router network interface.
   * @default - Generated automatically
   */
  readonly routerNetworkInterfaceName?: string;

  /**
   * Network configuration for the router network interface.
   */
  readonly configuration: RouterNetworkConfiguration;

  /**
   * Select a region for your router network interface.
   *
   * @default - Same region as the stack
   */
  readonly regionName?: string;

  /**
   * Tags to add to the network interface
   *
   * @default - No tagging
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Attributes for importing an existing Router Network Interface.
 */
export interface RouterNetworkInterfaceAttributes {
  /**
   * The Amazon Resource Name (ARN) of the router network interface.
   */
  readonly routerNetworkInterfaceArn: string;

  /**
   * The unique identifier of the router network interface.
   *
   * @default - accessing `routerNetworkInterfaceId` on the imported interface throws; only provide when available.
   */
  readonly routerNetworkInterfaceId?: string;
}

/**
 * Properties for public network configuration
 */
export interface PublicNetworkConfigurationProps {
  /**
   * CIDR blocks allowed to send inbound traffic to the network interface.
   *
   * @default - no inbound allowed (outbound only)
   */
  readonly cidr?: string[];
}

/**
 * Properties for VPC network configuration
 */
export interface VpcNetworkConfigurationProps {
  /** Security groups to associate with the network interface */
  readonly securityGroups: ISecurityGroup[];
  /** Subnet where the network interface will be created */
  readonly subnet: ISubnet;
}

/**
 * Factory class for creating Router Network configurations
 */
export class RouterNetworkConfiguration {
  /**
   * Create a public network configuration
   * @param props Public network configuration properties
   * @returns RouterNetworkConfiguration instance for public setup
   */
  public static publicNetwork(props: PublicNetworkConfigurationProps = {}): RouterNetworkConfiguration {
    const allowRules = (props.cidr ?? []).map(ip => ({ cidr: ip }));
    return new RouterNetworkConfiguration({
      public: { allowRules },
    }, props.cidr);
  }

  /**
   * Create a VPC network configuration
   * @param props VPC network configuration properties
   * @returns RouterNetworkConfiguration instance for VPC setup
   */
  public static vpc(props: VpcNetworkConfigurationProps): RouterNetworkConfiguration {
    return new RouterNetworkConfiguration({
      vpc: {
        securityGroupIds: props.securityGroups.map(sg => { return sg.securityGroupId; }),
        subnetId: props.subnet.subnetId,
      },
    });
  }

  private readonly _config: CfnRouterNetworkInterface.RouterNetworkInterfaceConfigurationProperty;
  private readonly _publicCidrs?: string[];

  private constructor(
    config: CfnRouterNetworkInterface.RouterNetworkInterfaceConfigurationProperty,
    publicCidrs?: string[],
  ) {
    this._config = config;
    this._publicCidrs = publicCidrs;
  }

  /**
   * Called when the network configuration is bound to a RouterNetworkInterface.
   * @internal
   */
  public _bind(scope: Construct): CfnRouterNetworkInterface.RouterNetworkInterfaceConfigurationProperty {
    if (this._publicCidrs !== undefined && !Token.isUnresolved(this._publicCidrs)) {
      for (const cidr of this._publicCidrs) {
        if (!Token.isUnresolved(cidr) && isOpenCidr(cidr)) {
          Annotations.of(scope).addWarningV2(
            '@aws-cdk/aws-mediaconnect-alpha:openRouterNetworkCidr',
            `Router network interface public CIDR '${cidr}' allows traffic from any IP. Restrict to the narrowest range your senders/receivers need.`,
          );
        }
      }
    }
    return { ...this._config };
  }
}

/**
 * Shared base for both real and imported router network interfaces.
 */
abstract class RouterNetworkInterfaceBase extends Resource implements IRouterNetworkInterface {
  public abstract readonly routerNetworkInterfaceArn: string;
  public abstract readonly routerNetworkInterfaceId: string;
  public abstract readonly createdAt?: string;
  public abstract readonly updatedAt?: string;

  public get routerNetworkInterfaceRef(): RouterNetworkInterfaceReference {
    return { routerNetworkInterfaceArn: this.routerNetworkInterfaceArn };
  }
}

/**
 * Defines a AWS Elemental MediaConnect Router Network Interface
 */
@propertyInjectable
export class RouterNetworkInterface extends RouterNetworkInterfaceBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.RouterNetworkInterface';

  /**
   * Import an existing Router Network Interface from its ARN.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param routerNetworkInterfaceArn The ARN of the Router Network Interface
   * @returns A Router Network Interface construct
   */
  public static fromRouterNetworkInterfaceArn(scope: Construct, id: string, routerNetworkInterfaceArn: string): IRouterNetworkInterface {
    return RouterNetworkInterface.fromRouterNetworkInterfaceAttributes(scope, id, { routerNetworkInterfaceArn });
  }

  /**
   * Import an existing Router Network Interface from its attributes.
   *
   * Provide `routerNetworkInterfaceId` when importing an interface that was deployed
   * externally — otherwise accessing it on the imported construct will throw.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param attrs The Router Network Interface attributes
   * @returns A Router Network Interface construct
   */
  public static fromRouterNetworkInterfaceAttributes(
    scope: Construct,
    id: string,
    attrs: RouterNetworkInterfaceAttributes,
  ): IRouterNetworkInterface {
    class Import extends RouterNetworkInterfaceBase {
      public readonly routerNetworkInterfaceArn = attrs.routerNetworkInterfaceArn;
      public readonly createdAt = undefined;
      public readonly updatedAt = undefined;

      public get routerNetworkInterfaceId(): string {
        if (attrs.routerNetworkInterfaceId) return attrs.routerNetworkInterfaceId;
        throw new ValidationError(
          lit`RouterNetworkInterfaceIdNotProvided`,
          `'routerNetworkInterfaceId' is not available on imported RouterNetworkInterface ${this.node.path}; pass it via fromRouterNetworkInterfaceAttributes`,
          this,
        );
      }
    }
    return new Import(scope, id);
  }

  public readonly routerNetworkInterfaceArn: string;
  public readonly routerNetworkInterfaceId: string;
  /**
   * The date and time the router network interface was created.
   * @attribute
   */
  public readonly createdAt?: string;
  /**
   * The date and time the router network interface was last updated.
   * @attribute
   */
  public readonly updatedAt?: string;

  constructor(scope: Construct, id: string, props: RouterNetworkInterfaceProps) {
    super(scope, id, {
      physicalName: props.routerNetworkInterfaceName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 128 }) }),
    });

    // Validate router network interface name if provided
    if (props.routerNetworkInterfaceName != null && props.routerNetworkInterfaceName !== '' && !Token.isUnresolved(props.routerNetworkInterfaceName)) {
      if (props.routerNetworkInterfaceName.length > 128) {
        throw new ValidationError(lit`RouterNetworkInterfaceNameLength`, `Router network interface name must be between 1 and 128 characters, got ${props.routerNetworkInterfaceName.length}`, this);
      }
      if (!/^[a-zA-Z0-9-]+$/.test(props.routerNetworkInterfaceName)) {
        throw new ValidationError(lit`RouterNetworkInterfaceNameFormat`, `Router network interface name must contain only alphanumeric characters and hyphens, got '${props.routerNetworkInterfaceName}'`, this);
      }
    }

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const routerNetworkInterface = new CfnRouterNetworkInterface(this, 'Resource', {
      name: this.physicalName,
      configuration: props.configuration._bind(this),
      regionName: props.regionName,
      tags: props.tags ? renderTags(props.tags) : undefined,
    });

    this.routerNetworkInterfaceArn = routerNetworkInterface.attrArn;
    this.routerNetworkInterfaceId = routerNetworkInterface.attrId;
    this.createdAt = routerNetworkInterface.attrCreatedAt;
    this.updatedAt = routerNetworkInterface.attrUpdatedAt;
  }
}
