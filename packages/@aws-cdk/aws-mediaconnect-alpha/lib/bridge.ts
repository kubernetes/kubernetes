import type { Bitrate, IResource } from 'aws-cdk-lib';
import { Duration, Resource, Lazy, Names, Token, ValidationError, UnscopedValidationError } from 'aws-cdk-lib';
import type { MetricOptions } from 'aws-cdk-lib/aws-cloudwatch';
import { Metric, Unit } from 'aws-cdk-lib/aws-cloudwatch';
import { CfnBridge } from 'aws-cdk-lib/aws-mediaconnect';
import type { IBridgeRef, BridgeReference } from 'aws-cdk-lib/aws-mediaconnect';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { IBridgeOutput } from './bridge-output';
import { BridgeOutput } from './bridge-output';
import type { IFlow } from './flow';
import type { IGateway } from './gateway';
import type { BridgeProtocol, BridgeNetworkSource, GatewayNetwork, VpcInterfaceConfig } from './shared';
import { FailoverMode, State } from './shared';

/**
 * The source of the bridge. A flow source originates in MediaConnect as an existing cloud flow.
 */
export interface BridgeFlowSource {
  /**
   * The cloud flow used as a source of this bridge.
   */
  readonly flow: IFlow;
  /**
   * The VPC interface attachment to use for this source.
   *
   * @default - no VPC interface
   */
  readonly vpcInterface?: VpcInterfaceConfig;
}

/**
 * A named flow source for an egress bridge.
 */
export interface BridgeFlowInput {
  /**
   * The name of the flow source. Must be unique among sources on the bridge.
   */
  readonly name: string;

  /**
   * The flow source configuration describing where the bridge consumes content from.
   */
  readonly source: BridgeFlowSource;
}

/**
 * A named network source for an ingress bridge.
 */
export interface BridgeNetworkInput {
  /**
   * The name of the network source. Must be unique among sources on the bridge.
   */
  readonly name: string;

  /**
   * The network source configuration describing the multicast endpoint the bridge listens to.
   */
  readonly source: BridgeNetworkSource;
}

/**
 * Configuration of bridge type
 */
export class BridgeType {
  /**
   * Ingress Bridge
   */
  public static readonly INGRESS_BRIDGE = new BridgeType('ingress_bridge');
  /**
   * Egress Bridge
   */
  public static readonly EGRESS_BRIDGE = new BridgeType('egress_bridge');

  /**
   * Use a custom bridge type value
   * @param value The bridge type string value
   */
  public static of(value: string): BridgeType {
    return new BridgeType(value);
  }

  /** @param value The bridge type string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * Base bridge configuration properties
 */
interface BridgeConfigurationBase {
  /**
   * The maximum expected bitrate (in bps) of the bridge.
   */
  readonly maxBitrate: Bitrate;
}

/**
 * Ingress bridge configuration
 */
export interface IngressBridgeConfiguration extends BridgeConfigurationBase {
  /**
   * The maximum number of outputs on the ingress bridge.
   */
  readonly maxOutputs: number;
  /**
   * The network sources for the ingress bridge.
   */
  readonly networkSources: BridgeNetworkInput[];
}

/**
 * Egress bridge configuration
 */
export interface EgressBridgeConfiguration extends BridgeConfigurationBase {
  /**
   * The flow sources for the egress bridge.
   */
  readonly flowSources: BridgeFlowInput[];
  /**
   * The network outputs for the egress bridge.
   */
  readonly networkOutputs: BridgeNetworkOutput[];
}

/**
 * Interface for Bridge
 */
export interface IBridge extends IResource, IBridgeRef {

  /**
   * The Amazon Resource Name (ARN) of the bridge.
   *
   * @attribute
   */
  readonly bridgeArn: string;

  /**
   * The name of the bridge.
   *
   * @attribute
   */
  readonly bridgeName: string;

  /**
   * The type of bridge (ingress or egress).
   *
   * @attribute
   */
  readonly bridgeType: BridgeType;

  /**
   * The current state of the bridge.
   *
   * @attribute
   */
  readonly bridgeState?: string;

  /**
   * Failover Configuration for Bridge
   */
  readonly isFailoverEnabled?: boolean;

  /**
   * Add a network output to this bridge (for egress bridges only).
   *
   * @param id Construct id for the new BridgeOutput
   * @param networkOutput The named output to add. The `name` is the output's identity on the bridge and must be unique among outputs.
   */
  addOutput(id: string, networkOutput: BridgeNetworkOutput): IBridgeOutput;

  /**
   * Create a CloudWatch metric for this bridge.
   *
   * Bridge metrics are dimensioned by `BridgeARN`. See the MediaConnect
   * documentation for available metric names (e.g. `IngressBridgeBitRate`,
   * `EgressBridgeBitRate`, `IngressBridgePacketLossPercent`).
   */
  metric(metricName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the bitrate of a specific bridge source.
   *
   * Uses `IngressBridgeSourceBitRate` for ingress bridges and
   * `EgressBridgeSourceBitRate` for egress bridges.
   *
   * @param bridgeSourceName The name of the bridge source
   * @default - average over 60 seconds
   */
  metricSourceBitrate(bridgeSourceName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the percentage of packets lost on a specific bridge source.
   *
   * Uses `IngressBridgeSourcePacketLossPercent` for ingress bridges and
   * `EgressBridgeSourcePacketLossPercent` for egress bridges.
   *
   * @param bridgeSourceName The name of the bridge source
   * @default - average over 60 seconds
   */
  metricSourcePacketLossPercent(bridgeSourceName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the total number of times the bridge switches between sources
   * when using the `FAILOVER` failover mode.
   *
   * Uses `IngressBridgeFailoverSwitches` for ingress bridges and
   * `EgressBridgeFailoverSwitches` for egress bridges.
   *
   * @default - sum over 60 seconds
   */
  metricFailoverSwitches(props?: MetricOptions): Metric;
}

/**
 * Properties for a bridge network output.
 */
export interface BridgeNetworkOutputProps {
  /**
   * The IP address where the output will send content.
   */
  readonly ipAddress: string;
  /**
   * The port to use for the output.
   */
  readonly port: number;
  /**
   * The gateway network this output sends content out of.
   *
   * Use {@link GatewayNetwork.define} to create the network and pass the same
   * instance to the gateway and to each output that uses it.
   */
  readonly network: GatewayNetwork;
  /**
   * The protocol to use for the output.
   */
  readonly protocol: BridgeProtocol;
  /**
   * Time to live (TTL) for the output packets in hops (1-255).
   *
   * TTL represents the maximum number of network hops a packet can traverse
   * before being discarded.
   */
  readonly ttl: number;
}

/**
 * A named network output for an egress bridge.
 */
export interface BridgeNetworkOutput {
  /**
   * The name of the network output. Must be unique among outputs on the bridge.
   *
   * Used as the physical name of the underlying CFN resource.
   */
  readonly name: string;

  /**
   * The network configuration describing where this output sends content.
   */
  readonly output: BridgeOutputConfiguration;
}

/**
 * Configuration for a bridge output.
 */
export class BridgeOutputConfiguration {
  /**
   * Create a network output configuration for a bridge.
   */
  public static network(props: BridgeNetworkOutputProps): BridgeOutputConfiguration {
    if (!Token.isUnresolved(props.ttl) && (props.ttl < 1 || props.ttl > 255)) {
      throw new UnscopedValidationError(lit`BridgeOutputTtlRange`, `TTL must be between 1 and 255 hops, got ${props.ttl}`);
    }
    return new BridgeOutputConfiguration(props);
  }

  private constructor(private readonly _network: BridgeNetworkOutputProps) {}

  /**
   * @internal
   */
  public _bind(): { ipAddress: string; networkName: string; port: number; protocol: string; ttl: number } {
    return {
      ipAddress: this._network.ipAddress,
      networkName: this._network.network.name,
      port: this._network.port,
      protocol: this._network.protocol.value,
      ttl: this._network.ttl,
    };
  }
}

/**
 * Options for bridge source failover.
 */
export interface BridgeFailoverOptions {
  /**
   * Whether failover is enabled. Set to `State.DISABLED` to keep the configuration
   * on the bridge without switching failover on.
   *
   * @default State.ENABLED
   */
  readonly state?: State;
  /**
   * The name of the source you want to treat as primary. If set, MediaConnect always
   * uses this source when it is available. When unset, both sources are treated with
   * equal priority.
   *
   * @default - both sources are equal priority
   */
  readonly primarySource?: string;
}

/**
 * Source failover configuration for a bridge.
 *
 * Bridges only support `FAILOVER` (switchover) mode — the `MERGE` mode available on
 * `Flow.sourceFailoverConfig` is not allowed on bridges by the service. Use
 * {@link BridgeFailoverConfig.failover} to construct the configuration.
 */
export class BridgeFailoverConfig {
  /**
   * Configure switchover-mode failover. The bridge swaps to the backup source when
   * the primary source stops receiving data.
   */
  public static failover(options: BridgeFailoverOptions = {}): BridgeFailoverConfig {
    return new BridgeFailoverConfig(options.state ?? State.ENABLED, options.primarySource);
  }

  private constructor(
    private readonly _state: State,
    private readonly _primarySource: string | undefined,
  ) {}

  /**
   * Whether this configuration has failover enabled.
   * @internal
   */
  public get _isEnabled(): boolean {
    return this._state === State.ENABLED;
  }

  /**
   * Called when the configuration is bound to a Bridge.
   * @internal
   */
  public _bind(): CfnBridge.FailoverConfigProperty {
    return {
      failoverMode: FailoverMode.FAILOVER.value,
      state: this._state,
      sourcePriority: this._primarySource ? { primarySource: this._primarySource } : undefined,
    };
  }
}

/**
 * Bridge configuration to set ingress and egress options on the bridge
 */
export class BridgeConfiguration {
  /**
   * An ingress bridge is a ground-to-cloud bridge. The content originates at your premises and is delivered to the cloud.
   */
  public static ingress(config: IngressBridgeConfiguration): BridgeConfiguration {
    if (!config.maxBitrate.isUnresolved()) {
      const bitrateBps = config.maxBitrate.toBps();
      if (bitrateBps < 1000000 || bitrateBps > 100000000) {
        throw new UnscopedValidationError(lit`BridgeIngressBitrateRange`, `Bridge ingress max bitrate must be between 1,000,000 and 100,000,000 bps, got ${bitrateBps}`);
      }
    }

    if (!Token.isUnresolved(config.networkSources) && config.networkSources.length > 2) {
      throw new UnscopedValidationError(
        lit`BridgeMaxNetworkSources`,
        `A bridge supports a maximum of 2 network sources, got ${config.networkSources.length}`,
      );
    }

    return new BridgeConfiguration(BridgeType.INGRESS_BRIDGE, config, undefined);
  }

  /**
   * An egress bridge is a cloud-to-ground bridge. The content comes from an existing MediaConnect flow and is delivered to your premises.
   */
  public static egress(config: EgressBridgeConfiguration): BridgeConfiguration {
    if (!config.maxBitrate.isUnresolved()) {
      const bitrateBps = config.maxBitrate.toBps();
      if (bitrateBps < 1000000 || bitrateBps > 100000000) {
        throw new UnscopedValidationError(lit`BridgeEgressBitrateRange`, `Bridge egress max bitrate must be between 1,000,000 and 100,000,000 bps, got ${bitrateBps}`);
      }
    }

    if (!Token.isUnresolved(config.flowSources) && config.flowSources.length > 2) {
      throw new UnscopedValidationError(
        lit`BridgeMaxFlowSources`,
        `A bridge supports a maximum of 2 flow sources, got ${config.flowSources.length}`,
      );
    }

    if (!Token.isUnresolved(config.networkOutputs) && config.networkOutputs.length > 2) {
      throw new UnscopedValidationError(
        lit`BridgeMaxNetworkOutputs`,
        `A bridge supports a maximum of 2 network outputs, got ${config.networkOutputs.length}`,
      );
    }

    return new BridgeConfiguration(BridgeType.EGRESS_BRIDGE, undefined, config);
  }

  private constructor(
    private readonly _bridgeType: BridgeType,
    private readonly _ingressConfig?: IngressBridgeConfiguration,
    private readonly _egressConfig?: EgressBridgeConfiguration,
  ) {}

  /**
   * Called when the bridge configuration is bound to a Bridge.
   * @internal
   */
  public _bind(): { bridgeType: BridgeType; ingressConfig?: IngressBridgeConfiguration; egressConfig?: EgressBridgeConfiguration } {
    return {
      bridgeType: this._bridgeType,
      ingressConfig: this._ingressConfig,
      egressConfig: this._egressConfig,
    };
  }
}

/**
 * Properties for the Bridge
 */
export interface BridgeProps {
  /**
   * The name of the bridge. Cannot be modified after the bridge is created.
   *
   * @default - autogenerated
   */
  readonly bridgeName?: string;
  /**
   * The bridge configuration specifying ingress or egress settings.
   */
  readonly config: BridgeConfiguration;
  /**
   * The gateway that the bridge is associated with.
   */
  readonly gateway: IGateway;
  /**
   * The source failover configuration for the bridge.
   *
   * @default - No failover configuration
   */
  readonly sourceFailoverConfig?: BridgeFailoverConfig;

}

/**
 * Attributes for importing an existing Bridge.
 */
export interface BridgeAttributes {
  /**
   * The ARN of the bridge.
   */
  readonly bridgeArn: string;

  /**
   * The name of the bridge.
   *
   * Not encoded in the bridge ARN, so must be provided explicitly if the imported
   * bridge needs to expose `bridgeName`.
   *
   * @default - bridgeName is not available on the imported construct
   */
  readonly bridgeName?: string;

  /**
   * Indicates what type of bridge is imported.
   *
   * Not encoded in the bridge ARN, so must be provided explicitly if the imported
   * bridge is used with `addOutput()` or other methods that need the bridge type.
   *
   * @default - bridgeType is not available on the imported construct
   */
  readonly bridgeType?: BridgeType;

  /**
   * Failover Configuration for Bridge
   *
   * @default false
   */
  readonly isFailoverEnabled?: boolean;
}

abstract class BridgeBase extends Resource implements IBridge {
  /**
   * Creates a Bridge construct that represents an external (imported) Bridge.
   */
  public static fromBridgeAttributes(scope: Construct, id: string, attrs: BridgeAttributes): IBridge {
    class Import extends BridgeBase implements IBridge {
      public readonly bridgeArn = attrs.bridgeArn;
      public readonly bridgeState = undefined;

      public get bridgeName(): string {
        if (attrs.bridgeName) return attrs.bridgeName;
        throw new ValidationError(
          lit`BridgeNameNotProvided`,
          `'bridgeName' was not provided when importing Bridge ${this.node.path}; provide it in fromBridgeAttributes()`,
          this,
        );
      }

      public get bridgeType(): BridgeType {
        if (attrs.bridgeType) return attrs.bridgeType;
        throw new ValidationError(
          lit`BridgeTypeNotProvided`,
          `'bridgeType' was not provided when importing Bridge ${this.node.path}; provide it in fromBridgeAttributes()`,
          this,
        );
      }

      // Imported bridges default to `false`. Pass `isFailoverEnabled: true` in the
      // attributes when the real service-side bridge has failover enabled and you
      // need to add a BridgeSource.
      public isFailoverEnabled?: boolean = attrs.isFailoverEnabled ?? false;
    }

    return new Import(scope, id);
  }

  /**
   * Import an existing Bridge from its ARN.
   *
   * Use `fromBridgeAttributes()` instead if you need access to `bridgeName`, `bridgeType`
   * (required for `addOutput()`), or `isFailoverEnabled`.
   */
  public static fromBridgeArn(scope: Construct, id: string, bridgeArn: string): IBridge {
    return Bridge.fromBridgeAttributes(scope, id, { bridgeArn });
  }

  public abstract readonly bridgeArn: string;
  public abstract readonly bridgeName: string;
  public abstract readonly bridgeType: BridgeType;
  public abstract readonly bridgeState?: string;
  public abstract readonly isFailoverEnabled?: boolean;

  /** A reference to this Bridge resource. */
  public get bridgeRef(): BridgeReference {
    return { bridgeArn: this.bridgeArn };
  }

  /**
   * Create a CloudWatch metric for this bridge.
   */
  public metric(metricName: string, props?: MetricOptions): Metric {
    return new Metric({
      metricName,
      namespace: 'AWS/MediaConnect',
      ...props,
      dimensionsMap: {
        BridgeARN: this.bridgeArn,
        ...props?.dimensionsMap,
      },
    }).attachTo(this);
  }

  /**
   * Metric for the bitrate of a specific bridge source.
   */
  public metricSourceBitrate(bridgeSourceName: string, props?: MetricOptions): Metric {
    const metricName = this.bridgeType.value === BridgeType.EGRESS_BRIDGE.value
      ? 'EgressBridgeSourceBitRate'
      : 'IngressBridgeSourceBitRate';
    return new Metric({
      metricName,
      namespace: 'AWS/MediaConnect',
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.BITS_PER_SECOND,
      ...props,
      dimensionsMap: {
        BridgeARN: this.bridgeArn,
        BridgeSourceName: bridgeSourceName,
        ...props?.dimensionsMap,
      },
    }).attachTo(this);
  }

  /**
   * Metric for the percentage of packets lost on a specific bridge source.
   */
  public metricSourcePacketLossPercent(bridgeSourceName: string, props?: MetricOptions): Metric {
    const metricName = this.bridgeType.value === BridgeType.EGRESS_BRIDGE.value
      ? 'EgressBridgeSourcePacketLossPercent'
      : 'IngressBridgeSourcePacketLossPercent';
    return new Metric({
      metricName,
      namespace: 'AWS/MediaConnect',
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.PERCENT,
      ...props,
      dimensionsMap: {
        BridgeARN: this.bridgeArn,
        BridgeSourceName: bridgeSourceName,
        ...props?.dimensionsMap,
      },
    }).attachTo(this);
  }

  /**
   * Metric for the total number of times the bridge switches between sources
   * when using the `FAILOVER` failover mode.
   */
  public metricFailoverSwitches(props?: MetricOptions): Metric {
    const metricName = this.bridgeType.value === BridgeType.EGRESS_BRIDGE.value
      ? 'EgressBridgeFailoverSwitches'
      : 'IngressBridgeFailoverSwitches';
    return this.metric(metricName, {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Add a network output to this bridge (for egress bridges only).
   *
   * @param id Construct id for the new BridgeOutput
   * @param networkOutput The named output to add. The `name` is the output's identity on the bridge and must be unique among outputs.
   */
  public addOutput(id: string, networkOutput: BridgeNetworkOutput): IBridgeOutput {
    if (this.bridgeType.value !== BridgeType.EGRESS_BRIDGE.value) {
      throw new ValidationError(
        lit`BridgeEgressOnly`,
        `addOutput can only be called on egress bridges, got '${this.bridgeType.value}'`,
        this,
      );
    }

    return new BridgeOutput(this, id, {
      bridgeOutputName: networkOutput.name,
      bridge: this,
      output: networkOutput.output,
    });
  }
}

/**
 * Defines a AWS Elemental MediaConnect Bridge
 */
@propertyInjectable
export class Bridge extends BridgeBase implements IBridge {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.Bridge';

  public readonly bridgeArn: string;
  public readonly bridgeName: string;
  public readonly bridgeType: BridgeType;
  public readonly bridgeState?: string;
  public readonly isFailoverEnabled?: boolean = false;

  constructor(scope: Construct, id: string, props: BridgeProps) {
    super(scope, id, {
      physicalName: props?.bridgeName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 64 }) }),
    });

    // Validate bridge name if provided
    if (props.bridgeName != null && props.bridgeName !== '' && !Token.isUnresolved(props.bridgeName)) {
      if (props.bridgeName.length > 64) {
        throw new ValidationError(lit`BridgeNameLength`, `Bridge name must be between 1 and 64 characters, got ${props.bridgeName.length}`, this);
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(props.bridgeName)) {
        throw new ValidationError(lit`BridgeNameFormat`, `Bridge name must contain only alphanumeric characters, hyphens, and underscores, got '${props.bridgeName}'`, this);
      }
    }

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.isFailoverEnabled = props.sourceFailoverConfig?._isEnabled ?? false;

    const config = props.config;
    const configBind = config._bind();
    const ingress = configBind.ingressConfig ? {
      ingressGatewayBridge: {
        maxBitrate: configBind.ingressConfig.maxBitrate.toBps(),
        maxOutputs: configBind.ingressConfig.maxOutputs,
      },
      sources: configBind.ingressConfig.networkSources.map(input => {
        return {
          networkSource: {
            name: input.name,
            multicastIp: input.source.multicastIp,
            networkName: input.source.network.name,
            port: input.source.port,
            protocol: input.source.protocol.value,
            multicastSourceSettings: {
              multicastSourceIp: input.source.multicastSourceIp,
            },
          },
        };
      }),
    } : undefined;
    const egress = configBind.egressConfig ? {
      egressGatewayBridge: {
        maxBitrate: configBind.egressConfig.maxBitrate.toBps(),
      },
      sources: configBind.egressConfig.flowSources.map(input => this.formatBridgeFlowSource(input)),
      outputs: configBind.egressConfig.networkOutputs.map(no => ({
        networkOutput: { name: no.name, ...no.output._bind() },
      })),
    } : undefined;

    const bridge = new CfnBridge(this, 'Resource', {
      name: this.physicalName,
      placementArn: props.gateway.gatewayArn,
      ingressGatewayBridge: ingress?.ingressGatewayBridge,
      egressGatewayBridge: egress?.egressGatewayBridge,
      sources: ingress?.sources ?? egress?.sources ?? [],
      outputs: egress?.outputs,
      sourceFailoverConfig: props.sourceFailoverConfig?._bind(),
    });

    this.bridgeArn = bridge.attrBridgeArn;
    this.bridgeName = this.physicalName;
    this.bridgeState = bridge.attrBridgeState;
    this.bridgeType = configBind.bridgeType;
  }

  /**
   * Format bridge flow source into BridgeSourceProperty format
   */
  private formatBridgeFlowSource(input: BridgeFlowInput): CfnBridge.BridgeSourceProperty {
    return {
      flowSource: {
        flowArn: input.source.flow.flowArn,
        name: input.name,
        flowVpcInterfaceAttachment: input.source.vpcInterface ? {
          vpcInterfaceName: input.source.vpcInterface.name,
        } : undefined,
      },
    };
  }
}
