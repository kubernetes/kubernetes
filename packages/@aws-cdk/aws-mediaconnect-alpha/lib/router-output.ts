import type { Bitrate, IResource } from 'aws-cdk-lib';
import { Arn, ArnFormat, Duration, Lazy, Names, Resource, Stack, Token, UnscopedValidationError, ValidationError } from 'aws-cdk-lib';
import type { MetricOptions } from 'aws-cdk-lib/aws-cloudwatch';
import { Metric, Unit } from 'aws-cdk-lib/aws-cloudwatch';
import type { Grant } from 'aws-cdk-lib/aws-iam';
import { CfnRouterOutput } from 'aws-cdk-lib/aws-mediaconnect';
import type { IRouterOutputRef, RouterOutputReference } from 'aws-cdk-lib/aws-mediaconnect';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { IFlow } from './flow';
import type { MaintenanceConfiguration, ForwardErrorCorrection, RoutingScope } from './router-input';
import type { IRouterNetworkInterface } from './router-network-interface';
import type { MediaLivePipeline, RouterSrtEncryption, TransitEncryption } from './shared';
import { renderTags, exceedsRouterTierBitrate, resolveRouterSrtEncryption, resolveTransitEncryption, validateMaintenanceTime } from './shared';

/**
 * Protocol options available for Router Output configurations
 */
export class RouterOutputProtocolOptions {
  /** Real-time Transport Protocol */
  public static readonly RTP = new RouterOutputProtocolOptions('RTP');
  /** Reliable Internet Stream Transport */
  public static readonly RIST = new RouterOutputProtocolOptions('RIST');
  /** Secure Reliable Transport - Caller mode */
  public static readonly SRT_CALLER = new RouterOutputProtocolOptions('SRT_CALLER');
  /** Secure Reliable Transport - Listener mode */
  public static readonly SRT_LISTENER = new RouterOutputProtocolOptions('SRT_LISTENER');

  /**
   * Use a custom protocol value
   * @param value The protocol string value
   */
  public static of(value: string): RouterOutputProtocolOptions {
    return new RouterOutputProtocolOptions(value);
  }

  /** @param value The protocol string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * Routing tier that determines the maximum bitrate (in Mbps) for a Router Output.
 */
export class RouterOutputTier {
  /** Up to 100 Mbps */
  public static readonly OUTPUT_100 = new RouterOutputTier('OUTPUT_100');
  /** Up to 50 Mbps */
  public static readonly OUTPUT_50 = new RouterOutputTier('OUTPUT_50');
  /** Up to 20 Mbps */
  public static readonly OUTPUT_20 = new RouterOutputTier('OUTPUT_20');

  /**
   * Use a custom tier value
   * @param value The tier string value
   */
  public static of(value: string): RouterOutputTier {
    return new RouterOutputTier(value);
  }

  /** @param value The tier string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * Interface for Router Output
 */
export interface IRouterOutput extends IResource, IRouterOutputRef {
  /**
   * The name of the router output.
   *
   * @attribute
   */
  readonly routerOutputName: string;

  /**
   * The Amazon Resource Name (ARN) of the router output.
   *
   * @attribute
   */
  readonly routerOutputArn: string;

  /**
   * The unique identifier of the router output.
   *
   * @attribute
   */
  readonly routerOutputId: string;

  /**
   * The IP address of the router output.
   *
   * @attribute
   */
  readonly ipAddress: string;

  /**
   * The timestamp when the router output was created.
   *
   * @attribute
   */
  readonly createdAt?: string;

  /**
   * The timestamp when the router output was last updated.
   *
   * @attribute
   */
  readonly updatedAt?: string;

  /**
   * Create a CloudWatch metric for this router output.
   *
   * Router output metrics are dimensioned by `RouterOutputARN`. See the MediaConnect
   * documentation for available metric names (e.g. `RouterOutputBitRate`,
   * `RouterOutputTotalPackets`).
   */
  metric(metricName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the bitrate of the router output's payload.
   * @default - average over 60 seconds
   */
  metricBitrate(props?: MetricOptions): Metric;

  /**
   * Metric for the total number of packets sent by the router output.
   * @default - sum over 60 seconds
   */
  metricTotalPackets(props?: MetricOptions): Metric;

  /**
   * Metric for the router output connection state (1 connected, 0 disconnected). Applies to SRT outputs only.
   * @default - minimum over 60 seconds
   */
  metricConnected(props?: MetricOptions): Metric;

  /**
   * Metric for the number of retransmitted packets requested through automatic repeat request (ARQ).
   * Applies to RIST and SRT outputs only.
   *
   * @default - sum over 60 seconds
   */
  metricArqRequests(props?: MetricOptions): Metric;
}

/**
 * Properties for creating a Router Output
 */
export interface RouterOutputProps {
  /**
   * Name of the Router Output
   * @default - Generated automatically
   */
  readonly routerOutputName?: string;
  /** The maximum bitrate for the router output. */
  readonly maximumBitrate: Bitrate;
  /** Indicates whether the router output is configured for Regional or global routing. */
  readonly routingScope: RoutingScope;
  /**
   * Select a tier based on your maximum bitrate requirements.
   * @default RouterOutputTier.OUTPUT_20
   */
  readonly tier?: RouterOutputTier;
  /** Configuration for the Router Output (standard, MediaConnect Flow, or MediaLive) */
  readonly configuration: RouterOutputConfiguration;
  /**
   * Maintenance window configuration
   * @default - Default maintenance window will be used
   */
  readonly maintenanceConfiguration?: MaintenanceConfiguration;
  /**
   * The AWS Region where the router output is located.
   * @default - Defaults to the same region as stack
   */
  readonly regionName?: string;
  /**
   * Tags to add to the Router Output
   * @default - No tagging
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Attributes for importing an existing Router Output.
 */
export interface RouterOutputAttributes {
  /**
   * The ARN of the router output.
   */
  readonly routerOutputArn: string;

  /**
   * The name of the router output.
   *
   * The name is not encoded in the ARN (the ARN contains only the service-generated ID),
   * so provide it here if you need to access `routerOutputName` on the imported construct.
   *
   * @default - routerOutputName is not available on the imported construct
   */
  readonly routerOutputName?: string;

  /**
   * The unique identifier of the router output.
   *
   * @default - parsed from the ARN when available
   */
  readonly routerOutputId?: string;
}

/**
 * Properties for RTP protocol configuration for outputs
 */
export interface RtpOutputProtocolProps {
  /** Destination IP address for RTP traffic */
  readonly destinationAddress: string;
  /** Port number for RTP traffic */
  readonly port: number;
  /**
   * Forward Error Correction setting
   * @default - undefined; when omitted, MediaConnect applies ForwardErrorCorrection.DISABLED at deploy time
   */
  readonly forwardErrorCorrection?: ForwardErrorCorrection;
}

/**
 * Properties for RIST protocol configuration for outputs
 */
export interface RistOutputProtocolProps {
  /** Destination IP address for RIST traffic */
  readonly destinationAddress: string;
  /** Port number for RIST traffic */
  readonly port: number;
}

/**
 * Properties for SRT Listener protocol configuration for outputs
 */
export interface SrtListenerOutputProtocolProps {
  /** Port number for SRT listener */
  readonly port: number;
  /** Minimum latency for SRT */
  readonly minimumLatency: Duration;
  /**
   * Optional encryption configuration for SRT streams
   * @default - No encryption
   */
  readonly encryptionConfiguration?: RouterSrtEncryption;
}

/**
 * Properties for SRT Caller protocol configuration for outputs
 */
export interface SrtCallerOutputProtocolProps {
  /** Destination IP address to connect to */
  readonly destinationAddress: string;
  /** Destination port to connect to */
  readonly destinationPort: number;
  /** Minimum latency for SRT */
  readonly minimumLatency: Duration;
  /**
   * Optional stream ID for SRT connection
   * @default - No stream ID
   */
  readonly streamId?: string;
  /**
   * Optional encryption configuration for SRT streams
   * @default - No encryption
   */
  readonly encryptionConfiguration?: RouterSrtEncryption;
}

/**
 * Factory class for creating Router Output protocol configurations.
 *
 * Supported protocols: RTP, RIST, SRT (Listener and Caller).
 */
export class RouterOutputProtocol {
  /**
   * Create an RTP protocol configuration
   *
   * @example
   *
   *    RouterOutputProtocol.rtp({
   *      destinationAddress: '10.0.0.1',
   *      port: 5000,
   *    });
   *
   * @param props RTP protocol properties
   * @returns RouterOutputProtocol instance configured for RTP
   */
  public static rtp(props: RtpOutputProtocolProps): RouterOutputProtocol {
    RouterOutputProtocol.validatePort(props.port);
    return new RouterOutputProtocol(RouterOutputProtocolOptions.RTP, {
      rtp: {
        destinationAddress: props.destinationAddress,
        destinationPort: props.port,
        forwardErrorCorrection: props.forwardErrorCorrection,
      },
    });
  }

  /**
   * Create a RIST protocol configuration
   *
   * @example
   *
   *    RouterOutputProtocol.rist({
   *      destinationAddress: '10.0.0.1',
   *      port: 5000,
   *    });
   *
   * @param props RIST protocol properties
   * @returns RouterOutputProtocol instance configured for RIST
   */
  public static rist(props: RistOutputProtocolProps): RouterOutputProtocol {
    RouterOutputProtocol.validatePort(props.port);
    return new RouterOutputProtocol(RouterOutputProtocolOptions.RIST, {
      rist: {
        destinationAddress: props.destinationAddress,
        destinationPort: props.port,
      },
    });
  }

  /**
   * Create an SRT Listener protocol configuration
   *
   * @example
   *
   *    RouterOutputProtocol.srtListener({
   *      port: 5000,
   *      minimumLatency: Duration.millis(125),
   *    });
   *
   * @param props SRT Listener protocol properties
   * @returns RouterOutputProtocol instance configured for SRT Listener
   */
  public static srtListener(props: SrtListenerOutputProtocolProps): RouterOutputProtocol {
    RouterOutputProtocol.validatePort(props.port);
    return new RouterOutputProtocol(RouterOutputProtocolOptions.SRT_LISTENER, {
      srtListener: {
        port: props.port,
        minimumLatencyMilliseconds: props.minimumLatency.toMilliseconds(),
      },
    }, props.encryptionConfiguration);
  }

  /**
   * Create an SRT Caller protocol configuration
   *
   * @example
   *
   *    RouterOutputProtocol.srtCaller({
   *      destinationAddress: '10.0.0.1',
   *      destinationPort: 5000,
   *      minimumLatency: Duration.millis(125),
   *    });
   *
   * @param props SRT Caller protocol properties
   * @returns RouterOutputProtocol instance configured for SRT Caller
   */
  public static srtCaller(props: SrtCallerOutputProtocolProps): RouterOutputProtocol {
    RouterOutputProtocol.validateSrtCallerPort(props.destinationPort);
    return new RouterOutputProtocol(RouterOutputProtocolOptions.SRT_CALLER, {
      srtCaller: {
        destinationAddress: props.destinationAddress,
        destinationPort: props.destinationPort,
        minimumLatencyMilliseconds: props.minimumLatency.toMilliseconds(),
        streamId: props.streamId,
      },
    }, props.encryptionConfiguration);
  }

  /**
   * Validate that port is within the valid range (3000-30000)
   */
  private static validatePort(port: number): void {
    if (!Token.isUnresolved(port) && (port < 3000 || port > 30000)) {
      throw new UnscopedValidationError(lit`RouterOutputPortRange`, `Port must be between 3000 and 30000, got ${port}`);
    }
  }

  /**
   * Validate that SRT Caller port is within the valid range (0-65535)
   */
  private static validateSrtCallerPort(port: number): void {
    if (!Token.isUnresolved(port) && (port < 0 || port > 65535)) {
      throw new UnscopedValidationError(lit`SrtCallerPortRange`, `SRT Caller port must be between 0 and 65535, got ${port}`);
    }
  }

  private readonly _protocolName: RouterOutputProtocolOptions;
  private readonly _config: CfnRouterOutput.RouterOutputProtocolConfigurationProperty;
  private readonly _encryption?: RouterSrtEncryption;

  private constructor(
    protocolName: RouterOutputProtocolOptions,
    config: CfnRouterOutput.RouterOutputProtocolConfigurationProperty,
    encryption?: RouterSrtEncryption,
  ) {
    this._protocolName = protocolName;
    this._config = config;
    this._encryption = encryption;
  }

  /**
   * Resolve this protocol to its CFN-ready form. Returns the protocol identifier
   * and the CFN configuration shape with SRT encryption auto-role materialized if needed.
   *
   * @param scope             Construct that scopes the auto-created encryption role
   * @param routerOutputArn   Optional ARN of the consuming router output — when provided,
   *                          scopes the auto-created encryption role's trust policy.
   *
   * @internal
   */
  public _bind(scope: Construct, routerOutputArn?: string): {
    name: RouterOutputProtocolOptions;
    config: CfnRouterOutput.RouterOutputProtocolConfigurationProperty;
    grant?: Grant;
  } {
    if (!this._encryption) {
      return { name: this._protocolName, config: this._config };
    }

    const { config: encryptionConfiguration, grant } = resolveRouterSrtEncryption(scope, 'EncryptionRole', this._encryption, routerOutputArn);

    if (this._config.srtListener) {
      return {
        name: this._protocolName,
        config: { srtListener: { ...this._config.srtListener, encryptionConfiguration } },
        grant,
      };
    }
    if (this._config.srtCaller) {
      return {
        name: this._protocolName,
        config: { srtCaller: { ...this._config.srtCaller, encryptionConfiguration } },
        grant,
      };
    }
    return { name: this._protocolName, config: this._config };
  }
}

/**
 * Properties for standard Router Output configuration
 */
export interface StandardOutputConfigurationProps {
  /** Network interface for the Router Output */
  readonly networkInterface: IRouterNetworkInterface;
  /** Protocol configuration for the output */
  readonly protocol: RouterOutputProtocol;
  /**
   * The availability zone where the router output is located.
   *
   * @default - assigned by the MediaConnect service
   */
  readonly availabilityZone?: string;
}

/**
 * Properties for MediaLive Router Output configuration with specific input connection
 */
export interface MediaLiveInputConnectionProps {
  /**
   * ARN of the MediaLive input to send output to.
   *
   * Note: This will change to accept an IInputRef (typed MediaLive Input reference)
   * when the @aws-cdk/aws-medialive-alpha L2 construct is released.
   */
  readonly mediaLiveInputArn: string;
  /** Pipeline ID for MediaLive input */
  readonly mediaLivePipelineId: MediaLivePipeline;
  /**
   * Optional transit encryption configuration
   * @default - Automatic encryption will be used
   */
  readonly destinationTransitEncryption?: TransitEncryption;
}

/**
 * Properties for MediaLive Router Output configuration without specific input connection
 */
export interface MediaLiveNoInputConnectionProps {
  /** Availability zone for the router output when not connecting to a specific input */
  readonly availabilityZone: string;
  /**
   * Optional transit encryption configuration
   * @default - Automatic encryption will be used
   */
  readonly destinationTransitEncryption?: TransitEncryption;
}

/**
 * Properties for MediaConnect Flow Router Output configuration with specific flow connection
 */
export interface MediaConnectFlowConnectionProps {
  /** MediaConnect Flow to send output to */
  readonly flow: IFlow;
  /**
   * Optional transit encryption configuration
   * @default - Automatic encryption will be used
   */
  readonly destinationTransitEncryption?: TransitEncryption;
}

/**
 * Properties for MediaConnect Flow Router Output configuration without specific flow connection
 */
export interface MediaConnectFlowNoConnectionProps {
  /** Availability zone for the router output when not connecting to a specific flow */
  readonly availabilityZone: string;
  /**
   * Optional transit encryption configuration
   * @default - Automatic encryption will be used
   */
  readonly destinationTransitEncryption?: TransitEncryption;
}

/**
 * Factory class for creating Router Output configurations
 */
export abstract class RouterOutputConfiguration {
  /**
   * Create a standard Router Output configuration with a single protocol
   *
   * @example
   *
   *    declare const networkInterface: RouterNetworkInterface;
   *
   *    RouterOutputConfiguration.standard({
   *      networkInterface,
   *      protocol: RouterOutputProtocol.rtp({
   *        destinationAddress: '10.0.0.1',
   *        port: 5000,
   *      }),
   *    });
   *
   * @param props Standard configuration properties
   * @returns RouterOutputConfiguration instance for standard setup
   */
  public static standard(props: StandardOutputConfigurationProps): RouterOutputConfiguration {
    return new StandardRouterOutputConfig(props);
  }

  /**
   * Create a MediaLive Router Output configuration with a specific input connection.
   *
   * Use this when the MediaLive input already exists and you want to connect immediately.
   *
   * @example
   *
   *    RouterOutputConfiguration.mediaLiveInput({
   *      mediaLiveInputArn: 'arn:aws:medialive:us-east-1:123456789012:input:1234567',
   *      mediaLivePipelineId: MediaLivePipeline.PIPELINE_0,
   *    });
   *
   * @param props MediaLive input connection properties
   * @returns RouterOutputConfiguration instance for MediaLive setup with input connection
   */
  public static mediaLiveInput(props: MediaLiveInputConnectionProps): RouterOutputConfiguration {
    return new MediaLiveInputRouterOutputConfig({
      mediaLiveInputArn: props.mediaLiveInputArn,
      mediaLivePipelineId: props.mediaLivePipelineId,
      destinationTransitEncryption: props.destinationTransitEncryption,
    });
  }

  /**
   * Create a MediaLive Router Output configuration without a specific input connection.
   *
   * Use this when you want to set up the router output before the MediaLive input exists.
   *
   * @example
   *
   *    RouterOutputConfiguration.mediaLiveInputWithoutConnection({
   *      availabilityZone: 'us-east-1a',
   *    });
   *
   * @param props MediaLive no input connection properties
   * @returns RouterOutputConfiguration instance for MediaLive setup without input connection
   */
  public static mediaLiveInputWithoutConnection(props: MediaLiveNoInputConnectionProps): RouterOutputConfiguration {
    return new MediaLiveInputRouterOutputConfig(
      { destinationTransitEncryption: props.destinationTransitEncryption },
      props.availabilityZone,
    );
  }

  /**
   * Create a MediaConnect Flow Router Output configuration with a specific flow connection.
   *
   * Use this when the target flow already exists and you want to connect immediately.
   *
   * @example
   *
   *    declare const flow: Flow;
   *
   *    RouterOutputConfiguration.mediaConnectFlow({
   *      flow,
   *    });
   *
   * @param props MediaConnect Flow connection properties
   * @returns RouterOutputConfiguration instance for MediaConnect Flow setup with flow connection
   */
  public static mediaConnectFlow(props: MediaConnectFlowConnectionProps): RouterOutputConfiguration {
    return new MediaConnectFlowRouterOutputConfig({
      flow: props.flow,
      destinationTransitEncryption: props.destinationTransitEncryption,
    });
  }

  /**
   * Create a MediaConnect Flow Router Output configuration without a specific flow connection.
   *
   * Use this when you want to set up the router output before the target flow exists.
   *
   * @example
   *
   *    RouterOutputConfiguration.mediaConnectFlowWithoutConnection({
   *      availabilityZone: 'us-east-1a',
   *    });
   *
   * @param props MediaConnect Flow no connection properties
   * @returns RouterOutputConfiguration instance for MediaConnect Flow setup without flow connection
   */
  public static mediaConnectFlowWithoutConnection(props: MediaConnectFlowNoConnectionProps): RouterOutputConfiguration {
    return new MediaConnectFlowRouterOutputConfig(
      { destinationTransitEncryption: props.destinationTransitEncryption },
      props.availabilityZone,
    );
  }

  /**
   * Resolve this configuration to its CloudFormation shape. Called by the RouterOutput
   * constructor — each concrete subclass supplies its own variant-specific rendering.
   * @internal
   */
  public abstract _bind(scope: Construct, routerOutputArn?: string): {
    config: CfnRouterOutput.RouterOutputConfigurationProperty; availabilityZone?: string; grant?: Grant;
  };
}

/**
 * Concrete variant: standard (single-protocol) router output configuration.
 * @internal
 */
class StandardRouterOutputConfig extends RouterOutputConfiguration {
  constructor(private readonly props: StandardOutputConfigurationProps) { super(); }

  public _bind(scope: Construct, routerOutputArn?: string) {
    const protocol = this.props.protocol._bind(scope, routerOutputArn);
    return {
      config: {
        standard: {
          networkInterfaceArn: this.props.networkInterface.routerNetworkInterfaceArn,
          protocol: protocol.name.value,
          protocolConfiguration: protocol.config,
        },
      },
      availabilityZone: this.props.availabilityZone,
      grant: protocol.grant,
    };
  }
}

/**
 * Concrete variant: MediaLive Input router output configuration (with or without connection).
 * @internal
 */
class MediaLiveInputRouterOutputConfig extends RouterOutputConfiguration {
  constructor(
    private readonly options: MediaLiveInputRouterOutputOptions,
    private readonly availabilityZone?: string,
  ) { super(); }

  public _bind(scope: Construct, routerOutputArn?: string) {
    const transit = resolveTransitEncryption(scope, 'DestinationTransitEncryptionRole', this.options.destinationTransitEncryption, routerOutputArn);
    return {
      config: {
        mediaLiveInput: {
          mediaLiveInputArn: this.options.mediaLiveInputArn,
          mediaLivePipelineId: this.options.mediaLivePipelineId,
          destinationTransitEncryption: transit.config,
        },
      },
      availabilityZone: this.availabilityZone,
      grant: transit.grant,
    };
  }
}

/**
 * Internal options for {@link MediaLiveInputRouterOutputConfig}.
 */
interface MediaLiveInputRouterOutputOptions {
  readonly mediaLiveInputArn?: string;
  readonly mediaLivePipelineId?: MediaLivePipeline;
  readonly destinationTransitEncryption?: TransitEncryption;
}

/**
 * Concrete variant: MediaConnect Flow router output configuration (with or without connection).
 * @internal
 */
class MediaConnectFlowRouterOutputConfig extends RouterOutputConfiguration {
  constructor(
    private readonly options: MediaConnectFlowRouterOutputOptions,
    private readonly availabilityZone?: string,
  ) { super(); }

  public _bind(scope: Construct, routerOutputArn?: string) {
    const transit = resolveTransitEncryption(scope, 'DestinationTransitEncryptionRole', this.options.destinationTransitEncryption, routerOutputArn);
    return {
      config: {
        mediaConnectFlow: {
          flowArn: this.options.flow?.flowArn,
          flowSourceArn: this.options.flow?.sourceArn,
          destinationTransitEncryption: transit.config,
        },
      },
      availabilityZone: this.availabilityZone,
      grant: transit.grant,
    };
  }
}

/**
 * Internal options for {@link MediaConnectFlowRouterOutputConfig}.
 */
interface MediaConnectFlowRouterOutputOptions {
  readonly flow?: IFlow;
  readonly destinationTransitEncryption?: TransitEncryption;
}

/**
 * A MediaConnect Router Output that sends media streams to destinations.
 *
 * Router Outputs support multiple configuration types:
 * - Standard: Single protocol output
 * - MediaLive Input: Output to AWS Elemental MediaLive
 * - MediaConnect Flow: Output to another MediaConnect Flow
 */
abstract class RouterOutputBase extends Resource implements IRouterOutput {
  public abstract readonly routerOutputArn: string;
  public abstract readonly routerOutputName: string;
  public abstract readonly routerOutputId: string;
  public abstract readonly ipAddress: string;
  public abstract readonly createdAt?: string;
  public abstract readonly updatedAt?: string;

  /**
   * A reference to this Router Output resource.
   * Required by the auto-generated IRouterOutputRef interface.
   */
  public get routerOutputRef(): RouterOutputReference {
    return { routerOutputArn: this.routerOutputArn };
  }

  /**
   * Create a CloudWatch metric for this router output.
   */
  public metric(metricName: string, props?: MetricOptions): Metric {
    return new Metric({
      metricName,
      namespace: 'AWS/MediaConnect',
      ...props,
      dimensionsMap: {
        RouterOutputARN: this.routerOutputArn,
        ...props?.dimensionsMap,
      },
    }).attachTo(this);
  }

  /**
   * Metric for the bitrate of the router output's payload.
   */
  public metricBitrate(props?: MetricOptions): Metric {
    return this.metric('RouterOutputBitRate', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.BITS_PER_SECOND,
      ...props,
    });
  }

  /**
   * Metric for the total number of packets sent by the router output.
   */
  public metricTotalPackets(props?: MetricOptions): Metric {
    return this.metric('RouterOutputTotalPackets', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the router output connection state (1 connected, 0 disconnected). Applies to SRT outputs only.
   */
  public metricConnected(props?: MetricOptions): Metric {
    return this.metric('RouterOutputConnected', {
      statistic: 'min',
      period: Duration.seconds(60),
      unit: Unit.NONE,
      ...props,
    });
  }

  /**
   * Metric for the number of retransmitted packets requested through ARQ. Applies to RIST and SRT outputs.
   */
  public metricArqRequests(props?: MetricOptions): Metric {
    return this.metric('RouterOutputARQRequests', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }
}

/**
 * Defines an AWS Elemental MediaConnect Router Output.
 */
@propertyInjectable
export class RouterOutput extends RouterOutputBase implements IRouterOutput {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.RouterOutput';

  /**
   * Import an existing Router Output from its ARN.
   *
   * The ARN only contains the service-generated ID (not the router output name). Accessing
   * `routerOutputName` on an ARN import throws — use `fromRouterOutputAttributes()` to provide
   * the name explicitly.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param routerOutputArn The ARN of the Router Output
   * @returns A Router Output construct
   */
  public static fromRouterOutputArn(scope: Construct, id: string, routerOutputArn: string): IRouterOutput {
    return RouterOutput.fromRouterOutputAttributes(scope, id, { routerOutputArn });
  }

  /**
   * Import an existing Router Output from attributes.
   *
   * Use this when you need to reference a router output by name.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param attrs The Router Output attributes
   * @returns A Router Output construct
   */
  public static fromRouterOutputAttributes(scope: Construct, id: string, attrs: RouterOutputAttributes): IRouterOutput {
    // Derive the ID from the ARN using CFN intrinsics so this works with token ARNs too.
    // The ARN format is arn:aws:mediaconnect:<region>:<account>:routerOutput:<id>, so the
    // resource name is the ID.
    const routerOutputId = attrs.routerOutputId ?? extractRouterOutputId(attrs.routerOutputArn);

    class Import extends RouterOutputBase implements IRouterOutput {
      public readonly routerOutputArn = attrs.routerOutputArn;
      public readonly routerOutputId = routerOutputId;
      public readonly createdAt = undefined;
      public readonly updatedAt = undefined;

      public get routerOutputName(): string {
        if (attrs.routerOutputName) return attrs.routerOutputName;
        throw new ValidationError(
          lit`RouterOutputNameNotProvided`,
          `'routerOutputName' was not provided when importing RouterOutput ${this.node.path}; provide it in fromRouterOutputAttributes()`,
          this,
        );
      }

      public get ipAddress(): string {
        throw new ValidationError(
          lit`RouterOutputIpAddressNotProvided`,
          `'ipAddress' is not available on imported RouterOutput ${this.node.path}`,
          this,
        );
      }
    }
    return new Import(scope, id);
  }

  /**
   * Returns the maximum bitrate in Mbps for a known tier, or undefined for custom tiers.
   */
  private static tierLimitMbps(tier: RouterOutputTier): number | undefined {
    if (tier.value === RouterOutputTier.OUTPUT_20.value) return 20;
    if (tier.value === RouterOutputTier.OUTPUT_50.value) return 50;
    if (tier.value === RouterOutputTier.OUTPUT_100.value) return 100;
    return undefined;
  }

  public readonly routerOutputArn: string;
  public readonly routerOutputName: string;
  public readonly routerOutputId: string;
  public readonly ipAddress: string;
  public readonly createdAt?: string;
  public readonly updatedAt?: string;

  constructor(scope: Construct, id: string, props: RouterOutputProps) {
    super(scope, id, {
      physicalName: props.routerOutputName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 128 }) }),
    });

    // Validate router output name if provided
    if (props.routerOutputName != null && props.routerOutputName !== '' && !Token.isUnresolved(props.routerOutputName)) {
      if (props.routerOutputName.length > 128) {
        throw new ValidationError(lit`RouterOutputNameLength`, `Router output name must be between 1 and 128 characters, got ${props.routerOutputName.length}`, this);
      }
      if (!/^[a-zA-Z0-9-]+$/.test(props.routerOutputName)) {
        throw new ValidationError(lit`RouterOutputNameFormat`, `Router output name must contain only alphanumeric characters and hyphens, got '${props.routerOutputName}'`, this);
      }
    }

    // Validate maximum bitrate
    if (!props.maximumBitrate.isUnresolved() && props.maximumBitrate.toBps() < 1000000) {
      throw new ValidationError(lit`RouterOutputMinBitrate`, `Maximum bitrate must be at least 1,000,000 bits/s (1 Mbps), got ${props.maximumBitrate.toBps()}`, this);
    }

    // Validate bitrate does not exceed tier limit
    const resolvedTier = props.tier ?? RouterOutputTier.OUTPUT_20;
    const tierMbps = RouterOutput.tierLimitMbps(resolvedTier);
    if (exceedsRouterTierBitrate(tierMbps, props.maximumBitrate)) {
      throw new ValidationError(lit`RouterOutputBitrateExceedsTier`, `Maximum bitrate ${props.maximumBitrate.toBps() / 1_000_000} Mbps exceeds the ${resolvedTier.value} tier limit of ${tierMbps} Mbps`, this);
    }

    // Validate maintenance time format
    if (props.maintenanceConfiguration) {
      validateMaintenanceTime(props.maintenanceConfiguration.time);
    }

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const stack = Stack.of(this);
    const targetRegion = props.regionName ?? stack.region;

    // Wildcard the id — pinning the live ARN (attrArn) would create a role → router-output → role cycle.
    const routerOutputArn = stack.formatArn({
      service: 'mediaconnect',
      region: targetRegion,
      resource: 'routerOutput',
      resourceName: '*',
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
    });

    const configBind = props.configuration._bind(this, routerOutputArn);

    // Check to see if region specified is also compatible with AZ configured for some of the Router Outputs configurations
    if (configBind.availabilityZone && !Token.isUnresolved(configBind.availabilityZone) && !Token.isUnresolved(targetRegion)
      && !configBind.availabilityZone.startsWith(targetRegion)) {
      throw new ValidationError(lit`RouterOutputAzRegionMismatch`, `Availability zone '${configBind.availabilityZone}' must be within region '${targetRegion}'`, this);
    }

    const routerOutput = new CfnRouterOutput(this, 'Resource', {
      name: this.physicalName,
      maximumBitrate: props.maximumBitrate.toBps(),
      routingScope: props.routingScope.value,
      tier: (props.tier ?? RouterOutputTier.OUTPUT_20).value,
      availabilityZone: configBind.availabilityZone,
      maintenanceConfiguration: props.maintenanceConfiguration ? {
        preferredDayTime: props.maintenanceConfiguration,
      } : {
        default: {},
      },
      configuration: configBind.config,
      regionName: targetRegion,
      tags: props.tags ? renderTags(props.tags) : undefined,
    });

    this.routerOutputArn = routerOutput.attrArn;
    this.routerOutputName = this.physicalName;
    this.routerOutputId = routerOutput.attrId;
    this.ipAddress = routerOutput.attrIpAddress;
    this.createdAt = routerOutput.attrCreatedAt;
    this.updatedAt = routerOutput.attrUpdatedAt;

    // Without this the router input can be created before the
    // policy is attached ("access denied" at create time). Ordering the grant before the
    // router input pulls the policy in as a dependency.
    configBind.grant?.applyBefore(routerOutput);
  }
}

function extractRouterOutputId(routerOutputArn: string): string {
  const parsed = Arn.split(routerOutputArn, ArnFormat.COLON_RESOURCE_NAME).resourceName;
  if (parsed === undefined) {
    throw new UnscopedValidationError(
      lit`InvalidRouterOutputArn`,
      `Could not parse Router Output ARN: ${routerOutputArn}. Expected format: arn:<partition>:mediaconnect:<region>:<account>:routerOutput:<id>`,
    );
  }
  return parsed;
}
