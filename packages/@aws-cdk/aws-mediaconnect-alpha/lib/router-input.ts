import type { Bitrate, IResource } from 'aws-cdk-lib';
import { ArnFormat, Duration, Fn, Lazy, Names, Resource, Stack, Token, UnscopedValidationError, ValidationError } from 'aws-cdk-lib';
import type { MetricOptions } from 'aws-cdk-lib/aws-cloudwatch';
import { Metric, Unit } from 'aws-cdk-lib/aws-cloudwatch';
import { CfnRouterInput } from 'aws-cdk-lib/aws-mediaconnect';
import type { IRouterInputRef, RouterInputReference } from 'aws-cdk-lib/aws-mediaconnect';
import type { ISecret } from 'aws-cdk-lib/aws-secretsmanager';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { IFlow } from './flow';
import type { IFlowOutput } from './flow-output';
import { RouterInputGrants } from './mediaconnect-grants.generated';
import type { IRouterNetworkInterface } from './router-network-interface';
import type { MaintenanceDay, MediaLivePipeline, RouterSrtEncryption, TransitEncryption } from './shared';
import { renderTags, exceedsRouterTierBitrate, renderRouterSrtEncryption, renderTransitEncryption, resolveTransitEncryption, validateMaintenanceTime } from './shared';

/**
 * Protocol options available for Router Input configurations
 */
export class RouterInputProtocolOptions {
  /** Real-time Transport Protocol */
  public static readonly RTP = new RouterInputProtocolOptions('RTP');
  /** Reliable Internet Stream Transport */
  public static readonly RIST = new RouterInputProtocolOptions('RIST');
  /** Secure Reliable Transport - Caller mode */
  public static readonly SRT_CALLER = new RouterInputProtocolOptions('SRT_CALLER');
  /** Secure Reliable Transport - Listener mode */
  public static readonly SRT_LISTENER = new RouterInputProtocolOptions('SRT_LISTENER');

  /**
   * Use a custom protocol value
   * @param value The protocol string value
   */
  public static of(value: string): RouterInputProtocolOptions {
    return new RouterInputProtocolOptions(value);
  }

  /** @param value The protocol string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * A single ingest endpoint where the router input listens for upstream content.
 *
 * Failover and merge configurations expose two endpoints; standard configurations
 * expose one. Returned by {@link IRouterInput.endpoints}.
 */
export interface RouterInputEndpoint {
  /**
   * The full ingest URL combining protocol scheme, IP, and port. For example:
   * `srt://203.0.113.10:5000` or `rtp://203.0.113.10:5001`.
   */
  readonly url: string;

  /**
   * The listening port at which the router input accepts upstream content.
   */
  readonly port: number;
}

/**
 * Interface for Router Input
 */
export interface IRouterInput extends IResource, IRouterInputRef {
  /**
   * The Amazon Resource Name (ARN) of the router input.
   *
   * @attribute
   */
  readonly routerInputArn: string;

  /**
   * The unique identifier of the router input.
   *
   * @attribute
   */
  readonly routerInputId: string;

  /**
   * The IP address of the router input.
   *
   * @attribute
   */
  readonly ipAddress: string;

  /**
   * The timestamp when the router input was created.
   *
   * @attribute
   */
  readonly createdAt?: string;

  /**
   * The timestamp when the router input was last updated.
   *
   * @attribute
   */
  readonly updatedAt?: string;

  /**
   * The ingest endpoints (URL + port) where the router input listens.
   *
   * Returns one entry for standard protocol-based variants (RTP, RIST, SRT listener),
   * and one entry per source for failover and merge configurations built from those
   * protocols. For example a failover RTP input will return:
   *
   * ```
   * [
   *   { url: 'rtp://203.0.113.10:5000', port: 5000 },
   *   { url: 'rtp://203.0.113.10:5001', port: 5001 },
   * ]
   * ```
   *
   * Accessing this on SRT caller (where the router input dials out to a remote
   * source), MediaConnect Flow, MediaLive Channel, or imported inputs throws — those
   * variants do not expose host:port pairs the input listens on.
   */
  readonly endpoints: RouterInputEndpoint[];

  /**
   * The Secrets Manager secret containing the transit encryption passphrase.
   *
   * Only available when the Router Input was created with explicit
   * `transitEncryption` configuration. Not available for
   * automatic encryption or imported inputs.
   */
  readonly transitEncryptionSecret?: ISecret;

  /**
   * Collection of grant methods for this router input.
   */
  readonly grants: RouterInputGrants;

  /**
   * Create a CloudWatch metric for this router input.
   *
   * Router input metrics are dimensioned by `RouterInputARN`. See the MediaConnect
   * documentation for available metric names (e.g. `RouterInputBitRate`,
   * `RouterInputNotRecoveredPackets`).
   */
  metric(metricName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the bitrate of the router input's payload.
   * @default - average over 60 seconds
   */
  metricBitrate(props?: MetricOptions): Metric;

  /**
   * Metric for packets lost in transit that were not recovered by error correction.
   * @default - sum over 60 seconds
   */
  metricNotRecoveredPackets(props?: MetricOptions): Metric;

  /**
   * Metric for the total number of packets received by the router input.
   * @default - sum over 60 seconds
   */
  metricTotalPackets(props?: MetricOptions): Metric;

  /**
   * Metric for the router input connection state (1 connected, 0 disconnected). Applies to SRT sources only.
   * @default - minimum over 60 seconds
   */
  metricConnected(props?: MetricOptions): Metric;

  /**
   * Metric for continuity counter errors in the transport stream.
   * @default - sum over 60 seconds
   */
  metricContinuityCounterErrors(props?: MetricOptions): Metric;

  /**
   * Metric for the recovery latency of the input stream. Applies to RIST, SRT, and RTP-FEC.
   * @default - average over 60 seconds
   */
  metricLatency(props?: MetricOptions): Metric;

  /**
   * Metric for the number of times the router input has switched sources in Failover mode.
   * @default - sum over 60 seconds
   */
  metricFailoverSwitches(props?: MetricOptions): Metric;
}

/**
 * Routing scope for the Router Input
 */
export class RoutingScope {
  /** Route traffic within the same AWS region */
  public static readonly REGIONAL = new RoutingScope('REGIONAL');
  /** Route traffic globally across AWS regions */
  public static readonly GLOBAL = new RoutingScope('GLOBAL');

  /**
   * Use a custom routing scope value
   * @param value The routing scope string value
   */
  public static of(value: string): RoutingScope {
    return new RoutingScope(value);
  }

  /** @param value The routing scope string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * Routing tier based on your maximum bitrate requirements.
 */
export class RouterInputTier {
  /** Supports a maximum bitrate up to 100 megabits per second. */
  public static readonly INPUT_100 = new RouterInputTier('INPUT_100');
  /** Supports a maximum bitrate up to 50 megabits per second. */
  public static readonly INPUT_50 = new RouterInputTier('INPUT_50');
  /** Supports a maximum bitrate up to 20 megabits per second. */
  public static readonly INPUT_20 = new RouterInputTier('INPUT_20');

  /**
   * Use a custom tier value
   * @param value The tier string value
   */
  public static of(value: string): RouterInputTier {
    return new RouterInputTier(value);
  }

  /** @param value The tier string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * Configuration for scheduled maintenance windows
 */
export interface MaintenanceConfiguration {
  /** Day of the week for maintenance (e.g., 'Monday', 'Tuesday') */
  readonly day: MaintenanceDay;
  /** Time of day for maintenance in HH:MM format (e.g., '02:00') */
  readonly time: string;
}

/**
 * Properties for creating a Router Input
 */
export interface RouterInputProps {
  /**
   * Name of the Router Input
   * @default - Generated automatically
   */
  readonly routerInputName?: string;
  /** The maximum bitrate for the router input. */
  readonly maximumBitrate: Bitrate;
  /** Indicates whether the router input is configured for Regional or global routing. */
  readonly routingScope: RoutingScope;
  /**
   * Select a tier based on your maximum bitrate requirements.
   * @default RouterInputTier.INPUT_20
   */
  readonly tier?: RouterInputTier;
  /** Configuration for the Router Input (standard, failover, merge, or MediaConnect flow) */
  readonly configuration: RouterInputConfiguration;
  /**
   * Maintenance window configuration
   * @default - Default maintenance window will be used
   */
  readonly maintenanceConfiguration?: MaintenanceConfiguration;
  /**
   * The AWS Region where the router input is located.
   * @default - Defaults to the same region as stack
   */
  readonly regionName?: string;
  /**
   * Tags to add to the Router Input
   * @default - No tagging
   */
  readonly tags?: { [key: string]: string };
  /**
   * Transit encryption configuration using AWS Secrets Manager.
   *
   * When provided without a role, a scoped IAM role is automatically created with read
   * access to the secret.
   *
   * @default - Automatic encryption will be configured
   */
  readonly transitEncryption?: TransitEncryption;
}

/**
 * Attributes for importing an existing Router Input.
 */
export interface RouterInputAttributes {
  /**
   * The Amazon Resource Name (ARN) of the router input.
   */
  readonly routerInputArn: string;

  /**
   * The unique identifier of the router input.
   *
   * @default - accessing `routerInputId` on the imported input throws; only provide when available.
   */
  readonly routerInputId?: string;

  /**
   * The IP address that the router input uses to ingest content.
   *
   * @default - accessing `ipAddress` on the imported input throws; only provide when available.
   */
  readonly ipAddress?: string;
}

/**
 * Forward Error Correction (FEC) options for RTP protocol
 */
export enum ForwardErrorCorrection {
  /** Enable Forward Error Correction */
  ENABLED='ENABLED',
  /** Disable Forward Error Correction */
  DISABLED='DISABLED',
}

/**
 * Identifies which protocol in a failover configuration's protocols array is primary.
 */
export enum PrimarySource {
  /** The first protocol in the failover protocols array. */
  FIRST_SOURCE = 0,
  /** The second protocol in the failover protocols array. */
  SECOND_SOURCE = 1,
}

/**
 * Internal CFN-level values for the source priority mode.
 * @internal
 */
enum SourcePriorityMode {
  /** Both sources are equal priority — the service picks one. */
  NO_PRIORITY = 'NO_PRIORITY',
  /** One source is primary, the other is backup. */
  PRIMARY_SECONDARY = 'PRIMARY_SECONDARY',
}

/**
 * Source priority configuration for failover Router Input configurations.
 */
export class SourcePriorityConfig {
  /**
   * Treat both sources with equal priority — MediaConnect picks one when needed.
   */
  public static none(): SourcePriorityConfig {
    return new SourcePriorityConfig(SourcePriorityMode.NO_PRIORITY, undefined);
  }

  /**
   * Designate one of the two sources as primary; the other is treated as backup.
   */
  public static primarySecondary(primary: PrimarySource): SourcePriorityConfig {
    return new SourcePriorityConfig(SourcePriorityMode.PRIMARY_SECONDARY, primary);
  }

  private constructor(
    private readonly _mode: SourcePriorityMode,
    private readonly _primary: PrimarySource | undefined,
  ) {}

  /**
   * Resolve the configuration to its CFN shape.
   * @internal
   */
  public _bind(): { sourcePriorityMode: string; primarySourceIndex?: number } {
    return {
      sourcePriorityMode: this._mode,
      primarySourceIndex: this._primary,
    };
  }
}

/**
 * Properties for RTP protocol configuration
 */
export interface RtpProtocolProps {
  /** Port number for RTP traffic */
  readonly port: number;
  /**
   * Forward Error Correction setting
   * @default - undefined; when omitted, MediaConnect applies ForwardErrorCorrection.DISABLED at deploy time
   */
  readonly forwardErrorCorrection?: ForwardErrorCorrection;
}

/**
 * Properties for RIST protocol configuration
 */
export interface RistProtocolProps {
  /** Port number for RIST traffic */
  readonly port: number;
  /** Recovery latency for RIST */
  readonly recoveryLatency: Duration;
}

/**
 * Properties for SRT Listener protocol configuration
 */
export interface SrtListenerProtocolProps {
  /** Port number for SRT listener */
  readonly port: number;
  /** Minimum latency for SRT */
  readonly minimumLatency: Duration;
  /**
   * Optional decryption configuration for encrypted SRT streams
   * @default - No decryption
   */
  readonly decryptionConfiguration?: RouterSrtEncryption;
}

/**
 * Properties for SRT Caller protocol configuration
 */
export interface SrtCallerProtocolProps {
  /** Source IP address to connect to */
  readonly sourceAddress: string;
  /** Source port to connect to */
  readonly sourcePort: number;
  /** Minimum latency for SRT */
  readonly minimumLatency: Duration;
  /**
   * Optional stream ID for SRT connection
   * @default - No stream ID
   */
  readonly streamId?: string;
  /**
   * Optional decryption configuration for encrypted SRT streams
   * @default - No decryption
   */
  readonly decryptionConfiguration?: RouterSrtEncryption;
}

/**
 * Identifies which configuration type a port conflict check is being performed for.
 * @internal
 */
enum ProtocolConfigType {
  /** Failover-mode router input configuration. */
  FAILOVER = 'failover',
  /** Merge-mode router input configuration. */
  MERGE = 'merge',
}

/**
 * Factory class for creating Router Input protocol configurations
 */
export class RouterInputProtocol {
  /**
   * Create an RTP protocol configuration
   * @param props RTP protocol properties
   * @returns RouterInputProtocol instance configured for RTP
   */
  public static rtp(props: RtpProtocolProps): RouterInputProtocol {
    RouterInputProtocol.validatePort(props.port);
    return new RouterInputProtocol(RouterInputProtocolOptions.RTP, {
      rtp: {
        port: props.port,
        forwardErrorCorrection: props.forwardErrorCorrection,
      },
    }, props.port);
  }

  /**
   * Create a RIST protocol configuration
   * @param props RIST protocol properties
   * @returns RouterInputProtocol instance configured for RIST
   */
  public static rist(props: RistProtocolProps): RouterInputProtocol {
    RouterInputProtocol.validatePort(props.port);
    return new RouterInputProtocol(RouterInputProtocolOptions.RIST, {
      rist: {
        port: props.port,
        recoveryLatencyMilliseconds: props.recoveryLatency.toMilliseconds(),
      },
    }, props.port);
  }

  /**
   * Create an SRT Listener protocol configuration
   * @param props SRT Listener protocol properties
   * @returns RouterInputProtocol instance configured for SRT Listener
   */
  public static srtListener(props: SrtListenerProtocolProps): RouterInputProtocol {
    RouterInputProtocol.validatePort(props.port);
    return new RouterInputProtocol(RouterInputProtocolOptions.SRT_LISTENER, {
      srtListener: {
        port: props.port,
        minimumLatencyMilliseconds: props.minimumLatency.toMilliseconds(),
      },
    }, props.port, props.decryptionConfiguration);
  }

  /**
   * Create an SRT Caller protocol configuration
   * @param props SRT Caller protocol properties
   * @returns RouterInputProtocol instance configured for SRT Caller
   */
  public static srtCaller(props: SrtCallerProtocolProps): RouterInputProtocol {
    RouterInputProtocol.validateSrtCallerPort(props.sourcePort);
    return new RouterInputProtocol(RouterInputProtocolOptions.SRT_CALLER, {
      srtCaller: {
        sourceAddress: props.sourceAddress,
        sourcePort: props.sourcePort,
        minimumLatencyMilliseconds: props.minimumLatency.toMilliseconds(),
        streamId: props.streamId,
      },
    }, undefined, props.decryptionConfiguration);
  }

  /**
   * Validate that port is within the valid range (3000-30000)
   */
  private static validatePort(port: number): void {
    if (!Token.isUnresolved(port) && (port < 3000 || port > 30000)) {
      throw new UnscopedValidationError(lit`RouterInputPortRange`, `Port must be between 3000 and 30000, got ${port}`);
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

  private readonly _protocolName: RouterInputProtocolOptions;
  private readonly _config: CfnRouterInput.RouterInputProtocolConfigurationProperty;
  private readonly _port?: number;
  private readonly _decryption?: RouterSrtEncryption;

  private constructor(
    protocolName: RouterInputProtocolOptions,
    config: CfnRouterInput.RouterInputProtocolConfigurationProperty,
    port?: number,
    decryption?: RouterSrtEncryption,
  ) {
    this._protocolName = protocolName;
    this._config = config;
    this._port = port;
    this._decryption = decryption;
  }

  /**
   * Resolve this protocol to its CFN-ready form. Returns the protocol identifier,
   * the CFN configuration shape (with SRT decryption auto-role materialized if needed),
   * and the ingest port if the protocol has one.
   *
   * @param scope         Construct that scopes the auto-created decryption role
   * @param routerInputArn  Optional ARN of the consuming router input — when provided,
   *                        scopes the auto-created decryption role's trust policy.
   *
   * @internal
   */
  public _bind(scope: Construct, routerInputArn?: string): {
    name: RouterInputProtocolOptions;
    config: CfnRouterInput.RouterInputProtocolConfigurationProperty;
    port?: number;
  } {
    if (!this._decryption) {
      return { name: this._protocolName, config: this._config, port: this._port };
    }

    const decryptionConfiguration = renderRouterSrtEncryption(scope, 'DecryptionRole', this._decryption, routerInputArn);

    if (this._config.srtListener) {
      return {
        name: this._protocolName,
        port: this._port,
        config: { srtListener: { ...this._config.srtListener, decryptionConfiguration } },
      };
    }
    if (this._config.srtCaller) {
      return {
        name: this._protocolName,
        port: this._port,
        config: { srtCaller: { ...this._config.srtCaller, decryptionConfiguration } },
      };
    }
    return { name: this._protocolName, config: this._config, port: this._port };
  }

  /**
   * Compute the ingest URL (scheme://ipAddress:port) for this protocol, if it has
   * a listening port.
   *
   * @internal
   */
  public _ingestUrl(ipAddress: string): string | undefined {
    if (this._port === undefined) return undefined;
    return renderIngestUrl(this._protocolName.value, this._port, ipAddress);
  }

  /**
   * The listening port for this protocol, if it has one. Undefined for SRT caller
   * (which has a `sourcePort` pointing at the upstream listener, not a port the input
   * itself listens on).
   *
   * @internal
   */
  public _listeningPort(): number | undefined {
    return this._port;
  }

  /**
   * Whether two protocols share the same protocol identifier.
   *
   * @internal
   */
  public _matchesKind(other: RouterInputProtocol): boolean {
    return this._protocolName.value === other._protocolName.value;
  }

  /**
   * Whether this protocol is SRT (Caller or Listener).
   *
   * @internal
   */
  public _isSrt(): boolean {
    return this._protocolName.value === RouterInputProtocolOptions.SRT_CALLER.value
      || this._protocolName.value === RouterInputProtocolOptions.SRT_LISTENER.value;
  }

  /**
   * Validate port compatibility between this protocol and another in a failover or
   * merge configuration. Throws on conflicts appropriate to the protocol kind.
   *
   * @internal
   */
  public _validatePortConflicts(other: RouterInputProtocol, configType: ProtocolConfigType): void {
    const protocolValue = this._protocolName.value;

    if (protocolValue === RouterInputProtocolOptions.RIST.value) {
      const rist1 = this._config.rist;
      const rist2 = other._config.rist;
      if (rist1 && rist2 && 'port' in rist1 && 'port' in rist2) {
        const port1 = rist1.port;
        const port2 = rist2.port;
        if (port1 === port2) {
          throw new UnscopedValidationError(
            lit`DuplicateRistPort`,
            `${configType} configuration cannot use the same RIST port ${port1} for both protocols, ` +
            'each protocol must use a different port number',
          );
        }
        if (Math.abs(port1 - port2) < 2) {
          throw new UnscopedValidationError(
            lit`ConsecutiveRistPort`,
            `${configType} configuration cannot use consecutive RIST ports ${Math.min(port1, port2)} and ${Math.max(port1, port2)}, ` +
            'RIST protocols require non-consecutive ports to avoid overlapping ranges (e.g., use 6000 and 6002)',
          );
        }
      }
      return;
    }

    if (protocolValue === RouterInputProtocolOptions.RTP.value) {
      const rtp1 = this._config.rtp;
      const rtp2 = other._config.rtp;
      if (rtp1 && rtp2 && 'port' in rtp1 && 'port' in rtp2 && rtp1.port === rtp2.port) {
        throw new UnscopedValidationError(
          lit`DuplicateRtpPort`,
          `${configType} configuration cannot use the same RTP port ${rtp1.port} for both protocols, ` +
          'each protocol must use a different port number',
        );
      }
      return;
    }

    if (protocolValue === RouterInputProtocolOptions.SRT_LISTENER.value) {
      const srt1 = this._config.srtListener;
      const srt2 = other._config.srtListener;
      if (srt1 && srt2 && 'port' in srt1 && 'port' in srt2 && srt1.port === srt2.port) {
        throw new UnscopedValidationError(
          lit`DuplicateSrtListenerPort`,
          `${configType} configuration cannot use the same SRT Listener port ${srt1.port} for both protocols, ` +
          'each protocol must use a different port number',
        );
      }
    }
  }
}

/**
 * Properties for standard Router Input configuration
 */
export interface StandardConfigurationProps {
  /** Network interface for the Router Input */
  readonly networkInterface: IRouterNetworkInterface;
  /** Protocol configuration for the input */
  readonly protocol: RouterInputProtocol;
}

/**
 * Properties for failover Router Input configuration
 */
export interface FailoverConfigurationProps {
  /** Network interface for the Router Input */
  readonly networkInterface: IRouterNetworkInterface;
  /** Array of exactly 2 protocol configurations for failover (must be same protocol type) */
  readonly protocols: RouterInputProtocol[];
  /**
   * Source priority configuration for failover.
   *
   * @default SourcePriorityConfig.none()
   */
  readonly sourcePriority?: SourcePriorityConfig;
}

/**
 * Properties for merge Router Input configuration
 */
export interface MergeConfigurationProps {
  /** Network interface for the Router Input */
  readonly networkInterface: IRouterNetworkInterface;
  /** Array of exactly 2 protocol configurations for merge (must be same non-SRT protocol type) */
  readonly protocols: RouterInputProtocol[];
  /** Recovery window for merge operation */
  readonly mergeRecoveryWindow: Duration;
}

/**
 * Properties for MediaConnect Flow Router Input configuration
 */
export interface MediaConnectFlowConfigurationProps {
  /** The MediaConnect flow to use as input */
  readonly flow: IFlow;
  /** The flow output that feeds this router input */
  readonly flowOutput: IFlowOutput;
  /**
   * Optional transit encryption configuration
   * @default - Automatic encryption will be used
   */
  readonly sourceTransitDecryption?: TransitEncryption;
}

/**
 * Properties for MediaConnect Flow Router Input configuration - without a connection
 */
export interface MediaConnectFlowConfigurationWithoutConnectionProps {
  /**
   * Availability Zone the router input will be placed in.
   */
  readonly availabilityZone: string;
  /**
   * Optional transit encryption configuration
   * @default - Automatic encryption will be used
   */
  readonly sourceTransitDecryption?: TransitEncryption;
}

/**
 * Properties for MediaLive Channel Router Input configuration.
 *
 * Use this when the MediaLive channel already exists and you want to ingest
 * from one of its outputs immediately.
 */
export interface MediaLiveChannelConfigurationProps {
  /**
   * ARN of the MediaLive channel to use as input.
   *
   * Note: This will change to accept a typed MediaLive channel reference
   * when the @aws-cdk/aws-medialive-alpha L2 construct is released.
   */
  readonly mediaLiveChannelArn: string;

  /**
   * The name of the MediaLive channel output to connect to this router input.
   */
  readonly mediaLiveChannelOutputName: string;

  /**
   * The MediaLive pipeline to connect to this router input.
   */
  readonly mediaLivePipelineId: MediaLivePipeline;

  /**
   * Optional transit encryption configuration.
   *
   * @default - Automatic encryption will be used
   */
  readonly sourceTransitDecryption?: TransitEncryption;
}

/**
 * Properties for MediaLive Channel Router Input configuration without a specific channel connection.
 *
 * Use this when you want to set up the router input before the target MediaLive channel exists.
 */
export interface MediaLiveChannelConfigurationWithoutConnectionProps {
  /**
   * Availability Zone the router input will be placed in.
   */
  readonly availabilityZone: string;

  /**
   * Optional transit encryption configuration.
   *
   * @default - Automatic encryption will be used
   */
  readonly sourceTransitDecryption?: TransitEncryption;
}

/**
 * Factory class for creating Router Input configurations
 */
export abstract class RouterInputConfiguration {
  /**
   * Create a standard Router Input configuration with a single protocol
   * @param props Standard configuration properties
   * @returns RouterInputConfiguration instance for standard setup
   */
  public static standard(props: StandardConfigurationProps): RouterInputConfiguration {
    return new StandardRouterInputConfig(props);
  }

  /**
   * Create a failover Router Input configuration with two matching protocols
   * @param props Failover configuration properties
   * @returns RouterInputConfiguration instance for failover setup
   * @throws Error if protocols don't match or count is not exactly 2
   */
  public static failover(props: FailoverConfigurationProps): RouterInputConfiguration {
    if (!Token.isUnresolved(props.protocols) && props.protocols.length !== 2) {
      throw new UnscopedValidationError(lit`FailoverProtocolCount`, `Failover configuration requires exactly 2 protocols, got ${props.protocols.length}`);
    }
    if (!props.protocols[0]._matchesKind(props.protocols[1])) {
      throw new UnscopedValidationError(lit`FailoverProtocolMismatch`, 'Protocols must match');
    }
    props.protocols[0]._validatePortConflicts(props.protocols[1], ProtocolConfigType.FAILOVER);
    return new FailoverRouterInputConfig(props);
  }

  /**
   * Create a merge Router Input configuration with two matching non-SRT protocols
   * @param props Merge configuration properties
   * @returns RouterInputConfiguration instance for merge setup
   * @throws Error if protocols don't match, count is not exactly 2, or SRT protocols are used
   */
  public static merge(props: MergeConfigurationProps): RouterInputConfiguration {
    if (!Token.isUnresolved(props.protocols) && props.protocols.length !== 2) {
      throw new UnscopedValidationError(lit`MergeProtocolCount`, `Merge configuration requires exactly 2 protocols, got ${props.protocols.length}`);
    }
    if (props.protocols.some(p => p._isSrt())) {
      throw new UnscopedValidationError(lit`MergeSrtNotSupported`, 'SRT protocols are not supported in merge configuration');
    }
    if (!props.protocols[0]._matchesKind(props.protocols[1])) {
      throw new UnscopedValidationError(lit`MergeProtocolMismatch`, 'Protocols must match');
    }
    props.protocols[0]._validatePortConflicts(props.protocols[1], ProtocolConfigType.MERGE);
    return new MergeRouterInputConfig(props);
  }

  /**
   * Create a MediaConnect Flow Router Input configuration without connecting to a specific flow
   * Use this when you want to prepare the router input for a flow connection later
   * @param props MediaConnect Flow configuration properties without connection
   * @returns RouterInputConfiguration instance for MediaConnect Flow setup without connection
   */
  public static mediaConnectFlowWithoutConnection(props: MediaConnectFlowConfigurationWithoutConnectionProps): RouterInputConfiguration {
    return new MediaConnectFlowRouterInputConfig({ sourceTransitDecryption: props.sourceTransitDecryption }, props.availabilityZone);
  }

  /**
   * Create a MediaConnect Flow Router Input configuration.
   *
   * Use this when the source flow already exists and you want to connect immediately.
   *
   * @param props MediaConnect Flow configuration properties
   * @returns RouterInputConfiguration instance for MediaConnect Flow setup
   */
  public static mediaConnectFlow(props: MediaConnectFlowConfigurationProps): RouterInputConfiguration {
    return new MediaConnectFlowRouterInputConfig({
      flow: props.flow,
      flowOutput: props.flowOutput,
      sourceTransitDecryption: props.sourceTransitDecryption,
    });
  }

  /**
   * Create a MediaLive Channel Router Input configuration.
   *
   * Use this when the source MediaLive channel already exists and you want to
   * ingest from one of its outputs immediately.
   *
   * @param props MediaLive channel configuration properties
   * @returns RouterInputConfiguration instance for MediaLive channel setup
   */
  public static mediaLiveChannel(props: MediaLiveChannelConfigurationProps): RouterInputConfiguration {
    return new MediaLiveChannelRouterInputConfig({
      mediaLiveChannelArn: props.mediaLiveChannelArn,
      mediaLiveChannelOutputName: props.mediaLiveChannelOutputName,
      mediaLivePipelineId: props.mediaLivePipelineId,
      sourceTransitDecryption: props.sourceTransitDecryption,
    });
  }

  /**
   * Create a MediaLive Channel Router Input configuration without a specific channel connection.
   *
   * Use this when you want to set up the router input before the target MediaLive channel exists.
   *
   * @param props MediaLive channel no-connection properties
   * @returns RouterInputConfiguration instance for MediaLive channel setup without connection
   */
  public static mediaLiveChannelWithoutConnection(props: MediaLiveChannelConfigurationWithoutConnectionProps): RouterInputConfiguration {
    return new MediaLiveChannelRouterInputConfig({ sourceTransitDecryption: props.sourceTransitDecryption }, props.availabilityZone);
  }

  /**
   * Resolve this configuration to its CloudFormation shape. Called by the RouterInput
   * constructor — each concrete subclass supplies its own variant-specific rendering.
   * @internal
   */
  public abstract _bind(scope: Construct, routerInputArn?: string): {
    config: CfnRouterInput.RouterInputConfigurationProperty; availabilityZone?: string;
  };

  /**
   * Compute the ingest endpoints for protocol-based inputs that listen on one or more
   * ports.
   *
   * Returns `undefined` when the variant has no concept of a listen port (SRT caller,
   * MediaConnect Flow, MediaLive Channel). Returns an array with one entry for
   * standard variants, and one entry per source for failover and merge.
   * @internal
   */
  public abstract _endpoints(ipAddress: string): RouterInputEndpoint[] | undefined;
}

/**
 * Render an ingest URL string (scheme://ip:port). Used by `_ingestUrl()` on individual
 * protocols and composed into the per-config-variant `_endpoints()` results.
 *
 * Extracts the base protocol for the URL scheme (e.g. SRT_LISTENER → srt, RTP → rtp).
 */
function renderIngestUrl(protocolName: string, port: number, ipAddress: string): string {
  const scheme = protocolName.toLowerCase().split('_')[0];
  return Fn.join('', [`${scheme}://`, ipAddress, `:${port}`]);
}

/**
 * Concrete variant: standard (single-protocol) router input configuration.
 * @internal
 */
class StandardRouterInputConfig extends RouterInputConfiguration {
  constructor(private readonly props: StandardConfigurationProps) { super(); }

  public _bind(scope: Construct, routerInputArn?: string) {
    const protocol = this.props.protocol._bind(scope, routerInputArn);
    return {
      config: {
        standard: {
          networkInterfaceArn: this.props.networkInterface.routerNetworkInterfaceArn,
          protocol: protocol.name.value,
          protocolConfiguration: protocol.config,
        },
      },
    };
  }

  public _endpoints(ipAddress: string): RouterInputEndpoint[] | undefined {
    const url = this.props.protocol._ingestUrl(ipAddress);
    const port = this.props.protocol._listeningPort();
    if (url === undefined || port === undefined) return undefined;
    return [{ url, port }];
  }
}

/**
 * Concrete variant: failover router input configuration.
 * @internal
 */
class FailoverRouterInputConfig extends RouterInputConfiguration {
  constructor(private readonly props: FailoverConfigurationProps) { super(); }

  public _bind(scope: Construct, routerInputArn?: string) {
    const priority = (this.props.sourcePriority ?? SourcePriorityConfig.none())._bind();
    return {
      config: {
        failover: {
          networkInterfaceArn: this.props.networkInterface.routerNetworkInterfaceArn,
          protocolConfigurations: this.props.protocols.map(p => p._bind(scope, routerInputArn).config),
          sourcePriorityMode: priority.sourcePriorityMode,
          primarySourceIndex: priority.primarySourceIndex,
        },
      },
    };
  }

  public _endpoints(ipAddress: string): RouterInputEndpoint[] | undefined {
    const endpoints: RouterInputEndpoint[] = [];
    for (const p of this.props.protocols) {
      const url = p._ingestUrl(ipAddress);
      const port = p._listeningPort();
      if (url === undefined || port === undefined) return undefined;
      endpoints.push({ url, port });
    }
    return endpoints;
  }
}

/**
 * Concrete variant: merge router input configuration.
 * @internal
 */
class MergeRouterInputConfig extends RouterInputConfiguration {
  constructor(private readonly props: MergeConfigurationProps) { super(); }

  public _bind(scope: Construct, routerInputArn?: string) {
    return {
      config: {
        merge: {
          networkInterfaceArn: this.props.networkInterface.routerNetworkInterfaceArn,
          protocolConfigurations: this.props.protocols.map(p => p._bind(scope, routerInputArn).config),
          mergeRecoveryWindowMilliseconds: this.props.mergeRecoveryWindow.toMilliseconds(),
        },
      },
    };
  }

  public _endpoints(ipAddress: string): RouterInputEndpoint[] | undefined {
    const endpoints: RouterInputEndpoint[] = [];
    for (const p of this.props.protocols) {
      const url = p._ingestUrl(ipAddress);
      const port = p._listeningPort();
      if (url === undefined || port === undefined) return undefined;
      endpoints.push({ url, port });
    }
    return endpoints;
  }
}

/**
 * Internal options for {@link MediaConnectFlowRouterInputConfig}.
 */
interface MediaConnectFlowRouterInputOptions {
  readonly flow?: IFlow;
  readonly flowOutput?: IFlowOutput;
  readonly sourceTransitDecryption?: TransitEncryption;
}

/**
 * Concrete variant: MediaConnect Flow router input configuration (with or without connection).
 * @internal
 */
class MediaConnectFlowRouterInputConfig extends RouterInputConfiguration {
  constructor(
    private readonly options: MediaConnectFlowRouterInputOptions,
    private readonly availabilityZone?: string,
  ) { super(); }

  public _bind(scope: Construct, routerInputArn?: string) {
    return {
      config: {
        mediaConnectFlow: {
          flowArn: this.options.flow?.flowArn,
          flowOutputArn: this.options.flowOutput?.flowOutputArn,
          sourceTransitDecryption: renderTransitEncryption(scope, 'SourceTransitDecryptionRole', this.options.sourceTransitDecryption, routerInputArn),
        },
      },
      availabilityZone: this.availabilityZone,
    };
  }

  public _endpoints(_ipAddress: string): RouterInputEndpoint[] | undefined {
    return undefined;
  }
}

/**
 * Internal options for {@link MediaLiveChannelRouterInputConfig}.
 */
interface MediaLiveChannelRouterInputOptions {
  readonly mediaLiveChannelArn?: string;
  readonly mediaLiveChannelOutputName?: string;
  readonly mediaLivePipelineId?: MediaLivePipeline;
  readonly sourceTransitDecryption?: TransitEncryption;
}

/**
 * Concrete variant: MediaLive Channel router input configuration (with or without connection).
 * @internal
 */
class MediaLiveChannelRouterInputConfig extends RouterInputConfiguration {
  constructor(
    private readonly options: MediaLiveChannelRouterInputOptions,
    private readonly availabilityZone?: string,
  ) { super(); }

  public _bind(scope: Construct, routerInputArn?: string) {
    return {
      config: {
        mediaLiveChannel: {
          mediaLiveChannelArn: this.options.mediaLiveChannelArn,
          mediaLiveChannelOutputName: this.options.mediaLiveChannelOutputName,
          mediaLivePipelineId: this.options.mediaLivePipelineId,
          sourceTransitDecryption: renderTransitEncryption(scope, 'SourceTransitDecryptionRole', this.options.sourceTransitDecryption, routerInputArn),
        },
      },
      availabilityZone: this.availabilityZone,
    };
  }

  public _endpoints(_ipAddress: string): RouterInputEndpoint[] | undefined {
    return undefined;
  }
}

/**
 * A MediaConnect Router Input that receives media streams and routes them to outputs.
 *
 * Router Inputs support multiple configuration types:
 * - Standard: Single protocol input
 * - Failover: Two matching protocols for redundancy
 * - Merge: Two matching non-SRT protocols for stream merging
 * - MediaConnect Flow: Input from an existing MediaConnect Flow
 */
abstract class RouterInputBase extends Resource implements IRouterInput {
  public abstract readonly routerInputArn: string;
  public abstract readonly routerInputId: string;
  public abstract readonly ipAddress: string;
  public abstract readonly createdAt?: string;
  public abstract readonly updatedAt?: string;
  public abstract readonly endpoints: RouterInputEndpoint[];
  public abstract readonly transitEncryptionSecret?: ISecret;
  public abstract readonly grants: RouterInputGrants;

  /**
   * A reference to this Router Input resource.
   * Required by the auto-generated IRouterInputRef interface.
   */
  public get routerInputRef(): RouterInputReference {
    return { routerInputArn: this.routerInputArn };
  }

  /**
   * Create a CloudWatch metric for this router input.
   */
  public metric(metricName: string, props?: MetricOptions): Metric {
    return new Metric({
      metricName,
      namespace: 'AWS/MediaConnect',
      ...props,
      dimensionsMap: {
        RouterInputARN: this.routerInputArn,
        ...props?.dimensionsMap,
      },
    }).attachTo(this);
  }

  /**
   * Metric for the bitrate of the router input's payload.
   */
  public metricBitrate(props?: MetricOptions): Metric {
    return this.metric('RouterInputBitRate', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.BITS_PER_SECOND,
      ...props,
    });
  }

  /**
   * Metric for packets lost in transit that were not recovered by error correction.
   */
  public metricNotRecoveredPackets(props?: MetricOptions): Metric {
    return this.metric('RouterInputNotRecoveredPackets', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the total number of packets received by the router input.
   */
  public metricTotalPackets(props?: MetricOptions): Metric {
    return this.metric('RouterInputTotalPackets', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the router input connection state (1 connected, 0 disconnected). Applies to SRT sources only.
   */
  public metricConnected(props?: MetricOptions): Metric {
    return this.metric('RouterInputConnected', {
      statistic: 'min',
      period: Duration.seconds(60),
      unit: Unit.NONE,
      ...props,
    });
  }

  /**
   * Metric for continuity counter errors in the transport stream.
   */
  public metricContinuityCounterErrors(props?: MetricOptions): Metric {
    return this.metric('RouterInputCCErrors', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the recovery latency of the input stream. Applies to RIST, SRT, and RTP-FEC.
   */
  public metricLatency(props?: MetricOptions): Metric {
    return this.metric('RouterInputLatency', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.MILLISECONDS,
      ...props,
    });
  }

  /**
   * Metric for the number of times the router input has switched sources in Failover mode.
   */
  public metricFailoverSwitches(props?: MetricOptions): Metric {
    return this.metric('RouterInputFailoverSwitches', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }
}

/**
 * Defines an AWS Elemental MediaConnect Router Input.
 */
@propertyInjectable
export class RouterInput extends RouterInputBase implements IRouterInput {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.RouterInput';

  /**
   * Import an existing Router Input from its ARN.
   *
   * Use `fromRouterInputAttributes()` instead if you need to expose `routerInputId`
   * or `ipAddress` on the imported construct.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param routerInputArn The ARN of the Router Input
   * @returns A Router Input construct
   */
  public static fromRouterInputArn(scope: Construct, id: string, routerInputArn: string): IRouterInput {
    return RouterInput.fromRouterInputAttributes(scope, id, { routerInputArn });
  }

  /**
   * Import an existing Router Input from its attributes.
   *
   * Provide `routerInputId` and/or `ipAddress` when importing an input that was deployed
   * externally — otherwise accessing those properties on the imported construct will throw.
   *
   * @param scope The parent construct
   * @param id The construct id
   * @param attrs The Router Input attributes
   * @returns A Router Input construct
   */
  public static fromRouterInputAttributes(scope: Construct, id: string, attrs: RouterInputAttributes): IRouterInput {
    class Import extends RouterInputBase implements IRouterInput {
      public readonly routerInputArn = attrs.routerInputArn;
      public readonly createdAt = undefined;
      public readonly updatedAt = undefined;
      public readonly transitEncryptionSecret = undefined;
      public readonly grants = RouterInputGrants.fromRouterInput(this);

      public get endpoints(): RouterInputEndpoint[] {
        throw new ValidationError(
          lit`RouterInputEndpointsNotAvailableImported`,
          `'endpoints' is not available on imported RouterInput ${this.node.path}; only RouterInputs constructed in this app for listening protocol variants (RTP, RIST, SRT listener, including failover and merge) expose ingest endpoints`,
          this,
        );
      }

      public get routerInputId(): string {
        if (attrs.routerInputId) return attrs.routerInputId;
        throw new ValidationError(
          lit`RouterInputIdNotProvided`,
          `'routerInputId' is not available on imported RouterInput ${this.node.path}; pass it via fromRouterInputAttributes`,
          this,
        );
      }

      public get ipAddress(): string {
        if (attrs.ipAddress) return attrs.ipAddress;
        throw new ValidationError(
          lit`RouterInputIpAddressNotProvided`,
          `'ipAddress' is not available on imported RouterInput ${this.node.path}; pass it via fromRouterInputAttributes`,
          this,
        );
      }
    }
    return new Import(scope, id);
  }

  /**
   * Returns the maximum bitrate in Mbps for a known tier, or undefined for custom tiers.
   */
  private static tierLimitMbps(tier: RouterInputTier): number | undefined {
    if (tier.value === RouterInputTier.INPUT_20.value) return 20;
    if (tier.value === RouterInputTier.INPUT_50.value) return 50;
    if (tier.value === RouterInputTier.INPUT_100.value) return 100;
    return undefined;
  }

  public readonly routerInputArn: string;
  public readonly routerInputId: string;
  public readonly ipAddress: string;
  public readonly createdAt?: string;
  public readonly updatedAt?: string;
  private readonly _endpoints: RouterInputEndpoint[] | undefined;
  public readonly transitEncryptionSecret?: ISecret;
  public readonly grants: RouterInputGrants;

  public get endpoints(): RouterInputEndpoint[] {
    if (this._endpoints === undefined) {
      throw new ValidationError(
        lit`RouterInputEndpointsNotAvailableConfig`,
        `'endpoints' is not available on this RouterInput ${this.node.path}; only listening protocol variants (RTP, RIST, SRT listener — including failover and merge built from those) expose ingest endpoints — SRT caller, MediaConnect Flow, and MediaLive Channel inputs do not`,
        this,
      );
    }
    return this._endpoints;
  }

  constructor(scope: Construct, id: string, props: RouterInputProps) {
    super(scope, id, {
      physicalName: props.routerInputName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 128 }) }),
    });

    // Validate router input name if provided
    if (props.routerInputName != null && props.routerInputName !== '' && !Token.isUnresolved(props.routerInputName)) {
      if (props.routerInputName.length > 128) {
        throw new ValidationError(lit`RouterInputNameLength`, `Router input name must be between 1 and 128 characters, got ${props.routerInputName.length}`, this);
      }
      if (!/^[a-zA-Z0-9-]+$/.test(props.routerInputName)) {
        throw new ValidationError(lit`RouterInputNameFormat`, `Router input name must contain only alphanumeric characters and hyphens, got '${props.routerInputName}'`, this);
      }
    }

    // Validate maximum bitrate
    if (!props.maximumBitrate.isUnresolved() && props.maximumBitrate.toBps() < 1000000) {
      throw new ValidationError(lit`RouterInputMinBitrate`, `Maximum bitrate must be at least 1,000,000 bits/s (1 Mbps), got ${props.maximumBitrate.toBps()}`, this);
    }

    // Validate bitrate does not exceed tier limit
    const resolvedTier = props.tier ?? RouterInputTier.INPUT_20;
    const tierMbps = RouterInput.tierLimitMbps(resolvedTier);
    if (exceedsRouterTierBitrate(tierMbps, props.maximumBitrate)) {
      throw new ValidationError(lit`RouterInputBitrateExceedsTier`, `Maximum bitrate ${props.maximumBitrate.toBps() / 1_000_000} Mbps exceeds the ${resolvedTier.value} tier limit of ${tierMbps} Mbps`, this);
    }

    // Validate maintenance time format
    if (props.maintenanceConfiguration) {
      validateMaintenanceTime(props.maintenanceConfiguration.time);
    }

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const stack = Stack.of(this);
    const targetRegion = props.regionName ?? stack.region;

    // Wildcard the id — pinning the live ARN (attrArn) would create a role → router-input → role cycle.
    const routerInputArn = stack.formatArn({
      service: 'mediaconnect',
      region: targetRegion,
      resource: 'routerInput',
      resourceName: '*',
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
    });

    const configBind = props.configuration._bind(this, routerInputArn);

    // Validate AZ matches region if provided
    if (configBind.availabilityZone && !configBind.availabilityZone.startsWith(targetRegion)) {
      throw new ValidationError(lit`RouterInputAzRegionMismatch`, `Availability zone '${configBind.availabilityZone}' must be within region '${targetRegion}'`, this);
    }

    const transit = resolveTransitEncryption(this, 'TransitEncryptionRole', props.transitEncryption, routerInputArn);

    const routerinput = new CfnRouterInput(this, 'Resource', {
      name: this.physicalName,
      maximumBitrate: props.maximumBitrate.toBps(),
      tier: (props.tier ?? RouterInputTier.INPUT_20).value,
      routingScope: props.routingScope.value,
      maintenanceConfiguration: props.maintenanceConfiguration ? {
        preferredDayTime: props.maintenanceConfiguration,
      } : {
        default: {},
      },
      configuration: configBind.config,
      transitEncryption: transit.config,
      regionName: targetRegion,
      availabilityZone: configBind.availabilityZone,
      tags: props.tags ? renderTags(props.tags) : undefined,
    });

    this.routerInputArn = routerinput.attrArn;
    this.routerInputId = routerinput.attrId;
    this.ipAddress = routerinput.attrIpAddress;
    this.createdAt = routerinput.attrCreatedAt;
    this.updatedAt = routerinput.attrUpdatedAt;

    // Without this the router input can be created before the
    // policy is attached ("access denied" at create time). Ordering the grant before the
    // router input pulls the policy in as a dependency.
    transit.grant?.applyBefore(routerinput);

    // Compute ingest endpoints for protocol-based variants
    this._endpoints = props.configuration._endpoints(routerinput.attrIpAddress);

    // Store the transit encryption secret if explicitly provided
    this.transitEncryptionSecret = props.transitEncryption?.secret;
    this.grants = RouterInputGrants.fromRouterInput(this);
  }
}

