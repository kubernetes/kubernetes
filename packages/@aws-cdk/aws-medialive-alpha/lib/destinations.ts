import type { IChannel as IMediaPackageV2Channel } from '@aws-cdk/aws-mediapackagev2-alpha';
import { Token, UnscopedValidationError } from 'aws-cdk-lib';
import type { IRole } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import type { IBucket } from 'aws-cdk-lib/aws-s3';
import type { ISecret } from 'aws-cdk-lib/aws-secretsmanager';
import type { IStringParameter } from 'aws-cdk-lib/aws-ssm';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';

/**
 * Validate a network port (1–65535), unless tokenized.
 * @internal
 */
function validatePort(port: number): void {
  if (!Token.isUnresolved(port) && (!Number.isInteger(port) || port < 1 || port > 65535)) {
    throw new UnscopedValidationError(lit`InvalidPort`, `port must be an integer between 1 and 65535, got ${JSON.stringify(port)}`);
  }
}

/**
 * A destination address (IP or host) and port for a transport-stream output.
 */
export interface TransportOutputDestinationProps {
  /** The destination address — a unicast or multicast IP, or a hostname. */
  readonly address: string;
  /** The destination port. */
  readonly port: number;
}

/**
 * Options for a URL-based output destination.
 */
export interface OutputDestinationOptions {
  /**
   * The username for accessing the downstream system.
   * @default - no credentials
   */
  readonly username?: string;
  /**
   * An SSM String Parameter holding the password for accessing the downstream system. MediaLive
   * reads it from Parameter Store at channel runtime, so the channel role is granted read access.
   * @default - no credentials
   */
  readonly password?: IStringParameter;
}

/** @internal */
function s3Url(bucket: IBucket, prefix?: string): string {
  return `s3ssl://${bucket.bucketName}${prefix ? `/${prefix}` : ''}`;
}

/**
 * A general URL-based output destination — an S3 bucket or an HTTP(S) endpoint.
 *
 * Used by HLS, MS Smooth, and CMAF Ingest output groups. Each output group's `destinations`
 * prop is typed to the destination valid for its protocol, so an invalid pairing (e.g. a UDP
 * destination on an HLS group) is a compile-time error rather than a deploy-time failure.
 */
export class OutputDestination {
  /** Deliver to an S3 bucket (auto-grants write to the channel role). */
  public static toBucket(bucket: IBucket, prefix?: string): OutputDestination {
    return new OutputDestination(s3Url(bucket, prefix), bucket);
  }
  /** Deliver to a raw URL — typically an `https://` origin (or an `s3ssl://` path). */
  public static url(url: string, options?: OutputDestinationOptions): OutputDestination {
    return new OutputDestination(url, undefined, options);
  }
  private constructor(
    private readonly destUrl: string,
    private readonly bucket?: IBucket,
    private readonly options?: OutputDestinationOptions,
  ) {}
  /** @internal */
  public _bind() {
    return { url: this.destUrl, username: this.options?.username, passwordParam: this.options?.password?.parameterName };
  }
  /** @internal */
  public _grantPermissions(role: IRole): void {
    this.bucket?.grantReadWrite(role);
    this.options?.password?.grantRead(role);
  }
}

/**
 * A destination for an Archive or Frame Capture output group — always an S3 bucket.
 */
export class S3OutputDestination {
  /** Deliver to an S3 bucket (auto-grants write to the channel role). */
  public static toBucket(bucket: IBucket, prefix?: string): S3OutputDestination {
    return new S3OutputDestination(s3Url(bucket, prefix), bucket);
  }
  /** Deliver to a raw `s3ssl://` path (escape hatch when you don't have a bucket construct). */
  public static url(url: string): S3OutputDestination {
    return new S3OutputDestination(url);
  }
  private constructor(private readonly destUrl: string, private readonly bucket?: IBucket) {}
  /** @internal */
  public _bind() {
    return { url: this.destUrl };
  }
  /** @internal */
  public _grantPermissions(role: IRole): void {
    this.bucket?.grantReadWrite(role);
  }
}

/**
 * A destination for a UDP output group — a UDP or RTP transport endpoint.
 */
export class UdpOutputDestination {
  /** Deliver over UDP — builds `udp://address:port`. */
  public static udp(props: TransportOutputDestinationProps): UdpOutputDestination {
    validatePort(props.port);
    return new UdpOutputDestination(`udp://${props.address}:${props.port}`);
  }
  /** Deliver over RTP — builds `rtp://address:port`. */
  public static rtp(props: TransportOutputDestinationProps): UdpOutputDestination {
    validatePort(props.port);
    return new UdpOutputDestination(`rtp://${props.address}:${props.port}`);
  }
  /** Deliver to a raw transport URL (escape hatch). */
  public static url(url: string): UdpOutputDestination {
    return new UdpOutputDestination(url);
  }
  private constructor(private readonly destUrl: string) {}
  /** @internal */
  public _bind() {
    return { url: this.destUrl };
  }
  /** @internal */
  public _grantPermissions(_role: IRole): void {}
}

/**
 * A destination for an RTMP output group. Use the static factory methods to create.
 */
export abstract class RtmpDestination {
  /**
   * Create an RTMP destination.
   * @param url The RTMP endpoint URL (e.g. rtmp://host/appname).
   * @param streamName The stream name (stream key).
   * @param options Optional credentials.
   */
  public static url(url: string, streamName: string, options?: OutputDestinationOptions): RtmpDestination {
    return new UrlRtmpDestination(url, streamName, options);
  }

  /** @internal */
  public abstract _bind(): { url: string; streamName: string; username?: string; passwordParam?: string };

  /** @internal - Grant the channel role read access to the password parameter, if configured. */
  public _grantPermissions(_role: IRole): void {}
}

/** @internal */
class UrlRtmpDestination extends RtmpDestination {
  constructor(
    private readonly url: string,
    private readonly streamName: string,
    private readonly options?: OutputDestinationOptions,
  ) { super(); }
  public _bind() {
    return {
      url: this.url,
      streamName: this.streamName,
      username: this.options?.username,
      passwordParam: this.options?.password?.parameterName,
    };
  }
  public override _grantPermissions(role: IRole): void {
    this.options?.password?.grantRead(role);
  }
}

/**
 * SRT caller destination properties.
 */
export interface SrtCallerDestinationProps {
  /** The address (IP or host) of the SRT listener to connect to. */
  readonly address: string;
  /** The port of the SRT listener to connect to. */
  readonly port: number;
  /**
   * The stream ID for the SRT connection.
   * @default - no stream ID
   */
  readonly streamId?: string;
  /**
   * The Secrets Manager secret containing the encryption passphrase.
   *
   * [disable-awslint:prefer-ref-interface]
   */
  readonly encryptionPassphraseSecret: ISecret;
}

/**
 * Options for a URL-based SRT caller destination (`SrtDestination.callerUrl`).
 */
export interface SrtCallerUrlOptions {
  /**
   * The stream ID for the SRT connection.
   * @default - no stream ID
   */
  readonly streamId?: string;
  /**
   * The Secrets Manager secret containing the encryption passphrase.
   *
   * [disable-awslint:prefer-ref-interface]
   */
  readonly encryptionPassphraseSecret: ISecret;
}

/**
 * SRT listener destination properties.
 *
 * In listener mode, MediaLive opens a socket on `listenerPort` and waits for the downstream
 * system to connect. The downstream system needs the channel's outbound IP and this port —
 * AWS does not require (or use) a destination URL in listener mode.
 */
export interface SrtListenerDestinationProps {
  /** The port that MediaLive will listen on. AWS reserves the range 5000–5200 for SRT listener output. */
  readonly listenerPort: number;
  /**
   * The stream ID for the SRT connection.
   * @default - no stream ID
   */
  readonly streamId?: string;
  /**
   * The Secrets Manager secret containing the encryption passphrase.
   *
   * [disable-awslint:prefer-ref-interface]
   */
  readonly encryptionPassphraseSecret: ISecret;
}

/**
 * A destination for an SRT output group. Use the static factory methods to create.
 */
export abstract class SrtDestination {
  /** Create a caller-mode SRT destination. MediaLive connects to the remote listener. */
  public static caller(props: SrtCallerDestinationProps): SrtDestination {
    validatePort(props.port);
    return new SrtCallerDestination(`srt://${props.address}:${props.port}`, props.streamId, props.encryptionPassphraseSecret);
  }
  /**
   * Create a caller-mode SRT destination from a full SRT URL.
   *
   * Use this when you already have a URL rather than a separate host and port — for example a
   * MediaConnect Router Input's ingest endpoint (`routerInput.endpoints[0].url`).
   */
  public static callerUrl(url: string, options: SrtCallerUrlOptions): SrtDestination {
    return new SrtCallerDestination(url, options.streamId, options.encryptionPassphraseSecret);
  }
  /** Create a listener-mode SRT destination. MediaLive listens for incoming connections. */
  public static listener(props: SrtListenerDestinationProps): SrtDestination {
    return new SrtListenerDestination(props);
  }

  /** @internal */
  public abstract _bind(): { url?: string; connectionMode?: string;
    encryptionPassphraseSecretArn?: string;
    listenerPort?: number; streamId?: string; };

  /** @internal - Grant permissions for encryption secrets. */
  public _grantPermissions(_role: IRole): void {}
}

/** @internal */
class SrtCallerDestination extends SrtDestination {
  constructor(
    private readonly url: string,
    private readonly streamId: string | undefined,
    private readonly encryptionPassphraseSecret: ISecret,
  ) { super(); }
  public _bind() {
    return {
      url: this.url,
      connectionMode: 'CALLER',
      streamId: this.streamId,
      encryptionPassphraseSecretArn: this.encryptionPassphraseSecret.secretArn,
    };
  }
  public override _grantPermissions(role: IRole): void {
    this.encryptionPassphraseSecret.grantRead(role);
  }
}

/** @internal */
class SrtListenerDestination extends SrtDestination {
  constructor(private readonly props: SrtListenerDestinationProps) {
    super();
    if (!Token.isUnresolved(props.listenerPort) && (!Number.isInteger(props.listenerPort)
      || props.listenerPort < 5000 || props.listenerPort > 5200)) {
      throw new UnscopedValidationError(lit`SrtListenerPort`, `SRT listener port must be an integer between 5000 and 5200, got ${JSON.stringify(props.listenerPort)}`);
    }
  }
  public _bind() {
    return {
      connectionMode: 'LISTENER',
      listenerPort: this.props.listenerPort,
      streamId: this.props.streamId,
      encryptionPassphraseSecretArn: this.props.encryptionPassphraseSecret.secretArn,
    };
  }
  public override _grantPermissions(role: IRole): void {
    this.props.encryptionPassphraseSecret.grantRead(role);
  }
}

/**
 * The pipeline endpoint for a MediaPackage V2 destination.
 */
export class MediaPackageV2EndpointId {
  /** Pipeline 0 endpoint */
  public static readonly ENDPOINT_1 = new MediaPackageV2EndpointId('ENDPOINT_1');
  /** Pipeline 1 endpoint */
  public static readonly ENDPOINT_2 = new MediaPackageV2EndpointId('ENDPOINT_2');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MediaPackageV2EndpointId {
    return new MediaPackageV2EndpointId(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * A MediaPackage V2 destination for a MediaLive output group.
 * Use the static factory method to create.
 */
export abstract class MediaPackageV2Destination {
  /**
   * Create a MediaPackage V2 destination.
   *
   * The region is resolved from the channel's `region` property. Import the MediaPackage V2
   * channel with its region (e.g. via `fromChannelAttributes`) for cross-region delivery.
   *
   * @param channel The MediaPackage V2 channel to send output to.
   * @param endpointId The pipeline endpoint to send to. When omitted, `channelEndpointId` and the
   * region are left unset and MediaLive maps the pipeline automatically (same-region only).
   */
  public static channel(channel: IMediaPackageV2Channel, endpointId?: MediaPackageV2EndpointId): MediaPackageV2Destination {
    return new DefaultMediaPackageV2Destination(channel, endpointId);
  }

  /** @internal */
  public abstract _bind(): { channelName: string; channelGroup: string; channelEndpointId?: string; mediaPackageRegionName?: string };

  /** @internal - Grant ingest permissions to the channel role. */
  public _grantPermissions(_role: IRole): void {}
}

/** @internal */
class DefaultMediaPackageV2Destination extends MediaPackageV2Destination {
  constructor(
    private readonly mpChannel: IMediaPackageV2Channel,
    private readonly endpointId?: MediaPackageV2EndpointId,
  ) { super(); }
  public _bind() {
    return {
      channelName: this.mpChannel.channelName,
      channelGroup: this.mpChannel.channelGroupName,
      channelEndpointId: this.endpointId?.value,
      // MediaLive requires an endpoint id whenever a region is set, so only emit the region
      // when an endpoint is provided. Omitting both selects same-region pipeline auto-mapping.
      mediaPackageRegionName: this.endpointId !== undefined ? this.mpChannel.region : undefined,
    };
  }
  public override _grantPermissions(role: IRole): void {
    this.mpChannel.grants.ingest(role);
  }
}

/**
 * Per-pipeline settings for a MediaConnect Router output destination.
 *
 * Today this carries only transit encryption, but it is a struct so that future per-pipeline
 * MediaConnect Router settings can be added without an API break.
 */
export interface MediaConnectRouterPipelineConfig {
  /**
   * A Secrets Manager secret holding the transit-encryption passphrase. When set, the pipeline
   * uses `SECRETS_MANAGER` encryption; the channel role is granted read access to the secret.
   *
   * @default - AUTOMATIC (service-managed) transit encryption
   */
  readonly encryptionSecret?: ISecret;
}

/**
 * Per-pipeline settings for `MediaConnectRouterSettings.perPipeline()`.
 *
 * MediaLive maps a channel's pipelines to MediaConnect Router output destinations positionally;
 * the console labels them "Destination A" (pipeline 0) and "Destination B" (pipeline 1). An
 * omitted pipeline uses AUTOMATIC transit encryption.
 */
export interface MediaConnectRouterPerPipelineSettings {
  /**
   * Settings for pipeline 0 ("Destination A" in the MediaLive console).
   * @default - AUTOMATIC transit encryption
   */
  readonly pipeline0?: MediaConnectRouterPipelineConfig;
  /**
   * Settings for pipeline 1 ("Destination B" in the console). STANDARD channels only —
   * SINGLE_PIPELINE channels have no second pipeline.
   * @default - AUTOMATIC transit encryption
   */
  readonly pipeline1?: MediaConnectRouterPipelineConfig;
}

/** @internal */
function renderRouterPipeline(config?: MediaConnectRouterPipelineConfig): CfnChannel.MediaConnectRouterOutputDestinationSettingsProperty {
  return config?.encryptionSecret !== undefined
    ? { encryptionType: 'SECRETS_MANAGER', secretArn: config.encryptionSecret.secretArn }
    : { encryptionType: 'AUTOMATIC' };
}

/**
 * Transit-encryption settings for a MediaConnect Router output group, applied per channel pipeline.
 *
 * Omit the output group's `routerSettings` entirely for AUTOMATIC (service-managed) encryption on
 * every pipeline. Use `shared()` to apply one secret across all pipelines, or `perPipeline()` to
 * control each pipeline independently.
 */
export abstract class MediaConnectRouterSettings {
  /**
   * Apply the same settings to every pipeline.
   *
   * @param settings the settings to apply to all pipelines. Omit `encryptionSecret` for AUTOMATIC.
   */
  public static shared(settings: MediaConnectRouterPipelineConfig = {}): MediaConnectRouterSettings {
    return new SharedMediaConnectRouterSettings(settings);
  }

  /**
   * Configure each pipeline independently. An omitted pipeline uses AUTOMATIC encryption.
   */
  public static perPipeline(settings: MediaConnectRouterPerPipelineSettings): MediaConnectRouterSettings {
    return new PerPipelineMediaConnectRouterSettings(settings);
  }

  /**
   * Render the per-pipeline destination settings for the given pipeline count (1 for
   * SINGLE_PIPELINE, 2 for STANDARD).
   * @internal
   */
  public abstract _bind(pipelineCount: number): CfnChannel.MediaConnectRouterOutputDestinationSettingsProperty[];

  /**
   * The secrets referenced by these settings, for granting read access to the channel role.
   * @internal
   */
  public abstract _secrets(): ISecret[];
}

/** @internal */
class SharedMediaConnectRouterSettings extends MediaConnectRouterSettings {
  constructor(private readonly settings: MediaConnectRouterPipelineConfig) { super(); }

  public _bind(pipelineCount: number): CfnChannel.MediaConnectRouterOutputDestinationSettingsProperty[] {
    const entry = renderRouterPipeline(this.settings);
    return Array.from({ length: pipelineCount }, () => entry);
  }

  public _secrets(): ISecret[] {
    return this.settings.encryptionSecret !== undefined ? [this.settings.encryptionSecret] : [];
  }
}

/** @internal */
class PerPipelineMediaConnectRouterSettings extends MediaConnectRouterSettings {
  constructor(private readonly settings: MediaConnectRouterPerPipelineSettings) { super(); }

  public _bind(pipelineCount: number): CfnChannel.MediaConnectRouterOutputDestinationSettingsProperty[] {
    if (pipelineCount === 1 && this.settings.pipeline1 !== undefined) {
      throw new UnscopedValidationError(
        lit`MediaConnectRouterPipelineOne`,
        'pipeline1 settings are not valid on a SINGLE_PIPELINE channel; use ChannelClass.STANDARD or remove pipeline1',
      );
    }
    const result = [renderRouterPipeline(this.settings.pipeline0)];
    if (pipelineCount === 2) {
      result.push(renderRouterPipeline(this.settings.pipeline1));
    }
    return result;
  }

  public _secrets(): ISecret[] {
    return [this.settings.pipeline0?.encryptionSecret, this.settings.pipeline1?.encryptionSecret]
      .filter((s): s is ISecret => s !== undefined);
  }
}
