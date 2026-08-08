import type { Bitrate, IResource } from 'aws-cdk-lib';
import { ArnFormat, Resource, Lazy, Names, Duration, Stack, Token, Fn, UnscopedValidationError, ValidationError } from 'aws-cdk-lib';
import type { MetricOptions } from 'aws-cdk-lib/aws-cloudwatch';
import { Metric, Unit } from 'aws-cdk-lib/aws-cloudwatch';
import { CfnFlow } from 'aws-cdk-lib/aws-mediaconnect';
import type { IFlowRef, FlowReference } from 'aws-cdk-lib/aws-mediaconnect';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import type { IFlowOutput, OutputConfiguration } from './flow-output';
import { FlowOutput } from './flow-output';
import type { SourceConfiguration } from './flow-source-configuration';
import { SourceProtocol } from './flow-source-configuration';
import { FlowGrants } from './mediaconnect-grants.generated';
import type { VpcInterfaceConfig, MaintenanceDay, Framerate, PixelAspectRatio } from './shared';
import { FailoverMode, NetworkInterface, State, toTitleCase, validateMaintenanceTime } from './shared';

/**
 * Interface for Flow
 */
export interface IFlow extends IResource, IFlowRef {

  /**
   * The Amazon Resource Name (ARN) of the flow.
   *
   * @attribute
   */
  readonly flowArn: string;

  /**
   * The Amazon Resource Name (ARN) of the source defined on the flow.
   *
   * @attribute
   */
  readonly sourceArn: string;

  /**
   * The IP address that the flow uses to send outbound content.
   *
   * @attribute
   */
  readonly egressIp: string;

  /**
   * The IP address that the flow listens on for incoming content.
   *
   * Available for listener-style source protocols (RTP, RTP-FEC, RIST, SRT listener,
   * Zixi push). Accessing this on SRT caller, entitlement, gateway bridge, router,
   * CDI, JPEG XS, or imported flows throws — those sources don't expose a listening
   * IP address.
   *
   * @attribute
   */
  readonly sourceIngestIp: string;

  /**
   * The port that the flow listens on for incoming content.
   *
   * Available for the same listener-style source protocols as `sourceIngestIp`.
   * Accessing this on SRT caller, entitlement, gateway bridge, router, CDI, JPEG XS,
   * or imported flows throws.
   *
   * @attribute
   */
  readonly sourceIngestPort: string;

  /**
   * The full ingest URL for the flow source, combining protocol, IP, and port.
   * For example: `srt://203.0.113.10:5000`
   *
   * Available for listener-style source protocols (RTP, RTP-FEC, RIST, SRT listener,
   * Zixi push) where the flow listens for an upstream sender. Accessing this on
   * SRT caller, entitlement, gateway bridge, router, CDI, JPEG XS, or imported
   * flows throws — those sources don't expose a single host:port ingest URL.
   */
  readonly sourceIngestUrl: string;

  /**
   * The Availability Zone that the flow was created in.
   *
   * @attribute
   */
  readonly flowAvailabilityZone?: string;

  /**
   * Failover Configuration for flow
   */
  readonly isFailoverEnabled?: boolean;

  /**
   * Collection of grant methods for this flow.
   */
  readonly grants: FlowGrants;

  /**
   * Create a CloudWatch metric for this flow.
   */
  metric(metricName: string, props?: MetricOptions): Metric;

  /**
   * Metric for the bitrate of content ingested into the flow.
   * @default - average over 60 seconds
   */
  metricSourceBitrate(props?: MetricOptions): Metric;

  /**
   * Metric for packets that were not recovered by the flow source.
   * @default - sum over 60 seconds
   */
  metricSourceNotRecoveredPackets(props?: MetricOptions): Metric;

  /**
   * Metric for the total number of packets received by the flow source.
   * @default - sum over 60 seconds
   */
  metricSourceTotalPackets(props?: MetricOptions): Metric;

  /**
   * Metric indicating which source is selected for ingest when using `Failover` failover mode.
   * @default - maximum over 60 seconds
   */
  metricSourceSelected(props?: MetricOptions): Metric;

  /**
   * Metric indicating the connection state of the source (1 connected, 0 disconnected).
   * Applies only to Zixi, SRT, and RIST sources.
   * @default - minimum over 60 seconds
   */
  metricSourceConnected(props?: MetricOptions): Metric;

  /**
   * Metric for the number of times the source transitioned from connected to disconnected.
   * @default - sum over 60 seconds
   */
  metricSourceDisconnections(props?: MetricOptions): Metric;

  /**
   * Metric for the number of packets lost during transit, measured before any error correction.
   * @default - sum over 60 seconds
   */
  metricSourceDroppedPackets(props?: MetricOptions): Metric;

  /**
   * Metric for the percentage of packets lost during transit, even if they were recovered.
   * @default - average over 60 seconds
   */
  metricSourcePacketLossPercent(props?: MetricOptions): Metric;

  /**
   * Metric for the round-trip time between the source and MediaConnect.
   * Applies only to RIST, Zixi, and SRT sources.
   * @default - average over 60 seconds
   */
  metricSourceRoundTripTime(props?: MetricOptions): Metric;

  /**
   * Metric for the current network jitter of the source, measured in milliseconds.
   * @default - average over 60 seconds
   */
  metricSourceJitter(props?: MetricOptions): Metric;

  /**
   * Add an output to this flow
   */
  addOutput(id: string, options: AddFlowOutputOptions): IFlowOutput;
}

/**
 * Options for adding an output to a flow.
 */
export interface AddFlowOutputOptions {
  /**
   * The output configuration.
   */
  readonly output: OutputConfiguration;
  /**
   * The name of the flow output.
   * @default - auto-generated
   */
  readonly flowOutputName?: string;
  /**
   * A description of the output.
   * @default - no description
   */
  readonly description?: string;
  /**
   * Whether the output is enabled.
   * @default State.ENABLED
   */
  readonly outputStatus?: State;
}

/**
 * Options for Flow Size
 */
export class FlowSize {
  /**
   * Medium flow size. The default — standard transport stream distribution; does not
   * support NDI, CDI, or JPEG XS.
   *
   * @see https://docs.aws.amazon.com/mediaconnect/latest/ug/flow-sizes-capabilities.html
   */
  public static readonly MEDIUM = new FlowSize('MEDIUM');
  /**
   * Large flow size. Required when the flow uses NDI sources or outputs. Supports
   * transport stream distribution plus NDI; does not support CDI or JPEG XS.
   *
   * @see https://docs.aws.amazon.com/mediaconnect/latest/ug/flow-sizes-capabilities.html
   */
  public static readonly LARGE = new FlowSize('LARGE');
  /**
   * Option for large 4x flow size. Required for JPEG XS and CDI protocols. Does not support transport streams or NDI.
   */
  public static readonly LARGE_4X = new FlowSize('LARGE_4X');

  /**
   * Use a custom flow size value
   * @param value The flow size string value
   */
  public static of(value: string): FlowSize {
    return new FlowSize(value);
  }

  /** @param value The flow size string value */
  private constructor(public readonly value: string) {}

  /** Returns the string value */
  public toString(): string {
    return this.value;
  }
}

/**
 * Configuration for scheduled maintenance windows.
 */
export interface MaintenanceWindow {
  /** A day of a week when the maintenance will happen. */
  readonly day: MaintenanceDay;
  /** The maintenance start time in UTC, 24-hour HH:MM format. Minutes must be 00 (e.g., '02:00', '13:00'). */
  readonly time: string;
}

/**
 * Options for merge-mode source failover.
 */
export interface MergeFailoverOptions {
  /**
   * Whether failover is enabled. Set to `State.DISABLED` to keep the configuration
   * on the flow without switching failover on.
   *
   * @default State.ENABLED
   */
  readonly state?: State;
  /**
   * Search window time to look for SMPTE 2022-7 packets. A larger recovery window
   * means a longer delay in transmitting the stream, but more room for error
   * correction. A smaller window means a shorter delay but less room for correction.
   *
   * Valid range: 100–15000 ms.
   *
   * @default - 200 ms
   */
  readonly recoveryWindow?: Duration;
}

/**
 * Options for switchover-mode source failover.
 */
export interface FailoverFailoverOptions {
  /**
   * Whether failover is enabled. Set to `State.DISABLED` to keep the configuration
   * on the flow without switching failover on.
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
 * Source failover configuration for a flow.
 *
 * MediaConnect supports two failover modes:
 * - {@link FailoverConfig.merge} combines two binary-identical sources into a single
 *   stream, recovering lost packets from the other source (SMPTE 2022-7).
 * - {@link FailoverConfig.failover} switches between a primary and backup source when
 *   the active source stops receiving data.
 */
export class FailoverConfig {
  /**
   * Configure merge-mode failover. Requires two binary-identical sources.
   */
  public static merge(options: MergeFailoverOptions = {}): FailoverConfig {
    if (options.recoveryWindow) {
      const ms = options.recoveryWindow.toMilliseconds();
      if (ms < 100 || ms > 15000) {
        throw new UnscopedValidationError(lit`RecoveryWindowRange`, `Recovery window must be between 100 and 15000 ms, got ${ms}`);
      }
    }
    return new FailoverConfig(FailoverMode.MERGE, options.state ?? State.ENABLED, options.recoveryWindow, undefined);
  }

  /**
   * Configure switchover-mode failover. The flow swaps to the backup source when the
   * primary source stops receiving data.
   */
  public static failover(options: FailoverFailoverOptions = {}): FailoverConfig {
    return new FailoverConfig(FailoverMode.FAILOVER, options.state ?? State.ENABLED, undefined, options.primarySource);
  }

  private constructor(
    private readonly _failoverMode: FailoverMode,
    private readonly _state: State,
    private readonly _recoveryWindow: Duration | undefined,
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
   * Called when the configuration is bound to a Flow.
   * @internal
   */
  public _bind(): CfnFlow.FailoverConfigProperty {
    return {
      failoverMode: this._failoverMode.value,
      recoveryWindow: this._recoveryWindow?.toMilliseconds(),
      sourcePriority: this._primarySource ? { primarySource: this._primarySource } : undefined,
      state: this._state,
    };
  }
}

/**
 * Configures a source monitoring metric.
 */
export interface MonitoringMetric {
  /**
   * Whether the metric is enabled or disabled. The threshold is persisted by the
   * service either way, so you can toggle the metric on and off without losing
   * the threshold value.
   */
  readonly state: State;
  /**
   * Threshold in seconds that triggers the metric's alert. Valid range: 10–60 seconds.
   */
  readonly threshold: Duration;
}

/**
 * Source monitoring settings.
 */
export interface SourceMonitoringConfig {
  /**
   * Indicates whether content quality analysis is enabled or disabled.
   *
   * @default - content quality analysis is disabled
   */
  readonly contentQualityAnalysisState?: State;
  /**
   * The current state of the thumbnail monitoring.
   *
   * @default - thumbnail monitoring is disabled
   */
  readonly thumbnailState?: State;
  /**
   * Silent-audio detection on the source.
   *
   * @default - silent audio monitoring is not configured
   */
  readonly silentAudio?: MonitoringMetric;
  /**
   * Black-frames detection on the source.
   *
   * @default - black frames monitoring is not configured
   */
  readonly blackFrames?: MonitoringMetric;
  /**
   * Frozen-frames detection on the source.
   *
   * @default - frozen frames monitoring is not configured
   */
  readonly frozenFrames?: MonitoringMetric;
}

/**
 * NDI Configuration
 */
export interface NdiDiscoveryServerConfig {
  /**
   * The unique network address of the NDI discovery server.
   */
  readonly discoveryServerAddress: string;
  /**
   * The port for the NDI discovery server. Defaults to 5959 if a custom port isn't specified.
   *
   * @default - undefined; when omitted, MediaConnect applies 5959 at deploy time
   */
  readonly discoveryServerPort?: number;
  /**
   * The VPC interface that the NDI discovery server uses to reach the flow.
   */
  readonly vpcInterface: VpcInterfaceConfig;
}

/**
 * Configuration for NDI Config
 */
export interface NdiConfig {
  /**
   * A prefix for the names of the NDI sources that the flow creates.
   * If a custom name isn't specified, MediaConnect generates a unique 12-character ID as the prefix.
   *
   * @default - MediaConnect generates a unique 12-character ID
   */
  readonly machineName?: string;
  /**
   * Specifies the configuration settings for individual NDI discovery servers.
   * A maximum of 3 servers is allowed.
   *
   * @default - no NDI discovery servers
   */
  readonly ndiDiscoveryServers?: NdiDiscoveryServerConfig[];
  /**
   * A setting that controls whether NDI sources or outputs can be used in the flow.
   * Must be ENABLED for the flow to support NDI sources or outputs.
   *
   * @default State.DISABLED
   */
  readonly ndiState?: State;
}

/**
 * Encoding profiles supported when transcoding an NDI source to a transport stream.
 */
export enum EncodingProfile {
  /**
   * H.264 profile tuned for contribution workflows — higher quality, higher bitrate.
   */
  CONTRIBUTION_H264_DEFAULT = 'CONTRIBUTION_H264_DEFAULT',
  /**
   * H.264 profile tuned for distribution workflows — lower bitrate, wider compatibility.
   */
  DISTRIBUTION_H264_DEFAULT = 'DISTRIBUTION_H264_DEFAULT',
}

/**
 * Encoding configuration applied to an NDI source when transcoding it to a transport
 * stream for downstream distribution.
 */
export interface EncodingConfig {
  /**
   * The encoding profile to use when transcoding the NDI source content to a transport
   * stream. You can change this value while the flow is running.
   *
   * @default - the MediaConnect service default
   */
  readonly encodingProfile?: EncodingProfile;

  /**
   * The maximum video bitrate to use when transcoding the NDI source to a transport stream.
   *
   * The supported range is 10 Mbps – 50 Mbps.
   *
   * @default - undefined; when omitted, MediaConnect applies 20 Mbps at deploy time
   */
  readonly videoMaxBitrate?: Bitrate;
}

/**
 * Option for Colorimetry
 */
export class Colorimetry {
  /** Option for BT601 */
  public static readonly BT601 = new Colorimetry('BT601');
  /** Option for BT709 */
  public static readonly BT709 = new Colorimetry('BT709');
  /** Option for BT2020 */
  public static readonly BT2020 = new Colorimetry('BT2020');
  /** Option for BT2100 */
  public static readonly BT2100 = new Colorimetry('BT2100');
  /** Option for ST2065-1 */
  public static readonly ST2065_1 = new Colorimetry('ST2065-1');
  /** Option for ST2065-3 */
  public static readonly ST2065_3 = new Colorimetry('ST2065-3');
  /** Option for XYZ */
  public static readonly XYZ = new Colorimetry('XYZ');

  /** Use a custom colorimetry value */
  public static of(value: string): Colorimetry { return new Colorimetry(value); }
  /** @param value The colorimetry string value */
  private constructor(public readonly value: string) {}
  /** Returns the string value */
  public toString(): string { return this.value; }
}

/**
 * Options for Video Range
 */
export class VideoRange {
  /** Option for Narrow */
  public static readonly NARROW = new VideoRange('NARROW');
  /** Option for Full */
  public static readonly FULL = new VideoRange('FULL');
  /** Option for Full Protect */
  public static readonly FULLPROTECT = new VideoRange('FULLPROTECT');

  /** Use a custom video range value */
  public static of(value: string): VideoRange { return new VideoRange(value); }
  /** @param value The video range string value */
  private constructor(public readonly value: string) {}
  /** Returns the string value */
  public toString(): string { return this.value; }
}

/**
 * Options for Scan Mode
 */
export class ScanMode {
  /** Option for Progressive */
  public static readonly PROGRESSIVE = new ScanMode('progressive');
  /** Option for Interlace */
  public static readonly INTERLACE = new ScanMode('interlace');
  /** Option for Progressive Segmented Frame */
  public static readonly PROGRESSIVE_SEGMENTED_FRAME = new ScanMode('progressive-segmented-frame');

  /** Use a custom scan mode value */
  public static of(value: string): ScanMode { return new ScanMode(value); }
  /** @param value The scan mode string value */
  private constructor(public readonly value: string) {}
  /** Returns the string value */
  public toString(): string { return this.value; }
}

/**
 * Options for Tcs
 */
export class Tcs {
  /** Option for SDR */
  public static readonly SDR = new Tcs('SDR');
  /** Option for PQ */
  public static readonly PQ = new Tcs('PQ');
  /** Option for HLG */
  public static readonly HLG = new Tcs('HLG');
  /** Option for Linear */
  public static readonly LINEAR = new Tcs('LINEAR');
  /** Option for BT2100LINPQ */
  public static readonly BT2100LINPQ = new Tcs('BT2100LINPQ');
  /** Option for BT2100LINHLG */
  public static readonly BT2100LINHLG = new Tcs('BT2100LINHLG');
  /** Option for ST2065-1 */
  public static readonly ST2065_1 = new Tcs('ST2065-1');
  /** Option for ST428-1 */
  public static readonly ST428_1 = new Tcs('ST428-1');
  /** Option for Density */
  public static readonly DENSITY = new Tcs('DENSITY');

  /** Use a custom TCS value */
  public static of(value: string): Tcs { return new Tcs(value); }
  /** @param value The TCS string value */
  private constructor(public readonly value: string) {}
  /** Returns the string value */
  public toString(): string { return this.value; }
}

/**
 * Options for FMTP Video
 */
export interface FmtpVideo {
  /**
   * The format used for the representation of color.
   *
   * @default - no colorimetry specified
   */
  readonly colorimetry?: Colorimetry;
  /**
   * The frame rate for the video stream.
   *
   * @default - no frame rate specified
   */
  readonly exactFramerate?: Framerate;
  /**
   * The pixel aspect ratio (PAR) of the video — the shape of a single pixel, not the
   * display aspect ratio. Use `PixelAspectRatio.SQUARE` for most modern content.
   *
   * @default - no pixel aspect ratio specified
   */
  readonly par?: PixelAspectRatio;
  /**
   * The encoding range of the video.
   *
   * @default - no video range specified
   */
  readonly videoRange?: VideoRange;
  /**
   * The type of compression that was used to smooth the video's appearance.
   *
   * @default - no scan mode specified
   */
  readonly scanMode?: ScanMode;
  /**
   * The transfer characteristic system (TCS) that is used in the video.
   *
   * @default - no TCS specified
   */
  readonly tcs?: Tcs;
}

/**
 * Audio Stream Order Options
 */
export class AudioStreamOrderOptions {
  /** Mono */
  public static readonly MONO = new AudioStreamOrderOptions('SMPTE2110.(M)');
  /** Dual Mono */
  public static readonly DUAL_MONO = new AudioStreamOrderOptions('SMPTE2110.(DM)');
  /** Standard Stereo */
  public static readonly STANDARD_STEREO = new AudioStreamOrderOptions('SMPTE2110.(ST)');
  /** Left Right Matrix Stereo */
  public static readonly LTRT_MATRIX_STEREO = new AudioStreamOrderOptions('SMPTE2110.(LtRt)');
  /** 5.1 Surround */
  public static readonly SURROUND_5_1 = new AudioStreamOrderOptions('SMPTE2110.(51)');
  /** 7.1 Surround */
  public static readonly SURROUND_7_1 = new AudioStreamOrderOptions('SMPTE2110.(71)');
  /** 22.2 Surround */
  public static readonly SURROUND_22_2 = new AudioStreamOrderOptions('SMPTE2110.(222)');
  /** One SDI Audio Group */
  public static readonly ONE_SDI_AUDIO_GROUP = new AudioStreamOrderOptions('SMPTE2110.(SGRP)');

  /** Use a custom audio stream order value */
  public static of(value: string): AudioStreamOrderOptions { return new AudioStreamOrderOptions(value); }
  /** @param value The audio stream order string value */
  private constructor(public readonly value: string) {}
  /** Returns the string value */
  public toString(): string { return this.value; }
}

/**
 * Options for Media Stream Type
 *
 * @internal
 */
enum MediaStreamType {
  /**
   * Option for Video
   */
  VIDEO='video',
  /**
   * Option for Audio
   */
  AUDIO='audio',
  /**
   * Option for ancillary data
   */
  ANCILLARY_DATA='ancillary-data',
}

/**
 * Options for Media Video format
 */
export class MediaVideoFormat {
  /** Option for 2160p */
  public static readonly UHD_2160P = new MediaVideoFormat('2160p');
  /** Option for 1080p */
  public static readonly HD_1080P = new MediaVideoFormat('1080p');
  /** Option for 1080i */
  public static readonly HD_1080I = new MediaVideoFormat('1080i');
  /** Option for 720p */
  public static readonly HD_720P = new MediaVideoFormat('720p');
  /** Option for 480p */
  public static readonly SD_480P = new MediaVideoFormat('480p');

  /** Use a custom video format value */
  public static of(value: string): MediaVideoFormat { return new MediaVideoFormat(value); }
  /** @param value The video format string value */
  private constructor(public readonly value: string) {}
  /** Returns the string value */
  public toString(): string { return this.value; }
}

/**
 * Base configuration for Media Stream.
 */
export interface MediaStreamBase {
  /**
   * A unique identifier for the media stream.
   */
  readonly mediaStreamId: number;
  /**
   * A name that helps you distinguish one media stream from another.
   */
  readonly mediaStreamName: string;
  /**
   * A description that can help you quickly identify what your media stream is used for.
   *
   * @default - no description
   */
  readonly description?: string;
}

/**
 * A media stream represents one component of your content, such as video, audio, or ancillary data.
 * After you add a media stream to your flow, you can associate it with sources and outputs that use
 * the ST 2110 JPEG XS or CDI protocol.
 */
export interface MediaStreamVideo extends MediaStreamBase {
  /**
   * Attributes that are related to the media stream.
   */
  readonly fmtp: FmtpVideo;
  /**
   * The sample rate for the stream. This value is measured in Hz.
   *
   * @default - default clock rate for the media stream type
   */
  readonly clockRate?: number;
  /**
   * The format type number (sometimes referred to as RTP payload type) of the media stream.
   * MediaConnect assigns this value to the media stream. For ST 2110 JPEG XS outputs, you need to provide this value to the receiver.
   *
   * @default - assigned by MediaConnect
   */
  readonly fmt?: number;
  /**
   * The resolution of the video.
   */
  readonly videoFormat: MediaVideoFormat;
}

/**
 * A media stream represents one component of your content, such as video, audio, or ancillary data.
 * After you add a media stream to your flow, you can associate it with sources and outputs that use
 * the ST 2110 JPEG XS or CDI protocol.
 */
export interface MediaStreamAudio extends MediaStreamBase {
  /**
   * The format of the audio channel.
   *
   * @default - the MediaConnect service default
   */
  readonly channelOrder?: AudioStreamOrderOptions;
  /**
   * The sample rate for the stream. This value is measured in Hz.
   *
   * @default - default clock rate for the media stream type
   */
  readonly clockRate?: number;
  /**
   * The format type number (sometimes referred to as RTP payload type) of the media stream.
   * MediaConnect assigns this value to the media stream. For ST 2110 JPEG XS outputs, you need to provide this value to the receiver.
   *
   * @default - assigned by MediaConnect
   */
  readonly fmt?: number;
  /**
   * The audio language, in a format that is recognized by the receiver.
   *
   * @default - no language specified
   */
  readonly lang?: string;
}

/**
 * A media stream represents one component of your content, such as video, audio, or ancillary data.
 * After you add a media stream to your flow, you can associate it with sources and outputs that use
 * the ST 2110 JPEG XS or CDI protocol.
 */
export interface MediaStreamAncillaryData extends MediaStreamBase { }

/**
 * Properties for the Flow
 */
export interface FlowProps {
  /**
   * The name of the flow.
   *
   * @default - autogenerated
   */
  readonly flowName?: string;

  /**
   * The Availability Zone that you want to create the flow in. These options are limited to the Availability Zones within the current AWS Region.
   *
   * @default - chosen by MediaConnect
   */
  readonly availabilityZone?: string;

  /**
   * Determines the processing capacity and feature set of the flow. Set this optional parameter to LARGE if you want to enable NDI outputs on the flow.
   *
   * @default - undefined; when omitted, MediaConnect applies FlowSize.MEDIUM at deploy time
   */
  readonly flowSize?: FlowSize;

  /**
   * The maintenance window for the flow.
   *
   * @default - chosen by MediaConnect
   */
  readonly maintenanceConfiguration?: MaintenanceWindow;

  /**
   * The media streams that are associated with the flow. After you associate a media stream with a source, you can also associate it with outputs on the flow.
   *
   * @default No Media Streams
   */
  readonly mediaStreams?: MediaStream[];

  /**
   * Specifies the configuration settings for a flow's NDI source or output. Required when the flow includes an NDI source or output.
   *
   * @default No NDI Configuration applied
   */
  readonly ndiConfig?: NdiConfig;

  /**
   * Encoding configuration applied to the NDI source when transcoding it to a transport
   * stream for downstream distribution.
   *
   * Required when the flow source uses the NDI SpeedHQ protocol
   * (see {@link SourceConfiguration.ndi}).
   *
   * @default - no encoding config; required for NDI sources
   */
  readonly encodingConfig?: EncodingConfig;

  /**
   * The settings for the source that you want to use for the new flow.
   */
  readonly source: SourceConfiguration;

  /**
   * The settings for source failover.
   *
   * @default No source failover configuration applied
   */
  readonly sourceFailoverConfig?: FailoverConfig;

  /**
   * The settings for source monitoring.
   *
   * @default No source monitoring configuration applied
   */
  readonly sourceMonitoringConfig?: SourceMonitoringConfig;

  /**
   * The VPC Interfaces for this flow.
   * @default No VPC Interface configuration applied
   */
  readonly vpcInterfaces?: VpcInterfaceConfig[];

}

/**
 * Simplified configuration for Media Streams
 */
export class MediaStream {
  /**
   * Configuration for MediaStream Video
   */
  public static video(config: MediaStreamVideo): MediaStream {
    return new MediaStream({
      mediaStreamId: config.mediaStreamId,
      mediaStreamName: config.mediaStreamName,
      videoFormat: config.videoFormat.value,
      clockRate: config.clockRate,
      fmt: config.fmt,
      description: config.description,
      attributes: {
        fmtp: {
          exactFramerate: config.fmtp.exactFramerate?.toString(),
          par: config.fmtp.par?.toString(),
          colorimetry: config.fmtp.colorimetry?.value,
          range: config.fmtp.videoRange?.value,
          scanMode: config.fmtp.scanMode?.value,
          tcs: config.fmtp.tcs?.value,
        },
      },
      mediaStreamType: MediaStreamType.VIDEO,
    });
  }

  /**
   * Configuration for MediaStream audio
   */
  public static audio(config: MediaStreamAudio): MediaStream {
    return new MediaStream({
      mediaStreamId: config.mediaStreamId,
      mediaStreamName: config.mediaStreamName,
      clockRate: config.clockRate,
      fmt: config.fmt,
      description: config.description,
      attributes: {
        fmtp: {
          channelOrder: config.channelOrder?.value,
        },
        lang: config.lang,
      },
      mediaStreamType: MediaStreamType.AUDIO,
    });
  }

  /**
   * Configuration for MediaStream ancillary data
   */
  public static ancillaryData(config: MediaStreamAncillaryData): MediaStream {
    return new MediaStream({
      mediaStreamId: config.mediaStreamId,
      mediaStreamName: config.mediaStreamName,
      description: config.description,
      mediaStreamType: MediaStreamType.ANCILLARY_DATA,
    });
  }

  private constructor(private readonly _config: CfnFlow.MediaStreamProperty) {}

  /**
   * @internal
   */
  public _bind(): CfnFlow.MediaStreamProperty {
    return this._config;
  }
}

/**
 * Attributes for importing an existing Flow.
 */
export interface FlowAttributes {
  /**
   * The ARN of the flow.
   */
  readonly flowArn: string;

  /**
   * Indicates whether failover configured
   *
   * @default false
   */
  readonly isFailoverEnabled?: boolean;

  /**
   * ARN of the source defined on the flow.
   *
   * Not encoded in the flow ARN, so must be provided explicitly when you need
   * access to `sourceArn` on the imported construct.
   *
   * @default - sourceArn is not available on the imported construct
   */
  readonly sourceArn?: string;

  /**
   * The IP address that the flow uses to send outbound content.
   *
   * @default - accessing `egressIp` on the imported flow throws; only provide when available.
   */
  readonly egressIp?: string;

  /**
   * The IP address that the flow listens on for incoming content.
   *
   * @default - accessing `sourceIngestIp` on the imported flow throws; only provide when available.
   */
  readonly sourceIngestIp?: string;

  /**
   * The port that the flow listens on for incoming content.
   *
   * @default - accessing `sourceIngestPort` on the imported flow throws; only provide when available.
   */
  readonly sourceIngestPort?: string;

  /**
   * The Availability Zone that the flow was created in.
   *
   * @default - `flowAvailabilityZone` is undefined on the imported flow.
   */
  readonly flowAvailabilityZone?: string;
}

abstract class FlowBase extends Resource implements IFlow {
  /**
   * Creates a Flow construct that represents an external (imported) Flow.
   */
  public static fromFlowAttributes(scope: Construct, id: string, attrs: FlowAttributes): IFlow {
    class Import extends FlowBase implements IFlow {
      public readonly flowArn = attrs.flowArn;
      public readonly flowAvailabilityZone = attrs.flowAvailabilityZone;

      // Imported flows default to `false`. Pass `isFailoverEnabled: true` in the
      // attributes when the real service-side flow has failover enabled and you
      // need to add a FlowSource.
      public isFailoverEnabled?: boolean = attrs.isFailoverEnabled ?? false;

      public get sourceArn(): string {
        if (attrs.sourceArn) return attrs.sourceArn;
        throw new ValidationError(
          lit`SourceArnNotProvided`,
          `'sourceArn' was not provided when importing Flow ${this.node.path}; provide 'sourceArn' in fromFlowAttributes()`,
          this,
        );
      }

      public get egressIp(): string {
        if (attrs.egressIp) return attrs.egressIp;
        throw new ValidationError(
          lit`EgressIpNotProvided`,
          `'egressIp' was not provided when importing Flow ${this.node.path}; provide it in fromFlowAttributes()`,
          this,
        );
      }

      public get sourceIngestIp(): string {
        if (attrs.sourceIngestIp) return attrs.sourceIngestIp;
        throw new ValidationError(
          lit`SourceIngestIpNotProvided`,
          `'sourceIngestIp' was not provided when importing Flow ${this.node.path}; provide it in fromFlowAttributes()`,
          this,
        );
      }

      public get sourceIngestPort(): string {
        if (attrs.sourceIngestPort) return attrs.sourceIngestPort;
        throw new ValidationError(
          lit`SourceIngestPortNotProvided`,
          `'sourceIngestPort' was not provided when importing Flow ${this.node.path}; provide it in fromFlowAttributes()`,
          this,
        );
      }

      public get sourceIngestUrl(): string {
        throw new ValidationError(
          lit`SourceIngestUrlNotAvailableImported`,
          `'sourceIngestUrl' is not available on imported Flow ${this.node.path}; only Flows constructed in this app for listener-style source protocols (RTP, RTP-FEC, RIST, SRT listener, Zixi push) expose a source ingest URL`,
          this,
        );
      }
    }

    return new Import(scope, id);
  }

  /**
   * Import an existing Flow from its ARN.
   *
   * Use `fromFlowAttributes()` instead if you need access to `sourceArn` or any other
   * attribute not encoded in the flow ARN.
   */
  public static fromFlowArn(scope: Construct, id: string, flowArn: string): IFlow {
    return Flow.fromFlowAttributes(scope, id, { flowArn });
  }

  public abstract readonly sourceArn: string;
  public abstract readonly flowArn: string;
  public abstract readonly egressIp: string;
  public abstract readonly sourceIngestIp: string;
  public abstract readonly sourceIngestPort: string;
  public abstract readonly sourceIngestUrl: string;
  public abstract readonly flowAvailabilityZone?: string;
  public abstract readonly isFailoverEnabled?: boolean;

  /**
   * The NDI state for this flow. `undefined` for imported flows (state is unknown).
   * @internal
   */
  protected _ndiState?: State;

  /**
   * The source protocol value for this flow. `undefined` for imported flows.
   * @internal
   */
  protected _sourceProtocol?: string;

  /**
   * Tracks the number of NDI outputs added to this flow. Used to enforce the
   * documented "max 1 NDI output per flow" limit.
   * @internal
   */
  protected _ndiOutputCount = 0;

  /**
   * Collection of grant methods for this flow.
   */
  public readonly grants = FlowGrants.fromFlow(this);

  /**
   * A reference to this Flow resource.
   * Required by the auto-generated IFlowRef interface.
   */
  public get flowRef(): FlowReference {
    return {
      flowArn: this.flowArn,
    };
  }

  /**
   * Add an output to this flow
   */
  public addOutput(id: string, options: AddFlowOutputOptions): IFlowOutput {
    const protocol = options.output._bind().protocol;

    if (protocol === 'ndi-speed-hq') {
      if (this._ndiState !== undefined && this._ndiState !== State.ENABLED) {
        throw new ValidationError(
          lit`NdiOutputRequiresNdiConfig`,
          'NDI outputs require ndiConfig with ndiState set to State.ENABLED on the flow',
          this,
        );
      }

      // NDI passthrough is not supported — a flow cannot have both an NDI source
      // and an NDI output. The service rejects this at deploy; fail fast here.
      if (this._sourceProtocol === SourceProtocol.NDI_SPEED_HQ.value) {
        throw new ValidationError(
          lit`NdiPassthroughNotSupported`,
          'NDI passthrough is not supported. A flow cannot have both an NDI source and an NDI output',
          this,
        );
      }

      // A flow supports a maximum of 1 NDI output (Large flow size, transport
      // stream protocols + 1 NDI output).
      // @see https://docs.aws.amazon.com/mediaconnect/latest/ug/flow-sizes-capabilities.html
      if (this._ndiOutputCount >= 1) {
        throw new ValidationError(
          lit`NdiOutputLimit`,
          'A flow supports a maximum of 1 NDI output',
          this,
        );
      }
      this._ndiOutputCount++;
    }

    return new FlowOutput(this, id, {
      flow: this,
      output: options.output,
      flowOutputName: options.flowOutputName,
      description: options.description,
      outputStatus: options.outputStatus,
    });
  }

  /**
   * Create a CloudWatch metric for this flow.
   */
  public metric(metricName: string, props?: MetricOptions): Metric {
    return new Metric({
      metricName,
      namespace: 'AWS/MediaConnect',
      dimensionsMap: {
        FlowARN: this.flowArn,
      },
      ...props,
    }).attachTo(this);
  }

  /**
   * Metric for the bitrate of content ingested into the flow.
   *
   * Emits the `SourceBitRate` metric with dimension `FlowARN`, which aggregates across
   * all sources on the flow. To narrow to a specific source on a multi-source flow, call
   * {@link metric} with an additional `SourceARN` dimension.
   *
   * @default - average over 60 seconds
   */
  public metricSourceBitrate(props?: MetricOptions): Metric {
    return this.metric('SourceBitRate', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.BITS_PER_SECOND,
      ...props,
    });
  }

  /**
   * Metric for packets that were lost in transit and not recovered by error correction.
   *
   * Emits the `SourceNotRecoveredPackets` metric with dimension `FlowARN`, which aggregates
   * across all sources on the flow.
   *
   * @default - sum over 60 seconds
   */
  public metricSourceNotRecoveredPackets(props?: MetricOptions): Metric {
    return this.metric('SourceNotRecoveredPackets', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the total number of packets received by the flow sources.
   *
   * Emits the `SourceTotalPackets` metric with dimension `FlowARN`, which aggregates across
   * all sources on the flow.
   *
   * @default - sum over 60 seconds
   */
  public metricSourceTotalPackets(props?: MetricOptions): Metric {
    return this.metric('SourceTotalPackets', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric indicating which source is selected for ingest when using `Failover` failover mode.
   * A value of 1 indicates the source is being used as the input; 0 indicates it is standby.
   *
   * Emits the `SourceSelected` metric with dimension `FlowARN`. To narrow to a specific source,
   * call {@link metric} with an additional `SourceARN` dimension.
   *
   * @default - maximum over 60 seconds
   */
  public metricSourceSelected(props?: MetricOptions): Metric {
    return this.metric('SourceSelected', {
      statistic: 'max',
      period: Duration.seconds(60),
      unit: Unit.NONE,
      ...props,
    });
  }

  /**
   * Metric indicating the connection state of the source. 1 for connected, 0 for disconnected.
   * Applies only to Zixi, SRT, and RIST sources.
   *
   * Emits the `SourceConnected` metric with dimension `FlowARN`.
   *
   * @default - minimum over 60 seconds
   */
  public metricSourceConnected(props?: MetricOptions): Metric {
    return this.metric('SourceConnected', {
      statistic: 'min',
      period: Duration.seconds(60),
      unit: Unit.NONE,
      ...props,
    });
  }

  /**
   * Metric for the number of times the source transitioned from connected to disconnected.
   *
   * Emits the `SourceDisconnections` metric with dimension `FlowARN`.
   *
   * @default - sum over 60 seconds
   */
  public metricSourceDisconnections(props?: MetricOptions): Metric {
    return this.metric('SourceDisconnections', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the number of packets lost during transit, measured before any error
   * correction takes place.
   *
   * Emits the `SourceDroppedPackets` metric with dimension `FlowARN`.
   *
   * @default - sum over 60 seconds
   */
  public metricSourceDroppedPackets(props?: MetricOptions): Metric {
    return this.metric('SourceDroppedPackets', {
      statistic: 'sum',
      period: Duration.seconds(60),
      unit: Unit.COUNT,
      ...props,
    });
  }

  /**
   * Metric for the percentage of packets lost during transit, even if they were recovered.
   *
   * Emits the `SourcePacketLossPercent` metric with dimension `FlowARN`.
   *
   * @default - average over 60 seconds
   */
  public metricSourcePacketLossPercent(props?: MetricOptions): Metric {
    return this.metric('SourcePacketLossPercent', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.PERCENT,
      ...props,
    });
  }

  /**
   * Metric for the round-trip time between the source and MediaConnect. Applies only to
   * RIST, Zixi, and SRT sources.
   *
   * Emits the `SourceRoundTripTime` metric with dimension `FlowARN`.
   *
   * @default - average over 60 seconds
   */
  public metricSourceRoundTripTime(props?: MetricOptions): Metric {
    return this.metric('SourceRoundTripTime', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.MILLISECONDS,
      ...props,
    });
  }

  /**
   * Metric for the current network jitter of the source, measured in milliseconds.
   *
   * Emits the `SourceJitter` metric with dimension `FlowARN`.
   *
   * @default - average over 60 seconds
   */
  public metricSourceJitter(props?: MetricOptions): Metric {
    return this.metric('SourceJitter', {
      statistic: 'avg',
      period: Duration.seconds(60),
      unit: Unit.MILLISECONDS,
      ...props,
    });
  }
}

/**
 * Defines an AWS Elemental MediaConnect Flow.
 */
@propertyInjectable
export class Flow extends FlowBase implements IFlow {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-mediaconnect-alpha.Flow';

  /**
   * The set of source protocols where the flow opens an IP:port and listens for
   * upstream content. Determines availability of `sourceIngestIp`, `sourceIngestPort`,
   * and `sourceIngestUrl`. Maps each protocol to the URL scheme used to construct
   * the ingest URL.
   */
  private static readonly LISTENER_PROTOCOL_SCHEMES: ReadonlyMap<string, string> = new Map([
    ['rtp', 'rtp'],
    ['rtp-fec', 'rtp'],
    ['rist', 'rist'],
    ['srt-listener', 'srt'],
    ['zixi-push', 'zixi'],
  ]);

  private static isListenerStyleProtocol(protocol: string | undefined): boolean {
    return protocol !== undefined && Flow.LISTENER_PROTOCOL_SCHEMES.has(protocol);
  }

  public readonly flowArn: string;
  public readonly sourceArn: string;
  public readonly egressIp: string;
  private readonly _sourceIngestIp: string;
  private readonly _sourceIngestPort: string;
  private readonly _hasListenerSource: boolean;
  public readonly flowAvailabilityZone?: string;
  public readonly isFailoverEnabled?: boolean = false;
  private vpcInterfaces: CfnFlow.VpcInterfaceProperty[] = [];

  public get sourceIngestIp(): string {
    if (!this._hasListenerSource) {
      throw new ValidationError(
        lit`SourceIngestIpNotAvailableSourceProtocol`,
        `'sourceIngestIp' is not available on this Flow ${this.node.path}; only listener-style source protocols (RTP, RTP-FEC, RIST, SRT listener, Zixi push) expose a source ingest IP — CDI, JPEG XS, entitlement, gateway bridge, router, and SRT caller sources do not`,
        this,
      );
    }
    return this._sourceIngestIp;
  }

  public get sourceIngestPort(): string {
    if (!this._hasListenerSource) {
      throw new ValidationError(
        lit`SourceIngestPortNotAvailableSourceProtocol`,
        `'sourceIngestPort' is not available on this Flow ${this.node.path}; only listener-style source protocols (RTP, RTP-FEC, RIST, SRT listener, Zixi push) expose a source ingest port — CDI, JPEG XS, entitlement, gateway bridge, router, and SRT caller sources do not`,
        this,
      );
    }
    return this._sourceIngestPort;
  }

  public get sourceIngestUrl(): string {
    if (!this._hasListenerSource) {
      throw new ValidationError(
        lit`SourceIngestUrlNotAvailableSourceProtocol`,
        `'sourceIngestUrl' is not available on this Flow ${this.node.path}; only listener-style source protocols (RTP, RTP-FEC, RIST, SRT listener, Zixi push) expose a source ingest URL — CDI, JPEG XS, entitlement, gateway bridge, router, and SRT caller sources do not`,
        this,
      );
    }
    return this.computeIngestUrl(this._sourceProtocol, this._sourceIngestIp, this._sourceIngestPort);
  }

  constructor(scope: Construct, id: string, props: FlowProps) {
    super(scope, id, {
      physicalName: props?.flowName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 64 }) }),
    });

    // Validate flow name if provided
    this.validateName(props);

    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    // Validate maintenance start hour format
    if (props.maintenanceConfiguration) {
      validateMaintenanceTime(props.maintenanceConfiguration.time);
    }

    // Validate source monitoring thresholds and content-quality requirements
    this.validateMonitoring(props);

    // Validate encoding config video max bitrate range (10-50 Mbps)
    this.validateEncoding(props);

    // Only iterate a known-concrete vpcInterfaces list; skip tokenized lists to avoid
    // pushing the token marker into our internal collection.
    if (props.vpcInterfaces !== undefined && !Token.isUnresolved(props.vpcInterfaces)) {
      props.vpcInterfaces.forEach(vpc => this.addVpcInterface(vpc));
    }
    this.isFailoverEnabled = props.sourceFailoverConfig?._isEnabled ?? false;

    const sourceConfig = props.source._bind(this, Stack.of(this).formatArn({
      service: 'mediaconnect',
      resource: 'flow',
      resourceName: `*:${this.physicalName}`,
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
    }));

    if ((sourceConfig.protocol === SourceProtocol.JPEGXS.value || sourceConfig.protocol === SourceProtocol.CDI.value)
      && props.flowSize?.value !== FlowSize.LARGE_4X.value) {
      throw new ValidationError(
        lit`FlowSizeRequired`,
        'Flows using JPEG XS and CDI protocols require LARGE_4X flow size; set flowSize to FlowSize.LARGE_4X',
        this,
      );
    }

    // NDI constraints (discovery server limit, protocol conflicts, flow size, source requirements)
    this.validateNdi(props, sourceConfig);

    // LARGE_4X only supports CDI and JPEG XS protocols
    if (props.flowSize?.value === FlowSize.LARGE_4X.value
      && sourceConfig.protocol !== SourceProtocol.JPEGXS.value
      && sourceConfig.protocol !== SourceProtocol.CDI.value) {
      throw new ValidationError(
        lit`FlowSizeLarge4xProtocol`,
        'LARGE_4X flow size only supports CDI and JPEG XS protocols',
        this,
      );
    }

    // VPC interface EFA-count limit
    this.validateVpcEfaCount(props);

    const flow = new CfnFlow(this, 'Resource', {
      name: this.physicalName,
      source: sourceConfig,
      availabilityZone: props?.availabilityZone,
      flowSize: props?.flowSize?.value,
      maintenance: props?.maintenanceConfiguration ? {
        maintenanceDay: toTitleCase(props.maintenanceConfiguration.day),
        maintenanceStartHour: props.maintenanceConfiguration.time,
      } : undefined,
      mediaStreams: props.mediaStreams ? props.mediaStreams.map(stream => stream._bind()) : undefined,
      ndiConfig: props.ndiConfig ? {
        machineName: props.ndiConfig.machineName,
        ndiState: props.ndiConfig.ndiState,
        ndiDiscoveryServers: props.ndiConfig.ndiDiscoveryServers?.map(server => ({
          discoveryServerAddress: server.discoveryServerAddress,
          discoveryServerPort: server.discoveryServerPort,
          vpcInterfaceAdapter: server.vpcInterface.name,
        })),
      } : undefined,
      encodingConfig: props.encodingConfig ? {
        encodingProfile: props.encodingConfig.encodingProfile,
        videoMaxBitrate: props.encodingConfig.videoMaxBitrate?.toBps(),
      } : undefined,
      sourceFailoverConfig: props.sourceFailoverConfig?._bind(),
      sourceMonitoringConfig: props.sourceMonitoringConfig ? {
        contentQualityAnalysisState: props.sourceMonitoringConfig.contentQualityAnalysisState,
        thumbnailState: props.sourceMonitoringConfig.thumbnailState,
        audioMonitoringSettings: props.sourceMonitoringConfig.silentAudio ? [{
          silentAudio: {
            state: props.sourceMonitoringConfig.silentAudio.state,
            thresholdSeconds: props.sourceMonitoringConfig.silentAudio.threshold.toSeconds(),
          },
        }] : undefined,
        videoMonitoringSettings: props.sourceMonitoringConfig.blackFrames || props.sourceMonitoringConfig.frozenFrames ? [{
          blackFrames: props.sourceMonitoringConfig.blackFrames ? {
            state: props.sourceMonitoringConfig.blackFrames.state,
            thresholdSeconds: props.sourceMonitoringConfig.blackFrames.threshold.toSeconds(),
          } : undefined,
          frozenFrames: props.sourceMonitoringConfig.frozenFrames ? {
            state: props.sourceMonitoringConfig.frozenFrames.state,
            thresholdSeconds: props.sourceMonitoringConfig.frozenFrames.threshold.toSeconds(),
          } : undefined,
        }] : undefined,
      } : undefined,
      vpcInterfaces: Lazy.any({ produce: () => this.vpcInterfaces }, { omitEmptyArray: true }),
    });

    this.flowArn = flow.attrFlowArn;
    this.sourceArn = flow.attrSourceSourceArn;
    this.egressIp = flow.attrEgressIp;
    this._sourceIngestIp = flow.attrSourceIngestIp;
    this._sourceIngestPort = flow.attrSourceSourceIngestPort;
    this.flowAvailabilityZone = flow.attrFlowAvailabilityZone;
    this._ndiState = props.ndiConfig?.ndiState ?? State.DISABLED;
    this._sourceProtocol = sourceConfig.protocol;
    this._hasListenerSource = Flow.isListenerStyleProtocol(sourceConfig.protocol);

    this.node.addValidation({
      validate: () => {
        const names = this.vpcInterfaces.map(v => v.name).filter(n => !Token.isUnresolved(n));
        const duplicates = [...new Set(names.filter((n, i) => names.indexOf(n) !== i))];
        return duplicates.length > 0
          ? [`VPC interface names must be unique within a flow. Duplicate name(s): ${duplicates.join(', ')}`]
          : [];
      },
    });
  }

  /** Validate the flow name length and character set, when a concrete value is provided. */
  private validateName(props: FlowProps): void {
    if (props.flowName != null && props.flowName !== '' && !Token.isUnresolved(props.flowName)) {
      if (props.flowName.length > 64) {
        throw new ValidationError(lit`FlowNameLength`, `Flow name must be between 1 and 64 characters, got ${props.flowName.length}`, this);
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(props.flowName)) {
        throw new ValidationError(lit`FlowNameFormat`, `Flow name must contain only alphanumeric characters, hyphens, and underscores, got '${props.flowName}'`, this);
      }
    }
  }

  /**
   * Validate source-monitoring thresholds (10-60 seconds) and that the per-metric checks are only
   * used when content-quality analysis is enabled.
   */
  private validateMonitoring(props: FlowProps): void {
    const monitoring = props.sourceMonitoringConfig;
    if (monitoring && (monitoring.silentAudio || monitoring.blackFrames || monitoring.frozenFrames)
      && monitoring.contentQualityAnalysisState !== State.ENABLED) {
      throw new ValidationError(
        lit`MonitoringMetricsRequireContentQualityAnalysis`,
        'silentAudio, blackFrames, and frozenFrames require contentQualityAnalysisState to be State.ENABLED',
        this,
      );
    }
    if (monitoring?.silentAudio) {
      const secs = monitoring.silentAudio.threshold.toSeconds();
      if (secs < 10 || secs > 60) {
        throw new ValidationError(lit`SilentAudioThresholdRange`, `Silent audio threshold must be between 10 and 60 seconds, got ${secs}`, this);
      }
    }
    if (monitoring?.blackFrames) {
      const secs = monitoring.blackFrames.threshold.toSeconds();
      if (secs < 10 || secs > 60) {
        throw new ValidationError(lit`BlackFramesThresholdRange`, `Black frames threshold must be between 10 and 60 seconds, got ${secs}`, this);
      }
    }
    if (monitoring?.frozenFrames) {
      const secs = monitoring.frozenFrames.threshold.toSeconds();
      if (secs < 10 || secs > 60) {
        throw new ValidationError(lit`FrozenFramesThresholdRange`, `Frozen frames threshold must be between 10 and 60 seconds, got ${secs}`, this);
      }
    }
  }

  /** Validate the encoding config's video max bitrate range (10-50 Mbps). */
  private validateEncoding(props: FlowProps): void {
    if (props.encodingConfig?.videoMaxBitrate) {
      const mbps = props.encodingConfig.videoMaxBitrate.toMbps();
      if (mbps < 10 || mbps > 50) {
        throw new ValidationError(
          lit`EncodingVideoMaxBitrateRange`,
          `Encoding config videoMaxBitrate must be between 10 and 50 Mbps, got ${mbps}`,
          this,
        );
      }
    }
  }

  /**
   * Validate NDI constraints: discovery-server limit, mutual exclusivity with CDI/JPEG XS, the
   * required LARGE flow size, and the requirements an NDI source places on the flow.
   */
  private validateNdi(props: FlowProps, sourceConfig: CfnFlow.SourceProperty): void {
    // NDI discovery servers limit (service rejects >3 at deploy time)
    const discoveryServers = props.ndiConfig?.ndiDiscoveryServers;
    if (discoveryServers !== undefined && !Token.isUnresolved(discoveryServers) && discoveryServers.length > 3) {
      throw new ValidationError(
        lit`NdiMaxDiscoveryServers`,
        `NDI config supports a maximum of 3 discovery servers, got ${discoveryServers.length}`,
        this,
      );
    }

    // NDI cannot coexist with CDI or JPEG XS — they require different flow sizes
    if (props.ndiConfig?.ndiState === State.ENABLED
      && (sourceConfig.protocol === SourceProtocol.CDI.value || sourceConfig.protocol === SourceProtocol.JPEGXS.value)) {
      throw new ValidationError(
        lit`NdiCdiConflict`,
        'NDI cannot be used with CDI or JPEG XS protocols; NDI requires LARGE flow size while CDI and JPEG XS require LARGE_4X',
        this,
      );
    }

    // NDI requires LARGE flow size (not MEDIUM, not LARGE_4X)
    if (props.ndiConfig?.ndiState === State.ENABLED
      && props.flowSize?.value !== FlowSize.LARGE.value) {
      throw new ValidationError(
        lit`NdiFlowSizeRequired`,
        'NDI requires LARGE flow size; set flowSize to FlowSize.LARGE',
        this,
      );
    }

    // NDI sources require NDI to be enabled on the flow
    if (sourceConfig.protocol === SourceProtocol.NDI_SPEED_HQ.value
      && props.ndiConfig?.ndiState !== State.ENABLED) {
      throw new ValidationError(
        lit`NdiSourceRequiresNdiConfig`,
        'NDI sources require ndiConfig with ndiState set to State.ENABLED',
        this,
      );
    }

    // NDI sources require LARGE flow size
    if (sourceConfig.protocol === SourceProtocol.NDI_SPEED_HQ.value
      && props.flowSize?.value !== FlowSize.LARGE.value) {
      throw new ValidationError(
        lit`NdiSourceFlowSizeRequired`,
        'NDI sources require LARGE flow size; set flowSize to FlowSize.LARGE',
        this,
      );
    }

    // NDI sources require encodingConfig to transcode into a transport stream
    if (sourceConfig.protocol === SourceProtocol.NDI_SPEED_HQ.value
      && !props.encodingConfig) {
      throw new ValidationError(
        lit`NdiSourceRequiresEncodingConfig`,
        'NDI sources require encodingConfig to transcode the NDI source into a transport stream',
        this,
      );
    }
  }

  /** Validate the EFA VPC-interface count (at most one), when the list is concrete. */
  private validateVpcEfaCount(props: FlowProps): void {
    // Skip EFA-count validation when the user passed a tokenized vpcInterfaces list;
    // we can't inspect individual entries in that case.
    const efaCount = props.vpcInterfaces !== undefined && !Token.isUnresolved(props.vpcInterfaces)
      ? props.vpcInterfaces.filter(vpc => vpc.networkInterfaceType?.value === NetworkInterface.EFA.value).length
      : 0;

    if (efaCount > 1) {
      throw new ValidationError(
        lit`FlowMaxEfaInterfaces`,
        `A flow can have a maximum of 1 EFA VPC interface, got ${efaCount}`,
        this,
      );
    }
  }

  /**
   * Build the URL for a listener-style source protocol. The caller must have already
   * checked `_hasListenerSource`; this method assumes the protocol is in
   * `LISTENER_PROTOCOL_SCHEMES` and throws an unscoped error otherwise.
   */
  private computeIngestUrl(protocol: string | undefined, ip: string, port: string): string {
    const scheme = protocol === undefined ? undefined : Flow.LISTENER_PROTOCOL_SCHEMES.get(protocol);
    if (scheme === undefined) {
      throw new UnscopedValidationError(
        lit`SourceIngestUrlComputeNotListener`,
        `internal: computeIngestUrl called for non-listener protocol '${protocol}'`,
      );
    }
    return Fn.join('', [`${scheme}://`, ip, ':', port]);
  }

  /**
   * Add a VPC interface to this flow.
   */
  @MethodMetadata()
  public addVpcInterface(vpc: VpcInterfaceConfig) {
    this.vpcInterfaces.push({
      name: vpc.name,
      networkInterfaceIds: vpc.networkInterfaceIds,
      networkInterfaceType: vpc.networkInterfaceType?.value,
      securityGroupIds: vpc.securityGroups.map(sg => sg.securityGroupId),
      subnetId: vpc.subnet.subnetId,
      roleArn: vpc.role.roleArn,
    });
    return vpc;
  }
}
