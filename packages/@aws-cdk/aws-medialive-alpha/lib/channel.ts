import type { Duration, IResource } from 'aws-cdk-lib';
import { Resource, Lazy, Names, Aws, ArnFormat, Stack, ValidationError } from 'aws-cdk-lib';
import type { MetricOptions } from 'aws-cdk-lib/aws-cloudwatch';
import { Metric } from 'aws-cdk-lib/aws-cloudwatch';
import type { ISubnetRef } from 'aws-cdk-lib/aws-ec2';
import type { ISecurityGroupRef } from 'aws-cdk-lib/aws-elasticache';
import type { IRole } from 'aws-cdk-lib/aws-iam';
import { Grant, PolicyStatement, Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import type { IChannelRef, ChannelReference, IInputSecurityGroupRef, IClusterRef, IChannelPlacementGroupRef } from 'aws-cdk-lib/aws-medialive';
import { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import { AvailBlankingState, BlackoutSlateState, NetworkEndBlackout, type AvailBlanking, type AvailSettings, type Scte35SegmentationScope, type BlackoutSlate } from './avail-settings';
import { ThumbnailState, MotionGraphicsInsertion, type FeatureActivations, type MotionGraphicsConfiguration, type NielsenConfiguration, type ThumbnailConfiguration } from './channel-features';
import type { ColorCorrection } from './color-correction';
import type { FileLocation } from './file-location';
import { InputType } from './input';
import { FailoverCondition, SourceEndBehavior, InputFilter, Smpte2038DataPreference, type InputAttachment } from './input-attachment';
import { InputSpecification } from './input-specification';
import { ChannelGrants } from './medialive-grants.generated';
import type { OutputGroupConfiguration } from './output-group';
import { OutputGroup } from './output-group';
import { extractResourceId } from './shared';
import { VideoCodecType } from './video-codec-settings';

export {
  VideoCodecSettings,
  H264Profile,
  H264RateControl,
  H264AdaptiveQuantization,
  GopSize,
  H265Profile,
  H265Tier,
  H265RateControl,
  Av1RateControl,
  AfdSignaling,
  ColorMetadata,
  ScanType,
  FlickerAq,
  GopBReference,
  LookAheadRateControl,
  TimecodeInsertion,
  SubgopLength,
  H264EntropyEncoding,
  H264ForceFieldPictures,
  H264Syntax,
  H264QualityLevel,
  H265AdaptiveQuantization,
  H265AlternativeTransferFunction,
  H265Deblocking,
  H265MvOverPictureBoundaries,
  H265MvTemporalPredictor,
  H265TilePadding,
  H265TreeblockSize,
  Av1BitDepth,
  Av1SceneChangeDetect,
  Av1SpatialAq,
  Av1TemporalAq,
  Av1TimecodeInsertion,
  H264SceneChangeDetect,
  H264SpatialAq,
  H264TemporalAq,
  H265SceneChangeDetect,
  TimecodeBurninFontSize,
  TimecodeBurninPosition,
} from './video-codec-settings';
export type {
  H264SettingsProps,
  H265SettingsProps,
  Av1SettingsProps,
  FrameCaptureSettingsProps,
  TimecodeBurninSettings,
} from './video-codec-settings';
export {
  AudioCodecSettings,
  AacProfile,
  AacCodingMode,
  AacRateControlMode,
  AacRawFormat,
  AacSpec,
  AacInputType,
  AacVbrQuality,
  Ac3AttenuationControl,
  Ac3BitstreamMode,
  Ac3CodingMode,
  Ac3DrcProfile,
  Ac3LfeFilter,
  Ac3MetadataControl,
  Eac3AttenuationControl,
  Eac3BitstreamMode,
  Eac3CodingMode,
  Eac3DcFilter,
  Eac3DrcLine,
  Eac3DrcRf,
  Eac3LfeControl,
  Eac3LfeFilter,
  Eac3MetadataControl,
  Eac3PassthroughControl,
  Eac3PhaseControl,
  Eac3StereoDownmix,
  Eac3SurroundExMode,
  Eac3SurroundMode,
  Eac3AtmosCodingMode,
  Eac3AtmosDrcLine,
  Eac3AtmosDrcRf,
  Mp2CodingMode,
  WavCodingMode,
} from './audio-codec-settings';
export type {
  AacSettingsProps,
  Ac3SettingsProps,
  Eac3SettingsProps,
  Eac3AtmosSettingsProps,
  Mp2SettingsProps,
  WavSettingsProps,
} from './audio-codec-settings';

/**
 * Represents a MediaLive Channel.
 */
export interface IChannel extends IResource, IChannelRef {
  /**
   * The ARN of the channel.
   * @attribute
   */
  readonly channelArn: string;
  /**
   * The ID of the channel.
   * @attribute
   */
  readonly channelId: string;
  /**
   * The IDs of the inputs attached to this channel.
   * @attribute
   */
  readonly channelInputs?: string[];

  /**
   * Collection of grant methods for this channel — start, stop, and update its schedule.
   */
  readonly grants: ChannelGrants;

  /**
   * Create a CloudWatch metric for this channel scoped to a specific pipeline.
   *
   * Channel metrics are published per-pipeline. `STANDARD` channels run two
   * redundant pipelines (`PIPELINE_0` and `PIPELINE_1`); to cover both, build
   * a metric for each. `SINGLE_PIPELINE` channels only publish on `PIPELINE_0`.
   * See the
   * {@link https://docs.aws.amazon.com/medialive/latest/ug/monitoring-eml-metrics.html | MediaLive metrics docs}
   * for the full set of metric names and recommended statistics.
   */
  metric(metricName: string, pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for the total number of active alerts on this channel.
   *
   * @default - max over 5 minutes
   */
  metricActiveAlerts(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for the rate of inbound network traffic to MediaLive in Mbps.
   *
   * @default - average over 5 minutes
   */
  metricNetworkIn(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for the rate of outbound network traffic from MediaLive in Mbps.
   *
   * @default - average over 5 minutes
   */
  metricNetworkOut(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for the input video frame rate (frames per second).
   *
   * @default - max over 5 minutes
   */
  metricInputVideoFrameRate(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for fill milliseconds — the time MediaLive has filled the video output
   * with fill frames because the input did not deliver content within the
   * expected window. A non-zero value indicates an unhealthy input.
   *
   * @default - max over 5 minutes
   */
  metricFillMsec(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for input loss seconds (RTP and MediaConnect inputs only).
   *
   * @default - sum over 5 minutes
   */
  metricInputLossSeconds(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for dropped frames. A non-zero value indicates the encoder cannot
   * keep up with the incoming video in real time.
   *
   * @default - sum over 5 minutes
   */
  metricDroppedFrames(pipeline: Pipeline, props?: MetricOptions): Metric;

  /**
   * Metric for SVQ time (speed-vs-quality), expressed as a percent. Indicates
   * the share of time MediaLive reduced quality optimisations to keep up with
   * real-time output.
   *
   * @default - max over 5 minutes
   */
  metricSvqTime(pipeline: Pipeline, props?: MetricOptions): Metric;
}

/**
 * MediaLive pipeline (channel pipeline 0 or 1).
 *
 * `STANDARD` channels run two redundant pipelines (`PIPELINE_0`, `PIPELINE_1`).
 * `SINGLE_PIPELINE` channels run only `PIPELINE_0`.
 */
export class Pipeline {
  /** Pipeline 0. Always available on every channel. */
  public static readonly PIPELINE_0 = new Pipeline('0');

  /** Pipeline 1. Available only on `STANDARD` channels. */
  public static readonly PIPELINE_1 = new Pipeline('1');

  private constructor(private readonly value: string) {}

  /** Returns the CloudWatch dimension value for this pipeline (`'0'` or `'1'`). */
  public toString(): string {
    return this.value;
  }
}

/**
 * The class of the channel. Determines the pipeline redundancy.
 */
export class ChannelClass {
  /** Single pipeline — no redundancy */
  public static readonly SINGLE_PIPELINE = new ChannelClass('SINGLE_PIPELINE');
  /** Standard — two pipelines for redundancy */
  public static readonly STANDARD = new ChannelClass('STANDARD');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ChannelClass {
    return new ChannelClass(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * The log level for the channel.
 */
export class LogLevel {
  /** Log errors only */
  public static readonly ERROR = new LogLevel('ERROR');
  /** Log warnings and errors */
  public static readonly WARNING = new LogLevel('WARNING');
  /** Log info, warnings, and errors */
  public static readonly INFO = new LogLevel('INFO');
  /** Log everything */
  public static readonly DEBUG = new LogLevel('DEBUG');
  /** Disable logging */
  public static readonly DISABLED = new LogLevel('DISABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): LogLevel {
    return new LogLevel(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * The source of timecode for the channel outputs.
 */
export class TimecodeSource {
  /** Use embedded timecode from the source. Falls back to zero-based if not detected. */
  public static readonly EMBEDDED = new TimecodeSource('EMBEDDED');
  /** Use the system clock (UTC). */
  public static readonly SYSTEMCLOCK = new TimecodeSource('SYSTEMCLOCK');
  /** Start at 00:00:00:00. */
  public static readonly ZEROBASED = new TimecodeSource('ZEROBASED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TimecodeSource {
    return new TimecodeSource(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Timecode configuration for the channel.
 */
export interface TimecodeConfig {
  /**
   * The source of timecode.
   * @default TimecodeSource.EMBEDDED
   */
  readonly source?: TimecodeSource;
  /**
   * The threshold in frames beyond which output timecode is resynchronized to the input timecode.
   * @default - no sync threshold
   */
  readonly syncThreshold?: number;
}

/**
 * Properties for creating a MediaLive Channel.
 */
export interface ChannelProps {
  /**
   * The name of the channel.
   * @default - auto-generated
   */
  readonly channelName?: string;

  /**
   * The class of the channel (STANDARD for redundancy, SINGLE_PIPELINE for cost savings).
   * @default ChannelClass.SINGLE_PIPELINE
   */
  readonly channelClass?: ChannelClass;

  /**
   * The IAM role for MediaLive to assume when running this channel.
   *
   * [disable-awslint:prefer-ref-interface]
   *
   * @default - a role is auto-created with confused-deputy prevention
   */
  readonly role?: IRole;

  /**
   * The input attachments for this channel. At least one is required.
   * Additional inputs can be added with `addInput()`.
   */
  readonly inputs: InputAttachment[];

  /**
   * The initial output groups for this channel. At least one is required.
   * Additional output groups can be added with `addOutputGroup()`.
   *
   * A single channel can contain multiple output groups with different codecs
   * (e.g. H.264 and H.265 ladders) sharing the same input.
   */
  readonly outputGroups: OutputGroupConfiguration[];

  /**
   * The log level for the channel.
   * @default LogLevel.DISABLED
   */
  readonly logLevel?: LogLevel;

  /**
   * The input specification for this channel. Defines the expected codec, bitrate, and
   * resolution of the inputs, and whether they are standard, CDI, or Elemental Link inputs.
   * @default - InputSpecification.standard() (AVC codec, 20 Mbps max, HD resolution)
   */
  readonly inputSpecification?: InputSpecification;

  /**
   * Global configuration settings for the channel.
   * @default - default global configuration
   */
  readonly globalConfiguration?: GlobalConfiguration;

  /**
   * Tags to add to the channel.
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };

  /**
   * Maintenance window configuration for the channel.
   * @default - default maintenance window
   */
  readonly maintenance?: MaintenanceSettings;

  /**
   * Timecode configuration for the channel.
   * @default - EMBEDDED source, no sync threshold
   */
  readonly timecodeConfig?: TimecodeConfig;

  /**
   * VPC output settings. When set, all output endpoints are created in the specified VPC.
   * @default - no VPC (outputs use public endpoints)
   */
  readonly vpc?: VpcOutputSettings;

  /**
   * Settings for blanking video, audio, and captions during ad avails.
   * @default - avail blanking disabled
   */
  readonly availBlanking?: AvailBlanking;

  /**
   * Ad avail handling configuration. Defines how SCTE-35 markers are processed.
   * @default - no avail configuration
   */
  readonly availSettings?: AvailSettings;

  /**
   * Which output groups receive SCTE-35 segmentation cues.
   * @default - service default
   */
  readonly scte35SegmentationScope?: Scte35SegmentationScope;

  /**
   * Feature activations for the channel (e.g. Input Prepare schedule actions).
   * @default - all features disabled
   */
  readonly featureActivations?: FeatureActivations;

  /**
   * Motion graphics overlay configuration.
   * @default - motion graphics disabled
   */
  readonly motionGraphicsConfiguration?: MotionGraphicsConfiguration;

  /**
   * Nielsen watermark configuration.
   * @default - no Nielsen configuration
   */
  readonly nielsenConfiguration?: NielsenConfiguration;

  /**
   * Thumbnail generation configuration.
   * @default - thumbnails disabled
   */
  readonly thumbnailConfiguration?: ThumbnailConfiguration;

  /**
   * Blackout slate configuration. Controls what is displayed during blackout events.
   * @default - blackout slate disabled
   */
  readonly blackoutSlate?: BlackoutSlate;

  /**
   * Global color correction rules applied to all outputs.
   * @default - no color corrections
   */
  readonly colorCorrections?: ColorCorrection[];

  /**
   * Anywhere settings for running the channel on AWS Elemental Anywhere.
   * @default - not an Anywhere channel
   */
  readonly anywhereSettings?: AnywhereSettings;

  /**
   * The engine version for the channel.
   * @default - service default
   */
  readonly channelEngineVersion?: string;

  /**
   * Linked channel settings for primary/follower channel configurations.
   * @default - not a linked channel
   */
  readonly linkedChannelSettings?: LinkedChannelSettings;

  /**
   * Input security groups to associate with the channel. Controls which IP addresses can connect
   * to the channel's outputs (pull-style outputs where downstream systems initiate connections).
   *
   * @default - no channel security groups
   */
  readonly channelSecurityGroups?: IInputSecurityGroupRef[];

  /**
   * An AWS Elemental Inference feed to send this channel's media to for inference
   * processing.
   *
   * Future breaking change: this will change when Elemental Inference is released as an L2 construct.
   *
   * @default - the channel is not associated to an inference feed
   */
  readonly inferenceFeedArn?: string;
}

/**
 * Action to take when the current input completes.
 */
export class InputEndAction {
  /** Restart at the beginning of the first input */
  public static readonly SWITCH_AND_LOOP_INPUTS = new InputEndAction('SWITCH_AND_LOOP_INPUTS');
  /** Do nothing — show slate or black until next input switch */
  public static readonly NONE = new InputEndAction('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): InputEndAction {
    return new InputEndAction(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * How MediaLive pipelines are synchronised. The legacy wire selector; the L2 derives it from
 * the `OutputLocking` configuration.
 *
 * @see https://docs.aws.amazon.com/medialive/latest/ug/plan-redundancy-mode.html
 * @see https://docs.aws.amazon.com/medialive/latest/ug/pipeline-lock.html
 */
enum OutputLockingMode {
  /** Synchronise pipelines to each other */
  PIPELINE_LOCKING = 'PIPELINE_LOCKING',
  /** Synchronise pipelines to the Unix epoch */
  EPOCH_LOCKING = 'EPOCH_LOCKING',
}

/**
 * Source of output timing.
 */
export class OutputTimingSource {
  /** Use the input clock */
  public static readonly INPUT_CLOCK = new OutputTimingSource('INPUT_CLOCK');
  /** Use the system clock */
  public static readonly SYSTEM_CLOCK = new OutputTimingSource('SYSTEM_CLOCK');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): OutputTimingSource {
    return new OutputTimingSource(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** The image MediaLive substitutes into the output on input loss. */
export class InputLossImageType {
  /** Substitute a solid color. */
  public static readonly COLOR = new InputLossImageType('COLOR');
  /** Substitute a slate image. */
  public static readonly SLATE = new InputLossImageType('SLATE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): InputLossImageType {
    return new InputLossImageType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Behavior on input loss: substitute black, optionally repeat the last frame, then show a solid
 * color or a slate image.
 */
export interface InputLossBehavior {
  /**
   * How long to substitute black before showing the input-loss image (up to Duration.seconds(1000);
   * Duration.seconds(1000) is interpreted as infinite).
   * @default - service default
   */
  readonly blackFrame?: Duration;
  /**
   * How long to repeat the previous picture before substituting black (up to Duration.seconds(1000);
   * Duration.seconds(1000) is interpreted as infinite).
   * @default - service default
   */
  readonly repeatFrame?: Duration;
  /**
   * Whether to substitute a solid color or a slate image after the black period.
   * @default - service default
   */
  readonly imageType?: InputLossImageType;
  /**
   * The image color as 6 hex characters (RGB). Used when InputLossImageType.COLOR.
   * @default - service default
   */
  readonly imageColor?: string;
  /**
   * The slate image to display. Used when `imageType` is SLATE. Provide a `FileLocation`
   * referencing an S3 bucket (`FileLocation.fromBucket`, which auto-grants read access) or
   * a URL (`FileLocation.url`).
   * @default - service default
   */
  readonly imageSlate?: FileLocation;
}

/** Properties for epoch output locking. */
export interface EpochOutputLockingProps {
  /**
   * A custom epoch (ISO-8601 timestamp) to lock outputs to.
   * @default - service default
   */
  readonly customEpoch?: string;
  /**
   * A jam-sync time (ISO-8601 timestamp).
   * @default - service default
   */
  readonly jamSyncTime?: string;
}

/** The method MediaLive uses to synchronise pipelines for pipeline output locking. */
export class PipelineLockingMethod {
  /** Use the timecode in the source (the default). Requires reliable embedded timecodes. */
  public static readonly SOURCE_TIMECODE = new PipelineLockingMethod('SOURCE_TIMECODE');
  /**
   * Lock frames the encoder identifies as having matching content (visual signature comparison).
   * Does not require embedded timecodes; existing timecodes are ignored for locking decisions.
   */
  public static readonly VIDEO_ALIGNMENT = new PipelineLockingMethod('VIDEO_ALIGNMENT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): PipelineLockingMethod {
    return new PipelineLockingMethod(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Properties for pipeline output locking. */
export interface PipelineOutputLockingProps {
  /**
   * A custom epoch (ISO-8601 timestamp) to lock outputs to.
   * @default - service default
   */
  readonly customEpoch?: string;
  /**
   * The method MediaLive uses to synchronise the pipelines.
   * @default - SOURCE_TIMECODE, applied by MediaLive
   */
  readonly method?: PipelineLockingMethod;
}

/**
 * Output locking synchronises the frames emitted by a channel's two pipelines. Use the static
 * factory methods to select a strategy:
 *
 * - `OutputLocking.pipeline()` — synchronise each pipeline's output to the other.
 * - `OutputLocking.epoch()` — synchronise each pipeline's output to the Unix epoch.
 * - `OutputLocking.disabled()` — do not synchronise pipelines.
 *
 * @see https://docs.aws.amazon.com/medialive/latest/ug/plan-redundancy-mode.html
 * @see https://docs.aws.amazon.com/medialive/latest/ug/pipeline-lock.html
 */
export abstract class OutputLocking {
  /** Disable output locking (optionally with a custom epoch). */
  public static disabled(customEpoch?: string): OutputLocking {
    return new DisabledOutputLocking(customEpoch);
  }
  /** Lock outputs to an epoch. */
  public static epoch(props: EpochOutputLockingProps = {}): OutputLocking {
    return new EpochOutputLocking(props);
  }
  /** Lock pipelines to each other. */
  public static pipeline(props: PipelineOutputLockingProps = {}): OutputLocking {
    return new PipelineOutputLocking(props);
  }

  /** @internal */
  public abstract _bind(): CfnChannel.OutputLockingSettingsProperty;

  /**
   * The legacy `OutputLockingMode` wire value implied by this strategy, or `undefined` for
   * disabled (which has no legacy mode value — it's expressed only via the settings block).
   * @internal
   */
  public abstract _mode(): OutputLockingMode | undefined;
}

/** @internal */
class DisabledOutputLocking extends OutputLocking {
  constructor(private readonly customEpoch?: string) { super(); }
  public _bind(): CfnChannel.OutputLockingSettingsProperty {
    return { disabledLockingSettings: { customEpoch: this.customEpoch } };
  }
  public _mode(): OutputLockingMode | undefined {
    return undefined;
  }
}

/** @internal */
class EpochOutputLocking extends OutputLocking {
  constructor(private readonly props: EpochOutputLockingProps) { super(); }
  public _bind(): CfnChannel.OutputLockingSettingsProperty {
    return { epochLockingSettings: { customEpoch: this.props.customEpoch, jamSyncTime: this.props.jamSyncTime } };
  }
  public _mode(): OutputLockingMode | undefined {
    return OutputLockingMode.EPOCH_LOCKING;
  }
}

/** @internal */
class PipelineOutputLocking extends OutputLocking {
  constructor(private readonly props: PipelineOutputLockingProps) { super(); }
  public _bind(): CfnChannel.OutputLockingSettingsProperty {
    return { pipelineLockingSettings: { customEpoch: this.props.customEpoch, pipelineLockingMethod: this.props.method?.value } };
  }
  public _mode(): OutputLockingMode | undefined {
    return OutputLockingMode.PIPELINE_LOCKING;
  }
}

/**
 * Global configuration settings that apply to the entire channel.
 */
export interface GlobalConfiguration {
  /**
   * The initial audio gain for the channel (-60 to 60 dB).
   * @default - service default
   */
  readonly initialAudioGain?: number;
  /**
   * Action to take when the current input completes.
   * @default - service default
   */
  readonly inputEndAction?: InputEndAction;
  /**
   * Source of output timing.
   * @default - service default
   */
  readonly outputTimingSource?: OutputTimingSource;
  /**
   * Enable support for low framerate inputs (e.g. music channels with less than 1 fps).
   * @default false
   */
  readonly supportLowFramerateInputs?: boolean;
  /**
   * Behavior on input loss (substitute black / repeat frame, then a color or slate image).
   * @default - service default
   */
  readonly inputLossBehavior?: InputLossBehavior;
  /**
   * How MediaLive pipelines are synchronised — `OutputLocking.pipeline()`,
   * `OutputLocking.epoch()`, or `OutputLocking.disabled()`.
   *
   * @see https://docs.aws.amazon.com/medialive/latest/ug/plan-redundancy-mode.html
   * @see https://docs.aws.amazon.com/medialive/latest/ug/pipeline-lock.html
   * @default - service default
   */
  readonly outputLocking?: OutputLocking;
}

/**
 * Day of the week for maintenance.
 */
export class MaintenanceDay {
  /** Monday */
  public static readonly MONDAY = new MaintenanceDay('MONDAY');
  /** Tuesday */
  public static readonly TUESDAY = new MaintenanceDay('TUESDAY');
  /** Wednesday */
  public static readonly WEDNESDAY = new MaintenanceDay('WEDNESDAY');
  /** Thursday */
  public static readonly THURSDAY = new MaintenanceDay('THURSDAY');
  /** Friday */
  public static readonly FRIDAY = new MaintenanceDay('FRIDAY');
  /** Saturday */
  public static readonly SATURDAY = new MaintenanceDay('SATURDAY');
  /** Sunday */
  public static readonly SUNDAY = new MaintenanceDay('SUNDAY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): MaintenanceDay {
    return new MaintenanceDay(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Maintenance window settings for the channel.
 */
export interface MaintenanceSettings {
  /**
   * The day of the week for maintenance.
   */
  readonly day: MaintenanceDay;
  /**
   * The start time for maintenance in UTC (HH:MM format, e.g. '02:00').
   * @default '02:00'
   */
  readonly time?: string;
}

/**
 * Anywhere settings for running the channel on AWS Elemental Anywhere.
 */
export interface AnywhereSettings {
  /**
   * The cluster for this channel.
   */
  readonly cluster: IClusterRef;
  /**
   * The channel placement group for this channel.
   *
   * @default - no placement group
   */
  readonly channelPlacementGroup?: IChannelPlacementGroupRef;
}

/**
 * Linked channel type.
 */
enum LinkedChannelType {
  /** This channel is the primary */
  PRIMARY = 'PRIMARY_CHANNEL',
  /** This channel follows a primary */
  FOLLOWER = 'FOLLOWING_CHANNEL',
}

/**
 * Linked channel settings for primary/follower channel configurations.
 * Use the static factory methods to create.
 */
export abstract class LinkedChannelSettings {
  /**
   * Configure this channel as a primary in a linked channel pair.
   */
  public static primary(): LinkedChannelSettings {
    return new PrimaryLinkedChannelSettings();
  }

  /**
   * Configure this channel as a follower of a primary channel.
   *
   * @param primaryChannel The primary channel this channel follows. Use a `Channel`
   * instance or import one with `Channel.fromChannelArn()`.
   */
  public static follower(primaryChannel: IChannel): LinkedChannelSettings {
    return new FollowerLinkedChannelSettings(primaryChannel);
  }

  /** @internal */
  public abstract _bind(): CfnChannel.LinkedChannelSettingsProperty;
}

/** @internal */
class PrimaryLinkedChannelSettings extends LinkedChannelSettings {
  public _bind(): CfnChannel.LinkedChannelSettingsProperty {
    return {
      primaryChannelSettings: {
        linkedChannelType: LinkedChannelType.PRIMARY,
      },
    };
  }
}

/** @internal */
class FollowerLinkedChannelSettings extends LinkedChannelSettings {
  constructor(private readonly primaryChannel: IChannel) { super(); }
  public _bind(): CfnChannel.LinkedChannelSettingsProperty {
    return {
      followerChannelSettings: {
        linkedChannelType: LinkedChannelType.FOLLOWER,
        primaryChannelArn: this.primaryChannel.channelArn,
      },
    };
  }
}

/**
 * VPC output settings for the channel.
 * When configured, all output endpoints are created within the specified VPC.
 */
export interface VpcOutputSettings {
  /**
   * The subnets to use for the channel's output endpoints.
   * For STANDARD channels, provide subnets in two different availability zones.
   * For SINGLE_PIPELINE channels, provide at least one subnet.
   */
  readonly subnets: ISubnetRef[];
  /**
   * The security groups to attach to the output VPC network interfaces.
   *
   * @default - VPC default security group
   */
  readonly securityGroups?: ISecurityGroupRef[];
  /**
   * Public address allocation IDs to associate with ENIs created in the output VPC.
   * Must specify one for SINGLE_PIPELINE, two for STANDARD channels.
   * @default - no public addresses
   */
  readonly publicAddressAllocationIds?: string[];
}

/**
 * Shared metric implementation for both real and imported channels.
 */
abstract class ChannelBase extends Resource implements IChannel {
  public abstract readonly channelArn: string;
  public abstract readonly channelId: string;
  public abstract readonly channelInputs?: string[];
  public abstract get channelRef(): ChannelReference;

  /** Collection of grant methods for this channel — start, stop, and update its schedule. */
  public readonly grants: ChannelGrants = ChannelGrants.fromChannel(this);

  /**
   * The channel class, or `undefined` for imported channels (skip pipeline validation).
   * @internal
   */
  protected abstract get _channelClass(): ChannelClass | undefined;

  public metric(metricName: string, pipeline: Pipeline, props?: MetricOptions): Metric {
    if (pipeline === Pipeline.PIPELINE_1 && this._channelClass?.value === ChannelClass.SINGLE_PIPELINE.value) {
      throw new ValidationError(
        lit`SinglePipelineChannelHasNoPipelineOne`,
        'Pipeline.PIPELINE_1 is not available on SINGLE_PIPELINE channels. Use Pipeline.PIPELINE_0, or set channelClass: ChannelClass.STANDARD',
        this,
      );
    }
    return new Metric({
      metricName,
      namespace: 'AWS/MediaLive',
      ...props,
      // Required dimensions applied last so a caller's dimensionsMap can't drop ChannelId/Pipeline.
      dimensionsMap: { ...props?.dimensionsMap, ChannelId: this.channelId, Pipeline: pipeline.toString() },
    });
  }

  public metricActiveAlerts(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('ActiveAlerts', pipeline, { statistic: 'max', ...props });
  }

  public metricNetworkIn(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('NetworkIn', pipeline, { statistic: 'avg', ...props });
  }

  public metricNetworkOut(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('NetworkOut', pipeline, { statistic: 'avg', ...props });
  }

  public metricInputVideoFrameRate(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('InputVideoFrameRate', pipeline, { statistic: 'max', ...props });
  }

  public metricFillMsec(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('FillMsec', pipeline, { statistic: 'max', ...props });
  }

  public metricInputLossSeconds(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('InputLossSeconds', pipeline, { statistic: 'sum', ...props });
  }

  public metricDroppedFrames(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('DroppedFrames', pipeline, { statistic: 'sum', ...props });
  }

  public metricSvqTime(pipeline: Pipeline, props?: MetricOptions): Metric {
    return this.metric('SvqTime', pipeline, { statistic: 'max', ...props });
  }
}

/**
 * Defines an AWS Elemental MediaLive Channel.
 */
@propertyInjectable
export class Channel extends ChannelBase {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.Channel';

  /** Import an existing channel by its ARN. The id is parsed out of the ARN. */
  public static fromChannelArn(scope: Construct, id: string, channelArn: string): IChannel {
    const channelId = extractResourceId(channelArn, 'Channel');

    class Import extends ChannelBase {
      public readonly channelArn = channelArn;
      public readonly channelId = channelId;
      public readonly channelInputs = undefined;
      public get channelRef(): ChannelReference {
        return { channelId: this.channelId, channelArn: this.channelArn };
      }
      // Imported channels are not validated against PIPELINE_1 use because we
      // cannot know how the channel was provisioned.
      protected readonly _channelClass = undefined;
    }
    return new Import(scope, id);
  }

  public readonly channelArn: string;
  public readonly channelId: string;
  public readonly channelInputs?: string[];

  /** The IAM role used by this channel. */
  public readonly role: IRole;

  /** A reference to this Channel resource. */
  public get channelRef(): ChannelReference {
    return { channelId: this.channelId, channelArn: this.channelArn };
  }

  /** @internal */
  protected readonly _channelClass: ChannelClass;

  private readonly inputAttachments: CfnChannel.InputAttachmentProperty[] = [];
  private readonly attachments: InputAttachment[] = [];
  private readonly hasAnywhereSettings: boolean;
  private readonly outputGroups: OutputGroup[] = [];

  /**
   * Whether the caller supplied their own `role`. When true, the channel makes no automatic
   * grants.
   */
  private readonly userProvidedRole: boolean;
  private readonly usesEpochLocking: boolean;

  constructor(scope: Construct, id: string, props: ChannelProps) {
    super(scope, id, {
      physicalName: props.channelName ?? Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 256 }) }),
    });

    addConstructMetadata(this, props);

    if (props.inputs.length < 1) {
      throw new ValidationError(lit`ChannelMinInputs`, 'A channel must have at least one input.', this);
    }
    if (props.outputGroups.length < 1) {
      throw new ValidationError(lit`ChannelMinOutputGroups`, 'A channel must have at least one output group.', this);
    }
    props.inputs.forEach(attachment => {
      this.attachments.push(attachment);
      this.inputAttachments.push(this.buildInputAttachment(attachment));
    });

    const usesEpochLocking = props.globalConfiguration?.outputLocking?._mode() === OutputLockingMode.EPOCH_LOCKING;
    props.outputGroups.forEach(config => {
      const outputGroup = new OutputGroup(config);
      if (usesEpochLocking) {
        outputGroup._setEpochLocking(true);
      }
      this.outputGroups.push(outputGroup);
    });

    // A channel may have at most one MediaConnect Router output group. Detected by output-group
    // settings shape rather than an `instanceof` check, per this module's static-type-check
    // convention.
    const mediaConnectRouterGroupCount = props.outputGroups
      .filter(config => config._bind().mediaConnectRouterGroupSettings !== undefined)
      .length;
    if (mediaConnectRouterGroupCount > 1) {
      throw new ValidationError(
        lit`MultipleMediaConnectRouterGroups`,
        `A channel may have at most one MediaConnect Router output group, but ${mediaConnectRouterGroupCount} were provided.`,
        this,
      );
    }

    const resolvedChannelClass = props.channelClass ?? ChannelClass.SINGLE_PIPELINE;
    this._channelClass = resolvedChannelClass;

    // SRT listener outputs require channel security groups — fail fast at synth.
    const hasSrtListenerOutput = props.outputGroups.some(config => config._hasSrtListenerDestination());
    if (hasSrtListenerOutput && (props.channelSecurityGroups?.length ?? 0) === 0) {
      throw new ValidationError(
        lit`SrtListenerRequiresChannelSecurityGroups`,
        'an SRT output in listener connection mode requires channelSecurityGroups on the channel',
        this,
      );
    }

    this.hasAnywhereSettings = props.anywhereSettings !== undefined;
    this.node.addValidation({ validate: () => this.validateInputs() });

    if (props.linkedChannelSettings && resolvedChannelClass.value !== ChannelClass.SINGLE_PIPELINE.value) {
      throw new ValidationError(lit`LinkedChannelClass`, 'Linked channel settings can only be configured on SINGLE_PIPELINE channels.', this);
    }

    // Epoch locking requires the output timing source to be the input clock. MediaLive rejects any
    // other timing source at deploy, so fail fast at synth.
    if (usesEpochLocking
      && props.globalConfiguration?.outputTimingSource?.value === OutputTimingSource.SYSTEM_CLOCK.value) {
      throw new ValidationError(
        lit`EpochLockingTimingSource`,
        'globalConfiguration.outputTimingSource must be INPUT_CLOCK when using epoch output locking.',
        this,
      );
    }

    // Epoch-locking checks are deferred to synth so they also cover groups added via
    // addOutputGroup()
    this.usesEpochLocking = usesEpochLocking;
    if (usesEpochLocking) {
      this.node.addValidation({ validate: () => this.validateEpochLocking() });
    }

    // Create a default role if not provided. When the caller brings their own role, the channel
    // makes no automatic grants.
    this.userProvidedRole = props.role !== undefined;
    const channelRole = props.role ?? new Role(this, 'Role', {
      assumedBy: new ServicePrincipal('medialive.amazonaws.com', {
        conditions: {
          // Confused-deputy prevention. SourceArn wildcards the channel ID because it's
          // service-generated (pinning it would create a role → channel → role cycle).
          StringEquals: { 'aws:SourceAccount': Aws.ACCOUNT_ID },
          ArnLike: {
            'aws:SourceArn': Stack.of(this).formatArn({
              service: 'medialive',
              resource: 'channel',
              resourceName: '*',
              arnFormat: ArnFormat.COLON_RESOURCE_NAME,
            }),
          },
        },
      }),
      description: 'Role for MediaLive channel',
    });

    const inputSpec = props.inputSpecification ?? InputSpecification.standard();

    const resource = new CfnChannel(this, 'Resource', {
      name: this.physicalName,
      channelClass: resolvedChannelClass.value,
      roleArn: channelRole.roleArn,
      logLevel: (props.logLevel ?? LogLevel.DISABLED).value,
      inferenceSettings: props.inferenceFeedArn ? {
        feedArn: props.inferenceFeedArn,
      } : undefined,
      tags: props.tags ? Object.entries(props.tags).map(([key, value]) => ({ key, value })) : undefined,
      maintenance: props.maintenance ? {
        maintenanceDay: props.maintenance.day.value,
        maintenanceStartTime: props.maintenance.time ?? '02:00',
      } : undefined,
      destinations: Lazy.any({
        produce: () => this.outputGroups
          .flatMap(og => og._bindDestination(resolvedChannelClass.value)),
      }, { omitEmptyArray: true }),
      inputAttachments: Lazy.any({ produce: () => this.inputAttachments }, { omitEmptyArray: true }),
      inputSpecification: inputSpec._bindInputSpecification(),
      encoderSettings: {
        timecodeConfig: {
          source: (props.timecodeConfig?.source ?? TimecodeSource.EMBEDDED).value,
          syncThreshold: props.timecodeConfig?.syncThreshold,
        },
        globalConfiguration: props.globalConfiguration ? {
          initialAudioGain: props.globalConfiguration.initialAudioGain,
          inputEndAction: props.globalConfiguration.inputEndAction?.value,
          outputLockingMode: props.globalConfiguration.outputLocking?._mode(),
          outputTimingSource: props.globalConfiguration.outputTimingSource?.value,
          supportLowFramerateInputs: props.globalConfiguration.supportLowFramerateInputs ? 'ENABLED' : 'DISABLED',
          inputLossBehavior: props.globalConfiguration.inputLossBehavior ? {
            blackFrameMsec: props.globalConfiguration.inputLossBehavior.blackFrame?.toMilliseconds(),
            repeatFrameMsec: props.globalConfiguration.inputLossBehavior.repeatFrame?.toMilliseconds(),
            inputLossImageType: props.globalConfiguration.inputLossBehavior.imageType?.value,
            inputLossImageColor: props.globalConfiguration.inputLossBehavior.imageColor,
            inputLossImageSlate: props.globalConfiguration.inputLossBehavior.imageSlate?._bind(),
          } : undefined,
          outputLockingSettings: props.globalConfiguration.outputLocking?._bind(),
        } : undefined,
        availBlanking: props.availBlanking ? {
          state: (props.availBlanking.state
            ?? (props.availBlanking.image ? AvailBlankingState.ENABLED : AvailBlankingState.DISABLED)).value,
          availBlankingImage: props.availBlanking.image?._bind(),
        } : undefined,
        availConfiguration: (props.availSettings || props.scte35SegmentationScope) ? {
          availSettings: props.availSettings?._bind(),
          scte35SegmentationScope: props.scte35SegmentationScope?.value,
        } : undefined,
        featureActivations: props.featureActivations ? {
          inputPrepareScheduleActions: props.featureActivations.inputPrepareScheduleActions?.value,
          outputStaticImageOverlayScheduleActions: props.featureActivations.outputStaticImageOverlayScheduleActions?.value,
        } : undefined,
        motionGraphicsConfiguration: props.motionGraphicsConfiguration ? {
          motionGraphicsInsertion: (props.motionGraphicsConfiguration.motionGraphicsInsertion ?? MotionGraphicsInsertion.DISABLED).value,
          motionGraphicsSettings: { htmlMotionGraphicsSettings: {} },
        } : undefined,
        nielsenConfiguration: props.nielsenConfiguration ? {
          distributorId: props.nielsenConfiguration.distributorId,
          nielsenPcmToId3Tagging: props.nielsenConfiguration.nielsenPcmToId3Tagging?.value,
        } : undefined,
        thumbnailConfiguration: props.thumbnailConfiguration ? {
          state: (props.thumbnailConfiguration.state ?? ThumbnailState.AUTO).value,
        } : undefined,
        blackoutSlate: props.blackoutSlate ? {
          state: (props.blackoutSlate.state
            ?? (props.blackoutSlate.image ? BlackoutSlateState.ENABLED : BlackoutSlateState.DISABLED)).value,
          blackoutSlateImage: props.blackoutSlate.image?._bind(),
          networkEndBlackout: (props.blackoutSlate.networkEndBlackout
            ?? (props.blackoutSlate.networkEndBlackoutImage ? NetworkEndBlackout.ENABLED : NetworkEndBlackout.DISABLED)).value,
          networkEndBlackoutImage: props.blackoutSlate.networkEndBlackoutImage?._bind(),
          networkId: props.blackoutSlate.networkId,
        } : undefined,
        colorCorrectionSettings: props.colorCorrections ? {
          globalColorCorrections: props.colorCorrections.map(c => ({
            inputColorSpace: c.inputColorSpace.value,
            outputColorSpace: c.outputColorSpace.value,
            uri: c.lut?._bind(),
          })),
        } : undefined,
        videoDescriptions: Lazy.any({
          produce: () => {
            const result: CfnChannel.VideoDescriptionProperty[] = [];
            this.outputGroups.flatMap(og => og._collectEncodes()).forEach(e => {
              const desc = e._bindVideo();
              if (desc?.name && !result.some(d => d.name === desc.name)) result.push(desc);
            });
            return result;
          },
        }, { omitEmptyArray: true }),
        audioDescriptions: Lazy.any({
          produce: () => {
            const result: CfnChannel.AudioDescriptionProperty[] = [];
            this.outputGroups.flatMap(og => og._collectEncodes()).forEach(e => {
              const desc = e._bindAudio();
              if (desc?.name && !result.some(d => d.name === desc.name)) result.push(desc);
            });
            return result;
          },
        }, { omitEmptyArray: true }),
        captionDescriptions: Lazy.any({
          produce: () => {
            const result: CfnChannel.CaptionDescriptionProperty[] = [];
            this.outputGroups.flatMap(og => og._collectEncodes()).forEach(e => {
              const desc = e._bindCaption();
              if (desc?.name && !result.some(d => d.name === desc.name)) result.push(desc);
            });
            return result;
          },
        }, { omitEmptyArray: true }),
        outputGroups: Lazy.any({ produce: () => this.outputGroups.map(og => og._bind()) }, { omitEmptyArray: true }),
      },
      vpc: props.vpc ? {
        subnetIds: props.vpc.subnets.map(s => s.subnetRef.subnetId),
        securityGroupIds: props.vpc.securityGroups?.map(sg => sg.securityGroupRef.securityGroupId),
        publicAddressAllocationIds: props.vpc.publicAddressAllocationIds,
      } : undefined,
      anywhereSettings: props.anywhereSettings ? {
        channelPlacementGroupId: props.anywhereSettings.channelPlacementGroup?.channelPlacementGroupRef.channelPlacementGroupId,
        clusterId: props.anywhereSettings.cluster.clusterRef.clusterId,
      } : undefined,
      channelEngineVersion: props.channelEngineVersion ? {
        version: props.channelEngineVersion,
      } : undefined,
      cdiInputSpecification: inputSpec._bindCdiInputSpecification(),
      linkedChannelSettings: props.linkedChannelSettings?._bind(),
      channelSecurityGroups: props.channelSecurityGroups?.map(sg => sg.inputSecurityGroupRef.inputSecurityGroupId),
    });

    this.channelArn = resource.attrArn;
    this.channelId = resource.ref;
    this.channelInputs = resource.attrInputs;
    this.role = channelRole;

    // Auto-grant permissions only for the channel-managed role. When the caller brings their own
    // role, they define every permission (principal and resource side).
    if (!this.userProvidedRole) {
      // Auto-grant permissions for S3 destinations and S3 input sources
      this.outputGroups.forEach(og => og._grantPermissions(channelRole));
      props.inputs.forEach(attachment => {
        attachment.input._grantPermissions(channelRole);
        attachment.automaticInputFailover?.secondaryInput._grantPermissions(channelRole);
      });
      // Grant the service-role permissions implied by channel-level features (logging, VPC output).
      this.grantServiceRolePermissions(channelRole, props, resource);
    }
  }

  /**
   * Grant the channel's service role the permissions implied by channel-level features.
   *
   * Mirrors the AWS-documented MediaLive trusted-entity requirements:
   * https://docs.aws.amazon.com/medialive/latest/ug/trusted-entity-requirements.html
   *
   * Resource-level scoping is applied where the service supports it; EC2 `Describe*`
   * actions don't support resource-level permissions, so they require `*`.
   */
  private grantServiceRolePermissions(role: IRole, props: ChannelProps, resource: CfnChannel): void {
    const stack = Stack.of(this);

    // Thumbnails — grant unless user explicitly disabled.
    const thumbnailsEnabled = props.thumbnailConfiguration === undefined
      || props.thumbnailConfiguration.state?.value === ThumbnailState.AUTO.value;
    if (thumbnailsEnabled) {
      role.addToPrincipalPolicy(new PolicyStatement({
        actions: ['s3:PutObject'],
        resources: ['*'], // Thumbnail destination can't be scoped
      }));
    }

    // CloudWatch Logs — required when logging is enabled.
    // MediaLive always writes to the fixed log group `ElementalMediaLive`.
    // See https://docs.aws.amazon.com/medialive/latest/ug/working-with-logs.html
    if ((props.logLevel ?? LogLevel.DISABLED).value !== LogLevel.DISABLED.value) {
      const logGroupArn = stack.formatArn({
        service: 'logs',
        resource: 'log-group',
        resourceName: 'ElementalMediaLive',
        arnFormat: ArnFormat.COLON_RESOURCE_NAME,
      });
      Grant.addToPrincipal({
        grantee: role,
        actions: [
          'logs:CreateLogGroup',
          'logs:CreateLogStream',
          'logs:PutLogEvents',
          'logs:PutMetricFilter',
          'logs:PutRetentionPolicy',
          'logs:DescribeLogStreams',
          'logs:DescribeLogGroups',
        ],
        // The group ARN covers group-level actions; the `:*` variant covers the log streams.
        resourceArns: [logGroupArn, `${logGroupArn}:*`],
      });
    }

    // EC2 — required for VPC output. Uses .applyBefore() because MediaLive validates
    // permissions at channel creation time.
    if (props.vpc) {
      const subnetArns = props.vpc.subnets.map(s => stack.formatArn({ service: 'ec2', resource: 'subnet', resourceName: s.subnetRef.subnetId }));
      const securityGroupArns = (props.vpc.securityGroups ?? []).map(sg => stack.formatArn({ service: 'ec2', resource: 'security-group', resourceName: sg.securityGroupRef.securityGroupId }));
      // ENI IDs are generated at runtime — wildcarded but scoped to region/account.
      const networkInterfaceArn = stack.formatArn({ service: 'ec2', resource: 'network-interface', resourceName: '*' });

      // Create/Delete scope to the ENIs plus the subnets and security groups they attach to.
      Grant.addToPrincipal({
        grantee: role,
        actions: [
          'ec2:CreateNetworkInterface',
          'ec2:CreateNetworkInterfacePermission',
          'ec2:DeleteNetworkInterface',
        ],
        resourceArns: [networkInterfaceArn, ...subnetArns, ...securityGroupArns],
      }).applyBefore(resource);

      // Describe* actions don't support resource-level permissions — EC2 requires `*`.
      // See https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-policies-ec2-console.html
      Grant.addToPrincipal({
        grantee: role,
        actions: [
          'ec2:DescribeNetworkInterfaces',
          'ec2:DescribeSubnets',
          'ec2:DescribeSecurityGroups',
          'ec2:DescribeAddresses',
        ],
        resourceArns: ['*'],
      }).applyBefore(resource);

      // Public IP association — ENI target is created at runtime, can't be scoped at synth.
      if (props.vpc.publicAddressAllocationIds !== undefined) {
        Grant.addToPrincipal({
          grantee: role,
          actions: ['ec2:AssociateAddress'],
          resourceArns: ['*'],
        }).applyBefore(resource);
      }
    }

    // ESAM avail mode — grant read access to the POIS password parameter, if configured.
    props.availSettings?._grantPermissions(role);

    // Input-loss slate image — grant read access when the slate is sourced from an S3 bucket.
    props.globalConfiguration?.inputLossBehavior?.imageSlate?._grantRead(role);

    // Avail-blanking and blackout-slate images — grant read access when sourced from S3.
    props.availBlanking?.image?._grantRead(role);
    props.blackoutSlate?.image?._grantRead(role);
    props.blackoutSlate?.networkEndBlackoutImage?._grantRead(role);

    // Color-correction LUT files — grant read access when sourced from an S3 bucket.
    props.colorCorrections?.forEach(c => c.lut?._grantRead(role));
  }

  /**
   * Attach an input to this channel.
   */
  public addInput(attachment: InputAttachment): void {
    this.attachments.push(attachment);
    this.inputAttachments.push(this.buildInputAttachment(attachment));
    // Skip auto-grants when the caller owns the role.
    if (!this.userProvidedRole) {
      attachment.input._grantPermissions(this.role);
      attachment.automaticInputFailover?.secondaryInput._grantPermissions(this.role);
    }
  }

  /**
   * Validate automatic-input-failover pairs against MediaLive's create-time rules. The secondary
   * input is rejected by the service at deploy unless it is also attached to the channel, so we
   * fail fast at synth.
   */
  /**
   * Synth-time input validations, run over all attached inputs (initial + addInput()).
   */
  private validateInputs(): string[] {
    const errors: string[] = [];
    const anywhereOnlyInputTypes: string[] = [InputType.SDI, InputType.SMPTE_2110_RECEIVER_GROUP, InputType.MULTICAST];

    for (const attachment of this.attachments) {
      const inputClass = attachment.input.inputClass;
      if (inputClass && inputClass !== this._channelClass.value) {
        errors.push(`Input '${attachment.input.node.id}' has input class '${inputClass}' which is incompatible`
          + ` with channel class '${this._channelClass.value}'.`);
      }

      const inputType = attachment.input.inputType;
      if (!this.hasAnywhereSettings && inputType && anywhereOnlyInputTypes.includes(inputType)) {
        errors.push(`Input '${attachment.input.node.id}' has type '${inputType}' which requires anywhereSettings to be configured on the channel.`);
      }

      const failover = attachment.automaticInputFailover;
      if (failover && !this.attachments.some(other => other.input === failover.secondaryInput)) {
        errors.push(`Input '${attachment.input.node.id}' declares automatic input failover to secondary input`
          + ` '${failover.secondaryInput.node.id}', but that secondary input is not attached to the channel.`
          + " Add the secondary input to the channel's 'inputs' as its own attachment.");
      }
    }

    return errors;
  }

  private buildInputAttachment(attachment: InputAttachment): CfnChannel.InputAttachmentProperty {
    // Default to LOOP for file-based inputs (MP4, TS) — they're typically used as slates or test content
    const isFileInput = attachment.input.inputType === 'MP4_FILE' || attachment.input.inputType === 'TS_FILE';
    const sourceEndBehavior = attachment.sourceEndBehavior ?? (isFileInput ? SourceEndBehavior.LOOP : SourceEndBehavior.CONTINUE);

    const failover = attachment.automaticInputFailover;
    if (failover) {
      // The failover pair must share an input class — MediaLive rejects mismatched classes.
      const primaryClass = attachment.input.inputClass;
      const secondaryClass = failover.secondaryInput.inputClass;
      if (primaryClass && secondaryClass && primaryClass !== secondaryClass) {
        throw new ValidationError(
          lit`FailoverInputClassMismatch`,
          `Automatic input failover requires the primary and secondary inputs to have the same input class, got '${primaryClass}' and '${secondaryClass}'.`,
          this,
        );
      }
    }

    return {
      inputId: attachment.input.inputId,
      inputAttachmentName: attachment.inputAttachmentName,
      automaticInputFailoverSettings: failover ? {
        secondaryInputId: failover.secondaryInput.inputId,
        inputPreference: failover.inputPreference?.value,
        errorClearTimeMsec: failover.errorClearTime?.toMilliseconds(),
        // MediaLive requires at least one failover condition — default to input loss.
        failoverConditions: (failover.failoverConditions ?? [FailoverCondition.inputLoss()]).map(c => c._bind()),
      } : undefined,
      inputSettings: {
        sourceEndBehavior: sourceEndBehavior.value,
        inputFilter: (attachment.inputFilter ?? InputFilter.AUTO).value,
        filterStrength: attachment.filterStrength ?? 1,
        deblockFilter: attachment.deblockFilter ? 'ENABLED' : 'DISABLED',
        denoiseFilter: attachment.denoiseFilter ? 'ENABLED' : 'DISABLED',
        smpte2038DataPreference: (attachment.smpte2038DataPreference ?? Smpte2038DataPreference.IGNORE).value,
        audioSelectors: attachment.audioSelectors?.map(s => s._bind()),
        captionSelectors: attachment.captionSelectors?.map(s => s._bind()),
        scte35Pid: attachment.scte35Pid,
        videoSelector: attachment.videoSelector ? {
          colorSpace: attachment.videoSelector.colorSpace?.value,
          colorSpaceUsage: attachment.videoSelector.colorSpaceUsage?.value,
          colorSpaceSettings: attachment.videoSelector.hdr10 ? {
            hdr10Settings: {
              maxCll: attachment.videoSelector.hdr10.maxContentLightLevel,
              maxFall: attachment.videoSelector.hdr10.maxFrameAverageLightLevel,
            },
          } : undefined,
          selectorSettings: attachment.videoSelector.selectBy?._bind(),
        } : undefined,
        networkInputSettings: attachment.networkInputSettings ? {
          serverValidation: attachment.networkInputSettings.serverValidation?.value,
          hlsInputSettings: attachment.networkInputSettings.hlsInputSettings ? {
            bandwidth: attachment.networkInputSettings.hlsInputSettings.bandwidth?.toBps(),
            bufferSegments: attachment.networkInputSettings.hlsInputSettings.bufferSegments,
            retries: attachment.networkInputSettings.hlsInputSettings.retries,
            retryInterval: attachment.networkInputSettings.hlsInputSettings.retryInterval?.toSeconds(),
            scte35Source: attachment.networkInputSettings.hlsInputSettings.scte35Source?.value,
          } : undefined,
          multicastInputSettings: attachment.networkInputSettings.multicastSourceIp
            ? { sourceIpAddress: attachment.networkInputSettings.multicastSourceIp }
            : undefined,
        } : undefined,
      },
      logicalInterfaceNames: attachment.logicalInterfaceNames,
    };
  }

  /**
   * Add an output group to this channel.
   *
   * Outputs are declared up front via the `outputs` prop on the group configuration.
   */
  public addOutputGroup(config: OutputGroupConfiguration): void {
    const outputGroup = new OutputGroup(config);
    if (this.usesEpochLocking) {
      outputGroup._setEpochLocking(true);
    }
    this.outputGroups.push(outputGroup);
    // Skip auto-grants when the caller owns the role.
    if (!this.userProvidedRole) {
      outputGroup._grantPermissions(this.role);
    }
  }

  /** Synth-time epoch-locking checks for configs MediaLive rejects at deploy. */
  private validateEpochLocking(): string[] {
    // An explicit SYSTEM_CLOCK program-date-time clock contradicts epoch locking.
    const clockError = this.outputGroups.some(og => og._hasExplicitSystemClock())
      ? ['an HLS output group programDateTimeClock must be INITIALIZE_FROM_OUTPUT_TIMECODE when using epoch output locking']
      : [];

    // H.264 is the only codec that falls back to INITIALIZE_FROM_SOURCE when framerate is omitted;
    // epoch locking requires an explicit rate. H.265/AV1/Frame Capture always carry one.
    const offenders = [...new Set(this.outputGroups
      .flatMap(og => og._collectEncodes())
      .filter(encode => encode._videoCodecType() === VideoCodecType.H264 && !encode._hasExplicitFramerate())
      .map(encode => encode.name))];
    const framerateError = offenders.length > 0
      ? ['epoch output locking requires an explicit frame rate on H.264 video encodes '
        + `(e.g. Framerate.FPS_30); these encodes follow the source frame rate: ${offenders.join(', ')}`]
      : [];

    return [...clockError, ...framerateError];
  }
}
