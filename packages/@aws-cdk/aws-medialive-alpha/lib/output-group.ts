export * from './destinations';
export * from './enums';
export * from './outputs';

import type { IChannel as IMediaPackageV2Channel } from '@aws-cdk/aws-mediapackagev2-alpha';
import type { Duration, SecretValue } from 'aws-cdk-lib';
import { UnscopedValidationError, Token } from 'aws-cdk-lib';
import type { IRole, IRoleRef } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type {
  OutputDestination, S3OutputDestination,
  UdpOutputDestination,
} from './destinations';
import { MediaPackageV2Destination, MediaPackageV2EndpointId, MediaConnectRouterSettings } from './destinations';
import type { EncodeConfiguration } from './encode-configuration';
import {
  HlsMode,
  HlsInputLossAction,
  RtmpAuthenticationScheme,
  HlsClientCache,
  HlsCodecSpecification,
  HlsDirectoryStructure,
  HlsDiscontinuityTags,
  HlsId3SegmentTaggingState,
  HlsIFrameOnlyPlaylists,
  HlsIncompleteSegmentBehavior,
  HlsManifestCompression,
  HlsManifestDurationFormat,
  HlsOutputSelection,
  HlsProgramDateTime,
  HlsProgramDateTimeClock,
  HlsRedundantManifest,
  HlsSegmentationMode,
  HlsStreamInfResolution,
  HlsTsFileMode,
  HttpTransferMode,
  HlsTimedMetadataId3Frame,
  UdpInputLossAction,
  SegmentLengthUnits,
  Id3Behavior,
  KlvBehavior,
  NielsenId3Behavior,
  Scte35Type,
  TimedMetadataId3Frame,
  TimedMetadataPassthrough,
  UdpTimedMetadataId3Frame,
  SrtInputLossAction,
} from './enums';
import type {
  S3CannedAcl,
  HlsEncryptionType, HlsIvInManifest, HlsIvSource,
  HlsCaptionLanguageSetting,
  RtmpCacheFullBehavior, RtmpCaptionData, RtmpInputLossAction,
  RtmpIncludeFillerNalUnits,
  MsSmoothAudioOnlyTimecodeControl, MsSmoothCertificateMode,
  MsSmoothEventIdMode, MsSmoothEventStopBehavior,
  MsSmoothInputLossAction, MsSmoothSegmentationMode,
  MsSmoothSparseTrackType, MsSmoothStreamManifestBehavior,
  MsSmoothTimestampOffsetMode,
  HlsAdMarkers, RtmpAdMarkers,
} from './enums';
import {
  MediaPackageV2Output, HlsOutput, UdpOutput, ArchiveOutput,
  RtmpOutput, SrtOutput, CmafIngestOutput, FrameCaptureOutput,
  MsSmoothOutput, MediaConnectRouterOutput, toDestinationId,
} from './outputs';
import type {
  Output,
  MediaConnectRouterOutputDefinition,
  MediaPackageV2OutputDefinition, HlsOutputDefinition,
  UdpOutputDefinition, ArchiveOutputDefinition,
  RtmpOutputDefinition, SrtOutputDefinition,
  CmafIngestOutputDefinition, FrameCaptureOutputDefinition,
  MsSmoothOutputDefinition,
} from './outputs';
import type { Segment } from './shared';

/** @internal */
function validateDestinationCount(
  groupName: string,
  channelClass: string,
  destinationCount: number,
  additionalCount: number,
  allowAdditional: boolean,
): void {
  const isStandard = channelClass === 'STANDARD';
  const requiredPrimary = isStandard ? 2 : 1;
  if (destinationCount !== requiredPrimary) {
    throw new UnscopedValidationError(
      lit`DestinationCount`,
      `Output group '${groupName}' requires exactly ${requiredPrimary}`
      + ` primary destination(s) for a ${channelClass} channel, but ${destinationCount} provided.`,
    );
  }
  if (additionalCount > 0 && !allowAdditional) {
    throw new UnscopedValidationError(
      lit`AdditionalDestinations`,
      `Output group '${groupName}' does not support additional destinations.`,
    );
  }
  const maxAdditional = isStandard ? 2 : 1;
  if (additionalCount > maxAdditional) {
    throw new UnscopedValidationError(
      lit`AdditionalDestinationCount`,
      `Output group '${groupName}' allows at most ${maxAdditional}`
      + ` additional destination(s) for a ${channelClass} channel, but ${additionalCount} provided.`,
    );
  }
}

// =============================================================================
// OutputGroupConfiguration (abstract base + subclasses)
// =============================================================================

/**
 * Configuration for an output group. Use the static factory methods to create.
 */
export abstract class OutputGroupConfiguration {
  /**
   * Create a MediaPackage V2 output group that delivers to a single channel, auto-mapping each
   * pipeline to a MediaPackage ingest endpoint based on the channel class.
   */
  public static mediaPackageV2(props: MediaPackageV2OutputGroupProps): OutputGroupConfiguration {
    return new MediaPackageV2OutputGroupConfiguration(props, props.channel, undefined);
  }
  /**
   * Create a MediaPackage V2 output group with explicit per-pipeline destinations (channel +
   * endpoint per pipeline). Use for cross-region delivery or pinning a pipeline to an endpoint.
   */
  public static mediaPackageV2PerPipeline(props: MediaPackageV2PerPipelineOutputGroupProps): OutputGroupConfiguration {
    return new MediaPackageV2OutputGroupConfiguration(props, undefined, props.destinations);
  }
  /**
   * Create a MediaConnect Router output group, delivering each channel pipeline to a MediaConnect
   * Router. Transit encryption defaults to AUTOMATIC; override per pipeline via `routerSettings`.
   *
   * The downstream wiring (which router input each pipeline feeds) is configured on the
   * MediaConnect side, referencing this output group by `name` and the pipeline id.
   */
  public static mediaConnectRouter(props: MediaConnectRouterOutputGroupProps): OutputGroupConfiguration {
    return new MediaConnectRouterOutputGroupConfiguration(props);
  }
  /** Create an HLS output group configuration. */
  public static hls(props: HlsOutputGroupProps): OutputGroupConfiguration {
    return new HlsOutputGroupConfiguration(props);
  }
  /** Create a UDP output group configuration. */
  public static udp(props: UdpOutputGroupProps): OutputGroupConfiguration {
    return new UdpOutputGroupConfiguration(props);
  }
  /** Create an Archive (S3) output group configuration. */
  public static archive(props: ArchiveOutputGroupProps): OutputGroupConfiguration {
    return new ArchiveOutputGroupConfiguration(props);
  }
  /** Create an RTMP output group configuration. */
  public static rtmp(props: RtmpOutputGroupProps): OutputGroupConfiguration {
    return new RtmpOutputGroupConfiguration(props);
  }
  /** Create an SRT output group configuration. */
  public static srt(props: SrtOutputGroupProps): OutputGroupConfiguration {
    return new SrtOutputGroupConfiguration(props);
  }
  /** Create a CMAF Ingest output group configuration. */
  public static cmafIngest(props: CmafIngestOutputGroupProps): OutputGroupConfiguration {
    return new CmafIngestOutputGroupConfiguration(props);
  }
  /**
   * Create a Frame Capture output group configuration.
   *
   * A channel that includes a Frame Capture output group must also include
   * a separate video output group (e.g. Archive, HLS, UDP). Frame Capture
   * cannot be the channel's only output group.
   */
  public static frameCapture(props: FrameCaptureOutputGroupProps): OutputGroupConfiguration {
    return new FrameCaptureOutputGroupConfiguration(props);
  }
  /** Create an MS Smooth output group configuration. */
  public static msSmooth(props: MsSmoothOutputGroupProps): OutputGroupConfiguration {
    return new MsSmoothOutputGroupConfiguration(props);
  }

  /**
   * The name of this output group.
   * @internal
   */
  public abstract readonly _name: string;

  /** @internal */
  public abstract _bind(): CfnChannel.OutputGroupSettingsProperty;
  /** @internal */
  public abstract _bindDestination(channelClass: string): CfnChannel.OutputDestinationProperty[];
  /** @internal - Grant permissions to the channel role for resources referenced by destinations. */
  public _grantPermissions(_role: IRoleRef): void {}
  /** @internal - Create the initial Output instances from the config's output definitions. */
  public abstract _createInitialOutputs(): Output[];
  /**
   * Whether any output in this group has an SRT listener-mode destination. Used by the channel to
   * decide whether channel security groups are required, without triggering destination binding.
   * @internal
   */
  public _hasSrtListenerDestination(): boolean { return false; }
  /**
   * HLS uses this to select a compatible default program-date-time clock
   * (the service requires INITIALIZE_FROM_OUTPUT_TIMECODE for epoch locking).
   * @internal
   */
  public _setEpochLocking(_active: boolean): void {}
  /**
   * Whether this group explicitly requests an HLS program date time clock of SYSTEM_CLOCK, which
   * is incompatible with epoch output locking. Only HLS groups can return true.
   * @internal
   */
  public _hasExplicitSystemClock(): boolean { return false; }
}

// =============================================================================
// Output group props interfaces
// =============================================================================

/**
 * Maps a captions channel to an ISO 693-2 language code.
 */
export interface CaptionLanguageMapping {
  /** The closed caption channel number (1-4). */
  readonly captionChannel: number;
  /** A three-character ISO 639-2 language code. */
  readonly languageCode: string;
  /** The textual description of the language. */
  readonly languageDescription: string;
}

/**
 * Maps a captions channel to an ISO 639-2 language code for a CMAF Ingest output group.
 *
 * Unlike `CaptionLanguageMapping`, the CMAF Ingest variant has no language description.
 */
export interface CmafCaptionLanguageMapping {
  /** The closed caption channel number (1-4). */
  readonly captionChannel: number;
  /** A three-character ISO 639-2 language code. */
  readonly languageCode: string;
}

/** Properties for HLS static key encryption. */
export interface HlsStaticKeyProps {
  /**
   * The URL of the license server that serves the static key.
   *
   * Required — MediaLive rejects this without a server URL, even though the underlying
   * CloudFormation property is typed as optional.
   */
  readonly keyProviderServerUrl: string;
  /** The static key value as a 32-character hexadecimal string. */
  readonly staticKeyValue: SecretValue;
}

/** Properties for HLS S3 CDN settings. */
export interface HlsS3CdnProps {
  /** The S3 canned ACL to apply to each output. @default - no canned ACL */
  readonly cannedAcl?: S3CannedAcl;
}

/** Properties for HLS Basic PUT CDN settings. */
export interface HlsBasicPutCdnProps {
  /** The number of seconds to wait before retrying a connection to the CDN. @default 1 */
  readonly connectionRetryInterval?: number;
  /** The size of the file cache for streaming outputs. @default Duration.seconds(300) */
  readonly filecacheDuration?: Duration;
  /** The number of retry attempts. @default 10 */
  readonly numRetries?: number;
  /** The number of seconds to wait before restarting after a failure. @default 1 */
  readonly restartDelay?: number;
}

/** Properties for HLS Akamai CDN settings. */
export interface HlsAkamaiCdnProps {
  /** The number of seconds to wait before retrying a connection to the CDN. @default 1 */
  readonly connectionRetryInterval?: number;
  /** The size of the file cache for streaming outputs. @default Duration.seconds(300) */
  readonly filecacheDuration?: Duration;
  /** Specifies whether to use chunked transfer encoding. @default HttpTransferMode.NON_CHUNKED */
  readonly httpTransferMode?: HttpTransferMode;
  /** The number of retry attempts. @default 10 */
  readonly numRetries?: number;
  /** The number of seconds to wait before restarting after a failure. @default 1 */
  readonly restartDelay?: number;
  /** The salt for Akamai authentication. @default - no salt */
  readonly salt?: string;
  /** The token for Akamai authentication. @default - no token */
  readonly token?: string;
}

/** Properties for HLS WebDAV CDN settings. */
export interface HlsWebdavCdnProps {
  /** The number of seconds to wait before retrying a connection to the CDN. @default 1 */
  readonly connectionRetryInterval?: number;
  /** The size of the file cache for streaming outputs. @default Duration.seconds(300) */
  readonly filecacheDuration?: Duration;
  /** Specifies whether to use chunked transfer encoding. @default HttpTransferMode.NON_CHUNKED */
  readonly httpTransferMode?: HttpTransferMode;
  /** The number of retry attempts. @default 10 */
  readonly numRetries?: number;
  /** The number of seconds to wait before restarting after a failure. @default 1 */
  readonly restartDelay?: number;
}

/**
 * Key provider settings for HLS encryption.
 */
export class HlsKeyProviderSettings {
  /** Use a static key for HLS encryption. */
  public static staticKey(props: HlsStaticKeyProps): HlsKeyProviderSettings {
    return new HlsKeyProviderSettings({
      staticKeySettings: {
        keyProviderServer: { uri: props.keyProviderServerUrl },
        staticKeyValue: props.staticKeyValue.unsafeUnwrap(), // Safe usage: rendered as CFN dynamic reference
      },
    });
  }

  private readonly config: CfnChannel.KeyProviderSettingsProperty;
  private constructor(config: CfnChannel.KeyProviderSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.KeyProviderSettingsProperty { return this.config; }
}

/**
 * CDN settings for HLS output groups.
 */
export class HlsCdnSettings {
  /** Use Amazon S3 as the CDN for HLS output. */
  public static s3(props?: HlsS3CdnProps): HlsCdnSettings {
    return new HlsCdnSettings({ hlsS3Settings: { cannedAcl: props?.cannedAcl?.value } });
  }
  /** Use a basic HTTP PUT for HLS output. */
  public static basicPut(props?: HlsBasicPutCdnProps): HlsCdnSettings {
    return new HlsCdnSettings({
      hlsBasicPutSettings: {
        connectionRetryInterval: props?.connectionRetryInterval ?? 1,
        filecacheDuration: props?.filecacheDuration?.toSeconds() ?? 300,
        numRetries: props?.numRetries ?? 10,
        restartDelay: props?.restartDelay ?? 1,
      },
    });
  }
  /** Use Akamai as the CDN for HLS output. */
  public static akamai(props?: HlsAkamaiCdnProps): HlsCdnSettings {
    return new HlsCdnSettings({
      hlsAkamaiSettings: {
        connectionRetryInterval: props?.connectionRetryInterval ?? 1,
        filecacheDuration: props?.filecacheDuration?.toSeconds() ?? 300,
        numRetries: props?.numRetries ?? 10,
        restartDelay: props?.restartDelay ?? 1,
        httpTransferMode: (props?.httpTransferMode ?? HttpTransferMode.NON_CHUNKED).value,
        salt: props?.salt,
        token: props?.token,
      },
    });
  }
  /** Use WebDAV as the CDN for HLS output. */
  public static webdav(props?: HlsWebdavCdnProps): HlsCdnSettings {
    return new HlsCdnSettings({
      hlsWebdavSettings: {
        connectionRetryInterval: props?.connectionRetryInterval ?? 1,
        filecacheDuration: props?.filecacheDuration?.toSeconds() ?? 300,
        numRetries: props?.numRetries ?? 10,
        restartDelay: props?.restartDelay ?? 1,
        httpTransferMode: (props?.httpTransferMode ?? HttpTransferMode.NON_CHUNKED).value,
      },
    });
  }

  private readonly config: CfnChannel.HlsCdnSettingsProperty;
  private constructor(config: CfnChannel.HlsCdnSettingsProperty) { this.config = config; }

  /** @internal */
  public _bind(): CfnChannel.HlsCdnSettingsProperty { return this.config; }
}

/**
 * Common properties shared by the MediaPackage V2 output group variants.
 *
 * @see https://docs.aws.amazon.com/medialive/latest/ug/creating-mediapackage-output-group.html
 */
export interface MediaPackageV2OutputGroupBaseProps {
  /** The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally. */
  readonly name: string;
  /**
   * Configure additional destinations to fan out the output to extra MediaPackage V2
   * channels, for example for cross-region delivery or backup packaging.
   * These correspond to Destination 3/4 in the AWS console. Each additional destination is a
   * single, explicit entry (channel + endpoint), independent of the channel class.
   *
   * @see https://docs.aws.amazon.com/medialive/latest/ug/creating-mediapackage-output-group.html
   * @default - no additional destinations
   */
  readonly additionalDestinations?: MediaPackageV2Destination[];
  /**
   * The length of each media segment.
   * @default - Segment.seconds(1)
   */
  readonly segment?: Segment;
  /**
   * The ID3 behavior.
   * @default Id3Behavior.DISABLED
   */
  readonly id3Behavior?: Id3Behavior;
  /**
   * The KLV behavior.
   * @default KlvBehavior.NO_PASSTHROUGH
   */
  readonly klvBehavior?: KlvBehavior;
  /**
   * The Nielsen ID3 behavior.
   * @default NielsenId3Behavior.NO_PASSTHROUGH
   */
  readonly nielsenId3Behavior?: NielsenId3Behavior;
  /**
   * The SCTE-35 type.
   * @default Scte35Type.SCTE_35_WITHOUT_SEGMENTATION
   */
  readonly scte35Type?: Scte35Type;
  /**
   * The timed metadata ID3 frame.
   * @default TimedMetadataId3Frame.NONE
   */
  readonly timedMetadataId3Frame?: TimedMetadataId3Frame;
  /**
   * The timed metadata interval.
   * @default Duration.seconds(10)
   */
  readonly timedMetadataId3Period?: Duration;
  /**
   * Whether timed metadata is passed through.
   * @default TimedMetadataPassthrough.DISABLED
   */
  readonly timedMetadataPassthrough?: TimedMetadataPassthrough;
  /**
   * Caption language mappings for the MediaPackage V2 output.
   * @default - no caption language mappings
   */
  readonly captionLanguageMappings?: CaptionLanguageMapping[];
  /**
   * The outputs for this output group.
   *
   * MediaPackage V2 uses CMAF ingest which requires one track per output.
   * Create a separate output for each encode (e.g. one for HD video, one for SD video, one for audio).
   * Do NOT put multiple encodes in a single output.
   */
  readonly outputs: MediaPackageV2OutputDefinition[];
}

/**
 * Properties for a MediaPackage V2 output group.
 *
 * Reference a single MediaPackage V2 channel; MediaLive wires each channel pipeline to a
 * MediaPackage ingest endpoint automatically (one for SINGLE_PIPELINE, both for STANDARD). For
 * per-pipeline control — for example sending pipeline 0 to a specific endpoint of a channel in
 * another region — use `OutputGroupConfiguration.mediaPackageV2PerPipeline()` instead.
 */
export interface MediaPackageV2OutputGroupProps extends MediaPackageV2OutputGroupBaseProps {
  /**
   * The MediaPackage V2 channel to deliver to. MediaLive maps the channel pipelines to the
   * channel's ingest endpoints automatically based on the channel class.
   */
  readonly channel: IMediaPackageV2Channel;
}

/**
 * Properties for a MediaPackage V2 output group with explicit per-pipeline destinations.
 *
 * Use this when you need to control the channel and endpoint each pipeline delivers to — for
 * example cross-region delivery, or pinning pipeline 0 to a specific endpoint.
 */
export interface MediaPackageV2PerPipelineOutputGroupProps extends MediaPackageV2OutputGroupBaseProps {
  /**
   * The primary MediaPackage V2 destinations — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations.
   */
  readonly destinations: MediaPackageV2Destination[];
}

/**
 * Properties for a MediaConnect Router output group.
 *
 * Delivers each channel pipeline to a MediaConnect Router. The downstream routing (which router
 * input each pipeline feeds) is configured on the MediaConnect side, referencing this group by
 * `name` and pipeline id.
 *
 * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-medialive-channel-mediaconnectroutergroupsettings.html
 */
export interface MediaConnectRouterOutputGroupProps {
  /** The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally. */
  readonly name: string;
  /**
   * The Availability Zones in which to write output to the MediaConnect Router. Provide exactly
   * one AZ for a `SINGLE_PIPELINE` channel, or two (one per pipeline) for a `STANDARD` channel.
   */
  readonly availabilityZones: string[];
  /**
   * Transit-encryption settings, applied per channel pipeline.
   * @default - AUTOMATIC (service-managed) transit encryption on every pipeline
   */
  readonly routerSettings?: MediaConnectRouterSettings;
  /**
   * The outputs for this output group.
   *
   * MediaConnect Router outputs use an MPEG-TS container, so each output may multiplex one video
   * encode and one or more audio encodes (as with UDP).
   */
  readonly outputs: MediaConnectRouterOutputDefinition[];
}

/**
 * Properties for an HLS output group.
 */
export interface HlsOutputGroupProps {
  /**
   * The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally.
   */
  readonly name: string;
  /**
   * The destinations for this output group — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations.
   */
  readonly destinations: OutputDestination[];
  /**
   * The length of each media segment. HLS supports whole-second segments only.
   * @default - Segment.seconds(2)
   */
  readonly segment?: Segment;
  /**
   * The number of segments to retain in the destination directory (LIVE mode only).
   * @default 21
   */
  readonly keepSegments?: number;
  /**
   * The maximum number of segments in the media manifest (LIVE mode only).
   * @default 10
   */
  readonly indexNSegments?: number;
  /**
   * The output mode — LIVE or VOD.
   * @default HlsMode.LIVE
   */
  readonly mode?: HlsMode;
  /**
   * The minimum segment length. HLS supports whole-second segments only.
   * @default - service default
   */
  readonly minSegment?: Segment;
  /**
   * Action to take when the input is lost.
   * @default HlsInputLossAction.EMIT_OUTPUT
   */
  readonly inputLossAction?: HlsInputLossAction;
  /**
   * Chooses one or more ad marker types to pass SCTE35 signals through to this group of Apple HLS outputs.
   * @default - no ad markers
   */
  readonly adMarkers?: HlsAdMarkers[];
  /**
   * A partial URI prefix that will be prepended to each output in the media .m3u8 file.
   * @default - no base URL content prefix
   */
  readonly baseUrlContent?: string;
  /**
   * Optional base URL content for pipeline 1 if different from pipeline 0.
   * @default - no base URL content 1
   */
  readonly baseUrlContent1?: string;
  /**
   * A partial URI prefix that will be prepended to each output in the media .m3u8 file for the manifest.
   * @default - no base URL manifest prefix
   */
  readonly baseUrlManifest?: string;
  /**
   * Optional base URL manifest for pipeline 1 if different from pipeline 0.
   * @default - no base URL manifest 1
   */
  readonly baseUrlManifest1?: string;
  /**
   * A mapping of up to 4 captions channels to captions languages.
   * Meaningful only if captionLanguageSetting is set to INSERT.
   * @default - no caption language mappings
   */
  readonly captionLanguageMappings?: CaptionLanguageMapping[];
  /**
   * Applies only to 608 embedded output captions.
   * @default - service default
   */
  readonly captionLanguageSetting?: HlsCaptionLanguageSetting;
  /**
   * When set to DISABLED, sets the #EXT-X-ALLOW-CACHE:no tag in the manifest.
   * @default HlsClientCache.ENABLED
   */
  readonly clientCache?: HlsClientCache;
  /**
   * The specification to use (RFC-6381 or the default RFC-4281) during m3u8 playlist generation.
   * @default HlsCodecSpecification.RFC_4281
   */
  readonly codecSpecification?: HlsCodecSpecification;
  /**
   * A 128-bit, 16-byte hex value represented by a 32-character text string used as the IV for encryption.
   * Used with encryptionType when ivSource is set to EXPLICIT.
   * @default - no constant IV
   */
  readonly constantIv?: string;
  /**
   * Places segments in subdirectories.
   * @default HlsDirectoryStructure.SINGLE_DIRECTORY
   */
  readonly directoryStructure?: HlsDirectoryStructure;
  /**
   * Specifies whether to insert EXT-X-DISCONTINUITY tags in the HLS child manifests.
   * @default HlsDiscontinuityTags.INSERT
   */
  readonly discontinuityTags?: HlsDiscontinuityTags;
  /**
   * Encrypts the segments with the specified encryption scheme.
   * @default - no encryption
   */
  readonly encryptionType?: HlsEncryptionType;
  /**
   * Settings to configure the CDN for the HLS output.
   * @default - service default
   */
  readonly hlsCdnSettings?: HlsCdnSettings;
  /**
   * State of HLS ID3 Segment Tagging.
   * @default HlsId3SegmentTaggingState.DISABLED
   */
  readonly hlsId3SegmentTagging?: HlsId3SegmentTaggingState;
  /**
   * Whether to create an I-frame-only manifest.
   * @default HlsIFrameOnlyPlaylists.DISABLED
   */
  readonly iFrameOnlyPlaylists?: HlsIFrameOnlyPlaylists;
  /**
   * Specifies whether to include the final (incomplete) segment in the media output.
   * @default HlsIncompleteSegmentBehavior.AUTO
   */
  readonly incompleteSegmentBehavior?: HlsIncompleteSegmentBehavior;
  /**
   * Whether the IV is listed in the manifest.
   * @default - service default
   */
  readonly ivInManifest?: HlsIvInManifest;
  /**
   * Whether the IV follows the segment number or is explicit.
   * @default - service default
   */
  readonly ivSource?: HlsIvSource;
  /**
   * Specifies how the key is represented in the resource identified by the URI.
   * @default - service default
   */
  readonly keyFormat?: string;
  /**
   * Either a single positive integer version value or a slash-delimited list of version values (1/2/3).
   * @default - service default
   */
  readonly keyFormatVersions?: string;
  /**
   * The key provider settings for HLS encryption.
   * @default - no key provider settings
   */
  readonly keyProviderSettings?: HlsKeyProviderSettings;
  /**
   * When set to GZIP, compresses HLS playlist.
   * @default HlsManifestCompression.NONE
   */
  readonly manifestCompression?: HlsManifestCompression;
  /**
   * Indicates whether the output manifest should use floating point or integer values for segment duration.
   * @default HlsManifestDurationFormat.FLOATING_POINT
   */
  readonly manifestDurationFormat?: HlsManifestDurationFormat;
  /**
   * Controls which manifests and segments are generated.
   * @default HlsOutputSelection.MANIFESTS_AND_SEGMENTS
   */
  readonly outputSelection?: HlsOutputSelection;
  /**
   * Includes or excludes the EXT-X-PROGRAM-DATE-TIME tag in .m3u8 manifest files.
   * @default HlsProgramDateTime.INCLUDE
   */
  readonly programDateTime?: HlsProgramDateTime;
  /**
   * Specifies the algorithm used to drive the HLS EXT-X-PROGRAM-DATE-TIME clock.
   *
   * @default - HlsProgramDateTimeClock.SYSTEM_CLOCK, or INITIALIZE_FROM_OUTPUT_TIMECODE with epoch locking
   */
  readonly programDateTimeClock?: HlsProgramDateTimeClock;
  /**
   * The period of insertion of the EXT-X-PROGRAM-DATE-TIME entry.
   * @default Duration.minutes(10)
   */
  readonly programDateTimePeriod?: Duration;
  /**
   * Whether the master manifest includes information about both pipelines.
   * @default HlsRedundantManifest.DISABLED
   */
  readonly redundantManifest?: HlsRedundantManifest;
  /**
   * The segmentation mode.
   * @default HlsSegmentationMode.USE_SEGMENT_DURATION
   */
  readonly segmentationMode?: HlsSegmentationMode;
  /**
   * The number of segments to write to a subdirectory before starting a new one.
   * @default 10000
   */
  readonly segmentsPerSubdirectory?: number;
  /**
   * Whether to include or exclude the RESOLUTION attribute for a video in the EXT-X-STREAM-INF tag.
   * @default HlsStreamInfResolution.INCLUDE
   */
  readonly streamInfResolution?: HlsStreamInfResolution;
  /**
   * Indicates the ID3 frame that has the timecode.
   * @default HlsTimedMetadataId3Frame.PRIV
   */
  readonly timedMetadataId3Frame?: HlsTimedMetadataId3Frame;
  /**
   * The timed metadata interval.
   * @default Duration.seconds(10)
   */
  readonly timedMetadataId3Period?: Duration;
  /**
   * Provides an extra delta offset to fine tune the timestamps.
   * @default - service default
   */
  readonly timestampDelta?: Duration;
  /**
   * Whether to emit segmented files or a single file.
   * @default HlsTsFileMode.SEGMENTED_FILES
   */
  readonly tsFileMode?: HlsTsFileMode;
  /**
   * The outputs for this HLS output group.
   * @default - no initial outputs
   */
  readonly outputs?: HlsOutputDefinition[];
}

/**
 * Properties for a UDP output group.
 */
export interface UdpOutputGroupProps {
  /**
   * The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally.
   */
  readonly name: string;
  /**
   * The destinations for this output group — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations.
   */
  readonly destinations: UdpOutputDestination[];
  /**
   * The output buffering. Applied at millisecond granularity.
   * @default - service default
   */
  readonly buffer?: Duration;
  /**
   * Action to take when the input is lost.
   * @default UdpInputLossAction.EMIT_PROGRAM
   */
  readonly inputLossAction?: UdpInputLossAction;
  /**
   * Indicates the ID3 frame that has the timecode.
   * @default UdpTimedMetadataId3Frame.PRIV
   */
  readonly timedMetadataId3Frame?: UdpTimedMetadataId3Frame;
  /**
   * The timed metadata interval.
   * @default - Duration.seconds(10)
   */
  readonly timedMetadataId3Period?: Duration;
  /**
   * The outputs for this UDP output group.
   * @default - no initial outputs
   */
  readonly outputs?: UdpOutputDefinition[];
}

/** Properties for an Archive (S3) output group. */
export interface ArchiveOutputGroupProps {
  /**
   * The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally.
   */
  readonly name: string;
  /**
   * The destinations for this output group — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations.
   */
  readonly destinations: S3OutputDestination[];
  /**
   * The duration of each archive file (rollover interval).
   * @default Duration.seconds(300)
   */
  readonly rolloverInterval?: Duration;
  /**
   * The S3 canned ACL to apply to each archive output.
   * @default - no canned ACL
   */
  readonly archiveS3CannedAcl?: S3CannedAcl;
  /**
   * The outputs for this Archive output group.
   * @default - no initial outputs
   */
  readonly outputs?: ArchiveOutputDefinition[];
}

/** Properties for an RTMP output group. */
export interface RtmpOutputGroupProps {
  /**
   * The name of this output group.
   */
  readonly name: string;
  /**
   * The authentication scheme for the RTMP connection.
   * @default RtmpAuthenticationScheme.COMMON
   */
  readonly authenticationScheme?: RtmpAuthenticationScheme;
  /**
   * The delay before restarting after a streaming output failure. A value of
   * `Duration.seconds(0)` means never restart.
   * @default Duration.seconds(1)
   */
  readonly restartDelay?: Duration;
  /**
   * Choose the ad marker type for this output group.
   * @default - no ad markers
   */
  readonly adMarkers?: RtmpAdMarkers[];
  /**
   * Controls behavior when the content cache fills up.
   * @default - service default
   */
  readonly cacheFullBehavior?: RtmpCacheFullBehavior;
  /**
   * The cache length, in seconds, that is used to calculate buffer size.
   * @default - service default
   */
  readonly cacheLength?: Duration;
  /**
   * Controls the types of data that pass to onCaptionInfo outputs.
   * @default - service default
   */
  readonly captionData?: RtmpCaptionData;
  /**
   * Controls whether filler NAL units are included in the output.
   * @default - service default
   */
  readonly includeFillerNalUnits?: RtmpIncludeFillerNalUnits;
  /**
   * Controls the behavior of this RTMP group if the input becomes unavailable.
   * @default - service default
   */
  readonly inputLossAction?: RtmpInputLossAction;
  /**
   * The outputs for this RTMP output group. Each output includes its own RTMP destination.
   */
  readonly outputs: RtmpOutputDefinition[];
}

/** Properties for an SRT output group. */
export interface SrtOutputGroupProps {
  /**
   * The name of this output group.
   */
  readonly name: string;
  /**
   * Controls the behavior of this SRT group if the input becomes unavailable.
   * @default SrtInputLossAction.EMIT_PROGRAM
   */
  readonly inputLossAction?: SrtInputLossAction;
  /**
   * The outputs for this SRT output group. Each output includes its own SRT destination.
   */
  readonly outputs: SrtOutputDefinition[];
}

/** Properties for a CMAF Ingest output group. */
export interface CmafIngestOutputGroupProps {
  /**
   * The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally.
   */
  readonly name: string;
  /**
   * The primary CMAF ingest destinations — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations to utilise both
   * pipelines for redundancy.
   */
  readonly destinations: OutputDestination[];
  /**
   * Configure additional destinations to fan out the CMAF ingest output to extra
   * endpoints, for example for cross-region delivery or backup packaging.
   *
   * Standard channels support up to 2 additional destinations.
   * Single pipeline channels support 1 additional destination.
   *
   * @default - no additional destinations
   */
  readonly additionalDestinations?: OutputDestination[];
  /**
   * The length of each media segment.
   * @default - Segment.seconds(1)
   */
  readonly segment?: Segment;
  /**
   * The ID3 behavior for the CMAF ingest output.
   * @default Id3Behavior.DISABLED
   */
  readonly id3Behavior?: Id3Behavior;
  /**
   * The name modifier for ID3 metadata.
   * @default - service default
   */
  readonly id3NameModifier?: string;
  /**
   * The KLV behavior for the CMAF ingest output.
   * @default KlvBehavior.NO_PASSTHROUGH
   */
  readonly klvBehavior?: KlvBehavior;
  /**
   * The name modifier for KLV metadata.
   * @default - service default
   */
  readonly klvNameModifier?: string;
  /**
   * The Nielsen ID3 behavior for the CMAF ingest output.
   * @default NielsenId3Behavior.NO_PASSTHROUGH
   */
  readonly nielsenId3Behavior?: NielsenId3Behavior;
  /**
   * The name modifier for Nielsen ID3 metadata.
   * @default - service default
   */
  readonly nielsenId3NameModifier?: string;
  /**
   * The name modifier for SCTE-35 messages.
   * @default - service default
   */
  readonly scte35NameModifier?: string;
  /**
   * The SCTE-35 type for the CMAF ingest output.
   * @default Scte35Type.SCTE_35_WITHOUT_SEGMENTATION
   */
  readonly scte35Type?: Scte35Type;
  /**
   * The number of milliseconds to delay the output from the second pipeline.
   * @default - service default
   */
  readonly sendDelayMs?: number;
  /**
   * Indicates the ID3 frame that has the timecode.
   * @default TimedMetadataId3Frame.NONE
   */
  readonly timedMetadataId3Frame?: TimedMetadataId3Frame;
  /**
   * The timed metadata interval.
   * @default - Duration.seconds(10)
   */
  readonly timedMetadataId3Period?: Duration;
  /**
   * Whether timed metadata is passed through.
   * @default TimedMetadataPassthrough.DISABLED
   */
  readonly timedMetadataPassthrough?: TimedMetadataPassthrough;
  /**
   * Maps captions channels to languages for this CMAF Ingest output group.
   * @default - no caption language mappings
   */
  readonly captionLanguageMappings?: CmafCaptionLanguageMapping[];
  /**
   * The outputs for this CMAF Ingest output group. Each output should contain a single encode.
   */
  readonly outputs: CmafIngestOutputDefinition[];
}

/** Properties for a Frame Capture output group. */
export interface FrameCaptureOutputGroupProps {
  /**
   * The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally.
   */
  readonly name: string;
  /**
   * The destinations for this output group — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations.
   */
  readonly destinations: S3OutputDestination[];
  /**
   * The S3 canned ACL to apply to each frame capture output.
   * @default - no canned ACL
   */
  readonly frameCaptureS3CannedAcl?: S3CannedAcl;
  /**
   * The outputs for this Frame Capture output group.
   * @default - no initial outputs
   */
  readonly outputs?: FrameCaptureOutputDefinition[];
}

/** Properties for an MS Smooth output group. */
export interface MsSmoothOutputGroupProps {
  /**
   * The name of this output group. Used as the destination reference ID. Underscores are normalised to hyphens internally.
   */
  readonly name: string;
  /**
   * The destinations for this output group — one per pipeline.
   *
   * Array position determines the pipeline mapping:
   * - `destinations[0]` → Pipeline 0
   * - `destinations[1]` → Pipeline 1 (STANDARD channels only)
   *
   * For a SINGLE_PIPELINE channel, provide exactly 1 destination.
   * For a STANDARD channel, provide exactly 2 destinations.
   */
  readonly destinations: OutputDestination[];
  /**
   * The value of the Acquisition Point Identity element used in each message placed in the sparse track.
   * @default - service default
   */
  readonly acquisitionPointId?: string;
  /**
   * If set to passthrough for an audio-only output, the fragment absolute time is set to the current timecode.
   * @default - service default
   */
  readonly audioOnlyTimecodeControl?: MsSmoothAudioOnlyTimecodeControl;
  /**
   * If set to VERIFY_AUTHENTICITY, verifies the HTTPS certificate chain to a trusted CA.
   * @default - service default
   */
  readonly certificateMode?: MsSmoothCertificateMode;
  /**
   * The number of seconds to wait before retrying the connection to the IIS server if the connection is lost.
   * @default - service default
   */
  readonly connectionRetryInterval?: Duration;
  /**
   * The Microsoft Smooth channel ID that is sent to the IIS server.
   * @default - service default
   */
  readonly eventId?: string;
  /**
   * Specifies whether to send a channel ID to the IIS server.
   * @default - service default
   */
  readonly eventIdMode?: MsSmoothEventIdMode;
  /**
   * When set to SEND_EOS, sends an EOS signal to an IIS server when stopping the channel.
   * @default - service default
   */
  readonly eventStopBehavior?: MsSmoothEventStopBehavior;
  /**
   * The size, in seconds, of the file cache for streaming outputs.
   * @default - service default
   */
  readonly filecacheDuration?: Duration;
  /**
   * The length, in seconds, of mp4 fragments to generate.
   * @default - service default
   */
  readonly fragmentLength?: Duration;
  /**
   * A parameter that controls output group behavior on an input loss.
   * @default - service default
   */
  readonly inputLossAction?: MsSmoothInputLossAction;
  /**
   * The number of retry attempts.
   * @default 10
   */
  readonly numRetries?: number;
  /**
   * The number of seconds before initiating a restart due to output failure.
   * @default - Duration.seconds(1)
   */
  readonly restartDelay?: Duration;
  /**
   * The segmentation mode.
   * @default - service default
   */
  readonly segmentationMode?: MsSmoothSegmentationMode;
  /**
   * The number of milliseconds to delay the output from the second pipeline.
   * @default - service default
   */
  readonly sendDelayMs?: number;
  /**
   * If set to SCTE_35, uses incoming SCTE-35 messages to generate a sparse track.
   * @default - service default
   */
  readonly sparseTrackType?: MsSmoothSparseTrackType;
  /**
   * When set to SEND, sends a stream manifest so that the publishing point doesn't start until all streams start.
   * @default - service default
   */
  readonly streamManifestBehavior?: MsSmoothStreamManifestBehavior;
  /**
   * The timestamp offset for the channel. Used only if timestampOffsetMode is set to USE_CONFIGURED_OFFSET.
   * @default - service default
   */
  readonly timestampOffset?: string;
  /**
   * The type of timestamp date offset to use.
   * @default - service default
   */
  readonly timestampOffsetMode?: MsSmoothTimestampOffsetMode;
  /**
   * The outputs for this MS Smooth output group.
   * @default - no initial outputs
   */
  readonly outputs?: MsSmoothOutputDefinition[];
}

// =============================================================================
// OutputGroupConfiguration subclasses
// =============================================================================

/** @internal */
class MediaPackageV2OutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(
    private readonly props: MediaPackageV2OutputGroupBaseProps,
    private readonly autoChannel?: IMediaPackageV2Channel,
    private readonly explicitDestinations?: MediaPackageV2Destination[],
  ) {
    super();
    this._name = props.name;
  }

  /**
   * The primary destinations. For the auto variant they are derived from the channel class
   * (one per pipeline, each with its endpoint); for the custom variant they are user-supplied.
   */
  private primaryDestinations(channelClass: string): MediaPackageV2Destination[] {
    if (this.autoChannel) {
      return channelClass === 'STANDARD'
        ? [
          MediaPackageV2Destination.channel(this.autoChannel, MediaPackageV2EndpointId.ENDPOINT_1),
          MediaPackageV2Destination.channel(this.autoChannel, MediaPackageV2EndpointId.ENDPOINT_2),
        ]
        : [MediaPackageV2Destination.channel(this.autoChannel, MediaPackageV2EndpointId.ENDPOINT_1)];
    }
    return this.explicitDestinations ?? [];
  }

  public _createInitialOutputs(): Output[] {
    return this.props.outputs.map(def => new MediaPackageV2Output(def, this._name));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    const additional = this.props.additionalDestinations ?? [];
    return {
      mediaPackageGroupSettings: {
        destination: { destinationRefId: toDestinationId(this.props.name) },
        mediapackageV2GroupSettings: {
          additionalDestinations: additional.length > 0
            ? additional.map((_dest, i) => ({
              destination: { destinationRefId: `${toDestinationId(this.props.name)}-additional-${i}` },
            }))
            : undefined,
          segmentLength: this.props.segment?._length() ?? 1,
          segmentLengthUnits: this.props.segment?._units() ?? SegmentLengthUnits.SECONDS,
          id3Behavior: (this.props.id3Behavior ?? Id3Behavior.DISABLED).value,
          klvBehavior: (this.props.klvBehavior ?? KlvBehavior.NO_PASSTHROUGH).value,
          nielsenId3Behavior: (this.props.nielsenId3Behavior ?? NielsenId3Behavior.NO_PASSTHROUGH).value,
          scte35Type: (this.props.scte35Type ?? Scte35Type.SCTE_35_WITHOUT_SEGMENTATION).value,
          timedMetadataId3Frame: (this.props.timedMetadataId3Frame ?? TimedMetadataId3Frame.NONE).value,
          timedMetadataId3Period: this.props.timedMetadataId3Period?.toSeconds() ?? 10,
          timedMetadataPassthrough: (this.props.timedMetadataPassthrough ?? TimedMetadataPassthrough.DISABLED).value,
          captionLanguageMappings: this.props.captionLanguageMappings,
        },
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    const additional = this.props.additionalDestinations ?? [];
    const primaries = this.primaryDestinations(_channelClass);
    // For the auto variant the primary count is correct by construction; for the custom variant
    // this enforces the per-pipeline count the user supplied.
    validateDestinationCount(this._name, _channelClass, primaries.length, additional.length, true);

    // Primary destinations share the output group's destination ID
    const primarySettings: CfnChannel.MediaPackageOutputDestinationSettingsProperty[] =
      primaries.map(dest => dest._bind());
    const result: CfnChannel.OutputDestinationProperty[] = [
      { id: toDestinationId(this.props.name), mediaPackageSettings: primarySettings },
    ];

    // Each additional destination is a separate top-level Destinations entry
    additional.forEach((dest, i) => {
      result.push({
        id: `${toDestinationId(this.props.name)}-additional-${i}`,
        mediaPackageSettings: [dest._bind()],
      });
    });

    return result;
  }
  public override _grantPermissions(role: IRole): void {
    const primaries = this.autoChannel
      ? [MediaPackageV2Destination.channel(this.autoChannel)]
      : (this.explicitDestinations ?? []);
    [...primaries, ...(this.props.additionalDestinations ?? [])].forEach(d => d._grantPermissions(role));
  }
}

/** @internal */
class MediaConnectRouterOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: MediaConnectRouterOutputGroupProps) {
    super();
    this._name = props.name;
    if (props.outputs.length > 1) {
      throw new UnscopedValidationError(
        lit`MediaConnectRouterSingleOutput`,
        `MediaConnect Router output group '${props.name}' may contain at most one output, but ${props.outputs.length} were provided`,
      );
    }
  }

  public _createInitialOutputs(): Output[] {
    return this.props.outputs.map(def => new MediaConnectRouterOutput(def, this._name));
  }

  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      mediaConnectRouterGroupSettings: {
        availabilityZones: this.props.availabilityZones,
      },
    };
  }

  public _bindDestination(channelClass: string): CfnChannel.OutputDestinationProperty[] {
    const pipelineCount = channelClass === 'STANDARD' ? 2 : 1;
    const azCount = this.props.availabilityZones.length;
    if (azCount !== pipelineCount) {
      throw new UnscopedValidationError(
        lit`MediaConnectRouterAvailabilityZones`,
        `MediaConnect Router output group '${this._name}' requires exactly ${pipelineCount} availabilityZone(s)`
        + ` for a ${channelClass} channel, but ${azCount} provided`,
      );
    }
    const settings = this.props.routerSettings ?? MediaConnectRouterSettings.shared();
    return [
      { id: toDestinationId(this._name), mediaConnectRouterSettings: settings._bind(pipelineCount) },
    ];
  }

  public override _grantPermissions(role: IRole): void {
    this.props.routerSettings?._secrets().forEach(secret => secret.grantRead(role));
  }
}

/** @internal */
class HlsOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  private epochLocking = false;
  constructor(private readonly props: HlsOutputGroupProps) {
    super();
    this._name = props.name;
  }
  public override _setEpochLocking(active: boolean): void {
    this.epochLocking = active;
  }
  /**
   * The default program date time clock. Epoch output locking requires
   * INITIALIZE_FROM_OUTPUT_TIMECODE; otherwise SYSTEM_CLOCK.
   */
  private _defaultProgramDateTimeClock(): HlsProgramDateTimeClock {
    return this.epochLocking
      ? HlsProgramDateTimeClock.INITIALIZE_FROM_OUTPUT_TIMECODE
      : HlsProgramDateTimeClock.SYSTEM_CLOCK;
  }
  /** Whether the user explicitly set an incompatible SYSTEM_CLOCK program-date-time clock. @internal */
  public _hasExplicitSystemClock(): boolean {
    return this.props.programDateTimeClock?.value === HlsProgramDateTimeClock.SYSTEM_CLOCK.value;
  }
  public _createInitialOutputs(): Output[] {
    return (this.props.outputs ?? []).map(def => new HlsOutput(def));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      hlsGroupSettings: {
        destination: { destinationRefId: toDestinationId(this.props.name) },
        segmentLength: this.props.segment?._toSeconds() ?? 2,
        keepSegments: this.props.keepSegments ?? 21,
        indexNSegments: this.props.indexNSegments ?? 10,
        mode: (this.props.mode ?? HlsMode.LIVE).value,
        minSegmentLength: this.props.minSegment?._toSeconds(),
        inputLossAction: (this.props.inputLossAction ?? HlsInputLossAction.EMIT_OUTPUT).value,
        adMarkers: this.props.adMarkers?.map(m => m.value),
        baseUrlContent: this.props.baseUrlContent,
        baseUrlContent1: this.props.baseUrlContent1,
        baseUrlManifest: this.props.baseUrlManifest,
        baseUrlManifest1: this.props.baseUrlManifest1,
        captionLanguageMappings: this.props.captionLanguageMappings,
        captionLanguageSetting: this.props.captionLanguageSetting?.value,
        clientCache: (this.props.clientCache ?? HlsClientCache.ENABLED).value,
        codecSpecification: (this.props.codecSpecification ?? HlsCodecSpecification.RFC_4281).value,
        constantIv: this.props.constantIv,
        directoryStructure: (this.props.directoryStructure ?? HlsDirectoryStructure.SINGLE_DIRECTORY).value,
        discontinuityTags: (this.props.discontinuityTags ?? HlsDiscontinuityTags.INSERT).value,
        encryptionType: this.props.encryptionType?.value,
        hlsCdnSettings: this.props.hlsCdnSettings?._bind(),
        hlsId3SegmentTagging: (this.props.hlsId3SegmentTagging ?? HlsId3SegmentTaggingState.DISABLED).value,
        iFrameOnlyPlaylists: (this.props.iFrameOnlyPlaylists ?? HlsIFrameOnlyPlaylists.DISABLED).value,
        incompleteSegmentBehavior: (this.props.incompleteSegmentBehavior ?? HlsIncompleteSegmentBehavior.AUTO).value,
        ivInManifest: this.props.ivInManifest?.value,
        ivSource: this.props.ivSource?.value,
        keyFormat: this.props.keyFormat,
        keyFormatVersions: this.props.keyFormatVersions,
        keyProviderSettings: this.props.keyProviderSettings?._bind(),
        manifestCompression: (this.props.manifestCompression ?? HlsManifestCompression.NONE).value,
        manifestDurationFormat: (this.props.manifestDurationFormat ?? HlsManifestDurationFormat.FLOATING_POINT).value,
        outputSelection: (this.props.outputSelection ?? HlsOutputSelection.MANIFESTS_AND_SEGMENTS).value,
        programDateTime: (this.props.programDateTime ?? HlsProgramDateTime.INCLUDE).value,
        programDateTimeClock: (this.props.programDateTimeClock ?? this._defaultProgramDateTimeClock()).value,
        programDateTimePeriod: this.props.programDateTimePeriod?.toSeconds() ?? 600,
        redundantManifest: (this.props.redundantManifest ?? HlsRedundantManifest.DISABLED).value,
        segmentationMode: (this.props.segmentationMode ?? HlsSegmentationMode.USE_SEGMENT_DURATION).value,
        segmentsPerSubdirectory: this.props.segmentsPerSubdirectory ?? 10_000,
        streamInfResolution: (this.props.streamInfResolution ?? HlsStreamInfResolution.INCLUDE).value,
        timedMetadataId3Frame: (this.props.timedMetadataId3Frame ?? HlsTimedMetadataId3Frame.PRIV).value,
        timedMetadataId3Period: this.props.timedMetadataId3Period?.toSeconds() ?? 10,
        timestampDeltaMilliseconds: this.props.timestampDelta?.toMilliseconds(),
        tsFileMode: (this.props.tsFileMode ?? HlsTsFileMode.SEGMENTED_FILES).value,
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    validateDestinationCount(this._name, _channelClass, this.props.destinations.length, 0, false);
    const settings = this.props.destinations.map(d => {
      const bound = d._bind();
      if (bound.url && !Token.isUnresolved(bound.url) && bound.url.endsWith('/')) {
        throw new UnscopedValidationError(
          lit`HlsDestinationTrailingSlash`,
          `HLS output group '${this._name}' destination URL must be a file prefix, not a folder (got '${bound.url}'). Remove the trailing slash.`,
        );
      }
      // An https destination pushes to a CDN/web server, which requires hlsCdnSettings.
      if (bound.url && !Token.isUnresolved(bound.url) && bound.url.startsWith('https://')
        && this.props.hlsCdnSettings === undefined) {
        throw new UnscopedValidationError(
          lit`HlsHttpsDestinationRequiresCdnSettings`,
          `HLS output group '${this._name}' uses an https destination URL, which requires hlsCdnSettings to be set (e.g. HlsCdnSettings.basicPut(), .akamai(), or .webdav()).`,
        );
      }
      return bound;
    });
    return [{ id: toDestinationId(this.props.name), settings }];
  }
  public override _grantPermissions(role: IRole): void {
    this.props.destinations.forEach(d => d._grantPermissions(role));
  }
}

/** @internal */
class UdpOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: UdpOutputGroupProps) {
    super();
    this._name = props.name;
  }
  public _createInitialOutputs(): Output[] {
    return (this.props.outputs ?? []).map(def => new UdpOutput(def, this.props));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      udpGroupSettings: {
        inputLossAction: (this.props.inputLossAction ?? UdpInputLossAction.EMIT_PROGRAM).value,
        timedMetadataId3Frame: (this.props.timedMetadataId3Frame ?? UdpTimedMetadataId3Frame.PRIV).value,
        timedMetadataId3Period: this.props.timedMetadataId3Period?.toSeconds() ?? 10,
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    validateDestinationCount(this._name, _channelClass, this.props.destinations.length, 0, false);
    const settings = this.props.destinations.map(d => d._bind());
    return [{ id: toDestinationId(this.props.name), settings }];
  }
  public override _grantPermissions(role: IRole): void {
    this.props.destinations.forEach(d => d._grantPermissions(role));
  }
}

/** @internal */
class ArchiveOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: ArchiveOutputGroupProps) {
    super();
    this._name = props.name;
    // An archive group must contain at least one output with a video encode (raw-container
    // outputs are audio-only, so a group of only raw outputs is rejected by the service).
    const outputs = props.outputs ?? [];
    if (outputs.length > 0 && !outputs.some(o => o.encodes.some(e => e._bindVideo() !== undefined))) {
      throw new UnscopedValidationError(
        lit`ArchiveGroupRequiresVideo`,
        `Archive output group '${props.name}' must contain at least one output with a video encode`,
      );
    }
  }
  public _createInitialOutputs(): Output[] {
    return (this.props.outputs ?? []).map(def => new ArchiveOutput(def));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      archiveGroupSettings: {
        destination: { destinationRefId: toDestinationId(this.props.name) },
        rolloverInterval: this.props.rolloverInterval?.toSeconds() ?? 300,
        archiveCdnSettings: this.props.archiveS3CannedAcl ? {
          archiveS3Settings: { cannedAcl: this.props.archiveS3CannedAcl.value },
        } : undefined,
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    validateDestinationCount(this._name, _channelClass, this.props.destinations.length, 0, false);
    const settings = this.props.destinations.map(d => d._bind());
    return [{ id: toDestinationId(this.props.name), settings }];
  }
  public override _grantPermissions(role: IRole): void {
    this.props.destinations.forEach(d => d._grantPermissions(role));
  }
}

/** @internal */
class RtmpOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: RtmpOutputGroupProps) {
    super();
    this._name = props.name;
  }
  public _createInitialOutputs(): Output[] {
    return this.props.outputs.map(def => new RtmpOutput(def, this._name));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      rtmpGroupSettings: {
        authenticationScheme: (this.props.authenticationScheme ?? RtmpAuthenticationScheme.COMMON).value,
        restartDelay: this.props.restartDelay?.toSeconds() ?? 1,
        adMarkers: this.props.adMarkers?.map(m => m.value),
        cacheFullBehavior: this.props.cacheFullBehavior?.value,
        cacheLength: this.props.cacheLength?.toSeconds(),
        captionData: this.props.captionData?.value,
        includeFillerNalUnits: this.props.includeFillerNalUnits?.value,
        inputLossAction: this.props.inputLossAction?.value,
      },
    };
  }
  public _bindDestination(channelClass: string): CfnChannel.OutputDestinationProperty[] {
    const expected = channelClass === 'STANDARD' ? 2 : 1;
    const seenIds = new Set<string>();
    return this.props.outputs.map((def) => {
      if (def.destinations.length !== expected) {
        throw new UnscopedValidationError(
          lit`RtmpDestinationCount`,
          `RTMP output '${def.outputName}' in group '${this._name}' requires exactly ${expected} destination(s) (one per pipeline) for a ${channelClass} channel, got ${def.destinations.length}`,
        );
      }
      const id = toDestinationId(def.outputName);
      if (id === '') {
        throw new UnscopedValidationError(
          lit`RtmpDestinationIdEmpty`,
          `RTMP output in group '${this._name}' outputName must contain at least one alphanumeric character`,
        );
      }
      if (seenIds.has(id)) {
        throw new UnscopedValidationError(
          lit`RtmpDestinationIdCollision`,
          `RTMP outputs in group '${this._name}' produce the same destination id '${id}' after sanitising output names; rename the outputs so they differ by more than '_' vs '-'`,
        );
      }
      seenIds.add(id);
      return { id, settings: def.destinations.map(d => d._bind()) };
    });
  }
  public override _grantPermissions(role: IRole): void {
    this.props.outputs.forEach(o => o.destinations.forEach(d => d._grantPermissions(role)));
  }
}

/** @internal */
class SrtOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  private readonly _outputs: SrtOutput[];
  constructor(private readonly props: SrtOutputGroupProps) {
    super();
    this._name = props.name;
    this._outputs = props.outputs.map(def => new SrtOutput(def, this._name));
  }
  public _createInitialOutputs(): Output[] {
    return this._outputs;
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      srtGroupSettings: {
        inputLossAction: (this.props.inputLossAction ?? SrtInputLossAction.EMIT_PROGRAM).value,
      },
    };
  }
  public _bindDestination(channelClass: string): CfnChannel.OutputDestinationProperty[] {
    const expected = channelClass === 'STANDARD' ? 2 : 1;
    const seenIds = new Set<string>();
    return this.props.outputs.map((def) => {
      if (def.destinations.length !== expected) {
        throw new UnscopedValidationError(
          lit`SrtDestinationCount`,
          `SRT output '${def.outputName}' in group '${this._name}' requires exactly ${expected} destination(s) (one per pipeline) for a ${channelClass} channel, got ${def.destinations.length}`,
        );
      }
      const id = toDestinationId(def.outputName);
      if (id === '') {
        throw new UnscopedValidationError(
          lit`SrtDestinationIdEmpty`,
          `SRT output in group '${this._name}' outputName must contain at least one alphanumeric character`,
        );
      }
      if (seenIds.has(id)) {
        throw new UnscopedValidationError(
          lit`SrtDestinationIdCollision`,
          `SRT outputs in group '${this._name}' produce the same destination id '${id}' after sanitising output names; rename the outputs so they differ by more than '_' vs '-'`,
        );
      }
      seenIds.add(id);
      return { id, srtSettings: def.destinations.map(d => d._bind()) };
    });
  }
  public override _grantPermissions(role: IRole): void {
    this.props.outputs.forEach(o => o.destinations.forEach(d => d._grantPermissions(role)));
  }
  public override _hasSrtListenerDestination(): boolean {
    return this.props.outputs.some(o => o.destinations.some(d => d._bind().connectionMode === 'LISTENER'));
  }
}

/** @internal */
class CmafIngestOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: CmafIngestOutputGroupProps) {
    super();
    this._name = props.name;
    // Eagerly reject destination URLs that don't end with '/' — this is API misuse that always
    // fails at deploy, so fail fast at construction rather than at synth.
    [...props.destinations, ...(props.additionalDestinations ?? [])].forEach(d => {
      const { url } = d._bind();
      if (url && !Token.isUnresolved(url) && !url.endsWith('/')) {
        throw new UnscopedValidationError(
          lit`CmafIngestDestinationTrailingSlash`,
          `CMAF Ingest output group '${this._name}' destination URL path must end with '/' (got '${url}').`,
        );
      }
    });
  }
  public _createInitialOutputs(): Output[] {
    return this.props.outputs.map(def => new CmafIngestOutput(def));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      cmafIngestGroupSettings: {
        destination: { destinationRefId: toDestinationId(this.props.name) },
        segmentLength: this.props.segment?._length() ?? 1,
        segmentLengthUnits: this.props.segment?._units() ?? SegmentLengthUnits.SECONDS,
        id3Behavior: (this.props.id3Behavior ?? Id3Behavior.DISABLED).value,
        id3NameModifier: this.props.id3NameModifier,
        klvBehavior: (this.props.klvBehavior ?? KlvBehavior.NO_PASSTHROUGH).value,
        klvNameModifier: this.props.klvNameModifier,
        nielsenId3Behavior: (this.props.nielsenId3Behavior ?? NielsenId3Behavior.NO_PASSTHROUGH).value,
        nielsenId3NameModifier: this.props.nielsenId3NameModifier,
        scte35NameModifier: this.props.scte35NameModifier,
        scte35Type: (this.props.scte35Type ?? Scte35Type.SCTE_35_WITHOUT_SEGMENTATION).value,
        sendDelayMs: this.props.sendDelayMs,
        timedMetadataId3Frame: (this.props.timedMetadataId3Frame ?? TimedMetadataId3Frame.NONE).value,
        timedMetadataId3Period: this.props.timedMetadataId3Period?.toSeconds() ?? 10,
        timedMetadataPassthrough: (this.props.timedMetadataPassthrough ?? TimedMetadataPassthrough.DISABLED).value,
        captionLanguageMappings: this.props.captionLanguageMappings,
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    const additional = this.props.additionalDestinations ?? [];
    validateDestinationCount(this._name, _channelClass, this.props.destinations.length, additional.length, true);

    const primarySettings = this.props.destinations.map(d => d._bind());
    const result: CfnChannel.OutputDestinationProperty[] = [
      { id: toDestinationId(this.props.name), settings: primarySettings },
    ];

    additional.forEach((dest, i) => {
      result.push({
        id: `${toDestinationId(this.props.name)}-additional-${i}`,
        settings: [dest._bind()],
      });
    });

    return result;
  }
  public override _grantPermissions(role: IRole): void {
    [...this.props.destinations, ...(this.props.additionalDestinations ?? [])].forEach(d => d._grantPermissions(role));
  }
}

/** @internal */
class FrameCaptureOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: FrameCaptureOutputGroupProps) {
    super();
    this._name = props.name;
  }
  public _createInitialOutputs(): Output[] {
    return (this.props.outputs ?? []).map(def => new FrameCaptureOutput(def));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      frameCaptureGroupSettings: {
        destination: { destinationRefId: toDestinationId(this.props.name) },
        frameCaptureCdnSettings: this.props.frameCaptureS3CannedAcl ? {
          frameCaptureS3Settings: { cannedAcl: this.props.frameCaptureS3CannedAcl.value },
        } : undefined,
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    validateDestinationCount(this._name, _channelClass, this.props.destinations.length, 0, false);
    const settings = this.props.destinations.map(d => d._bind());
    return [{ id: toDestinationId(this.props.name), settings }];
  }
  public override _grantPermissions(role: IRole): void {
    this.props.destinations.forEach(d => d._grantPermissions(role));
  }
}

/** @internal */
class MsSmoothOutputGroupConfiguration extends OutputGroupConfiguration {
  public override readonly _name: string;
  constructor(private readonly props: MsSmoothOutputGroupProps) {
    super();
    this._name = props.name;

    // All outputs in an MS Smooth group share the group's single destination, so MediaLive
    // distinguishes them by nameModifier alone — it must be set and unique across outputs.
    // Otherwise this fails at deploy with "Outputs must have unique destinations".
    const outputs = props.outputs ?? [];
    if (outputs.length > 1) {
      const seenModifiers = new Set<string>();
      outputs.forEach(def => {
        if (!def.nameModifier) {
          throw new UnscopedValidationError(
            lit`MsSmoothNameModifierRequired`,
            `MS Smooth output group '${this._name}' has multiple outputs sharing one destination;`
            + ` output '${def.outputName}' must set a unique 'nameModifier'.`,
          );
        }
        if (seenModifiers.has(def.nameModifier)) {
          throw new UnscopedValidationError(
            lit`MsSmoothNameModifierCollision`,
            `MS Smooth output group '${this._name}' has more than one output with nameModifier`
            + ` '${def.nameModifier}'; each output's nameModifier must be unique within the group.`,
          );
        }
        seenModifiers.add(def.nameModifier);
      });
    }
  }
  public _createInitialOutputs(): Output[] {
    return (this.props.outputs ?? []).map(def => new MsSmoothOutput(def));
  }
  public _bind(): CfnChannel.OutputGroupSettingsProperty {
    return {
      msSmoothGroupSettings: {
        destination: { destinationRefId: toDestinationId(this.props.name) },
        acquisitionPointId: this.props.acquisitionPointId,
        audioOnlyTimecodeControl: this.props.audioOnlyTimecodeControl?.value,
        certificateMode: this.props.certificateMode?.value,
        connectionRetryInterval: this.props.connectionRetryInterval?.toSeconds(),
        eventId: this.props.eventId,
        eventIdMode: this.props.eventIdMode?.value,
        eventStopBehavior: this.props.eventStopBehavior?.value,
        filecacheDuration: this.props.filecacheDuration?.toSeconds(),
        fragmentLength: this.props.fragmentLength?.toSeconds(),
        inputLossAction: this.props.inputLossAction?.value,
        numRetries: this.props.numRetries ?? 10,
        restartDelay: this.props.restartDelay?.toSeconds() ?? 1,
        segmentationMode: this.props.segmentationMode?.value,
        sendDelayMs: this.props.sendDelayMs,
        sparseTrackType: this.props.sparseTrackType?.value,
        streamManifestBehavior: this.props.streamManifestBehavior?.value,
        timestampOffset: this.props.timestampOffset,
        timestampOffsetMode: this.props.timestampOffsetMode?.value,
      },
    };
  }
  public _bindDestination(_channelClass: string): CfnChannel.OutputDestinationProperty[] {
    validateDestinationCount(this._name, _channelClass, this.props.destinations.length, 0, false);
    const settings = this.props.destinations.map(d => d._bind());
    return [{ id: toDestinationId(this.props.name), settings }];
  }
  public override _grantPermissions(role: IRole): void {
    this.props.destinations.forEach(d => d._grantPermissions(role));
  }
}

// =============================================================================
// OutputGroup (orchestrates outputs within a group)
// =============================================================================

/**
 * An output group within a MediaLive channel.
 *
 * @internal
 */
export class OutputGroup {
  private readonly outputs: Output[] = [];

  constructor(private readonly config: OutputGroupConfiguration) {
    this.outputs.push(...config._createInitialOutputs());
  }

  /** @internal */
  public _bind(): CfnChannel.OutputGroupProperty {
    return {
      name: this.config._name,
      outputGroupSettings: this.config._bind(),
      outputs: this.outputs.map(o => o._bind()),
    };
  }

  /** @internal */
  public _bindDestination(channelClass: string): CfnChannel.OutputDestinationProperty[] {
    return this.config._bindDestination(channelClass);
  }

  /** @internal */
  public _collectEncodes(): EncodeConfiguration[] {
    return this.outputs.flatMap(o => o._getEncodes());
  }

  /** @internal - Propagate epoch-locking state to the underlying config. */
  public _setEpochLocking(active: boolean): void {
    this.config._setEpochLocking(active);
  }

  /** @internal - Whether the underlying config explicitly requests an incompatible SYSTEM_CLOCK. */
  public _hasExplicitSystemClock(): boolean {
    return this.config._hasExplicitSystemClock();
  }

  /** @internal */
  public _grantPermissions(role: IRoleRef): void {
    this.config._grantPermissions(role);
    // Grant read access to any external files referenced by the outputs (burn-in caption fonts,
    // audio-only cover-art images) sourced from S3 buckets.
    this.outputs.forEach(o => o._grantPermissions(role));
  }
}
