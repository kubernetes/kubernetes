import type { Duration } from 'aws-cdk-lib';
import { UnscopedValidationError, Token } from 'aws-cdk-lib';
import type { IRole, IRoleRef } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { AudioCodecType } from './audio-codec-settings';
import type { RtmpDestination, SrtDestination } from './destinations';
import type { EncodeConfiguration } from './encode-configuration';
import type { FecMode, H265PackagingType, RtmpCertificateMode, SrtEncryptionType } from './enums';
import { MediaPackageV2HlsSetting } from './enums';
import type { M2tsSettings } from './m2ts-settings';
import { HlsSettings } from './m3u8-settings';
import type { UdpOutputGroupProps } from './output-group';
import { VideoCodecType } from './video-codec-settings';

/**
 * Convert an output name to a valid MediaLive channel `DestinationId`.
 *
 * Derives a valid CFN destination ID from a name (output group name or per-output name).
 * The service requires destination IDs to contain only alphanumeric characters and hyphens,
 * with no leading/trailing hyphen. Since names may contain underscores or other characters,
 * we sanitise here.
 *
 * @internal
 */
export function toDestinationId(outputName: string): string {
  if (Token.isUnresolved(outputName)) return outputName;
  return outputName
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

/** @internal */
interface ValidateCodecsOptions {
  readonly outputName: string;
  readonly groupName?: string;
  readonly allowedVideo: VideoCodecType[];
  readonly allowedAudio: AudioCodecType[];
  readonly encodes: EncodeConfiguration[];
}

/** @internal */
export function validateCodecs(opts: ValidateCodecsOptions): void {
  opts.encodes.forEach(encode => {
    const vType = encode._videoCodecType();
    if (vType && !opts.allowedVideo.includes(vType)) {
      const label = opts.groupName ? `${opts.outputName}' in group '${opts.groupName}` : opts.outputName;
      throw new UnscopedValidationError(
        lit`UnsupportedVideoCodec`,
        `Output '${label}' does not support video codec '${vType}'.`
        + ` Supported: ${opts.allowedVideo.join(', ')}.`,
      );
    }
    const aType = encode._audioCodecType();
    if (aType && !opts.allowedAudio.includes(aType)) {
      const label = opts.groupName ? `${opts.outputName}' in group '${opts.groupName}` : opts.outputName;
      throw new UnscopedValidationError(
        lit`UnsupportedAudioCodec`,
        `Output '${label}' does not support audio codec '${aType}'.`
        + ` Supported: ${opts.allowedAudio.join(', ')}.`,
      );
    }
  });
}

/**
 * Validates that an MPEG-TS or RTMP output contains at most one video encode.
 * These container formats support a single video PID. Multiple audio encodes are allowed.
 * @internal
 */
export function validateSingleVideoEncode(encodes: EncodeConfiguration[], outputName: string, groupName?: string): void {
  const videoEncodes = encodes.filter(e => e._videoCodecType() !== undefined);
  if (videoEncodes.length > 1) {
    const label = groupName ? `${outputName}' in group '${groupName}` : outputName;
    throw new UnscopedValidationError(
      lit`MultipleVideoEncodes`,
      `Output '${label}' contains ${videoEncodes.length} video encodes, but only one is allowed per output.`,
    );
  }
}

/**
 * Validate FEC output settings: SMPTE 2022-1 ranges, and that FEC is only used with rtp://
 * destinations (the only protocol MediaLive supports for FEC). Tokenized values are skipped.
 * @internal
 */
export function validateFec(fec: FecOutputSettings | undefined, outputName: string, destinationUrls: string[]): void {
  if (!fec) {
    return;
  }
  if (fec.columnDepth !== undefined && !Token.isUnresolved(fec.columnDepth)
    && (fec.columnDepth < 4 || fec.columnDepth > 20)) {
    throw new UnscopedValidationError(
      lit`FecColumnDepthRange`,
      `UDP output '${outputName}' fec.columnDepth must be between 4 and 20, got ${JSON.stringify(fec.columnDepth)}`,
    );
  }
  if (fec.rowLength !== undefined && !Token.isUnresolved(fec.rowLength)
    && (fec.rowLength < 1 || fec.rowLength > 20)) {
    throw new UnscopedValidationError(
      lit`FecRowLengthRange`,
      `UDP output '${outputName}' fec.rowLength must be between 1 and 20, got ${JSON.stringify(fec.rowLength)}`,
    );
  }
  // MediaLive only supports FEC on rtp:// destinations. The destinations are shared by the whole
  // UDP group, so any non-rtp destination makes this FEC output invalid.
  const nonRtp = destinationUrls.find(url => !Token.isUnresolved(url) && !url.startsWith('rtp://'));
  if (nonRtp !== undefined) {
    throw new UnscopedValidationError(
      lit`FecRequiresRtp`,
      `UDP output '${outputName}' has FEC enabled, which requires every destination in the group to use the rtp:// protocol, got ${JSON.stringify(nonRtp)}`,
    );
  }
}

/**
 * Base output definition — shared by all output group types.
 */
export interface OutputDefinition {
  /**
   * The encode configurations to wire to this output.
   */
  readonly encodes: EncodeConfiguration[];
  /**
   * The name of this output. Must be unique across all outputs in the channel.
   */
  readonly outputName: string;
}

// =============================================================================
// Output (abstract base + subclasses per output group type)
// =============================================================================

/**
 * Represents an output within an output group.
 * Each output group type has its own Output subclass that knows how to render its output settings.
 */
export abstract class Output {
  private readonly encodes: EncodeConfiguration[] = [];

  /** @internal */
  constructor(protected readonly outputName: string) {
    if (!Token.isUnresolved(outputName)) {
      if (outputName.length < 1 || outputName.length > 256) {
        throw new UnscopedValidationError(lit`OutputNameLength`, 'Output name must be between 1 and 256 characters.');
      }
      if (!/^[a-zA-Z0-9_-]+$/.test(outputName)) {
        throw new UnscopedValidationError(lit`OutputNameFormat`, 'Output name must contain only alphanumeric characters, hyphens, and underscores.');
      }
    }
  }

  /** @internal */
  public _getEncodes(): EncodeConfiguration[] {
    return this.encodes;
  }

  /**
   * Grant the channel role read access to external files referenced by this output. The base
   * grants files referenced by the output's encodes (e.g. burn-in caption fonts); subclasses
   * override to add output-type-specific files (e.g. audio-only cover art).
   * @internal
   */
  public _grantPermissions(role: IRoleRef): void {
    this.encodes.forEach(e => e._grantRead(role));
  }

  /**
   * Wire an encode configuration to this output.
   * @internal
   */
  public _addEncode(encode: EncodeConfiguration): void {
    this.encodes.push(encode);
  }

  /**
   * Render the output settings for this output type.
   * @internal
   */
  protected abstract _bindOutputSettings(): CfnChannel.OutputSettingsProperty;

  /**
   * Validate encodes for this output type. Called at synth time.
   * @internal
   */
  protected _validate(): void {
    // Default: no validation. Subclasses override.
  }

  /** @internal */
  public _bind(): CfnChannel.OutputProperty {
    this._validate();

    const videoNames = this.encodes.flatMap(e => e._bindVideo()?.name ?? []);
    const audioNames = this.encodes.flatMap(e => e._bindAudio()?.name ?? []);
    const captionNames = this.encodes.flatMap(e => e._bindCaption()?.name ?? []);

    return {
      outputSettings: this._bindOutputSettings(),
      outputName: this.outputName,
      videoDescriptionName: videoNames[0],
      audioDescriptionNames: audioNames.length > 0 ? audioNames : undefined,
      captionDescriptionNames: captionNames.length > 0 ? captionNames : undefined,
    };
  }
}

// =============================================================================
// Per-type output definitions (extend base with type-specific settings)
// =============================================================================

/**
 * Output definition for a MediaPackage V2 output group.
 *
 * MediaPackage V2 uses CMAF ingest which requires one media track (video or audio) per output.
 * In-band captions (burn-in, embedded) can ride alongside the primary encode because they do not
 * produce a separate track.
 */
export interface MediaPackageV2OutputDefinition {
  /**
   * The primary encode for this output — one video or one audio track.
   */
  readonly encode: EncodeConfiguration;
  /**
   * Caption encodes that ride alongside the primary encode. Only in-band caption types are
   * allowed (burn-in, embedded) — out-of-band captions must go in their own output.
   *
   * @default - no captions on this output
   */
  readonly captions?: EncodeConfiguration[];
  /**
   * The name of this output. Must be unique across all outputs in the channel.
   */
  readonly outputName: string;
  /**
   * The audio group ID for audio outputs.
   * @default - service default
   */
  readonly audioGroupId?: string;
  /**
   * For audio outputs, whether MediaPackage sets this rendition as the auto-select rendition in the
   * HLS manifest.
   * @default MediaPackageV2HlsSetting.OMIT
   */
  readonly hlsAutoSelect?: MediaPackageV2HlsSetting;
  /**
   * For audio outputs, whether MediaPackage sets this rendition as the default rendition in the HLS
   * manifest.
   * @default MediaPackageV2HlsSetting.OMIT
   */
  readonly hlsDefault?: MediaPackageV2HlsSetting;
  /**
   * The audio rendition sets for video outputs.
   * @default - service default
   */
  readonly audioRenditionSets?: string;
}

/**
 * Output definition for an HLS output group.
 */
export interface HlsOutputDefinition extends OutputDefinition {
  /**
   * A string concatenated to the end of the destination file name.
   * @default - service default
   */
  readonly nameModifier?: string;
  /**
   * A string concatenated to the end of segment file names.
   * @default - no segment modifier
   */
  readonly segmentModifier?: string;
  /**
   * For H.265 video, whether to package as HEV1 or HVC1.
   * @default - service default
   */
  readonly h265PackagingType?: H265PackagingType;
  /**
   * The per-output HLS settings (standard, audio-only, fMP4, or frame-capture). Use the
   * `HlsSettings` factory methods. Standard outputs additionally configure the M3U8 container
   * via `HlsSettings.standard({ m3u8Settings })`.
   * @default - HlsSettings.standard() with service-default M3U8 settings
   */
  readonly hlsSettings?: HlsSettings;
}

/**
 * Forward Error Correction (FEC) settings for a UDP output (SMPTE 2022-1).
 */
export interface FecOutputSettings {
  /**
   * Whether to enable column-only or column-and-row FEC.
   * @default - service default
   */
  readonly mode?: FecMode;
  /**
   * Parameter D from SMPTE 2022-1 — the height of the FEC protection matrix (number of
   * transport stream packets per column error-correction packet). Must be 4..20.
   * @default - service default
   */
  readonly columnDepth?: number;
  /**
   * Parameter L from SMPTE 2022-1 — the width of the FEC protection matrix. Must be 1..20
   * for column-only FEC, or 4..20 for column-and-row FEC.
   * @default - service default
   */
  readonly rowLength?: number;
}

/**
 * Output definition for a UDP output group.
 */
export interface UdpOutputDefinition extends OutputDefinition {
  /**
   * The output buffering (overrides group-level setting). Applied at millisecond granularity.
   * @default - uses group-level buffer
   */
  readonly buffer?: Duration;
  /**
   * MPEG-TS (M2TS) container settings for this output.
   * @default - service defaults
   */
  readonly m2tsSettings?: M2tsSettings;
  /**
   * Forward Error Correction (FEC) settings for this output.
   * @default - no FEC
   */
  readonly fec?: FecOutputSettings;
}

/**
 * The container (transport stream) for an Archive output. Use the static factory methods to
 * select between an MPEG-TS (M2TS) container or a raw container.
 */
export abstract class ArchiveContainer {
  /** An MPEG-TS (M2TS) container, optionally configured via `M2tsSettings`. */
  public static m2ts(settings?: M2tsSettings): ArchiveContainer {
    return new M2tsArchiveContainer(settings);
  }
  /** A raw container (no transport-stream wrapping). */
  public static raw(): ArchiveContainer {
    return new RawArchiveContainer();
  }

  /** @internal */
  public abstract _bind(): CfnChannel.ArchiveContainerSettingsProperty;

  /**
   * Whether this is a raw container. Raw containers hold a single uncompressed (WAV) audio
   * stream — no video, no compressed audio.
   * @internal
   */
  public abstract _isRaw(): boolean;
}

/** @internal */
class M2tsArchiveContainer extends ArchiveContainer {
  constructor(private readonly settings?: M2tsSettings) { super(); }
  public _bind(): CfnChannel.ArchiveContainerSettingsProperty {
    return { m2TsSettings: this.settings?._bind() ?? {} };
  }
  public _isRaw(): boolean {
    return false;
  }
}

/** @internal */
class RawArchiveContainer extends ArchiveContainer {
  public _bind(): CfnChannel.ArchiveContainerSettingsProperty {
    return { rawSettings: {} };
  }
  public _isRaw(): boolean {
    return true;
  }
}

/**
 * Output definition for an Archive output group.
 */
export interface ArchiveOutputDefinition extends OutputDefinition {
  /**
   * A string concatenated to the end of the destination file name. Required if the output group
   * contains more than one output of the same container type.
   * @default - no name modifier
   */
  readonly nameModifier?: string;
  /**
   * The output file extension.
   * @default - auto-selected from container type
   */
  readonly extension?: string;
  /**
   * The container (transport stream) for this output — an MPEG-TS (M2TS) or raw container.
   * Use the `ArchiveContainer` factory methods.
   * @default - ArchiveContainer.m2ts() with service-default M2TS settings
   */
  readonly container?: ArchiveContainer;
}

/**
 * Output definition for an RTMP output group.
 * `outputName` is normalised (underscores → hyphens) for use as the destination reference ID.
 */
export interface RtmpOutputDefinition extends OutputDefinition {
  /**
   * The RTMP destination(s) for this output — one per channel pipeline.
   *
   * MediaLive publishes each RTMP output to one destination per pipeline (the console
   * calls these "Destination A" and "Destination B"). Provide a single destination for a
   * `SINGLE_PIPELINE` channel, or two (A then B) for a `STANDARD` channel.
   */
  readonly destinations: RtmpDestination[];
  /**
   * The TLS certificate verification mode.
   * @default - service default
   */
  readonly certificateMode?: RtmpCertificateMode;
  /**
   * The interval between connection retry attempts.
   * @default - service default
   */
  readonly connectionRetryInterval?: Duration;
  /**
   * The number of retry attempts.
   * @default - service default
   */
  readonly numRetries?: number;
}

/**
 * Output definition for an SRT output group.
 * `outputName` is normalised (underscores → hyphens) for use as the destination reference ID.
 */
export interface SrtOutputDefinition extends OutputDefinition {
  /**
   * The SRT destination(s) for this output — one per channel pipeline.
   *
   * MediaLive publishes each SRT output to one destination per pipeline (the console
   * calls these "Destination A" and "Destination B"). Provide a single destination for a
   * `SINGLE_PIPELINE` channel, or two (A then B) for a `STANDARD` channel. Each pipeline
   * can target a different listener and carry its own encryption passphrase.
   */
  readonly destinations: SrtDestination[];
  /**
   * The output buffering. Applied at millisecond granularity.
   * @default - service default
   */
  readonly buffer?: Duration;
  /**
   * The encryption type for the SRT output.
   * @default - no encryption
   */
  readonly encryptionType?: SrtEncryptionType;
  /**
   * The SRT latency.
   * @default - service default
   */
  readonly latency?: Duration;
  /**
   * MPEG-TS (M2TS) container settings for this output.
   * @default - service defaults
   */
  readonly m2tsSettings?: M2tsSettings;
}

/**
 * Output definition for a CMAF Ingest output group.
 *
 * CMAF Ingest requires one media track (video or audio) per output. In-band captions (burn-in,
 * embedded) can ride alongside the primary encode.
 */
export interface CmafIngestOutputDefinition {
  /**
   * The primary encode for this output — one video or one audio track.
   */
  readonly encode: EncodeConfiguration;
  /**
   * Caption encodes that ride alongside the primary encode. Only in-band caption types are
   * allowed (burn-in, embedded) — out-of-band captions must go in their own output.
   *
   * @default - no captions on this output
   */
  readonly captions?: EncodeConfiguration[];
  /**
   * The name of this output. Must be unique across all outputs in the channel.
   */
  readonly outputName: string;
  /**
   * A string concatenated to the end of the destination file name.
   * @default - no name modifier
   */
  readonly nameModifier?: string;
}

/**
 * Output definition for a Frame Capture output group.
 */
export interface FrameCaptureOutputDefinition extends OutputDefinition {
  /**
   * A string concatenated to the end of the destination file name.
   * Required if the output group contains more than one output.
   * @default - no name modifier
   */
  readonly nameModifier?: string;
}

/**
 * Output definition for an MS Smooth output group.
 */
export interface MsSmoothOutputDefinition extends OutputDefinition {
  /**
   * A string concatenated to the end of the destination file name. Required if the output group
   * contains more than one output.
   * @default - no name modifier
   */
  readonly nameModifier?: string;
  /**
   * For H.265 video, whether to package as HEV1 or HVC1.
   * @default - service default
   */
  readonly h265PackagingType?: H265PackagingType;
}

// =============================================================================
// Output subclasses (one per output group type)
// =============================================================================

/** @internal */
export class MediaPackageV2Output extends Output {
  constructor(private readonly def: MediaPackageV2OutputDefinition, private readonly groupName: string) {
    super(def.outputName);
    this._addEncode(def.encode);
    for (const caption of def.captions ?? []) {
      this._addEncode(caption);
    }
  }

  protected _validate(): void {
    const encode = this.def.encode;
    if (encode._bindVideo() && !encode._hasExplicitFramerate()) {
      throw new UnscopedValidationError(
        lit`MediaPackageFramerateRequired`,
        `MediaPackage V2 output group '${this.groupName}' requires explicit framerate on video encode '${encode.name}'.`,
      );
    }
    // Validate the primary encode is not a caption
    if (encode._bindCaption()) {
      throw new UnscopedValidationError(
        lit`MediaPackagePrimaryEncodeNotCaption`,
        `MediaPackage V2 output '${this.def.outputName}': 'encode' must be a video or audio encode. Captions must go in the 'captions' prop alongside a video encode.`,
      );
    }
    // Validate captions are actually caption encodes and are in-band only
    for (const caption of this.def.captions ?? []) {
      if (!caption._bindCaption()) {
        throw new UnscopedValidationError(
          lit`MediaPackageCaptionsMustBeCaptions`,
          `MediaPackage V2 output '${this.def.outputName}': 'captions' must contain caption encodes only, got '${caption.name}'.`,
        );
      }
      if (!caption._isInBandCaption()) {
        throw new UnscopedValidationError(
          lit`MediaPackageCaptionsMustBeInBand`,
          `MediaPackage V2 output '${this.def.outputName}': caption '${caption.name}' uses an out-of-band destination (e.g. WebVTT, TTML). Only in-band types (burn-in, embedded) can ride alongside the primary encode. Put out-of-band captions in their own output.`,
        );
      }
    }
    // Only one embedded caption per output
    const embeddedCount = (this.def.captions ?? []).filter(c => c._isEmbeddedCaption()).length
      + (encode._bindCaption() && encode._isEmbeddedCaption() ? 1 : 0);
    if (embeddedCount > 1) {
      throw new UnscopedValidationError(
        lit`MediaPackageOneEmbeddedCaption`,
        `MediaPackage V2 output '${this.def.outputName}': only one embedded caption is allowed per output.`,
      );
    }
    validateCodecs({
      outputName: this.groupName,
      // MediaPackage V2 uses CMAF Ingest under the hood, which supports AV1 in addition to
      // H.264/H.265 — see https://docs.aws.amazon.com/mediapackage/latest/userguide/cmaf-ingest.html.
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265, VideoCodecType.AV1, VideoCodecType.FRAME_CAPTURE],
      allowedAudio: [AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS, AudioCodecType.PASSTHROUGH],
      encodes: [this.def.encode],
    });
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      mediaPackageOutputSettings: {
        mediaPackageV2DestinationSettings: {
          audioGroupId: this.def.audioGroupId,
          audioRenditionSets: this.def.audioRenditionSets,
          hlsAutoSelect: (this.def.hlsAutoSelect ?? MediaPackageV2HlsSetting.OMIT).value,
          hlsDefault: (this.def.hlsDefault ?? MediaPackageV2HlsSetting.OMIT).value,
        },
      },
    };
  }
}

/** @internal */
export class HlsOutput extends Output {
  constructor(private readonly def: HlsOutputDefinition) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    validateCodecs({
      outputName: this.outputName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265, VideoCodecType.FRAME_CAPTURE],
      allowedAudio: [AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS, AudioCodecType.PASSTHROUGH],
      encodes: this._getEncodes(),
    });
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      hlsOutputSettings: {
        nameModifier: this.def.nameModifier,
        segmentModifier: this.def.segmentModifier,
        h265PackagingType: this.def.h265PackagingType?.value,
        hlsSettings: (this.def.hlsSettings ?? HlsSettings.standard())._bind(),
      },
    };
  }

  public override _grantPermissions(role: IRole): void {
    super._grantPermissions(role);
    // Audio-only HLS settings may reference a cover-art image in S3.
    this.def.hlsSettings?._grantRead(role);
  }
}

/** @internal */
export class UdpOutput extends Output {
  constructor(private readonly def: UdpOutputDefinition, private readonly groupProps: UdpOutputGroupProps) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    validateCodecs({
      outputName: this.outputName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265],
      allowedAudio: [
        AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS,
        AudioCodecType.MP2, AudioCodecType.PASSTHROUGH,
      ],
      encodes: this._getEncodes(),
    });
    validateSingleVideoEncode(this._getEncodes(), this.outputName);
    validateFec(this.def.fec, this.outputName, this.groupProps.destinations.map(d => d._bind().url));
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      udpOutputSettings: {
        bufferMsec: this.def.buffer?.toMilliseconds() ?? this.groupProps.buffer?.toMilliseconds(),
        destination: {
          destinationRefId: this.groupProps.name,
        },
        containerSettings: {
          m2TsSettings: this.def.m2tsSettings?._bind() ?? {},
        },
        fecOutputSettings: this.def.fec ? {
          includeFec: this.def.fec.mode?.value,
          columnDepth: this.def.fec.columnDepth,
          rowLength: this.def.fec.rowLength,
        } : undefined,
      },
    };
  }
}

/** @internal */
export class ArchiveOutput extends Output {
  constructor(private readonly def: ArchiveOutputDefinition) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    const isRaw = (this.def.container ?? ArchiveContainer.m2ts())._isRaw();
    if (isRaw) {
      // Raw containers require an explicit file extension.
      validateCodecs({
        outputName: this.outputName,
        allowedVideo: [VideoCodecType.H264, VideoCodecType.H265],
        allowedAudio: [
          AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS,
          AudioCodecType.MP2, AudioCodecType.WAV, AudioCodecType.PASSTHROUGH,
        ],
        encodes: this._getEncodes(),
      });
      if (this.def.extension === undefined) {
        throw new UnscopedValidationError(
          lit`RawArchiveExtensionRequired`,
          `Archive output '${this.outputName}' uses a raw container, which requires an explicit 'extension'`,
        );
      }
      return;
    }
    validateCodecs({
      outputName: this.outputName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265],
      allowedAudio: [
        AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS,
        AudioCodecType.MP2, AudioCodecType.PASSTHROUGH,
      ],
      encodes: this._getEncodes(),
    });
    validateSingleVideoEncode(this._getEncodes(), this.outputName);
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      archiveOutputSettings: {
        nameModifier: this.def.nameModifier,
        extension: this.def.extension,
        containerSettings: (this.def.container ?? ArchiveContainer.m2ts())._bind(),
      },
    };
  }
}

/** @internal */
export class RtmpOutput extends Output {
  constructor(private readonly def: RtmpOutputDefinition, private readonly groupName: string) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    validateCodecs({
      outputName: this.outputName,
      groupName: this.groupName,
      allowedVideo: [VideoCodecType.H264],
      allowedAudio: [AudioCodecType.AAC],
      encodes: this._getEncodes(),
    });
    validateSingleVideoEncode(this._getEncodes(), this.outputName, this.groupName);
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      rtmpOutputSettings: {
        certificateMode: this.def.certificateMode?.value,
        connectionRetryInterval: this.def.connectionRetryInterval?.toSeconds(),
        numRetries: this.def.numRetries,
        destination: {
          destinationRefId: toDestinationId(this.outputName),
        },
      },
    };
  }
}

/** @internal */
export class SrtOutput extends Output {
  constructor(private readonly def: SrtOutputDefinition, private readonly groupName: string) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    validateCodecs({
      outputName: this.outputName,
      groupName: this.groupName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265],
      allowedAudio: [
        AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS,
        AudioCodecType.MP2, AudioCodecType.PASSTHROUGH,
      ],
      encodes: this._getEncodes(),
    });
    validateSingleVideoEncode(this._getEncodes(), this.outputName, this.groupName);
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      srtOutputSettings: {
        bufferMsec: this.def.buffer?.toMilliseconds(),
        encryptionType: this.def.encryptionType?.value,
        latency: this.def.latency?.toMilliseconds(),
        // Each SRT output has its own channel destination, keyed by the output name.
        destination: {
          destinationRefId: toDestinationId(this.outputName),
        },
        containerSettings: {
          m2TsSettings: this.def.m2tsSettings?._bind() ?? {},
        },
      },
    };
  }
}

/** @internal */
export class CmafIngestOutput extends Output {
  constructor(private readonly def: CmafIngestOutputDefinition) {
    super(def.outputName);
    this._addEncode(def.encode);
    for (const caption of def.captions ?? []) {
      this._addEncode(caption);
    }
  }

  protected _validate(): void {
    // Validate captions are actually caption encodes and are in-band only
    for (const caption of this.def.captions ?? []) {
      if (!caption._bindCaption()) {
        throw new UnscopedValidationError(
          lit`CmafIngestCaptionsMustBeCaptions`,
          `CMAF Ingest output '${this.def.outputName}': 'captions' must contain caption encodes only, got '${caption.name}'.`,
        );
      }
      if (!caption._isInBandCaption()) {
        throw new UnscopedValidationError(
          lit`CmafIngestCaptionsMustBeInBand`,
          `CMAF Ingest output '${this.def.outputName}': caption '${caption.name}' uses an out-of-band destination. Only in-band types (burn-in, embedded) can ride alongside the primary encode. Put out-of-band captions in their own output.`,
        );
      }
    }
    // Only one embedded caption per output
    const embeddedCount = (this.def.captions ?? []).filter(c => c._isEmbeddedCaption()).length
      + (this.def.encode._bindCaption() && this.def.encode._isEmbeddedCaption() ? 1 : 0);
    if (embeddedCount > 1) {
      throw new UnscopedValidationError(
        lit`CmafIngestOneEmbeddedCaption`,
        `CMAF Ingest output '${this.def.outputName}': only one embedded caption is allowed per output.`,
      );
    }
    validateCodecs({
      outputName: this.outputName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265, VideoCodecType.AV1],
      allowedAudio: [AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS, AudioCodecType.PASSTHROUGH],
      encodes: [this.def.encode],
    });
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      cmafIngestOutputSettings: {
        nameModifier: this.def.nameModifier,
      },
    };
  }
}

/** @internal */
export class FrameCaptureOutput extends Output {
  constructor(private readonly def: FrameCaptureOutputDefinition) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    const allowedVideo: VideoCodecType[] = [VideoCodecType.FRAME_CAPTURE];
    this._getEncodes().forEach(encode => {
      const vType = encode._videoCodecType();
      if (vType && !allowedVideo.includes(vType)) {
        throw new UnscopedValidationError(
          lit`UnsupportedVideoCodec`,
          `Frame Capture output '${this.outputName}' does not support video codec '${vType}'.`
          + ` Supported: ${allowedVideo.join(', ')}.`,
        );
      }
      const aType = encode._audioCodecType();
      if (aType) {
        throw new UnscopedValidationError(lit`UnsupportedAudioCodec`, 'Frame Capture outputs do not support audio.');
      }
    });
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      frameCaptureOutputSettings: {
        nameModifier: this.def.nameModifier,
      },
    };
  }
}

/** @internal */
export class MsSmoothOutput extends Output {
  constructor(private readonly def: MsSmoothOutputDefinition) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    validateCodecs({
      outputName: this.outputName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265],
      allowedAudio: [AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3],
      encodes: this._getEncodes(),
    });
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      msSmoothOutputSettings: {
        nameModifier: this.def.nameModifier,
        h265PackagingType: this.def.h265PackagingType?.value,
      },
    };
  }
}

/**
 * Output definition for a MediaConnect Router output group.
 */
export interface MediaConnectRouterOutputDefinition extends OutputDefinition {
  /**
   * MPEG-TS (M2TS) container settings for this output.
   * @default - service defaults
   */
  readonly m2tsSettings?: M2tsSettings;
}

/** @internal */
export class MediaConnectRouterOutput extends Output {
  constructor(private readonly def: MediaConnectRouterOutputDefinition, private readonly groupName: string) {
    super(def.outputName);
    def.encodes.forEach(e => this._addEncode(e));
  }

  protected _validate(): void {
    validateCodecs({
      outputName: this.outputName,
      groupName: this.groupName,
      allowedVideo: [VideoCodecType.H264, VideoCodecType.H265],
      allowedAudio: [
        AudioCodecType.AAC, AudioCodecType.AC3, AudioCodecType.EAC3, AudioCodecType.EAC3_ATMOS,
        AudioCodecType.MP2, AudioCodecType.PASSTHROUGH,
      ],
      encodes: this._getEncodes(),
    });
    validateSingleVideoEncode(this._getEncodes(), this.outputName, this.groupName);
  }

  protected _bindOutputSettings(): CfnChannel.OutputSettingsProperty {
    return {
      mediaConnectRouterOutputSettings: {
        destination: {
          destinationRefId: toDestinationId(this.groupName),
        },
        containerSettings: {
          m2TsSettings: this.def.m2tsSettings?._bind() ?? {},
        },
      },
    };
  }
}
