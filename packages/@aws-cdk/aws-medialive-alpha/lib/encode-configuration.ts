import { Token, UnscopedValidationError } from 'aws-cdk-lib';
import type { IRole, IRoleRef } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type { AudioCodecSettings, AudioCodecType } from './audio-codec-settings';
import type { CaptionDestination } from './caption-settings';
import type { VideoCodecSettings, VideoCodecType } from './video-codec-settings';

/**
 * Base interface for an encode configuration (video, audio, or caption).
 *
 * The same EncodeConfiguration instance can be shared across multiple output groups within a channel.
 * The channel automatically deduplicates encode descriptions by name at synth time.
 */
export abstract class EncodeConfiguration {
  /**
   * Create a video encode configuration.
   */
  public static video(props: VideoEncodeProps): EncodeConfiguration {
    return new VideoEncodeConfiguration(props);
  }

  /**
   * Create an audio encode configuration.
   */
  public static audio(props: AudioEncodeProps): EncodeConfiguration {
    return new AudioEncodeConfiguration(props);
  }

  /**
   * Create a caption encode configuration.
   */
  public static caption(props: CaptionEncodeProps): EncodeConfiguration {
    return new CaptionEncodeConfiguration(props);
  }

  /** The unique name for this encode, used to reference it from outputs. */
  public abstract readonly name: string;

  /**
   * Whether this encode has explicit framerate and PAR configured.
   * Required for MediaPackage V2 video outputs.
   * @internal
   */
  public abstract _hasExplicitFramerate(): boolean;

  /** @internal */
  public abstract _bindVideo(): CfnChannel.VideoDescriptionProperty | undefined;
  /** @internal */
  public abstract _bindAudio(): CfnChannel.AudioDescriptionProperty | undefined;
  /** @internal */
  public abstract _bindCaption(): CfnChannel.CaptionDescriptionProperty | undefined;
  /** @internal */
  public abstract _videoCodecType(): VideoCodecType | undefined;
  /** @internal */
  public abstract _audioCodecType(): AudioCodecType | undefined;

  /**
   * Grant the channel role read access to any external files this encode references (e.g. a
   * burn-in caption font in S3). Default is a no-op; caption encodes override it.
   * @internal
   */
  public _grantRead(_role: IRoleRef): void {}

  /**
   * Whether this is an in-band caption encode (burn-in, embedded) that does not produce a
   * separate track. Returns false for non-caption encodes and out-of-band caption types.
   * @internal
   */
  public _isInBandCaption(): boolean {
    return false;
  }

  /**
   * Whether this is an embedded-family caption encode. Only one per output is allowed.
   * @internal
   */
  public _isEmbeddedCaption(): boolean {
    return false;
  }
}

/**
 * How to respond to AFD values in the input stream.
 */
export class RespondToAfd {
  /** Clip input video based on AFD values */
  public static readonly RESPOND = new RespondToAfd('RESPOND');
  /** Pass AFD values through without clipping */
  public static readonly PASSTHROUGH = new RespondToAfd('PASSTHROUGH');
  /** Ignore AFD values */
  public static readonly NONE = new RespondToAfd('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): RespondToAfd {
    return new RespondToAfd(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Video scaling behavior.
 */
export class ScalingBehavior {
  /** May insert black boxes to match output resolution */
  public static readonly DEFAULT = new ScalingBehavior('DEFAULT');
  /** Stretch video to fill the output resolution */
  public static readonly STRETCH_TO_OUTPUT = new ScalingBehavior('STRETCH_TO_OUTPUT');
  /** Intelligently crop the video to focus on key subjects (9:16 vertical). Requires an Elemental Inference feed on the channel via `inferenceFeed`. Do NOT include `FeedOutput.cropping()` on the feed — MediaLive auto-inserts it. */
  public static readonly SMART_CROP = new ScalingBehavior('SMART_CROP');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ScalingBehavior {
    return new ScalingBehavior(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties for a video encode configuration.
 */
export interface VideoEncodeProps {
  /**
   * A unique name for this video encode.
   */
  readonly name: string;
  /**
   * The width of the output video in pixels. Must be an even number.
   */
  readonly width: number;
  /**
   * The height of the output video in pixels. Must be an even number.
   */
  readonly height: number;
  /**
   * The codec for the video encode.
   *
   * Choose the codec explicitly (e.g. `VideoCodecSettings.h264(...)`)
   */
  readonly codec: VideoCodecSettings;
  /**
   * How to respond to AFD values in the input stream.
   * @default RespondToAfd.NONE
   */
  readonly respondToAfd?: RespondToAfd;
  /**
   * The video scaling behavior.
   * @default ScalingBehavior.DEFAULT
   */
  readonly scalingBehavior?: ScalingBehavior;
  /**
   * The anti-alias filter strength (0-100). 0 is softest, 100 is sharpest.
   * @default 50
   */
  readonly sharpness?: number;
}

/**
 * Audio normalization algorithm.
 */
export class AudioNormalizationAlgorithm {
  /** CALM Act specification (ITU-R BS.1770-1) */
  public static readonly ITU_1770_1 = new AudioNormalizationAlgorithm('ITU_1770_1');
  /** EBU R-128 specification (ITU-R BS.1770-2) */
  public static readonly ITU_1770_2 = new AudioNormalizationAlgorithm('ITU_1770_2');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioNormalizationAlgorithm {
    return new AudioNormalizationAlgorithm(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Audio normalization algorithm control.
 */
export class AudioNormalizationAlgorithmControl {
  /** Correct the audio using the chosen algorithm */
  public static readonly CORRECT_AUDIO = new AudioNormalizationAlgorithmControl('CORRECT_AUDIO');
  /** Measure audio but do not adjust */
  public static readonly MEASURE_ONLY = new AudioNormalizationAlgorithmControl('MEASURE_ONLY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioNormalizationAlgorithmControl {
    return new AudioNormalizationAlgorithmControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Peak calculation method for audio normalization.
 */
export class AudioNormalizationPeakCalculation {
  /** Calculate and log the TruePeak for each audio track. */
  public static readonly TRUE_PEAK = new AudioNormalizationPeakCalculation('TRUE_PEAK');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioNormalizationPeakCalculation {
    return new AudioNormalizationPeakCalculation(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Audio normalization settings.
 */
export interface AudioNormalizationSettings {
  /**
   * The normalization algorithm.
   * @default - service default
   */
  readonly algorithm?: AudioNormalizationAlgorithm;
  /**
   * Whether to correct or only measure.
   * @default - service default
   */
  readonly algorithmControl?: AudioNormalizationAlgorithmControl;
  /**
   * The target loudness in LKFS. CALM Act recommends -24, EBU R-128 recommends -23.
   * @default - service default
   */
  readonly targetLkfs?: number;
  /**
   * Whether to use a peak limiter and how to calculate peak levels.
   * @default - service default
   */
  readonly peakCalculation?: AudioNormalizationPeakCalculation;
  /**
   * The peak limiter threshold in dBFS. Only used when peak limiting is enabled.
   * @default - service default
   */
  readonly peakLimiterThreshold?: number;
}

/**
 * An input channel level for audio remixing.
 */
export interface InputChannelLevel {
  /**
   * The index of the input channel to use as a source.
   */
  readonly inputChannel: number;
  /**
   * The remixing gain in dB (-60 to 6).
   * @default 0
   */
  readonly gain?: number;
}

/**
 * A mapping from input channels to an output channel.
 */
export interface AudioChannelMapping {
  /**
   * The index of the output channel being produced.
   */
  readonly outputChannel: number;
  /**
   * The input channels and their gain levels to mix into this output channel.
   */
  readonly inputChannelLevels: InputChannelLevel[];
}

/**
 * Audio remix settings for channel remapping.
 */
export interface RemixSettings {
  /**
   * The channel mappings from input to output.
   */
  readonly channelMappings: AudioChannelMapping[];
  /**
   * The number of input channels.
   * @default - auto-detected
   */
  readonly channelsIn?: number;
  /**
   * The number of output channels. Valid values: 1, 2, 4, 6, 8.
   * @default - auto-detected
   */
  readonly channelsOut?: number;
}

/**
 * CBET insertion behavior when prior encoding is detected on the same layer.
 */
export class NielsenCbetStepaside {
  /**
   * Existing Nielsen watermarks are removed. New watermarks are inserted throughout the audio.
   */
  public static readonly DISABLED = new NielsenCbetStepaside('DISABLED');
  /**
   * Existing Nielsen watermarks are left intact. New watermarks are inserted only in portions
   * of the audio where there are no existing watermarks.
   */
  public static readonly ENABLED = new NielsenCbetStepaside('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): NielsenCbetStepaside {
    return new NielsenCbetStepaside(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Timezone applied to the timestamps in a Nielsen NAES II/NW watermark.
 */
export class NielsenWatermarkTimezone {
  /** America/Puerto Rico */
  public static readonly AMERICA_PUERTO_RICO = new NielsenWatermarkTimezone('AMERICA_PUERTO_RICO');
  /** US Alaska */
  public static readonly US_ALASKA = new NielsenWatermarkTimezone('US_ALASKA');
  /** US Arizona */
  public static readonly US_ARIZONA = new NielsenWatermarkTimezone('US_ARIZONA');
  /** US Central */
  public static readonly US_CENTRAL = new NielsenWatermarkTimezone('US_CENTRAL');
  /** US Eastern */
  public static readonly US_EASTERN = new NielsenWatermarkTimezone('US_EASTERN');
  /** US Hawaii */
  public static readonly US_HAWAII = new NielsenWatermarkTimezone('US_HAWAII');
  /** US Mountain */
  public static readonly US_MOUNTAIN = new NielsenWatermarkTimezone('US_MOUNTAIN');
  /** US Pacific */
  public static readonly US_PACIFIC = new NielsenWatermarkTimezone('US_PACIFIC');
  /** US Samoa */
  public static readonly US_SAMOA = new NielsenWatermarkTimezone('US_SAMOA');
  /** Coordinated Universal Time */
  public static readonly UTC = new NielsenWatermarkTimezone('UTC');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): NielsenWatermarkTimezone {
    return new NielsenWatermarkTimezone(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Nielsen CBET watermark settings.
 */
export interface NielsenCbetSettings {
  /**
   * The CBET check digit string.
   */
  readonly cbetCheckDigitString: string;
  /**
   * The CBET Source ID (CSID).
   */
  readonly csid: string;
  /**
   * The CBET stepaside behavior when prior encoding is detected.
   * @default - service default
   */
  readonly cbetStepaside?: NielsenCbetStepaside;
}

/**
 * Nielsen NAES II/NW watermark settings.
 */
export interface NielsenNaesIiNwSettings {
  /**
   * The check digit string for the watermark.
   */
  readonly checkDigitString: string;
  /**
   * The Nielsen Source ID (SID).
   */
  readonly sid: number;
  /**
   * The timezone for the timestamps in the watermark.
   * @default - Coordinated Universal Time (UTC)
   */
  readonly timezone?: NielsenWatermarkTimezone;
}

/**
 * Nielsen watermark distribution type.
 */
export class NielsenDistributionType {
  /** Program content */
  public static readonly PROGRAM_CONTENT = new NielsenDistributionType('PROGRAM_CONTENT');
  /** Final distributor */
  public static readonly FINAL_DISTRIBUTOR = new NielsenDistributionType('FINAL_DISTRIBUTOR');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): NielsenDistributionType {
    return new NielsenDistributionType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Nielsen watermark settings for audio.
 */
export interface NielsenWatermarksSettings {
  /**
   * The distribution type for the watermark.
   * @default - service default
   */
  readonly distributionType?: NielsenDistributionType;
  /**
   * Nielsen CBET watermark settings.
   * @default - no CBET watermarks
   */
  readonly cbetSettings?: NielsenCbetSettings;
  /**
   * Nielsen NAES II/NW watermark settings.
   * @default - no NAES II/NW watermarks
   */
  readonly naesIiNwSettings?: NielsenNaesIiNwSettings;
}

/**
 * Audio watermarking settings.
 */
export interface AudioWatermarkSettings {
  /**
   * Nielsen watermark settings.
   * @default - no Nielsen watermarks
   */
  readonly nielsenWatermarks?: NielsenWatermarksSettings;
}

/**
 * Determines how the audio type is signaled in the output.
 */
export class AudioTypeControl {
  /**
   * If the input contains an ISO 639 audioType it is passed through; otherwise the
   * configured `audioType` is used.
   */
  public static readonly FOLLOW_INPUT = new AudioTypeControl('FOLLOW_INPUT');
  /** The configured `audioType` is always used. */
  public static readonly USE_CONFIGURED = new AudioTypeControl('USE_CONFIGURED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioTypeControl {
    return new AudioTypeControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Determines how the audio language code is signaled in the output.
 */
export class AudioLanguageCodeControl {
  /**
   * If the input contains a language code it is passed through; otherwise the configured
   * `languageCode` is used as a fallback.
   */
  public static readonly FOLLOW_INPUT = new AudioLanguageCodeControl('FOLLOW_INPUT');
  /** The configured `languageCode` is always used. */
  public static readonly USE_CONFIGURED = new AudioLanguageCodeControl('USE_CONFIGURED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioLanguageCodeControl {
    return new AudioLanguageCodeControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * The audio type, as defined in ISO/IEC 13818-1.
 */
export class AudioType {
  /** Clean effects (no dialogue). */
  public static readonly CLEAN_EFFECTS = new AudioType('CLEAN_EFFECTS');
  /** Hearing impaired. */
  public static readonly HEARING_IMPAIRED = new AudioType('HEARING_IMPAIRED');
  /** Undefined. */
  public static readonly UNDEFINED = new AudioType('UNDEFINED');
  /** Visual impaired commentary. */
  public static readonly VISUAL_IMPAIRED_COMMENTARY = new AudioType('VISUAL_IMPAIRED_COMMENTARY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioType {
    return new AudioType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * DVB DASH accessibility signaling for an audio output.
 */
export class DvbDashAccessibility {
  /** Visually impaired. */
  public static readonly VISUALLY_IMPAIRED = new DvbDashAccessibility('DVBDASH_1_VISUALLY_IMPAIRED');
  /** Hard of hearing. */
  public static readonly HARD_OF_HEARING = new DvbDashAccessibility('DVBDASH_2_HARD_OF_HEARING');
  /** Supplemental commentary. */
  public static readonly SUPPLEMENTAL_COMMENTARY = new DvbDashAccessibility('DVBDASH_3_SUPPLEMENTAL_COMMENTARY');
  /** Director's commentary. */
  public static readonly DIRECTORS_COMMENTARY = new DvbDashAccessibility('DVBDASH_4_DIRECTORS_COMMENTARY');
  /** Educational notes. */
  public static readonly EDUCATIONAL_NOTES = new DvbDashAccessibility('DVBDASH_5_EDUCATIONAL_NOTES');
  /** Main program. */
  public static readonly MAIN_PROGRAM = new DvbDashAccessibility('DVBDASH_6_MAIN_PROGRAM');
  /** Clean feed. */
  public static readonly CLEAN_FEED = new DvbDashAccessibility('DVBDASH_7_CLEAN_FEED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): DvbDashAccessibility {
    return new DvbDashAccessibility(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * A DASH role to assign to an audio output (used when the output carries DVB DASH accessibility
 * signaling).
 */
export class AudioDashRole {
  /** Alternate. */
  public static readonly ALTERNATE = new AudioDashRole('ALTERNATE');
  /** Commentary. */
  public static readonly COMMENTARY = new AudioDashRole('COMMENTARY');
  /** Description. */
  public static readonly DESCRIPTION = new AudioDashRole('DESCRIPTION');
  /** Dub. */
  public static readonly DUB = new AudioDashRole('DUB');
  /** Emergency. */
  public static readonly EMERGENCY = new AudioDashRole('EMERGENCY');
  /** Enhanced audio intelligibility. */
  public static readonly ENHANCED_AUDIO_INTELLIGIBILITY = new AudioDashRole('ENHANCED-AUDIO-INTELLIGIBILITY');
  /** Karaoke. */
  public static readonly KARAOKE = new AudioDashRole('KARAOKE');
  /** Main. */
  public static readonly MAIN = new AudioDashRole('MAIN');
  /** Supplementary. */
  public static readonly SUPPLEMENTARY = new AudioDashRole('SUPPLEMENTARY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioDashRole {
    return new AudioDashRole(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties for an audio encode configuration.
 */
export interface AudioEncodeProps {
  /**
   * A unique name for this audio encode.
   */
  readonly name: string;
  /**
   * The name of the audio selector in the input to use as the source. Must match the `name` of an
   * `AudioSelector` on the input attachment. When omitted, MediaLive uses the input's default audio.
   * @default - the input's default audio
   */
  readonly audioSelectorName?: string;
  /**
   * The codec for the audio encode.
   *
   * Choose the codec explicitly (e.g. `AudioCodecSettings.aac(...)`)
   */
  readonly codec: AudioCodecSettings;
  /**
   * The ISO 639-2 language code for the audio output track (e.g. 'eng', 'spa').
   * @default - follow input
   */
  readonly languageCode?: string;
  /**
   * How the audio language code is signaled in the output. When `FOLLOW_INPUT`, a configured
   * `languageCode` is used only as a fallback when the input has none.
   * @default - USE_CONFIGURED when `languageCode` is set, otherwise FOLLOW_INPUT
   */
  readonly languageCodeControl?: AudioLanguageCodeControl;
  /**
   * The display name for the audio track (e.g. 'English', 'Director Commentary').
   * Used for HLS and MS Smooth outputs.
   * @default - no stream name
   */
  readonly streamName?: string;
  /**
   * Audio normalization settings for loudness correction.
   * @default - no normalization
   */
  readonly audioNormalization?: AudioNormalizationSettings;
  /**
   * The audio type when audioTypeControl is USE_CONFIGURED. The values are defined in ISO-IEC 13818-1.
   * @default - follow input
   */
  readonly audioType?: AudioType;
  /**
   * How the audio type is signaled in the output.
   * @default - USE_CONFIGURED when `audioType` is set, otherwise FOLLOW_INPUT
   */
  readonly audioTypeControl?: AudioTypeControl;
  /**
   * The DASH roles to assign to this audio output. Applies only when the output is configured
   * for DVB DASH accessibility signaling.
   * @default - no DASH roles
   */
  readonly audioDashRoles?: AudioDashRole[];
  /**
   * DVB DASH accessibility signaling for this audio output.
   * @default - no DVB DASH accessibility signaling
   */
  readonly dvbDashAccessibility?: DvbDashAccessibility;
  /**
   * Audio remix settings for channel remapping.
   * @default - no remixing
   */
  readonly remixSettings?: RemixSettings;
  /**
   * Audio watermarking settings (e.g. Nielsen watermarks).
   * @default - no watermarking
   */
  readonly audioWatermarkSettings?: AudioWatermarkSettings;
}

/** @internal */
class VideoEncodeConfiguration extends EncodeConfiguration {
  public readonly name: string;
  private readonly props: VideoEncodeProps;

  constructor(props: VideoEncodeProps) {
    super();
    if (!Token.isUnresolved(props.width) && props.width % 2 !== 0) {
      throw new UnscopedValidationError(lit`VideoWidthEven`, `Video width must be an even number, got ${props.width}.`);
    }
    if (!Token.isUnresolved(props.height) && props.height % 2 !== 0) {
      throw new UnscopedValidationError(lit`VideoHeightEven`, `Video height must be an even number, got ${props.height}.`);
    }
    this.name = props.name;
    this.props = props;
  }

  public _hasExplicitFramerate(): boolean {
    return this.props.codec._hasExplicitFramerate();
  }

  public _bindVideo(): CfnChannel.VideoDescriptionProperty {
    return {
      name: this.name,
      width: this.props.width,
      height: this.props.height,
      respondToAfd: (this.props.respondToAfd ?? RespondToAfd.NONE).value,
      scalingBehavior: (this.props.scalingBehavior ?? ScalingBehavior.DEFAULT).value,
      sharpness: this.props.sharpness ?? 50,
      codecSettings: this.props.codec._bind(),
    };
  }

  public _bindAudio(): undefined {
    return undefined;
  }

  public _bindCaption(): undefined {
    return undefined;
  }

  public _videoCodecType(): VideoCodecType {
    return this.props.codec._codecType;
  }

  public _audioCodecType(): undefined {
    return undefined;
  }
}

/** @internal */
class AudioEncodeConfiguration extends EncodeConfiguration {
  public readonly name: string;
  private readonly props: AudioEncodeProps;

  constructor(props: AudioEncodeProps) {
    super();
    this.name = props.name;
    this.props = props;
  }

  public _hasExplicitFramerate(): boolean {
    return true; // Not applicable for audio
  }

  public _bindVideo(): undefined {
    return undefined;
  }

  public _bindAudio(): CfnChannel.AudioDescriptionProperty {
    const codecSettings = this.props.codec;
    return {
      name: this.name,
      audioSelectorName: this.props.audioSelectorName,
      audioTypeControl: this.props.audioTypeControl?.value
        ?? (this.props.audioType !== undefined ? AudioTypeControl.USE_CONFIGURED : AudioTypeControl.FOLLOW_INPUT).value,
      languageCode: this.props.languageCode,
      languageCodeControl: this.props.languageCodeControl?.value
        ?? (this.props.languageCode !== undefined ? AudioLanguageCodeControl.USE_CONFIGURED : AudioLanguageCodeControl.FOLLOW_INPUT).value,
      streamName: this.props.streamName,
      audioType: this.props.audioType?.value,
      audioDashRoles: this.props.audioDashRoles?.map(r => r.value),
      dvbDashAccessibility: this.props.dvbDashAccessibility?.value,
      codecSettings: codecSettings._bind(),
      audioNormalizationSettings: this.props.audioNormalization ? {
        algorithm: this.props.audioNormalization.algorithm?.value,
        algorithmControl: this.props.audioNormalization.algorithmControl?.value,
        targetLkfs: this.props.audioNormalization.targetLkfs,
        peakCalculation: this.props.audioNormalization.peakCalculation?.value,
        peakLimiterThreshold: this.props.audioNormalization.peakLimiterThreshold,
      } : undefined,
      remixSettings: this.props.remixSettings ? {
        channelMappings: this.props.remixSettings.channelMappings.map(m => ({
          outputChannel: m.outputChannel,
          inputChannelLevels: m.inputChannelLevels.map(l => ({
            inputChannel: l.inputChannel,
            gain: l.gain ?? 0,
          })),
        })),
        channelsIn: this.props.remixSettings.channelsIn,
        channelsOut: this.props.remixSettings.channelsOut,
      } : undefined,
      audioWatermarkingSettings: this.props.audioWatermarkSettings ? {
        nielsenWatermarksSettings: this.props.audioWatermarkSettings.nielsenWatermarks ? {
          nielsenDistributionType: this.props.audioWatermarkSettings.nielsenWatermarks.distributionType?.value,
          nielsenCbetSettings: this.props.audioWatermarkSettings.nielsenWatermarks.cbetSettings ? {
            cbetCheckDigitString: this.props.audioWatermarkSettings.nielsenWatermarks.cbetSettings.cbetCheckDigitString,
            cbetStepaside: this.props.audioWatermarkSettings.nielsenWatermarks.cbetSettings.cbetStepaside?.value,
            csid: this.props.audioWatermarkSettings.nielsenWatermarks.cbetSettings.csid,
          } : undefined,
          nielsenNaesIiNwSettings: this.props.audioWatermarkSettings.nielsenWatermarks.naesIiNwSettings ? {
            checkDigitString: this.props.audioWatermarkSettings.nielsenWatermarks.naesIiNwSettings.checkDigitString,
            sid: this.props.audioWatermarkSettings.nielsenWatermarks.naesIiNwSettings.sid,
            timezone: this.props.audioWatermarkSettings.nielsenWatermarks.naesIiNwSettings.timezone?.value,
          } : undefined,
        } : undefined,
      } : undefined,
    };
  }

  public _bindCaption(): undefined {
    return undefined;
  }

  public _videoCodecType(): undefined {
    return undefined;
  }

  public _audioCodecType(): AudioCodecType {
    return this.props.codec._codecType;
  }
}

/**
 * Whether a caption track implements accessibility features (written descriptions of dialog,
 * music, and sounds). Signaled in HLS and MediaPackage output groups.
 */
export class CaptionAccessibility {
  /** The captions do not implement accessibility features. */
  public static readonly DOES_NOT_IMPLEMENT_ACCESSIBILITY_FEATURES = new CaptionAccessibility('DOES_NOT_IMPLEMENT_ACCESSIBILITY_FEATURES');
  /** The captions implement accessibility features. */
  public static readonly IMPLEMENTS_ACCESSIBILITY_FEATURES = new CaptionAccessibility('IMPLEMENTS_ACCESSIBILITY_FEATURES');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionAccessibility {
    return new CaptionAccessibility(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * A DASH role to assign to a captions output (used when the output carries DVB DASH accessibility
 * signaling).
 */
export class CaptionDashRole {
  /** Alternate. */
  public static readonly ALTERNATE = new CaptionDashRole('ALTERNATE');
  /** Caption. */
  public static readonly CAPTION = new CaptionDashRole('CAPTION');
  /** Commentary. */
  public static readonly COMMENTARY = new CaptionDashRole('COMMENTARY');
  /** Description. */
  public static readonly DESCRIPTION = new CaptionDashRole('DESCRIPTION');
  /** Dub. */
  public static readonly DUB = new CaptionDashRole('DUB');
  /** Easy reader. */
  public static readonly EASYREADER = new CaptionDashRole('EASYREADER');
  /** Emergency. */
  public static readonly EMERGENCY = new CaptionDashRole('EMERGENCY');
  /** Forced subtitle. */
  public static readonly FORCED_SUBTITLE = new CaptionDashRole('FORCED-SUBTITLE');
  /** Karaoke. */
  public static readonly KARAOKE = new CaptionDashRole('KARAOKE');
  /** Main. */
  public static readonly MAIN = new CaptionDashRole('MAIN');
  /** Metadata. */
  public static readonly METADATA = new CaptionDashRole('METADATA');
  /** Subtitle. */
  public static readonly SUBTITLE = new CaptionDashRole('SUBTITLE');
  /** Supplementary. */
  public static readonly SUPPLEMENTARY = new CaptionDashRole('SUPPLEMENTARY');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionDashRole {
    return new CaptionDashRole(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Properties for a caption encode configuration.
 */
export interface CaptionEncodeProps {
  /**
   * A unique name for this caption encode.
   */
  readonly name: string;
  /**
   * The name of the caption selector in the input to use as the source.
   */
  readonly captionSelectorName: string;
  /**
   * The output caption format. Use the `CaptionDestination` factory methods (e.g.
   * `CaptionDestination.burnIn()`, `.webvtt()`, `.embedded()`).
   */
  readonly destination: CaptionDestination;
  /**
   * The ISO 639-2 language code for the captions (e.g. 'eng', 'spa').
   * @default - no language code
   */
  readonly languageCode?: string;
  /**
   * Human-readable description of the captions (e.g. 'English', 'Spanish').
   * @default - no language description
   */
  readonly languageDescription?: string;
  /**
   * Whether this caption track implements accessibility features.
   * @default - The captions do not implement accessibility features
   */
  readonly accessibility?: CaptionAccessibility;
  /**
   * The DASH roles to assign to this captions output. Applies only when the output is configured
   * for DVB DASH accessibility signaling.
   * @default - no DASH roles
   */
  readonly captionDashRoles?: CaptionDashRole[];
  /**
   * DVB DASH accessibility signaling for this captions output.
   * @default - no DVB DASH accessibility signaling
   */
  readonly dvbDashAccessibility?: DvbDashAccessibility;
}

/** @internal */
class CaptionEncodeConfiguration extends EncodeConfiguration {
  public readonly name: string;
  private readonly props: CaptionEncodeProps;

  constructor(props: CaptionEncodeProps) {
    super();
    this.name = props.name;
    this.props = props;
  }

  public _hasExplicitFramerate(): boolean {
    return true; // Not applicable for captions
  }

  public _bindVideo(): undefined {
    return undefined;
  }

  public _bindAudio(): undefined {
    return undefined;
  }

  public _bindCaption(): CfnChannel.CaptionDescriptionProperty {
    return {
      name: this.name,
      captionSelectorName: this.props.captionSelectorName,
      destinationSettings: this.props.destination._bind(),
      languageCode: this.props.languageCode,
      languageDescription: this.props.languageDescription,
      accessibility: (this.props.accessibility ?? CaptionAccessibility.DOES_NOT_IMPLEMENT_ACCESSIBILITY_FEATURES).value,
      captionDashRoles: this.props.captionDashRoles?.map(r => r.value),
      dvbDashAccessibility: this.props.dvbDashAccessibility?.value,
    };
  }

  public _videoCodecType(): undefined {
    return undefined;
  }

  public _audioCodecType(): undefined {
    return undefined;
  }

  public override _grantRead(role: IRole): void {
    this.props.destination._grantRead(role);
  }

  public override _isInBandCaption(): boolean {
    return this.props.destination._isInBand();
  }

  public override _isEmbeddedCaption(): boolean {
    return this.props.destination._isEmbedded();
  }
}
