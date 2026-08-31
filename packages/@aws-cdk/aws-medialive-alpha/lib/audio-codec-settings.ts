import type { Bitrate } from 'aws-cdk-lib';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';

/**
 * Audio sample rate for AAC, MP2, and WAV codecs.
 *
 * Use one of the standard presets or `AudioSampleRate.of(hz)` for a custom value.
 */
export class AudioSampleRate {
  /** 8,000 Hz */
  public static readonly HZ_8000 = new AudioSampleRate(8000);
  /** 12,000 Hz */
  public static readonly HZ_12000 = new AudioSampleRate(12000);
  /** 16,000 Hz */
  public static readonly HZ_16000 = new AudioSampleRate(16000);
  /** 22,050 Hz */
  public static readonly HZ_22050 = new AudioSampleRate(22050);
  /** 24,000 Hz */
  public static readonly HZ_24000 = new AudioSampleRate(24000);
  /** 32,000 Hz */
  public static readonly HZ_32000 = new AudioSampleRate(32000);
  /** 44,100 Hz */
  public static readonly HZ_44100 = new AudioSampleRate(44100);
  /** 48,000 Hz */
  public static readonly HZ_48000 = new AudioSampleRate(48000);
  /** 88,200 Hz */
  public static readonly HZ_88200 = new AudioSampleRate(88200);
  /** 96,000 Hz */
  public static readonly HZ_96000 = new AudioSampleRate(96000);

  /**
   * A custom sample rate in Hz.
   * @param hz the sample rate in Hz
   */
  public static of(hz: number): AudioSampleRate {
    return new AudioSampleRate(hz);
  }

  private constructor(private readonly hz: number) {}

  /** @internal */
  public _toHz(): number {
    return this.hz;
  }
}

/**
 * Audio bit depth for WAV codec.
 *
 * Use one of the standard presets or `AudioBitDepth.of(bits)` for a custom value.
 */
export class AudioBitDepth {
  /** 16-bit */
  public static readonly DEPTH_16 = new AudioBitDepth(16);
  /** 24-bit */
  public static readonly DEPTH_24 = new AudioBitDepth(24);

  /**
   * A custom bit depth.
   * @param bits the bit depth
   */
  public static of(bits: number): AudioBitDepth {
    return new AudioBitDepth(bits);
  }

  private constructor(private readonly bits: number) {}

  /** @internal */
  public _toBits(): number {
    return this.bits;
  }
}

/**
 * The type of audio codec. Users select a codec via the
 * `AudioCodecSettings` factory methods, never by passing this type directly.
 * @internal
 */
export enum AudioCodecType {
  /** AAC */
  AAC = 'AAC',
  /** Dolby Digital (AC3) */
  AC3 = 'AC3',
  /** Dolby Digital Plus (EAC3) */
  EAC3 = 'EAC3',
  /** Dolby Digital Plus with Atmos (EAC3 Atmos) */
  EAC3_ATMOS = 'EAC3_ATMOS',
  /** MPEG-1 Layer II (MP2) */
  MP2 = 'MP2',
  /** WAV */
  WAV = 'WAV',
  /** Passthrough (no transcoding) */
  PASSTHROUGH = 'PASSTHROUGH',
}

/**
 * AAC profile.
 */
export class AacProfile {
  /** HEV1 */
  public static readonly HEV1 = new AacProfile('HEV1');
  /** HEV2 */
  public static readonly HEV2 = new AacProfile('HEV2');
  /** LC (Low Complexity) */
  public static readonly LC = new AacProfile('LC');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacProfile {
    return new AacProfile(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AAC coding mode.
 */
export class AacCodingMode {
  /** Ad receiver mix — receives stereo audio description + control track per ETSI TS 101 154 Annex E. */
  public static readonly AD_RECEIVER_MIX = new AacCodingMode('AD_RECEIVER_MIX');
  /** 1.0 (mono) */
  public static readonly CODING_MODE_1_0 = new AacCodingMode('CODING_MODE_1_0');
  /** 1+1 (dual mono) */
  public static readonly CODING_MODE_1_1 = new AacCodingMode('CODING_MODE_1_1');
  /** 2.0 (stereo) */
  public static readonly CODING_MODE_2_0 = new AacCodingMode('CODING_MODE_2_0');
  /** 5.1 surround */
  public static readonly CODING_MODE_5_1 = new AacCodingMode('CODING_MODE_5_1');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacCodingMode {
    return new AacCodingMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AAC rate control mode.
 */
export class AacRateControlMode {
  /** Constant bitrate */
  public static readonly CBR = new AacRateControlMode('CBR');
  /** Variable bitrate */
  public static readonly VBR = new AacRateControlMode('VBR');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacRateControlMode {
    return new AacRateControlMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AAC raw format.
 */
export class AacRawFormat {
  /** LATM/LOAS */
  public static readonly LATM_LOAS = new AacRawFormat('LATM_LOAS');
  /** None */
  public static readonly NONE = new AacRawFormat('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacRawFormat {
    return new AacRawFormat(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AAC specification.
 */
export class AacSpec {
  /** MPEG-2 AAC */
  public static readonly MPEG2 = new AacSpec('MPEG2');
  /** MPEG-4 AAC */
  public static readonly MPEG4 = new AacSpec('MPEG4');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacSpec {
    return new AacSpec(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AAC input type.
 */
export class AacInputType {
  /** Broadcaster mixed AD */
  public static readonly BROADCASTER_MIXED_AD = new AacInputType('BROADCASTER_MIXED_AD');
  /** Normal */
  public static readonly NORMAL = new AacInputType('NORMAL');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacInputType {
    return new AacInputType(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AAC VBR quality level.
 */
export class AacVbrQuality {
  /** High */
  public static readonly HIGH = new AacVbrQuality('HIGH');
  /** Low */
  public static readonly LOW = new AacVbrQuality('LOW');
  /** Medium high */
  public static readonly MEDIUM_HIGH = new AacVbrQuality('MEDIUM_HIGH');
  /** Medium low */
  public static readonly MEDIUM_LOW = new AacVbrQuality('MEDIUM_LOW');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AacVbrQuality {
    return new AacVbrQuality(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// AC3 enums
// =============================================================================

/**
 * AC3 attenuation control.
 */
export class Ac3AttenuationControl {
  /** Apply 3 dB attenuation to surround channels */
  public static readonly ATTENUATE_3_DB = new Ac3AttenuationControl('ATTENUATE_3_DB');
  /** No attenuation */
  public static readonly NONE = new Ac3AttenuationControl('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Ac3AttenuationControl {
    return new Ac3AttenuationControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AC3 bitstream mode.
 */
export class Ac3BitstreamMode {
  /** Commentary */
  public static readonly COMMENTARY = new Ac3BitstreamMode('COMMENTARY');
  /** Complete main */
  public static readonly COMPLETE_MAIN = new Ac3BitstreamMode('COMPLETE_MAIN');
  /** Dialogue */
  public static readonly DIALOGUE = new Ac3BitstreamMode('DIALOGUE');
  /** Emergency */
  public static readonly EMERGENCY = new Ac3BitstreamMode('EMERGENCY');
  /** Hearing impaired */
  public static readonly HEARING_IMPAIRED = new Ac3BitstreamMode('HEARING_IMPAIRED');
  /** Music and effects */
  public static readonly MUSIC_AND_EFFECTS = new Ac3BitstreamMode('MUSIC_AND_EFFECTS');
  /** Visually impaired */
  public static readonly VISUALLY_IMPAIRED = new Ac3BitstreamMode('VISUALLY_IMPAIRED');
  /** Voice over */
  public static readonly VOICE_OVER = new Ac3BitstreamMode('VOICE_OVER');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Ac3BitstreamMode {
    return new Ac3BitstreamMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AC3 coding mode.
 */
export class Ac3CodingMode {
  /** 1.0 (mono) */
  public static readonly CODING_MODE_1_0 = new Ac3CodingMode('CODING_MODE_1_0');
  /** 1+1 (dual mono) */
  public static readonly CODING_MODE_1_1 = new Ac3CodingMode('CODING_MODE_1_1');
  /** 2.0 (stereo) */
  public static readonly CODING_MODE_2_0 = new Ac3CodingMode('CODING_MODE_2_0');
  /** 3/2 (5.0 surround) */
  public static readonly CODING_MODE_3_2_LFE = new Ac3CodingMode('CODING_MODE_3_2_LFE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Ac3CodingMode {
    return new Ac3CodingMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AC3 DRC profile.
 */
export class Ac3DrcProfile {
  /** Film standard */
  public static readonly FILM_STANDARD = new Ac3DrcProfile('FILM_STANDARD');
  /** None */
  public static readonly NONE = new Ac3DrcProfile('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Ac3DrcProfile {
    return new Ac3DrcProfile(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AC3 LFE filter.
 */
export class Ac3LfeFilter {
  /** Disabled */
  public static readonly DISABLED = new Ac3LfeFilter('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Ac3LfeFilter('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Ac3LfeFilter {
    return new Ac3LfeFilter(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * AC3 metadata control.
 */
export class Ac3MetadataControl {
  /** Follow input */
  public static readonly FOLLOW_INPUT = new Ac3MetadataControl('FOLLOW_INPUT');
  /** Use configured */
  public static readonly USE_CONFIGURED = new Ac3MetadataControl('USE_CONFIGURED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Ac3MetadataControl {
    return new Ac3MetadataControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// EAC3 enums
// =============================================================================

/**
 * EAC3 attenuation control.
 */
export class Eac3AttenuationControl {
  /** Apply 3 dB attenuation to surround channels */
  public static readonly ATTENUATE_3_DB = new Eac3AttenuationControl('ATTENUATE_3_DB');
  /** No attenuation */
  public static readonly NONE = new Eac3AttenuationControl('NONE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3AttenuationControl {
    return new Eac3AttenuationControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 bitstream mode.
 */
export class Eac3BitstreamMode {
  /** Commentary */
  public static readonly COMMENTARY = new Eac3BitstreamMode('COMMENTARY');
  /** Complete main */
  public static readonly COMPLETE_MAIN = new Eac3BitstreamMode('COMPLETE_MAIN');
  /** Emergency */
  public static readonly EMERGENCY = new Eac3BitstreamMode('EMERGENCY');
  /** Hearing impaired */
  public static readonly HEARING_IMPAIRED = new Eac3BitstreamMode('HEARING_IMPAIRED');
  /** Visually impaired */
  public static readonly VISUALLY_IMPAIRED = new Eac3BitstreamMode('VISUALLY_IMPAIRED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3BitstreamMode {
    return new Eac3BitstreamMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 coding mode.
 */
export class Eac3CodingMode {
  /** 1.0 (mono) */
  public static readonly CODING_MODE_1_0 = new Eac3CodingMode('CODING_MODE_1_0');
  /** 2.0 (stereo) */
  public static readonly CODING_MODE_2_0 = new Eac3CodingMode('CODING_MODE_2_0');
  /** 3/2 (5.0 surround) */
  public static readonly CODING_MODE_3_2 = new Eac3CodingMode('CODING_MODE_3_2');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3CodingMode {
    return new Eac3CodingMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 DC filter.
 */
export class Eac3DcFilter {
  /** Disabled */
  public static readonly DISABLED = new Eac3DcFilter('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Eac3DcFilter('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3DcFilter {
    return new Eac3DcFilter(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 DRC line mode profile.
 */
export class Eac3DrcLine {
  /** Film light */
  public static readonly FILM_LIGHT = new Eac3DrcLine('FILM_LIGHT');
  /** Film standard */
  public static readonly FILM_STANDARD = new Eac3DrcLine('FILM_STANDARD');
  /** Music light */
  public static readonly MUSIC_LIGHT = new Eac3DrcLine('MUSIC_LIGHT');
  /** Music standard */
  public static readonly MUSIC_STANDARD = new Eac3DrcLine('MUSIC_STANDARD');
  /** None */
  public static readonly NONE = new Eac3DrcLine('NONE');
  /** Speech */
  public static readonly SPEECH = new Eac3DrcLine('SPEECH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3DrcLine {
    return new Eac3DrcLine(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 DRC RF mode profile.
 */
export class Eac3DrcRf {
  /** Film light */
  public static readonly FILM_LIGHT = new Eac3DrcRf('FILM_LIGHT');
  /** Film standard */
  public static readonly FILM_STANDARD = new Eac3DrcRf('FILM_STANDARD');
  /** Music light */
  public static readonly MUSIC_LIGHT = new Eac3DrcRf('MUSIC_LIGHT');
  /** Music standard */
  public static readonly MUSIC_STANDARD = new Eac3DrcRf('MUSIC_STANDARD');
  /** None */
  public static readonly NONE = new Eac3DrcRf('NONE');
  /** Speech */
  public static readonly SPEECH = new Eac3DrcRf('SPEECH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3DrcRf {
    return new Eac3DrcRf(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 LFE control.
 */
export class Eac3LfeControl {
  /** LFE */
  public static readonly LFE = new Eac3LfeControl('LFE');
  /** No LFE */
  public static readonly NO_LFE = new Eac3LfeControl('NO_LFE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3LfeControl {
    return new Eac3LfeControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 LFE filter.
 */
export class Eac3LfeFilter {
  /** Disabled */
  public static readonly DISABLED = new Eac3LfeFilter('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Eac3LfeFilter('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3LfeFilter {
    return new Eac3LfeFilter(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 metadata control.
 */
export class Eac3MetadataControl {
  /** Follow input */
  public static readonly FOLLOW_INPUT = new Eac3MetadataControl('FOLLOW_INPUT');
  /** Use configured */
  public static readonly USE_CONFIGURED = new Eac3MetadataControl('USE_CONFIGURED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3MetadataControl {
    return new Eac3MetadataControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 passthrough control.
 */
export class Eac3PassthroughControl {
  /** No passthrough */
  public static readonly NO_PASSTHROUGH = new Eac3PassthroughControl('NO_PASSTHROUGH');
  /** When possible */
  public static readonly WHEN_POSSIBLE = new Eac3PassthroughControl('WHEN_POSSIBLE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3PassthroughControl {
    return new Eac3PassthroughControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 phase control.
 */
export class Eac3PhaseControl {
  /** No shift */
  public static readonly NO_SHIFT = new Eac3PhaseControl('NO_SHIFT');
  /** Shift 90 degrees */
  public static readonly SHIFT_90_DEGREES = new Eac3PhaseControl('SHIFT_90_DEGREES');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3PhaseControl {
    return new Eac3PhaseControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 stereo downmix preference.
 */
export class Eac3StereoDownmix {
  /** DPL2 */
  public static readonly DPL2 = new Eac3StereoDownmix('DPL2');
  /** Lo/Ro */
  public static readonly LO_RO = new Eac3StereoDownmix('LO_RO');
  /** Lt/Rt */
  public static readonly LT_RT = new Eac3StereoDownmix('LT_RT');
  /** Not indicated */
  public static readonly NOT_INDICATED = new Eac3StereoDownmix('NOT_INDICATED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3StereoDownmix {
    return new Eac3StereoDownmix(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 surround ex mode.
 */
export class Eac3SurroundExMode {
  /** Disabled */
  public static readonly DISABLED = new Eac3SurroundExMode('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Eac3SurroundExMode('ENABLED');
  /** Not indicated */
  public static readonly NOT_INDICATED = new Eac3SurroundExMode('NOT_INDICATED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3SurroundExMode {
    return new Eac3SurroundExMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 surround mode.
 */
export class Eac3SurroundMode {
  /** Disabled */
  public static readonly DISABLED = new Eac3SurroundMode('DISABLED');
  /** Enabled */
  public static readonly ENABLED = new Eac3SurroundMode('ENABLED');
  /** Not indicated */
  public static readonly NOT_INDICATED = new Eac3SurroundMode('NOT_INDICATED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3SurroundMode {
    return new Eac3SurroundMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// EAC3 Atmos enums
// =============================================================================

/**
 * EAC3 Atmos coding mode.
 */
export class Eac3AtmosCodingMode {
  /** 5.1.4 surround */
  public static readonly CODING_MODE_5_1_4 = new Eac3AtmosCodingMode('CODING_MODE_5_1_4');
  /** 7.1.4 surround */
  public static readonly CODING_MODE_7_1_4 = new Eac3AtmosCodingMode('CODING_MODE_7_1_4');
  /** 9.1.6 surround */
  public static readonly CODING_MODE_9_1_6 = new Eac3AtmosCodingMode('CODING_MODE_9_1_6');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3AtmosCodingMode {
    return new Eac3AtmosCodingMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 Atmos DRC line mode profile.
 */
export class Eac3AtmosDrcLine {
  /** Film light */
  public static readonly FILM_LIGHT = new Eac3AtmosDrcLine('FILM_LIGHT');
  /** Film standard */
  public static readonly FILM_STANDARD = new Eac3AtmosDrcLine('FILM_STANDARD');
  /** Music light */
  public static readonly MUSIC_LIGHT = new Eac3AtmosDrcLine('MUSIC_LIGHT');
  /** Music standard */
  public static readonly MUSIC_STANDARD = new Eac3AtmosDrcLine('MUSIC_STANDARD');
  /** None */
  public static readonly NONE = new Eac3AtmosDrcLine('NONE');
  /** Speech */
  public static readonly SPEECH = new Eac3AtmosDrcLine('SPEECH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3AtmosDrcLine {
    return new Eac3AtmosDrcLine(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * EAC3 Atmos DRC RF mode profile.
 */
export class Eac3AtmosDrcRf {
  /** Film light */
  public static readonly FILM_LIGHT = new Eac3AtmosDrcRf('FILM_LIGHT');
  /** Film standard */
  public static readonly FILM_STANDARD = new Eac3AtmosDrcRf('FILM_STANDARD');
  /** Music light */
  public static readonly MUSIC_LIGHT = new Eac3AtmosDrcRf('MUSIC_LIGHT');
  /** Music standard */
  public static readonly MUSIC_STANDARD = new Eac3AtmosDrcRf('MUSIC_STANDARD');
  /** None */
  public static readonly NONE = new Eac3AtmosDrcRf('NONE');
  /** Speech */
  public static readonly SPEECH = new Eac3AtmosDrcRf('SPEECH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Eac3AtmosDrcRf {
    return new Eac3AtmosDrcRf(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// MP2 enums
// =============================================================================

/**
 * MP2 coding mode.
 */
export class Mp2CodingMode {
  /** 1.0 (mono) */
  public static readonly CODING_MODE_1_0 = new Mp2CodingMode('CODING_MODE_1_0');
  /** 2.0 (stereo) */
  public static readonly CODING_MODE_2_0 = new Mp2CodingMode('CODING_MODE_2_0');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Mp2CodingMode {
    return new Mp2CodingMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// WAV enums
// =============================================================================

/**
 * WAV coding mode.
 */
export class WavCodingMode {
  /** 1.0 (mono) */
  public static readonly CODING_MODE_1_0 = new WavCodingMode('CODING_MODE_1_0');
  /** 2.0 (stereo) */
  public static readonly CODING_MODE_2_0 = new WavCodingMode('CODING_MODE_2_0');
  /** 4.0 */
  public static readonly CODING_MODE_4_0 = new WavCodingMode('CODING_MODE_4_0');
  /** 8.0 */
  public static readonly CODING_MODE_8_0 = new WavCodingMode('CODING_MODE_8_0');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): WavCodingMode {
    return new WavCodingMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

// =============================================================================
// Props interfaces
// =============================================================================

/**
 * Properties for AAC codec settings.
 */
export interface AacSettingsProps {
  /**
   * The average bitrate.
   * @default Bitrate.kbps(192)
   */
  readonly bitrate?: Bitrate;
  /**
   * The AAC profile.
   * @default AacProfile.LC
   */
  readonly profile?: AacProfile;
  /**
   * The coding mode (mono, stereo, 5.1).
   * @default AacCodingMode.CODING_MODE_2_0
   */
  readonly codingMode?: AacCodingMode;
  /**
   * The rate control mode.
   * @default AacRateControlMode.CBR
   */
  readonly rateControlMode?: AacRateControlMode;
  /**
   * The sample rate.
   * @default AudioSampleRate.HZ_48000
   */
  readonly sampleRate?: AudioSampleRate;
  /**
   * Set to broadcasterMixedAd when the input contains pre-mixed main audio + AD (narration) as a stereo pair.
   * @default AacInputType.NORMAL
   */
  readonly inputType?: AacInputType;
  /**
   * Sets the LATM/LOAS AAC output for raw containers.
   * @default AacRawFormat.NONE
   */
  readonly rawFormat?: AacRawFormat;
  /**
   * The AAC specification (MPEG-4 or MPEG-2) used to encode the audio.
   *
   * Set to `AacSpec.MPEG2` to emit MPEG-2 AAC instead of MPEG-4 AAC for raw or MPEG-2 Transport
   * Stream containers.
   * @default AacSpec.MPEG4
   */
  readonly spec?: AacSpec;
  /**
   * The VBR quality level. Used only if rateControlMode is VBR.
   * @default - service default
   */
  readonly vbrQuality?: AacVbrQuality;
}

/**
 * Properties for AC3 codec settings.
 */
export interface Ac3SettingsProps {
  /**
   * The average bitrate.
   * @default - service default
   */
  readonly bitrate?: Bitrate;
  /**
   * The Dolby Digital coding mode.
   * @default Ac3CodingMode.CODING_MODE_2_0
   */
  readonly codingMode?: Ac3CodingMode;
  /**
   * The dialogue normalization level (1–31).
   * @default - service default
   */
  readonly dialNorm?: number;
  /**
   * Applies a 3 dB attenuation to the surround channels. Used only for the 3/2 coding mode.
   * @default - service default
   */
  readonly attenuationControl?: Ac3AttenuationControl;
  /**
   * Specifies the bitstream mode (bsmod) for the emitted AC-3 stream.
   * @default Ac3BitstreamMode.COMPLETE_MAIN
   */
  readonly bitstreamMode?: Ac3BitstreamMode;
  /**
   * If set to filmStandard, adds dynamic range compression signaling to the output bitstream.
   * @default - service default
   */
  readonly drcProfile?: Ac3DrcProfile;
  /**
   * When set to enabled, applies a 120Hz lowpass filter to the LFE channel prior to encoding.
   * Valid only in codingMode32Lfe mode.
   * @default Ac3LfeFilter.DISABLED
   */
  readonly lfeFilter?: Ac3LfeFilter;
  /**
   * When set to followInput, encoder metadata is sourced from the DD, DD+, or DolbyE decoder that supplies this audio data.
   * @default - service default
   */
  readonly metadataControl?: Ac3MetadataControl;
}

/**
 * Properties for EAC3 codec settings.
 */
export interface Eac3SettingsProps {
  /**
   * The average bitrate.
   * @default - service default
   */
  readonly bitrate?: Bitrate;
  /**
   * The Dolby Digital Plus coding mode.
   * @default Eac3CodingMode.CODING_MODE_3_2
   */
  readonly codingMode?: Eac3CodingMode;
  /**
   * The dialogue normalization level (1–31).
   * @default - service default
   */
  readonly dialNorm?: number;
  /**
   * When set to attenuate3Db, applies a 3 dB attenuation to the surround channels. Used only for the 3/2 coding mode.
   * @default Eac3AttenuationControl.NONE
   */
  readonly attenuationControl?: Eac3AttenuationControl;
  /**
   * Specifies the bitstream mode (bsmod) for the emitted E-AC-3 stream.
   * @default Eac3BitstreamMode.COMPLETE_MAIN
   */
  readonly bitstreamMode?: Eac3BitstreamMode;
  /**
   * When set to enabled, activates a DC highpass filter for all input channels.
   * @default - service default
   */
  readonly dcFilter?: Eac3DcFilter;
  /**
   * Sets the Dolby dynamic range compression profile.
   * @default - service default
   */
  readonly drcLine?: Eac3DrcLine;
  /**
   * Sets the profile for heavy Dolby dynamic range compression, ensuring that the instantaneous signal peaks do not exceed specified levels.
   * @default - service default
   */
  readonly drcRf?: Eac3DrcRf;
  /**
   * When encoding 3/2 audio, setting to lfe enables the LFE channel.
   * @default - service default
   */
  readonly lfeControl?: Eac3LfeControl;
  /**
   * When set to enabled, applies a 120Hz lowpass filter to the LFE channel prior to encoding.
   * Valid only with a codingMode32 coding mode.
   * @default - service default
   */
  readonly lfeFilter?: Eac3LfeFilter;
  /**
   * The Left only/Right only center mix level. Used only for the 3/2 coding mode.
   * @default - service default
   */
  readonly loRoCenterMixLevel?: number;
  /**
   * The Left only/Right only surround mix level. Used only for a 3/2 coding mode.
   * @default - service default
   */
  readonly loRoSurroundMixLevel?: number;
  /**
   * The Left total/Right total center mix level. Used only for a 3/2 coding mode.
   * @default - service default
   */
  readonly ltRtCenterMixLevel?: number;
  /**
   * The Left total/Right total surround mix level. Used only for the 3/2 coding mode.
   * @default - service default
   */
  readonly ltRtSurroundMixLevel?: number;
  /**
   * When set to followInput, encoder metadata is sourced from the DD, DD+, or DolbyE decoder that supplies this audio data.
   * @default - service default
   */
  readonly metadataControl?: Eac3MetadataControl;
  /**
   * When set to whenPossible, input DD+ audio will be passed through if it is present on the input.
   * @default - service default
   */
  readonly passthroughControl?: Eac3PassthroughControl;
  /**
   * When set to shift90Degrees, applies a 90-degree phase shift to the surround channels. Used only for a 3/2 coding mode.
   * @default - service default
   */
  readonly phaseControl?: Eac3PhaseControl;
  /**
   * A stereo downmix preference. Used only for the 3/2 coding mode.
   * @default - service default
   */
  readonly stereoDownmix?: Eac3StereoDownmix;
  /**
   * When encoding 3/2 audio, sets whether an extra center back surround channel is matrix encoded into the left and right surround channels.
   * @default - service default
   */
  readonly surroundExMode?: Eac3SurroundExMode;
  /**
   * When encoding 2/0 audio, sets whether Dolby Surround is matrix-encoded into the two channels.
   * @default - service default
   */
  readonly surroundMode?: Eac3SurroundMode;
}

/**
 * Properties for EAC3 Atmos codec settings.
 */
export interface Eac3AtmosSettingsProps {
  /**
   * The average bitrate.
   * @default - service default
   */
  readonly bitrate?: Bitrate;
  /**
   * The coding mode (e.g. CODING_MODE_5_1_4, CODING_MODE_7_1_4, CODING_MODE_9_1_6).
   * @default Eac3AtmosCodingMode.CODING_MODE_5_1_4
   */
  readonly codingMode?: Eac3AtmosCodingMode;
  /**
   * The dialogue normalization level (1–31).
   * @default - service default
   */
  readonly dialNorm?: number;
  /**
   * Sets the Dolby dynamic range compression line mode profile.
   * @default - service default
   */
  readonly drcLine?: Eac3AtmosDrcLine;
  /**
   * Sets the Dolby dynamic range compression RF mode profile.
   * @default - service default
   */
  readonly drcRf?: Eac3AtmosDrcRf;
  /**
   * Height channel trim level.
   * @default - service default
   */
  readonly heightTrim?: number;
  /**
   * Surround channel trim level.
   * @default - service default
   */
  readonly surroundTrim?: number;
}

/**
 * Properties for MP2 codec settings.
 */
export interface Mp2SettingsProps {
  /**
   * The average bitrate.
   * @default - service default
   */
  readonly bitrate?: Bitrate;
  /**
   * The MPEG2 Audio coding mode.
   * @default Mp2CodingMode.CODING_MODE_2_0
   */
  readonly codingMode?: Mp2CodingMode;
  /**
   * The sample rate.
   * @default AudioSampleRate.HZ_48000
   */
  readonly sampleRate?: AudioSampleRate;
}

/**
 * Properties for WAV codec settings.
 */
export interface WavSettingsProps {
  /**
   * The bit depth of the WAV output.
   * @default AudioBitDepth.DEPTH_16
   */
  readonly bitDepth?: AudioBitDepth;
  /**
   * The audio coding mode for the WAV audio.
   * @default WavCodingMode.CODING_MODE_2_0
   */
  readonly codingMode?: WavCodingMode;
  /**
   * The sample rate.
   * @default AudioSampleRate.HZ_48000
   */
  readonly sampleRate?: AudioSampleRate;
}

// =============================================================================
// Abstract class and implementations
// =============================================================================

/**
 * Audio codec settings. Use the static factory methods to create.
 */
export abstract class AudioCodecSettings {
  /**
   * Create AAC codec settings.
   */
  public static aac(props?: AacSettingsProps): AudioCodecSettings {
    return new AacAudioCodecSettings(props ?? {});
  }

  /**
   * Create AC3 codec settings.
   */
  public static ac3(props?: Ac3SettingsProps): AudioCodecSettings {
    return new Ac3AudioCodecSettings(props ?? {});
  }

  /**
   * Create EAC3 (Dolby Digital Plus) codec settings.
   */
  public static eac3(props?: Eac3SettingsProps): AudioCodecSettings {
    return new Eac3AudioCodecSettings(props ?? {});
  }

  /**
   * Create EAC3 Atmos (Dolby Digital Plus with Atmos) codec settings.
   */
  public static eac3Atmos(props?: Eac3AtmosSettingsProps): AudioCodecSettings {
    return new Eac3AtmosAudioCodecSettings(props ?? {});
  }

  /**
   * Create MP2 codec settings.
   */
  public static mp2(props?: Mp2SettingsProps): AudioCodecSettings {
    return new Mp2AudioCodecSettings(props ?? {});
  }

  /**
   * Create WAV codec settings.
   */
  public static wav(props?: WavSettingsProps): AudioCodecSettings {
    return new WavAudioCodecSettings(props ?? {});
  }

  /**
   * Create passthrough audio settings (no transcoding).
   */
  public static passthrough(): AudioCodecSettings {
    return new PassthroughAudioCodecSettings();
  }

  /** @internal */
  public abstract readonly _codecType: AudioCodecType;
  /** @internal */
  public abstract _bind(): CfnChannel.AudioCodecSettingsProperty;
}

/** @internal */
class AacAudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.AAC;
  constructor(private readonly props: AacSettingsProps) {
    super();
  }

  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    const p = this.props;
    return {
      aacSettings: {
        bitrate: p.bitrate?.toBps() ?? 192000,
        profile: (p.profile ?? AacProfile.LC).value,
        codingMode: (p.codingMode ?? AacCodingMode.CODING_MODE_2_0).value,
        rateControlMode: (p.rateControlMode ?? AacRateControlMode.CBR).value,
        sampleRate: (p.sampleRate ?? AudioSampleRate.HZ_48000)._toHz(),
        rawFormat: (p.rawFormat ?? AacRawFormat.NONE).value,
        spec: (p.spec ?? AacSpec.MPEG4).value,
        inputType: (p.inputType ?? AacInputType.NORMAL).value,
        vbrQuality: p.vbrQuality?.value,
      },
    };
  }
}

/** @internal */
class Ac3AudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.AC3;
  constructor(private readonly props: Ac3SettingsProps) {
    super();
  }

  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    const p = this.props;
    return {
      ac3Settings: {
        bitrate: p.bitrate?.toBps(),
        codingMode: (p.codingMode ?? Ac3CodingMode.CODING_MODE_2_0).value,
        dialnorm: p.dialNorm,
        attenuationControl: p.attenuationControl?.value,
        bitstreamMode: (p.bitstreamMode ?? Ac3BitstreamMode.COMPLETE_MAIN).value,
        drcProfile: p.drcProfile?.value,
        lfeFilter: (p.lfeFilter ?? Ac3LfeFilter.DISABLED).value,
        metadataControl: p.metadataControl?.value,
      },
    };
  }
}

/** @internal */
class Eac3AudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.EAC3;
  constructor(private readonly props: Eac3SettingsProps) {
    super();
  }

  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    const p = this.props;
    return {
      eac3Settings: {
        bitrate: p.bitrate?.toBps(),
        codingMode: (p.codingMode ?? Eac3CodingMode.CODING_MODE_3_2).value,
        dialnorm: p.dialNorm,
        attenuationControl: (p.attenuationControl ?? Eac3AttenuationControl.NONE).value,
        bitstreamMode: (p.bitstreamMode ?? Eac3BitstreamMode.COMPLETE_MAIN).value,
        dcFilter: p.dcFilter?.value,
        drcLine: p.drcLine?.value,
        drcRf: p.drcRf?.value,
        lfeControl: p.lfeControl?.value,
        lfeFilter: p.lfeFilter?.value,
        loRoCenterMixLevel: p.loRoCenterMixLevel,
        loRoSurroundMixLevel: p.loRoSurroundMixLevel,
        ltRtCenterMixLevel: p.ltRtCenterMixLevel,
        ltRtSurroundMixLevel: p.ltRtSurroundMixLevel,
        metadataControl: p.metadataControl?.value,
        passthroughControl: p.passthroughControl?.value,
        phaseControl: p.phaseControl?.value,
        stereoDownmix: p.stereoDownmix?.value,
        surroundExMode: p.surroundExMode?.value,
        surroundMode: p.surroundMode?.value,
      },
    };
  }
}

/** @internal */
class Eac3AtmosAudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.EAC3_ATMOS;
  constructor(private readonly props: Eac3AtmosSettingsProps) {
    super();
  }

  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    const p = this.props;
    return {
      eac3AtmosSettings: {
        bitrate: p.bitrate?.toBps(),
        codingMode: (p.codingMode ?? Eac3AtmosCodingMode.CODING_MODE_5_1_4).value,
        dialnorm: p.dialNorm,
        drcLine: p.drcLine?.value,
        drcRf: p.drcRf?.value,
        heightTrim: p.heightTrim,
        surroundTrim: p.surroundTrim,
      },
    };
  }
}

/** @internal */
class Mp2AudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.MP2;
  constructor(private readonly props: Mp2SettingsProps) {
    super();
  }

  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    const p = this.props;
    return {
      mp2Settings: {
        bitrate: p.bitrate?.toBps(),
        codingMode: (p.codingMode ?? Mp2CodingMode.CODING_MODE_2_0).value,
        sampleRate: (p.sampleRate ?? AudioSampleRate.HZ_48000)._toHz(),
      },
    };
  }
}

/** @internal */
class WavAudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.WAV;
  constructor(private readonly props: WavSettingsProps) {
    super();
  }

  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    const p = this.props;
    return {
      wavSettings: {
        bitDepth: (p.bitDepth ?? AudioBitDepth.DEPTH_16)._toBits(),
        codingMode: (p.codingMode ?? WavCodingMode.CODING_MODE_2_0).value,
        sampleRate: (p.sampleRate ?? AudioSampleRate.HZ_48000)._toHz(),
      },
    };
  }
}

/** @internal */
class PassthroughAudioCodecSettings extends AudioCodecSettings {
  public readonly _codecType = AudioCodecType.PASSTHROUGH;
  public _bind(): CfnChannel.AudioCodecSettingsProperty {
    return {
      passThroughSettings: {},
    };
  }
}
