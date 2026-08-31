import { UnscopedValidationError } from 'aws-cdk-lib';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type { AudioNormalizationSettings, RemixSettings } from './encode-configuration';

/**
 * Policy for how MediaLive identifies the audio stream when selecting by language, on a
 * transport-stream PMT update.
 */
export class AudioLanguageSelectionPolicy {
  /**
   * Strictly identify audio by its language descriptor. If the matching language is no
   * longer present after a PMT update, mute is encoded until the language returns.
   */
  public static readonly STRICT = new AudioLanguageSelectionPolicy('STRICT');
  /**
   * On a PMT update, fall back to another audio stream of the same type in the program if a
   * stream with the same language can't be found.
   */
  public static readonly LOOSE = new AudioLanguageSelectionPolicy('LOOSE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): AudioLanguageSelectionPolicy {
    return new AudioLanguageSelectionPolicy(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Options for selecting an HLS audio rendition.
 */
export interface HlsRenditionSelectionOptions {
  /** The `GROUP-ID` in the `#EXT-X-MEDIA` tag of the target HLS audio rendition. */
  readonly groupId: string;
  /** The `NAME` in the `#EXT-X-MEDIA` tag of the target HLS audio rendition. */
  readonly renditionName: string;
}

/**
 * Which Dolby E program to decode from a selected audio track.
 */
export class DolbyEProgramSelection {
  /** Decode all channels. */
  public static readonly ALL_CHANNELS = new DolbyEProgramSelection('ALL_CHANNELS');
  /** Decode Dolby E program 1. */
  public static readonly PROGRAM_1 = new DolbyEProgramSelection('PROGRAM_1');
  /** Decode Dolby E program 2. */
  public static readonly PROGRAM_2 = new DolbyEProgramSelection('PROGRAM_2');
  /** Decode Dolby E program 3. */
  public static readonly PROGRAM_3 = new DolbyEProgramSelection('PROGRAM_3');
  /** Decode Dolby E program 4. */
  public static readonly PROGRAM_4 = new DolbyEProgramSelection('PROGRAM_4');
  /** Decode Dolby E program 5. */
  public static readonly PROGRAM_5 = new DolbyEProgramSelection('PROGRAM_5');
  /** Decode Dolby E program 6. */
  public static readonly PROGRAM_6 = new DolbyEProgramSelection('PROGRAM_6');
  /** Decode Dolby E program 7. */
  public static readonly PROGRAM_7 = new DolbyEProgramSelection('PROGRAM_7');
  /** Decode Dolby E program 8. */
  public static readonly PROGRAM_8 = new DolbyEProgramSelection('PROGRAM_8');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): DolbyEProgramSelection {
    return new DolbyEProgramSelection(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Configuration for a single audio PID in a PID-based selector.
 */
export interface AudioPidConfig {
  /** The packet identifier (PID) value from within the source. */
  readonly pid: number;
  /**
   * Which Dolby E program to decode from this PID.
   * @default - no Dolby E decoding
   */
  readonly dolbyEDecode?: DolbyEProgramSelection;
  /**
   * Pre-mixer settings for this PID (gain, channel remix, loudness normalization).
   * @default - no pre-mixing
   */
  readonly premixSettings?: AudioPreMixerSettings;
}

/**
 * Configuration for a single audio track in a track-based selector.
 */
export interface AudioTrackConfig {
  /** The 1-based track number to extract. */
  readonly track: number;
  /**
   * Pre-mixer settings for this track (gain, channel remix, loudness normalization).
   * @default - no pre-mixing
   */
  readonly premixSettings?: AudioPreMixerSettings;
}

/**
 * Audio pre-mixer settings for normalizing audio before interleaving.
 * Applied per-PID or per-track before tracks are combined.
 */
export class AudioPreMixerSettings {
  /**
   * Create pre-mixer settings.
   */
  public static of(props: AudioPreMixerSettingsProps): AudioPreMixerSettings {
    return new AudioPreMixerSettings(props);
  }

  private constructor(private readonly props: AudioPreMixerSettingsProps) {}

  /** @internal */
  public _bind(): CfnChannel.AudioPreMixerSettingsProperty {
    return {
      gainDb: this.props.gainDb,
      channels: this.props.channels,
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
      audioNormalizationSettings: this.props.audioNormalizationSettings ? {
        algorithm: this.props.audioNormalizationSettings.algorithm?.value,
        algorithmControl: this.props.audioNormalizationSettings.algorithmControl?.value,
        targetLkfs: this.props.audioNormalizationSettings.targetLkfs,
        peakCalculation: this.props.audioNormalizationSettings.peakCalculation?.value,
        peakLimiterThreshold: this.props.audioNormalizationSettings.peakLimiterThreshold,
      } : undefined,
    };
  }
}

/**
 * Properties for audio pre-mixer settings.
 */
export interface AudioPreMixerSettingsProps {
  /**
   * The gain adjustment in decibels (dB).
   * @default - no gain adjustment
   */
  readonly gainDb?: number;
  /**
   * The number of audio channels to remix to. Overridden by `remixSettings` if specified.
   * @default - pass through original channel count
   */
  readonly channels?: number;
  /**
   * Remix settings for fine-grained channel mapping and gain levels.
   * @default - no remixing
   */
  readonly remixSettings?: RemixSettings;
  /**
   * Audio normalization settings for loudness control.
   * @default - no normalization
   */
  readonly audioNormalizationSettings?: AudioNormalizationSettings;
}

/**
 * An audio selector that identifies which audio to extract from the input.
 */
export abstract class AudioSelector {
  /**
   * Select audio by language code.
   *
   * @param name a name for this selector
   * @param languageCode the three-letter (ISO 639-2) language code to select
   * @param languageSelectionPolicy how strictly to identify the stream by language on a PMT update
   */
  public static byLanguage(name: string, languageCode: string, languageSelectionPolicy?: AudioLanguageSelectionPolicy): AudioSelector {
    return new LanguageAudioSelector(name, languageCode, languageSelectionPolicy);
  }

  /**
   * Select one or more audio PIDs from the source.
   *
   * @param name a name for this selector
   * @param pids the PID configurations (each with optional Dolby E decode and pre-mix settings)
   */
  public static byPid(name: string, pids: AudioPidConfig[]): AudioSelector {
    return new PidAudioSelector(name, pids);
  }

  /**
   * Select one or more audio tracks (1-based) from the source.
   *
   * @param name a name for this selector
   * @param tracks the track configurations (each with optional pre-mix settings)
   * @param dolbyEProgramSelection which Dolby E program to decode from the selected track(s)
   */
  public static byTrack(name: string, tracks: AudioTrackConfig[], dolbyEProgramSelection?: DolbyEProgramSelection): AudioSelector {
    return new TrackAudioSelector(name, tracks, dolbyEProgramSelection);
  }

  /**
   * Select an HLS audio rendition by its `#EXT-X-MEDIA` `GROUP-ID` and `NAME`.
   *
   * @param name a name for this selector
   * @param options the target rendition's group id and name
   */
  public static hlsRendition(name: string, options: HlsRenditionSelectionOptions): AudioSelector {
    return new HlsRenditionAudioSelector(name, options);
  }

  /**
   * Select the default audio track (no specific selector settings).
   */
  public static default(name: string): AudioSelector {
    return new DefaultAudioSelector(name);
  }

  /**
   * The name of this audio selector. Reference it from features that monitor a specific
   * selector, such as an audio-silence {@link FailoverCondition}.
   */
  public readonly name: string;

  protected constructor(name: string) {
    this.name = name;
  }

  /** @internal */
  public abstract _bind(): CfnChannel.AudioSelectorProperty;
}

/** @internal */
class LanguageAudioSelector extends AudioSelector {
  constructor(name: string, private readonly languageCode: string, private readonly policy?: AudioLanguageSelectionPolicy) { super(name); }
  public _bind(): CfnChannel.AudioSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        audioLanguageSelection: {
          languageCode: this.languageCode,
          languageSelectionPolicy: this.policy?.value,
        },
      },
    };
  }
}

/** @internal */
class PidAudioSelector extends AudioSelector {
  constructor(name: string, private readonly pids: AudioPidConfig[]) {
    super(name);
    if (pids.length === 0) {
      throw new UnscopedValidationError(lit`AudioPidSelectorEmpty`, 'byPid() requires at least one PID configuration');
    }
  }
  public _bind(): CfnChannel.AudioSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        audioPidSelection: {
          pids: this.pids.map(p => ({
            pid: p.pid,
            dolbyEDecode: p.dolbyEDecode ? { programSelection: p.dolbyEDecode.value } : undefined,
            premixSettings: p.premixSettings?._bind(),
          })),
        },
      },
    };
  }
}

/** @internal */
class TrackAudioSelector extends AudioSelector {
  constructor(name: string, private readonly tracks: AudioTrackConfig[],
    private readonly dolbyEProgramSelection?: DolbyEProgramSelection) {
    super(name);
    if (tracks.length === 0) {
      throw new UnscopedValidationError(lit`AudioTrackSelectorEmpty`, 'byTrack() requires at least one track configuration');
    }
  }
  public _bind(): CfnChannel.AudioSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        audioTrackSelection: {
          tracks: this.tracks.map(t => ({
            track: t.track,
            premixSettings: t.premixSettings?._bind(),
          })),
          dolbyEDecode: this.dolbyEProgramSelection ? { programSelection: this.dolbyEProgramSelection.value } : undefined,
        },
      },
    };
  }
}

/** @internal */
class HlsRenditionAudioSelector extends AudioSelector {
  constructor(name: string, private readonly options: HlsRenditionSelectionOptions) { super(name); }
  public _bind(): CfnChannel.AudioSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        audioHlsRenditionSelection: {
          groupId: this.options.groupId,
          name: this.options.renditionName,
        },
      },
    };
  }
}

/** @internal */
class DefaultAudioSelector extends AudioSelector {
  constructor(name: string) { super(name); }
  public _bind(): CfnChannel.AudioSelectorProperty {
    return { name: this.name };
  }
}
