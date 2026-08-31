import { Token, UnscopedValidationError } from 'aws-cdk-lib';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';

/**
 * The OCR language to use when converting an image-based caption source to text.
 */
export class OcrLanguage {
  /** German */
  public static readonly DEU = new OcrLanguage('DEU');
  /** English */
  public static readonly ENG = new OcrLanguage('ENG');
  /** French */
  public static readonly FRA = new OcrLanguage('FRA');
  /** Dutch */
  public static readonly NLD = new OcrLanguage('NLD');
  /** Portuguese */
  public static readonly POR = new OcrLanguage('POR');
  /** Spanish */
  public static readonly SPA = new OcrLanguage('SPA');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): OcrLanguage {
    return new OcrLanguage(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Whether to upconvert 608 captions to 708.
 */
export class Convert608To708 {
  /** Pass through 608 captions without upconverting. */
  public static readonly DISABLED = new Convert608To708('DISABLED');
  /** Upconvert 608 captions to 708 (608 data is also passed through). */
  public static readonly UPCONVERT = new Convert608To708('UPCONVERT');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Convert608To708 {
    return new Convert608To708(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * SCTE-20 detection mode for an embedded caption source.
 */
export class Scte20Detection {
  /** Handle streams with intermittent or non-aligned SCTE-20 and embedded captions. */
  public static readonly AUTO = new Scte20Detection('AUTO');
  /** Do not detect SCTE-20 captions. */
  public static readonly OFF = new Scte20Detection('OFF');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): Scte20Detection {
    return new Scte20Detection(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Options for an embedded (CEA-608/708) caption source.
 */
export interface EmbeddedCaptionSourceOptions {
  /**
   * Whether to upconvert 608 captions to 708.
   * @default Convert608To708.DISABLED
   */
  readonly convert608To708?: Convert608To708;
  /**
   * SCTE-20 detection mode.
   * @default Scte20Detection.OFF
   */
  readonly scte20Detection?: Scte20Detection;
  /**
   * The 608/708 channel number within the video track to extract captions from.
   * @default - MediaLive service default
   */
  readonly source608ChannelNumber?: number;
}

/**
 * Options for an ancillary caption source.
 */
export interface AncillaryCaptionSourceOptions {
  /**
   * The captions channel (1-4) to extract from the ancillary captions. Required when
   * converting to another caption format; ignored when passing through as embedded.
   * @default - MediaLive ignores the channel (passthrough)
   */
  readonly sourceChannelNumber?: number;
}

/**
 * Options for a DVB-Sub caption source.
 */
export interface DvbSubCaptionSourceOptions {
  /**
   * The PID of the source content. Unused for DVB-Sub passthrough.
   * @default - MediaLive service default
   */
  readonly pid?: number;
  /**
   * The OCR language to use when converting this image-based source to text.
   * @default - MediaLive service default
   */
  readonly ocrLanguage?: OcrLanguage;
}

/**
 * Options for an SCTE-20 caption source.
 */
export interface Scte20CaptionSourceOptions {
  /**
   * Whether to upconvert 608 captions to 708.
   * @default Convert608To708.DISABLED
   */
  readonly convert608To708?: Convert608To708;
  /**
   * The 608/708 channel number within the video track to extract captions from.
   * @default - MediaLive service default
   */
  readonly source608ChannelNumber?: number;
}

/**
 * Options for an SCTE-27 caption source.
 */
export interface Scte27CaptionSourceOptions {
  /**
   * The PID to extract captions from. See the MediaLive docs for how PID and `languageCode`
   * interact.
   * @default - MediaLive service default
   */
  readonly pid?: number;
  /**
   * The OCR language to use when converting this image-based source to text.
   * @default - MediaLive service default
   */
  readonly ocrLanguage?: OcrLanguage;
}

/**
 * A display rectangle, expressed as percentages of the underlying video frame, for captions
 * converted to EBU-TT-D or TTML.
 */
export interface CaptionRectangle {
  /** Height of the rectangle, as a percentage of the frame height (0–100). */
  readonly height: number;
  /** Width of the rectangle, as a percentage of the frame width (0–100). */
  readonly width: number;
  /** Left edge position, as a percentage of the frame width (0–100). */
  readonly leftOffset: number;
  /** Top edge position, as a percentage of the frame height (0–100). */
  readonly topOffset: number;
}

/**
 * Options for a Teletext caption source.
 */
export interface TeletextCaptionSourceOptions {
  /**
   * The Teletext page number to extract captions from, as a hexadecimal string with no
   * `0x` prefix (range `100`–`8FF`). Unused for passthrough.
   * @default - MediaLive service default
   */
  readonly pageNumber?: string;
  /**
   * The caption rectangle to use when converting this source to EBU-TT-D or TTML.
   * @default - no rectangle (service default)
   */
  readonly outputRectangle?: CaptionRectangle;
}

/**
 * Controls whether MediaLive delays video to synchronize captions with audio and video output.
 */
export class CaptionSynchronizationMode {
  /** MediaLive does not delay video for caption alignment. */
  public static readonly NO_VIDEO_DELAY = new CaptionSynchronizationMode('NO_VIDEO_DELAY');
  /** MediaLive delays video to ensure captions are synchronized with audio and video. */
  public static readonly VIDEO_ALIGNED_CAPTIONS = new CaptionSynchronizationMode('VIDEO_ALIGNED_CAPTIONS');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionSynchronizationMode {
    return new CaptionSynchronizationMode(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Options for a smart subtitle caption source (AI-generated subtitles via Elemental Inference).
 */
export interface SmartSubtitleSourceOptions {
  /**
   * The name of the Elemental Inference feed output that provides the subtitles.
   * @default - service default
   */
  readonly inferenceFeedOutput?: string;
  /**
   * Controls whether MediaLive delays video to synchronize captions with audio and video output.
   * @default - service default
   */
  readonly captionSynchronizationMode?: CaptionSynchronizationMode;
}

/**
 * A caption selector that identifies which captions to extract from the input. Create with
 * the static factory methods — one per caption source format.
 */
export abstract class CaptionSelector {
  /**
   * Select captions by language code (no specific source format).
   */
  public static byLanguage(name: string, languageCode: string): CaptionSelector {
    return new LanguageCaptionSelector(name, languageCode);
  }

  /**
   * Select embedded (CEA-608/708) captions.
   */
  public static embedded(name: string, options: EmbeddedCaptionSourceOptions = {}): CaptionSelector {
    return new EmbeddedCaptionSelector(name, options);
  }

  /**
   * Select ancillary captions.
   */
  public static ancillary(name: string, options: AncillaryCaptionSourceOptions = {}): CaptionSelector {
    return new AncillaryCaptionSelector(name, options);
  }

  /**
   * Select ARIB captions.
   */
  public static arib(name: string): CaptionSelector {
    return new AribCaptionSelector(name);
  }

  /**
   * Select DVB-Sub (image-based) captions.
   */
  public static dvbSub(name: string, options: DvbSubCaptionSourceOptions = {}): CaptionSelector {
    return new DvbSubCaptionSelector(name, options);
  }

  /**
   * Select SCTE-20 captions.
   */
  public static scte20(name: string, options: Scte20CaptionSourceOptions = {}): CaptionSelector {
    return new Scte20CaptionSelector(name, options);
  }

  /**
   * Select SCTE-27 (image-based) captions.
   */
  public static scte27(name: string, options: Scte27CaptionSourceOptions = {}): CaptionSelector {
    return new Scte27CaptionSelector(name, options);
  }

  /**
   * Select Teletext captions.
   */
  public static teletext(name: string, options: TeletextCaptionSourceOptions = {}): CaptionSelector {
    return new TeletextCaptionSelector(name, options);
  }

  /**
   * Select smart subtitles generated by Elemental Inference.
   */
  public static smartSubtitle(name: string, options: SmartSubtitleSourceOptions = {}): CaptionSelector {
    return new SmartSubtitleCaptionSelector(name, options);
  }

  /**
   * The name of this caption selector, used to associate it with caption outputs. Unique
   * within a channel.
   */
  public readonly name: string;

  protected constructor(name: string) {
    this.name = name;
  }

  /** @internal */
  public abstract _bind(): CfnChannel.CaptionSelectorProperty;
}

/** @internal */
class LanguageCaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly languageCode: string) { super(name); }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return { name: this.name, languageCode: this.languageCode };
  }
}

/** @internal */
class EmbeddedCaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: EmbeddedCaptionSourceOptions) {
    super(name);
    if (options.source608ChannelNumber !== undefined && !Token.isUnresolved(options.source608ChannelNumber)
      && (options.source608ChannelNumber < 1 || options.source608ChannelNumber > 4)) {
      throw new UnscopedValidationError(lit`Source608ChannelNumberRange`, `source608ChannelNumber must be between 1 and 4, got ${options.source608ChannelNumber}`);
    }
  }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    const hasSettings = this.options.convert608To708 !== undefined
      || this.options.scte20Detection !== undefined
      || this.options.source608ChannelNumber !== undefined;
    return {
      name: this.name,
      selectorSettings: hasSettings ? {
        embeddedSourceSettings: {
          convert608To708: (this.options.convert608To708 ?? Convert608To708.DISABLED).value,
          scte20Detection: (this.options.scte20Detection ?? Scte20Detection.OFF).value,
          source608ChannelNumber: this.options.source608ChannelNumber,
        },
      } : undefined,
    };
  }
}

/** @internal */
class AncillaryCaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: AncillaryCaptionSourceOptions) {
    super(name);
    if (options.sourceChannelNumber !== undefined && !Token.isUnresolved(options.sourceChannelNumber)
      && (options.sourceChannelNumber < 1 || options.sourceChannelNumber > 4)) {
      throw new UnscopedValidationError(lit`AncillaryChannelNumberRange`, `sourceChannelNumber must be between 1 and 4, got ${options.sourceChannelNumber}`);
    }
  }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        ancillarySourceSettings: { sourceAncillaryChannelNumber: this.options.sourceChannelNumber },
      },
    };
  }
}

/** @internal */
class AribCaptionSelector extends CaptionSelector {
  constructor(name: string) { super(name); }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return { name: this.name, selectorSettings: { aribSourceSettings: {} } };
  }
}

/** @internal */
class DvbSubCaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: DvbSubCaptionSourceOptions) { super(name); }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        dvbSubSourceSettings: { pid: this.options.pid, ocrLanguage: this.options.ocrLanguage?.value },
      },
    };
  }
}

/** @internal */
class Scte20CaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: Scte20CaptionSourceOptions) {
    super(name);
    if (options.source608ChannelNumber !== undefined && !Token.isUnresolved(options.source608ChannelNumber)
      && (options.source608ChannelNumber < 1 || options.source608ChannelNumber > 4)) {
      throw new UnscopedValidationError(lit`Source608ChannelNumberRange`, `source608ChannelNumber must be between 1 and 4, got ${options.source608ChannelNumber}`);
    }
  }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        scte20SourceSettings: {
          convert608To708: (this.options.convert608To708 ?? Convert608To708.DISABLED).value,
          source608ChannelNumber: this.options.source608ChannelNumber,
        },
      },
    };
  }
}

/** @internal */
class Scte27CaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: Scte27CaptionSourceOptions) { super(name); }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        scte27SourceSettings: { pid: this.options.pid, ocrLanguage: this.options.ocrLanguage?.value },
      },
    };
  }
}

/** @internal */
class TeletextCaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: TeletextCaptionSourceOptions) {
    super(name);
    if (options.outputRectangle) {
      const r = options.outputRectangle;
      if (!Token.isUnresolved(r.height) && (r.height < 0 || r.height > 100)) {
        throw new UnscopedValidationError(lit`CaptionRectangleRange`, `outputRectangle.height must be between 0 and 100, got ${r.height}`);
      }
      if (!Token.isUnresolved(r.width) && (r.width < 0 || r.width > 100)) {
        throw new UnscopedValidationError(lit`CaptionRectangleRange`, `outputRectangle.width must be between 0 and 100, got ${r.width}`);
      }
      if (!Token.isUnresolved(r.leftOffset) && (r.leftOffset < 0 || r.leftOffset > 100)) {
        throw new UnscopedValidationError(lit`CaptionRectangleRange`, `outputRectangle.leftOffset must be between 0 and 100, got ${r.leftOffset}`);
      }
      if (!Token.isUnresolved(r.topOffset) && (r.topOffset < 0 || r.topOffset > 100)) {
        throw new UnscopedValidationError(lit`CaptionRectangleRange`, `outputRectangle.topOffset must be between 0 and 100, got ${r.topOffset}`);
      }
      if (!Token.isUnresolved(r.leftOffset) && !Token.isUnresolved(r.width) && r.leftOffset + r.width > 100) {
        throw new UnscopedValidationError(lit`CaptionRectangleOverflow`, `outputRectangle leftOffset (${r.leftOffset}) + width (${r.width}) must not exceed 100`);
      }
      if (!Token.isUnresolved(r.topOffset) && !Token.isUnresolved(r.height) && r.topOffset + r.height > 100) {
        throw new UnscopedValidationError(lit`CaptionRectangleOverflow`, `outputRectangle topOffset (${r.topOffset}) + height (${r.height}) must not exceed 100`);
      }
    }
  }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        teletextSourceSettings: {
          pageNumber: this.options.pageNumber,
          outputRectangle: this.options.outputRectangle ? {
            height: this.options.outputRectangle.height,
            width: this.options.outputRectangle.width,
            leftOffset: this.options.outputRectangle.leftOffset,
            topOffset: this.options.outputRectangle.topOffset,
          } : undefined,
        },
      },
    };
  }
}

/** @internal */
class SmartSubtitleCaptionSelector extends CaptionSelector {
  constructor(name: string, private readonly options: SmartSubtitleSourceOptions) { super(name); }
  public _bind(): CfnChannel.CaptionSelectorProperty {
    return {
      name: this.name,
      selectorSettings: {
        smartSubtitleSourceSettings: {
          inferenceFeedOutput: this.options.inferenceFeedOutput,
          captionSynchronizationMode: this.options.captionSynchronizationMode?.value,
        },
      },
    };
  }
}
