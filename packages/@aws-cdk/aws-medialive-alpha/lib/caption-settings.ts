import { Token, Tokenization, UnscopedValidationError } from 'aws-cdk-lib';
import type { IRole } from 'aws-cdk-lib/aws-iam';
import type { CfnChannel } from 'aws-cdk-lib/aws-medialive';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type { FileLocation } from './file-location';

/** Caption alignment for burn-in and DVB-Sub outputs. */
export class CaptionAlignment {
  /** Center the captions. */
  public static readonly CENTERED = new CaptionAlignment('CENTERED');
  /** Left-align the captions. */
  public static readonly LEFT = new CaptionAlignment('LEFT');
  /** Smart: left-justify live subtitles, center-justify pre-recorded subtitles. */
  public static readonly SMART = new CaptionAlignment('SMART');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionAlignment {
    return new CaptionAlignment(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Font color for burn-in and DVB-Sub captions. */
export class CaptionFontColor {
  /** Black. */
  public static readonly BLACK = new CaptionFontColor('BLACK');
  /** Blue. */
  public static readonly BLUE = new CaptionFontColor('BLUE');
  /** Green. */
  public static readonly GREEN = new CaptionFontColor('GREEN');
  /** Red. */
  public static readonly RED = new CaptionFontColor('RED');
  /** White. */
  public static readonly WHITE = new CaptionFontColor('WHITE');
  /** Yellow. */
  public static readonly YELLOW = new CaptionFontColor('YELLOW');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionFontColor {
    return new CaptionFontColor(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Font outline color for burn-in and DVB-Sub captions. */
export class CaptionOutlineColor {
  /** Black. */
  public static readonly BLACK = new CaptionOutlineColor('BLACK');
  /** Blue. */
  public static readonly BLUE = new CaptionOutlineColor('BLUE');
  /** Green. */
  public static readonly GREEN = new CaptionOutlineColor('GREEN');
  /** Red. */
  public static readonly RED = new CaptionOutlineColor('RED');
  /** White. */
  public static readonly WHITE = new CaptionOutlineColor('WHITE');
  /** Yellow. */
  public static readonly YELLOW = new CaptionOutlineColor('YELLOW');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionOutlineColor {
    return new CaptionOutlineColor(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Background color for burn-in and DVB-Sub captions. */
export class CaptionBackgroundColor {
  /** Black. */
  public static readonly BLACK = new CaptionBackgroundColor('BLACK');
  /** None (transparent). */
  public static readonly NONE = new CaptionBackgroundColor('NONE');
  /** White. */
  public static readonly WHITE = new CaptionBackgroundColor('WHITE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionBackgroundColor {
    return new CaptionBackgroundColor(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Shadow color for burn-in and DVB-Sub captions. */
export class CaptionShadowColor {
  /** Black. */
  public static readonly BLACK = new CaptionShadowColor('BLACK');
  /** None (transparent). */
  public static readonly NONE = new CaptionShadowColor('NONE');
  /** White. */
  public static readonly WHITE = new CaptionShadowColor('WHITE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionShadowColor {
    return new CaptionShadowColor(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Controls whether a fixed grid is used to generate the subtitle bitmap (Teletext input). */
export class CaptionTeletextGridControl {
  /** Fixed grid. */
  public static readonly FIXED = new CaptionTeletextGridControl('FIXED');
  /** Scaled grid. */
  public static readonly SCALED = new CaptionTeletextGridControl('SCALED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): CaptionTeletextGridControl {
    return new CaptionTeletextGridControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether EBU-TT-D fills the gap between multi-line captions. */
export class EbuTtDFillLineGap {
  /** Leave the gap unfilled. */
  public static readonly DISABLED = new EbuTtDFillLineGap('DISABLED');
  /** Fill with the captions background color. */
  public static readonly ENABLED = new EbuTtDFillLineGap('ENABLED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): EbuTtDFillLineGap {
    return new EbuTtDFillLineGap(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether EBU-TT-D includes source style information. */
export class EbuTtDStyleControl {
  /** Set the font family to monospaced and exclude other style info. */
  public static readonly EXCLUDE = new EbuTtDStyleControl('EXCLUDE');
  /** Include source style (color, position) in the font data. */
  public static readonly INCLUDE = new EbuTtDStyleControl('INCLUDE');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): EbuTtDStyleControl {
    return new EbuTtDStyleControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether TTML passes through source style/position. */
export class TtmlStyleControl {
  /** Pass through style and position from a TTML-like source. */
  public static readonly PASSTHROUGH = new TtmlStyleControl('PASSTHROUGH');
  /** Use the configured style. */
  public static readonly USE_CONFIGURED = new TtmlStyleControl('USE_CONFIGURED');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): TtmlStyleControl {
    return new TtmlStyleControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/** Whether WebVTT passes through source style/position. */
export class WebvttStyleControl {
  /** Do not pass through style; output contains no font styling. */
  public static readonly NO_STYLE_DATA = new WebvttStyleControl('NO_STYLE_DATA');
  /** Pass through style (valid only for EMBEDDED or TELETEXT sources). */
  public static readonly PASSTHROUGH = new WebvttStyleControl('PASSTHROUGH');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): WebvttStyleControl {
    return new WebvttStyleControl(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * Font size for burn-in and DVB-Sub captions.
 *
 * Use {@link CaptionFontSize.AUTO} to scale the font size with the output resolution, or
 * {@link CaptionFontSize.of} to set an exact size in points.
 */
export class CaptionFontSize {
  /** Scale the font size automatically to match the output resolution. */
  public static readonly AUTO = new CaptionFontSize('auto');

  /**
   * An exact font size, in points.
   *
   * @param points the font size in points; must be a positive integer
   */
  public static of(points: number): CaptionFontSize {
    if (!Token.isUnresolved(points) && (!Number.isInteger(points) || points <= 0)) {
      throw new UnscopedValidationError(lit`CaptionFontSizeInvalid`, `caption font size must be a positive integer, got ${JSON.stringify(points)}`);
    }
    return new CaptionFontSize(Tokenization.stringifyNumber(points));
  }

  private constructor(private readonly value: string) {}

  /** @internal */
  public _bind(): string {
    return this.value;
  }
}

/**
 * Font and positioning settings for a rendered caption output (burn-in or DVB-Sub).
 */
export interface CaptionFontStyleProps {
  /**
   * Caption alignment. With explicit x/y positions, the font is justified relative to them.
   * @default CaptionAlignment.CENTERED
   */
  readonly alignment?: CaptionAlignment;
  /**
   * The color of the rectangle behind the captions.
   * @default - service default
   */
  readonly backgroundColor?: CaptionBackgroundColor;
  /**
   * The opacity of the background rectangle (0 transparent .. 255 opaque).
   * @default - service default
   */
  readonly backgroundOpacity?: number;
  /**
   * An external font file (.ttf or .tte) used for burn-in. Provide a `FileLocation` referencing
   * an S3 bucket (`FileLocation.fromBucket`, which auto-grants read access) or a URL
   * (`FileLocation.url`).
   * @default - service default font
   */
  readonly font?: FileLocation;
  /**
   * The color of the burned-in captions.
   * @default CaptionFontColor.WHITE
   */
  readonly fontColor?: CaptionFontColor;
  /**
   * The opacity of the burned-in captions (0 transparent .. 255 opaque).
   * @default 255
   */
  readonly fontOpacity?: number;
  /**
   * The font resolution in DPI.
   * @default 96
   */
  readonly fontResolution?: number;
  /**
   * Font size — `CaptionFontSize.AUTO` to scale with the output, or `CaptionFontSize.of(points)`
   * for an exact size in points.
   * @default CaptionFontSize.AUTO
   */
  readonly fontSize?: CaptionFontSize;
  /**
   * The font outline color.
   * @default CaptionOutlineColor.BLACK
   */
  readonly outlineColor?: CaptionOutlineColor;
  /**
   * The font outline size in pixels.
   * @default 2
   */
  readonly outlineSize?: number;
  /**
   * The color of the shadow cast by the captions.
   * @default CaptionShadowColor.NONE
   */
  readonly shadowColor?: CaptionShadowColor;
  /**
   * The opacity of the shadow (0 transparent .. 255 opaque).
   * @default 0
   */
  readonly shadowOpacity?: number;
  /**
   * The horizontal offset of the shadow in pixels (negative shifts left).
   * @default - service default
   */
  readonly shadowXOffset?: number;
  /**
   * The vertical offset of the shadow in pixels (negative shifts up).
   * @default - service default
   */
  readonly shadowYOffset?: number;
  /**
   * For Teletext input, the number of lines for the captions bitmap.
   * @default - service default
   */
  readonly subtitleRows?: string;
  /**
   * Whether a fixed grid is used to generate the subtitle bitmap (Teletext input).
   * @default CaptionTeletextGridControl.FIXED
   */
  readonly teletextGridControl?: CaptionTeletextGridControl;
  /**
   * The horizontal position of the captions in pixels from the left.
   * @default - determined by `alignment`
   */
  readonly xPosition?: number;
  /**
   * The vertical position of the captions in pixels from the top.
   * @default - positioned towards the bottom
   */
  readonly yPosition?: number;
}

/**
 * Properties for burn-in captions.
 */
export interface BurnInDestinationProps extends CaptionFontStyleProps {}

/**
 * Properties for DVB-Sub captions.
 */
export interface DvbSubDestinationProps extends CaptionFontStyleProps {}

/** Properties for EBU-TT-D caption output. */
export interface EbuTtDDestinationProps {
  /**
   * The copyright holder included in the TTML copyright metadata tag.
   * @default - no copyright holder
   */
  readonly copyrightHolder?: string;
  /**
   * The default font size.
   * @default - service default
   */
  readonly defaultFontSize?: number;
  /**
   * The default line height.
   * @default - service default
   */
  readonly defaultLineHeight?: number;
  /**
   * How to handle the gap between multi-line captions.
   * @default - service default
   */
  readonly fillLineGap?: EbuTtDFillLineGap;
  /**
   * A comma-separated list of font families to include (valid only when `styleControl` is INCLUDE).
   * @default - 'monospaced'
   */
  readonly fontFamily?: string;
  /**
   * Whether source style information is included in the EBU-TT-D font data.
   * @default - service default
   */
  readonly styleControl?: EbuTtDStyleControl;
}

/** Properties for TTML caption output. */
export interface TtmlDestinationProps {
  /**
   * Whether to pass through style and position from a TTML-like source.
   * @default TtmlStyleControl.PASSTHROUGH
   */
  readonly styleControl?: TtmlStyleControl;
}

/** Properties for WebVTT caption output. */
export interface WebvttDestinationProps {
  /**
   * Whether to pass through source color and position to the WebVTT output.
   * PASSTHROUGH is only valid for EMBEDDED or TELETEXT sources.
   * @default WebvttStyleControl.NO_STYLE_DATA
   */
  readonly styleControl?: WebvttStyleControl;
}

/**
 * The output caption format for a caption encode. Use the static factory methods to select one of
 * the supported destination types.
 */
export abstract class CaptionDestination {
  /** Burned-in captions rendered into the video. */
  public static burnIn(props: BurnInDestinationProps = {}): CaptionDestination {
    return new BurnInCaptionDestination(props);
  }
  /** DVB-Sub bitmap captions. */
  public static dvbSub(props: DvbSubDestinationProps = {}): CaptionDestination {
    return new DvbSubCaptionDestination(props);
  }
  /** EBU-TT-D sidecar captions. */
  public static ebuTtD(props: EbuTtDDestinationProps = {}): CaptionDestination {
    return new EbuTtDCaptionDestination(props);
  }
  /** TTML sidecar captions. */
  public static ttml(props: TtmlDestinationProps = {}): CaptionDestination {
    return new TtmlCaptionDestination(props);
  }
  /** WebVTT sidecar captions. */
  public static webvtt(props: WebvttDestinationProps = {}): CaptionDestination {
    return new WebvttCaptionDestination(props);
  }
  /** ARIB captions. */
  public static arib(): CaptionDestination {
    return new SimpleCaptionDestination({ aribDestinationSettings: {} });
  }
  /** Embedded (CEA-608/708) captions. */
  public static embedded(): CaptionDestination {
    return new SimpleCaptionDestination({ embeddedDestinationSettings: {} }, true, true);
  }
  /** Embedded plus SCTE-20 captions. */
  public static embeddedPlusScte20(): CaptionDestination {
    return new SimpleCaptionDestination({ embeddedPlusScte20DestinationSettings: {} }, true, true);
  }
  /** SCTE-20 plus embedded captions. */
  public static scte20PlusEmbedded(): CaptionDestination {
    return new SimpleCaptionDestination({ scte20PlusEmbeddedDestinationSettings: {} }, true, true);
  }
  /** SMPTE-TT sidecar captions. */
  public static smpteTt(): CaptionDestination {
    return new SimpleCaptionDestination({ smpteTtDestinationSettings: {} });
  }
  /** Teletext captions. */
  public static teletext(): CaptionDestination {
    return new SimpleCaptionDestination({ teletextDestinationSettings: {} });
  }
  /** RTMP CaptionInfo captions. */
  public static rtmpCaptionInfo(): CaptionDestination {
    return new SimpleCaptionDestination({ rtmpCaptionInfoDestinationSettings: {} });
  }

  /** @internal */
  public abstract _bind(): CfnChannel.CaptionDestinationSettingsProperty;

  /**
   * Whether this caption type produces no separate track (rendered into the video or passed
   * through in the video stream). Burn-in, embedded, embeddedPlusScte20, and scte20PlusEmbedded
   * are in-band.
   * @internal
   */
  public _isInBand(): boolean {
    return false;
  }

  /**
   * Whether this is an embedded-family caption type (embedded, embeddedPlusScte20,
   * scte20PlusEmbedded). Only one embedded caption per output is allowed.
   * @internal
   */
  public _isEmbedded(): boolean {
    return false;
  }

  /**
   * Grant the channel role read access to any external files this destination references (e.g. a
   * burn-in font in S3). Default is a no-op; burn-in and DVB-Sub destinations override it.
   * @internal
   */
  public _grantRead(_role: IRole): void {}
}

/** @internal */
function bindFontStyle(p: CaptionFontStyleProps): CfnChannel.BurnInDestinationSettingsProperty {
  return {
    alignment: (p.alignment ?? CaptionAlignment.CENTERED).value,
    backgroundColor: p.backgroundColor?.value,
    backgroundOpacity: p.backgroundOpacity,
    font: p.font?._bind(),
    fontColor: (p.fontColor ?? CaptionFontColor.WHITE).value,
    fontOpacity: p.fontOpacity ?? 255,
    fontResolution: p.fontResolution ?? 96,
    fontSize: (p.fontSize ?? CaptionFontSize.AUTO)._bind(),
    outlineColor: (p.outlineColor ?? CaptionOutlineColor.BLACK).value,
    outlineSize: p.outlineSize ?? 2,
    shadowColor: (p.shadowColor ?? CaptionShadowColor.NONE).value,
    shadowOpacity: p.shadowOpacity ?? 0,
    shadowXOffset: p.shadowXOffset,
    shadowYOffset: p.shadowYOffset,
    subtitleRows: p.subtitleRows,
    teletextGridControl: (p.teletextGridControl ?? CaptionTeletextGridControl.FIXED).value,
    xPosition: p.xPosition,
    yPosition: p.yPosition,
  };
}

/** @internal */
class BurnInCaptionDestination extends CaptionDestination {
  constructor(private readonly props: BurnInDestinationProps) { super(); }
  public _bind(): CfnChannel.CaptionDestinationSettingsProperty {
    return { burnInDestinationSettings: bindFontStyle(this.props) };
  }
  public override _isInBand(): boolean {
    return true;
  }
  public override _grantRead(role: IRole): void {
    this.props.font?._grantRead(role);
  }
}

/** @internal */
class DvbSubCaptionDestination extends CaptionDestination {
  constructor(private readonly props: DvbSubDestinationProps) { super(); }
  public _bind(): CfnChannel.CaptionDestinationSettingsProperty {
    return { dvbSubDestinationSettings: bindFontStyle(this.props) };
  }
  public override _grantRead(role: IRole): void {
    this.props.font?._grantRead(role);
  }
}

/** @internal */
class EbuTtDCaptionDestination extends CaptionDestination {
  constructor(private readonly props: EbuTtDDestinationProps) { super(); }
  public _bind(): CfnChannel.CaptionDestinationSettingsProperty {
    return {
      ebuTtDDestinationSettings: {
        copyrightHolder: this.props.copyrightHolder,
        defaultFontSize: this.props.defaultFontSize,
        defaultLineHeight: this.props.defaultLineHeight,
        fillLineGap: this.props.fillLineGap?.value,
        fontFamily: this.props.fontFamily,
        styleControl: this.props.styleControl?.value,
      },
    };
  }
}

/** @internal */
class TtmlCaptionDestination extends CaptionDestination {
  constructor(private readonly props: TtmlDestinationProps) { super(); }
  public _bind(): CfnChannel.CaptionDestinationSettingsProperty {
    return { ttmlDestinationSettings: { styleControl: (this.props.styleControl ?? TtmlStyleControl.PASSTHROUGH).value } };
  }
}

/** @internal */
class WebvttCaptionDestination extends CaptionDestination {
  constructor(private readonly props: WebvttDestinationProps) { super(); }
  public _bind(): CfnChannel.CaptionDestinationSettingsProperty {
    return { webvttDestinationSettings: { styleControl: (this.props.styleControl ?? WebvttStyleControl.NO_STYLE_DATA).value } };
  }
}

/** @internal */
class SimpleCaptionDestination extends CaptionDestination {
  constructor(
    private readonly settings: CfnChannel.CaptionDestinationSettingsProperty,
    private readonly inBand: boolean = false,
    private readonly embedded: boolean = false,
  ) { super(); }
  public _bind(): CfnChannel.CaptionDestinationSettingsProperty {
    return this.settings;
  }
  public override _isInBand(): boolean {
    return this.inBand;
  }
  public override _isEmbedded(): boolean {
    return this.embedded;
  }
}
