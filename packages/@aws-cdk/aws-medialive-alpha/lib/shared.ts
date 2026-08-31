import { Arn, ArnFormat, UnscopedValidationError, Token } from 'aws-cdk-lib';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import { SegmentLengthUnits } from './enums';

/**
 * Parse the trailing `:<id>` resource name out of a MediaLive ARN
 * (`arn:<partition>:medialive:<region>:<account>:<resourceType>:<id>`).
 *
 * @param arn The full resource ARN.
 * @param resourceType The MediaLive resource type, used only in the error message
 * (e.g. `'Channel'`, `'Input'`).
 * @internal
 */
export function extractResourceId(arn: string, resourceType: string): string {
  const parsed = Arn.split(arn, ArnFormat.COLON_RESOURCE_NAME).resourceName;
  if (parsed === undefined) {
    throw new UnscopedValidationError(
      lit`InvalidMediaLiveArn`,
      `Could not parse MediaLive ${resourceType} ARN: ${arn}. Expected format: arn:<partition>:medialive:<region>:<account>:<resourceType>:<id>`,
    );
  }
  return parsed;
}

/**
 * The length of a media segment for an output group.
 *
 * Express the length in whole seconds with {@link Segment.seconds} or in milliseconds
 * with {@link Segment.milliseconds}. Some output groups (e.g. HLS) only support
 * whole-second segments and will reject sub-second millisecond values.
 */
export class Segment {
  /**
   * A segment length in whole seconds.
   * @param value Number of seconds (non-negative integer).
   */
  public static seconds(value: number): Segment {
    return new Segment(value, SegmentLengthUnits.SECONDS);
  }

  /**
   * A segment length in milliseconds.
   * @param value Number of milliseconds (non-negative integer).
   */
  public static milliseconds(value: number): Segment {
    return new Segment(value, SegmentLengthUnits.MILLISECONDS);
  }

  /** @internal */
  private readonly _value: number;
  /** @internal */
  private readonly _unitsValue: SegmentLengthUnits;

  private constructor(value: number, units: SegmentLengthUnits) {
    if (!Token.isUnresolved(value) && (!Number.isInteger(value) || value < 0)) {
      throw new UnscopedValidationError(lit`SegmentLength`, `Segment length must be a non-negative integer, got ${value}`);
    }
    this._value = value;
    this._unitsValue = units;
  }

  /**
   * The raw length value, in the configured units.
   * @internal
   */
  public _length(): number {
    return this._value;
  }

  /**
   * The length units.
   * @internal
   */
  public _units(): SegmentLengthUnits {
    return this._unitsValue;
  }

  /**
   * The length in whole seconds, for output groups whose segment length is seconds-only.
   * Throws when the segment is expressed in sub-second milliseconds.
   * @internal
   */
  public _toSeconds(): number {
    if (this._unitsValue === SegmentLengthUnits.SECONDS) {
      return this._value;
    }
    if (this._value % 1000 !== 0) {
      throw new UnscopedValidationError(lit`SegmentLengthNotWholeSeconds`, `Segment length for this output group must be a whole number of seconds, got ${this._value} milliseconds`);
    }
    return this._value / 1000;
  }
}

/**
 * The pixel aspect ratio (PAR) of the video.
 *
 * Use the predefined constants for standard ratios, or {@link PixelAspectRatio.of} for
 * a custom ratio.
 */
export class PixelAspectRatio {
  /** Square pixels (`1:1`). */
  public static readonly SQUARE = new PixelAspectRatio(1, 1);

  /**
   * Define a pixel aspect ratio.
   *
   * @param numerator Numerator of the ratio.
   * @param denominator Denominator of the ratio.
   */
  public static of(numerator: number, denominator: number): PixelAspectRatio {
    if (!Token.isUnresolved(numerator) && (!Number.isInteger(numerator) || numerator <= 0)) {
      throw new UnscopedValidationError(lit`PixelAspectRatioNumerator`, `Pixel aspect ratio numerator must be a positive integer, got ${numerator}`);
    }
    if (!Token.isUnresolved(denominator) && (!Number.isInteger(denominator) || denominator <= 0)) {
      throw new UnscopedValidationError(lit`PixelAspectRatioDenominator`, `Pixel aspect ratio denominator must be a positive integer, got ${denominator}`);
    }
    return new PixelAspectRatio(numerator, denominator);
  }

  /** @internal */
  private readonly _numeratorValue: number;
  /** @internal */
  private readonly _denominatorValue: number;

  /**
   * @param numerator Numerator of the ratio.
   * @param denominator Denominator of the ratio.
   */
  private constructor(numerator: number, denominator: number) {
    this._numeratorValue = numerator;
    this._denominatorValue = denominator;
  }

  /** Returns the string value in `numerator:denominator` form. */
  public toString(): string {
    return `${this._numeratorValue}:${this._denominatorValue}`;
  }

  /**
   * The numerator of the ratio.
   * @internal
   */
  public _numerator(): number {
    return this._numeratorValue;
  }

  /**
   * The denominator of the ratio.
   * @internal
   */
  public _denominator(): number {
    return this._denominatorValue;
  }
}
