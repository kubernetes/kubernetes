import type { Lut } from './file-location';

/**
 * A color space supported for 3D-LUT color conversion in a color-correction rule.
 */
export class ColorSpace {
  /** HDR10. */
  public static readonly HDR10 = new ColorSpace('HDR10');
  /** HLG (Rec. 2020). */
  public static readonly HLG_2020 = new ColorSpace('HLG_2020');
  /** Rec. 601 (SD). */
  public static readonly REC_601 = new ColorSpace('REC_601');
  /** Rec. 709 (HD). */
  public static readonly REC_709 = new ColorSpace('REC_709');

  /** A value not yet modelled by AWS CDK. */
  public static of(value: string): ColorSpace {
    return new ColorSpace(value);
  }

  /** The underlying string value passed to CloudFormation. */
  public readonly value: string;

  private constructor(value: string) {
    this.value = value;
  }
}

/**
 * A color space correction rule.
 */
export interface ColorCorrection {
  /**
   * The input color space to match.
   */
  readonly inputColorSpace: ColorSpace;
  /**
   * The output color space to convert to.
   */
  readonly outputColorSpace: ColorSpace;
  /**
   * The 3D LUT file for the color correction. MediaLive reads the LUT from S3 at runtime, so it
   * must be an S3 location — provide it via `Lut.fromBucket()` (which uses the secure `s3ssl://`
   * form and auto-grants the channel role read access) or `Lut.url()` with an `s3://`/`s3ssl://` URL.
   * @default - no LUT file
   */
  readonly lut?: Lut;
}
