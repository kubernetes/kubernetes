import { UnscopedValidationError, Token } from 'aws-cdk-lib';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';

/**
 * A video frame rate expressed as a rational number (numerator/denominator).
 *
 * Use the predefined constants for standard rates, or {@link Framerate.of} for a custom rate.
 */
export class Framerate {
  /** 24 fps (cinema) — `24/1`. */
  public static readonly FPS_24 = new Framerate(24, 1);
  /** 25 fps — `25/1`. */
  public static readonly FPS_25 = new Framerate(25, 1);
  /** 29.97 fps — `30000/1001`. */
  public static readonly FPS_29_97 = new Framerate(30000, 1001);
  /** 30 fps — `30/1`. */
  public static readonly FPS_30 = new Framerate(30, 1);
  /** 50 fps — `50/1`. */
  public static readonly FPS_50 = new Framerate(50, 1);
  /** 59.94 fps — `60000/1001`. */
  public static readonly FPS_59_94 = new Framerate(60000, 1001);
  /** 60 fps — `60/1`. */
  public static readonly FPS_60 = new Framerate(60, 1);

  /**
   * Define a custom frame rate.
   *
   * @param numerator Numerator of the rate.
   * @param denominator Denominator of the rate.
   */
  public static of(numerator: number, denominator: number): Framerate {
    if (!Token.isUnresolved(numerator) && (!Number.isInteger(numerator) || numerator <= 0)) {
      throw new UnscopedValidationError(lit`FramerateNumerator`, `Frame rate numerator must be a positive integer, got ${numerator}`);
    }
    if (!Token.isUnresolved(denominator) && (!Number.isInteger(denominator) || denominator <= 0)) {
      throw new UnscopedValidationError(lit`FramerateDenominator`, `Frame rate denominator must be a positive integer, got ${denominator}`);
    }
    return new Framerate(numerator, denominator);
  }

  /** @internal */
  private readonly _numeratorValue: number;
  /** @internal */
  private readonly _denominatorValue: number;

  private constructor(numerator: number, denominator: number) {
    this._numeratorValue = numerator;
    this._denominatorValue = denominator;
  }

  /** Returns the string value. */
  public toString(): string {
    return `${this._numeratorValue}/${this._denominatorValue}`;
  }

  /**
   * Returns the number value.
   * @internal
   */
  public _numerator(): number {
    return this._numeratorValue;
  }

  /**
   * Returns the number value.
   * @internal
   */
  public _denominator(): number {
    return this._denominatorValue;
  }
}
