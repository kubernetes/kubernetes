import { UnscopedValidationError } from './errors';
import { lit } from './private/literal-string';
import { Token } from './token';

/**
 * Represents the amount of digital storage.
 *
 * The amount can be specified either as a literal value (e.g: `10`) which
 * cannot be negative, or as an unresolved number token.
 *
 * When the amount is passed as a token, unit conversion is not possible.
 */
export class Size {
  /**
   * Create a Storage representing an amount bytes.
   *
   * @param amount the amount of bytes to be represented
   *
   * @returns a new `Size` instance
   */
  public static bytes(amount: number): Size {
    return new Size(amount, StorageUnit.Bytes);
  }

  /**
   * Create a Storage representing an amount kibibytes.
   * 1 KiB = 1024 bytes
   *
   * @param amount the amount of kibibytes to be represented
   *
   * @returns a new `Size` instance
   */
  public static kibibytes(amount: number): Size {
    return new Size(amount, StorageUnit.Kibibytes);
  }

  /**
   * Create a Storage representing an amount mebibytes.
   * 1 MiB = 1024 KiB
   *
   * @param amount the amount of mebibytes to be represented
   *
   * @returns a new `Size` instance
   */
  public static mebibytes(amount: number): Size {
    return new Size(amount, StorageUnit.Mebibytes);
  }

  /**
   * Create a Storage representing an amount gibibytes.
   * 1 GiB = 1024 MiB
   *
   * @param amount the amount of gibibytes to be represented
   *
   * @returns a new `Size` instance
   */
  public static gibibytes(amount: number): Size {
    return new Size(amount, StorageUnit.Gibibytes);
  }

  /**
   * Create a Storage representing an amount tebibytes.
   * 1 TiB = 1024 GiB
   *
   * @param amount the amount of tebibytes to be represented
   *
   * @returns a new `Size` instance
   */
  public static tebibytes(amount: number): Size {
    return new Size(amount, StorageUnit.Tebibytes);
  }

  /**
   * Create a Storage representing an amount pebibytes.
   * 1 PiB = 1024 TiB
   *
   * @deprecated use `pebibytes` instead
   */
  public static pebibyte(amount: number): Size {
    return Size.pebibytes(amount);
  }

  /**
   * Create a Storage representing an amount pebibytes.
   * 1 PiB = 1024 TiB
   *
   * @param amount the amount of pebibytes to be represented
   *
   * @returns a new `Size` instance
   */
  public static pebibytes(amount: number): Size {
    return new Size(amount, StorageUnit.Pebibytes);
  }

  private readonly amount: number;
  private readonly unit: StorageUnit;

  private constructor(amount: number, unit: StorageUnit) {
    if (!Token.isUnresolved(amount) && amount < 0) {
      throw new UnscopedValidationError(lit`StorageAmountsCannotNegative`, `Storage amounts cannot be negative. Received: ${amount}`);
    }
    this.amount = amount;
    this.unit = unit;
  }

  /**
   * Return this storage as a total number of bytes.
   *
   * @param opts the conversion options
   *
   * @returns the quantity expressed in bytes
   */
  public toBytes(opts: SizeConversionOptions = {}): number {
    return convert(this.amount, this.unit, StorageUnit.Bytes, opts);
  }

  /**
   * Return this storage as a total number of kibibytes.
   *
   * @param opts the conversion options
   *
   * @returns the quantity of bytes expressed in kibibytes
   */
  public toKibibytes(opts: SizeConversionOptions = {}): number {
    return convert(this.amount, this.unit, StorageUnit.Kibibytes, opts);
  }

  /**
   * Return this storage as a total number of mebibytes.
   *
   * @param opts the conversion options
   *
   * @returns the quantity of bytes expressed in mebibytes
   */
  public toMebibytes(opts: SizeConversionOptions = {}): number {
    return convert(this.amount, this.unit, StorageUnit.Mebibytes, opts);
  }

  /**
   * Return this storage as a total number of gibibytes.
   *
   * @param opts the conversion options
   *
   * @returns the quantity of bytes expressed in gibibytes
   */
  public toGibibytes(opts: SizeConversionOptions = {}): number {
    return convert(this.amount, this.unit, StorageUnit.Gibibytes, opts);
  }

  /**
   * Return this storage as a total number of tebibytes.
   *
   * @param opts the conversion options
   *
   * @returns the quantity of bytes expressed in tebibytes
   */
  public toTebibytes(opts: SizeConversionOptions = {}): number {
    return convert(this.amount, this.unit, StorageUnit.Tebibytes, opts);
  }

  /**
   * Return this storage as a total number of pebibytes.
   *
   * @param opts the conversion options
   *
   * @returns the quantity of bytes expressed in pebibytes
   */
  public toPebibytes(opts: SizeConversionOptions = {}): number {
    return convert(this.amount, this.unit, StorageUnit.Pebibytes, opts);
  }

  /**
   * Checks if size is a token or a resolvable object
   */
  public isUnresolved() {
    return Token.isUnresolved(this.amount);
  }

  /**
   * Convert this Size object to the most appropriate readable number.
   */
  public toHumanString() {
    if (this.isUnresolved()) {
      return `${Token.asString(this.amount)} ${this.unit.shortLabel}`;
    }

    const bytes = this.toBytes();

    for (const unit of allStorageUnits()) {
      if (bytes >= unit.inBytes) {
        const am = convert(bytes, StorageUnit.Bytes, unit, { rounding: SizeRoundingBehavior.NONE });
        return `${am.toFixed(2).replace(/\.?0+$/, '')} ${unit.shortLabel}`;
      }
    }

    return `${bytes} bytes`;
  }

  /**
   * Convert this Size object to its constructor form
   */
  public toString() {
    return `Size.${this.unit.label}(${this.amount})`;
  }
}

function allStorageUnits() {
  return [
    StorageUnit.Pebibytes,
    StorageUnit.Tebibytes,
    StorageUnit.Gibibytes,
    StorageUnit.Mebibytes,
    StorageUnit.Kibibytes,
    StorageUnit.Bytes,
  ];
}

/**
 * Rounding behaviour when converting between units of `Size`.
 */
export enum SizeRoundingBehavior {
  /** Fail the conversion if the result is not an integer. */
  FAIL,
  /** If the result is not an integer, round it to the closest integer less than the result */
  FLOOR,
  /** Don't round. Return even if the result is a fraction. */
  NONE,
}

/**
 * Options for how to convert time to a different unit.
 */
export interface SizeConversionOptions {
  /**
   * How conversions should behave when it encounters a non-integer result
   * @default SizeRoundingBehavior.FAIL
   */
  readonly rounding?: SizeRoundingBehavior;
}

class StorageUnit {
  public static readonly Bytes = new StorageUnit('bytes', 'bytes', 1);
  public static readonly Kibibytes = new StorageUnit('kibibytes', 'KiB', 1024);
  public static readonly Mebibytes = new StorageUnit('mebibytes', 'MiB', 1024 * 1024);
  public static readonly Gibibytes = new StorageUnit('gibibytes', 'GiB', 1024 * 1024 * 1024);
  public static readonly Tebibytes = new StorageUnit('tebibytes', 'TiB', 1024 * 1024 * 1024 * 1024);
  public static readonly Pebibytes = new StorageUnit('pebibytes', 'PiB', 1024 * 1024 * 1024 * 1024 * 1024);

  private constructor(public readonly label: string, public readonly shortLabel: string, public readonly inBytes: number) {
    // MAX_SAFE_INTEGER is 2^53, so by representing storage in kibibytes,
    // the highest storage we can represent is 8 exbibytes.
  }

  public toString() {
    return this.label;
  }
}

function convert(amount: number, fromUnit: StorageUnit, toUnit: StorageUnit, options: SizeConversionOptions = {}) {
  const rounding = options.rounding ?? SizeRoundingBehavior.FAIL;
  if (fromUnit.inBytes === toUnit.inBytes) { return amount; }
  if (Token.isUnresolved(amount)) {
    throw new UnscopedValidationError(lit`MustBeSizeSpecifiedSize`, `Size must be specified as 'Size.${toUnit}()' here since its value comes from a token and cannot be converted (got Size.${fromUnit})`);
  }

  const multiplier = fromUnit.inBytes / toUnit.inBytes;
  const value = amount * multiplier;
  switch (rounding) {
    case SizeRoundingBehavior.NONE:
      return value;
    case SizeRoundingBehavior.FLOOR:
      return Math.floor(value);
    default:
    case SizeRoundingBehavior.FAIL:
      if (!Number.isInteger(value)) {
        throw new UnscopedValidationError(lit`CannotConvertedIntoWhole`, `'${amount} ${fromUnit}' cannot be converted into a whole number of ${toUnit}.`);
      }
      return value;
  }
}
