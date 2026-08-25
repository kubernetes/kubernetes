import { Token, UnscopedValidationError } from 'aws-cdk-lib';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';

/**
 * A column of a table.
 */
export interface Column {
  /**
   * Name of the column.
   */
  readonly name: string;

  /**
   * Type of the column.
   */
  readonly type: Type;

  /**
   * Coment describing the column.
   *
   * @default none
   */
  readonly comment?: string;
}

/**
 * The type of a column in a table schema.
 *
 * Instances are opaque: obtain one from a `Schema` factory (for example
 * `Schema.STRING`, `Schema.decimal(...)`, `Schema.array(...)`) or, for a type the
 * `Schema` factories don't model, from `Schema.custom(...)`.
 */
export class Type {
  /**
   * Create a `Type` from its parts.
   *
   * @internal
   */
  public static _of(inputString: string, isPrimitive: boolean): Type {
    return new Type(inputString, isPrimitive);
  }

  /**
   * Indicates whether this type is a primitive data type.
   */
  public readonly isPrimitive: boolean;

  /**
   * Glue InputString for this type.
   */
  public readonly inputString: string;

  private constructor(inputString: string, isPrimitive: boolean) {
    this.inputString = inputString;
    this.isPrimitive = isPrimitive;
  }
}

/**
 * @see https://docs.aws.amazon.com/athena/latest/ug/data-types.html
 */
export class Schema {
  public static readonly BOOLEAN: Type = Type._of('boolean', true);

  public static readonly BINARY: Type = Type._of('binary', true);

  /**
   * A 64-bit signed INTEGER in two’s complement format, with a minimum value of -2^63 and a maximum value of 2^63-1.
   */
  public static readonly BIG_INT: Type = Type._of('bigint', true);

  public static readonly DOUBLE: Type = Type._of('double', true);

  public static readonly FLOAT: Type = Type._of('float', true);

  /**
   * A 32-bit signed INTEGER in two’s complement format, with a minimum value of -2^31 and a maximum value of 2^31-1.
   */
  public static readonly INTEGER: Type = Type._of('int', true);

  /**
   * A 16-bit signed INTEGER in two’s complement format, with a minimum value of -2^15 and a maximum value of 2^15-1.
   */
  public static readonly SMALL_INT: Type = Type._of('smallint', true);

  /**
   * A 8-bit signed INTEGER in two’s complement format, with a minimum value of -2^7 and a maximum value of 2^7-1
   */
  public static readonly TINY_INT: Type = Type._of('tinyint', true);

  /**
   * Date type.
   */
  public static readonly DATE: Type = Type._of('date', true);

  /**
   * Timestamp type (date and time).
   */
  public static readonly TIMESTAMP: Type = Type._of('timestamp', true);

  /**
   * Arbitrary-length string type.
   */
  public static readonly STRING: Type = Type._of('string', true);

  /**
   * Creates a decimal type.
   *
   * @param precision the total number of digits, between 1 and 38
   * @param scale the number of digits in the fractional part, between 0 and 38; the default is 0
   * @see https://docs.aws.amazon.com/athena/latest/ug/data-types.html
   */
  public static decimal(precision: number, scale?: number): Type {
    if (Token.isResolved(precision) && (precision < 1 || precision > 38 || precision % 1 !== 0)) {
      throw new UnscopedValidationError(lit`DecimalPrecisionOutOfRange`, `decimal precision must be a positive integer between 1 and 38, got ${precision}`);
    }
    if (scale !== undefined && Token.isResolved(scale) && (scale < 0 || scale > 38 || scale % 1 !== 0)) {
      throw new UnscopedValidationError(lit`DecimalScaleOutOfRange`, `decimal scale must be an integer between 0 and 38, got ${scale}`);
    }
    return Type._of(scale !== undefined ? `decimal(${precision},${scale})` : `decimal(${precision})`, true);
  }

  /**
   * Fixed length character data, with a specified length between 1 and 255.
   *
   * @param length length between 1 and 255
   */
  public static char(length: number): Type {
    if (length <= 0 || length > 255) {
      throw new UnscopedValidationError(lit`CharLengthOutOfRange`, `char length must be (inclusively) between 1 and 255, but was ${length}`);
    }
    if (length % 1 !== 0) {
      throw new UnscopedValidationError(lit`CharLengthNotInteger`, `char length must be a positive integer, was ${length}`);
    }
    return Type._of(`char(${length})`, true);
  }

  /**
   * Variable length character data, with a specified length between 1 and 65535.
   *
   * @param length length between 1 and 65535.
   */
  public static varchar(length: number): Type {
    if (length <= 0 || length > 65535) {
      throw new UnscopedValidationError(lit`VarcharLengthOutOfRange`, `varchar length must be (inclusively) between 1 and 65535, but was ${length}`);
    }
    if (length % 1 !== 0) {
      throw new UnscopedValidationError(lit`VarcharLengthNotInteger`, `varchar length must be a positive integer, was ${length}`);
    }
    return Type._of(`varchar(${length})`, true);
  }

  /**
   * Creates an array of some other type.
   *
   * @param itemType type contained by the array.
   */
  public static array(itemType: Type): Type {
    return Type._of(`array<${itemType.inputString}>`, false);
  }

  /**
   * Creates a map of some primitive key type to some value type.
   *
   * @param keyType type of key, must be a primitive.
   * @param valueType type fo the value indexed by the key.
   */
  public static map(keyType: Type, valueType: Type): Type {
    if (!keyType.isPrimitive) {
      throw new UnscopedValidationError(lit`MapKeyTypeNotPrimitive`, `the key type of a 'map' must be a primitive, but was ${keyType.inputString}`);
    }
    return Type._of(`map<${keyType.inputString},${valueType.inputString}>`, false);
  }

  /**
   * Creates a nested structure containing individually named and typed columns.
   *
   * @param columns the columns of the structure.
   */
  public static struct(columns: Column[]): Type {
    return Type._of(`struct<${columns.map(column => {
      return `${column.name}:${column.type.inputString}`;
    }).join(',')}>`, false);
  }

  /**
   * Creates a custom type from a raw Glue input string.
   *
   * Escape hatch for column types the other `Schema` factories don't model. The
   * `inputString` is emitted verbatim and is not validated.
   *
   * @param inputString the Glue input string for the type (for example `interval_day_to_second`).
   * @param isPrimitive whether the type is a primitive (non-nested) data type. Defaults to true.
   */
  public static custom(inputString: string, isPrimitive?: boolean): Type {
    return Type._of(inputString, isPrimitive ?? true);
  }
}
