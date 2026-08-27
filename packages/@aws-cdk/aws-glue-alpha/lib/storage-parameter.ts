import type * as kms from 'aws-cdk-lib/aws-kms';

/**
 * The compression type.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"compression_type"_
 */
export enum CompressionType {
  /**
   * No compression.
   */
  NONE = 'none',

  /**
   * Burrows-Wheeler compression.
   */
  BZIP2 = 'bzip2',

  /**
   * Deflate compression.
   */
  GZIP = 'gzip',

  /**
   * Compression algorithm focused on high compression and decompression speeds, rather than the maximum possible compression.
   */
  SNAPPY = 'snappy',
}

/**
 * Specifies the action to perform when query results contain invalid UTF-8 character values.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"invalid_char_handling"_
 */
export enum InvalidCharHandlingAction {
  /**
   * Doesn't perform invalid character handling.
   */
  DISABLED = 'DISABLED',

  /**
   * Cancels queries that return data containing invalid UTF-8 values.
   */
  FAIL = 'FAIL',

  /**
   * Replaces invalid UTF-8 values with null.
   */
  SET_TO_NULL = 'SET_TO_NULL',

  /**
   * Replaces each value in the row with null.
   */
  DROP_ROW = 'DROP_ROW',

  /**
   * Replaces the invalid character with the replacement character you specify using `REPLACEMENT_CHAR`.
   */
  REPLACE = 'REPLACE',
}

/**
 * Specifies the action to perform when ORC data contains an integer (for example, BIGINT or int64) that is larger than the column definition (for example, SMALLINT or int16).
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"numeric_overflow_handling"_
 */
export enum NumericOverflowHandlingAction {
  /**
   * Invalid character handling is turned off.
   */
  DISABLED = 'DISABLED',

  /**
   * Cancel the query when the data includes invalid characters.
   */
  FAIL = 'FAIL',

  /**
   * Set invalid characters to null.
   */
  SET_TO_NULL = 'SET_TO_NULL',

  /**
   * Set each value in the row to null.
   */
  DROP_ROW = 'DROP_ROW',
}

/**
 * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARBYTE data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"surplus_bytes_handling"_
 */
export enum SurplusBytesHandlingAction {
  /**
   * Replaces data that exceeds the column width with null.
   */
  SET_TO_NULL = 'SET_TO_NULL',

  /**
   * Doesn't perform surplus byte handling.
   */
  DISABLED = 'DISABLED',

  /**
   * Cancels queries that return data exceeding the column width.
   */
  FAIL = 'FAIL',

  /**
   * Drop all rows that contain data exceeding column width.
   */
  DROP_ROW = 'DROP_ROW',

  /**
   * Removes the characters that exceed the maximum number of characters defined for the column.
   */
  TRUNCATE = 'TRUNCATE',
}

/**
 * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARCHAR, CHAR, or string data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"surplus_char_handling"_
 */
export enum SurplusCharHandlingAction {
  /**
   * Replaces data that exceeds the column width with null.
   */
  SET_TO_NULL = 'SET_TO_NULL',

  /**
   * Doesn't perform surplus character handling.
   */
  DISABLED = 'DISABLED',

  /**
   * Cancels queries that return data exceeding the column width.
   */
  FAIL = 'FAIL',

  /**
   * Replaces each value in the row with null.
   */
  DROP_ROW = 'DROP_ROW',

  /**
   * Removes the characters that exceed the maximum number of characters defined for the column.
   */
  TRUNCATE = 'TRUNCATE',
}

/**
 * Identifies if the file contains less or more values for a row than the number of columns specified in the external table definition. This property is only available for an uncompressed text file format.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"column_count_mismatch_handling"_
 */
export enum ColumnCountMismatchHandlingAction {
  /**
   * Column count mismatch handling is turned off.
   */
  DISABLED = 'DISABLED',

  /**
   * Fail the query if the column count mismatch is detected.
   */
  FAIL = 'FAIL',

  /**
   * Fill missing values with NULL and ignore the additional values in each row.
   */
  SET_TO_NULL = 'SET_TO_NULL',

  /**
   * Drop all rows that contain column count mismatch error from the scan.
   */
  DROP_ROW = 'DROP_ROW',
}

/**
 * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARCHAR, CHAR, or string data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"surplus_char_handling"_
 */
export enum WriteParallel {
  /**
   * Write data in parallel.
   */
  ON = 'on',

  /**
   * Write data serially.
   */
  OFF = 'off',
}

/**
 * Specifies how to map columns when the table uses ORC data format.
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_ > _"orc.schema.resolution"_
 */
export enum OrcColumnMappingType {
  /**
   * Map columns by name.
   */
  NAME = 'name',

  /**
   * Map columns by position.
   */
  POSITION = 'position',
}

/**
 * The storage parameter keys that are currently known, this list is not exhaustive and other keys may be used.
 */
export enum StorageParameters {
  /**
   * The number of rows to skip at the top of a CSV file when the table is being created.
   */
  SKIP_HEADER_LINE_COUNT = 'skip.header.line.count',

  /**
   * Determines whether data handling is on for the table.
   */
  DATA_CLEANSING_ENABLED = 'data_cleansing_enabled',

  /**
   * The type of compression used on the table, when the file name does not contain an extension. This value overrides the compression type specified through the extension.
   */
  COMPRESSION_TYPE = 'compression_type',

  /**
   * Specifies the action to perform when query results contain invalid UTF-8 character values.
   */
  INVALID_CHAR_HANDLING = 'invalid_char_handling',

  /**
   * Specifies the replacement character to use when you set `INVALID_CHAR_HANDLING` to `REPLACE`.
   */
  REPLACEMENT_CHAR = 'replacement_char',

  /**
   * Specifies the action to perform when ORC data contains an integer (for example, BIGINT or int64) that is larger than the column definition (for example, SMALLINT or int16).
   */
  NUMERIC_OVERFLOW_HANDLING = 'numeric_overflow_handling',

  /**
   * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARBYTE data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
   */
  SURPLUS_BYTES_HANDLING = 'surplus_bytes_handling',

  /**
   * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARCHAR, CHAR, or string data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
   */
  SURPLUS_CHAR_HANDLING = 'surplus_char_handling',

  /**
   * Identifies if the file contains less or more values for a row than the number of columns specified in the external table definition. This property is only available for an uncompressed text file format.
   */
  COLUMN_COUNT_MISMATCH_HANDLING = 'column_count_mismatch_handling',

  /**
   * A property that sets the numRows value for the table definition. To explicitly update an external table's statistics, set the numRows property to indicate the size of the table. Amazon Redshift doesn't analyze external tables to generate the table statistics that the query optimizer uses to generate a query plan. If table statistics aren't set for an external table, Amazon Redshift generates a query execution plan based on an assumption that external tables are the larger tables and local tables are the smaller tables.
   */
  NUM_ROWS = 'num_rows',

  /**
   * A property that sets number of rows to skip at the beginning of each source file.
   */
  SERIALIZATION_NULL_FORMAT = 'serialization.null.format',

  /**
   * A property that sets the column mapping type for tables that use ORC data format. This property is ignored for other data formats.
   */
  ORC_SCHEMA_RESOLUTION = 'orc.schema.resolution',

  /**
   * A property that sets whether CREATE EXTERNAL TABLE AS should write data in parallel. When 'write.parallel' is set to off, CREATE EXTERNAL TABLE AS writes to one or more data files serially onto Amazon S3. This table property also applies to any subsequent INSERT statement into the same external table.
   */
  WRITE_PARALLEL = 'write.parallel',

  /**
   * A property that sets the maximum size (in MB) of each file written to Amazon S3 by CREATE EXTERNAL TABLE AS. The size must be a valid integer between 5 and 6200. The default maximum file size is 6,200 MB. This table property also applies to any subsequent INSERT statement into the same external table.
   */
  WRITE_MAX_FILESIZE_MB = 'write.maxfilesize.mb',

  /**
   * You can specify an AWS Key Management Service key to enable Server–Side Encryption (SSE) for Amazon S3 objects.
   */
  WRITE_KMS_KEY_ID = 'write.kms.key.id',
}

/**
 * A storage parameter. The list of storage parameters available is not exhaustive and other keys may be used.
 *
 * If you would like to specify a storage parameter that is not available as a static member of this class, use the `StorageParameter.custom` method.
 *
 * The list of storage parameters currently known within the CDK is listed.
 *
 * @see https://docs.aws.amazon.com/glue/latest/dg/table-properties-crawler.html
 *
 * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html#r_CREATE_EXTERNAL_TABLE-parameters - under _"TABLE PROPERTIES"_
 */
export class StorageParameter {
  /**
   * The number of rows to skip at the top of a CSV file when the table is being created.
   */
  public static skipHeaderLineCount(value: number): StorageParameter {
    return new StorageParameter('skip.header.line.count', value.toString());
  }

  /**
   * Determines whether data handling is on for the table.
   */
  public static dataCleansingEnabled(value: boolean): StorageParameter {
    return new StorageParameter('data_cleansing_enabled', value.toString());
  }

  /**
   * The type of compression used on the table, when the file name does not contain an extension. This value overrides the compression type specified through the extension.
   */
  public static compressionType(value: CompressionType): StorageParameter {
    return new StorageParameter('compression_type', value);
  }

  /**
   * Specifies the action to perform when query results contain invalid UTF-8 character values.
   */
  public static invalidCharHandling(value: InvalidCharHandlingAction): StorageParameter {
    return new StorageParameter('invalid_char_handling', value);
  }

  /**
   * Specifies the replacement character to use when you set `INVALID_CHAR_HANDLING` to `REPLACE`.
   */
  public static replacementChar(value: string): StorageParameter {
    return new StorageParameter('replacement_char', value);
  }

  /**
   * Specifies the action to perform when ORC data contains an integer (for example, BIGINT or int64) that is larger than the column definition (for example, SMALLINT or int16).
   */
  public static numericOverflowHandling(value: NumericOverflowHandlingAction): StorageParameter {
    return new StorageParameter('numeric_overflow_handling', value);
  }

  /**
   * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARBYTE data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
   */
  public static surplusBytesHandling(value: SurplusBytesHandlingAction): StorageParameter {
    return new StorageParameter('surplus_bytes_handling', value);
  }

  /**
   * Specifies how to handle data being loaded that exceeds the length of the data type defined for columns containing VARCHAR, CHAR, or string data. By default, Redshift Spectrum sets the value to null for data that exceeds the width of the column.
   */
  public static surplusCharHandling(value: SurplusCharHandlingAction): StorageParameter {
    return new StorageParameter('surplus_char_handling', value);
  }

  /**
   * Identifies if the file contains less or more values for a row than the number of columns specified in the external table definition. This property is only available for an uncompressed text file format.
   */
  public static columnCountMismatchHandling(value: ColumnCountMismatchHandlingAction): StorageParameter {
    return new StorageParameter('column_count_mismatch_handling', value);
  }

  /**
   * A property that sets the numRows value for the table definition. To explicitly update an external table's statistics, set the numRows property to indicate the size of the table. Amazon Redshift doesn't analyze external tables to generate the table statistics that the query optimizer uses to generate a query plan. If table statistics aren't set for an external table, Amazon Redshift generates a query execution plan based on an assumption that external tables are the larger tables and local tables are the smaller tables.
   */
  public static numRows(value: number): StorageParameter {
    return new StorageParameter('num_rows', value.toString());
  }

  /**
   * A property that sets number of rows to skip at the beginning of each source file.
   */
  public static serializationNullFormat(value: string): StorageParameter {
    return new StorageParameter('serialization.null.format', value);
  }

  /**
   * A property that sets the column mapping type for tables that use ORC data format. This property is ignored for other data formats. If this property is omitted, columns are mapped by `OrcColumnMappingType.NAME` by default.
   *
   * @default OrcColumnMappingType.NAME
   */
  public static orcSchemaResolution(value: OrcColumnMappingType): StorageParameter {
    return new StorageParameter('orc.schema.resolution', value);
  }

  /**
   * A property that sets whether CREATE EXTERNAL TABLE AS should write data in parallel. When 'write.parallel' is set to off, CREATE EXTERNAL TABLE AS writes to one or more data files serially onto Amazon S3. This table property also applies to any subsequent INSERT statement into the same external table.
   *
   * @default WriteParallel.ON
   */
  public static writeParallel(value: WriteParallel): StorageParameter {
    return new StorageParameter('write.parallel', value);
  }

  /**
   * A property that sets the maximum size (in MB) of each file written to Amazon S3 by CREATE EXTERNAL TABLE AS. The size must be a valid integer between 5 and 6200. The default maximum file size is 6,200 MB. This table property also applies to any subsequent INSERT statement into the same external table.
   */
  public static writeMaxFileSizeMb(value: number): StorageParameter {
    return new StorageParameter('write.maxfilesize.mb', value.toString());
  }

  /**
   * Enables server-side encryption (SSE-KMS) with the given AWS KMS key on the
   * files Redshift Spectrum writes for this table (via `CREATE EXTERNAL TABLE AS`
   * or `INSERT`).
   *
   * Redshift Spectrum accepts either the key's ARN or its bare key ID for the
   * `write.kms.key.id` property, and encrypts the written objects with that key;
   * this renders the ARN. To use the S3 bucket's default KMS key instead, set the
   * literal value `auto` via `StorageParameter.custom('write.kms.key.id', 'auto')` —
   * this typed factory cannot express `auto`.
   *
   * @see https://docs.aws.amazon.com/redshift/latest/dg/r_CREATE_EXTERNAL_TABLE.html
   */
  public static writeKmsKeyId(key: kms.IKeyRef): StorageParameter {
    return new StorageParameter('write.kms.key.id', key.keyRef.keyArn);
  }

  /**
   * A custom storage parameter.
   * @param key - The key of the storage parameter.
   * @param value - The value of the storage parameter.
   */
  public static custom(key: string, value: string): StorageParameter {
    return new StorageParameter(key, value);
  }

  protected constructor(public readonly key: string, public readonly value: string) {}
}
