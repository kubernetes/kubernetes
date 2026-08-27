import type { Construct } from 'constructs';
import type { CommonDestinationProps, CommonDestinationS3Props } from './common';
import { BackupMode } from './common';
import type { DestinationBindOptions, DestinationConfig, IDestination } from './destination';
import type { IInputFormat, IOutputFormat, SchemaConfiguration } from './record-format';
import * as iam from '../../aws-iam';
import type * as s3 from '../../aws-s3';
import { createBackupConfig, createBufferingHints, createDynamicPartitioningConfiguration, createEncryptionConfig, createLoggingOptions, createProcessingConfig, createTimezoneName, ERROR_OUTPUT_TYPE, PARTITION_KEY_LAMBDA, PARTITION_KEY_QUERY } from './private/helpers';
import * as cdk from '../../core';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Props for defining an S3 destination of an Amazon Data Firehose delivery stream.
 */
export interface S3BucketProps extends CommonDestinationS3Props, CommonDestinationProps {
  /**
   * Specify a file extension.
   * It will override the default file extension appended by Data Format Conversion or S3 compression features such as `.parquet` or `.gz`.
   *
   * File extension must start with a period (`.`) and can contain allowed characters: `0-9a-z!-_.*'()`.
   *
   * @see https://docs.aws.amazon.com/firehose/latest/dev/create-destination.html#create-destination-s3
   * @default - The default file extension appended by Data Format Conversion or S3 compression features
   */
  readonly fileExtension?: string;

  /**
   * The time zone you prefer.
   *
   * AWS Kinesis Data Firehose supports standard IANA time zone identifiers (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo').
   *
   * @see https://docs.aws.amazon.com/firehose/latest/dev/s3-object-name.html
   * @default - UTC
   */
  readonly timeZone?: cdk.TimeZone;

  /**
   * The input format, output format, and schema config for converting data from the JSON format to the Parquet or ORC format before writing to Amazon S3.
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-kinesisfirehose-deliverystream-extendeds3destinationconfiguration.html#cfn-kinesisfirehose-deliverystream-extendeds3destinationconfiguration-dataformatconversionconfiguration
   *
   * @default - no data format conversion is done
   */
  readonly dataFormatConversion?: DataFormatConversionProps;

  /**
   * Specify dynamic partitioning.
   * @see https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning.html
   * @default - Dynamic partitioning is disabled.
   */
  readonly dynamicPartitioning?: DynamicPartitioningProps;
}

/**
 * Props for specifying data format conversion for Firehose
 *
 * @see https://docs.aws.amazon.com/firehose/latest/dev/record-format-conversion.html
 */
export interface DataFormatConversionProps {
  /**
   * Whether data format conversion is enabled or not.
   *
   * @default `true`
   */
  readonly enabled?: boolean;

  /**
   * The schema configuration to use in converting the input format to output format
   */
  readonly schemaConfiguration: SchemaConfiguration;

  /**
   * The input format to convert from for record format conversion
   */
  readonly inputFormat: IInputFormat;

  /**
   * The output format to convert to for record format conversion
   */
  readonly outputFormat: IOutputFormat;
}

/**
 * Props for defining dynamic partitioning.
 *
 * @see https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning.html
 */
export interface DynamicPartitioningProps {
  /**
   * Whether to enable the dynamic partitioning.
   *
   * You cannot enable dynamic partitioning for an existing Firehose stream that does not have dynamic partitioning already enabled.
   *
   * @see https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning-enable.html
   */
  readonly enabled: boolean;

  /**
   * The total amount of time that Data Firehose spends on retries.
   *
   * Minimum: Duration.seconds(0)
   * Maximum: Duration.seconds(7200)
   *
   * @default Duration.seconds(300)
   */
  readonly retryDuration?: cdk.Duration;
}

/**
 * An S3 bucket destination for data from an Amazon Data Firehose delivery stream.
 */
export class S3Bucket implements IDestination {
  constructor(private readonly bucket: s3.IBucket, private readonly props: S3BucketProps = {}) {
    if (this.props.s3Backup?.mode === BackupMode.FAILED) {
      throw new cdk.UnscopedValidationError(lit`S3BackupModeFailedNotSupported`, 'S3 destinations do not support BackupMode.FAILED');
    }

    if (this.props.dataFormatConversion && this.props.compression) {
      throw new cdk.UnscopedValidationError(lit`DataFormatConversionCompressionConflict`, 'When data record format conversion is enabled, compression cannot be set on the S3 Destination. Compression may only be set in the OutputFormat. By default, this compression is SNAPPY');
    }

    validateOutputPrefix(this.props.dataOutputPrefix, this.props.errorOutputPrefix);
    // CFN validation message: "Dynamic Partitioning Namespaces can only be part of a prefix expression when Dynamic Partitioning is enabled."
    if (
      !this.props.dynamicPartitioning?.enabled &&
      (this.props.dataOutputPrefix?.includes(`!{${PARTITION_KEY_LAMBDA}:`) || this.props.dataOutputPrefix?.includes(`!{${PARTITION_KEY_QUERY}:`))
    ) {
      throw new cdk.UnscopedValidationError(lit`DynamicPartitioningNamespaceWithoutEnabled`, `When dynamic partitioning is not enabled, the dataOutputPrefix cannot contain neither ${PARTITION_KEY_LAMBDA} nor ${PARTITION_KEY_QUERY}.`);
    }
  }

  bind(scope: Construct, _options: DestinationBindOptions): DestinationConfig {
    const role = this.props.role ?? new iam.Role(scope, 'S3 Destination Role', {
      assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
    });

    const bucketGrant = this.bucket.grantReadWrite(role);

    const { loggingOptions, dependables: loggingDependables } = createLoggingOptions(scope, {
      loggingConfig: this.props.loggingConfig,
      role,
      streamId: 'S3Destination',
    }) ?? {};

    const { backupConfig, dependables: backupDependables } = createBackupConfig(scope, role, this.props.s3Backup) ?? {};

    const fileExtension = this.props.fileExtension;
    if (fileExtension && !cdk.Token.isUnresolved(fileExtension)) {
      if (!fileExtension.startsWith('.')) {
        throw new cdk.ValidationError(lit`FileExtensionMustStartWithPeriod`, "fileExtension must start with '.'", scope);
      }
      if (/[^0-9a-z!\-_.*'()]/.test(fileExtension)) {
        throw new cdk.ValidationError(lit`FileExtensionInvalidCharacters`, "fileExtension can contain allowed characters: 0-9a-z!-_.*'()", scope);
      }
    }

    const dataFormatConfig = this.props.dataFormatConversion;

    const dataFormatConversionConfiguration = dataFormatConfig ? {
      enabled: dataFormatConfig.enabled ?? true,
      schemaConfiguration: dataFormatConfig.schemaConfiguration.bind(scope, { role: role }),
      inputFormatConfiguration: dataFormatConfig.inputFormat.createInputFormatConfig(),
      outputFormatConfiguration: dataFormatConfig.outputFormat.createOutputFormatConfig(),
    } : undefined;

    return {
      extendedS3DestinationConfiguration: {
        cloudWatchLoggingOptions: loggingOptions,
        processingConfiguration: createProcessingConfig(scope, this.props, {
          role,
          dynamicPartitioningEnabled: this.props.dynamicPartitioning?.enabled,
          prefix: this.props.dataOutputPrefix,
        }),
        roleArn: role.roleArn,
        s3BackupConfiguration: backupConfig,
        s3BackupMode: this.getS3BackupMode(),
        bufferingHints: createBufferingHints(scope, this.props.bufferingInterval, this.props.bufferingSize,
          dataFormatConversionConfiguration?.enabled, this.props.dynamicPartitioning?.enabled),
        bucketArn: this.bucket.bucketArn,
        dataFormatConversionConfiguration: dataFormatConversionConfiguration,
        compressionFormat: this.props.compression?.value,
        encryptionConfiguration: createEncryptionConfig(role, this.props.encryptionKey),
        errorOutputPrefix: this.props.errorOutputPrefix,
        prefix: this.props.dataOutputPrefix,
        fileExtension: this.props.fileExtension,
        customTimeZone: createTimezoneName(scope, this.props.timeZone),
        dynamicPartitioningConfiguration: createDynamicPartitioningConfiguration(scope, this.props.dynamicPartitioning),
      },
      dependables: [bucketGrant, ...(loggingDependables ?? []), ...(backupDependables ?? [])],
    };
  }

  private getS3BackupMode(): string | undefined {
    return this.props.s3Backup?.bucket || this.props.s3Backup?.mode === BackupMode.ALL
      ? 'Enabled'
      : undefined;
  }
}

/**
 * Validates output prefixes
 * @see https://docs.aws.amazon.com/firehose/latest/dev/s3-prefixes.html#prefix-rules
 */
function validateOutputPrefix(prefix?: string, errorOutputPrefix?: string) {
  // The sequence !{ can only appear in !{namespace:value} expressions.
  if (prefix) validateOutputPrefixExpression(prefix, 'dataOutputPrefix');
  if (errorOutputPrefix) validateOutputPrefixExpression(errorOutputPrefix, 'errorOutputPrefix');
  // ErrorOutputPrefix can be null only if Prefix contains no expressions.
  if (prefix?.includes('!{') && !errorOutputPrefix) {
    throw new cdk.UnscopedValidationError(lit`ErrorOutputPrefixRequiredWithExpressions`, 'Specify the errorOutputPrefix in order to use expressions in the dataOutputPrefix.');
  }
  // If you specify an expression for ErrorOutputPrefix, you must include at least one instance of !{firehose:error-output-type}.
  if (errorOutputPrefix?.includes('!{') && !errorOutputPrefix.includes(ERROR_OUTPUT_TYPE)) {
    throw new cdk.UnscopedValidationError(lit`ErrorOutputPrefixMustIncludeErrorOutputType`, `The errorOutputPrefix expression must include at least one instance of ${ERROR_OUTPUT_TYPE}.`);
  }
  // Prefix can't contain !{firehose:error-output-type}.
  if (prefix?.includes(ERROR_OUTPUT_TYPE)) {
    throw new cdk.UnscopedValidationError(lit`DataOutputPrefixCannotContainErrorOutputType`, `The dataOutputPrefix cannot contain ${ERROR_OUTPUT_TYPE}.`);
  }
  // You cannot use partitionKeyFromLambda and partitionKeyFromQuery namespaces when creating ErrorOutputPrefix expressions.
  if (errorOutputPrefix?.includes(`!{${PARTITION_KEY_LAMBDA}:`) || errorOutputPrefix?.includes(`!{${PARTITION_KEY_QUERY}:`)) {
    throw new cdk.UnscopedValidationError(lit`ErrorOutputPrefixCannotUsePartitionKeyNamespaces`, `You cannot use ${PARTITION_KEY_LAMBDA} and ${PARTITION_KEY_QUERY} namespaces in errorOutputPreix.`);
  }
}

function validateOutputPrefixExpression(prefix: string, prop: string) {
  if (/!\{(?!(?:firehose|timestamp|partitionKeyFrom(?:Lambda|Query)):[^{}]+\})/.test(prefix)) {
    throw new cdk.UnscopedValidationError(lit`InvalidExpressionFormat`, `The expression must be of the form !{namespace:value} and include a valid namespace at ${prop}.`);
  }
}
