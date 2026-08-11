import type { Construct, IDependable } from 'constructs';
import { Node } from 'constructs';
import type * as iam from '../../../aws-iam';
import type * as kms from '../../../aws-kms';
import * as logs from '../../../aws-logs';
import * as s3 from '../../../aws-s3';
import * as cdk from '../../../core';
import { lit } from '../../../core/lib/private/literal-string';
import { undefinedIfAllValuesAreEmpty } from '../../../core/lib/util';
import type { CommonDestinationProps, DestinationS3BackupProps } from '../common';
import type { CfnDeliveryStream } from '../kinesisfirehose.generated';
import type { ILoggingConfig } from '../logging-config';
import type { DataProcessorBindOptions, IDataProcessor } from '../processor';
import type { DynamicPartitioningProps } from '../s3-bucket';

export const PARTITION_KEY_QUERY = 'partitionKeyFromQuery';
export const PARTITION_KEY_LAMBDA = 'partitionKeyFromLambda';
export const ERROR_OUTPUT_TYPE = '!{firehose:error-output-type}';

export interface DestinationLoggingProps {
  /**
   * Configuration that determines whether to log errors during data transformation or delivery failures,
   * and specifies the CloudWatch log group for storing error logs.
   *
   * @default - errors will be logged and a log group will be created for you.
   */
  readonly loggingConfig?: ILoggingConfig;

  /**
   * The IAM role associated with this destination.
   */
  readonly role: iam.IRole;

  /**
   * The ID of the stream that is created in the log group where logs will be placed.
   *
   * Must be unique within the log group, so should be different every time this function is called.
   */
  readonly streamId: string;
}

interface ConfigWithDependables {
  /**
   * Resources that were created by the sub-config creator that must be deployed before the delivery stream is deployed.
   */
  readonly dependables: IDependable[];
}

export interface DestinationLoggingConfig extends ConfigWithDependables {
  /**
   * Logging options that will be injected into the destination configuration.
   */
  readonly loggingOptions: CfnDeliveryStream.CloudWatchLoggingOptionsProperty;
}

export interface DestinationBackupConfig extends ConfigWithDependables {
  /**
   * S3 backup configuration that will be injected into the destination configuration.
   */
  readonly backupConfig: CfnDeliveryStream.S3DestinationConfigurationProperty;
}

export function createLoggingOptions(scope: Construct, props: DestinationLoggingProps): DestinationLoggingConfig | undefined {
  if (props.loggingConfig?.logging !== false || props.loggingConfig?.logGroup) {
    const logGroup = props.loggingConfig?.logGroup ?? Node.of(scope).tryFindChild('LogGroup') as logs.ILogGroup ?? new logs.LogGroup(scope, 'LogGroup');
    const logGroupGrant = logGroup.grantWrite(props.role);
    return {
      loggingOptions: {
        enabled: true,
        logGroupName: logGroup.logGroupName,
        logStreamName: logGroup.addStream(props.streamId).logStreamName,
      },
      dependables: [logGroupGrant],
    };
  }
  return undefined;
}

export function createBufferingHints(
  scope: Construct,
  interval?: cdk.Duration,
  size?: cdk.Size,
  dataFormatConversionEnabled?: boolean,
  dynamicPartitioningEnabled?: boolean,
): CfnDeliveryStream.BufferingHintsProperty | undefined {
  if (!interval && !size && !dataFormatConversionEnabled && !dynamicPartitioningEnabled) {
    return undefined;
  }

  const intervalInSeconds = interval?.toSeconds() ?? 300;
  if (!cdk.Token.isUnresolved(intervalInSeconds)) {
    if (intervalInSeconds > 900) {
      throw new cdk.ValidationError(lit`BufferingIntervalTooLarge`, `Buffering interval must be less than 900 seconds, got ${intervalInSeconds} seconds.`, scope);
    }
    if (dynamicPartitioningEnabled && intervalInSeconds < 60) {
      // From testing, CFN deployment will fail if `BufferingHints.IntervalInSeconds` is less than 60.
      // The message is: "BufferingHints.IntervalInSeconds must be at least 60 seconds when Dynamic Partitioning is enabled."
      throw new cdk.ValidationError(lit`DynamicPartitioningBufferingIntervalTooSmall`, `When dynamic partitioning is enabled, buffering interval must be at least 60 seconds, got ${intervalInSeconds} seconds.`, scope);
    }
  }

  const defaultSizeInMBs = (dataFormatConversionEnabled || dynamicPartitioningEnabled) ? 128 : 5;
  const sizeInMBs = size?.toMebibytes() ?? defaultSizeInMBs;
  if (!cdk.Token.isUnresolved(sizeInMBs)) {
    if (sizeInMBs > 128) {
      throw new cdk.ValidationError(lit`BufferingSizeTooLarge`, `Buffering size must be at most 128 MiBs, got ${sizeInMBs} MiBs.`, scope);
    }
    if ((dataFormatConversionEnabled || dynamicPartitioningEnabled) && sizeInMBs < 64) {
      // From testing, CFN deployment will fail if `BufferingHints.SizeInMBs` is less than 64.
      // The message is: "BufferingHints.SizeInMBs must be at least 64 when Dynamic Partitioning is enabled."
      throw new cdk.ValidationError(lit`DataFormatConversionBufferingSizeTooSmall`, `When data format conversion or dynamic partitioning is enabled, buffering size must be at least 64 MiBs, got ${sizeInMBs} MiBs.`, scope);
    }
    if (sizeInMBs < 1) {
      throw new cdk.ValidationError(lit`BufferingSizeTooSmall`, `Buffering size must be at least 1 MiB, got ${sizeInMBs} MiBs.`, scope);
    }
  }

  return { intervalInSeconds, sizeInMBs };
}

export function createEncryptionConfig(
  role: iam.IRole,
  encryptionKey?: kms.IKey,
): CfnDeliveryStream.EncryptionConfigurationProperty | undefined {
  encryptionKey?.grantEncryptDecrypt(role);
  return encryptionKey
    ? { kmsEncryptionConfig: { awskmsKeyArn: encryptionKey.keyArn } }
    : undefined;
}

export function createProcessingConfig(
  scope: Construct,
  props: CommonDestinationProps,
  options: DataProcessorBindOptions,
): CfnDeliveryStream.ProcessingConfigurationProperty | undefined {
  if (props.processor && props.processors) {
    throw new cdk.ValidationError(lit`ProcessorAndProcessorsConflict`, "You can specify either 'processors' or 'processor', not both.", scope);
  }
  const processorsFromProps = props.processor ? [props.processor] : props.processors;
  const processors = (processorsFromProps ?? []).map((dp) => renderDataProcessor(dp, scope, options));
  const processorTypes = new Set(processors.map((p) => p.type));

  if (processorTypes.has('CloudWatchLogProcessing') && !processorTypes.has('Decompression')) {
    throw new cdk.ValidationError(lit`CloudWatchLogProcessorRequiresDecompression`, 'CloudWatchLogProcessor can only be enabled with DecompressionProcessor', scope);
  }
  if (options.dynamicPartitioningEnabled) {
    const withLambda = processorTypes.has('Lambda');
    const withInline = processorTypes.has('MetadataExtraction');
    // CFN validation message: "S3 Prefix should contain Dynamic Partitioning namespaces when Dynamic Partitioning is enabled"
    if (!options.prefix) {
      throw new cdk.ValidationError(lit`DynamicPartitioningRequiresDataOutputPrefix`, 'When dynamic partitioning is enabled, you must specify dataOutputPrefix.', scope);
    }
    // CFN validation message: "Processing Configuration is not enabled when DataPartitioning is enabled"
    if (!withLambda && !withInline) {
      throw new cdk.ValidationError(lit`DynamicPartitioningRequiresProcessors`, 'When dynamic partitioning is enabled, you must specify ether LambdaFunctionProcessor or MetadataExtractionProcessor, or both.', scope);
    }
    // CFN validation message: "S3 Prefix should contain Dynamic Partitioning namespaces when Dynamic Partitioning is enabled"
    if (withLambda && !withInline && !options.prefix.includes(`!{${PARTITION_KEY_LAMBDA}:`)) {
      throw new cdk.ValidationError(lit`DynamicPartitioningLambdaRequiresPartitionKey`, `When dynamic partitioning is enabled and the only LambdaFunctionProcessor is specified, you must specify at least one instance of !{${PARTITION_KEY_LAMBDA}:keyID}.`, scope);
    }
    // CFN validation message: "Lambda should be present when S3 Prefix contains keys from partitionKeyFromLambda namespace"
    if (!withLambda && options.prefix.includes(`!{${PARTITION_KEY_LAMBDA}:`)) {
      throw new cdk.ValidationError(lit`LambdaPartitionKeyRequiresProcessor`, `The dataOutputPrefix cannot contain !{${PARTITION_KEY_LAMBDA}:keyID} when LambdaFunctionProcessor is not specified.`, scope);
    }
    // CFN validation message: "MetadataExtraction processor should be present when S3 Prefix has partitionKeyFromQuery namespace"
    if (!withInline && options.prefix.includes(`!{${PARTITION_KEY_QUERY}:`)) {
      throw new cdk.ValidationError(lit`QueryPartitionKeyRequiresProcessor`, `The dataOutputPrefix cannot contain !{${PARTITION_KEY_QUERY}:keyID} when MetadataExtractionProcessor is not specified.`, scope);
    }
  }

  if (processors.length === 0) return undefined;

  return {
    enabled: true,
    processors,
  };
}

function renderDataProcessor(
  processor: IDataProcessor,
  scope: Construct,
  options: DataProcessorBindOptions,
): CfnDeliveryStream.ProcessorProperty {
  const processorConfig = processor.bind(scope, options);

  if (processorConfig.parameters) {
    return {
      type: processorConfig.processorType,
      parameters: processorConfig.parameters,
    };
  }

  const parameters = [{ parameterName: 'RoleArn', parameterValue: options.role.roleArn }];
  parameters.push(processorConfig.processorIdentifier);
  if (processor.props.bufferInterval) {
    parameters.push({ parameterName: 'BufferIntervalInSeconds', parameterValue: processor.props.bufferInterval.toSeconds().toString() });
  }
  if (processor.props.bufferSize) {
    parameters.push({ parameterName: 'BufferSizeInMBs', parameterValue: processor.props.bufferSize.toMebibytes().toString() });
  }
  if (processor.props.retries) {
    parameters.push({ parameterName: 'NumberOfRetries', parameterValue: processor.props.retries.toString() });
  }
  return {
    type: processorConfig.processorType,
    parameters,
  };
}

export function createBackupConfig(scope: Construct, role: iam.IRole, props: DestinationS3BackupProps & { mode: NonNullable<DestinationS3BackupProps['mode']> }): DestinationBackupConfig;
export function createBackupConfig(scope: Construct, role: iam.IRole, props?: DestinationS3BackupProps): DestinationBackupConfig | undefined;
export function createBackupConfig(scope: Construct, role: iam.IRole, props?: DestinationS3BackupProps): DestinationBackupConfig | undefined {
  if (!props || (props.mode === undefined && !props.bucket)) {
    return undefined;
  }

  const bucket = props.bucket ?? new s3.Bucket(scope, 'BackupBucket');
  const bucketGrant = bucket.grantReadWrite(role);

  const { loggingOptions, dependables: loggingDependables } = createLoggingOptions(scope, {
    loggingConfig: props.loggingConfig,
    role,
    streamId: 'S3Backup',
  }) ?? {};

  return {
    backupConfig: {
      bucketArn: bucket.bucketArn,
      roleArn: role.roleArn,
      prefix: props.dataOutputPrefix,
      errorOutputPrefix: props.errorOutputPrefix,
      bufferingHints: createBufferingHints(scope, props.bufferingInterval, props.bufferingSize),
      compressionFormat: props.compression?.value,
      encryptionConfiguration: createEncryptionConfig(role, props.encryptionKey),
      cloudWatchLoggingOptions: loggingOptions,
    },
    dependables: [bucketGrant, ...(loggingDependables ?? [])],
  };
}

export function createDynamicPartitioningConfiguration(
  scope: Construct,
  props?: DynamicPartitioningProps,
): CfnDeliveryStream.DynamicPartitioningConfigurationProperty | undefined {
  if (!props) return undefined;

  const durationInSeconds = props.retryDuration?.toSeconds();
  if (durationInSeconds != null && !cdk.Token.isUnresolved(durationInSeconds) && durationInSeconds > 7200) {
    throw new cdk.ValidationError(lit`RetryDurationTooLarge`, `Retry duration must be less than 7200 seconds, got ${durationInSeconds} seconds.`, scope);
  }

  return {
    enabled: props.enabled,
    retryOptions: undefinedIfAllValuesAreEmpty({
      durationInSeconds,
    }),
  };
}
