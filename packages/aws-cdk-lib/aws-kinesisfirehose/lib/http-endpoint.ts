import type { Construct } from 'constructs';
import {
  BackupMode as S3BackupMode,
  type CommonDestinationProps,
} from './common';
import type {
  DestinationBindOptions,
  DestinationConfig,
  IDestination,
} from './destination';
import * as iam from '../../aws-iam';
import type { ISecret } from '../../aws-secretsmanager';
import {
  createBackupConfig,
  createBufferingHints,
  createLoggingOptions,
  createProcessingConfig,
} from './private/helpers';
import * as cdk from '../../core';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Kinesis Data Firehose uses the content encoding to compress the body of a request before sending the request to the destination.
 */
export enum HttpCompression {
  /**
   * GZIP
   */
  GZIP = 'GZIP',
  /**
   * NONE
   */
  NONE = 'NONE',
}

/**
 * Describes the S3 bucket backup options for the data that Kinesis Data Firehose delivers to the Http endpoint destination.
 */
export enum HttpBackupMode {
  /**
   * Back up only the documents that Kinesis Data Firehose could not deliver.
   */
  FAILED = 'FailedDataOnly',
  /**
   * Back up all documents.
   */
  ALL = 'AllData',
}

/**
 * The buffering options that can be used before data is delivered to the specified destination.
 */
export interface HttpBufferingHints {
  /**
   * The higher interval allows more time to collect data and the size of data may be bigger. The lower interval sends the data more frequently and may be more advantageous when looking at shorter cycles of data activity.
   * @default 300 seconds
   */
  readonly interval?: cdk.Duration;
  /**
   * The higher buffer size may be lower in cost with higher latency. The lower buffer size will be faster in delivery with higher cost and less latency.
   * @default 5 MiB
   */
  readonly size?: cdk.Size;
}

/**
 * Describes the retry behavior in case Kinesis Data Firehose is unable to deliver data to the specified Http endpoint destination, or if it doesn't receive a valid acknowledgment of receipt from the specified Http endpoint destination.
 */
export interface HttpRetryOptions {
  /**
   * The total amount of time that Kinesis Data Firehose spends on retries.
   */
  readonly duration: cdk.Duration;
}

/**
 * Describes the configuration of the Http endpoint to which Kinesis Firehose delivers data.
 */
export interface HttpEndpointConfig {
  /**
   * The URL of the Http endpoint selected as the destination.
   */
  readonly url: string;
  /**
   * The access key used to authenticate with the Http endpoint.
   *
   * Used only when `secret` is not set. If both `accessKey` and `secret` are provided,
   * `secret` (AWS Secrets Manager) takes precedence and this value is ignored. The access
   * key is rendered into the CloudFormation template.
   *
   * @default - none; authentication uses `secret` if provided, otherwise no access key
   */
  readonly accessKey?: cdk.SecretValue;
  /**
   * A Secrets Manager secret used to authenticate with the Http endpoint.
   *
   * When set, Firehose retrieves the credential from AWS Secrets Manager, and this takes
   * precedence over `accessKey`.
   *
   * @default - none; `accessKey` is used if provided
   */
  readonly secret?: ISecret;
  /**
   * The name of the Http endpoint selected as the destination.
   * @default - None
   */
  readonly name?: string;
}

/**
 * Describes the metadata sent to the Http endpoint destination.
 */
export interface HttpAttribute {
  /**
   * The name of the Http endpoint common attribute.
   */
  readonly name: string;
  /**
   * The value of the Http endpoint common attribute.
   */
  readonly value: string;
}

/**
 * Props for defining an Http destination of a Kinesis Data Firehose delivery stream.
 */
export interface HttpEndpointProps extends CommonDestinationProps {
  /**
   * Describes the configuration of the Http endpoint to which Kinesis Firehose delivers data.
   */
  readonly endpointConfig: HttpEndpointConfig;
  /**
   * Describes the S3 bucket backup options for the data that Kinesis Data Firehose delivers to the Http endpoint destination.
   * @default - Failed data only
   */
  readonly backupMode?: HttpBackupMode;
  /**
   * Compress the body of a request before sending the request to the destination.
   * @default - None
   */
  readonly requestCompression?: HttpCompression;
  /**
   * The buffering options that can be used before data is delivered to the specified destination.
   * @default - None
   */
  readonly bufferingHints?: HttpBufferingHints;
  /**
   * The total amount of time that Kinesis Data Firehose spends on retries
   * @default - None
   */
  readonly retryOptions?: HttpRetryOptions;
  /**
   * Describes the metadata sent to the Http endpoint destination.
   * @default - None
   */
  readonly attributes?: HttpAttribute[];
}

/**
 *  An Http destination for data from a Kinesis Data Firehose delivery stream.
 */
export class HttpEndpoint implements IDestination {
  constructor(private readonly props: HttpEndpointProps) {}
  bind(scope: Construct, _options: DestinationBindOptions): DestinationConfig {
    const role =
      this.props.role ??
      new iam.Role(scope, 'Http Destination Role', {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com', {
          conditions: {
            StringEquals: {
              'aws:SourceAccount': cdk.Stack.of(scope).account,
            },
            ArnLike: {
              'aws:SourceArn': cdk.Stack.of(scope).formatArn({
                service: 'firehose',
                resource: 'deliverystream',
                resourceName: '*',
              }),
            },
          },
        }),
      });

    const { loggingOptions, dependables: loggingDependables } =
      createLoggingOptions(scope, {
        loggingConfig: this.props.loggingConfig,
        role,
        streamId: 'HttpDestination',
      }) ?? {};

    const { backupConfig, dependables: backupDependables } = createBackupConfig(
      scope,
      role,
      {
        ...this.props.s3Backup,
        mode: this.props.backupMode === HttpBackupMode.ALL ? S3BackupMode.ALL : S3BackupMode.FAILED,
      },
    );

    if (this.props.endpointConfig.secret) {
      this.props.endpointConfig.secret.grantRead(role);
    }

    if (this.props.retryOptions) {
      const durationInSeconds = this.props.retryOptions.duration.toSeconds();
      if (!cdk.Token.isUnresolved(durationInSeconds) && durationInSeconds > 7200) {
        throw new cdk.ValidationError(lit`HttpRetryDurationTooLarge`, `Retry duration must be at most 7200 seconds, got ${durationInSeconds} seconds.`, scope);
      }
    }

    return {
      httpEndpointDestinationConfiguration: {
        endpointConfiguration: {
          url: this.props.endpointConfig.url,
          ...(this.props.endpointConfig.accessKey && {
            accessKey: this.props.endpointConfig.accessKey.unsafeUnwrap(),
          }),
          ...(this.props.endpointConfig.name && {
            name: this.props.endpointConfig.name,
          }),
        },
        ...(this.props.retryOptions && {
          retryOptions: {
            durationInSeconds: this.props.retryOptions.duration.toSeconds(),
          },
        }),
        bufferingHints: createBufferingHints(scope, this.props.bufferingHints?.interval, this.props.bufferingHints?.size),
        requestConfiguration: {
          contentEncoding:
						this.props.requestCompression ?? HttpCompression.NONE,
          ...(this.props.attributes && {
            commonAttributes: [
              ...this.props.attributes.map((attr) => ({
                attributeName: attr.name,
                attributeValue: attr.value,
              })),
            ],
          }),
        },
        cloudWatchLoggingOptions: loggingOptions,
        processingConfiguration: createProcessingConfig(
          scope,
          this.props,
          { role },
        ),
        roleArn: role.roleArn,
        s3BackupMode: this.props.backupMode ?? HttpBackupMode.FAILED,
        s3Configuration: backupConfig,
        ...(this.props.endpointConfig.secret && {
          secretsManagerConfiguration: {
            secretArn: this.props.endpointConfig.secret.secretArn,
            enabled: true,
            roleArn: role.roleArn,
          },
        }),
      },
      dependables: [
        ...(loggingDependables ?? []),
        ...backupDependables,
      ],
    };
  }
}
