import type { CommonDestinationProps } from './common';
import {
  type HttpAttribute,
  type HttpBufferingHints,
  type HttpRetryOptions,
  HttpBackupMode,
  HttpCompression,
  HttpEndpoint,
} from './http-endpoint';
import type { ISecret } from '../../aws-secretsmanager';
import { Duration, Size } from '../../core';

/**
 * A Datadog endpoint URL for use with Kinesis Data Firehose.
 *
 * Use one of the predefined static members for a known Datadog region, or
 * `DatadogEndpoint.of(url)` for a custom URL.
 */
export class DatadogEndpoint {
  /**
   * Logs — US1
   */
  public static readonly LOGS_US1 = new DatadogEndpoint('https://aws-kinesis-http-intake.logs.datadoghq.com/v1/input');
  /**
   * Logs — US3
   */
  public static readonly LOGS_US3 = new DatadogEndpoint('https://aws-kinesis-http-intake.logs.us3.datadoghq.com/api/v2/logs?dd-protocol=aws-kinesis-firehose');
  /**
   * Logs — US5
   */
  public static readonly LOGS_US5 = new DatadogEndpoint('https://aws-kinesis-http-intake.logs.us5.datadoghq.com/api/v2/logs?dd-protocol=aws-kinesis-firehose');
  /**
   * Logs — AP1 (Japan)
   */
  public static readonly LOGS_AP1 = new DatadogEndpoint('https://aws-kinesis-http-intake.logs.ap1.datadoghq.com/api/v2/logs?dd-protocol=aws-kinesis-firehose');
  /**
   * Logs — EU
   */
  public static readonly LOGS_EU = new DatadogEndpoint('https://aws-kinesis-http-intake.logs.datadoghq.eu/v1/input');
  /**
   * Logs — US Government
   */
  public static readonly LOGS_GOV = new DatadogEndpoint('https://aws-kinesis-http-intake.logs.ddog-gov.com/v1/input');

  /**
   * Metrics — US
   */
  public static readonly METRICS_US = new DatadogEndpoint('https://awsmetrics-intake.datadoghq.com/v1/input');
  /**
   * Metrics — US5
   */
  public static readonly METRICS_US5 = new DatadogEndpoint('https://event-platform-intake.us5.datadoghq.com/api/v2/awsmetrics?dd-protocol=aws-kinesis-firehose');
  /**
   * Metrics — AP1 (Japan)
   */
  public static readonly METRICS_AP1 = new DatadogEndpoint('https://event-platform-intake.ap1.datadoghq.com/api/v2/awsmetrics?dd-protocol=aws-kinesis-firehose');
  /**
   * Metrics — EU
   */
  public static readonly METRICS_EU = new DatadogEndpoint('https://awsmetrics-intake.datadoghq.eu/v1/input');

  /**
   * Configurations — US1
   */
  public static readonly CONFIGURATION_US1 = new DatadogEndpoint('https://cloudplatform-intake.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose');
  /**
   * Configurations — US3
   */
  public static readonly CONFIGURATION_US3 = new DatadogEndpoint('https://cloudplatform-intake.us3.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose');
  /**
   * Configurations — US5
   */
  public static readonly CONFIGURATION_US5 = new DatadogEndpoint('https://cloudplatform-intake.us5.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose');
  /**
   * Configurations — AP1 (Japan)
   */
  public static readonly CONFIGURATION_AP1 = new DatadogEndpoint('https://cloudplatform-intake.ap1.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose');
  /**
   * Configurations — EU
   */
  public static readonly CONFIGURATION_EU = new DatadogEndpoint('https://cloudplatform-intake.datadoghq.eu/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose');
  /**
   * Configurations — US Government
   */
  public static readonly CONFIGURATION_US_GOV = new DatadogEndpoint('https://cloudplatform-intake.ddog-gov.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose');

  /**
   * Use a custom Datadog endpoint URL.
   *
   * This is an escape hatch for endpoints not covered by the predefined members (for
   * example a new region or a proxy in front of Datadog). The value is passed through
   * as-is and is not validated by CDK. It must be a valid HTTPS Datadog intake URL: the
   * Firehose API requires the endpoint URL to match `https://`, so an invalid value fails
   * at deployment. The caller is responsible for providing a correct URL.
   *
   * @param url The full HTTPS endpoint URL.
   */
  public static of(url: string): DatadogEndpoint {
    return new DatadogEndpoint(url);
  }

  private constructor(
    /**
     * The endpoint URL string.
     */
    public readonly url: string,
  ) {}
}

/**
 * Props for defining a Datadog destination of a Kinesis Data Firehose delivery stream.
 */
export interface DatadogProps extends CommonDestinationProps {
  /**
   * The API key used to authenticate with Datadog.
   *
   * Delivered to Firehose through AWS Secrets Manager (Firehose retrieves it at runtime rather
   * than embedding it in the template).
   */
  readonly apiKey: ISecret;
  /**
   * The Datadog endpoint to send data to.
   */
  readonly endpoint: DatadogEndpoint;
  /**
   * Datadog tags to apply for filtering.
   *
   * @default - No tags.
   */
  readonly tags?: HttpAttribute[];
  /**
   * Describes the S3 bucket backup options for the data that Kinesis Data Firehose delivers to Datadog.
   *
   * @default HttpBackupMode.FAILED
   */
  readonly backupMode?: HttpBackupMode;
  /**
   * Buffering hints for data delivery to the Datadog endpoint.
   *
   * @default - interval of 60 seconds, size of 4 MiB
   */
  readonly bufferingHints?: HttpBufferingHints;
  /**
   * Retry behavior when Kinesis Data Firehose cannot deliver data to Datadog.
   *
   * @default - duration of 60 seconds
   */
  readonly retryOptions?: HttpRetryOptions;
  /**
   * Content encoding applied to the request body before delivery.
   *
   * @default HttpCompression.GZIP
   */
  readonly requestCompression?: HttpCompression;
}

/**
 * A Datadog destination for data from a Kinesis Data Firehose delivery stream.
 */
export class Datadog extends HttpEndpoint {
  constructor(props: DatadogProps) {
    super({
      ...props,
      endpointConfig: {
        url: props.endpoint.url,
        secret: props.apiKey,
      },
      requestCompression: props.requestCompression ?? HttpCompression.GZIP,
      bufferingHints: props.bufferingHints ?? {
        interval: Duration.seconds(60),
        size: Size.mebibytes(4),
      },
      retryOptions: props.retryOptions ?? { duration: Duration.seconds(60) },
      backupMode: props.backupMode ?? HttpBackupMode.FAILED,
      attributes: props.tags ?? [],
    });
  }
}
