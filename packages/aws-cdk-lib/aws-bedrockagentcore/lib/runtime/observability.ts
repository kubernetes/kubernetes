import { Construct } from 'constructs';
import { Effect, PolicyStatement, ServicePrincipal, PolicyDocument } from '../../../aws-iam';
import type { IDeliveryStream } from '../../../aws-kinesisfirehose';
import * as logs from '../../../aws-logs';
import type * as s3 from '../../../aws-s3';
import * as xray from '../../../aws-xray';
import { ArnFormat, Lazy, Names, Stack, Tags } from '../../../core';

/**
 * Maximum length for delivery source and destination names
 */
const MAX_DELIVERY_NAME_LENGTH = 60;

/**
 * Log types for AgentCore Runtime observability
 */
export class LogType {
  /**
   * Application logs for agent runtime invocations
   */
  public static readonly APPLICATION_LOGS = new LogType('APPLICATION_LOGS');

  /**
   * Usage logs for session-level resource consumption
   */
  public static readonly USAGE_LOGS = new LogType('USAGE_LOGS');

  /**
   * A custom log type value
   *
   * @param value The log type value
   */
  public static of(value: string): LogType {
    return new LogType(value);
  }

  private constructor(
    /**
     * The log type value
     */
    public readonly value: string,
  ) {}
}

/**
 * Configuration for logging with log type and destination
 */
export interface LoggingConfig {
  /**
   * The type of logs to deliver
   */
  readonly logType: LogType;

  /**
   * The destination for logs
   */
  readonly destination: LoggingDestination;
}

/**
 * Configuration returned by LoggingDestination.bind()
 * @internal
 */
interface LoggingDestinationBindConfig {
  /**
   * The delivery destination construct
   */
  readonly deliveryDestination: logs.CfnDeliveryDestination;
}

/**
 * Options for configuring runtime observability delivery.
 * @internal
 */
export interface RuntimeObservabilityOptions {
  /**
   * Whether to create resource policies for log/trace delivery.
   *
   * When `false`, the `AWS::Logs::ResourcePolicy` and `AWS::XRay::ResourcePolicy`
   * are not created. This is useful when deploying many runtimes per account/Region,
   * as each resource policy consumes an account-level quota slot (CloudWatch Logs: 10,
   * X-Ray: lower). For same-account `/aws/vendedlogs/` delivery, the log-delivery
   * service-linked role provides the necessary write access without an explicit policy.
   *
   * @default true
   */
  readonly manageDeliveryResourcePolicy?: boolean;
}

/**
 * Represents a logging destination for AgentCore Runtime
 *
 * Use the static factory methods to create instances:
 * - `LoggingDestination.cloudWatchLogs(logGroup)` - Send logs to CloudWatch Logs
 * - `LoggingDestination.s3(bucket)` - Send logs to S3
 * - `LoggingDestination.firehose(stream)` - Send logs to Kinesis Data Firehose
 */
export abstract class LoggingDestination {
  /**
   * Create a logging destination that sends logs to a CloudWatch Log Group
   *
   * @param logGroup The CloudWatch Log Group to send logs to
   */
  public static cloudWatchLogs(logGroup: logs.ILogGroup): LoggingDestination {
    return new CloudWatchLogsDestination(logGroup);
  }

  /**
   * Create a logging destination that sends logs to an S3 bucket
   *
   * @param bucket The S3 bucket to send logs to
   */
  public static s3(bucket: s3.IBucket): LoggingDestination {
    return new S3Destination(bucket);
  }

  /**
   * Create a logging destination that sends logs to a Kinesis Data Firehose delivery stream
   *
   * @param stream The Firehose delivery stream to send logs to
   */
  public static firehose(stream: IDeliveryStream): LoggingDestination {
    return new FirehoseDestination(stream);
  }

  /**
   * Bind this destination to a scope and create the delivery destination resource
   *
   * @param scope The construct scope
   * @param id The construct id
   * @param options Observability options
   * @internal
   */
  public abstract _bind(scope: Construct, id: string, options: RuntimeObservabilityOptions): LoggingDestinationBindConfig;
}

/**
 * CloudWatch Logs destination implementation
 */
class CloudWatchLogsDestination extends LoggingDestination {
  constructor(private readonly logGroup: logs.ILogGroup) {
    super();
  }

  public _bind(scope: Construct, id: string, options: RuntimeObservabilityOptions): LoggingDestinationBindConfig {
    const stack = Stack.of(scope);

    const managePolicy = options.manageDeliveryResourcePolicy ?? true;

    let resourcePolicy: logs.ResourcePolicy | undefined;

    if (managePolicy) {
      // Get or create a singleton resource policy for logs delivery
      const policyId = 'CdkLogGroupLogsDeliveryPolicy';
      resourcePolicy = stack.node.tryFindChild(policyId) as logs.ResourcePolicy | undefined;

      if (!resourcePolicy) {
        resourcePolicy = new logs.ResourcePolicy(stack, policyId);
      }

      // Grant permissions for this specific log group
      // @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-infrastructure-V2-CloudWatchLogs.html
      resourcePolicy.document.addStatements(new PolicyStatement({
        effect: Effect.ALLOW,
        principals: [new ServicePrincipal('delivery.logs.amazonaws.com')],
        actions: ['logs:CreateLogStream', 'logs:PutLogEvents'],
        resources: [`${this.logGroup.logGroupArn}:log-stream:*`],
        conditions: {
          StringEquals: { 'aws:SourceAccount': stack.account },
          ArnLike: {
            'aws:SourceArn': stack.formatArn({
              service: 'logs',
              resource: '*',
            }),
          },
        },
      }));
    }

    const deliveryDestination = new logs.CfnDeliveryDestination(scope, `${id}Dest`, {
      name: Lazy.string({ produce: (): string => Names.uniqueResourceName(deliveryDestination, { maxLength: MAX_DELIVERY_NAME_LENGTH }) }),
      deliveryDestinationType: 'CWL',
      destinationResourceArn: this.logGroup.logGroupArn,
    });

    if (resourcePolicy) {
      deliveryDestination.node.addDependency(resourcePolicy);
    }

    return { deliveryDestination };
  }
}

/**
 * S3 destination implementation
 */
class S3Destination extends LoggingDestination {
  constructor(private readonly bucket: s3.IBucket) {
    super();
  }

  public _bind(scope: Construct, id: string, _options: RuntimeObservabilityOptions): LoggingDestinationBindConfig {
    const stack = Stack.of(scope);

    // Add bucket policy for logs delivery
    // @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-infrastructure-V2-S3.html
    this.bucket.addToResourcePolicy(new PolicyStatement({
      effect: Effect.ALLOW,
      principals: [new ServicePrincipal('delivery.logs.amazonaws.com')],
      actions: ['s3:PutObject'],
      resources: [`${this.bucket.bucketArn}/AWSLogs/${stack.account}/*`],
      conditions: {
        StringEquals: {
          's3:x-amz-acl': 'bucket-owner-full-control',
          'aws:SourceAccount': stack.account,
        },
        ArnLike: {
          'aws:SourceArn': stack.formatArn({
            service: 'logs',
            resource: 'delivery-source',
            resourceName: '*',
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        },
      },
    }));

    const deliveryDestination = new logs.CfnDeliveryDestination(scope, `${id}Dest`, {
      name: Lazy.string({ produce: (): string => Names.uniqueResourceName(deliveryDestination, { maxLength: MAX_DELIVERY_NAME_LENGTH }) }),
      deliveryDestinationType: 'S3',
      destinationResourceArn: this.bucket.bucketArn,
    });

    return { deliveryDestination };
  }
}

/**
 * Firehose destination implementation
 */
class FirehoseDestination extends LoggingDestination {
  constructor(private readonly stream: IDeliveryStream) {
    super();
  }

  public _bind(scope: Construct, id: string, _options: RuntimeObservabilityOptions): LoggingDestinationBindConfig {
    // Firehose uses a service-linked role that requires this tag to grant access
    // @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-infrastructure-V2-Firehose.html
    Tags.of(this.stream).add('LogDeliveryEnabled', 'true');

    const deliveryDestination = new logs.CfnDeliveryDestination(scope, `${id}Dest`, {
      name: Lazy.string({ produce: (): string => Names.uniqueResourceName(deliveryDestination, { maxLength: MAX_DELIVERY_NAME_LENGTH }) }),
      deliveryDestinationType: 'FH',
      destinationResourceArn: this.stream.deliveryStreamArn,
    });

    return { deliveryDestination };
  }
}

/**
 * Internal X-Ray resource policy wrapper that allows adding statements after creation.
 * This is similar to logs.ResourcePolicy but for X-Ray.
 */
class XRayResourcePolicy extends Construct {
  public readonly document: PolicyDocument;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.document = new PolicyDocument();

    new xray.CfnResourcePolicy(this, 'ResourcePolicy', {
      policyName: Lazy.string({ produce: () => Names.uniqueResourceName(this, { maxLength: 128 }) }),
      policyDocument: Lazy.string({ produce: () => JSON.stringify(this.document.toJSON()) }),
    });
  }
}

/**
 * Configure X-Ray tracing delivery for a runtime
 *
 * @param scope The construct scope
 * @param sourceArn The ARN of the source resource (runtime)
 * @param options Observability options
 * @internal
 */
export function configureTracingDelivery(
  scope: Construct,
  sourceArn: string,
  options: RuntimeObservabilityOptions = {},
): logs.CfnDelivery {
  const stack = Stack.of(scope);

  // Create delivery source for traces
  const deliverySource = new logs.CfnDeliverySource(scope, 'TracesDeliverySource', {
    name: Lazy.string({ produce: (): string => Names.uniqueResourceName(deliverySource, { maxLength: MAX_DELIVERY_NAME_LENGTH }) }),
    logType: 'TRACES',
    resourceArn: sourceArn,
  });

  const managePolicy = options.manageDeliveryResourcePolicy ?? true;

  let xrayPolicy: XRayResourcePolicy | undefined;

  if (managePolicy) {
    // Get or create X-Ray resource policy (singleton per stack)
    const policyId = 'CdkXRayLogsDeliveryPolicy';
    xrayPolicy = stack.node.tryFindChild(policyId) as XRayResourcePolicy | undefined;

    if (!xrayPolicy) {
      xrayPolicy = new XRayResourcePolicy(stack, policyId);
    }

    // Grant permissions for this specific source resource
    // The xray:PutTraceSegments action requires resources: ['*'] per AWS documentation.
    // The conditions below restrict this broad scope to only the specific source resource
    // (via logs:LogGeneratingResourceArns) and the current account and delivery-source ARN.
    // @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-infrastructure-V2-XRAY.html
    xrayPolicy.document.addStatements(new PolicyStatement({
      effect: Effect.ALLOW,
      principals: [new ServicePrincipal('delivery.logs.amazonaws.com')],
      actions: ['xray:PutTraceSegments'],
      resources: ['*'],
      conditions: {
        'ForAllValues:ArnLike': { 'logs:LogGeneratingResourceArns': [sourceArn] },
        'StringEquals': { 'aws:SourceAccount': stack.account },
        'ArnLike': {
          'aws:SourceArn': stack.formatArn({
            service: 'logs',
            resource: 'delivery-source',
            resourceName: '*',
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        },
      },
    }));
  }

  // Create delivery destination for X-Ray
  const deliveryDestination = new logs.CfnDeliveryDestination(scope, 'TracesDeliveryDest', {
    name: Lazy.string({ produce: (): string => Names.uniqueResourceName(deliveryDestination, { maxLength: MAX_DELIVERY_NAME_LENGTH }) }),
    deliveryDestinationType: 'XRAY',
  });

  if (xrayPolicy) {
    deliveryDestination.node.addDependency(xrayPolicy);
  }

  // Create delivery to connect source and destination
  const delivery = new logs.CfnDelivery(scope, 'TracesDelivery', {
    deliverySourceName: deliverySource.deliverySourceRef.deliverySourceName,
    deliveryDestinationArn: deliveryDestination.attrArn,
  });

  delivery.node.addDependency(deliverySource);
  delivery.node.addDependency(deliveryDestination);

  return delivery;
}

/**
 * Configure logging delivery for a runtime
 *
 * @param scope The construct scope
 * @param sourceArn The ARN of the source resource (runtime)
 * @param loggingConfigs Array of logging configurations
 * @param options Observability options
 * @internal
 */
export function configureLoggingDelivery(
  scope: Construct,
  sourceArn: string,
  loggingConfigs: LoggingConfig[],
  options: RuntimeObservabilityOptions = {},
): logs.CfnDelivery[] {
  const deliveries: logs.CfnDelivery[] = [];

  // Group configs by log type to create one source per log type
  const configsByLogType = new Map<string, LoggingConfig[]>();
  for (const config of loggingConfigs) {
    const existing = configsByLogType.get(config.logType.value) ?? [];
    existing.push(config);
    configsByLogType.set(config.logType.value, existing);
  }

  // Create delivery sources and deliveries for each log type
  for (const [logType, configs] of configsByLogType) {
    // Convert log type to PascalCase for construct IDs (e.g., APPLICATION_LOGS → ApplicationLogs)
    const logTypeId = logType.split('_').map(word => word.charAt(0) + word.slice(1).toLowerCase()).join('');

    // Create a delivery source for this log type
    const deliverySource: logs.CfnDeliverySource = new logs.CfnDeliverySource(scope, `${logTypeId}DeliverySource`, {
      name: Lazy.string({ produce: (): string => Names.uniqueResourceName(deliverySource, { maxLength: MAX_DELIVERY_NAME_LENGTH }) }),
      logType: logType,
      resourceArn: sourceArn,
    });

    // Create delivery for each destination
    configs.forEach((config, index) => {
      const id = configs.length === 1 ? logTypeId : `${logTypeId}${index}`;
      const bindConfig = config.destination._bind(scope, id, options);

      const delivery = new logs.CfnDelivery(scope, `${id}Delivery`, {
        deliverySourceName: deliverySource.deliverySourceRef.deliverySourceName,
        deliveryDestinationArn: bindConfig.deliveryDestination.attrArn,
      });

      delivery.node.addDependency(deliverySource);
      delivery.node.addDependency(bindConfig.deliveryDestination);

      if (deliveries.length > 0) {
        delivery.node.addDependency(deliveries[deliveries.length - 1]);
      }

      deliveries.push(delivery);
    });
  }

  return deliveries;
}
