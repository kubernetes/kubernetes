import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as events from 'aws-cdk-lib/aws-events';
import * as iam from 'aws-cdk-lib/aws-iam';
import type * as logs from 'aws-cdk-lib/aws-logs';
import * as cdk from 'aws-cdk-lib/core';
import { lit } from 'aws-cdk-lib/core/lib/helpers-internal';
import type * as constructs from 'constructs';
import type { Code } from '../code';
import type { IConnection } from '../connection';
import type { MetricType, WorkerType, GlueVersion } from '../constants';
import { JobState } from '../constants';
import { warnOnPlaintextSecrets } from '../private/secret-detection';
import type { ISecurityConfiguration } from '../security-configuration';

/**
 * Interface representing a new or an imported Glue Job
 */
export interface IJob extends cdk.IResource, iam.IGrantable {
  /**
   * The name of the job.
   * @attribute
   */
  readonly jobName: string;

  /**
   * The ARN of the job.
   * @attribute
   */
  readonly jobArn: string;

  /**
   * Defines a CloudWatch event rule triggered when something happens with this job.
   *
   * @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/EventTypes.html#glue-event-types
   */
  onEvent(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines a CloudWatch event rule triggered when this job moves to the SUCCEEDED state.
   *
   * @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/EventTypes.html#glue-event-types
   */
  onSuccess(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines a CloudWatch event rule triggered when this job moves to the FAILED state.
   *
   * @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/EventTypes.html#glue-event-types
   */
  onFailure(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Defines a CloudWatch event rule triggered when this job moves to the TIMEOUT state.
   *
   * @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/EventTypes.html#glue-event-types
   */
  onTimeout(id: string, options?: events.OnEventOptions): events.Rule;

  /**
   * Create a CloudWatch metric.
   *
   * @param metricName name of the metric typically prefixed with `glue.driver.`, `glue.<executorId>.` or `glue.ALL.`.
   * @param type the metric type.
   * @param props metric options.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/monitoring-awsglue-with-cloudwatch-metrics.html
   */
  metric(metricName: string, type: MetricType, props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Create a CloudWatch Metric indicating job success.
   */
  metricSuccess(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Create a CloudWatch Metric indicating job failure.
   */
  metricFailure(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Create a CloudWatch Metric indicating job timeout.
   */
  metricTimeout(props?: cloudwatch.MetricOptions): cloudwatch.Metric;
}

/**
 * Properties for enabling Continuous Logging for Glue Jobs.
 *
 * @see https://docs.aws.amazon.com/glue/latest/dg/monitor-continuous-logging-enable.html
 * @see https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html
 */
export interface ContinuousLoggingProps {
  /**
   * Enable continuous logging.
   */
  readonly enabled: boolean;

  /**
   * Specify a custom CloudWatch log group name.
   *
   * @default - a log group is created with name `/aws-glue/jobs/logs-v2/`.
   */
  readonly logGroup?: logs.ILogGroup;

  /**
   * Specify a custom CloudWatch log stream prefix.
   *
   * @default - the job run ID.
   */
  readonly logStreamPrefix?: string;

  /**
   * Filter out non-useful Apache Spark driver/executor and Apache Hadoop YARN heartbeat log messages.
   *
   * @default true
   */
  readonly quiet?: boolean;

  /**
   * Apply the provided conversion pattern.
   *
   * This is a Log4j Conversion Pattern to customize driver and executor logs.
   *
   * @default `%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n`
   */
  readonly conversionPattern?: string;
}

/**
 * A base class is needed to be able to import existing Jobs into a CDK app to
 * reference as part of a larger stack or construct. JobBase has the subset
 * of attributes required to identify and reference an existing Glue Job,
 * as well as some CloudWatch metric convenience functions to configure an
 * event-driven flow using the job.
 */
export abstract class JobBase extends cdk.Resource implements IJob {
  public abstract readonly jobArn: string;
  public abstract readonly jobName: string;
  public abstract readonly grantPrincipal: iam.IPrincipal;

  /**
   * Create a CloudWatch Event Rule for this Glue Job when it's in a given state
   *
   * @param id construct id
   * @param options event options. Note that some values are overridden if provided, these are
   *  - eventPattern.source = ['aws.glue']
   *  - eventPattern.detailType = ['Glue Job State Change', 'Glue Job Run Status']
   *  - eventPattern.detail.jobName = [this.jobName]
   *
   * @see https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/EventTypes.html#glue-event-types
   */
  public onEvent(id: string, options: events.OnEventOptions = {}): events.Rule {
    const rule = new events.Rule(this, id, options);
    rule.addTarget(options.target);
    rule.addEventPattern({
      source: ['aws.glue'],
      detailType: ['Glue Job State Change', 'Glue Job Run Status'],
      detail: {
        jobName: [this.jobName],
      },
    });
    return rule;
  }

  /**
   * Create a CloudWatch Event Rule for the transition into the input jobState.
   *
   * @param id construct id.
   * @param jobState the job state.
   * @param options optional event options.
   */
  protected onStateChange(id: string, jobState: JobState, options: events.OnEventOptions = {}): events.Rule {
    const rule = this.onEvent(id, {
      description: `Rule triggered when Glue job ${this.jobName} is in ${jobState} state`,
      ...options,
    });
    rule.addEventPattern({
      detail: {
        state: [jobState],
      },
    });
    return rule;
  }

  /**
   * Create a CloudWatch Event Rule matching JobState.SUCCEEDED.
   *
   * @param id construct id.
   * @param options optional event options. default is {}.
   */
  public onSuccess(id: string, options: events.OnEventOptions = {}): events.Rule {
    return this.onStateChange(id, JobState.SUCCEEDED, options);
  }

  /**
   * Return a CloudWatch Event Rule matching FAILED state.
   *
   * @param id construct id.
   * @param options optional event options. default is {}.
   */
  public onFailure(id: string, options: events.OnEventOptions = {}): events.Rule {
    return this.onStateChange(id, JobState.FAILED, options);
  }

  /**
   * Return a CloudWatch Event Rule matching TIMEOUT state.
   *
   * @param id construct id.
   * @param options optional event options. default is {}.
   */
  public onTimeout(id: string, options: events.OnEventOptions = {}): events.Rule {
    return this.onStateChange(id, JobState.TIMEOUT, options);
  }

  /**
   * Create a CloudWatch metric.
   *
   * @param metricName name of the metric typically prefixed with `glue.driver.`, `glue.<executorId>.` or `glue.ALL.`.
   * @param type the metric type.
   * @param props metric options.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/monitoring-awsglue-with-cloudwatch-metrics.html
   */
  public metric(metricName: string, type: MetricType, props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return new cloudwatch.Metric({
      metricName,
      namespace: 'Glue',
      dimensionsMap: {
        JobName: this.jobName,
        JobRunId: 'ALL',
        Type: type,
      },
      ...props,
    }).attachTo(this);
  }

  /**
   * Return a CloudWatch Metric indicating job success.
   *
   * This metric is based on the Rule returned by no-args onSuccess() call.
   */
  public metricSuccess(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return metricRule(this.metricJobStateRule('SuccessMetricRule', JobState.SUCCEEDED), props);
  }

  /**
   * Return a CloudWatch Metric indicating job failure.
   *
   * This metric is based on the Rule returned by no-args onFailure() call.
   */
  public metricFailure(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return metricRule(this.metricJobStateRule('FailureMetricRule', JobState.FAILED), props);
  }

  /**
   * Return a CloudWatch Metric indicating job timeout.
   *
   * This metric is based on the Rule returned by no-args onTimeout() call.
   */
  public metricTimeout(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return metricRule(this.metricJobStateRule('TimeoutMetricRule', JobState.TIMEOUT), props);
  }

  /**
   * Creates or retrieves a singleton event rule for the input job state for use with the metric JobState methods.
   *
   * @param id construct id.
   * @param jobState the job state.
   */
  private metricJobStateRule(id: string, jobState: JobState): events.Rule {
    return this.node.tryFindChild(id) as events.Rule ?? this.onStateChange(id, jobState);
  }

  /**
   * Returns the job arn
   */
  protected buildJobArn(scope: constructs.Construct, jobName: string) : string {
    return cdk.Stack.of(scope).formatArn({
      service: 'glue',
      resource: 'job',
      resourceName: jobName,
    });
  }
}

/**
 * A subset of Job attributes are required for importing an existing job
 * into a CDK project. This is only used when using fromJobAttributes
 * to identify and reference the existing job.
 */
export interface JobAttributes {
  /**
   * The name of the job.
   */
  readonly jobName: string;

  /**
   * The IAM role assumed by Glue to run this job.
   *
   * @default - undefined
   */
  readonly role?: iam.IRole;

}

/**
 * JobProps will be used to create new Glue Jobs using this L2 Construct.
 */
export interface JobProps {
  /**
   * Script Code Location (required)
   * Script to run when the Glue job executes. Can be uploaded
   * from the local directory structure using fromAsset
   * or referenced via S3 location using fromBucket
   */
  readonly script: Code;

  /**
   * IAM Role (required)
   * IAM Role to use for Glue job execution
   * Must be specified by the developer because the L2 doesn't have visibility
   * into the actions the script(s) takes during the job execution
   * The role must trust the Glue service principal (glue.amazonaws.com)
   * and be granted sufficient permissions.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/getting-started-access.html
   */
  readonly role: iam.IRole;

  /**
   * Name of the Glue job (optional)
   * Developer-specified name of the Glue job
   *
   * @default - a name is automatically generated
   */
  readonly jobName?: string;

  /**
   * Description (optional)
   * Developer-specified description of the Glue job
   *
   * @default - no value
   */
  readonly description?: string;

  /**
   * Number of Workers (optional)
   * Number of workers for Glue to use during job execution
   *
   * @default 10
   */
  readonly numberOfWorkers?: number;

  /**
   * Worker Type (optional)
   * Type of Worker for Glue to use during job execution
   * Enum options: Standard, G_1X, G_2X, G_025X. G_4X, G_8X, Z_2X
   *
   * @default WorkerType.G_1X
   */
  readonly workerType?: WorkerType;

  /**
   * Max Concurrent Runs (optional)
   * The maximum number of runs this Glue job can concurrently run
   *
   * An error is returned when this threshold is reached. The maximum value
   * you can specify is controlled by a service limit.
   *
   * @default 1
   */
  readonly maxConcurrentRuns?: number;

  /**
   * Default Arguments (optional)
   * The default arguments for every run of this Glue job,
   * specified as name-value pairs.
   *
   * These are emitted verbatim into the CloudFormation template, so avoid
   * placing secrets here in plaintext. Pass secrets to the job at runtime
   * through AWS Secrets Manager instead. A synthesis-time warning is emitted
   * when an argument key looks like a credential and holds a plaintext literal.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html
   * for a list of reserved parameters
   * @default - no arguments
   */
  readonly defaultArguments?: { [key: string]: string };

  /**
   * Connections (optional)
   * List of connections to use for this Glue job
   * Connections are used to connect to other AWS Service or resources within a VPC.
   *
   * @default [] - no connections are added to the job
   */
  readonly connections?: IConnection[];

  /**
   * Max Retries (optional)
   * Maximum number of retry attempts Glue performs if the job fails
   *
   * @default 0
   */
  readonly maxRetries?: number;

  /**
   * Timeout (optional)
   * The maximum time that a job run can consume resources before it is
   * terminated and enters TIMEOUT status. Specified in minutes.
   *
   * @default 2880 (2 days for non-streaming)
   *
   */
  readonly timeout?: cdk.Duration;

  /**
   * Security Configuration (optional)
   * Defines the encryption options for the Glue job
   *
   * @default - no security configuration.
   */
  readonly securityConfiguration?: ISecurityConfiguration;

  /**
   * Tags (optional)
   * A list of key:value pairs of tags to apply to this Glue job resources
   *
   * @default {} - no tags
   */
  readonly tags?: { [key: string]: string };

  /**
   * Glue Version
   * The version of Glue to use to execute this job
   *
   * @default 3.0 for ETL
   */
  readonly glueVersion?: GlueVersion;

  /**
   * Enables the collection of metrics for job profiling.
   *
   * @default - no profiling metrics emitted.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html
   */
  readonly enableProfilingMetrics? :boolean;

  /**
   * Enables continuous logging with the specified props.
   *
   * @default - continuous logging is enabled.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/monitor-continuous-logging-enable.html
   * @see https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html
   **/
  readonly continuousLogging?: ContinuousLoggingProps;
}

/**
 * A Glue Job.
 * @resource AWS::Glue::Job
 */
export abstract class Job extends JobBase {
  /**
   * Identifies an existing Glue Job from a subset of attributes that can
   * be referenced from within another Stack or Construct.
   *
   * @param scope The scope creating construct (usually `this`)
   * @param id The construct's id.
   * @param attrs Attributes for the Glue Job we want to import
   */
  public static fromJobAttributes(scope: constructs.Construct, id: string, attrs: JobAttributes): IJob {
    class Import extends JobBase {
      public readonly jobName = attrs.jobName;
      public readonly jobArn = this.buildJobArn(scope, attrs.jobName);
      public readonly grantPrincipal = attrs.role ?? new iam.UnknownPrincipal({ resource: this });
    }

    return new Import(scope, id);
  }

  /**
   * The IAM role Glue assumes to run this job.
   */
  public readonly abstract role: iam.IRole;

  /**
   * Check no usage of reserved arguments.
   *
   * @see https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-glue-arguments.html
   */
  protected checkNoReservedArgs(defaultArguments?: { [key: string]: string }) {
    if (defaultArguments) {
      const reservedArgs = new Set(['--debug', '--mode', '--JOB_NAME']);
      Object.keys(defaultArguments).forEach((arg) => {
        if (reservedArgs.has(arg)) {
          throw new cdk.ValidationError(lit`ReservedArgumentUsed`, `The ${arg} argument is reserved by Glue. Don't set it`, this);
        }
      });
    }
    warnOnPlaintextSecrets(
      this,
      defaultArguments,
      '@aws-cdk/aws-glue-alpha:plaintextJobArgumentSecret',
      'Pass secrets to the job at runtime through AWS Secrets Manager instead of embedding them in `defaultArguments`.',
    );
    return defaultArguments;
  }

  /**
   * Setup Continuous Logging Properties
   * @param role The IAM role to use for continuous logging
   * @param props The properties for continuous logging configuration
   * @returns String containing the args for the continuous logging command
   */
  protected setupContinuousLogging(role: iam.IRole, props: ContinuousLoggingProps | undefined) : any {
    // If the developer has explicitly disabled continuous logging return no args
    if (props && !props.enabled) {
      return {};
    }

    // Else we turn on continuous logging by default. Determine what log group to use.
    const args: {[key: string]: string} = {
      '--enable-continuous-cloudwatch-log': 'true',
    };

    if (props?.quiet) {
      args['--enable-continuous-log-filter'] = 'true';
    }

    // If the developer provided a log group, add its name to the args and update the role.
    if (props?.logGroup) {
      args['--continuous-log-logGroup'] = props.logGroup.logGroupName;
      props.logGroup.grantWrite(role);
    }

    if (props?.logStreamPrefix) {
      args['--continuous-log-logStreamPrefix'] = props.logStreamPrefix;
    }

    if (props?.conversionPattern) {
      args['--continuous-log-conversionPattern'] = props.conversionPattern;
    }

    return args;
  }

  protected codeS3ObjectUrl(code: Code) {
    const s3Location = code.bind(this, this.role).s3Location;
    return `s3://${s3Location.bucketName}/${s3Location.objectKey}`;
  }
}

/**
 * Create a CloudWatch Metric that's based on Glue Job events
 * {@see https://docs.aws.amazon.com/AmazonCloudWatch/latest/events/EventTypes.html#glue-event-types}
 * The metric has namespace = 'AWS/Events', metricName = 'TriggeredRules' and RuleName = rule.ruleName dimension.
 *
 * @param rule for use in setting RuleName dimension value
 * @param props metric properties
 */
function metricRule(rule: events.IRule, props?: cloudwatch.MetricOptions): cloudwatch.Metric {
  return new cloudwatch.Metric({
    namespace: 'AWS/Events',
    metricName: 'TriggeredRules',
    dimensionsMap: { RuleName: rule.ruleName },
    statistic: cloudwatch.Stats.SUM,
    ...props,
  }).attachTo(rule);
}

