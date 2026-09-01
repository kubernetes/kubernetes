import type { Construct } from 'constructs';
import type { ISlackChannelConfigurationRef, SlackChannelConfigurationReference } from './chatbot.generated';
import { CfnSlackChannelConfiguration } from './chatbot.generated';
import * as cloudwatch from '../../aws-cloudwatch';
import type * as notifications from '../../aws-codestarnotifications';
import * as iam from '../../aws-iam';
import * as logs from '../../aws-logs';
import type * as sns from '../../aws-sns';
import * as cdk from '../../core';
import type { IArrayBox } from '../../core/lib/helpers-internal';
import { Box } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';

/**
 * Properties for a new Slack channel configuration
 */
export interface SlackChannelConfigurationProps {

  /**
   * The name of Slack channel configuration
   */
  readonly slackChannelConfigurationName: string;

  /**
   * The permission role of Slack channel configuration
   *
   * @default - A role will be created.
   */
  readonly role?: iam.IRole;

  /**
   * The ID of the Slack workspace authorized with AWS Chatbot.
   *
   * To get the workspace ID, you must perform the initial authorization flow with Slack in the AWS Chatbot console.
   * Then you can copy and paste the workspace ID from the console.
   * For more details, see steps 1-4 in Setting Up AWS Chatbot with Slack in the AWS Chatbot User Guide.
   * @see https://docs.aws.amazon.com/chatbot/latest/adminguide/setting-up.html#Setup_intro
   */
  readonly slackWorkspaceId: string;

  /**
   * The ID of the Slack channel.
   *
   * To get the ID, open Slack, right click on the channel name in the left pane, then choose Copy Link.
   * The channel ID is the 9-character string at the end of the URL. For example, ABCBBLZZZ.
   */
  readonly slackChannelId: string;

  /**
   * The SNS topics that deliver notifications to AWS Chatbot.
   *
   * @default None
   */
  readonly notificationTopics?: sns.ITopic[];

  /**
   * Specifies the logging level for this configuration.
   * This property affects the log entries pushed to Amazon CloudWatch Logs.
   *
   * @default LoggingLevel.NONE
   */
  readonly loggingLevel?: LoggingLevel;

  /**
   * The number of days log events are kept in CloudWatch Logs. When updating
   * this property, unsetting it doesn't remove the log retention policy. To
   * remove the retention policy, set the value to `INFINITE`.
   *
   * @default logs.RetentionDays.INFINITE
   */
  readonly logRetention?: logs.RetentionDays;

  /**
   * The IAM role for the Lambda function associated with the custom resource
   * that sets the retention policy.
   *
   * @default - A new role is created.
   */
  readonly logRetentionRole?: iam.IRole;

  /**
   * When log retention is specified, a custom resource attempts to create the CloudWatch log group.
   * These options control the retry policy when interacting with CloudWatch APIs.
   *
   * @default - Default AWS SDK retry options.
   */
  readonly logRetentionRetryOptions?: logs.LogRetentionRetryOptions;

  /**
   * A list of IAM managed policies that are applied as channel guardrails.
   * @default - The AWS managed 'AdministratorAccess' policy is applied as a default if this is not set.
   */
  readonly guardrailPolicies?: iam.IManagedPolicy[];

  /**
   * Enables use of a user role requirement in your chat configuration.
   *
   * @default false
   */
  readonly userRoleRequired?: boolean;
}

/**
 * Logging levels include ERROR, INFO, or NONE.
 */
export enum LoggingLevel {
  /**
   * ERROR
   */
  ERROR = 'ERROR',

  /**
   * INFO
   */
  INFO = 'INFO',

  /**
   * NONE
   */
  NONE = 'NONE',
}

/**
 * Represents a Slack channel configuration
 */
// eslint-disable-next-line max-len
export interface ISlackChannelConfiguration extends cdk.IResource, iam.IGrantable, notifications.INotificationRuleTarget, ISlackChannelConfigurationRef {

  /**
   * The ARN of the Slack channel configuration
   * In the form of arn:aws:chatbot:{region}:{account}:chat-configuration/slack-channel/{slackChannelName}
   * @attribute
   */
  readonly slackChannelConfigurationArn: string;

  /**
   * The name of Slack channel configuration
   * @attribute
   */
  readonly slackChannelConfigurationName: string;

  /**
   * The permission role of Slack channel configuration
   * @attribute
   *
   * @default - A role will be created.
   */
  readonly role?: iam.IRole;

  /**
   * Adds a statement to the IAM role.
   */
  addToRolePolicy(statement: iam.PolicyStatement): void;

  /**
   * Return the given named metric for this SlackChannelConfiguration
   */
  metric(metricName: string, props?: cloudwatch.MetricOptions): cloudwatch.Metric;
}

/**
 * Either a new or imported Slack channel configuration
 */
abstract class SlackChannelConfigurationBase extends cdk.Resource implements ISlackChannelConfiguration {
  abstract readonly slackChannelConfigurationArn: string;

  abstract readonly slackChannelConfigurationName: string;

  abstract readonly grantPrincipal: iam.IPrincipal;

  abstract readonly role?: iam.IRole;

  /**
   * Adds extra permission to iam-role of Slack channel configuration
   */
  public addToRolePolicy(statement: iam.PolicyStatement): void {
    if (!this.role) {
      return;
    }

    this.role.addToPrincipalPolicy(statement);
  }

  /**
   * Return the given named metric for this SlackChannelConfiguration
   */
  public metric(metricName: string, props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    // AWS Chatbot publishes metrics to us-east-1 regardless of stack region
    // https://docs.aws.amazon.com/chatbot/latest/adminguide/monitoring-cloudwatch.html
    return new cloudwatch.Metric({
      namespace: 'AWS/Chatbot',
      region: 'us-east-1',
      dimensionsMap: {
        ConfigurationName: this.slackChannelConfigurationName,
      },
      metricName,
      ...props,
    });
  }

  public bindAsNotificationRuleTarget(_scope: Construct): notifications.NotificationRuleTargetConfig {
    return {
      targetType: 'AWSChatbotSlack',
      targetAddress: this.slackChannelConfigurationArn,
    };
  }

  public get slackChannelConfigurationRef(): SlackChannelConfigurationReference {
    return { slackChannelConfigurationArn: this.slackChannelConfigurationArn };
  }
}

/**
 * A new Slack channel configuration
 */
@propertyInjectable
@noBoxStackTraces
export class SlackChannelConfiguration extends SlackChannelConfigurationBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-chatbot.SlackChannelConfiguration';

  /**
   * Import an existing Slack channel configuration provided an ARN
   * @param scope The parent creating construct
   * @param id The construct's name
   * @param slackChannelConfigurationArn configuration ARN (i.e. arn:aws:chatbot::123456789012:chat-configuration/slack-channel/my-slack)
   *
   * @returns a reference to the existing Slack channel configuration
   */
  public static fromSlackChannelConfigurationArn(scope: Construct, id: string, slackChannelConfigurationArn: string): ISlackChannelConfiguration {
    const re = /^slack-channel\//;
    const resourceName = cdk.Arn.extractResourceName(slackChannelConfigurationArn, 'chat-configuration');

    if (!cdk.Token.isUnresolved(slackChannelConfigurationArn) && !re.test(resourceName)) {
      throw new cdk.ValidationError(lit`InvalidSlackChannelArn`, 'The ARN of a Slack integration must be in the form: arn:<partition>:chatbot:<region>:<account>:chat-configuration/slack-channel/<slackChannelName>', scope);
    }

    class Import extends SlackChannelConfigurationBase {
      /**
       * @attribute
       */
      readonly slackChannelConfigurationArn = slackChannelConfigurationArn;
      readonly role?: iam.IRole = undefined;
      readonly grantPrincipal: iam.IPrincipal;

      /**
       * Returns a name of Slack channel configuration
       *
       * NOTE:
       * For example: arn:aws:chatbot::123456789012:chat-configuration/slack-channel/my-slack
       * The ArnComponents API will return `slack-channel/my-slack`
       * It need to handle that to gets a correct name.`my-slack`
       */
      readonly slackChannelConfigurationName: string;

      constructor(s: Construct, i: string) {
        super(s, i);
        this.grantPrincipal = new iam.UnknownPrincipal({ resource: this });

        // handle slackChannelConfigurationName as specified above
        if (cdk.Token.isUnresolved(slackChannelConfigurationArn)) {
          this.slackChannelConfigurationName = cdk.Fn.select(1, cdk.Fn.split('slack-channel/', resourceName));
        } else {
          this.slackChannelConfigurationName = resourceName.substring('slack-channel/'.length);
        }
      }
    }

    return new Import(scope, id);
  }

  /**
   * Return the given named metric for All SlackChannelConfigurations
   */
  public static metricAll(metricName: string, props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    // AWS Chatbot publishes metrics to us-east-1 regardless of stack region
    // https://docs.aws.amazon.com/chatbot/latest/adminguide/monitoring-cloudwatch.html
    return new cloudwatch.Metric({
      namespace: 'AWS/Chatbot',
      region: 'us-east-1',
      metricName,
      ...props,
    });
  }

  readonly slackChannelConfigurationArn: string;

  readonly slackChannelConfigurationName: string;

  readonly role?: iam.IRole;

  readonly grantPrincipal: iam.IPrincipal;

  /**
   * The SNS topic that deliver notifications to AWS Chatbot.
   * @attribute
   */
  private readonly notificationTopics: IArrayBox<sns.ITopic>;

  constructor(scope: Construct, id: string, props: SlackChannelConfigurationProps) {
    super(scope, id, {
      physicalName: props.slackChannelConfigurationName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.role = props.role || new iam.Role(this, 'ConfigurationRole', {
      assumedBy: new iam.ServicePrincipal('chatbot.amazonaws.com'),
    });

    this.grantPrincipal = this.role;

    this.notificationTopics = Box.fromArray(props.notificationTopics ?? []);

    const configuration = new CfnSlackChannelConfiguration(this, 'Resource', {
      configurationName: props.slackChannelConfigurationName,
      iamRoleArn: this.role.roleArn,
      slackWorkspaceId: props.slackWorkspaceId,
      slackChannelId: props.slackChannelId,
      snsTopicArns: cdk.Token.asList(this.notificationTopics.map(topic => topic.topicArn), { displayHint: 'snsTopicArns' }),
      loggingLevel: props.loggingLevel?.toString(),
      guardrailPolicies: cdk.Lazy.list({ produce: () => props.guardrailPolicies?.map(policy => policy.managedPolicyArn) }, { omitEmpty: true } ),
      userRoleRequired: props.userRoleRequired,
    });

    // Log retention
    // AWS Chatbot publishes logs to us-east-1 regardless of stack region https://docs.aws.amazon.com/chatbot/latest/adminguide/cloudwatch-logs.html
    if (props.logRetention) {
      new logs.LogRetention(this, 'LogRetention', {
        logGroupName: `/aws/chatbot/${props.slackChannelConfigurationName}`,
        retention: props.logRetention,
        role: props.logRetentionRole,
        logGroupRegion: 'us-east-1',
        logRetentionRetryOptions: props.logRetentionRetryOptions,
      });
    }

    this.slackChannelConfigurationArn = configuration.ref;
    this.slackChannelConfigurationName = props.slackChannelConfigurationName;
  }

  /**
   * Adds a SNS topic that deliver notifications to AWS Chatbot.
   */
  @MethodMetadata()
  public addNotificationTopic(notificationTopic: sns.ITopic): void {
    this.notificationTopics.push(notificationTopic);
  }
}

