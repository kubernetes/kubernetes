import type { Construct } from 'constructs';
import { QueuePolicy } from './policy';
import { QueueGrants } from './sqs-grants.generated';
import type { IQueueRef, QueueReference } from './sqs.generated';
import * as cloudwatch from '../../aws-cloudwatch';
import * as iam from '../../aws-iam';
import type { GrantOnKeyResult, IEncryptedResource, IGrantable } from '../../aws-iam';
import type * as kms from '../../aws-kms';
import type { IResource, ResourceProps } from '../../core';
import { Resource } from '../../core';

/**
 * Represents an SQS queue
 */
export interface IQueue extends IResource, IQueueRef {
  /**
   * The ARN of this queue
   * @attribute
   */
  readonly queueArn: string;

  /**
   * The URL of this queue
   * @attribute
   */
  readonly queueUrl: string;

  /**
   * The name of this queue
   * @attribute
   */
  readonly queueName: string;

  /**
   * If this queue is server-side encrypted, this is the KMS encryption key.
   */
  readonly encryptionMasterKey?: kms.IKey;

  /**
   * Whether this queue is an Amazon SQS FIFO queue. If false, this is a standard queue.
   */
  readonly fifo: boolean;

  /**
   * Whether the contents of the queue are encrypted, and by what type of key.
   */
  readonly encryptionType?: QueueEncryption;

  /**
   * Adds a statement to the IAM resource policy associated with this queue.
   *
   * If this queue was created in this stack (`new Queue`), a queue policy
   * will be automatically created upon the first call to `addToPolicy`. If
   * the queue is imported (`Queue.import`), then this is a no-op.
   */
  addToResourcePolicy(statement: iam.PolicyStatement): iam.AddToResourcePolicyResult;

  /**
   * Grant permissions to consume messages from a queue
   *
   * This will grant the following permissions:
   *
   *   - sqs:ChangeMessageVisibility
   *   - sqs:DeleteMessage
   *   - sqs:ReceiveMessage
   *   - sqs:GetQueueAttributes
   *   - sqs:GetQueueUrl
   *
   * @param grantee Principal to grant consume rights to
   */
  grantConsumeMessages(grantee: iam.IGrantable): iam.Grant;

  /**
   * Grant access to send messages to a queue to the given identity.
   *
   * This will grant the following permissions:
   *
   *  - sqs:SendMessage
   *  - sqs:GetQueueAttributes
   *  - sqs:GetQueueUrl
   *
   * @param grantee Principal to grant send rights to
   */
  grantSendMessages(grantee: iam.IGrantable): iam.Grant;

  /**
   * Grant an IAM principal permissions to purge all messages from the queue.
   *
   * This will grant the following permissions:
   *
   *  - sqs:PurgeQueue
   *  - sqs:GetQueueAttributes
   *  - sqs:GetQueueUrl
   *
   * @param grantee Principal to grant send rights to
   */
  grantPurge(grantee: iam.IGrantable): iam.Grant;

  /**
   * Grant the actions defined in queueActions to the identity Principal given
   * on this SQS queue resource.
   *
   * @param grantee Principal to grant right to
   * @param queueActions The actions to grant
   */
  grant(grantee: iam.IGrantable, ...queueActions: string[]): iam.Grant;

  /**
   * The number of messages waiting to be picked up plus the number in flight
   *
   * `ApproximateNumberOfMessagesVisible + ApproximateNumberOfMessagesNotVisible`, as a metric math
   * expression. Prefer this over `metricApproximateNumberOfMessagesVisible` when scaling consumers
   * in: receiving a message lowers `Visible`, so a policy watching only `Visible` cannot tell a
   * consumer that just picked up work from one that finished it.
   *
   * `statistic`, `unit` and dimensions apply to both underlying metrics, `label`, `color` and
   * `period` to the expression.
   *
   * Maximum over 5 minutes
   */
  metricApproximateNumberOfMessagesOutstanding(props?: cloudwatch.MetricOptions): cloudwatch.MathExpression;
}

/**
 * Reference to a new or existing Amazon SQS queue
 */
export abstract class QueueBase extends Resource implements IQueue, IEncryptedResource {
  /**
   * The ARN of this queue
   */
  public abstract readonly queueArn: string;

  /**
   * The URL of this queue
   */
  public abstract readonly queueUrl: string;

  /**
   * The name of this queue
   */
  public abstract readonly queueName: string;

  /**
   * If this queue is server-side encrypted, this is the KMS encryption key.
   */
  public abstract readonly encryptionMasterKey?: kms.IKey;

  /**
   * Whether this queue is an Amazon SQS FIFO queue. If false, this is a standard queue.
   */
  public abstract readonly fifo: boolean;

  /**
   * Whether the contents of the queue are encrypted, and by what type of key.
   */
  public abstract readonly encryptionType?: QueueEncryption;

  /**
   * Collection of grant methods for a Queue
   */
  public readonly grants = QueueGrants.fromQueue(this);

  /**
   * Controls automatic creation of policy objects.
   *
   * Set by subclasses.
   */
  protected abstract readonly autoCreatePolicy: boolean;

  private policy?: QueuePolicy;

  constructor(scope: Construct, id: string, props: ResourceProps = {}) {
    super(scope, id, props);

    this.node.addValidation({ validate: () => this.policy?.document.validateForResourcePolicy() ?? [] });
  }

  public grantOnKey(grantee: IGrantable, ...actions: string[]): GrantOnKeyResult {
    const grant = this.encryptionMasterKey
      ? this.encryptionMasterKey.grant(grantee, ...actions)
      : undefined;
    return { grant };
  }

  public get queueRef(): QueueReference {
    return {
      queueUrl: this.queueUrl,
      queueArn: this.queueArn,
    };
  }

  /**
   * Adds a statement to the IAM resource policy associated with this queue.
   *
   * If this queue was created in this stack (`new Queue`), a queue policy
   * will be automatically created upon the first call to `addToPolicy`. If
   * the queue is imported (`Queue.import`), then this is a no-op.
   */
  public addToResourcePolicy(statement: iam.PolicyStatement): iam.AddToResourcePolicyResult {
    if (!this.policy && this.autoCreatePolicy) {
      this.policy = new QueuePolicy(this, 'Policy', { queues: [this] });
    }

    if (this.policy) {
      this.policy.document.addStatements(statement);
      return { statementAdded: true, policyDependable: this.policy };
    }

    return { statementAdded: false };
  }

  /**
   * Grant permissions to consume messages from a queue
   *
   * This will grant the following permissions:
   *
   *   - sqs:ChangeMessageVisibility
   *   - sqs:DeleteMessage
   *   - sqs:ReceiveMessage
   *   - sqs:GetQueueAttributes
   *   - sqs:GetQueueUrl
   *
   * If encryption is used, permission to use the key to decrypt the contents of the queue will also be granted to the same principal.
   *
   * This will grant the following KMS permissions:
   *
   *   - kms:Decrypt
   *
   *
   * The use of this method is discouraged. Please use `grants.consumeMessages()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee Principal to grant consume rights to
   */
  public grantConsumeMessages(grantee: iam.IGrantable) {
    return this.grants.consumeMessages(grantee);
  }

  /**
   * Grant access to send messages to a queue to the given identity.
   *
   * This will grant the following permissions:
   *
   *  - sqs:SendMessage
   *  - sqs:GetQueueAttributes
   *  - sqs:GetQueueUrl
   *
   * If encryption is used, permission to use the key to encrypt/decrypt the contents of the queue will also be granted to the same principal.
   *
   * This will grant the following KMS permissions:
   *
   *  - kms:Decrypt
   *  - kms:Encrypt
   *  - kms:ReEncrypt*
   *  - kms:GenerateDataKey*
   *
   *
   * The use of this method is discouraged. Please use `grants.sendMessages()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee Principal to grant send rights to
   */
  public grantSendMessages(grantee: iam.IGrantable) {
    return this.grants.sendMessages(grantee);
  }

  /**
   * Grant an IAM principal permissions to purge all messages from the queue.
   *
   * This will grant the following permissions:
   *
   *  - sqs:PurgeQueue
   *  - sqs:GetQueueAttributes
   *  - sqs:GetQueueUrl
   *
   *
   * The use of this method is discouraged. Please use `grants.purge()` instead.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee Principal to grant send rights to
   */
  public grantPurge(grantee: iam.IGrantable) {
    return this.grants.purge(grantee);
  }

  /**
   * Grant the actions defined in queueActions to the identity Principal given
   * on this SQS queue resource.
   *
   * [disable-awslint:no-grants]
   *
   * @param grantee Principal to grant right to
   * @param actions The actions to grant
   */
  public grant(grantee: iam.IGrantable, ...actions: string[]) {
    return iam.Grant.addToPrincipalOrResource({
      grantee,
      actions,
      resourceArns: [this.queueArn],
      resource: this,
    });
  }

  public metricApproximateNumberOfMessagesOutstanding(props?: cloudwatch.MetricOptions): cloudwatch.MathExpression {
    return new cloudwatch.MathExpression({
      expression: 'visible + notVisible',
      usingMetrics: {
        visible: this.metricApproximateNumberOfMessagesVisible(props),
        notVisible: this.metricApproximateNumberOfMessagesNotVisible(props),
      },
      label: props?.label ?? 'Approximate number of messages outstanding',
      color: props?.color,
      period: props?.period,
    });
  }
}
/**
 * Reference to a queue
 */
export interface QueueAttributes {
  /**
   * The ARN of the queue.
   */
  readonly queueArn: string;

  /**
   * The URL of the queue.
   * @see https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-queue-message-identifiers.html#sqs-general-identifiers
   *
   * @default - 'https://sqs.<region-endpoint>/<account-ID>/<queue-name>'
   */
  readonly queueUrl?: string;

  /**
   * The name of the queue.
   * @default if queue name is not specified, the name will be derived from the queue ARN
   */
  readonly queueName?: string;

  /**
   * KMS encryption key, if this queue is server-side encrypted by a KMS key.
   *
   * @default - None
   */
  readonly keyArn?: string;

  /**
   * Whether this queue is an Amazon SQS FIFO queue. If false, this is a standard queue.
   *
   * In case of a FIFO queue which is imported from a token, this value has to be explicitly set to true.
   *
   * @default - if fifo is not specified, the property will be determined based on the queue name (not possible for FIFO queues imported from a token)
   */
  readonly fifo?: boolean;
}

/**
 * What kind of encryption to apply to this queue
 */
export enum QueueEncryption {
  /**
   * Messages in the queue are not encrypted
   */
  UNENCRYPTED = 'NONE',

  /**
   * Server-side KMS encryption with a KMS key managed by SQS.
   */
  KMS_MANAGED = 'KMS_MANAGED',

  /**
   * Server-side encryption with a KMS key managed by the user.
   *
   * If `encryptionKey` is specified, this key will be used, otherwise, one will be defined.
   */
  KMS = 'KMS',

  /**
   * Server-side encryption key managed by SQS (SSE-SQS).
   *
   * To learn more about SSE-SQS on Amazon SQS, please visit the
   * [Amazon SQS documentation](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-server-side-encryption.html).
   */
  SQS_MANAGED = 'SQS_MANAGED',
}
