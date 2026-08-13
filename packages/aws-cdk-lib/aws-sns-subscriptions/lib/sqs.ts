import { Construct } from 'constructs';
import type { SubscriptionProps } from './subscription';
import * as iam from '../../aws-iam';
import * as sns from '../../aws-sns';
import * as sqs from '../../aws-sqs';
import { FeatureFlags, Names, ValidationError } from '../../core';
import * as cxapi from '../../cx-api';
import { regionFromArn, snsServicePrincipal } from './private/util';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Properties for an SQS subscription
 */
export interface SqsSubscriptionProps extends SubscriptionProps {
  /**
   * The message to the queue is the same as it was sent to the topic
   *
   * If false, the message will be wrapped in an SNS envelope.
   *
   * @default false
   */
  readonly rawMessageDelivery?: boolean;
}

/**
 * Use an SQS queue as a subscription target
 */
export class SqsSubscription implements sns.ITopicSubscription {
  constructor(private readonly queue: sqs.IQueue, private readonly props: SqsSubscriptionProps = {}) {
  }

  /**
   * Returns a configuration for an SQS queue to subscribe to an SNS topic
   */
  public bind(topic: sns.ITopic): sns.TopicSubscriptionConfig {
    // Create subscription under *consuming* construct to make sure it ends up
    // in the correct stack in cases of cross-stack subscriptions.
    if (!Construct.isConstruct(this.queue)) {
      throw new ValidationError(lit`SuppliedQueueObjectInstanceConstruct`, 'The supplied Queue object must be an instance of Construct', topic);
    }
    const principal = snsServicePrincipal(topic, this.queue);

    // if the queue is encrypted by AWS managed KMS key (alias/aws/sqs),
    // throw error message
    if (this.queue.encryptionType === sqs.QueueEncryption.KMS_MANAGED) {
      throw new ValidationError(lit`QueueEncryptedManagedKeyCannot`, 'SQS queue encrypted by AWS managed KMS key cannot be used as SNS subscription', topic);
    }

    // if the dead-letter queue is encrypted by AWS managed KMS key (alias/aws/sqs),
    // throw error message
    if (this.props.deadLetterQueue && this.props.deadLetterQueue.encryptionType === sqs.QueueEncryption.KMS_MANAGED) {
      throw new ValidationError(lit`QueueEncryptedManagedKeyCannot`, 'SQS queue encrypted by AWS managed KMS key cannot be used as dead-letter queue', topic);
    }

    // add a statement to the queue resource policy which allows this topic
    // to send messages to the queue.
    const queuePolicyDependable = this.queue.addToResourcePolicy(new iam.PolicyStatement({
      resources: [this.queue.queueArn],
      actions: ['sqs:SendMessage'],
      principals: [principal],
      conditions: {
        ArnEquals: { 'aws:SourceArn': topic.topicArn },
      },
    })).policyDependable;

    // if the queue is encrypted, add a statement to the key resource policy
    // which allows this topic to decrypt KMS keys
    if (this.queue.encryptionMasterKey) {
      this.queue.encryptionMasterKey.addToResourcePolicy(new iam.PolicyStatement({
        resources: ['*'],
        actions: ['kms:Decrypt', 'kms:GenerateDataKey'],
        principals: [principal],
        conditions: FeatureFlags.of(topic).isEnabled(cxapi.SNS_SUBSCRIPTIONS_SQS_DECRYPTION_POLICY)
          ? { ArnEquals: { 'aws:SourceArn': topic.topicArn } }
          : undefined,
      }));
    }

    // if the topic and queue are created in different stacks
    // then we need to make sure the topic is created first
    if (topic instanceof sns.Topic && topic.stack !== this.queue.stack) {
      this.queue.stack.addStackDependency(topic.stack);
    }

    return {
      subscriberScope: this.queue,
      subscriberId: Names.nodeUniqueId(topic.node),
      endpoint: this.queue.queueArn,
      protocol: sns.SubscriptionProtocol.SQS,
      rawMessageDelivery: this.props.rawMessageDelivery,
      filterPolicy: this.props.filterPolicy,
      filterPolicyWithMessageBody: this.props.filterPolicyWithMessageBody,
      region: regionFromArn(topic, this.queue),
      deadLetterQueue: this.props.deadLetterQueue,
      subscriptionDependency: queuePolicyDependable,
    };
  }
}
