import { Match, Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as firehose from '../../aws-kinesisfirehose';
import * as s3 from '../../aws-s3';
import * as sns from '../../aws-sns';
import * as sqs from '../../aws-sqs';
import { App, Stack } from '../../core';
import * as subs from '../lib';

describe('Amazon Data Firehose delivery stream subscription', () => {
  let stack: Stack;
  let topic: sns.Topic;

  beforeEach(() => {
    stack = new Stack();
    topic = new sns.Topic(stack, 'MyTopic');
  });

  test('creates configurations and role', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      Region: Match.absent(),
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [{
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Principal: { Service: 'sns.amazonaws.com' },
        }],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [{
          Action: [
            'firehose:DescribeDeliveryStream',
            'firehose:ListDeliveryStreams',
            'firehose:ListTagsForDeliveryStream',
            'firehose:PutRecord',
            'firehose:PutRecordBatch',
          ],
          Effect: 'Allow',
          Resource: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
        }],
      },
    });
  });

  test('creates configurations with raw message delivery', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream, {
      rawMessageDelivery: true,
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      RawMessageDelivery: true,
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
  });

  test('creates configurations with filter policy', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream, {
      filterPolicy: {
        color: sns.SubscriptionFilter.stringFilter({ allowlist: ['red', 'blue'] }),
      },
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      FilterPolicy: { color: ['red', 'blue'] },
      Protocol: 'firehose',
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
  });

  test('creates configurations with message body filter policy', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream, {
      filterPolicyWithMessageBody: {
        color: sns.FilterOrPolicy.filter(sns.SubscriptionFilter.stringFilter({ allowlist: ['red', 'blue'] })),
      },
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      FilterPolicy: { color: ['red', 'blue'] },
      FilterPolicyScope: 'MessageBody',
      Protocol: 'firehose',
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
  });

  test('creates configurations with user provided role', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('sns.amazonaws.com'),
    });
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream, { role }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      SubscriptionRoleArn: { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
  });

  test('creates configurations with user provided role', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('sns.amazonaws.com'),
    });
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream, { role }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      SubscriptionRoleArn: { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
  });

  test('creates configurations with user provided dlq', () => {
    // GIVEN
    const bucket = new s3.Bucket(stack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    const deadLetterQueue = new sqs.Queue(stack, 'DeadLetterQueue');
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream, { deadLetterQueue }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      RedrivePolicy: {
        deadLetterTargetArn: { 'Fn::GetAtt': ['DeadLetterQueue9F481546', 'Arn'] },
      },
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { Ref: 'MyTopic86869434' },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::SQS::QueuePolicy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'sqs:SendMessage',
            Condition: {
              ArnEquals: {
                'aws:SourceArn': { Ref: 'MyTopic86869434' },
              },
            },
            Effect: 'Allow',
            Principal: { Service: 'sns.amazonaws.com' },
            Resource: { 'Fn::GetAtt': ['DeadLetterQueue9F481546', 'Arn'] },
          },
        ],
      },
      Queues: [{ Ref: 'DeadLetterQueue9F481546' }],
    });
  });
});

describe('Amazon Data Firehose delivery stream subscription cross-stack', () => {
  test('creates cross-region configuration', () => {
    // GIVEN
    const app = new App();
    const topicStack = new Stack(app, 'TopicStack', { env: { account: '111111111111', region: 'us-east-1' } });
    const firehoseStack = new Stack(app, 'FirehoseStack', { env: { account: '111111111111', region: 'us-east-2' } });

    const topic = new sns.Topic(topicStack, 'Topic', {
      displayName: 'displayName',
      topicName: 'topicName',
    });

    const bucket = new s3.Bucket(firehoseStack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(firehoseStack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream));

    // THEN
    Template.fromStack(topicStack).hasResourceProperties('AWS::SNS::Topic', {
      DisplayName: 'displayName',
      TopicName: 'topicName',
    });
    Template.fromStack(firehoseStack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      Region: 'us-east-1',
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':sns:us-east-1:111111111111:topicName']] },
    });
    Template.fromStack(firehoseStack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [{
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Principal: { Service: 'sns.amazonaws.com' },
        }],
      },
    });
    Template.fromStack(firehoseStack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [{
          Action: [
            'firehose:DescribeDeliveryStream',
            'firehose:ListDeliveryStreams',
            'firehose:ListTagsForDeliveryStream',
            'firehose:PutRecord',
            'firehose:PutRecordBatch',
          ],
          Effect: 'Allow',
          Resource: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
        }],
      },
    });
    expect(firehoseStack.dependencies).toContain(topicStack);
  });

  test('creates same-region configuration', () => {
    // GIVEN
    const app = new App();
    const topicStack = new Stack(app, 'TopicStack');
    const firehoseStack = new Stack(app, 'FirehoseStack');

    const topic = new sns.Topic(topicStack, 'Topic', {
      displayName: 'displayName',
      topicName: 'topicName',
    });

    const bucket = new s3.Bucket(firehoseStack, 'Bucket');
    const deliveryStream = new firehose.DeliveryStream(firehoseStack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    // WHEN
    topic.addSubscription(new subs.FirehoseSubscription(deliveryStream));

    // THEN
    Template.fromStack(topicStack).hasResourceProperties('AWS::SNS::Topic', {
      DisplayName: 'displayName',
      TopicName: 'topicName',
    });
    Template.fromStack(firehoseStack).hasResourceProperties('AWS::SNS::Subscription', {
      Endpoint: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
      Protocol: 'firehose',
      Region: Match.absent(),
      SubscriptionRoleArn: { 'Fn::GetAtt': ['DeliveryStreamTopicSubscriptionRole4964AFE6', 'Arn'] },
      TopicArn: { 'Fn::ImportValue': 'TopicStack:ExportsOutputRefTopicBFC7AF6ECB4A357A' },
    });
    Template.fromStack(firehoseStack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [{
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Principal: { Service: 'sns.amazonaws.com' },
        }],
      },
    });
    Template.fromStack(firehoseStack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [{
          Action: [
            'firehose:DescribeDeliveryStream',
            'firehose:ListDeliveryStreams',
            'firehose:ListTagsForDeliveryStream',
            'firehose:PutRecord',
            'firehose:PutRecordBatch',
          ],
          Effect: 'Allow',
          Resource: { 'Fn::GetAtt': ['DeliveryStream58CF96DB', 'Arn'] },
        }],
      },
    });
  });
});
