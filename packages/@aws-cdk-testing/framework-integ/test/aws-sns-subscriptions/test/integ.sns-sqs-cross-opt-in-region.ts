import * as sns from 'aws-cdk-lib/aws-sns';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as cdk from 'aws-cdk-lib';
import * as subs from 'aws-cdk-lib/aws-sns-subscriptions';

/// !cdk-integ * pragma:enable-lookups
const app = new cdk.App();

const topicStack = new cdk.Stack(app, 'TopicStack', {
  env: {
    account: process.env.CDK_INTEG_ACCOUNT || process.env.CDK_DEFAULT_ACCOUNT,
    region: 'ap-southeast-4',
  },
});
const topic = new sns.Topic(topicStack, 'MyTopic', {
  topicName: cdk.PhysicalName.GENERATE_IF_NEEDED,
});

const queueStack = new cdk.Stack(app, 'QueueStack', {
  env: { region: 'us-east-1' },
});
const queue = new sqs.Queue(queueStack, 'MyQueue');

topic.addSubscription(new subs.SqsSubscription(queue));

app.synth();
