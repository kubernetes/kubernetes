import { Template } from '../../assertions';
import * as kms from '../../aws-kms';
import * as lambda from '../../aws-lambda';
import * as sns from '../../aws-sns';
import * as sqs from '../../aws-sqs';
import { App, CfnParameter, Duration, RemovalPolicy, Stack, Token } from '../../core';
import * as cxapi from '../../cx-api';
import * as subs from '../lib';

/* eslint-disable @stylistic/quote-props */
const restrictSqsDescryption = { [cxapi.SNS_SUBSCRIPTIONS_SQS_DECRYPTION_POLICY]: true };
let stack: Stack;
let topic: sns.Topic;

beforeEach(() => {
  stack = new Stack();
  topic = new sns.Topic(stack, 'MyTopic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });
});

test('url subscription', () => {
  topic.addSubscription(new subs.UrlSubscription('https://foobar.com/'));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopichttpsfoobarcomDEA92AB5': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': 'https://foobar.com/',
          'Protocol': 'https',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
    },
  });
});

test('url subscription with user provided dlq', () => {
  const dlQueue = new sqs.Queue(stack, 'DeadLetterQueue', {
    queueName: 'MySubscription_DLQ',
    retentionPeriod: Duration.days(14),
  });
  topic.addSubscription(new subs.UrlSubscription('https://foobar.com/', {
    deadLetterQueue: dlQueue,
  }));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopichttpsfoobarcomDEA92AB5': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': 'https://foobar.com/',
          'Protocol': 'https',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'RedrivePolicy': {
            'deadLetterTargetArn': {
              'Fn::GetAtt': [
                'DeadLetterQueue9F481546',
                'Arn',
              ],
            },
          },
        },
      },
      'DeadLetterQueue9F481546': {
        'Type': 'AWS::SQS::Queue',
        'DeletionPolicy': 'Delete',
        'UpdateReplacePolicy': 'Delete',
        'Properties': {
          'MessageRetentionPeriod': 1209600,
          'QueueName': 'MySubscription_DLQ',
        },
      },
      'DeadLetterQueuePolicyB1FB890C': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Ref': 'MyTopic86869434',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'DeadLetterQueue9F481546',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'DeadLetterQueue9F481546',
            },
          ],
        },
      },
    },
  });
});

test('url subscription (with raw delivery)', () => {
  topic.addSubscription(new subs.UrlSubscription('https://foobar.com/', {
    rawMessageDelivery: true,
  }));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopichttpsfoobarcomDEA92AB5': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': 'https://foobar.com/',
          'Protocol': 'https',
          'TopicArn': { 'Ref': 'MyTopic86869434' },
          'RawMessageDelivery': true,
        },
      },
    },
  });
});

test('url subscription (unresolved url with protocol)', () => {
  const urlToken = Token.asString({ Ref: 'my-url-1' });
  topic.addSubscription(new subs.UrlSubscription(urlToken, { protocol: sns.SubscriptionProtocol.HTTPS }));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicTokenSubscription141DD1BE2': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-url-1',
          },
          'Protocol': 'https',
          'TopicArn': { 'Ref': 'MyTopic86869434' },
        },
      },
    },
  });
});

test('url subscription (double unresolved url with protocol)', () => {
  const urlToken1 = Token.asString({ Ref: 'my-url-1' });
  const urlToken2 = Token.asString({ Ref: 'my-url-2' });

  topic.addSubscription(new subs.UrlSubscription(urlToken1, { protocol: sns.SubscriptionProtocol.HTTPS }));
  topic.addSubscription(new subs.UrlSubscription(urlToken2, { protocol: sns.SubscriptionProtocol.HTTPS }));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicTokenSubscription141DD1BE2': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-url-1',
          },
          'Protocol': 'https',
          'TopicArn': { 'Ref': 'MyTopic86869434' },
        },
      },
      'MyTopicTokenSubscription293BFE3F9': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-url-2',
          },
          'Protocol': 'https',
          'TopicArn': { 'Ref': 'MyTopic86869434' },
        },
      },
    },
  });
});

test('url subscription (unknown protocol)', () => {
  expect(() => topic.addSubscription(new subs.UrlSubscription('some-protocol://foobar.com/')))
    .toThrow(/URL must start with either http:\/\/ or https:\/\//);
});

test('url subscription (unresolved url without protocol)', () => {
  const urlToken = Token.asString({ Ref: 'my-url-1' });

  expect(() => topic.addSubscription(new subs.UrlSubscription(urlToken)))
    .toThrow(/Must provide protocol if url is unresolved/);
});

test('queue subscription', () => {
  const queue = new sqs.Queue(stack, 'MyQueue');

  topic.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'DeletionPolicy': 'Delete',
        'UpdateReplacePolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Ref': 'MyTopic86869434',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueMyTopic9B00631B': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

test('queue subscription cross region', () => {
  const app = new App();
  const topicStack = new Stack(app, 'TopicStack', {
    env: {
      account: '11111111111',
      region: 'us-east-1',
    },
  });
  const queueStack = new Stack(app, 'QueueStack', {
    env: {
      account: '11111111111',
      region: 'us-east-2',
    },
  });

  const topic1 = new sns.Topic(topicStack, 'Topic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });

  const queue = new sqs.Queue(queueStack, 'MyQueue');

  topic1.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(topicStack).templateMatches({
    'Resources': {
      'TopicBFC7AF6E': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
    },
  });

  Template.fromStack(queueStack).templateMatches({
    'Resources': {
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':sns:us-east-1:11111111111:topicName',
                        ],
                      ],
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueTopicStackTopicFBF76EB349BDFA94': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  'Ref': 'AWS::Partition',
                },
                ':sns:us-east-1:11111111111:topicName',
              ],
            ],
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
          'Region': 'us-east-1',
        },
      },
    },
  });
});

test('queue subscription cross region, env agnostic', () => {
  const app = new App();
  const topicStack = new Stack(app, 'TopicStack', {});
  const queueStack = new Stack(app, 'QueueStack', {});

  const topic1 = new sns.Topic(topicStack, 'Topic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });

  const queue = new sqs.Queue(queueStack, 'MyQueue');

  topic1.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(topicStack).templateMatches({
    'Resources': {
      'TopicBFC7AF6E': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
    },
    'Outputs': {
      'ExportsOutputRefTopicBFC7AF6ECB4A357A': {
        'Value': {
          'Ref': 'TopicBFC7AF6E',
        },
        'Export': {
          'Name': 'TopicStack:ExportsOutputRefTopicBFC7AF6ECB4A357A',
        },
      },
    },
  });

  Template.fromStack(queueStack).templateMatches({
    'Resources': {
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Fn::ImportValue': 'TopicStack:ExportsOutputRefTopicBFC7AF6ECB4A357A',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueTopicStackTopicFBF76EB349BDFA94': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Fn::ImportValue': 'TopicStack:ExportsOutputRefTopicBFC7AF6ECB4A357A',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

test('queue subscription cross region, topic env agnostic', () => {
  const app = new App();
  const topicStack = new Stack(app, 'TopicStack', {});
  const queueStack = new Stack(app, 'QueueStack', {
    env: {
      account: '11111111111',
      region: 'us-east-1',
    },
  });

  const topic1 = new sns.Topic(topicStack, 'Topic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });

  const queue = new sqs.Queue(queueStack, 'MyQueue');

  topic1.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(topicStack).templateMatches({
    'Resources': {
      'TopicBFC7AF6E': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
    },
  });

  Template.fromStack(queueStack).templateMatches({
    'Resources': {
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':sns:',
                          {
                            'Ref': 'AWS::Region',
                          },
                          ':',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':topicName',
                        ],
                      ],
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueTopicStackTopicFBF76EB349BDFA94': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  'Ref': 'AWS::Partition',
                },
                ':sns:',
                {
                  'Ref': 'AWS::Region',
                },
                ':',
                {
                  'Ref': 'AWS::AccountId',
                },
                ':topicName',
              ],
            ],
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

test('queue subscription cross region, queue env agnostic', () => {
  const app = new App();
  const topicStack = new Stack(app, 'TopicStack', {
    env: {
      account: '11111111111',
      region: 'us-east-1',
    },
  });
  const queueStack = new Stack(app, 'QueueStack', {});

  const topic1 = new sns.Topic(topicStack, 'Topic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });

  const queue = new sqs.Queue(queueStack, 'MyQueue');

  topic1.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(topicStack).templateMatches({
    'Resources': {
      'TopicBFC7AF6E': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
    },
  });

  Template.fromStack(queueStack).templateMatches({
    'Resources': {
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':sns:us-east-1:11111111111:topicName',
                        ],
                      ],
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueTopicStackTopicFBF76EB349BDFA94': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  'Ref': 'AWS::Partition',
                },
                ':sns:us-east-1:11111111111:topicName',
              ],
            ],
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
          'Region': 'us-east-1',
        },
      },
    },
  });
});
test('queue subscription with user provided dlq', () => {
  const queue = new sqs.Queue(stack, 'MyQueue');
  const dlQueue = new sqs.Queue(stack, 'DeadLetterQueue', {
    queueName: 'MySubscription_DLQ',
    retentionPeriod: Duration.days(14),
  });

  topic.addSubscription(new subs.SqsSubscription(queue, {
    deadLetterQueue: dlQueue,
  }));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'DeletionPolicy': 'Delete',
        'UpdateReplacePolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Ref': 'MyTopic86869434',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueMyTopic9B00631B': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
          'RedrivePolicy': {
            'deadLetterTargetArn': {
              'Fn::GetAtt': [
                'DeadLetterQueue9F481546',
                'Arn',
              ],
            },
          },
        },
      },
      'DeadLetterQueue9F481546': {
        'Type': 'AWS::SQS::Queue',
        'DeletionPolicy': 'Delete',
        'UpdateReplacePolicy': 'Delete',
        'Properties': {
          'MessageRetentionPeriod': 1209600,
          'QueueName': 'MySubscription_DLQ',
        },
      },
      'DeadLetterQueuePolicyB1FB890C': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Ref': 'MyTopic86869434',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'DeadLetterQueue9F481546',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'DeadLetterQueue9F481546',
            },
          ],
        },
      },
    },
  });
});

test('queue subscription (with raw delivery)', () => {
  const queue = new sqs.Queue(stack, 'MyQueue');

  topic.addSubscription(new subs.SqsSubscription(queue, { rawMessageDelivery: true }));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    'Endpoint': {
      'Fn::GetAtt': [
        'MyQueueE6CA6235',
        'Arn',
      ],
    },
    'Protocol': 'sqs',
    'TopicArn': {
      'Ref': 'MyTopic86869434',
    },
    'RawMessageDelivery': true,
  });
});

test('encrypted queue subscription', () => {
  const key = new kms.Key(stack, 'MyKey', {
    removalPolicy: RemovalPolicy.DESTROY,
  });

  const queue = new sqs.Queue(stack, 'MyQueue', {
    encryption: sqs.QueueEncryption.KMS,
    encryptionMasterKey: key,
  });

  topic.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyKey6AB29FA6': {
        'Type': 'AWS::KMS::Key',
        'Properties': {
          'KeyPolicy': {
            'Statement': [
              {
                'Action': 'kms:*',
                'Effect': 'Allow',
                'Principal': {
                  'AWS': {
                    'Fn::Join': [
                      '',
                      [
                        'arn:',
                        {
                          'Ref': 'AWS::Partition',
                        },
                        ':iam::',
                        {
                          'Ref': 'AWS::AccountId',
                        },
                        ':root',
                      ],
                    ],
                  },
                },
                'Resource': '*',
              },
              {
                'Action': [
                  'kms:Decrypt',
                  'kms:GenerateDataKey',
                ],
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': '*',
              },
            ],
            'Version': '2012-10-17',
          },
        },
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'Properties': {
          'KmsMasterKeyId': {
            'Fn::GetAtt': [
              'MyKey6AB29FA6',
              'Arn',
            ],
          },
        },
        'DeletionPolicy': 'Delete',
        'UpdateReplacePolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Ref': 'MyTopic86869434',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueMyTopic9B00631B': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

describe('Restrict sqs decryption feature flag', () => {
  test('Restrict decryption of sqs to sns service principal', () => {
    const stackUnderTest = new Stack(
      new App(),
    );
    const topicUnderTest = new sns.Topic(stackUnderTest, 'MyTopic', {
      topicName: 'topicName',
      displayName: 'displayName',
    });
    const key = new kms.Key(stackUnderTest, 'MyKey', {
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const queue = new sqs.Queue(stackUnderTest, 'MyQueue', {
      encryptionMasterKey: key,
    });

    topicUnderTest.addSubscription(new subs.SqsSubscription(queue));

    Template.fromStack(stackUnderTest).templateMatches({
      'Resources': {
        'MyKey6AB29FA6': {
          'Type': 'AWS::KMS::Key',
          'Properties': {
            'KeyPolicy': {
              'Statement': [
                {
                  'Action': 'kms:*',
                  'Effect': 'Allow',
                  'Principal': {
                    'AWS': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':iam::',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':root',
                        ],
                      ],
                    },
                  },
                  'Resource': '*',
                },
                {
                  'Action': [
                    'kms:Decrypt',
                    'kms:GenerateDataKey',
                  ],
                  'Effect': 'Allow',
                  'Principal': {
                    'Service': 'sns.amazonaws.com',
                  },
                  'Resource': '*',
                },
              ],
              'Version': '2012-10-17',
            },
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
  });
  test('Restrict decryption of sqs to sns topic', () => {
    const stackUnderTest = new Stack(
      new App({
        context: restrictSqsDescryption,
      }),
    );
    const topicUnderTest = new sns.Topic(stackUnderTest, 'MyTopic', {
      topicName: 'topicName',
      displayName: 'displayName',
    });
    const key = new kms.Key(stackUnderTest, 'MyKey', {
      removalPolicy: RemovalPolicy.DESTROY,
    });

    const queue = new sqs.Queue(stackUnderTest, 'MyQueue', {
      encryptionMasterKey: key,
    });

    topicUnderTest.addSubscription(new subs.SqsSubscription(queue));

    Template.fromStack(stackUnderTest).templateMatches({
      'Resources': {
        'MyKey6AB29FA6': {
          'Type': 'AWS::KMS::Key',
          'Properties': {
            'KeyPolicy': {
              'Statement': [
                {
                  'Action': 'kms:*',
                  'Effect': 'Allow',
                  'Principal': {
                    'AWS': {
                      'Fn::Join': [
                        '',
                        [
                          'arn:',
                          {
                            'Ref': 'AWS::Partition',
                          },
                          ':iam::',
                          {
                            'Ref': 'AWS::AccountId',
                          },
                          ':root',
                        ],
                      ],
                    },
                  },
                  'Resource': '*',
                },
                {
                  'Action': [
                    'kms:Decrypt',
                    'kms:GenerateDataKey',
                  ],
                  'Effect': 'Allow',
                  'Principal': {
                    'Service': 'sns.amazonaws.com',
                  },
                  'Resource': '*',
                  'Condition': {
                    'ArnEquals': {
                      'aws:SourceArn': {
                        'Ref': 'MyTopic86869434',
                      },
                    },
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
  });
});

test('throws an error when a queue is encrypted by AWS managed KMS kye for queue subscription', () => {
  // WHEN
  const queue = new sqs.Queue(stack, 'MyQueue', {
    encryption: sqs.QueueEncryption.KMS_MANAGED,
  });

  // THEN
  expect(() => topic.addSubscription(new subs.SqsSubscription(queue)))
    .toThrow(/SQS queue encrypted by AWS managed KMS key cannot be used as SNS subscription/);
});

test('throws an error when a dead-letter queue is encrypted by AWS managed KMS kye for queue subscription', () => {
  // WHEN
  const queue = new sqs.Queue(stack, 'MyQueue');
  const dlq = new sqs.Queue(stack, 'MyDLQ', {
    encryption: sqs.QueueEncryption.KMS_MANAGED,
  });

  // THEN
  expect(() => topic.addSubscription(new subs.SqsSubscription(queue, {
    deadLetterQueue: dlq,
  })))
    .toThrow(/SQS queue encrypted by AWS managed KMS key cannot be used as dead-letter queue/);
});

test('importing SQS queue and specify this as subscription', () => {
  // WHEN
  const queue = sqs.Queue.fromQueueArn(stack, 'Queue', 'arn:aws:sqs:us-east-1:123456789012:queue1');
  topic.addSubscription(new subs.SqsSubscription(queue));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    'Endpoint': 'arn:aws:sqs:us-east-1:123456789012:queue1',
    'Protocol': 'sqs',
    'TopicArn': {
      'Ref': 'MyTopic86869434',
    },
  });
});

test('lambda subscription', () => {
  const func = new lambda.Function(stack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });

  topic.addSubscription(new subs.LambdaSubscription(func));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyFuncServiceRole54065130': {
        'Type': 'AWS::IAM::Role',
        'Properties': {
          'AssumeRolePolicyDocument': {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'ManagedPolicyArns': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    'Ref': 'AWS::Partition',
                  },
                  ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                ],
              ],
            },
          ],
        },
      },
      'MyFunc8A243A2C': {
        'Type': 'AWS::Lambda::Function',
        'Properties': {
          'Code': {
            'ZipFile': 'exports.handler = function(e, c, cb) { return cb() }',
          },
          'Handler': 'index.handler',
          'Role': {
            'Fn::GetAtt': [
              'MyFuncServiceRole54065130',
              'Arn',
            ],
          },
          'Runtime': 'nodejs24.x',
        },
        'DependsOn': [
          'MyFuncServiceRole54065130',
        ],
      },
      'MyFuncAllowInvokeMyTopicDD0A15B8': {
        'Type': 'AWS::Lambda::Permission',
        'Properties': {
          'Action': 'lambda:InvokeFunction',
          'FunctionName': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
          'Principal': 'sns.amazonaws.com',
          'SourceArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
      'MyFuncMyTopic93B6FB00': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'lambda',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

test('lambda subscription, cross region env agnostic', () => {
  const app = new App();
  const topicStack = new Stack(app, 'TopicStack', {});
  const lambdaStack = new Stack(app, 'LambdaStack', {});

  const topic1 = new sns.Topic(topicStack, 'Topic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });
  const func = new lambda.Function(lambdaStack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });

  topic1.addSubscription(new subs.LambdaSubscription(func));

  Template.fromStack(lambdaStack).templateMatches({
    'Resources': {
      'MyFuncServiceRole54065130': {
        'Type': 'AWS::IAM::Role',
        'Properties': {
          'AssumeRolePolicyDocument': {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'ManagedPolicyArns': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    'Ref': 'AWS::Partition',
                  },
                  ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                ],
              ],
            },
          ],
        },
      },
      'MyFunc8A243A2C': {
        'Type': 'AWS::Lambda::Function',
        'Properties': {
          'Code': {
            'ZipFile': 'exports.handler = function(e, c, cb) { return cb() }',
          },
          'Role': {
            'Fn::GetAtt': [
              'MyFuncServiceRole54065130',
              'Arn',
            ],
          },
          'Handler': 'index.handler',
          'Runtime': 'nodejs24.x',
        },
        'DependsOn': [
          'MyFuncServiceRole54065130',
        ],
      },
      'MyFuncAllowInvokeTopicStackTopicFBF76EB3D4A699EF': {
        'Type': 'AWS::Lambda::Permission',
        'Properties': {
          'Action': 'lambda:InvokeFunction',
          'FunctionName': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
          'Principal': 'sns.amazonaws.com',
          'SourceArn': {
            'Fn::ImportValue': 'TopicStack:ExportsOutputRefTopicBFC7AF6ECB4A357A',
          },
        },
      },
      'MyFuncTopic3B7C24C5': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'lambda',
          'TopicArn': {
            'Fn::ImportValue': 'TopicStack:ExportsOutputRefTopicBFC7AF6ECB4A357A',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

test('lambda subscription, cross region', () => {
  const app = new App();
  const topicStack = new Stack(app, 'TopicStack', {
    env: {
      account: '11111111111',
      region: 'us-east-1',
    },
  });
  const lambdaStack = new Stack(app, 'LambdaStack', {
    env: {
      account: '11111111111',
      region: 'us-east-2',
    },
  });

  const topic1 = new sns.Topic(topicStack, 'Topic', {
    topicName: 'topicName',
    displayName: 'displayName',
  });
  const func = new lambda.Function(lambdaStack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });

  topic1.addSubscription(new subs.LambdaSubscription(func));

  Template.fromStack(lambdaStack).templateMatches({
    'Resources': {
      'MyFuncServiceRole54065130': {
        'Type': 'AWS::IAM::Role',
        'Properties': {
          'AssumeRolePolicyDocument': {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'ManagedPolicyArns': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    'Ref': 'AWS::Partition',
                  },
                  ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                ],
              ],
            },
          ],
        },
      },
      'MyFunc8A243A2C': {
        'Type': 'AWS::Lambda::Function',
        'Properties': {
          'Code': {
            'ZipFile': 'exports.handler = function(e, c, cb) { return cb() }',
          },
          'Role': {
            'Fn::GetAtt': [
              'MyFuncServiceRole54065130',
              'Arn',
            ],
          },
          'Handler': 'index.handler',
          'Runtime': lambda.Runtime.NODEJS_LATEST.name,
        },
        'DependsOn': [
          'MyFuncServiceRole54065130',
        ],
      },
      'MyFuncAllowInvokeTopicStackTopicFBF76EB3D4A699EF': {
        'Type': 'AWS::Lambda::Permission',
        'Properties': {
          'Action': 'lambda:InvokeFunction',
          'FunctionName': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
          'Principal': 'sns.amazonaws.com',
          'SourceArn': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  'Ref': 'AWS::Partition',
                },
                ':sns:us-east-1:11111111111:topicName',
              ],
            ],
          },
        },
      },
      'MyFuncTopic3B7C24C5': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'lambda',
          'TopicArn': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  'Ref': 'AWS::Partition',
                },
                ':sns:us-east-1:11111111111:topicName',
              ],
            ],
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
          'Region': 'us-east-1',
        },
      },
    },
  });
});

test('email subscription', () => {
  topic.addSubscription(new subs.EmailSubscription('foo@bar.com'));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicfoobarcomA344CADA': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': 'foo@bar.com',
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
    },
  });
});

test('email subscription with unresolved', () => {
  const emailToken = Token.asString({ Ref: 'my-email-1' });
  topic.addSubscription(new subs.EmailSubscription(emailToken));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicTokenSubscription141DD1BE2': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-email-1',
          },
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
    },
  });
});

test('email and url subscriptions with unresolved', () => {
  const emailToken = Token.asString({ Ref: 'my-email-1' });
  const urlToken = Token.asString({ Ref: 'my-url-1' });
  topic.addSubscription(new subs.EmailSubscription(emailToken));
  topic.addSubscription(new subs.UrlSubscription(urlToken, { protocol: sns.SubscriptionProtocol.HTTPS }));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicTokenSubscription141DD1BE2': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-email-1',
          },
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
      'MyTopicTokenSubscription293BFE3F9': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-url-1',
          },
          'Protocol': 'https',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
    },
  });
});

test('email and url subscriptions with unresolved - four subscriptions', () => {
  const emailToken1 = Token.asString({ Ref: 'my-email-1' });
  const emailToken2 = Token.asString({ Ref: 'my-email-2' });
  const emailToken3 = Token.asString({ Ref: 'my-email-3' });
  const emailToken4 = Token.asString({ Ref: 'my-email-4' });

  topic.addSubscription(new subs.EmailSubscription(emailToken1));
  topic.addSubscription(new subs.EmailSubscription(emailToken2));
  topic.addSubscription(new subs.EmailSubscription(emailToken3));
  topic.addSubscription(new subs.EmailSubscription(emailToken4));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicTokenSubscription141DD1BE2': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-email-1',
          },
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
      'MyTopicTokenSubscription293BFE3F9': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-email-2',
          },
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
      'MyTopicTokenSubscription335C2B4CA': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-email-3',
          },
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
      'MyTopicTokenSubscription4DBE52A3F': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-email-4',
          },
          'Protocol': 'email',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
    },
  });
});

test('multiple subscriptions', () => {
  const queue = new sqs.Queue(stack, 'MyQueue');
  const func = new lambda.Function(stack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });

  topic.addSubscription(new subs.SqsSubscription(queue));
  topic.addSubscription(new subs.LambdaSubscription(func));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'DeletionPolicy': 'Delete',
        'UpdateReplacePolicy': 'Delete',
      },
      'MyQueuePolicy6BBEDDAC': {
        'Type': 'AWS::SQS::QueuePolicy',
        'Properties': {
          'PolicyDocument': {
            'Statement': [
              {
                'Action': 'sqs:SendMessage',
                'Condition': {
                  'ArnEquals': {
                    'aws:SourceArn': {
                      'Ref': 'MyTopic86869434',
                    },
                  },
                },
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'sns.amazonaws.com',
                },
                'Resource': {
                  'Fn::GetAtt': [
                    'MyQueueE6CA6235',
                    'Arn',
                  ],
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'Queues': [
            {
              'Ref': 'MyQueueE6CA6235',
            },
          ],
        },
      },
      'MyQueueMyTopic9B00631B': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sqs',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyQueueE6CA6235',
              'Arn',
            ],
          },
        },
      },
      'MyFuncServiceRole54065130': {
        'Type': 'AWS::IAM::Role',
        'Properties': {
          'AssumeRolePolicyDocument': {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'lambda.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          'ManagedPolicyArns': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    'Ref': 'AWS::Partition',
                  },
                  ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                ],
              ],
            },
          ],
        },
      },
      'MyFunc8A243A2C': {
        'Type': 'AWS::Lambda::Function',
        'Properties': {
          'Code': {
            'ZipFile': 'exports.handler = function(e, c, cb) { return cb() }',
          },
          'Handler': 'index.handler',
          'Role': {
            'Fn::GetAtt': [
              'MyFuncServiceRole54065130',
              'Arn',
            ],
          },
          'Runtime': 'nodejs24.x',
        },
        'DependsOn': [
          'MyFuncServiceRole54065130',
        ],
      },
      'MyFuncAllowInvokeMyTopicDD0A15B8': {
        'Type': 'AWS::Lambda::Permission',
        'Properties': {
          'Action': 'lambda:InvokeFunction',
          'FunctionName': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
          'Principal': 'sns.amazonaws.com',
          'SourceArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
      'MyFuncMyTopic93B6FB00': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'lambda',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': {
            'Fn::GetAtt': [
              'MyFunc8A243A2C',
              'Arn',
            ],
          },
        },
      },
    },
  });
});

test('throws with mutliple subscriptions of the same subscriber', () => {
  const queue = new sqs.Queue(stack, 'MyQueue');

  topic.addSubscription(new subs.SqsSubscription(queue));

  expect(() => topic.addSubscription(new subs.SqsSubscription(queue)))
    .toThrow(/A subscription with id \"MyTopic\" already exists under the scope Default\/MyQueue/);
});

test('with filter policy', () => {
  const func = new lambda.Function(stack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });

  topic.addSubscription(new subs.LambdaSubscription(func, {
    filterPolicy: {
      color: sns.SubscriptionFilter.stringFilter({
        allowlist: ['red'],
        matchPrefixes: ['bl', 'ye'],
        matchSuffixes: ['ue', 'ow'],
      }),
      size: sns.SubscriptionFilter.stringFilter({
        denylist: ['small', 'medium'],
      }),
      price: sns.SubscriptionFilter.numericFilter({
        between: { start: 100, stop: 200 },
      }),
    },
  }));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    'FilterPolicy': {
      'color': [
        'red',
        {
          'prefix': 'bl',
        },
        {
          'prefix': 'ye',
        },
        {
          'suffix': 'ue',
        },
        {
          'suffix': 'ow',
        },
      ],
      'size': [
        {
          'anything-but': [
            'small',
            'medium',
          ],
        },
      ],
      'price': [
        {
          'numeric': [
            '>=',
            100,
            '<=',
            200,
          ],
        },
      ],
    },
  });
});

test('with filter policy scope MessageBody', () => {
  const func = new lambda.Function(stack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });

  topic.addSubscription(new subs.LambdaSubscription(func, {
    filterPolicyWithMessageBody: {
      color: sns.FilterOrPolicy.policy({
        background: sns.FilterOrPolicy.filter(sns.SubscriptionFilter.stringFilter({
          allowlist: ['red'],
          matchPrefixes: ['bl', 'ye'],
          matchSuffixes: ['ue', 'ow'],
        })),
      }),
      size: sns.FilterOrPolicy.filter(sns.SubscriptionFilter.stringFilter({
        denylist: ['small', 'medium'],
      })),
    },
  }));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    'FilterPolicy': {
      'color': {
        'background': [
          'red',
          {
            'prefix': 'bl',
          },
          {
            'prefix': 'ye',
          },
          {
            'suffix': 'ue',
          },
          {
            'suffix': 'ow',
          },
        ],
      },
      'size': [
        {
          'anything-but': [
            'small',
            'medium',
          ],
        },
      ],
    },
    FilterPolicyScope: 'MessageBody',
  });
});

test('region property is present on an imported topic - sqs', () => {
  const imported = sns.Topic.fromTopicArn(stack, 'mytopic', 'arn:aws:sns:us-east-1:123456789012:mytopic');
  const queue = new sqs.Queue(stack, 'myqueue');
  imported.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    Region: 'us-east-1',
  });
});

test('region property on an imported topic as a parameter - sqs', () => {
  const topicArn = new CfnParameter(stack, 'topicArn');
  const imported = sns.Topic.fromTopicArn(stack, 'mytopic', topicArn.valueAsString);
  const queue = new sqs.Queue(stack, 'myqueue');
  imported.addSubscription(new subs.SqsSubscription(queue));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    Region: {
      'Fn::Select': [3, { 'Fn::Split': [':', { 'Ref': 'topicArn' }] }],
    },
  });
});

test('region property is present on an imported topic - lambda', () => {
  const imported = sns.Topic.fromTopicArn(stack, 'mytopic', 'arn:aws:sns:us-east-1:123456789012:mytopic');
  const func = new lambda.Function(stack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });
  imported.addSubscription(new subs.LambdaSubscription(func));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    Region: 'us-east-1',
  });
});

test('region property on an imported topic as a parameter - lambda', () => {
  const topicArn = new CfnParameter(stack, 'topicArn');
  const imported = sns.Topic.fromTopicArn(stack, 'mytopic', topicArn.valueAsString);
  const func = new lambda.Function(stack, 'MyFunc', {
    runtime: lambda.Runtime.NODEJS_LATEST,
    handler: 'index.handler',
    code: lambda.Code.fromInline('exports.handler = function(e, c, cb) { return cb() }'),
  });
  imported.addSubscription(new subs.LambdaSubscription(func));

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::Subscription', {
    Region: {
      'Fn::Select': [3, { 'Fn::Split': [':', { 'Ref': 'topicArn' }] }],
    },
  });
});

test('sms subscription', () => {
  topic.addSubscription(new subs.SmsSubscription('+15551231234'));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopic155512312349C8DEEEE': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Protocol': 'sms',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
          'Endpoint': '+15551231234',
        },
      },
    },
  });
});

test('sms subscription with unresolved', () => {
  const smsToken = Token.asString({ Ref: 'my-sms-1' });
  topic.addSubscription(new subs.SmsSubscription(smsToken));

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'MyTopic86869434': {
        'Type': 'AWS::SNS::Topic',
        'Properties': {
          'DisplayName': 'displayName',
          'TopicName': 'topicName',
        },
      },
      'MyTopicTokenSubscription141DD1BE2': {
        'Type': 'AWS::SNS::Subscription',
        'Properties': {
          'Endpoint': {
            'Ref': 'my-sms-1',
          },
          'Protocol': 'sms',
          'TopicArn': {
            'Ref': 'MyTopic86869434',
          },
        },
      },
    },
  });
});
