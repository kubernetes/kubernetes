import { Match, Template } from '../../assertions';
import * as iam from '../../aws-iam/index';
import { Stack } from '../../core';
import type { IStreamConsumer } from '../lib';
import { Stream, StreamConsumer } from '../lib';

describe('Kinesis stream consumer', () => {
  let stack: Stack;
  let stream: Stream;

  beforeEach(() => {
    stack = new Stack();
    stream = new Stream(stack, 'Stream', {});
  });

  describe('stream consumer from attributes', () => {
    let consumer: IStreamConsumer;

    beforeEach(() => {
      consumer = StreamConsumer.fromStreamConsumerAttributes(stack, 'MyConsumer', {
        streamConsumerArn: 'arn:aws:kinesis:us-east-1:123456789012:stream/stream-name/consumer/consumer-name:123456',
      });
    });

    test('has expected properties', () => {
      expect(consumer.streamConsumerArn).toEqual('arn:aws:kinesis:us-east-1:123456789012:stream/stream-name/consumer/consumer-name:123456');
      expect(consumer.streamConsumerName).toEqual('consumer-name');
      expect(consumer.stream.streamArn).toEqual('arn:aws:kinesis:us-east-1:123456789012:stream/stream-name');
    });

    test('addToResourcePolicy is a no-op', () => {
      consumer.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['kinesis:DescribeStreamConsumer'],
        principals: [new iam.AnyPrincipal()],
        resources: [consumer.streamConsumerArn],
      }));

      Template.fromStack(stack).resourceCountIs('AWS::Kinesis::ResourcePolicy', 0);
    });

    test('grantRead grants stream and consumer permissions', () => {
      const user = new iam.User(stack, 'MyUser');
      consumer.grantRead(user);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        Users: [stack.resolve(user.userName)],
        PolicyDocument: {
          Statement: Match.arrayWith([
            // stream read permissions
            {
              Action: [
                'kinesis:DescribeStreamSummary',
                'kinesis:GetRecords',
                'kinesis:GetShardIterator',
                'kinesis:ListShards',
                'kinesis:SubscribeToShard',
                'kinesis:DescribeStream',
                'kinesis:ListStreams',
                'kinesis:DescribeStreamConsumer',
              ],
              Effect: 'Allow',
              Resource: 'arn:aws:kinesis:us-east-1:123456789012:stream/stream-name',
            },
            // consumer read permissions
            {
              Action: [
                'kinesis:DescribeStreamConsumer',
                'kinesis:SubscribeToShard',
              ],
              Effect: 'Allow',
              Resource: 'arn:aws:kinesis:us-east-1:123456789012:stream/stream-name/consumer/consumer-name:123456',
            },
          ]),
        },
      });
      Template.fromStack(stack).resourceCountIs('AWS::Kinesis::ResourcePolicy', 0);
    });
  });

  describe('new stream consumer', () => {
    let consumer: StreamConsumer;

    beforeEach(() => {
      consumer = new StreamConsumer(stack, 'StreamConsumer', {
        streamConsumerName: 'MyStreamConsumer',
        stream,
      });
    });

    test('creates stream consumer resource', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Kinesis::StreamConsumer', {
        ConsumerName: 'MyStreamConsumer',
        StreamARN: stack.resolve(stream.streamArn),
      });
    });

    test('addToResourcePolicy creates a consumer resource policy ', () => {
      consumer.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['kinesis:DescribeStreamConsumer'],
        principals: [new iam.AnyPrincipal()],
        resources: [consumer.streamConsumerArn],
      }));

      Template.fromStack(stack).hasResourceProperties('AWS::Kinesis::ResourcePolicy', {
        ResourceArn: stack.resolve(consumer.streamConsumerArn),
        ResourcePolicy: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 'kinesis:DescribeStreamConsumer',
              Effect: 'Allow',
              Principal: { AWS: '*' },
              Resource: stack.resolve(consumer.streamConsumerArn),
            },
          ],
        },
      });
    });

    test('grantRead grants stream and consumer permissions', () => {
      const user = new iam.User(stack, 'MyUser');
      consumer.grantRead(user);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        Users: [stack.resolve(user.userName)],
        PolicyDocument: {
          Statement: Match.arrayWith([
            // stream read permissions
            {
              Action: [
                'kinesis:DescribeStreamSummary',
                'kinesis:GetRecords',
                'kinesis:GetShardIterator',
                'kinesis:ListShards',
                'kinesis:SubscribeToShard',
                'kinesis:DescribeStream',
                'kinesis:ListStreams',
                'kinesis:DescribeStreamConsumer',
              ],
              Effect: 'Allow',
              Resource: stack.resolve(stream.streamArn),
            },
            // consumer read permissions
            {
              Action: [
                'kinesis:DescribeStreamConsumer',
                'kinesis:SubscribeToShard',
              ],
              Effect: 'Allow',
              Resource: stack.resolve(consumer.streamConsumerArn),
            },
          ]),
        },
      });
    });

    test('grantRead when stream/consumer and grantee are in different accounts', () => {
      const stackA = new Stack(undefined, 'StackA', { env: { account: '123456789012' } });
      const streamFromStackA = new Stream(stackA, 'Stream', {
        streamName: 'MyStream',
      });
      const streamConsumerFromStackA = new StreamConsumer(stackA, 'StreamConsumer', {
        streamConsumerName: 'MyStreamConsumer',
        stream: streamFromStackA,
      });

      const stackB = new Stack(undefined, 'StackB', { env: { account: '234567890123' } });
      const roleFromStackB = new iam.Role(stackB, 'MyRole', {
        assumedBy: new iam.AccountPrincipal('234567890123'),
        roleName: 'MyRole',
      });

      streamConsumerFromStackA.grantRead(roleFromStackB);

      // Grantee stack has the correct IAM Policy
      Template.fromStack(stackB).hasResourceProperties('AWS::IAM::Policy', {
        Roles: [stackB.resolve(roleFromStackB.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([
            // stream read permissions
            {
              Action: [
                'kinesis:DescribeStreamSummary',
                'kinesis:GetRecords',
                'kinesis:GetShardIterator',
                'kinesis:ListShards',
                'kinesis:SubscribeToShard',
                'kinesis:DescribeStream',
                'kinesis:ListStreams',
                'kinesis:DescribeStreamConsumer',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': ['',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':kinesis:',
                    { Ref: 'AWS::Region' },
                    ':123456789012:stream/MyStream',
                  ]],
              },
            },
            // consumer read permissions
            {
              Action: [
                'kinesis:DescribeStreamConsumer',
                'kinesis:SubscribeToShard',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': ['',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':kinesis:',
                    { Ref: 'AWS::Region' },
                    ':123456789012:stream/MyStream/consumer/MyStreamConsumer:*',
                  ]],
              },
            },
          ]),
        },
      });

      // Stream stack
      // - has the correct Stream resource policy
      Template.fromStack(stackA).hasResourceProperties('AWS::Kinesis::ResourcePolicy', {
        ResourceArn: {
          'Fn::GetAtt': ['Stream790BDEE4', 'Arn'],
        },
        ResourcePolicy: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: [
                'kinesis:DescribeStreamSummary',
                'kinesis:GetRecords',
                'kinesis:GetShardIterator',
                'kinesis:ListShards',
                'kinesis:DescribeStream',
              ],
              Effect: 'Allow',
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':iam::234567890123:role/MyRole',
                    ],
                  ],
                },
              },
              Resource: {
                'Fn::GetAtt': ['Stream790BDEE4', 'Arn'],
              },
            },
          ],
        },
      });
      // - has the correct StreamConsumer resource policy
      Template.fromStack(stackA).hasResourceProperties('AWS::Kinesis::ResourcePolicy', {
        ResourceArn: {
          'Fn::GetAtt': ['StreamConsumer58240CBA', 'ConsumerARN'],
        },
        ResourcePolicy: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: [
                'kinesis:DescribeStreamConsumer',
                'kinesis:SubscribeToShard',
              ],
              Effect: 'Allow',
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':iam::234567890123:role/MyRole',
                    ],
                  ],
                },
              },
              Resource: {
                'Fn::GetAtt': ['StreamConsumer58240CBA', 'ConsumerARN'],
              },
            },
          ],
        },
      });
    });
  });
});
