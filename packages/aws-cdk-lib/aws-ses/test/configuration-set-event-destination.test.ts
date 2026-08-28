import { Template } from '../../assertions';
import * as events from '../../aws-events';
import * as iam from '../../aws-iam';
import * as firehose from '../../aws-kinesisfirehose';
import * as sns from '../../aws-sns';
import { Stack } from '../../core';
import { CloudWatchDimensionSource, ConfigurationSet, ConfigurationSetEventDestination, EventDestination } from '../lib';

let stack: Stack;
let configurationSet: ConfigurationSet;
beforeEach(() => {
  stack = new Stack();
  configurationSet = new ConfigurationSet(stack, 'ConfigurationSet');
});

test('sns destination', () => {
  const topic = new sns.Topic(stack, 'Topic');

  new ConfigurationSetEventDestination(stack, 'Sns', {
    configurationSet,
    destination: EventDestination.snsTopic(topic),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::SES::ConfigurationSetEventDestination', {
    ConfigurationSetName: { Ref: 'ConfigurationSet3DD38186' },
    EventDestination: {
      Enabled: true,
      SnsDestination: { TopicARN: { Ref: 'TopicBFC7AF6E' } },
      MatchingEventTypes: [
        'send',
        'reject',
        'bounce',
        'complaint',
        'delivery',
        'open',
        'click',
        'renderingFailure',
        'deliveryDelay',
        'subscription',
      ],
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::SNS::TopicPolicy', {
    PolicyDocument: {
      Statement: [{
        Action: 'sns:Publish',
        Condition: {
          StringEquals: {
            'AWS:SourceAccount': { Ref: 'AWS::AccountId' },
            'AWS:SourceArn': {
              'Fn::Join': ['', [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':ses:',
                { Ref: 'AWS::Region' },
                ':',
                { Ref: 'AWS::AccountId' },
                ':configuration-set/',
                { Ref: 'ConfigurationSet3DD38186' },
              ]],
            },
          },
        },
        Effect: 'Allow',
        Principal: { Service: 'ses.amazonaws.com' },
        Resource: { Ref: 'TopicBFC7AF6E' },
      }],
    },
  });
});

test('cloudwatch dimensions destination', () => {
  new ConfigurationSetEventDestination(stack, 'Sns', {
    configurationSet,
    destination: EventDestination.cloudWatchDimensions([
      {
        source: CloudWatchDimensionSource.MESSAGE_TAG,
        name: 'ses:from-domain',
        defaultValue: 'no_domain',
      },
      {
        source: CloudWatchDimensionSource.MESSAGE_TAG,
        name: 'ses:source-ip',
        defaultValue: 'no_ip',
      },
    ]),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::SES::ConfigurationSetEventDestination', {
    ConfigurationSetName: { Ref: 'ConfigurationSet3DD38186' },
    EventDestination: {
      CloudWatchDestination: {
        DimensionConfigurations: [
          {
            DefaultDimensionValue: 'no_domain',
            DimensionName: 'ses:from-domain',
            DimensionValueSource: 'messageTag',
          },
          {
            DefaultDimensionValue: 'no_ip',
            DimensionName: 'ses:source-ip',
            DimensionValueSource: 'messageTag',
          },
        ],
      },
    },
  });
});

test('default bus as event bridge destination', () => {
  const bus = events.EventBus.fromEventBusName(stack, 'EventBus', 'default');
  new ConfigurationSetEventDestination(stack, 'Destination', {
    configurationSet,
    destination: EventDestination.eventBus(bus),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::SES::ConfigurationSetEventDestination', {
    ConfigurationSetName: { Ref: 'ConfigurationSet3DD38186' },
    EventDestination: {
      Enabled: true,
      EventBridgeDestination: {
        EventBusArn: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':events:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':event-bus/default',
          ]],
        },
      },
      MatchingEventTypes: [
        'send',
        'reject',
        'bounce',
        'complaint',
        'delivery',
        'open',
        'click',
        'renderingFailure',
        'deliveryDelay',
        'subscription',
      ],
    },
  });
});

test('throw if event bridge destination is not default event bus', () => {
  expect(() => {
    const bus = events.EventBus.fromEventBusName(stack, 'EventBus', 'test');
    new ConfigurationSetEventDestination(stack, 'Destination', {
      configurationSet,
      destination: EventDestination.eventBus(bus),
    });
  }).toThrow(/Only the default bus can be used as an event destination/);
});

test('kinesis firehose delivery stream destination only specify stream ARN', () => {
  const deliveryStream = firehose.DeliveryStream.fromDeliveryStreamName(stack, 'Firehose', 'my-deliverystream');

  new ConfigurationSetEventDestination(stack, 'Destination', {
    configurationSet,
    destination: EventDestination.firehoseDeliveryStream({
      deliveryStream,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::SES::ConfigurationSetEventDestination', {
    ConfigurationSetName: { Ref: 'ConfigurationSet3DD38186' },
    EventDestination: {
      Enabled: true,
      KinesisFirehoseDestination: {
        DeliveryStreamARN: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':firehose:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':deliverystream/my-deliverystream',
          ]],
        },
        IAMRoleARN: {
          'Fn::GetAtt': [
            'DestinationFirehoseDeliveryStreamIamRole8D1AB908',
            'Arn',
          ],
        },
      },
      MatchingEventTypes: [
        'send',
        'reject',
        'bounce',
        'complaint',
        'delivery',
        'open',
        'click',
        'renderingFailure',
        'deliveryDelay',
        'subscription',
      ],
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
    AssumeRolePolicyDocument: {
      Statement: [
        {
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Principal: {
            Service: 'ses.amazonaws.com',
          },
          Condition: {
            StringEquals: {
              'AWS:SourceAccount': {
                Ref: 'AWS::AccountId',
              },
              'AWS:SourceArn': {
                'Fn::Join': ['', [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':ses:',
                  { Ref: 'AWS::Region' },
                  ':',
                  { Ref: 'AWS::AccountId' },
                  ':configuration-set/',
                  { Ref: 'ConfigurationSet3DD38186' },
                ]],
              },
            },
          },
        },
      ],
      Version: '2012-10-17',
    },
    Policies: [
      {
        PolicyDocument: {
          Statement: [
            {
              Action: 'firehose:PutRecordBatch',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': ['', [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':firehose:',
                  { Ref: 'AWS::Region' },
                  ':',
                  { Ref: 'AWS::AccountId' },
                  ':deliverystream/my-deliverystream',
                ]],
              },
            },
          ],
          Version: '2012-10-17',
        },
      },
    ],
  });
});

test('kinesis firehose delivery stream destination specify stream ARN and IAM Role ARN', () => {
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('ses.amazonaws.com'),
  });

  const deliveryStream = firehose.DeliveryStream.fromDeliveryStreamName(stack, 'Firehose', 'my-deliverystream');

  new ConfigurationSetEventDestination(stack, 'Destination', {
    configurationSet,
    destination: EventDestination.firehoseDeliveryStream({
      deliveryStream,
      role: role,
    }),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::SES::ConfigurationSetEventDestination', {
    ConfigurationSetName: { Ref: 'ConfigurationSet3DD38186' },
    EventDestination: {
      Enabled: true,
      KinesisFirehoseDestination: {
        DeliveryStreamARN: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':firehose:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':deliverystream/my-deliverystream',
          ]],
        },
        IAMRoleARN: {
          'Fn::GetAtt': [
            'Role1ABCC5F0',
            'Arn',
          ],
        },
      },
      MatchingEventTypes: [
        'send',
        'reject',
        'bounce',
        'complaint',
        'delivery',
        'open',
        'click',
        'renderingFailure',
        'deliveryDelay',
        'subscription',
      ],
    },
  });
});

test('configurationSetEventDestinationRef of a destination imported by attributes is fully readable', () => {
  // WHEN
  const destination = ConfigurationSetEventDestination.fromConfigurationSetEventDestinationAttributes(stack, 'Imported', {
    configurationSet: ConfigurationSet.fromConfigurationSetName(stack, 'ImportedSet', 'MySet'),
    configurationSetEventDestinationId: 'MyDestination',
  });

  // THEN
  expect(destination.configurationSetEventDestinationRef.configurationSetEventDestinationId).toEqual('MyDestination');
  expect(destination.configurationSetEventDestinationRef.configurationSetName).toEqual('MySet');
});

test('fails to read configurationSetName of a destination imported by id', () => {
  // WHEN
  const destination = ConfigurationSetEventDestination.fromConfigurationSetEventDestinationId(stack, 'Imported', 'MyDestination');

  // THEN
  expect(() => destination.configurationSetEventDestinationRef.configurationSetName).toThrow(
    'configurationSetName is not available on a ConfigurationSetEventDestination imported by id; use ConfigurationSetEventDestination.fromConfigurationSetEventDestinationAttributes() to get a complete reference',
  );
});
