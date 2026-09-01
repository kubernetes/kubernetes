import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import * as sqs from '../../aws-sqs';
import { Aws, CfnResource, Stack, Arn, App, PhysicalName, CfnOutput } from '../../core';
import { EventBus, IncludeDetail, Level } from '../lib';

describe('event bus', () => {
  test('default event bus', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new EventBus(stack, 'Bus');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'Bus',
    });
  });

  test('default event bus with logConfig', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new EventBus(stack, 'Bus', {
      logConfig: {
        includeDetail: IncludeDetail.FULL,
        level: Level.TRACE,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'Bus',
      LogConfig: {
        IncludeDetail: 'FULL',
        Level: 'TRACE',
      },
    });
  });

  test('default event bus with empty props object', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new EventBus(stack, 'Bus', {});

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'Bus',
    });
  });

  test('named event bus', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new EventBus(stack, 'Bus', {
      eventBusName: 'myEventBus',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'myEventBus',
    });
  });

  test('event bus with description', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new EventBus(stack, 'myEventBus', {
      description: 'myEventBusDescription',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Description: 'myEventBusDescription',
    });
  });

  test('partner event bus', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new EventBus(stack, 'Bus', {
      eventSourceName: 'aws.partner/PartnerName/acct1/repo1',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'aws.partner/PartnerName/acct1/repo1',
      EventSourceName: 'aws.partner/PartnerName/acct1/repo1',
    });
  });

  test('imported event bus', () => {
    const stack = new Stack();

    const eventBus = new EventBus(stack, 'Bus');

    const importEB = EventBus.fromEventBusArn(stack, 'ImportBus', eventBus.eventBusArn);

    // WHEN
    new CfnResource(stack, 'Res', {
      type: 'Test::Resource',
      properties: {
        EventBusArn1: eventBus.eventBusArn,
        EventBusArn2: importEB.eventBusArn,
      },
    });

    Template.fromStack(stack).hasResourceProperties('Test::Resource', {
      EventBusArn1: { 'Fn::GetAtt': ['BusEA82B648', 'Arn'] },
      EventBusArn2: { 'Fn::GetAtt': ['BusEA82B648', 'Arn'] },
    });
  });

  test('imported event bus from name', () => {
    const stack = new Stack();

    const eventBus = new EventBus(stack, 'Bus', { eventBusName: 'test-bus-to-import-by-name' });

    const importEB = EventBus.fromEventBusName(stack, 'ImportBus', eventBus.eventBusName);

    // WHEN
    expect(stack.resolve(eventBus.eventBusName)).toEqual(stack.resolve(importEB.eventBusName));
  });

  test('same account imported event bus has right resource env', () => {
    const stack = new Stack();

    const eventBus = new EventBus(stack, 'Bus');

    const importEB = EventBus.fromEventBusArn(stack, 'ImportBus', eventBus.eventBusArn);

    // WHEN
    expect(stack.resolve(importEB.env.account)).toEqual({ 'Fn::Select': [4, { 'Fn::Split': [':', { 'Fn::GetAtt': ['BusEA82B648', 'Arn'] }] }] });
    expect(stack.resolve(importEB.env.region)).toEqual({ 'Fn::Select': [3, { 'Fn::Split': [':', { 'Fn::GetAtt': ['BusEA82B648', 'Arn'] }] }] });
  });

  test('cross account imported event bus has right resource env', () => {
    const stack = new Stack();

    const arnParts = {
      resource: 'bus',
      service: 'events',
      account: 'myAccount',
      region: 'us-west-1',
    };

    const arn = Arn.format(arnParts, stack);

    const importEB = EventBus.fromEventBusArn(stack, 'ImportBus', arn);

    // WHEN
    expect(importEB.env.account).toEqual(arnParts.account);
    expect(importEB.env.region).toEqual(arnParts.region);
  });

  test('can get bus name', () => {
    // GIVEN
    const stack = new Stack();
    const bus = new EventBus(stack, 'Bus', {
      eventBusName: 'myEventBus',
    });

    // WHEN
    new CfnResource(stack, 'Res', {
      type: 'Test::Resource',
      properties: {
        EventBusName: bus.eventBusName,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Test::Resource', {
      EventBusName: { Ref: 'BusEA82B648' },
    });
  });

  test('can get bus arn', () => {
    // GIVEN
    const stack = new Stack();
    const bus = new EventBus(stack, 'Bus', {
      eventBusName: 'myEventBus',
    });

    // WHEN
    new CfnResource(stack, 'Res', {
      type: 'Test::Resource',
      properties: {
        EventBusArn: bus.eventBusArn,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Test::Resource', {
      EventBusArn: { 'Fn::GetAtt': ['BusEA82B648', 'Arn'] },
    });
  });

  test('event bus name cannot be default', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const createInvalidBus = () => new EventBus(stack, 'Bus', {
      eventBusName: 'default',
    });

    // THEN
    expect(() => {
      createInvalidBus();
    }).toThrow(/'eventBusName' must not be 'default'/);
  });

  test('event bus name cannot contain slash', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const createInvalidBus = () => new EventBus(stack, 'Bus', {
      eventBusName: 'my/bus',
    });

    // THEN
    expect(() => {
      createInvalidBus();
    }).toThrow(/'eventBusName' must not contain '\/'/);
  });

  test('event bus cannot have name and source name', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const createInvalidBus = () => new EventBus(stack, 'Bus', {
      eventBusName: 'myBus',
      eventSourceName: 'myBus',
    });

    // THEN
    expect(() => {
      createInvalidBus();
    }).toThrow(/'eventBusName' and 'eventSourceName' cannot both be provided/);
  });

  test('event bus name cannot be empty string', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const createInvalidBus = () => new EventBus(stack, 'Bus', {
      eventBusName: '',
    });

    // THEN
    expect(() => {
      createInvalidBus();
    }).toThrow(/'eventBusName' must satisfy: /);
  });

  test('does not throw if eventBusName is a token', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN / THEN
    expect(() => new EventBus(stack, 'EventBus', {
      eventBusName: Aws.STACK_NAME,
    })).not.toThrow();
  });

  test('event bus source name must follow pattern', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const createInvalidBus = () => new EventBus(stack, 'Bus', {
      eventSourceName: 'invalid-partner',
    });

    // THEN
    expect(() => {
      createInvalidBus();
    }).toThrow(/'eventSourceName' must satisfy: \/\^aws/);
  });

  test('does not throw if eventSourceName is a token', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN / THEN
    expect(() => new EventBus(stack, 'EventBus', {
      eventSourceName: Aws.STACK_NAME,
    })).not.toThrow();
  });

  test('event bus source name cannot be empty string', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const createInvalidBus = () => new EventBus(stack, 'Bus', {
      eventSourceName: '',
    });

    // THEN
    expect(() => {
      createInvalidBus();
    }).toThrow(/'eventSourceName' must satisfy: /);
  });

  test('event bus description cannot be too long', () => {
    // GIVEN
    const stack = new Stack();
    const tooLongDescription = 'a'.repeat(513);

    // WHEN / THEN
    expect(() => {
      new EventBus(stack, 'EventBusWithTooLongDescription', {
        description: tooLongDescription,
      });
    }).toThrow('description must be less than or equal to 512 characters, got 513');
  });

  testDeprecated('can grant PutEvents', () => {
    // GIVEN
    const stack = new Stack();
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    EventBus.grantPutEvents(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'events:PutEvents',
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      Roles: [
        {
          Ref: 'Role1ABCC5F0',
        },
      ],
    });
  });

  test('can grant PutEvents using grantAllPutEvents', () => {
    // GIVEN
    const stack = new Stack();
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    EventBus.grantAllPutEvents(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'events:PutEvents',
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      Roles: [
        {
          Ref: 'Role1ABCC5F0',
        },
      ],
    });
  });

  test('can grant PutEvents to a specific event bus', () => {
    // GIVEN
    const stack = new Stack();
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    const eventBus = new EventBus(stack, 'EventBus');

    // WHEN
    eventBus.grantPutEventsTo(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'events:PutEvents',
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': [
                'EventBus7B8748AA',
                'Arn',
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
      Roles: [
        {
          Ref: 'Role1ABCC5F0',
        },
      ],
    });
  });

  test('can archive events', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const event = new EventBus(stack, 'Bus');

    event.archive('MyArchive', {
      eventPattern: {
        account: [stack.account],
      },
      archiveName: 'MyArchive',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'Bus',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Events::Archive', {
      SourceArn: {
        'Fn::GetAtt': [
          'BusEA82B648',
          'Arn',
        ],
      },
      Description: {
        'Fn::Join': [
          '',
          [
            'Event Archive for ',
            {
              Ref: 'BusEA82B648',
            },
            ' Event Bus',
          ],
        ],
      },
      EventPattern: {
        account: [
          {
            Ref: 'AWS::AccountId',
          },
        ],
      },
      RetentionDays: 0,
      ArchiveName: 'MyArchive',
    });
  });

  test('can archive events from an imported EventBus', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const bus = new EventBus(stack, 'Bus');

    const importedBus = EventBus.fromEventBusArn(stack, 'ImportedBus', bus.eventBusArn);

    importedBus.archive('MyArchive', {
      eventPattern: {
        account: [stack.account],
      },
      archiveName: 'MyArchive',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'Bus',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Events::Archive', {
      SourceArn: {
        'Fn::GetAtt': [
          'BusEA82B648',
          'Arn',
        ],
      },
      Description: {
        'Fn::Join': [
          '',
          [
            'Event Archive for ',
            {
              'Fn::Select': [
                1,
                {
                  'Fn::Split': [
                    '/',
                    {
                      'Fn::Select': [
                        5,
                        {
                          'Fn::Split': [
                            ':',
                            {
                              'Fn::GetAtt': [
                                'BusEA82B648',
                                'Arn',
                              ],
                            },
                          ],
                        },
                      ],
                    },
                  ],
                },
              ],
            },
            ' Event Bus',
          ],
        ],
      },
      EventPattern: {
        account: [
          {
            Ref: 'AWS::AccountId',
          },
        ],
      },
      RetentionDays: 0,
      ArchiveName: 'MyArchive',
    });
  });

  test('cross account event bus uses generated physical name', () => {
    // GIVEN
    const app = new App();
    const stack1 = new Stack(app, 'Stack1', {
      env: {
        account: '111111111111',
        region: 'us-east-1',
      },
    });
    const stack2 = new Stack(app, 'Stack2', {
      env: {
        account: '222222222222',
        region: 'us-east-1',
      },
    });

    // WHEN
    const bus1 = new EventBus(stack1, 'Bus', {
      eventBusName: PhysicalName.GENERATE_IF_NEEDED,
    });

    new CfnOutput(stack2, 'BusName', { value: bus1.eventBusName });

    // THEN
    Template.fromStack(stack1).hasResourceProperties('AWS::Events::EventBus', {
      Name: 'stack1stack1busca19bdf823d8f39f1c0f',
    });
  });

  test('can add one event bus policy', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const bus = new EventBus(stack, 'Bus');

    // WHEN
    bus.addToResourcePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      principals: [new iam.AccountPrincipal('111111111111')],
      actions: ['events:PutEvents'],
      sid: '123',
      resources: [bus.eventBusArn],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBusPolicy', {
      StatementId: 'cdk-123',
      EventBusName: {
        Ref: 'BusEA82B648',
      },
      Statement: {
        Action: 'events:PutEvents',
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
                ':iam::111111111111:root',
              ],
            ],
          },
        },
        Sid: 'cdk-123',
        Resource: {
          'Fn::GetAtt': [
            'BusEA82B648',
            'Arn',
          ],
        },
      },
    });
  });

  test('can add more than one event bus policy', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const bus = new EventBus(stack, 'Bus');

    const statement1 = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      principals: [new iam.ArnPrincipal('arn')],
      actions: ['events:PutEvents'],
      sid: 'statement1',
      resources: [bus.eventBusArn],
    });

    const statement2 = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      principals: [new iam.ArnPrincipal('arn')],
      actions: ['events:DeleteRule'],
      sid: 'statement2',
      resources: [bus.eventBusArn],
    });

    // WHEN
    const add1 = bus.addToResourcePolicy(statement1);
    const add2 = bus.addToResourcePolicy(statement2);

    // THEN
    expect(add1.statementAdded).toBe(true);
    expect(add2.statementAdded).toBe(true);
    Template.fromStack(stack).resourceCountIs('AWS::Events::EventBusPolicy', 2);
  });

  test('Event Bus policy statements must have a sid', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const bus = new EventBus(stack, 'Bus');

    // THEN
    expect(() => bus.addToResourcePolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      principals: [new iam.ArnPrincipal('arn')],
      actions: ['events:PutEvents'],
    }))).toThrow('Event Bus policy statements must have a sid');
  });

  test('set dead letter queue', () => {
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const dlq = new sqs.Queue(stack, 'DLQ');
    new EventBus(stack, 'Bus', {
      deadLetterQueue: dlq,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      DeadLetterConfig: {
        Arn: {
          'Fn::GetAtt': ['DLQ581697C4', 'Arn'],
        },
      },
    });
  });
  test('Event Bus with a customer managed key', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack');
    const key = new kms.Key(stack, 'Key');

    // WHEN
    new EventBus(stack, 'Bus', {
      kmsKey: key,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBus', {
      KmsKeyIdentifier: stack.resolve(key.keyArn),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Statement: [
          {
            Action: 'kms:*',
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
                    ':iam::',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':root',
                  ],
                ],
              },
            },
            Resource: '*',
          },
          {
            Action: [
              'kms:Decrypt',
              'kms:GenerateDataKey',
              'kms:DescribeKey',
            ],
            Condition: {
              StringEquals: {
                'aws:SourceAccount': {
                  Ref: 'AWS::AccountId',
                },
                'aws:SourceArn': {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':events:',
                      {
                        Ref: 'AWS::Region',
                      },
                      ':',
                      {
                        Ref: 'AWS::AccountId',
                      },
                      ':event-bus/StackBusAA0A1E4B',
                    ],
                  ],
                },
                'kms:EncryptionContext:aws:events:event-bus:arn': {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        Ref: 'AWS::Partition',
                      },
                      ':events:',
                      {
                        Ref: 'AWS::Region',
                      },
                      ':',
                      {
                        Ref: 'AWS::AccountId',
                      },
                      ':event-bus/StackBusAA0A1E4B',
                    ],
                  ],
                },
              },
            },
            Effect: 'Allow',
            Principal: {
              Service: 'events.amazonaws.com',
            },
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
    });
  });
});
