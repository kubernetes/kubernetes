import { Match, Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import { CfnParameter, Duration, Stack, App, Token } from '../../core';
import * as sqs from '../lib';
import { QueueGrants } from '../lib';
import { validateRedriveAllowPolicy } from '../lib/validate-queue-props';

/* eslint-disable @stylistic/quote-props */

test('default properties', () => {
  const stack = new Stack();
  const q = new sqs.Queue(stack, 'Queue');

  expect(q.fifo).toEqual(false);

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'Queue4A7E3555': {
        'Type': 'AWS::SQS::Queue',
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });

  Template.fromStack(stack).hasResource('AWS::SQS::Queue', {
    DeletionPolicy: 'Delete',
  });
});

test('with a dead letter queue', () => {
  const stack = new Stack();
  const dlq = new sqs.Queue(stack, 'DLQ');
  const dlqProps = { queue: dlq, maxReceiveCount: 3 };
  const queue = new sqs.Queue(stack, 'Queue', { deadLetterQueue: dlqProps });

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'DLQ581697C4': {
        'Type': 'AWS::SQS::Queue',
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
      'Queue4A7E3555': {
        'Type': 'AWS::SQS::Queue',
        'Properties': {
          'RedrivePolicy': {
            'deadLetterTargetArn': {
              'Fn::GetAtt': [
                'DLQ581697C4',
                'Arn',
              ],
            },
            'maxReceiveCount': 3,
          },
        },
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });

  expect(queue.deadLetterQueue).toEqual(dlqProps);
});

test('multiple prop validation errors are presented to the user (out-of-range retentionPeriod and deliveryDelay)', () => {
  // GIVEN
  const stack = new Stack();

  // THEN
  expect(() => new sqs.Queue(stack, 'MyQueue', {
    retentionPeriod: Duration.seconds(30),
    deliveryDelay: Duration.minutes(16),
  })).toThrow('Queue initialization failed due to the following validation error(s):\n- delivery delay must be between 0 and 900 seconds, but 960 was provided\n- message retention period must be between 60 and 1,209,600 seconds, but 30 was provided');
});

test('message retention period must be between 1 minute to 14 days', () => {
  // GIVEN
  const stack = new Stack();

  // THEN
  expect(() => new sqs.Queue(stack, 'MyQueue', {
    retentionPeriod: Duration.seconds(30),
  })).toThrow('Queue initialization failed due to the following validation error(s):\n- message retention period must be between 60 and 1,209,600 seconds, but 30 was provided');

  expect(() => new sqs.Queue(stack, 'AnotherQueue', {
    retentionPeriod: Duration.days(15),
  })).toThrow('Queue initialization failed due to the following validation error(s):\n- message retention period must be between 60 and 1,209,600 seconds, but 1296000 was provided');
});

test('message retention period can be provided as a parameter', () => {
  // GIVEN
  const stack = new Stack();
  const parameter = new CfnParameter(stack, 'my-retention-period', {
    type: 'Number',
    default: 60,
  });

  // WHEN
  new sqs.Queue(stack, 'MyQueue', {
    retentionPeriod: Duration.seconds(parameter.valueAsNumber),
  });

  // THEN
  Template.fromStack(stack).templateMatches({
    'Parameters': {
      'myretentionperiod': {
        'Type': 'Number',
        'Default': 60,
      },
    },
    'Resources': {
      'MyQueueE6CA6235': {
        'Type': 'AWS::SQS::Queue',
        'Properties': {
          'MessageRetentionPeriod': {
            'Ref': 'myretentionperiod',
          },
        },
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });
});

test.each([
  { size: 1023, valid: false, description: 'just below lower bound' },
  { size: 1024, valid: true, description: 'at lower bound' },
  { size: 1048576, valid: true, description: 'at upper bound' },
  { size: 1048577, valid: false, description: 'just above upper bound' },
])('maxMessageSizeBytes validation for $size bytes ($description)', ({ size, valid }) => {
  const stack = new Stack();
  const constructId = `QueueWithSize${size}`;
  const action = () => new sqs.Queue(stack, constructId, { maxMessageSizeBytes: size });

  if (valid) {
    expect(action).not.toThrow();
  } else {
    expect(action).toThrow(`Queue initialization failed due to the following validation error(s):\n- maximum message size must be between 1,024 and 1,048,576 bytes, but ${size} was provided`);
  }
});

test('maxMessageSizeBytes works with CDK tokens', () => {
  const stack = new Stack();
  const parameter = new CfnParameter(stack, 'MessageSize', { type: 'Number' });

  // Should not throw for tokens (validation skipped)
  expect(() => new sqs.Queue(stack, 'TokenQueue', {
    maxMessageSizeBytes: parameter.valueAsNumber,
  })).not.toThrow();
});

test('multiple validation errors include maxMessageSizeBytes', () => {
  const stack = new Stack();

  expect(() => new sqs.Queue(stack, 'MultiError', {
    maxMessageSizeBytes: 2000000,
    retentionPeriod: Duration.seconds(30),
  })).toThrow(/maximum message size must be between 1,024 and 1,048,576 bytes.*message retention period must be between 60 and 1,209,600 seconds/s);
});

test('maxMessageSizeBytes synthesizes correct CloudFormation', () => {
  const stack = new Stack();

  new sqs.Queue(stack, 'LargeMessageQueue', {
    maxMessageSizeBytes: 1048576,
  });

  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::SQS::Queue', {
    MaximumMessageSize: 1048576,
  });
});

test('addToPolicy will automatically create a policy for this queue', () => {
  const stack = new Stack();
  const queue = new sqs.Queue(stack, 'MyQueue');
  queue.addToResourcePolicy(new iam.PolicyStatement({
    resources: ['*'],
    actions: ['sqs:*'],
    principals: [new iam.ArnPrincipal('arn')],
  }));

  Template.fromStack(stack).templateMatches({
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
                'Action': 'sqs:*',
                'Effect': 'Allow',
                'Principal': {
                  'AWS': 'arn',
                },
                'Resource': '*',
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
    },
  });
});

describe('validateRedriveAllowPolicy', () => {
  test('does not throw for valid policy', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const redriveAllowPolicy = { redrivePermission: sqs.RedrivePermission.ALLOW_ALL };

    // THEN
    expect(() => validateRedriveAllowPolicy(stack, redriveAllowPolicy)).not.toThrow();
  });

  test('throws when sourceQueues is provided with ALLOW_ALL permission', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const sourceQueue = new sqs.Queue(stack, 'SourceQueue');
    const redriveAllowPolicy = {
      redrivePermission: sqs.RedrivePermission.ALLOW_ALL,
      sourceQueues: [sourceQueue],
    };

    // THEN
    expect(() => validateRedriveAllowPolicy(stack, redriveAllowPolicy))
      .toThrow("Queue initialization failed due to the following validation error(s):\n- sourceQueues cannot be configured when RedrivePermission is set to 'allowAll' or 'denyAll'");
  });

  test('throws when sourceQueues is not provided with BY_QUEUE permission', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const redriveAllowPolicy = {
      redrivePermission: sqs.RedrivePermission.BY_QUEUE,
    };

    // THEN
    expect(() => validateRedriveAllowPolicy(stack, redriveAllowPolicy))
      .toThrow("Queue initialization failed due to the following validation error(s):\n- At least one source queue must be specified when RedrivePermission is set to 'byQueue'");
  });

  test('throws when more than 10 sourceQueues are provided', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const sourceQueues = Array(11).fill(null).map((_, i) => new sqs.Queue(stack, `SourceQueue${i}`));
    const redriveAllowPolicy = {
      redrivePermission: sqs.RedrivePermission.BY_QUEUE,
      sourceQueues,
    };

    // THEN
    expect(() => validateRedriveAllowPolicy(stack, redriveAllowPolicy))
      .toThrow("Queue initialization failed due to the following validation error(s):\n- Up to 10 sourceQueues can be specified. Set RedrivePermission to 'allowAll' to specify more");
  });
});

describe('export and import', () => {
  test('importing works correctly', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const imports = sqs.Queue.fromQueueArn(stack, 'Imported', 'arn:aws:sqs:us-east-1:123456789012:queue1');

    // THEN

    // "import" returns an IQueue bound to `Fn::ImportValue`s.
    expect(stack.resolve(imports.queueArn)).toEqual('arn:aws:sqs:us-east-1:123456789012:queue1');
    expect(stack.resolve(imports.queueUrl)).toEqual({
      'Fn::Join':
      ['', ['https://sqs.us-east-1.', { Ref: 'AWS::URLSuffix' }, '/123456789012/queue1']],
    });
    expect(stack.resolve(imports.queueName)).toEqual('queue1');
  });

  test('importing fifo and standard queues are detected correctly', () => {
    const stack = new Stack();
    const stdQueue = sqs.Queue.fromQueueArn(stack, 'StdQueue', 'arn:aws:sqs:us-east-1:123456789012:queue1');
    const fifoQueue = sqs.Queue.fromQueueArn(stack, 'FifoQueue', 'arn:aws:sqs:us-east-1:123456789012:queue2.fifo');
    expect(stdQueue.fifo).toEqual(false);
    expect(fifoQueue.fifo).toEqual(true);
  });

  test('importing with keyArn and encryptionType is set correctly', () => {
    const stack = new Stack();
    const queue = sqs.Queue.fromQueueAttributes(stack, 'Queue', {
      queueArn: 'arn:aws:sqs:us-east-1:123456789012:queue1',
      keyArn: 'arn:aws:kms:us-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab',
    });
    expect(queue.encryptionType).toEqual(sqs.QueueEncryption.KMS);
  });

  test('import queueArn from token, fifo and standard queues can be defined', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const stdQueue1 = sqs.Queue.fromQueueAttributes(stack, 'StdQueue1', {
      queueArn: Token.asString({ Ref: 'ARN' }),
    });
    const stdQueue2 = sqs.Queue.fromQueueAttributes(stack, 'StdQueue2', {
      queueArn: Token.asString({ Ref: 'ARN' }),
      fifo: false,
    });
    const fifoQueue = sqs.Queue.fromQueueAttributes(stack, 'FifoQueue', {
      queueArn: Token.asString({ Ref: 'ARN' }),
      fifo: true,
    });

    // THEN
    expect(stdQueue1.fifo).toEqual(false);
    expect(stdQueue2.fifo).toEqual(false);
    expect(fifoQueue.fifo).toEqual(true);
  });

  test('import queueArn from token, check attributes', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const stdQueue1 = sqs.Queue.fromQueueArn(stack, 'StdQueue', Token.asString({ Ref: 'ARN' }));

    // THEN
    expect(stack.resolve(stdQueue1.queueArn)).toEqual({
      Ref: 'ARN',
    });
    expect(stack.resolve(stdQueue1.queueName)).toEqual({
      'Fn::Select': [5, { 'Fn::Split': [':', { Ref: 'ARN' }] }],
    });
    expect(stack.resolve(stdQueue1.queueUrl)).toEqual({
      'Fn::Join':
        ['',
          ['https://sqs.',
            { 'Fn::Select': [3, { 'Fn::Split': [':', { Ref: 'ARN' }] }] },
            '.',
            { Ref: 'AWS::URLSuffix' },
            '/',
            { 'Fn::Select': [4, { 'Fn::Split': [':', { Ref: 'ARN' }] }] },
            '/',
            { 'Fn::Select': [5, { 'Fn::Split': [':', { Ref: 'ARN' }] }] }]],
    });
    expect(stdQueue1.fifo).toEqual(false);
  });

  test('fails if fifo flag is set for non fifo queue name', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'my-stack');

    // THEN
    expect(() => sqs.Queue.fromQueueAttributes(stack, 'ImportedStdQueue', {
      queueArn: 'arn:aws:sqs:us-west-2:123456789012:queue1',
      fifo: true,
    })).toThrow(/FIFO queue names must end in '.fifo'/);
  });

  test('fails if fifo flag is false for fifo queue name', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'my-stack');

    // THEN
    expect(() => sqs.Queue.fromQueueAttributes(stack, 'ImportedFifoQueue', {
      queueArn: 'arn:aws:sqs:us-west-2:123456789012:queue1.fifo',
      fifo: false,
    })).toThrow(/Non-FIFO queue name may not end in '.fifo'/);
  });

  test('importing works correctly for cross region queue', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    const imports = sqs.Queue.fromQueueArn(stack, 'Imported', 'arn:aws:sqs:us-west-2:123456789012:queue1');

    // THEN

    // "import" returns an IQueue bound to `Fn::ImportValue`s.
    expect(stack.resolve(imports.queueArn)).toEqual('arn:aws:sqs:us-west-2:123456789012:queue1');
    expect(stack.resolve(imports.queueUrl)).toEqual({
      'Fn::Join':
      ['', ['https://sqs.us-west-2.', { Ref: 'AWS::URLSuffix' }, '/123456789012/queue1']],
    });
    expect(stack.resolve(imports.queueName)).toEqual('queue1');
  });

  test('sets account for imported queue env by fromQueueAttributes', () => {
    const stack = new Stack();
    const imported = sqs.Queue.fromQueueAttributes(stack, 'Imported', {
      queueArn: 'arn:aws:sqs:us-west-2:999999999999:queue',
    });

    expect(imported.env.account).toEqual('999999999999');
  });

  test('sets region for imported queue env by fromQueueAttributes', () => {
    const stack = new Stack();
    const imported = sqs.Queue.fromQueueAttributes(stack, 'Imported', {
      queueArn: 'arn:aws:sqs:us-west-2:999999999999:queue',
    });

    expect(imported.env.region).toEqual('us-west-2');
  });

  test('sets account for imported queue env by fromQueueArn', () => {
    const stack = new Stack();
    const imported = sqs.Queue.fromQueueArn(stack, 'Imported', 'arn:aws:sqs:us-west-2:999999999999:queue');

    expect(imported.env.account).toEqual('999999999999');
  });

  test('sets region for imported queue env by fromQueueArn', () => {
    const stack = new Stack();
    const imported = sqs.Queue.fromQueueArn(stack, 'Imported', 'arn:aws:sqs:us-west-2:123456789012:queue');

    expect(imported.env.region).toEqual('us-west-2');
  });
});

describe('grants', () => {
  test('grantConsumeMessages', () => {
    testGrant((q, p) => q.grantConsumeMessages(p),
      'sqs:ReceiveMessage',
      'sqs:ChangeMessageVisibility',
      'sqs:GetQueueUrl',
      'sqs:DeleteMessage',
      'sqs:GetQueueAttributes',
    );
  });

  test('grantSendMessages', () => {
    testGrant((q, p) => q.grantSendMessages(p),
      'sqs:SendMessage',
      'sqs:GetQueueAttributes',
      'sqs:GetQueueUrl',
    );
  });

  test('grantPurge', () => {
    testGrant((q, p) => q.grantPurge(p),
      'sqs:PurgeQueue',
      'sqs:GetQueueAttributes',
      'sqs:GetQueueUrl',
    );
  });

  test('grant() is general purpose', () => {
    testGrant((q, p) => q.grant(p, 'service:hello', 'service:world'),
      'service:hello',
      'service:world',
    );
  });

  test('grants also work on imported queues', () => {
    const stack = new Stack();
    const queue = sqs.Queue.fromQueueAttributes(stack, 'Import', {
      queueArn: 'arn:aws:sqs:us-east-1:123456789012:queue1',
      queueUrl: 'https://queue-url',
    });

    const user = new iam.User(stack, 'User');

    queue.grantPurge(user);

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': [
              'sqs:PurgeQueue',
              'sqs:GetQueueAttributes',
              'sqs:GetQueueUrl',
            ],
            'Effect': 'Allow',
            'Resource': 'arn:aws:sqs:us-east-1:123456789012:queue1',
          },
        ],
        'Version': '2012-10-17',
      },
    });
  });

  test('grant on CfnQueue with KMS key grants key permissions', () => {
    const stack = new Stack();
    const key = new kms.CfnKey(stack, 'Key');
    const cfnQueue = new sqs.CfnQueue(stack, 'Queue', {
      kmsMasterKeyId: key.attrKeyId,
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    QueueGrants.fromQueue(cfnQueue).sendMessages(role);

    Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: [
              'kms:Decrypt',
              'kms:Encrypt',
              'kms:ReEncrypt*',
              'kms:GenerateDataKey*',
            ],
            Resource: '*',
            Principal: { 'AWS': { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] } },
          }),
        ]),
      },
    });
  });

  test('grant on CfnQueue adds statement to queue policy', () => {
    const stack = new Stack();
    const cfnQueue = new sqs.CfnQueue(stack, 'Queue');
    const principal = new iam.ServicePrincipal('lambda.amazonaws.com');

    QueueGrants.fromQueue(cfnQueue).sendMessages(principal);

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::SQS::QueuePolicy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: [
              'sqs:SendMessage',
              'sqs:GetQueueAttributes',
              'sqs:GetQueueUrl',
            ],
            Principal: { Service: 'lambda.amazonaws.com' },
            Resource: { 'Fn::GetAtt': ['Queue', 'Arn'] },
          }),
        ]),
      },
      Queues: [{ 'Ref': 'Queue' }],
    });
  });
});

describe('queue encryption', () => {
  test('encryptionMasterKey can be set to a custom KMS key', () => {
    const stack = new Stack();

    const key = new kms.Key(stack, 'CustomKey');
    const queue = new sqs.Queue(stack, 'Queue', { encryptionMasterKey: key });

    expect(queue.encryptionMasterKey).toEqual(key);
    expect(queue.encryptionType).toEqual(sqs.QueueEncryption.KMS);
    Template.fromStack(stack).hasResourceProperties('AWS::SQS::Queue', {
      'KmsMasterKeyId': { 'Fn::GetAtt': ['CustomKey1E6D0D07', 'Arn'] },
    });
  });

  test('a kms key will be allocated if encryption = kms but a master key is not specified', () => {
    const stack = new Stack();

    const queue = new sqs.Queue(stack, 'Queue', { encryption: sqs.QueueEncryption.KMS });

    Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', Match.anyValue());
    Template.fromStack(stack).hasResourceProperties('AWS::SQS::Queue', {
      'KmsMasterKeyId': {
        'Fn::GetAtt': [
          'QueueKey39FCBAE6',
          'Arn',
        ],
      },
    });
    expect(queue.encryptionType).toEqual(sqs.QueueEncryption.KMS);
  });

  test('it is possible to use a managed kms key', () => {
    const stack = new Stack();

    const queue = new sqs.Queue(stack, 'Queue', { encryption: sqs.QueueEncryption.KMS_MANAGED });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'Properties': {
            'KmsMasterKeyId': 'alias/aws/sqs',
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
    expect(queue.encryptionType).toEqual(sqs.QueueEncryption.KMS_MANAGED);
  });

  test('grant also affects key on encrypted queue', () => {
    // GIVEN
    const stack = new Stack();
    const queue = new sqs.Queue(stack, 'Queue', {
      encryption: sqs.QueueEncryption.KMS,
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('someone'),
    });

    // WHEN
    queue.grantSendMessages(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': [
              'sqs:SendMessage',
              'sqs:GetQueueAttributes',
              'sqs:GetQueueUrl',
            ],
            'Effect': 'Allow',
            'Resource': { 'Fn::GetAtt': ['Queue4A7E3555', 'Arn'] },
          },
          {
            'Action': [
              'kms:Decrypt',
              'kms:Encrypt',
              'kms:ReEncrypt*',
              'kms:GenerateDataKey*',
            ],
            'Effect': 'Allow',
            'Resource': { 'Fn::GetAtt': ['QueueKey39FCBAE6', 'Arn'] },
          },
        ],
        'Version': '2012-10-17',
      },
    });
  });

  test('it is possible to use sqs managed server side encryption', () => {
    const stack = new Stack();

    const queue = new sqs.Queue(stack, 'Queue', { encryption: sqs.QueueEncryption.SQS_MANAGED });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'Properties': {
            'SqsManagedSseEnabled': true,
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
    expect(queue.encryptionType).toEqual(sqs.QueueEncryption.SQS_MANAGED);
  });

  test('it is possible to disable encryption (unencrypted)', () => {
    const stack = new Stack();

    const queue = new sqs.Queue(stack, 'Queue', { encryption: sqs.QueueEncryption.UNENCRYPTED });
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'Properties': {
            'SqsManagedSseEnabled': false,
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
    expect(queue.encryptionType).toEqual(sqs.QueueEncryption.UNENCRYPTED);
  });

  test('encryptionMasterKey is not supported if encryption type SQS_MANAGED is used', () => {
    // GIVEN
    const stack = new Stack();
    const key = new kms.Key(stack, 'CustomKey');

    // THEN
    expect(() => new sqs.Queue(stack, 'Queue', {
      encryption: sqs.QueueEncryption.SQS_MANAGED,
      encryptionMasterKey: key,
    })).toThrow(/'encryptionMasterKey' is not supported if encryption type 'SQS_MANAGED' is used/);
  });

  test('encryptionType is always KMS, when an encryptionMasterKey is provided', () => {
    // GIVEN
    const stack = new Stack();
    const key = new kms.Key(stack, 'CustomKey');
    const queue = new sqs.Queue(stack, 'Queue', {
      encryption: sqs.QueueEncryption.KMS_MANAGED,
      encryptionMasterKey: key,
    });

    // THEN
    expect(queue.encryptionType).toBe(sqs.QueueEncryption.KMS);
  });
});

describe('encryption in transit', () => {
  test('enforceSSL can be enabled', () => {
    const stack = new Stack();
    new sqs.Queue(stack, 'Queue', { enforceSSL: true });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
        'QueuePolicy25439813': {
          'Type': 'AWS::SQS::QueuePolicy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  'Action': 'sqs:*',
                  'Condition': {
                    'Bool': {
                      'aws:SecureTransport': 'false',
                    },
                  },
                  'Effect': 'Deny',
                  'Principal': {
                    'AWS': '*',
                  },
                  'Resource': {
                    'Fn::GetAtt': [
                      'Queue4A7E3555',
                      'Arn',
                    ],
                  },
                },
              ],
              'Version': '2012-10-17',
            },
          },
        },
      },
    });
  });
});

test('test ".fifo" suffixed queues register as fifo', () => {
  const stack = new Stack();
  const queue = new sqs.Queue(stack, 'Queue', {
    queueName: 'MyQueue.fifo',
  });

  expect(queue.fifo).toEqual(true);

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'Queue4A7E3555': {
        'Type': 'AWS::SQS::Queue',
        'Properties': {
          'QueueName': 'MyQueue.fifo',
          'FifoQueue': true,
        },
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });
});

test('test a fifo queue is observed when the "fifo" property is specified', () => {
  const stack = new Stack();
  const queue = new sqs.Queue(stack, 'Queue', {
    fifo: true,
  });

  expect(queue.fifo).toEqual(true);

  Template.fromStack(stack).templateMatches({
    'Resources': {
      'Queue4A7E3555': {
        'Type': 'AWS::SQS::Queue',
        'Properties': {
          'FifoQueue': true,
        },
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });
});

test('test a fifo queue is observed when high throughput properties are specified', () => {
  const stack = new Stack();
  const queue = new sqs.Queue(stack, 'Queue', {
    fifo: true,
    fifoThroughputLimit: sqs.FifoThroughputLimit.PER_MESSAGE_GROUP_ID,
    deduplicationScope: sqs.DeduplicationScope.MESSAGE_GROUP,
  });

  expect(queue.fifo).toEqual(true);
  Template.fromStack(stack).templateMatches({
    'Resources': {
      'Queue4A7E3555': {
        'Type': 'AWS::SQS::Queue',
        'Properties': {
          'DeduplicationScope': 'messageGroup',
          'FifoQueue': true,
          'FifoThroughputLimit': 'perMessageGroupId',
        },
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });
});

test('test a queue throws when fifoThroughputLimit specified on non fifo queue', () => {
  const stack = new Stack();
  expect(() => {
    new sqs.Queue(stack, 'Queue', {
      fifo: false,
      fifoThroughputLimit: sqs.FifoThroughputLimit.PER_MESSAGE_GROUP_ID,
    });
  }).toThrow();
});

test('test a queue throws when deduplicationScope specified on non fifo queue', () => {
  const stack = new Stack();
  expect(() => {
    new sqs.Queue(stack, 'Queue', {
      fifo: false,
      deduplicationScope: sqs.DeduplicationScope.MESSAGE_GROUP,
    });
  }).toThrow();
});

test('fifo: false is dropped from properties', () => {
  // GIVEN
  const stack = new Stack();

  // WHEN
  new sqs.Queue(stack, 'Queue', {
    fifo: false,
  });

  // THEN
  Template.fromStack(stack).templateMatches({
    'Resources': {
      'Queue4A7E3555': {
        'Type': 'AWS::SQS::Queue',
        'Properties': Match.absent(),
        'UpdateReplacePolicy': 'Delete',
        'DeletionPolicy': 'Delete',
      },
    },
  });
});

test('test metrics', () => {
  // GIVEN
  const stack = new Stack();
  const queue = new sqs.Queue(stack, 'Queue');

  // THEN
  expect(stack.resolve(queue.metricNumberOfMessagesSent())).toEqual({
    dimensions: { QueueName: { 'Fn::GetAtt': ['Queue4A7E3555', 'QueueName'] } },
    namespace: 'AWS/SQS',
    metricName: 'NumberOfMessagesSent',
    period: Duration.minutes(5),
    statistic: 'Sum',
  });

  expect(stack.resolve(queue.metricSentMessageSize())).toEqual({
    dimensions: { QueueName: { 'Fn::GetAtt': ['Queue4A7E3555', 'QueueName'] } },
    namespace: 'AWS/SQS',
    metricName: 'SentMessageSize',
    period: Duration.minutes(5),
    statistic: 'Average',
  });
});

describe('metricApproximateNumberOfMessagesOutstanding', () => {
  test('sums the visible and not visible message counts', () => {
    // GIVEN
    const stack = new Stack();
    const queue = new sqs.Queue(stack, 'Queue');

    // WHEN
    const metric = queue.metricApproximateNumberOfMessagesOutstanding();

    // THEN
    expect(metric.expression).toEqual('visible + notVisible');
    expect(metric.label).toEqual('Approximate number of messages outstanding');
    expect(metric.period).toEqual(Duration.minutes(5));
    expect(stack.resolve(metric.usingMetrics.visible)).toEqual({
      dimensions: { QueueName: { 'Fn::GetAtt': ['Queue4A7E3555', 'QueueName'] } },
      namespace: 'AWS/SQS',
      metricName: 'ApproximateNumberOfMessagesVisible',
      period: Duration.minutes(5),
      statistic: 'Maximum',
    });
    expect(stack.resolve(metric.usingMetrics.notVisible)).toEqual({
      dimensions: { QueueName: { 'Fn::GetAtt': ['Queue4A7E3555', 'QueueName'] } },
      namespace: 'AWS/SQS',
      metricName: 'ApproximateNumberOfMessagesNotVisible',
      period: Duration.minutes(5),
      statistic: 'Maximum',
    });
  });

  test('applies statistic and period to both underlying metrics', () => {
    // GIVEN
    const stack = new Stack();
    const queue = new sqs.Queue(stack, 'Queue');

    // WHEN
    const metric = queue.metricApproximateNumberOfMessagesOutstanding({
      statistic: 'Average',
      period: Duration.minutes(1),
      label: 'Outstanding work',
    });

    // THEN
    expect(metric.label).toEqual('Outstanding work');
    expect(metric.period).toEqual(Duration.minutes(1));
    for (const id of ['visible', 'notVisible']) {
      expect(stack.resolve(metric.usingMetrics[id])).toEqual(expect.objectContaining({
        period: Duration.minutes(1),
        statistic: 'Average',
      }));
    }
    // Both metrics already carry the expression's period, so nothing gets overridden
    expect(metric.warningsV2).toBeUndefined();
  });

  test('can be alarmed on', () => {
    // GIVEN
    const stack = new Stack();
    const queue = new sqs.Queue(stack, 'Queue');

    // WHEN
    queue.metricApproximateNumberOfMessagesOutstanding().createAlarm(stack, 'Alarm', {
      threshold: 100,
      evaluationPeriods: 3,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
      EvaluationPeriods: 3,
      Threshold: 100,
      Metrics: Match.arrayWith([
        Match.objectLike({
          Id: 'expr_1',
          Expression: 'visible + notVisible',
          Label: 'Approximate number of messages outstanding',
        }),
        Match.objectLike({
          Id: 'visible',
          MetricStat: Match.objectLike({
            Metric: Match.objectLike({
              Namespace: 'AWS/SQS',
              MetricName: 'ApproximateNumberOfMessagesVisible',
            }),
            Period: 300,
            Stat: 'Maximum',
          }),
          ReturnData: false,
        }),
        Match.objectLike({
          Id: 'notVisible',
          MetricStat: Match.objectLike({
            Metric: Match.objectLike({
              Namespace: 'AWS/SQS',
              MetricName: 'ApproximateNumberOfMessagesNotVisible',
            }),
            Period: 300,
            Stat: 'Maximum',
          }),
          ReturnData: false,
        }),
      ]),
    });
  });
});

test('fails if queue policy has no actions', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'my-stack');
  const queue = new sqs.Queue(stack, 'Queue');

  // WHEN
  queue.addToResourcePolicy(new iam.PolicyStatement({
    resources: ['*'],
    principals: [new iam.ArnPrincipal('arn')],
  }));

  // THEN
  expect(() => app.synth()).toThrow(/A PolicyStatement must specify at least one \'action\' or \'notAction\'/);
});

test('fails if queue policy has no IAM principals', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'my-stack');
  const queue = new sqs.Queue(stack, 'Queue');

  // WHEN
  queue.addToResourcePolicy(new iam.PolicyStatement({
    resources: ['*'],
    actions: ['sqs:*'],
  }));

  // THEN
  expect(() => app.synth()).toThrow(/A PolicyStatement used in a resource-based policy must specify at least one IAM principal/);
});

describe('redriveAllowPolicy', () => {
  test('Default settings for the dead letter source queue permission', () => {
    const stack = new Stack();
    new sqs.Queue(stack, 'Queue', {
      redriveAllowPolicy: {},
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'Properties': {
            'RedriveAllowPolicy': {
              'redrivePermission': 'allowAll',
            },
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
  });

  test.each([
    [sqs.RedrivePermission.ALLOW_ALL, 'allowAll'],
    [sqs.RedrivePermission.DENY_ALL, 'denyAll'],
  ])('redrive permission can be set to %s', (permission, expected) => {
    const stack = new Stack();
    new sqs.Queue(stack, 'Queue', {
      redriveAllowPolicy: {
        redrivePermission: permission,
      },
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'Properties': {
            'RedriveAllowPolicy': {
              'redrivePermission': expected,
            },
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
  });

  test('explicit specification of dead letter source queues', () => {
    const stack = new Stack();
    const sourceQueue1 = new sqs.Queue(stack, 'SourceQueue1');
    const sourceQueue2 = new sqs.Queue(stack, 'SourceQueue2');
    new sqs.Queue(stack, 'Queue', { redriveAllowPolicy: { sourceQueues: [sourceQueue1, sourceQueue2] } });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'SourceQueue1F4BBA4BB': {
          'Type': 'AWS::SQS::Queue',
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
        'SourceQueue22481CB5A': {
          'Type': 'AWS::SQS::Queue',
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
        'Queue4A7E3555': {
          'Type': 'AWS::SQS::Queue',
          'Properties': {
            'RedriveAllowPolicy': {
              'redrivePermission': 'byQueue',
              'sourceQueueArns': [
                {
                  'Fn::GetAtt': [
                    'SourceQueue1F4BBA4BB',
                    'Arn',
                  ],
                },
                {
                  'Fn::GetAtt': [
                    'SourceQueue22481CB5A',
                    'Arn',
                  ],
                },
              ],
            },
          },
          'UpdateReplacePolicy': 'Delete',
          'DeletionPolicy': 'Delete',
        },
      },
    });
  });

  test('throw if sourceQueues is not specified when redrivePermission is byQueue', () => {
    const stack = new Stack();
    expect(() => {
      new sqs.Queue(stack, 'Queue', {
        redriveAllowPolicy: {
          redrivePermission: sqs.RedrivePermission.BY_QUEUE,
        },
      });
    }).toThrow(/At least one source queue must be specified when RedrivePermission is set to 'byQueue'/);
  });

  test('throw if dead letter source queues are specified with allowAll permission', () => {
    const stack = new Stack();
    const sourceQueue1 = new sqs.Queue(stack, 'SourceQueue1');
    expect(() => {
      new sqs.Queue(stack, 'Queue', {
        redriveAllowPolicy: {
          sourceQueues: [sourceQueue1],
          redrivePermission: sqs.RedrivePermission.ALLOW_ALL,
        },
      });
    }).toThrow(/sourceQueues cannot be configured when RedrivePermission is set to 'allowAll' or 'denyAll'/);
  });

  test('throw if souceQueues length is greater than 10', () => {
    const stack = new Stack();
    const sourceQueues: sqs.IQueue[] = [];
    for (let i = 0; i < 11; i++) {
      sourceQueues.push(new sqs.Queue(stack, `SourceQueue${i}`));
    }
    expect(() => {
      new sqs.Queue(stack, 'Queue', {
        redriveAllowPolicy: {
          sourceQueues,
          redrivePermission: sqs.RedrivePermission.BY_QUEUE,
        },
      });
    }).toThrow(/Up to 10 sourceQueues can be specified. Set RedrivePermission to 'allowAll' to specify more/);
  });

  test('throw if sourceQueues is blank array when redrivePermission is byQueue', () => {
    const stack = new Stack();
    expect(() => {
      new sqs.Queue(stack, 'Queue', {
        redriveAllowPolicy: {
          sourceQueues: [],
          redrivePermission: sqs.RedrivePermission.BY_QUEUE,
        },
      });
    }).toThrow(/At least one source queue must be specified when RedrivePermission is set to 'byQueue'/);
  });
});

function testGrant(action: (q: sqs.Queue, principal: iam.IPrincipal) => void, ...expectedActions: string[]) {
  const stack = new Stack();
  const queue = new sqs.Queue(stack, 'MyQueue');
  const principal = new iam.User(stack, 'User');

  action(queue, principal);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': expectedActions,
          'Effect': 'Allow',
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
  });
}
