import type { Construct } from 'constructs';
import { Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import { Bucket } from '../../aws-s3';
import { App, CfnParameter, Fn, RemovalPolicy, Stack } from '../../core';
import type { ILogGroup, ILogSubscriptionDestination } from '../lib';
import { LogGroupGrants, LogGroup, RetentionDays, LogGroupClass, DataProtectionPolicy, DataIdentifier, CustomDataIdentifier, FilterPattern, FieldIndexPolicy, ParserProcessor, ParserProcessorType, JsonMutatorType, JsonMutatorProcessor, CfnLogGroup } from '../lib';

describe('log group', () => {
  test('set kms key when provided', () => {
    // GIVEN
    const stack = new Stack();
    const encryptionKey = new kms.Key(stack, 'Key');

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      encryptionKey,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      KmsKeyId: { 'Fn::GetAtt': ['Key961B73FD', 'Arn'] },
    });
  });

  test('fixed retention', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      retention: RetentionDays.ONE_WEEK,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      RetentionInDays: 7,
    });
  });

  test('default retention', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new LogGroup(stack, 'LogGroup');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      RetentionInDays: 731,
    });
  });

  test('infinite retention/dont delete log group by default', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      retention: RetentionDays.INFINITE,
    });

    // THEN
    Template.fromStack(stack).templateMatches({
      Resources: {
        LogGroupF5B46931: {
          Type: 'AWS::Logs::LogGroup',
          DeletionPolicy: 'Retain',
          UpdateReplacePolicy: 'Retain',
        },
      },
    });
  });

  test('infinite retention via legacy method', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      // Don't know why TypeScript doesn't complain about passing Infinity to
      // something where an enum is expected, but better keep this behavior for
      // existing clients.
      retention: Infinity,
    });

    // THEN
    Template.fromStack(stack).templateMatches({
      Resources: {
        LogGroupF5B46931: {
          Type: 'AWS::Logs::LogGroup',
          DeletionPolicy: 'Retain',
          UpdateReplacePolicy: 'Retain',
        },
      },
    });
  });

  test('unresolved retention', () => {
    // GIVEN
    const stack = new Stack();

    const parameter = new CfnParameter(stack, 'RetentionInDays', { default: 30, type: 'Number' });

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      retention: parameter.valueAsNumber,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      RetentionInDays: {
        Ref: 'RetentionInDays',
      },
    });
  });

  test('with INFREQUENT_ACCESS log group class', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'TestStack');

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      logGroupClass: LogGroupClass.INFREQUENT_ACCESS,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupClass: LogGroupClass.INFREQUENT_ACCESS,
    });
  });

  test('with STANDARD log group class', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'TestStack');

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      logGroupClass: LogGroupClass.STANDARD,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupClass: LogGroupClass.STANDARD,
    });
  });

  // when LogGroupClass is not specified, leave it to CFN and/or backend to default to STANDARD
  test('with default log group class', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new LogGroup(stack, 'LogGroup');

    // THEN
    Template.fromStack(stack).resourcePropertiesCountIs('AWS::Logs::LogGroup', {
      LogGroupClass: LogGroupClass.STANDARD,
    }, 0);

    Template.fromStack(stack).resourcePropertiesCountIs('AWS::Logs::LogGroup', {
      LogGroupClass: LogGroupClass.INFREQUENT_ACCESS,
    }, 0);
  });

  test('will delete log group if asked to', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new LogGroup(stack, 'LogGroup', {
      retention: Infinity,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // THEN
    Template.fromStack(stack).templateMatches({
      Resources: {
        LogGroupF5B46931: {
          Type: 'AWS::Logs::LogGroup',
          DeletionPolicy: 'Delete',
          UpdateReplacePolicy: 'Delete',
        },
      },
    });
  });

  test('import from ARN, same region', () => {
    // GIVEN
    const stack2 = new Stack();

    // WHEN
    const imported = LogGroup.fromLogGroupArn(stack2, 'lg', 'arn:aws:logs:us-east-1:123456789012:log-group:my-log-group');
    imported.addStream('MakeMeAStream');

    // THEN
    expect(imported.logGroupName).toEqual('my-log-group');
    expect(imported.logGroupArn).toEqual('arn:aws:logs:us-east-1:123456789012:log-group:my-log-group:*');
    Template.fromStack(stack2).hasResourceProperties('AWS::Logs::LogStream', {
      LogGroupName: 'my-log-group',
    });
  });

  test('import from ARN, different region', () => {
    // GIVEN
    const stack = new Stack();
    const importRegion = 'asgard-1';

    // WHEN
    const imported = LogGroup.fromLogGroupArn(stack, 'lg',
      `arn:aws:logs:${importRegion}:123456789012:log-group:my-log-group`);
    imported.addStream('MakeMeAStream');

    // THEN
    expect(imported.logGroupName).toEqual('my-log-group');
    expect(imported.logGroupArn).toEqual(`arn:aws:logs:${importRegion}:123456789012:log-group:my-log-group:*`);
    expect(imported.env.region).not.toEqual(stack.region);
    expect(imported.env.region).toEqual(importRegion);

    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogStream', {
      LogGroupName: 'my-log-group',
    });
    Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 0);
  });

  test('import from name', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const imported = LogGroup.fromLogGroupName(stack, 'lg', 'my-log-group');
    imported.addStream('MakeMeAStream');

    // THEN
    expect(imported.logGroupName).toEqual('my-log-group');
    expect(imported.logGroupArn).toMatch(/^arn:.+:logs:.+:.+:log-group:my-log-group:\*$/);

    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogStream', {
      LogGroupName: 'my-log-group',
    });
  });

  describe('loggroups imported by name have stream wildcard appended to grant ARN', () => void dataDrivenTests([
    // Regardless of whether the user put :* there already because of this bug, we
    // don't want to append it twice.
    '',
    ':*',
  ], (suffix: string) => {
    // GIVEN
    const stack = new Stack();
    const user = new iam.User(stack, 'Role');
    const imported = LogGroup.fromLogGroupName(stack, 'lg', `my-log-group${suffix}`);

    // WHEN
    imported.grantWrite(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: ['logs:CreateLogStream', 'logs:PutLogEvents'],
            Effect: 'Allow',
            Resource: {
              'Fn::Join': ['', [
                'arn:',
                { Ref: 'AWS::Partition' },
                ':logs:',
                { Ref: 'AWS::Region' },
                ':',
                { Ref: 'AWS::AccountId' },
                ':log-group:my-log-group:*',
              ]],
            },
          },
        ],
      },
    });

    expect(imported.logGroupName).toEqual('my-log-group');
  }));

  describe('loggroups imported by ARN have stream wildcard appended to grant ARN', () => void dataDrivenTests([
    // Regardless of whether the user put :* there already because of this bug, we
    // don't want to append it twice.
    '',
    ':*',
  ], (suffix: string) => {
    // GIVEN
    const stack = new Stack();
    const user = new iam.User(stack, 'Role');
    const imported = LogGroup.fromLogGroupArn(stack, 'lg', `arn:aws:logs:us-west-1:123456789012:log-group:my-log-group${suffix}`);

    // WHEN
    imported.grantWrite(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: ['logs:CreateLogStream', 'logs:PutLogEvents'],
            Effect: 'Allow',
            Resource: 'arn:aws:logs:us-west-1:123456789012:log-group:my-log-group:*',
          },
        ],
      },
    });

    expect(imported.logGroupName).toEqual('my-log-group');
  }));

  test('extractMetric', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');

    // WHEN
    const metric = lg.extractMetric('$.myField', 'MyService', 'Field');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::MetricFilter', {
      FilterPattern: '{ $.myField = "*" }',
      LogGroupName: { Ref: 'LogGroupF5B46931' },
      MetricTransformations: [
        {
          MetricName: 'Field',
          MetricNamespace: 'MyService',
          MetricValue: '$.myField',
        },
      ],
    });

    expect(metric.namespace).toEqual('MyService');
    expect(metric.metricName).toEqual('Field');
  });

  test('extractMetric allows passing in namespaces with "/"', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');

    // WHEN
    const metric = lg.extractMetric('$.myField', 'MyNamespace/MyService', 'Field');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::MetricFilter', {
      FilterPattern: '{ $.myField = "*" }',
      MetricTransformations: [
        {
          MetricName: 'Field',
          MetricNamespace: 'MyNamespace/MyService',
          MetricValue: '$.myField',
        },
      ],
    });

    expect(metric.namespace).toEqual('MyNamespace/MyService');
    expect(metric.metricName).toEqual('Field');
  });

  test('grant write', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');
    const user = new iam.User(stack, 'User');

    // WHEN
    lg.grantWrite(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: ['logs:CreateLogStream', 'logs:PutLogEvents'],
            Effect: 'Allow',
            Resource: { 'Fn::GetAtt': ['LogGroupF5B46931', 'Arn'] },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('grant read', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');
    const user = new iam.User(stack, 'User');

    // WHEN
    lg.grantRead(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: ['logs:FilterLogEvents', 'logs:GetLogEvents', 'logs:GetLogGroupFields', 'logs:DescribeLogGroups', 'logs:DescribeLogStreams'],
            Effect: 'Allow',
            Resource: { 'Fn::GetAtt': ['LogGroupF5B46931', 'Arn'] },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('grant to service principal', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');
    const sp = new iam.ServicePrincipal('es.amazonaws.com');

    // WHEN
    lg.grantWrite(sp);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::ResourcePolicy', {
      PolicyDocument: {
        'Fn::Join': [
          '',
          [
            '{"Statement":[{"Action":["logs:CreateLogStream","logs:PutLogEvents"],"Effect":"Allow","Principal":{"Service":"es.amazonaws.com"},"Resource":"',
            {
              'Fn::GetAtt': [
                'LogGroupF5B46931',
                'Arn',
              ],
            },
            '"}],"Version":"2012-10-17"}',
          ],
        ],
      },
      PolicyName: 'LogGroupPolicy643B329C',
    });
  });

  test('grant write to service principal for CfnLogGroup', () => {
    // GIVEN
    const stack = new Stack();
    const cfnLogGroup = new CfnLogGroup(stack, 'CfnLogGroup');
    const sp = new iam.ServicePrincipal('lambda.amazonaws.com');

    // WHEN
    LogGroupGrants.fromLogGroup(cfnLogGroup).write(sp);

    // THEN
    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::Logs::ResourcePolicy', {
      PolicyDocument: {
        'Fn::Join': [
          '',
          [
            '{"Statement":[{"Action":["logs:CreateLogStream","logs:PutLogEvents"],"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Resource":"',
            {
              'Fn::GetAtt': [
                'CfnLogGroup',
                'Arn',
              ],
            },
            '"}],"Version":"2012-10-17"}',
          ],
        ],
      },
    });
  });

  test('when added to log groups, IAM users are converted into account IDs in the resource policy', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');

    // WHEN
    lg.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: ['logs:PutLogEvents'],
      principals: [new iam.ArnPrincipal('arn:aws:iam::123456789012:user/user-name')],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::ResourcePolicy', {
      PolicyDocument: '{"Statement":[{"Action":"logs:PutLogEvents","Effect":"Allow","Principal":{"AWS":"123456789012"},"Resource":"*"}],"Version":"2012-10-17"}',
      PolicyName: 'LogGroupPolicy643B329C',
    });
  });

  test('log groups accept the AnyPrincipal policy', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');

    // WHEN
    lg.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: ['logs:PutLogEvents'],
      principals: [new iam.AnyPrincipal()],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::ResourcePolicy', {
      PolicyDocument: JSON.stringify({
        Statement: [{
          Action: 'logs:PutLogEvents',
          Effect: 'Allow',
          Principal: { AWS: '*' },
          Resource: '*',
        }],
        Version: '2012-10-17',
      }),
    });
  });

  test('imported values are treated as if they are ARNs and converted to account IDs via CFN pseudo parameters', () => {
    // GIVEN
    const stack = new Stack();
    const lg = new LogGroup(stack, 'LogGroup');

    // WHEN
    lg.addToResourcePolicy(new iam.PolicyStatement({
      resources: ['*'],
      actions: ['logs:PutLogEvents'],
      principals: [iam.Role.fromRoleArn(stack, 'Role', Fn.importValue('SomeRole'))],
    }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::ResourcePolicy', {
      PolicyDocument: {
        'Fn::Join': [
          '',
          [
            '{\"Statement\":[{\"Action\":\"logs:PutLogEvents\",\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"',
            {
              'Fn::Select': [
                4,
                { 'Fn::Split': [':', { 'Fn::ImportValue': 'SomeRole' }] },
              ],
            },
            '\"},\"Resource\":\"*\"}],\"Version\":\"2012-10-17\"}',
          ],
        ],
      },
    });
  });

  test('correctly returns physical name of the log group', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const logGroup = new LogGroup(stack, 'LogGroup', {
      logGroupName: 'my-log-group',
    });

    // THEN
    expect(logGroup.logGroupPhysicalName()).toEqual('my-log-group');
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: 'my-log-group',
    });
  });

  test('set data protection policy with custom name and description and no audit destinations', () => {
    // GIVEN
    const stack = new Stack();

    const dataProtectionPolicy = new DataProtectionPolicy({
      name: 'test-policy-name',
      description: 'test description',
      identifiers: [DataIdentifier.EMAILADDRESS],
    });

    // WHEN
    const logGroupName = 'test-log-group';
    new LogGroup(stack, 'LogGroup', {
      logGroupName: logGroupName,
      dataProtectionPolicy: dataProtectionPolicy,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: logGroupName,
      DataProtectionPolicy: {
        Name: 'test-policy-name',
        Description: 'test description',
        Version: '2021-06-01',
        Statement: [
          {
            Sid: 'audit-statement-cdk',
            DataIdentifier: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/EmailAddress',
                  ],
                ],
              },
            ],
            Operation: {
              Audit: {
                FindingsDestination: {},
              },
            },
          },
          {
            Sid: 'redact-statement-cdk',
            DataIdentifier: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/EmailAddress',
                  ],
                ],
              },
            ],
            Operation: {
              Deidentify: {
                MaskConfig: {},
              },
            },
          },
        ],
      },
    });
  });

  test('set data protection policy string-based data identifier', () => {
    // GIVEN
    const stack = new Stack();

    const dataProtectionPolicy = new DataProtectionPolicy({
      name: 'test-policy-name',
      description: 'test description',
      identifiers: [new DataIdentifier('NewIdentifier')],
    });

    // WHEN
    const logGroupName = 'test-log-group';
    new LogGroup(stack, 'LogGroup', {
      logGroupName: logGroupName,
      dataProtectionPolicy: dataProtectionPolicy,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: logGroupName,
      DataProtectionPolicy: {
        Name: 'test-policy-name',
        Description: 'test description',
        Version: '2021-06-01',
        Statement: [
          {
            Sid: 'audit-statement-cdk',
            DataIdentifier: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/NewIdentifier',
                  ],
                ],
              },
            ],
            Operation: {
              Audit: {
                FindingsDestination: {},
              },
            },
          },
          {
            Sid: 'redact-statement-cdk',
            DataIdentifier: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/NewIdentifier',
                  ],
                ],
              },
            ],
            Operation: {
              Deidentify: {
                MaskConfig: {},
              },
            },
          },
        ],
      },
    });
  });

  test('set data protection policy with audit destinations', () => {
    // GIVEN
    const stack = new Stack();

    const auditLogGroup = new LogGroup(stack, 'LogGroupAudit', { logGroupName: 'audit-log-group' });
    const auditS3Bucket = new Bucket(stack, 'BucketAudit', { bucketName: 'audit-bucket' });
    const auditDeliveryStreamName = 'delivery-stream-name';

    const dataProtectionPolicy = new DataProtectionPolicy({
      identifiers: [DataIdentifier.EMAILADDRESS],
      logGroupAuditDestination: auditLogGroup,
      s3BucketAuditDestination: auditS3Bucket,
      deliveryStreamNameAuditDestination: auditDeliveryStreamName,
    });

    // WHEN
    const logGroupName = 'test-log-group';
    new LogGroup(stack, 'LogGroup', {
      logGroupName: logGroupName,
      dataProtectionPolicy: dataProtectionPolicy,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: logGroupName,
      DataProtectionPolicy: {
        Name: 'data-protection-policy-cdk',
        Description: 'cdk generated data protection policy',
        Version: '2021-06-01',
        Statement: [
          {
            Sid: 'audit-statement-cdk',
            DataIdentifier: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/EmailAddress',
                  ],
                ],
              },
            ],
            Operation: {
              Audit: {
                FindingsDestination: {
                  CloudWatchLogs: {
                    LogGroup: {
                      Ref: 'LogGroupAudit2C8B7F73',
                    },
                  },
                  Firehose: {
                    DeliveryStream: auditDeliveryStreamName,
                  },
                  S3: {
                    Bucket: {
                      Ref: 'BucketAudit1DED3529',
                    },
                  },
                },
              },
            },
          },
          {
            Sid: 'redact-statement-cdk',
            DataIdentifier: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/EmailAddress',
                  ],
                ],
              },
            ],
            Operation: {
              Deidentify: {
                MaskConfig: {},
              },
            },
          },
        ],
      },
    });
  });

  test('set data protection policy with custom data identifier', () => {
    // GIVEN
    const stack = new Stack();

    const dataProtectionPolicy = new DataProtectionPolicy({
      name: 'test-policy-name',
      description: 'test description',
      identifiers: [new CustomDataIdentifier('EmployeeId', 'EmployeeId-\\d{9}')],
    });

    // WHEN
    const logGroupName = 'test-log-group';
    new LogGroup(stack, 'LogGroup', {
      logGroupName: logGroupName,
      dataProtectionPolicy: dataProtectionPolicy,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: logGroupName,
      DataProtectionPolicy: {
        Name: 'test-policy-name',
        Description: 'test description',
        Version: '2021-06-01',
        Configuration: {
          CustomDataIdentifier: [
            {
              Name: 'EmployeeId',
              Regex: 'EmployeeId-\\d{9}',
            },
          ],
        },
        Statement: [
          {
            Sid: 'audit-statement-cdk',
            DataIdentifier: [
              'EmployeeId',
            ],
            Operation: {
              Audit: {
                FindingsDestination: {},
              },
            },
          },
          {
            Sid: 'redact-statement-cdk',
            DataIdentifier: [
              'EmployeeId',
            ],
            Operation: {
              Deidentify: {
                MaskConfig: {},
              },
            },
          },
        ],
      },
    });
  });

  test('set data protection policy with mix of managed and custom data identifiers', () => {
    // GIVEN
    const stack = new Stack();

    const dataProtectionPolicy = new DataProtectionPolicy({
      name: 'test-policy-name',
      description: 'test description',
      identifiers: [new CustomDataIdentifier('EmployeeId', 'EmployeeId-\\d{9}'), DataIdentifier.EMAILADDRESS],
    });

    // WHEN
    const logGroupName = 'test-log-group';
    new LogGroup(stack, 'LogGroup', {
      logGroupName: logGroupName,
      dataProtectionPolicy: dataProtectionPolicy,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: logGroupName,
      DataProtectionPolicy: {
        Name: 'test-policy-name',
        Description: 'test description',
        Version: '2021-06-01',
        Configuration: {
          CustomDataIdentifier: [
            {
              Name: 'EmployeeId',
              Regex: 'EmployeeId-\\d{9}',
            },
          ],
        },
        Statement: [
          {
            Sid: 'audit-statement-cdk',
            DataIdentifier: [
              'EmployeeId',
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/EmailAddress',
                  ],
                ],
              },
            ],
            Operation: {
              Audit: {
                FindingsDestination: {},
              },
            },
          },
          {
            Sid: 'redact-statement-cdk',
            DataIdentifier: [
              'EmployeeId',
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dataprotection::aws:data-identifier/EmailAddress',
                  ],
                ],
              },
            ],
            Operation: {
              Deidentify: {
                MaskConfig: {},
              },
            },
          },
        ],
      },
    });
  });
});

test('set field index policy with four fields indexed', () => {
  // GIVEN
  const stack = new Stack();

  const fieldIndexPolicy = new FieldIndexPolicy({
    fields: ['Operation', 'RequestId', 'timestamp', 'message'],
  });

  // WHEN
  const logGroupName = 'test-field-index-log-group';
  new LogGroup(stack, 'LogGroup', {
    logGroupName: logGroupName,
    fieldIndexPolicies: [fieldIndexPolicy],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
    LogGroupName: logGroupName,
    FieldIndexPolicies: [{
      Fields: [
        'Operation',
        'RequestId',
        'timestamp',
        'message',
      ],
    }],
  });
});

test('set more than 20 field indexes in a field index policy', () => {
  let message;
  try {
    // GIVEN
    const stack = new Stack();
    const fieldIndexPolicy = new FieldIndexPolicy({
      fields: createMoreThan20FieldIndexes(),
    });

    // WHEN
    const logGroupName = 'test-field-multiple-field-index-policies';
    new LogGroup(stack, 'LogGroup', {
      logGroupName: logGroupName,
      fieldIndexPolicies: [fieldIndexPolicy],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: logGroupName,
      FieldIndexPolicies: [{
        Fields: ['abc'],
      }],
    });
  } catch (e) {
    message = (e as Error).message;
  }

  expect(message).toBeDefined();
  expect(message).toEqual('A maximum of 20 fields can be indexed per log group');
});

describe('subscription filter', () => {
  test('add subscription filter with custom name', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const logGroup = new LogGroup(stack, 'LogGroup');
    logGroup.addSubscriptionFilter('Subscription', {
      destination: new FakeDestination(),
      filterPattern: FilterPattern.literal('some pattern'),
      filterName: 'CustomSubscriptionFilterName',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Logs::SubscriptionFilter', {
      DestinationArn: 'arn:bogus',
      FilterPattern: 'some pattern',
      LogGroupName: { Ref: 'LogGroupF5B46931' },
      FilterName: 'CustomSubscriptionFilterName',
    });
  });
});

test('encrypting log group with referenced alias', () => {
  const stack = new Stack();

  const alias = kms.Alias.fromAliasName(stack, 'KmsAlias', 'alias/some-alias');

  new LogGroup(stack, 'LogGroup', {
    encryptionKey: alias,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
    KmsKeyId: {
      'Fn::Join': [
        '',
        [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':kms:',
          { Ref: 'AWS::Region' },
          ':',
          { Ref: 'AWS::AccountId' },
          ':alias/some-alias',
        ],
      ],
    },
  });
});

test('create a Add Key transformer against a log group', () => {
  // GIVEN
  const stack = new Stack();
  const logGroup = new LogGroup(stack, 'aws_cdk_test_log_group');

  const jsonParser = new ParserProcessor({
    type: ParserProcessorType.JSON,
  });

  const addKeysProcesor = new JsonMutatorProcessor({
    type: JsonMutatorType.ADD_KEYS,
    addKeysOptions: {
      entries: [
        { key: 'test_key1', value: 'test_value1', overwriteIfExists: true },
        { key: 'test_key2', value: 'test_value2' },
        { key: 'test_key3', value: 'test_value3', overwriteIfExists: false },
      ],
    },
  });

  // WHEN

  logGroup.addTransformer(
    'Transformer',
    {
      transformerName: 'MyTransformer',
      transformerConfig: [jsonParser, addKeysProcesor],
    },
  );

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Logs::Transformer', {
    LogGroupIdentifier: { Ref: 'awscdktestloggroup30AE39AB' },
    TransformerConfig: [
      {
        ParseJSON: { Source: '@message' },
      },
      {
        AddKeys: {
          Entries: [
            { Key: 'test_key1', Value: 'test_value1', OverwriteIfExists: true },
            { Key: 'test_key2', Value: 'test_value2', OverwriteIfExists: false },
            { Key: 'test_key3', Value: 'test_value3', OverwriteIfExists: false },
          ],
        },
      },
    ],
  });
});

test('enable deletion protection', () => {
  // GIVEN
  const stack = new Stack();

  // WHEN
  new LogGroup(stack, 'LogGroup', {
    deletionProtectionEnabled: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
    DeletionProtectionEnabled: true,
  });
});

function dataDrivenTests(cases: string[], body: (suffix: string) => void): void {
  for (let i = 0; i < cases.length; i++) {
    const args = cases[i]; // Need to capture inside loop for safe use inside closure.
    test(`case ${i + 1}`, () => {
      body(args);
    });
  }
}

function createMoreThan20FieldIndexes(): string[] {
  let arr: string[] = [];
  for (let i = 0; i < 23; i++) {
    arr.push('abc' + i.toString());
  }
  return arr;
}

class FakeDestination implements ILogSubscriptionDestination {
  public bind(_scope: Construct, _sourceLogGroup: ILogGroup) {
    return {
      arn: 'arn:bogus',
    };
  }
}
