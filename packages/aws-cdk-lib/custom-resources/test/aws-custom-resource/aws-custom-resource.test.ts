import { Template } from '../../../assertions';
import * as ec2 from '../../../aws-ec2';
import * as iam from '../../../aws-iam';
import * as logs from '../../../aws-logs';
import * as cdk from '../../../core';
import { App, Stack } from '../../../core';
import { LOG_API_RESPONSE_DATA_PROPERTY_TRUE_DEFAULT } from '../../../cx-api';
import { AwsCustomResource, AwsCustomResourcePolicy, Logging, PhysicalResourceId, PhysicalResourceIdReference } from '../../lib';

/* eslint-disable @stylistic/quote-props */

test('aws sdk js custom resource with onCreate and onDelete', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    resourceType: 'Custom::LogRetentionPolicy',
    onCreate: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      physicalResourceId: PhysicalResourceId.of('loggroup'),
    },
    onDelete: {
      service: 'CloudWatch',
      action: 'tagResource',
      parameters: {
        ResourceARN: 'dummy',
        Tags: [{ Key: 'Name', Value: 'prod' }],
      },
      physicalResourceId: PhysicalResourceId.of('add_tag'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::LogRetentionPolicy', {
    'Create': JSON.stringify({
      'service': 'CloudWatchLogs',
      'action': 'putRetentionPolicy',
      'parameters': {
        'logGroupName': '/aws/lambda/loggroup',
        'retentionInDays': 90,
      },
      'physicalResourceId': {
        'id': 'loggroup',
      },
    }),
    'Delete': JSON.stringify({
      'service': 'CloudWatch',
      'action': 'tagResource',
      'parameters': {
        'ResourceARN': 'dummy',
        'Tags': [{ 'Key': 'Name', 'Value': 'prod' }],
      },
      'physicalResourceId': {
        'id': 'add_tag',
      },
    }),
    'InstallLatestAwsSdk': true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': 'logs:PutRetentionPolicy',
          'Effect': 'Allow',
          'Resource': '*',
        },
        {
          'Action': 'cloudwatch:TagResource',
          'Effect': 'Allow',
          'Resource': '*',
        },
      ],
      'Version': '2012-10-17',
    },
  });
});

test('onCreate defaults to onUpdate', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    resourceType: 'Custom::S3PutObject',
    onUpdate: {
      service: 's3',
      action: 'putObject',
      parameters: {
        Bucket: 'my-bucket',
        Key: 'my-key',
        Body: 'my-body',
      },
      physicalResourceId: PhysicalResourceId.fromResponse('ETag'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::S3PutObject', {
    'Create': JSON.stringify({
      'service': 's3',
      'action': 'putObject',
      'parameters': {
        'Bucket': 'my-bucket',
        'Key': 'my-key',
        'Body': 'my-body',
      },
      'physicalResourceId': {
        'responsePath': 'ETag',
      },
    }),
    'Update': JSON.stringify({
      'service': 's3',
      'action': 'putObject',
      'parameters': {
        'Bucket': 'my-bucket',
        'Key': 'my-key',
        'Body': 'my-body',
      },
      'physicalResourceId': {
        'responsePath': 'ETag',
      },
    }),
  });
});

test('with custom policyStatements', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onUpdate: {
      service: 'S3',
      action: 'putObject',
      parameters: {
        Bucket: 'my-bucket',
        Key: 'my-key',
        Body: 'my-body',
      },
      physicalResourceId: PhysicalResourceId.fromResponse('ETag'),
    },
    policy: AwsCustomResourcePolicy.fromStatements([
      new iam.PolicyStatement({
        actions: ['s3:PutObject'],
        resources: ['arn:aws:s3:::my-bucket/my-key'],
      }),
    ]),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': 's3:PutObject',
          'Effect': 'Allow',
          'Resource': 'arn:aws:s3:::my-bucket/my-key',
        },
      ],
      'Version': '2012-10-17',
    },
  });
});

test('fails when no calls are specified', () => {
  const stack = new cdk.Stack();
  expect(() => new AwsCustomResource(stack, 'AwsSdk', {
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  })).toThrow(/`onCreate`.+`onUpdate`.+`onDelete`/);
});

// test patterns for physicalResourceId
// | # |    onCreate.physicalResourceId    |   onUpdate.physicalResourceId    | Error thrown? |
// |---|-----------------------------------|----------------------------------|---------------|
// | 1 | ANY_VALUE                         | ANY_VALUE                        | no            |
// | 2 | ANY_VALUE                         | undefined                        | no            |
// | 3 | undefined                         | ANY_VALLUE                       | yes           |
// | 4 | undefined                         | undefined                        | yes           |
// | 5 | ANY_VALUE                         | undefined (*omit whole onUpdate) | no            |
// | 6 | undefined                         | undefined (*omit whole onUpdate) | yes           |
// | 7 | ANY_VALUE (*copied from onUpdate) | ANY_VALUE                        | no            |
// | 8 | undefined (*copied from onUpdate) | undefined                        | yes           |
describe('physicalResourceId patterns', () => {
  // physicalResourceId pattern #1
  test('physicalResourceId is specified both in onCreate and onUpdate then success', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::AthenaNotebook',
      onCreate: {
        service: 'Athena',
        action: 'createNotebook',
        physicalResourceId: PhysicalResourceId.of('id'),
        parameters: {
          WorkGroup: 'WorkGroupA',
          Name: 'Notebook1',
        },
      },
      onUpdate: {
        service: 'Athena',
        action: 'updateNotebookMetadata',
        physicalResourceId: PhysicalResourceId.of('id'),
        parameters: {
          Name: 'Notebook1',
          NotebookId: new PhysicalResourceIdReference(),
        },
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::AthenaNotebook', {
      Create: JSON.stringify({
        service: 'Athena',
        action: 'createNotebook',
        physicalResourceId: {
          id: 'id',
        },
        parameters: {
          WorkGroup: 'WorkGroupA',
          Name: 'Notebook1',
        },
      }),
      Update: JSON.stringify({
        service: 'Athena',
        action: 'updateNotebookMetadata',
        physicalResourceId: {
          id: 'id',
        },
        parameters: {
          Name: 'Notebook1',
          NotebookId: 'PHYSICAL:RESOURCEID:',
        },
      }),
    });
  });

  // physicalResourceId pattern #2
  test('physicalResourceId is specified in onCreate, is not in onUpdate then absent', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::AthenaNotebook',
      onCreate: {
        service: 'Athena',
        action: 'createNotebook',
        physicalResourceId: PhysicalResourceId.fromResponse('NotebookId'),
        parameters: {
          WorkGroup: 'WorkGroupA',
          Name: 'Notebook1',
        },
      },
      onUpdate: {
        service: 'Athena',
        action: 'updateNotebookMetadata',
        parameters: {
          Name: 'Notebook1',
          NotebookId: new PhysicalResourceIdReference(),
        },
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::AthenaNotebook', {
      Create: JSON.stringify({
        service: 'Athena',
        action: 'createNotebook',
        physicalResourceId: {
          responsePath: 'NotebookId',
        },
        parameters: {
          WorkGroup: 'WorkGroupA',
          Name: 'Notebook1',
        },
      }),
      Update: JSON.stringify({
        service: 'Athena',
        action: 'updateNotebookMetadata',
        parameters: {
          Name: 'Notebook1',
          NotebookId: 'PHYSICAL:RESOURCEID:',
        },
      }),
    });
  });

  // physicalResourceId pattern #3
  test('physicalResourceId is not specified in onCreate but onUpdate then fail', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    expect(() => {
      new AwsCustomResource(stack, 'AwsSdk', {
        resourceType: 'Custom::AthenaNotebook',
        onCreate: {
          service: 'Athena',
          action: 'createNotebook',
          parameters: {
            WorkGroup: 'WorkGroupA',
            Name: 'Notebook1',
          },
        },
        onUpdate: {
          service: 'Athena',
          action: 'updateNotebookMetadata',
          physicalResourceId: PhysicalResourceId.of('id'),
          parameters: {
            Name: 'Notebook1',
            NotebookId: new PhysicalResourceIdReference(),
          },
        },
        policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
      });
    }).toThrow(/'physicalResourceId' must be specified for 'onCreate' call./);
  });

  // physicalResourceId pattern #4
  test('physicalResourceId is not specified both in onCreate and onUpdate then fail', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    expect(() => {
      new AwsCustomResource(stack, 'AwsSdk', {
        resourceType: 'Custom::AthenaNotebook',
        onCreate: {
          service: 'Athena',
          action: 'createNotebook',
          parameters: {
            WorkGroup: 'WorkGroupA',
            Name: 'Notebook1',
          },
        },
        onUpdate: {
          service: 'Athena',
          action: 'updateNotebookMetadata',
          parameters: {
            Name: 'Notebook1',
            NotebookId: new PhysicalResourceIdReference(),
          },
        },
        policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
      });
    }).toThrow(/'physicalResourceId' must be specified for 'onCreate' call./);
  });

  // physicalResourceId pattern #5
  test('physicalResourceId is specified in onCreate with empty onUpdate then success', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::AthenaNotebook',
      onCreate: {
        service: 'Athena',
        action: 'createNotebook',
        physicalResourceId: PhysicalResourceId.of('id'),
        parameters: {
          WorkGroup: 'WorkGroupA',
          Name: 'Notebook1',
        },
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::AthenaNotebook', {
      Create: JSON.stringify({
        service: 'Athena',
        action: 'createNotebook',
        physicalResourceId: {
          id: 'id',
        },
        parameters: {
          WorkGroup: 'WorkGroupA',
          Name: 'Notebook1',
        },
      }),
    });
  });

  // physicalResourceId pattern #6
  test('physicalResourceId is not specified onCreate with empty onUpdate then fail', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    expect(() => {
      new AwsCustomResource(stack, 'AwsSdk', {
        resourceType: 'Custom::AthenaNotebook',
        onCreate: {
          service: 'Athena',
          action: 'createNotebook',
          parameters: {
            WorkGroup: 'WorkGroupA',
            Name: 'Notebook1',
          },
        },
        policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
      });
    }).toThrow(/'physicalResourceId' must be specified for 'onCreate' call./);
  });

  // physicalResourceId pattern #7
  test('onCreate and onUpdate both have physicalResourceId when physicalResourceId is specified in onUpdate, even when onCreate is unspecified', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::AthenaNotebook',
      onUpdate: {
        service: 'Athena',
        action: 'updateNotebookMetadata',
        physicalResourceId: PhysicalResourceId.of('id'),
        parameters: {
          Name: 'Notebook1',
          NotebookId: 'XXXX',
        },
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    Template.fromStack(stack).hasResourceProperties('Custom::AthenaNotebook', {
      Create: JSON.stringify({
        service: 'Athena',
        action: 'updateNotebookMetadata',
        physicalResourceId: {
          id: 'id',
        },
        parameters: {
          Name: 'Notebook1',
          NotebookId: 'XXXX',
        },
      }),
      Update: JSON.stringify({
        service: 'Athena',
        action: 'updateNotebookMetadata',
        physicalResourceId: {
          id: 'id',
        },
        parameters: {
          Name: 'Notebook1',
          NotebookId: 'XXXX',
        },
      }),
    });
  });

  // physicalResourceId pattern #8
  test('Omitting physicalResourceId in onCreate when onUpdate is undefined throws an error', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    expect(() => {
      new AwsCustomResource(stack, 'AwsSdk', {
        resourceType: 'Custom::AthenaNotebook',
        onUpdate: {
          service: 'Athena',
          action: 'updateNotebookMetadata',
          parameters: {
            Name: 'Notebook1',
            NotebookId: 'XXXX',
          },
        },
        policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
      });
    }).toThrow(/'physicalResourceId' must be specified for 'onUpdate' call when 'onCreate' is omitted./);
  });
});

test('booleans are encoded in the stringified parameters object', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    resourceType: 'Custom::ServiceAction',
    onCreate: {
      service: 'service',
      action: 'action',
      parameters: {
        trueBoolean: true,
        trueString: 'true',
        falseBoolean: false,
        falseString: 'false',
      },
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::ServiceAction', {
    'Create': JSON.stringify({
      'service': 'service',
      'action': 'action',
      'parameters': {
        'trueBoolean': true,
        'trueString': 'true',
        'falseBoolean': false,
        'falseString': 'false',
      },
      'physicalResourceId': {
        'id': 'id',
      },
    }),
  });
});

test('fails PhysicalResourceIdReference is passed to onCreate parameters', () => {
  const stack = new cdk.Stack();
  expect(() => new AwsCustomResource(stack, 'AwsSdk', {
    resourceType: 'Custom::ServiceAction',
    onCreate: {
      service: 'service',
      action: 'action',
      parameters: {
        physicalResourceIdReference: new PhysicalResourceIdReference(),
      },
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  })).toThrow('`PhysicalResourceIdReference` must not be specified in `onCreate` parameters.');
});

test('encodes physical resource id reference', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    resourceType: 'Custom::ServiceAction',
    onUpdate: {
      service: 'service',
      action: 'action',
      parameters: {
        trueBoolean: true,
        trueString: 'true',
        falseBoolean: false,
        falseString: 'false',
        physicalResourceIdReference: new PhysicalResourceIdReference(),
      },
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::ServiceAction', {
    'Create': JSON.stringify({
      'service': 'service',
      'action': 'action',
      'parameters': {
        'trueBoolean': true,
        'trueString': 'true',
        'falseBoolean': false,
        'falseString': 'false',
        'physicalResourceIdReference': 'PHYSICAL:RESOURCEID:',
      },
      'physicalResourceId': {
        'id': 'id',
      },
    }),
  });
});

test('timeout defaults to 2 minutes', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Timeout: 120,
  });
});

test('memorySize defaults to 512 M if installLatestAwsSdk is true', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    installLatestAwsSdk: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    MemorySize: 512,
  });
});

test('memorySize is undefined if installLatestAwsSdk is false', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    installLatestAwsSdk: false,
  });

  // THEN
  Template.fromStack(stack).resourcePropertiesCountIs('AWS::Lambda::Function', {
    MemorySize: 512,
  }, 0);
});

test('use memorySize prop if installLatestAwsSdk is true', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    installLatestAwsSdk: true,
    memorySize: 1024,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    MemorySize: 1024,
  });
});

test('use memorySize prop if installLatestAwsSdk is false', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    installLatestAwsSdk: false,
    memorySize: 1024,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    MemorySize: 1024,
  });
});

test('can specify timeout', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    timeout: cdk.Duration.minutes(15),
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Timeout: 900,
  });
});

test('implements IGrantable', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
  });
  const customResource = new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // WHEN
  role.grantPassRole(customResource.grantPrincipal);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: 'iam:PassRole',
          Effect: 'Allow',
          Resource: {
            'Fn::GetAtt': [
              'Role1ABCC5F0',
              'Arn',
            ],
          },
        },
      ],
      Version: '2012-10-17',
    },
  });
});

test('can use existing role', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const role = iam.Role.fromRoleArn(stack, 'Role', 'arn:aws:iam::123456789012:role/CoolRole');

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    role,
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Role: 'arn:aws:iam::123456789012:role/CoolRole',
  });

  Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 0);
});

test('getData', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const awsSdk = new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // WHEN
  const token = awsSdk.getResponseFieldReference('Data');

  // THEN
  expect(stack.resolve(token)).toEqual({
    'Fn::GetAtt': [
      'AwsSdkE966FE43',
      'Data',
    ],
  });
});

test('fails when getData is used with `ignoreErrorCodesMatching`', () => {
  const stack = new cdk.Stack();

  const resource = new AwsCustomResource(stack, 'AwsSdk', {
    onUpdate: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      ignoreErrorCodesMatching: '.*',
      physicalResourceId: PhysicalResourceId.of('Id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  expect(() => resource.getResponseFieldReference('ShouldFail')).toThrow(/`getData`.+`ignoreErrorCodesMatching`/);
});

test('fails when getDataString is used with `ignoreErrorCodesMatching`', () => {
  const stack = new cdk.Stack();

  const resource = new AwsCustomResource(stack, 'AwsSdk', {
    onUpdate: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      ignoreErrorCodesMatching: '.*',
      physicalResourceId: PhysicalResourceId.of('Id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  expect(() => resource.getResponseField('ShouldFail')).toThrow(/`getDataString`.+`ignoreErrorCodesMatching`/);
});

test('fail when `PhysicalResourceId.fromResponse` is used with `ignoreErrorCodesMatching', () => {
  const stack = new cdk.Stack();
  expect(() => new AwsCustomResource(stack, 'AwsSdkOnUpdate', {
    onUpdate: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      ignoreErrorCodesMatching: '.*',
      physicalResourceId: PhysicalResourceId.fromResponse('Response'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  })).toThrow(/`PhysicalResourceId.fromResponse`.+`ignoreErrorCodesMatching`/);

  expect(() => new AwsCustomResource(stack, 'AwsSdkOnCreate', {
    onCreate: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      ignoreErrorCodesMatching: '.*',
      physicalResourceId: PhysicalResourceId.fromResponse('Response'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  })).toThrow(/`PhysicalResourceId.fromResponse`.+`ignoreErrorCodesMatching`/);

  expect(() => new AwsCustomResource(stack, 'AwsSdkOnDelete', {
    onDelete: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      ignoreErrorCodesMatching: '.*',
      physicalResourceId: PhysicalResourceId.fromResponse('Response'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  })).toThrow(/`PhysicalResourceId.fromResponse`.+`ignoreErrorCodesMatching`/);
});

test('getDataString', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const awsSdk = new AwsCustomResource(stack, 'AwsSdk1', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk2', {
    onCreate: {
      service: 'service',
      action: 'action',
      parameters: {
        a: awsSdk.getResponseField('Data'),
      },
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::AWS', {
    Create: {
      'Fn::Join': [
        '',
        [
          '{"service":"service","action":"action","parameters":{"a":"',
          {
            'Fn::GetAtt': [
              'AwsSdk155B91071',
              'Data',
            ],
          },
          '"},"physicalResourceId":{"id":"id"}}',
        ],
      ],
    },
  });
});

test('can specify log retention', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    logRetention: logs.RetentionDays.ONE_WEEK,
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::LogRetention', {
    LogGroupName: {
      'Fn::Join': [
        '',
        [
          '/aws/lambda/',
          {
            Ref: 'AWS679f53fac002430cb0da5b7982bd22872D164C4C',
          },
        ],
      ],
    },
    RetentionInDays: 7,
  });
});

test('can specify log group', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    logGroup: new logs.LogGroup(stack, 'LogGroup', {
      logGroupName: '/test/log/group/name',
      retention: logs.RetentionDays.ONE_WEEK,
    }),
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', {
    LogGroupName: '/test/log/group/name',
    RetentionInDays: 7,
  });
});

test('disable AWS SDK installation', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    installLatestAwsSdk: false,
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::AWS', {
    'InstallLatestAwsSdk': false,
  });
});

test('can specify function name', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    functionName: 'my-cool-function',
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    FunctionName: 'my-cool-function',
  });
});

test('separate policies per custom resource', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'Custom1', {
    onCreate: {
      service: 'service1',
      action: 'action1',
      physicalResourceId: PhysicalResourceId.of('id1'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });
  new AwsCustomResource(stack, 'Custom2', {
    onCreate: {
      service: 'service2',
      action: 'action2',
      physicalResourceId: PhysicalResourceId.of('id2'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: 'service1:Action1',
          Effect: 'Allow',
          Resource: '*',
        },
      ],
      Version: '2012-10-17',
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: 'service2:Action2',
          Effect: 'Allow',
          Resource: '*',
        },
      ],
      Version: '2012-10-17',
    },
  });
});

test('tokens can be used as dictionary keys', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const dummy = new cdk.CfnResource(stack, 'MyResource', {
    type: 'AWS::My::Resource',
  });

  // WHEN
  new AwsCustomResource(stack, 'Custom1', {
    onCreate: {
      service: 'service1',
      action: 'action1',
      physicalResourceId: PhysicalResourceId.of('id1'),
      parameters: {
        [dummy.ref]: {
          Foo: 1234,
          Bar: dummy.getAtt('Foorz'),
        },
      },
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::AWS', {
    Create: {
      'Fn::Join': [
        '',
        [
          '{"service":"service1","action":"action1","physicalResourceId":{"id":"id1"},"parameters":{"',
          {
            'Ref': 'MyResource',
          },
          '":{"Foo":1234,"Bar":"',
          {
            'Fn::GetAtt': [
              'MyResource',
              'Foorz',
            ],
          },
          '"}}}',
        ],
      ],
    },
  });
});

test('assumedRoleArn adds statement for sts:assumeRole', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      assumedRoleArn: 'arn:aws:iam::111111111111:role/Role',
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Resource: 'arn:aws:iam::111111111111:role/Role',
        },
      ],
      Version: '2012-10-17',
    },
  });
});

test('fails when at least one of policy or role is not specified', () => {
  const stack = new cdk.Stack();
  expect(() => new AwsCustomResource(stack, 'AwsSdk', {
    onUpdate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
      parameters: {
        param: 'param',
      },
    },
  })).toThrow(/`policy`.+`role`/);
});

test('can provide no policy if using existing role', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const role = iam.Role.fromRoleArn(stack, 'Role', 'arn:aws:iam::123456789012:role/CoolRole');
  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    role,
  });
  // THEN
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 0);
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
});

test('can specify VPC', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'TestVpc');

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    vpc,
    vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    VpcConfig: {
      SubnetIds: stack.resolve(vpc.privateSubnets.map(subnet => subnet.subnetId)),
    },
  });
});

test('specifying public subnets results in a synthesis error', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'TestVpc');

  // THEN
  expect(() => {
    new AwsCustomResource(stack, 'AwsSdk', {
      onCreate: {
        service: 'service',
        action: 'action',
        physicalResourceId: PhysicalResourceId.of('id'),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
    });
  }).toThrow(/Lambda Functions in a public subnet/);
});

test('not specifying vpcSubnets when only public subnets exist on a VPC results in an error', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'TestPublicOnlyVpc', {
    subnetConfiguration: [{ name: 'public', subnetType: ec2.SubnetType.PUBLIC }],
  });

  // THEN
  expect(() => {
    new AwsCustomResource(stack, 'AwsSdk', {
      onCreate: {
        service: 'service',
        action: 'action',
        physicalResourceId: PhysicalResourceId.of('id'),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
      vpc,
    });
  }).toThrow(/Lambda Functions in a public subnet/);
});

test('vpcSubnets filter is not required when only isolated subnets exist', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'TestPrivateOnlyVpc', {
    subnetConfiguration: [
      { name: 'test1private', subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      { name: 'test2private', subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
    ],
  });

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    vpc,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    VpcConfig: {
      SubnetIds: stack.resolve(vpc.isolatedSubnets.map(subnet => subnet.subnetId)),
    },
  });
});

test('vpcSubnets filter is not required for the default VPC configuration', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const vpc = new ec2.Vpc(stack, 'TestVpc');

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    vpc,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    VpcConfig: {
      SubnetIds: stack.resolve(vpc.privateSubnets.map(subnet => subnet.subnetId)),
    },
  });
});

test('vpcSubnets without vpc results in an error', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  expect(() => new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
  })).toThrow('Cannot configure \'vpcSubnets\' without configuring a VPC');
});

test.each([
  [undefined, true],
  [true, true],
  [false, false],
])('feature flag %p, installLatestAwsSdk %p', (flag, expected) => {
  // GIVEN
  const app = new App({
    context: {
      '@aws-cdk/customresources:installLatestAwsSdkDefault': flag,
    },
  });
  const stack = new Stack(app, 'Stack');

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    resourceType: 'Custom::LogRetentionPolicy',
    onCreate: {
      service: 'CloudWatchLogs',
      action: 'putRetentionPolicy',
      parameters: {
        logGroupName: '/aws/lambda/loggroup',
        retentionInDays: 90,
      },
      physicalResourceId: PhysicalResourceId.of('loggroup'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::LogRetentionPolicy', {
    'InstallLatestAwsSdk': expected,
  });
});

test('default removal policy is DESTROY', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
  });

  // THEN
  Template.fromStack(stack).hasResource('Custom::AWS', {
    DeletionPolicy: 'Delete',
  });
});

test('can specify removal policy', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    removalPolicy: cdk.RemovalPolicy.RETAIN,
  });

  // THEN
  Template.fromStack(stack).hasResource('Custom::AWS', {
    DeletionPolicy: 'Retain',
  });
});

test('set serviceTimeout', () => {
  // GIVEN
  const stack = new cdk.Stack();

  // WHEN
  new AwsCustomResource(stack, 'AwsSdk', {
    onCreate: {
      service: 'service',
      action: 'action',
      physicalResourceId: PhysicalResourceId.of('id'),
    },
    policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    serviceTimeout: cdk.Duration.seconds(60),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::AWS', {
    ServiceTimeout: '60',
  });
});

describe('logging configuration', () => {
  test('without logging configured', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::S3PutObject',
      onUpdate: {
        service: 's3',
        action: 'putObject',
        parameters: {
          Bucket: 'my-bucket',
          Key: 'my-key',
          Body: 'my-body',
        },
        physicalResourceId: PhysicalResourceId.fromResponse('ETag'),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::S3PutObject', {
      'Create': JSON.stringify({
        'service': 's3',
        'action': 'putObject',
        'parameters': {
          'Bucket': 'my-bucket',
          'Key': 'my-key',
          'Body': 'my-body',
        },
        'physicalResourceId': {
          'responsePath': 'ETag',
        },
      }),
      'Update': JSON.stringify({
        'service': 's3',
        'action': 'putObject',
        'parameters': {
          'Bucket': 'my-bucket',
          'Key': 'my-key',
          'Body': 'my-body',
        },
        'physicalResourceId': {
          'responsePath': 'ETag',
        },
      }),
    });
  });

  test('without logging configured and feature flag enabled', () => {
    // GIVEN
    const app = new cdk.App({ postCliContext: { [LOG_API_RESPONSE_DATA_PROPERTY_TRUE_DEFAULT]: true } });
    const stack = new cdk.Stack(app);

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::S3PutObject',
      onUpdate: {
        service: 's3',
        action: 'putObject',
        parameters: {
          Bucket: 'my-bucket',
          Key: 'my-key',
          Body: 'my-body',
        },
        physicalResourceId: PhysicalResourceId.fromResponse('ETag'),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::S3PutObject', {
      'Create': JSON.stringify({
        'service': 's3',
        'action': 'putObject',
        'parameters': {
          'Bucket': 'my-bucket',
          'Key': 'my-key',
          'Body': 'my-body',
        },
        'physicalResourceId': {
          'responsePath': 'ETag',
        },
        'logApiResponseData': true,
      }),
      'Update': JSON.stringify({
        'service': 's3',
        'action': 'putObject',
        'parameters': {
          'Bucket': 'my-bucket',
          'Key': 'my-key',
          'Body': 'my-body',
        },
        'physicalResourceId': {
          'responsePath': 'ETag',
        },
        'logApiResponseData': true,
      }),
    });
  });

  test('with Logging.all() configured', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::LogRetentionPolicy',
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: PhysicalResourceId.of('loggroup'),
        logging: Logging.all(),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetentionPolicy', {
      Create: JSON.stringify({
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: {
          id: 'loggroup',
        },
      }),
    });
  });

  test('with Logging.all() configured and feature flag enabled', () => {
    // GIVEN
    const app = new App({ postCliContext: { [LOG_API_RESPONSE_DATA_PROPERTY_TRUE_DEFAULT]: true } });
    const stack = new Stack(app);

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::LogRetentionPolicy',
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: PhysicalResourceId.of('loggroup'),
        logging: Logging.all(),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetentionPolicy', {
      Create: JSON.stringify({
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: {
          id: 'loggroup',
        },
        logApiResponseData: true,
      }),
    });
  });

  test('with Logging.withDataHidden() configured', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::LogRetentionPolicy',
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: PhysicalResourceId.of('loggroup'),
        logging: Logging.withDataHidden(),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetentionPolicy', {
      Create: JSON.stringify({
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: {
          id: 'loggroup',
        },
        logApiResponseData: false,
      }),
    });
  });

  test('with Logging.withDataHidden() configured and feature flag enabled', () => {
    // GIVEN
    const app = new App({ postCliContext: { [LOG_API_RESPONSE_DATA_PROPERTY_TRUE_DEFAULT]: true } });
    const stack = new Stack(app);

    // WHEN
    new AwsCustomResource(stack, 'AwsSdk', {
      resourceType: 'Custom::LogRetentionPolicy',
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: PhysicalResourceId.of('loggroup'),
        logging: Logging.withDataHidden(),
      },
      policy: AwsCustomResourcePolicy.fromSdkCalls({ resources: AwsCustomResourcePolicy.ANY_RESOURCE }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetentionPolicy', {
      Create: JSON.stringify({
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: '/aws/lambda/loggroup',
          retentionInDays: 90,
        },
        physicalResourceId: {
          id: 'loggroup',
        },
        logApiResponseData: false,
      }),
    });
  });
});
