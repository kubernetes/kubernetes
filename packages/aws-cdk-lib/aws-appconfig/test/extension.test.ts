import { Template } from '../../assertions';
import { EventBus } from '../../aws-events';
import { Effect, Role, ServicePrincipal } from '../../aws-iam';
import * as lambda from '../../aws-lambda';
import { Topic } from '../../aws-sns';
import { Queue } from '../../aws-sqs';
import * as cdk from '../../core';
import {
  Action,
  ActionPoint,
  Application,
  ConfigurationContent,
  Extension,
  HostedConfiguration,
  Parameter,
  LambdaDestination,
  SqsDestination,
  SnsDestination,
  EventBridgeDestination,
} from '../lib';

describe('extension', () => {
  test('simple extension with lambda', () => {
    const stack = new cdk.Stack();
    const func = new lambda.Function(stack, 'MyFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_COMPLETE,
            ActionPoint.ON_DEPLOYMENT_ROLLED_BACK,
            ActionPoint.AT_DEPLOYMENT_TICK,
          ],
          eventDestination: new LambdaDestination(func),
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: { 'Fn::GetAtt': ['MyFunction3BAA72D1', 'Arn'] },
          },
        ],
        ON_DEPLOYMENT_ROLLED_BACK: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: { 'Fn::GetAtt': ['MyFunction3BAA72D1', 'Arn'] },
          },
        ],
        AT_DEPLOYMENT_TICK: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: { 'Fn::GetAtt': ['MyFunction3BAA72D1', 'Arn'] },
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: { 'Fn::GetAtt': ['MyFunction3BAA72D1', 'Arn'] },
                Action: [
                  'lambda:InvokeFunction',
                  'lambda:InvokeAsync',
                ],
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    });
  });

  test('simple extension with two lambdas', () => {
    const stack = new cdk.Stack();
    const func1 = new lambda.Function(stack, 'MyFunction1', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    const func2 = new lambda.Function(stack, 'MyFunction2', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    Object.defineProperty(func1, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    Object.defineProperty(func2, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-diff-function',
    });
    new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_COMPLETE,
          ],
          eventDestination: new LambdaDestination(func1),
        }),
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_ROLLED_BACK,
          ],
          eventDestination: new LambdaDestination(func2),
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
        ON_DEPLOYMENT_ROLLED_BACK: [
          {
            Name: 'MyExtension-1',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRoleBE614F182C70A', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-diff-function',
          },
        ],
      },
    });
    Template.fromStack(stack).resourcePropertiesCountIs('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
                Action: [
                  'lambda:InvokeFunction',
                  'lambda:InvokeAsync',
                ],
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    }, 1);
    Template.fromStack(stack).resourcePropertiesCountIs('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:lambda:us-east-1:123456789012:function:my-diff-function',
                Action: [
                  'lambda:InvokeFunction',
                  'lambda:InvokeAsync',
                ],
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    }, 1);
  });

  test('extension with all props using lambda', () => {
    const stack = new cdk.Stack();
    const func = new lambda.Function(stack, 'MyFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    const appconfig = new Application(stack, 'MyApplication', {
      applicationName: 'MyApplication',
    });
    const ext = new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_COMPLETE,
          ],
          eventDestination: new LambdaDestination(func),
        }),
      ],
      extensionName: 'TestExtension',
      description: 'This is my extension',
      parameters: [
        Parameter.required('testVariable', 'hello'),
        Parameter.notRequired('testNotRequiredVariable'),
      ],
      latestVersionNumber: 1,
    });
    appconfig.addExtension(ext);

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'TestExtension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'TestExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRoleCA98491716207', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
      Description: 'This is my extension',
      Parameters: {
        testVariable: { Required: true },
        testNotRequiredVariable: { Required: false },
      },
      LatestVersionNumber: 1,
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyExtension89A915D0', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyExtension89A915D0', 'VersionNumber'],
      },
      Parameters: {
        testVariable: 'hello',
      },
      ResourceIdentifier: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':appconfig:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':application/',
            { Ref: 'MyApplication5C63EC1D' },
          ],
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
                Action: [
                  'lambda:InvokeFunction',
                  'lambda:InvokeAsync',
                ],
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    });
  });

  test('extension with queue', () => {
    const stack = new cdk.Stack();
    const queue = new Queue(stack, 'MyQueue');
    Object.defineProperty(queue, 'queueArn', {
      value: 'arn:aws:sqs:us-east-1:123456789012:my-queue',
    });
    new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_ROLLED_BACK,
          ],
          eventDestination: new SqsDestination(queue),
          name: 'ActionTestName',
          description: 'This is my action description',
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_ROLLED_BACK: [
          {
            Description: 'This is my action description',
            Name: 'ActionTestName',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole76B022BC4F2BC', 'Arn'] },
            Uri: 'arn:aws:sqs:us-east-1:123456789012:my-queue',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:sqs:us-east-1:123456789012:my-queue',
                Action: 'sqs:SendMessage',
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    });
  });

  test('extension with topic', () => {
    const stack = new cdk.Stack();
    const topic = new Topic(stack, 'MyTopic');
    Object.defineProperty(topic, 'topicArn', {
      value: 'arn:aws:sns:us-east-1:123456789012:my-topic',
    });
    new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_BAKING,
          ],
          eventDestination: new SnsDestination(topic),
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_BAKING: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: 'arn:aws:sns:us-east-1:123456789012:my-topic',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:sns:us-east-1:123456789012:my-topic',
                Action: 'sns:Publish',
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    });
  });

  test('extension with bus', () => {
    const stack = new cdk.Stack();
    const bus = new EventBus(stack, 'MyEventBus');
    Object.defineProperty(bus, 'eventBusArn', {
      value: 'arn:aws:events:us-east-1:123456789012:event-bus/aws.partner/PartnerName/acct1/repo1',
    });
    new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_BAKING,
          ],
          eventDestination: new EventBridgeDestination(bus),
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_BAKING: [
          {
            Name: 'MyExtension-0',
            Uri: 'arn:aws:events:us-east-1:123456789012:event-bus/aws.partner/PartnerName/acct1/repo1',
          },
        ],
      },
    });
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 0);
  });

  test('extension with associated environment', () => {
    const stack = new cdk.Stack();
    const func = new lambda.Function(stack, 'MyFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    const app = new Application(stack, 'MyApplication');
    const env = app.addEnvironment('MyEnvironment');
    const ext = new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_COMPLETE,
          ],
          eventDestination: new LambdaDestination(func),
        }),
      ],
    });
    env.addExtension(ext);

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyExtension89A915D0', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyExtension89A915D0', 'VersionNumber'],
      },
      ResourceIdentifier: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':appconfig:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':application/',
            { Ref: 'MyApplication5C63EC1D' },
            '/environment/',
            { Ref: 'MyApplicationMyEnvironment10D94356' },
          ],
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
                Action: [
                  'lambda:InvokeFunction',
                  'lambda:InvokeAsync',
                ],
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    });
  });

  test('extension with associated configuration profile', () => {
    const stack = new cdk.Stack();
    const func = new lambda.Function(stack, 'MyFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    const app = new Application(stack, 'MyApplication');
    const configProfile = new HostedConfiguration(stack, 'MyConfiguration', {
      application: app,
      content: ConfigurationContent.fromInlineJson('This is my content.'),
    });
    const ext = new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_COMPLETE,
          ],
          eventDestination: new LambdaDestination(func),
        }),
      ],
    });
    configProfile.addExtension(ext);

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyExtensionRole467D6FCDEEFA5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyExtension89A915D0', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyExtension89A915D0', 'VersionNumber'],
      },
      ResourceIdentifier: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':appconfig:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':application/',
            { Ref: 'MyApplication5C63EC1D' },
            '/configurationprofile/',
            { Ref: 'MyConfigurationConfigurationProfileEE0ECA85' },
          ],
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Policies: [
        {
          PolicyDocument: {
            Statement: [
              {
                Effect: Effect.ALLOW,
                Resource: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
                Action: [
                  'lambda:InvokeFunction',
                  'lambda:InvokeAsync',
                ],
              },
            ],
          },
          PolicyName: 'AllowAppConfigInvokeExtensionEventSourcePolicy',
        },
      ],
    });
  });

  test('extension with execution role', () => {
    const stack = new cdk.Stack();
    const func = new lambda.Function(stack, 'MyFunction', {
      runtime: lambda.Runtime.PYTHON_3_8,
      code: lambda.Code.fromInline('# dummy func'),
      handler: 'index.handler',
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('appconfig.amazonaws.com'),
    });
    Object.defineProperty(role, 'roleArn', {
      value: 'arn:iam::123456789012:role/my-role',
    });
    new Extension(stack, 'MyExtension', {
      actions: [
        new Action({
          actionPoints: [
            ActionPoint.ON_DEPLOYMENT_COMPLETE,
          ],
          eventDestination: new LambdaDestination(func),
          executionRole: role,
        }),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'MyExtension-0',
            RoleArn: 'arn:iam::123456789012:role/my-role',
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
  });

  test('from extension arn', () => {
    const stack = new cdk.Stack();
    const extension = Extension.fromExtensionArn(stack, 'MyExtension',
      'arn:aws:appconfig:us-west-2:123456789012:extension/abc123/1');

    expect(extension.extensionId).toEqual('abc123');
    expect(extension.extensionVersionNumber).toEqual(1);
    expect(extension.env.account).toEqual('123456789012');
    expect(extension.env.region).toEqual('us-west-2');
  });

  test('from extension arn with no resource name', () => {
    const stack = new cdk.Stack();
    expect(() => {
      Extension.fromExtensionArn(stack, 'MyExtension',
        'arn:aws:appconfig:us-west-2:123456789012:extension/');
    }).toThrow('Missing required /$/{extensionId}//$/{extensionVersionNumber} from configuration profile ARN:');
  });

  test('from extension arn with no extension id', () => {
    const stack = new cdk.Stack();
    expect(() => {
      Extension.fromExtensionArn(stack, 'MyExtension',
        'arn:aws:appconfig:us-west-2:123456789012:extension//1');
    }).toThrow('Missing required parameters for extension ARN: format should be /$/{extensionId}//$/{extensionVersionNumber}');
  });

  test('from extension arn with no extension version number', () => {
    const stack = new cdk.Stack();
    expect(() => {
      Extension.fromExtensionArn(stack, 'MyExtension',
        'arn:aws:appconfig:us-west-2:123456789012:extension/abc123/');
    }).toThrow('Missing required parameters for extension ARN: format should be /$/{extensionId}//$/{extensionVersionNumber}');
  });

  test('from extension id', () => {
    const cdkApp = new cdk.App();
    const stack = new cdk.Stack(cdkApp, 'Stack', {
      env: {
        region: 'us-west-2',
        account: '123456789012',
      },
    });
    const extension = Extension.fromExtensionAttributes(stack, 'Extension', {
      extensionId: 'abc123',
      extensionVersionNumber: 1,
    });

    expect(extension.extensionId).toEqual('abc123');
    expect(extension.extensionVersionNumber).toEqual(1);
    expect(extension.env.account).toEqual('123456789012');
    expect(extension.env.region).toEqual('us-west-2');
  });
});
