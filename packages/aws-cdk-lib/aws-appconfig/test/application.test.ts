import { Template } from '../../assertions';
import { FargateTaskDefinition } from '../../aws-ecs';
import { Code, Function, Runtime } from '../../aws-lambda';
import * as cdk from '../../core';
import {
  Application,
  Platform,
  LambdaDestination,
  Parameter,
  ActionPoint,
} from '../lib';

describe('appconfig', () => {
  test('basic appconfig', () => {
    const stack = new cdk.Stack();
    new Application(stack, 'MyAppConfig');

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Application', {
      Name: 'MyAppConfig',
    });
  });

  test('appconfig with name', () => {
    const stack = new cdk.Stack();
    new Application(stack, 'MyAppConfig', {
      applicationName: 'TestApp',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Application', {
      Name: 'TestApp',
    });
  });

  test('appconfig with description', () => {
    const stack = new cdk.Stack();
    new Application(stack, 'MyAppConfig', {
      description: 'This is my description',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Application', {
      Name: 'MyAppConfig',
      Description: 'This is my description',
    });
  });

  test('get lambda layer arn', () => {
    expect(Application.getLambdaLayerVersionArn('us-east-1')).toEqual('arn:aws:lambda:us-east-1:027255383542:layer:AWS-AppConfig-Extension:128');
    expect(Application.getLambdaLayerVersionArn('us-east-1', Platform.ARM_64)).toEqual('arn:aws:lambda:us-east-1:027255383542:layer:AWS-AppConfig-Extension-Arm64:61');
  });

  test('add agent to ecs', () => {
    const stack = new cdk.Stack();
    const taskDef = new FargateTaskDefinition(stack, 'TaskDef');
    Application.addAgentToEcs(taskDef);

    Template.fromStack(stack).hasResourceProperties('AWS::ECS::TaskDefinition', {
      ContainerDefinitions: [
        {
          Image: 'public.ecr.aws/aws-appconfig/aws-appconfig-agent:latest',
          Name: 'AppConfigAgentContainer',
          Essential: true,
        },
      ],
    });
  });

  test('specifying action point for extensible action on', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    appconfig.on(ActionPoint.ON_DEPLOYMENT_STEP, new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        ON_DEPLOYMENT_STEP: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: { 'Fn::GetAtt': ['MyFunc8A243A2C', 'Arn'] },
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('pre create hosted configuration version', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    appconfig.preCreateHostedConfigurationVersion(new LambdaDestination(func), {
      description: 'This is my description',
      extensionName: 'MyExtension',
      latestVersionNumber: 1,
      parameters: [
        Parameter.required('myparam', 'val'),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyExtension',
      Description: 'This is my description',
      LatestVersionNumber: 1,
      Actions: {
        PRE_CREATE_HOSTED_CONFIGURATION_VERSION: [
          {
            Name: 'MyExtension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionE4CCERole467D69791333F', 'Arn'] },
            Uri: { 'Fn::GetAtt': ['MyFunc8A243A2C', 'Arn'] },
          },
        ],
      },
      Parameters: {
        myparam: { Required: true },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionE4CCE34485313', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionE4CCE34485313', 'VersionNumber'],
      },
      Parameters: {
        myparam: 'val',
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('pre start deployment', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.preStartDeployment(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        PRE_START_DEPLOYMENT: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('on deployment start', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    Object.defineProperty(appconfig, 'applicationArn', {
      value: 'arn:aws:appconfig:us-east-1:123456789012:application/abc123',
    });
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.onDeploymentStart(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        ON_DEPLOYMENT_START: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('on deployment step', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.onDeploymentStep(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        ON_DEPLOYMENT_STEP: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('on deployment complete', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.onDeploymentComplete(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        ON_DEPLOYMENT_COMPLETE: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('on deployment bake', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.onDeploymentBaking(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        ON_DEPLOYMENT_BAKING: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('on deployment rolled back', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.onDeploymentRolledBack(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        ON_DEPLOYMENT_ROLLED_BACK: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('at deployment tick', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.atDeploymentTick(new LambdaDestination(func));

    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::Extension', {
      Name: 'MyAppConfig-Extension',
      Actions: {
        AT_DEPLOYMENT_TICK: [
          {
            Name: 'MyAppConfig-Extension-0',
            RoleArn: { 'Fn::GetAtt': ['MyAppConfigExtensionF845ERole0D30970E5A7E5', 'Arn'] },
            Uri: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
          },
        ],
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::AppConfig::ExtensionAssociation', {
      ExtensionIdentifier: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'Id'],
      },
      ExtensionVersionNumber: {
        'Fn::GetAtt': ['MyAppConfigExtensionF845EC11D4079', 'VersionNumber'],
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
            { Ref: 'MyAppConfigB4B63E75' },
          ],
        ],
      },
    });
  });

  test('create same extension twice', () => {
    const stack = new cdk.Stack();
    const appconfig = new Application(stack, 'MyAppConfig');
    const func = new Function(stack, 'MyFunc', {
      handler: 'index.handler',
      runtime: Runtime.PYTHON_3_9,
      code: Code.fromInline('# this is my code'),
    });
    Object.defineProperty(func, 'functionArn', {
      value: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });
    appconfig.preStartDeployment(new LambdaDestination(func));

    expect(() => {
      appconfig.preStartDeployment(new LambdaDestination(func));
    }).toThrow();
  });

  test('from application arn', () => {
    const stack = new cdk.Stack();
    const app = Application.fromApplicationArn(stack, 'Application',
      'arn:aws:appconfig:us-west-2:123456789012:application/abc123');

    expect(app.applicationId).toEqual('abc123');
  });

  test('from application arn with no resource name', () => {
    const stack = new cdk.Stack();
    expect(() => {
      Application.fromApplicationArn(stack, 'Application',
        'arn:aws:appconfig:us-west-2:123456789012:application/');
    }).toThrow('Missing required application id from application ARN');
  });

  test('from application id', () => {
    const cdkApp = new cdk.App();
    const stack = new cdk.Stack(cdkApp, 'Stack', {
      env: {
        region: 'us-west-2',
        account: '123456789012',
      },
    });
    const app = Application.fromApplicationId(stack, 'Application', 'abc123');

    expect(app.applicationId).toEqual('abc123');
  });
});
