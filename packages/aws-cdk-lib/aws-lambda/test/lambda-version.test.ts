import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Template } from '../../assertions';
import * as cdk from '../../core';
import * as lambda from '../lib';

const THE_RUNTIME = new lambda.Runtime('node99.x', lambda.RuntimeFamily.NODEJS, {
  supportsInlineCode: true,
});

describe('lambda version', () => {
  test('can import a Lambda version by ARN', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const version = lambda.Version.fromVersionArn(stack, 'Version', 'arn:aws:lambda:region:account-id:function:function-name:version');

    new cdk.CfnOutput(stack, 'ARN', { value: version.functionArn });
    new cdk.CfnOutput(stack, 'Name', { value: version.functionName });
    new cdk.CfnOutput(stack, 'FunctionArn', { value: version.lambda.functionArn });

    // THEN
    Template.fromStack(stack).templateMatches({
      Outputs: {
        ARN: {
          Value: 'arn:aws:lambda:region:account-id:function:function-name:version',
        },
        Name: {
          Value: 'function-name:version',
        },
        FunctionArn: {
          Value: 'arn:aws:lambda:region:account-id:function:function-name',
        },
      },
    });
  });

  test('can import an imported Lambda version by ARN', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const func = new lambda.Function(stack, 'Fn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
    });
    const version = new lambda.Version(stack, 'Version', {
      lambda: func,
      maxEventAge: cdk.Duration.hours(1),
      retryAttempts: 0,
    });
    const importedVersion = lambda.Version.fromVersionArn(stack, 'ImportedVersion', version.functionArn);
    expect(importedVersion.functionArn).toBe(version.functionArn);
  });

  test('create a version with event invoke config', () => {
    // GIVEN
    const stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge(
      { id: 'CloudFormation-Validate::W3030', reason: 'Tests intentionally use a bogus runtime' },
    );
    const fn = new lambda.Function(stack, 'Fn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
    });

    // WHEN
    new lambda.Version(stack, 'Version', {
      lambda: fn,
      maxEventAge: cdk.Duration.hours(1),
      retryAttempts: 0,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: {
        Ref: 'Fn9270CBC0',
      },
      Qualifier: {
        'Fn::GetAtt': [
          'Version6A868472',
          'Version',
        ],
      },
      MaximumEventAgeInSeconds: 3600,
      MaximumRetryAttempts: 0,
    });
  });

  test('throws when calling configureAsyncInvoke on already configured version', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Fn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
    });
    const version = new lambda.Version(stack, 'Version', {
      lambda: fn,
      maxEventAge: cdk.Duration.hours(1),
      retryAttempts: 0,
    });

    // THEN
    expect(() => version.configureAsyncInvoke({ retryAttempts: 1 })).toThrow(/An EventInvokeConfig has already been configured/);
  });

  test('event invoke config on imported versions', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const version1 = lambda.Version.fromVersionArn(stack, 'Version1', 'arn:aws:lambda:region:account-id:function:function-name:version1');
    const version2 = lambda.Version.fromVersionArn(stack, 'Version2', 'arn:aws:lambda:region:account-id:function:function-name:version2');

    // WHEN
    version1.configureAsyncInvoke({
      retryAttempts: 1,
    });
    version2.configureAsyncInvoke({
      retryAttempts: 0,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: 'function-name',
      Qualifier: 'version1',
      MaximumRetryAttempts: 1,
    });
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: 'function-name',
      Qualifier: 'version2',
      MaximumRetryAttempts: 0,
    });
  });

  testDeprecated('addAlias can be used to add an alias that points to a version', () => {
    // GIVEN
    const stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge(
      { id: 'CloudFormation-Validate::W3030', reason: 'Tests intentionally use a bogus runtime' },
    );
    const fn = new lambda.Function(stack, 'Fn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
    });
    const version = fn.currentVersion;

    // WHEN
    version.addAlias('foo');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Alias', {
      FunctionName: {
        Ref: 'Fn9270CBC0',
      },
      FunctionVersion: {
        'Fn::GetAtt': [
          'FnCurrentVersion17A89ABB30f50e285b0533137b4b353595c6ba57',
          'Version',
        ],
      },
      Name: 'foo',
    });
  });

  test('edgeArn', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Fn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
    });
    const version = fn.currentVersion;

    // THEN
    expect(stack.resolve(version.edgeArn)).toEqual({ Ref: 'FnCurrentVersion17A89ABB30f50e285b0533137b4b353595c6ba57' });
  });

  test('edgeArn throws with $LATEST', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const version = lambda.Version.fromVersionArn(stack, 'Version', 'arn:aws:lambda:region:account-id:function:function-name:$LATEST');

    // THEN
    expect(() => version.edgeArn).toThrow(/\$LATEST function version cannot be used for Lambda@Edge/);
  });

  test('edgeArn throws at synthesis if underlying function is not edge compatible', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    const fn = new lambda.Function(stack, 'Fn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
    });
    const version = fn.currentVersion;

    // WHEN
    new lambda.Function(stack, 'OtherFn', {
      runtime: THE_RUNTIME,
      handler: 'index.handler',
      code: lambda.Code.fromInline('foo'),
      environment: {
        EDGE_ARN: version.edgeArn, // Consume edgeArn
      },
    });
    // make fn incompatible for Lambda@Edge after consuming edgeArn
    fn.addEnvironment('KEY1', 'value1');
    fn.addEnvironment('KEY2', 'value2');

    // THEN
    expect(() => app.synth()).toThrow(/KEY1,KEY2/);
  });

  test('throws when adding FunctionUrl to a Version', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('hello()'),
      handler: 'index.hello',
      runtime: THE_RUNTIME,
    });
    const version = new lambda.Version(stack, 'Version', {
      lambda: fn,
      maxEventAge: cdk.Duration.hours(1),
      retryAttempts: 0,
    });

    // WHEN
    expect(() => {
      version.addFunctionUrl();
    }).toThrow(/FunctionUrl cannot be used with a Version/);
  });

  test('version\'s implementation of IFunctionRef should point to the version', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('hello()'),
      handler: 'index.hello',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    // WHEN
    const ver = new lambda.Version(stack, 'Version', {
      lambda: fn,
    });

    // THEN
    expect(ver.functionRef.functionArn).toEqual(ver.functionArn);
  });

  test('should throw error when version has provisioned concurrency and function has tenancy config', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('hello()'),
      handler: 'index.hello',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tenancyConfig: lambda.TenancyConfig.PER_TENANT,
    });

    // WHEN & THEN
    expect(() => new lambda.Version(stack, 'Version', {
      lambda: fn,
      provisionedConcurrentExecutions: 10,
    })).toThrow('Provisioned Concurrency is not supported for functions with tenant isolation mode');
  });

  test('provisionedConcurrentExecutions can be a token', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('hello()'),
      handler: 'index.hello',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    const pce = new cdk.CfnParameter(stack, 'ProvisionedConcurrentExecutions', { type: 'Number' });

    // WHEN
    new lambda.Version(stack, 'Version', {
      lambda: fn,
      provisionedConcurrentExecutions: pce.valueAsNumber,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Version', {
      ProvisionedConcurrencyConfig: {
        ProvisionedConcurrentExecutions: { Ref: 'ProvisionedConcurrentExecutions' },
      },
    });
  });

  describe('version scaling configuration', () => {
    test('version with min and max execution environments', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      new lambda.Version(stack, 'Version', {
        lambda: fn,
        minExecutionEnvironments: 1,
        maxExecutionEnvironments: 10,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Version', {
        FunctionScalingConfig: {
          MinExecutionEnvironments: 1,
          MaxExecutionEnvironments: 10,
        },
      });
    });

    test('version with only min execution environments', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      new lambda.Version(stack, 'Version', {
        lambda: fn,
        minExecutionEnvironments: 2,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Version', {
        FunctionScalingConfig: {
          MinExecutionEnvironments: 2,
        },
      });
    });

    test('version with only max execution environments', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      new lambda.Version(stack, 'Version', {
        lambda: fn,
        maxExecutionEnvironments: 5,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Version', {
        FunctionScalingConfig: {
          MaxExecutionEnvironments: 5,
        },
      });
    });

    test('throws when minExecutionEnvironments is negative', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      expect(() => {
        new lambda.Version(stack, 'Version', {
          lambda: fn,
          minExecutionEnvironments: -1,
        });
      }).toThrow(/minExecutionEnvironments must be a non-negative integer/);
    });

    test('throws when maxExecutionEnvironments is negative', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      expect(() => {
        new lambda.Version(stack, 'Version', {
          lambda: fn,
          maxExecutionEnvironments: -1,
        });
      }).toThrow(/maxExecutionEnvironments must be a non-negative integer/);
    });

    test('throws when minExecutionEnvironments is greater than capacityProviderMaxExecutionEnvironments', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      expect(() => {
        new lambda.Version(stack, 'Version', {
          lambda: fn,
          minExecutionEnvironments: 10,
          maxExecutionEnvironments: 5,
        });
      }).toThrow(/minExecutionEnvironments must be less than or equal to maxExecutionEnvironments/);
    });

    test('accepts tokens for execution environment scaling config', () => {
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Fn', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      const tokenMin = cdk.Token.asNumber(cdk.Fn.ref('MinEnvParam'));
      const tokenMax = cdk.Token.asNumber(cdk.Fn.ref('MaxEnvParam'));

      // WHEN - should not throw
      new lambda.Version(stack, 'Version', {
        lambda: fn,
        minExecutionEnvironments: tokenMin,
        maxExecutionEnvironments: tokenMax,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Version', {
        FunctionScalingConfig: {
          MinExecutionEnvironments: { Ref: 'MinEnvParam' },
          MaxExecutionEnvironments: { Ref: 'MaxEnvParam' },
        },
      });
    });
  });
});
