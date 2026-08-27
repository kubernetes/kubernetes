import * as path from 'path';
import { testDeprecated, bockfs } from '@aws-cdk/cdk-build-tools';
import type * as constructs from 'constructs';
import * as _ from 'lodash';
import { Annotations, Match, Template } from '../../assertions';
import { ProfilingGroup } from '../../aws-codeguruprofiler';
import * as ec2 from '../../aws-ec2';
import * as efs from '../../aws-efs';
import * as iam from '../../aws-iam';
import { AccountPrincipal } from '../../aws-iam';
import * as kms from '../../aws-kms';
import * as logs from '../../aws-logs';
import * as s3 from '../../aws-s3';
import * as s3files from '../../aws-s3files';
import * as signer from '../../aws-signer';
import * as sns from '../../aws-sns';
import * as sqs from '../../aws-sqs';
import * as cdk from '../../core';
import { Aspects, Lazy, Size } from '../../core';
import { JSII_RUNTIME_SYMBOL } from '../../core/lib/constants';
import { getWarnings } from '../../core/test/util';
import * as cxapi from '../../cx-api';
import * as lambda from '../lib';
import { AdotLambdaLayerJavaSdkVersion } from '../lib/adot-layers';
import { calculateFunctionHash } from '../lib/function-hash';

describe('function', () => {
  const dockerLambdaHandlerPath = path.join(__dirname, 'docker-lambda-handler');
  test('default function', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument:
      {
        Statement:
          [{
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: { Service: 'lambda.amazonaws.com' },
          }],
        Version: '2012-10-17',
      },
      ManagedPolicyArns:
        [{ 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']] }],
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties:
      {
        Code: { ZipFile: 'foo' },
        Handler: 'index.handler',
        Role: { 'Fn::GetAtt': ['MyLambdaServiceRole4539ECB6', 'Arn'] },
        Runtime: 'nodejs24.x',
      },
      DependsOn: ['MyLambdaServiceRole4539ECB6'],
    });
  });

  test('function without layers omits Layers property', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Layers: Match.absent(),
    });
  });

  test('adds policy permissions', () => {
    const stack = new cdk.Stack();
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      initialPolicy: [new iam.PolicyStatement({ actions: ['*'], resources: ['*'] })],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument:
      {
        Statement:
          [{
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: { Service: 'lambda.amazonaws.com' },
          }],
        Version: '2012-10-17',
      },
      ManagedPolicyArns:
        [{ 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']] }],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: '*',
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: { ZipFile: 'foo' },
        Handler: 'index.handler',
        Role: { 'Fn::GetAtt': ['MyLambdaServiceRole4539ECB6', 'Arn'] },
        Runtime: 'nodejs24.x',
      },
      DependsOn: ['MyLambdaServiceRoleDefaultPolicy5BBC6F68', 'MyLambdaServiceRole4539ECB6'],
    });
  });

  test('fails if inline code is used for an invalid runtime', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'bar',
      runtime: lambda.Runtime.DOTNET_CORE_2,
    })).toThrow();
  });

  describe('addPermissions', () => {
    test('can be used to add permissions to the Lambda function', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);

      fn.addPermission('S3Permission', {
        action: 'lambda:*',
        principal: new iam.ServicePrincipal('s3.amazonaws.com'),
        sourceAccount: stack.account,
        sourceArn: 'arn:aws:s3:::my_bucket',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: {
                Service: 'lambda.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
        ManagedPolicyArns:
          [{ 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole']] }],
      });

      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties: {
          Code: {
            ZipFile: 'foo',
          },
          Handler: 'bar',
          Role: {
            'Fn::GetAtt': [
              'MyLambdaServiceRole4539ECB6',
              'Arn',
            ],
          },
          Runtime: 'python3.14',
        },
        DependsOn: [
          'MyLambdaServiceRole4539ECB6',
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:*',
        FunctionName: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
        Principal: 's3.amazonaws.com',
        SourceAccount: {
          Ref: 'AWS::AccountId',
        },
        SourceArn: 'arn:aws:s3:::my_bucket',
      });
    });

    test('can supply principalOrgID via permission property', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);
      const org = new iam.OrganizationPrincipal('o-xxxxxxxxxx');
      const account = new iam.AccountPrincipal('123456789012');

      fn.addPermission('S3Permission', {
        action: 'lambda:*',
        principal: account,
        organizationId: org.organizationId,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:*',
        FunctionName: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
        Principal: account.accountId,
        PrincipalOrgID: org.organizationId,
      });
    });

    test('fails if the principal is not a service, account, arn, or organization principal', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);

      expect(() => fn.addPermission('F1', { principal: new iam.CanonicalUserPrincipal('org') }))
        .toThrow(/Invalid principal type for Lambda permission statement/);

      fn.addPermission('S1', { principal: new iam.ServicePrincipal('my-service') });
      fn.addPermission('S2', { principal: new iam.AccountPrincipal('account') });
      fn.addPermission('S3', { principal: new iam.ArnPrincipal('my:arn') });
      fn.addPermission('S4', { principal: new iam.OrganizationPrincipal('o-12345abcde') });
    });

    test('does not show warning if skipPermissions is set', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app);
      const imported = lambda.Function.fromFunctionAttributes(stack, 'Imported', {
        functionArn: 'arn:aws:lambda:us-west-2:123456789012:function:my-function',
        skipPermissions: true,
      });
      imported.addPermission('Permission', {
        action: 'lambda:InvokeFunction',
        principal: new AccountPrincipal('123456789010'),
      });

      expect(getWarnings(app.synth()).length).toBe(0);
    });

    test('shows warning if skipPermissions is not set', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app);
      const imported = lambda.Function.fromFunctionAttributes(stack, 'Imported', {
        functionArn: 'arn:aws:lambda:us-west-2:123456789012:function:my-function',
      });
      imported.addPermission('Permission', {
        action: 'lambda:InvokeFunction',
        principal: new AccountPrincipal('123456789010'),
      });

      expect(getWarnings(app.synth())).toEqual([
        {
          message: expect.stringMatching(/^addPermission\(\) has no effect on a Lambda Function with region=us-west-2, account=123456789012, in a Stack with region=\${Token\[AWS\.Region\.\d+]}, account=\${Token\[AWS\.AccountId\.\d+]}. Suppress this warning if this is intentional, or pass sameEnvironment=true to fromFunctionAttributes\(\) if you would like to add the permissions\. \[ack: UnclearLambdaEnvironment]$/),
          path: '/Default/Imported',
        },
      ]);
    });

    test('applies source account/ARN conditions if the principal has conditions', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);
      const sourceAccount = 'some-account';
      const sourceArn = 'arn:aws:s3:::my-bucket';
      const service = 'my-service';
      const principal = new iam.PrincipalWithConditions(new iam.ServicePrincipal(service), {
        ArnLike: {
          'aws:SourceArn': sourceArn,
        },
        StringEquals: {
          'aws:SourceAccount': sourceAccount,
        },
      });

      fn.addPermission('S1', { principal: principal });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
        Principal: service,
        SourceAccount: sourceAccount,
        SourceArn: sourceArn,
      });
    });

    test('applies source arn condition if principal has conditions', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);
      const sourceArn = 'arn:aws:sqs:us-east-1:111111111111:MyQueue';
      const service = 'my-service';
      const principal = new iam.PrincipalWithConditions(new iam.ServicePrincipal(service), {
        ArnLike: {
          'aws:SourceArn': sourceArn,
        },
      });

      fn.addPermission('S1', { principal: principal });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
        Principal: service,
        SourceArn: sourceArn,
      });
    });

    test('applies principal org id conditions if the principal has conditions', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);
      const principalOrgId = 'org-xxxxxxxxxx';
      const service = 'my-service';
      const principal = new iam.PrincipalWithConditions(new iam.ServicePrincipal(service), {
        StringEquals: {
          'aws:PrincipalOrgID': principalOrgId,
        },
      });

      fn.addPermission('S1', { principal: principal });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
        Principal: service,
        PrincipalOrgID: principalOrgId,
      });
    });

    test('fails if the principal has conditions that are not supported', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);

      expect(() => fn.addPermission('F1', {
        principal: new iam.PrincipalWithConditions(new iam.ServicePrincipal('my-service'), {
          ArnEquals: {
            'aws:SourceArn': 'source-arn',
          },
        }),
      })).toThrow(/PrincipalWithConditions had unsupported conditions for Lambda permission statement/);
      expect(() => fn.addPermission('F2', {
        principal: new iam.PrincipalWithConditions(new iam.ServicePrincipal('my-service'), {
          StringLike: {
            'aws:SourceAccount': 'source-account',
          },
        }),
      })).toThrow(/PrincipalWithConditions had unsupported conditions for Lambda permission statement/);
      expect(() => fn.addPermission('F3', {
        principal: new iam.PrincipalWithConditions(new iam.ServicePrincipal('my-service'), {
          ArnLike: {
            's3:DataAccessPointArn': 'data-access-point-arn',
          },
        }),
      })).toThrow(/PrincipalWithConditions had unsupported conditions for Lambda permission statement/);
    });

    test('fails if the principal has condition combinations that are not supported', () => {
      const stack = new cdk.Stack();
      const fn = newTestLambda(stack);

      expect(() => fn.addPermission('F2', {
        principal: new iam.PrincipalWithConditions(new iam.ServicePrincipal('my-service'), {
          StringEquals: {
            'aws:SourceAccount': 'source-account',
            'aws:PrincipalOrgID': 'principal-org-id',
          },
          ArnLike: {
            'aws:SourceArn': 'source-arn',
          },
        }),
      })).toThrow(/PrincipalWithConditions had unsupported condition combinations for Lambda permission statement/);
    });

    test('BYORole @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy enabled', () => {
      // GIVEN
      const app = new cdk.App({
        context: {
          [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: true,
        },
      });
      const stack = new cdk.Stack(app);
      const role = new iam.Role(stack, 'SomeRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      });
      role.addToPolicy(new iam.PolicyStatement({ actions: ['confirm:itsthesame'], resources: ['*'] }));

      // WHEN
      const fn = new lambda.Function(stack, 'Function', {
        code: new lambda.InlineCode('test'),
        runtime: lambda.Runtime.PYTHON_3_9,
        handler: 'index.test',
        role,
        initialPolicy: [
          new iam.PolicyStatement({ actions: ['inline:inline'], resources: ['*'] }),
        ],
      });

      fn.addToRolePolicy(new iam.PolicyStatement({ actions: ['explicit:explicit'], resources: ['*'] }));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            { Action: 'confirm:itsthesame', Effect: 'Allow', Resource: '*' },
            { Action: 'inline:inline', Effect: 'Allow', Resource: '*' },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            { Action: 'explicit:explicit', Effect: 'Allow', Resource: '*' },
          ],
        },
      });
    });

    test('BYORole @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy disabled', () => {
      // GIVEN
      const app = new cdk.App({
        context: {
          [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: false,
        },
      });
      const stack = new cdk.Stack(app);
      const role = new iam.Role(stack, 'SomeRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      });
      role.addToPolicy(new iam.PolicyStatement({ actions: ['confirm:itsthesame'], resources: ['*'] }));

      // WHEN
      const fn = new lambda.Function(stack, 'Function', {
        code: new lambda.InlineCode('test'),
        runtime: lambda.Runtime.PYTHON_3_9,
        handler: 'index.test',
        role,
        initialPolicy: [
          new iam.PolicyStatement({ actions: ['inline:inline'], resources: ['*'] }),
        ],
      });

      fn.addToRolePolicy(new iam.PolicyStatement({ actions: ['explicit:explicit'], resources: ['*'] }));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            { Action: 'confirm:itsthesame', Effect: 'Allow', Resource: '*' },
            { Action: 'inline:inline', Effect: 'Allow', Resource: '*' },
            { Action: 'explicit:explicit', Effect: 'Allow', Resource: '*' },
          ],
        },
      });
    });
  });

  test('fromFunctionArn', () => {
    // GIVEN
    const stack2 = new cdk.Stack();

    // WHEN
    const imported = lambda.Function.fromFunctionArn(stack2, 'Imported', 'arn:aws:lambda:us-east-1:123456789012:function:ProcessKinesisRecords');

    // THEN
    expect(imported.functionArn).toEqual('arn:aws:lambda:us-east-1:123456789012:function:ProcessKinesisRecords');
    expect(imported.functionName).toEqual('ProcessKinesisRecords');
  });

  test('Function.fromFunctionName', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const imported = lambda.Function.fromFunctionName(stack, 'Imported', 'my-function');

    // THEN
    expect(stack.resolve(imported.functionArn)).toStrictEqual({
      'Fn::Join': ['', [
        'arn:',
        { Ref: 'AWS::Partition' },
        ':lambda:',
        { Ref: 'AWS::Region' },
        ':',
        { Ref: 'AWS::AccountId' },
        ':function:my-function',
      ]],
    });
    expect(stack.resolve(imported.functionName)).toStrictEqual({
      'Fn::Select': [6, {
        'Fn::Split': [':', {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':lambda:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':function:my-function',
          ]],
        }],
      }],
    });
  });

  describe('Function.fromFunctionAttributes()', () => {
    let stack: cdk.Stack;

    beforeEach(() => {
      const app = new cdk.App();
      stack = new cdk.Stack(app, 'Base', {
        env: { account: '111111111111', region: 'stack-region' },
      });
    });

    describe('for a function in a different account and region', () => {
      let func: lambda.IFunction;

      beforeEach(() => {
        func = lambda.Function.fromFunctionAttributes(stack, 'iFunc', {
          functionArn: 'arn:aws:lambda:function-region:222222222222:function:function-name',
        });
      });

      test("the function's region is taken from the ARN", () => {
        expect(func.env.region).toBe('function-region');
      });

      test("the function's account is taken from the ARN", () => {
        expect(func.env.account).toBe('222222222222');
      });
    });
  });

  describe('addPermissions', () => {
    test('imported Function w/ resolved account and function arn', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'Imports', {
        env: { account: '123456789012', region: 'us-east-1' },
      });

      // WHEN
      const iFunc = lambda.Function.fromFunctionAttributes(stack, 'iFunc', {
        functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:BaseFunction',
      });
      iFunc.addPermission('iFunc', {
        principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Permission', 1);
    });

    test('imported Function w/ unresolved account', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'Imports');

      // WHEN
      const iFunc = lambda.Function.fromFunctionAttributes(stack, 'iFunc', {
        functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:BaseFunction',
      });
      iFunc.addPermission('iFunc', {
        principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Permission', 0);
    });

    test('imported Function w/ unresolved account & allowPermissions set', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'Imports');

      // WHEN
      const iFunc = lambda.Function.fromFunctionAttributes(stack, 'iFunc', {
        functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:BaseFunction',
        sameEnvironment: true, // since this is false, by default, for env agnostic stacks
      });
      iFunc.addPermission('iFunc', {
        principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Permission', 1);
    });

    test('imported Function w/different account', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'Base', {
        env: { account: '111111111111' },
      });

      // WHEN
      const iFunc = lambda.Function.fromFunctionAttributes(stack, 'iFunc', {
        functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:BaseFunction',
      });
      iFunc.addPermission('iFunc', {
        principal: new iam.ServicePrincipal('cloudformation.amazonaws.com'),
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Permission', 0);
    });

    describe('annotations on different IFunctions', () => {
      let stack: cdk.Stack;
      let fn: lambda.Function;
      let warningMessage: string;
      beforeEach(() => {
        warningMessage = 'AWS Lambda has changed their authorization strategy';
        stack = new cdk.Stack();
        fn = new lambda.Function(stack, 'MyLambda', {
          code: lambda.Code.fromAsset(path.join(__dirname, 'my-lambda-handler')),
          handler: 'index.handler',
          runtime: lambda.Runtime.PYTHON_3_9,
        });
      });

      describe('permissions on functions', () => {
        test('without lambda:InvokeFunction', () => {
          // WHEN
          fn.addPermission('MyPermission', {
            action: 'lambda.GetFunction',
            principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
          });

          // Simulate a workflow where a user has created a currentVersion with the intent to invoke it later.
          fn.currentVersion;

          // THEN
          Annotations.fromStack(stack).hasNoWarning('/Default/MyLambda', Match.stringLikeRegexp(warningMessage));
        });

        describe('with lambda:InvokeFunction', () => {
          test('without invoking currentVersion', () => {
            // WHEN
            fn.addPermission('MyPermission', {
              principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
            });

            // THEN
            Annotations.fromStack(stack).hasNoWarning('/Default/MyLambda', Match.stringLikeRegexp(warningMessage));
          });

          test('with currentVersion invoked first', () => {
            // GIVEN
            // Simulate a workflow where a user has created a currentVersion with the intent to invoke it later.
            fn.currentVersion;

            // WHEN
            fn.addPermission('MyPermission', {
              principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
            });

            // THEN
            Annotations.fromStack(stack).hasWarning('/Default/MyLambda', Match.stringLikeRegexp(warningMessage));
          });

          test('with currentVersion invoked after permissions created', () => {
            // WHEN
            fn.addPermission('MyPermission', {
              principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
            });

            // Simulate a workflow where a user has created a currentVersion after adding permissions to the function.
            fn.currentVersion;

            // THEN
            Annotations.fromStack(stack).hasWarning('/Default/MyLambda', Match.stringLikeRegexp(warningMessage));
          });

          test('multiple currentVersion calls does not result in multiple warnings', () => {
            // WHEN
            fn.currentVersion;

            fn.addPermission('MyPermission', {
              principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
            });

            fn.currentVersion;

            // THEN
            const warns = Annotations.fromStack(stack).findWarning('/Default/MyLambda', Match.stringLikeRegexp(warningMessage));
            expect(warns).toHaveLength(1);
          });
        });
      });

      test('permission on versions', () => {
        // GIVEN
        const version = new lambda.Version(stack, 'MyVersion', {
          lambda: fn.currentVersion,
        });

        // WHEN
        version.addPermission('MyPermission', {
          principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // THEN
        Annotations.fromStack(stack).hasNoWarning('/Default/MyVersion', Match.stringLikeRegexp(warningMessage));
      });

      test('permission on latest version', () => {
        // WHEN
        fn.latestVersion.addPermission('MyPermission', {
          principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // THEN
        // cannot add permissions on latest version, so no warning necessary
        Annotations.fromStack(stack).hasNoWarning('/Default/MyLambda/$LATEST', Match.stringLikeRegexp(warningMessage));
      });

      test('function.addAlias', () => {
        // WHEN
        fn.addAlias('prod');

        // THEN
        Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Alias', {
          Name: 'prod',
          FunctionName: { Ref: 'MyLambdaCCE802FB' },
          FunctionVersion: { 'Fn::GetAtt': ['MyLambdaCurrentVersionE7A382CCe2d14849ae02766d3abd365a8a0f12ae', 'Version'] },
        });
      });

      describe('permission on alias', () => {
        test('of current version', () => {
          // GIVEN
          const version = new lambda.Version(stack, 'MyVersion', {
            lambda: fn.currentVersion,
          });
          const alias = new lambda.Alias(stack, 'MyAlias', {
            aliasName: 'alias',
            version,
          });

          // WHEN
          alias.addPermission('MyPermission', {
            principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
          });

          // THEN
          Annotations.fromStack(stack).hasNoWarning('/Default/MyAlias', Match.stringLikeRegexp(warningMessage));
        });

        test('of latest version', () => {
          // GIVEN
          const alias = new lambda.Alias(stack, 'MyAlias', {
            aliasName: 'alias',
            version: fn.latestVersion,
          });

          // WHEN
          alias.addPermission('MyPermission', {
            principal: new iam.ServicePrincipal('lambda.amazonaws.com'),
          });

          // THEN
          Annotations.fromStack(stack).hasNoWarning('/Default/MyAlias', Match.stringLikeRegexp(warningMessage));
        });
      });
    });
  });

  test('Lambda code can be read from a local directory via an asset', () => {
    // GIVEN
    const app = new cdk.App({
      context: {
        [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false,
      },
    });
    const stack = new cdk.Stack(app);
    new lambda.Function(stack, 'MyLambda', {
      code: lambda.Code.fromAsset(path.join(__dirname, 'my-lambda-handler')),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Code: {
        S3Bucket: {
          Ref: 'AssetParameters9678c34eca93259d11f2d714177347afd66c50116e1e08996eff893d3ca81232S3Bucket1354C645',
        },
        S3Key: {
          'Fn::Join': ['', [
            { 'Fn::Select': [0, { 'Fn::Split': ['||', { Ref: 'AssetParameters9678c34eca93259d11f2d714177347afd66c50116e1e08996eff893d3ca81232S3VersionKey5D873FAC' }] }] },
            { 'Fn::Select': [1, { 'Fn::Split': ['||', { Ref: 'AssetParameters9678c34eca93259d11f2d714177347afd66c50116e1e08996eff893d3ca81232S3VersionKey5D873FAC' }] }] },
          ]],
        },
      },
      Handler: 'index.handler',
      Role: {
        'Fn::GetAtt': [
          'MyLambdaServiceRole4539ECB6',
          'Arn',
        ],
      },
      Runtime: 'python3.9',
    });
  });

  test('default function with SQS DLQ when client sets deadLetterQueueEnabled to true and functionName defined by client @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy enabled', () => {
    const app = new cdk.App({
      context: {
        [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: true,
      },
    });
    const stack = new cdk.Stack(app);

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      functionName: 'OneFunctionToRuleThemAll',
      deadLetterQueueEnabled: true,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'lambda.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
      ManagedPolicyArns: [
        {
          'Fn::Join': [
            '',
            [
              'arn:',
              {
                Ref: 'AWS::Partition',
              },
              ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            ],
          ],
        },
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'sqs:SendMessage',
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': [
                'MyLambdaDeadLetterQueue399EEA2D',
                'Arn',
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdainlinePolicyAddedToExecutionRole0E0144580',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
        DeadLetterConfig: {
          TargetArn: {
            'Fn::GetAtt': [
              'MyLambdaDeadLetterQueue399EEA2D',
              'Arn',
            ],
          },
        },
        FunctionName: 'OneFunctionToRuleThemAll',
      },
      DependsOn: [
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('default function with SQS DLQ when client sets deadLetterQueueEnabled to true and functionName defined by client @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy disabled', () => {
    const app = new cdk.App({
      context: {
        [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: false,
      },
    });
    const stack = new cdk.Stack(app);

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      functionName: 'OneFunctionToRuleThemAll',
      deadLetterQueueEnabled: true,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'lambda.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
      ManagedPolicyArns: [
        {
          'Fn::Join': [
            '',
            [
              'arn:',
              {
                Ref: 'AWS::Partition',
              },
              ':iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            ],
          ],
        },
      ],
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'sqs:SendMessage',
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': [
                'MyLambdaDeadLetterQueue399EEA2D',
                'Arn',
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
        DeadLetterConfig: {
          TargetArn: {
            'Fn::GetAtt': [
              'MyLambdaDeadLetterQueue399EEA2D',
              'Arn',
            ],
          },
        },
        FunctionName: 'OneFunctionToRuleThemAll',
      },
      DependsOn: [
        'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('default function with SQS DLQ when client sets deadLetterQueueEnabled to true and functionName not defined by client', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueueEnabled: true,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::SQS::Queue', {
      MessageRetentionPeriod: 1209600,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      DeadLetterConfig: {
        TargetArn: {
          'Fn::GetAtt': [
            'MyLambdaDeadLetterQueue399EEA2D',
            'Arn',
          ],
        },
      },
    });
  });

  test('default function with SQS DLQ when client sets deadLetterQueueEnabled to false', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueueEnabled: false,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Code: {
        ZipFile: 'foo',
      },
      Handler: 'index.handler',
      Role: {
        'Fn::GetAtt': [
          'MyLambdaServiceRole4539ECB6',
          'Arn',
        ],
      },
      Runtime: 'nodejs24.x',
    });
  });

  test('default function with SQS DLQ when client provides Queue to be used as DLQ', () => {
    const stack = new cdk.Stack();

    const dlQueue = new sqs.Queue(stack, 'DeadLetterQueue', {
      queueName: 'MyLambda_DLQ',
      retentionPeriod: cdk.Duration.days(14),
    });

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueue: dlQueue,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'sqs:SendMessage',
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': [
                'DeadLetterQueue9F481546',
                'Arn',
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      DeadLetterConfig: {
        TargetArn: {
          'Fn::GetAtt': [
            'DeadLetterQueue9F481546',
            'Arn',
          ],
        },
      },
    });
  });

  test('default function with SQS DLQ when client provides Queue to be used as DLQ and deadLetterQueueEnabled set to true', () => {
    const stack = new cdk.Stack();

    const dlQueue = new sqs.Queue(stack, 'DeadLetterQueue', {
      queueName: 'MyLambda_DLQ',
      retentionPeriod: cdk.Duration.days(14),
    });

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueueEnabled: true,
      deadLetterQueue: dlQueue,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'sqs:SendMessage',
            Effect: 'Allow',
            Resource: {
              'Fn::GetAtt': [
                'DeadLetterQueue9F481546',
                'Arn',
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      DeadLetterConfig: {
        TargetArn: {
          'Fn::GetAtt': [
            'DeadLetterQueue9F481546',
            'Arn',
          ],
        },
      },
    });
  });

  test('error when default function with SQS DLQ when client provides Queue to be used as DLQ and deadLetterQueueEnabled set to false', () => {
    const stack = new cdk.Stack();

    const dlQueue = new sqs.Queue(stack, 'DeadLetterQueue', {
      queueName: 'MyLambda_DLQ',
      retentionPeriod: cdk.Duration.days(14),
    });

    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueueEnabled: false,
      deadLetterQueue: dlQueue,
    })).toThrow(/deadLetterQueue defined but deadLetterQueueEnabled explicitly set to false/);
  });

  test('default function with SNS DLQ when client provides Topic to be used as DLQ', () => {
    const stack = new cdk.Stack();

    const dlTopic = new sns.Topic(stack, 'DeadLetterTopic');

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterTopic: dlTopic,
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          {
            Action: 'sns:Publish',
            Effect: 'Allow',
            Resource: {
              Ref: 'DeadLetterTopicC237650B',
            },
          },
        ]),
      },
    });
    template.hasResourceProperties('AWS::Lambda::Function', {
      DeadLetterConfig: {
        TargetArn: {
          Ref: 'DeadLetterTopicC237650B',
        },
      },
    });
  });

  test('error when default function with SNS DLQ when client provides Topic to be used as DLQ and deadLetterQueueEnabled set to false', () => {
    const stack = new cdk.Stack();

    const dlTopic = new sns.Topic(stack, 'DeadLetterTopic');

    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueueEnabled: false,
      deadLetterTopic: dlTopic,
    })).toThrow(/deadLetterQueue and deadLetterTopic cannot be specified together at the same time/);
  });

  test('error when default function with SNS DLQ when client provides Topic to be used as DLQ and deadLetterQueueEnabled set to true', () => {
    const stack = new cdk.Stack();

    const dlTopic = new sns.Topic(stack, 'DeadLetterTopic');

    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueueEnabled: true,
      deadLetterTopic: dlTopic,
    })).toThrow(/deadLetterQueue and deadLetterTopic cannot be specified together at the same time/);
  });

  test('error when both topic and queue are presented as DLQ', () => {
    const stack = new cdk.Stack();

    const dlQueue = new sqs.Queue(stack, 'DLQ');
    const dlTopic = new sns.Topic(stack, 'DeadLetterTopic');

    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      deadLetterQueue: dlQueue,
      deadLetterTopic: dlTopic,
    })).toThrow(/deadLetterQueue and deadLetterTopic cannot be specified together at the same time/);
  });

  test('default function with Active tracing @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy enabled', () => {
    const app = new cdk.App({
      context: {
        [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: true,
      },
    });
    const stack = new cdk.Stack(app);

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tracing: lambda.Tracing.ACTIVE,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'xray:PutTraceSegments',
              'xray:PutTelemetryRecords',
            ],
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdainlinePolicyAddedToExecutionRole0E0144580',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
        TracingConfig: {
          Mode: 'Active',
        },
      },
      DependsOn: [
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('default function with Active tracing @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy disabled', () => {
    const app = new cdk.App({
      context: {
        [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: false,
      },
    });
    const stack = new cdk.Stack(app);

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tracing: lambda.Tracing.ACTIVE,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'xray:PutTraceSegments',
              'xray:PutTelemetryRecords',
            ],
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
        TracingConfig: {
          Mode: 'Active',
        },
      },
      DependsOn: [
        'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('default function with PassThrough tracing @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy enabled', () => {
    const app = new cdk.App({
      context: {
        [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: true,
      },
    });
    const stack = new cdk.Stack(app);

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tracing: lambda.Tracing.PASS_THROUGH,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'xray:PutTraceSegments',
              'xray:PutTelemetryRecords',
            ],
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdainlinePolicyAddedToExecutionRole0E0144580',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
        TracingConfig: {
          Mode: 'PassThrough',
        },
      },
      DependsOn: [
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('default function with PassThrough tracing @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy disabled', () => {
    const app = new cdk.App({
      context: {
        [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: false,
      },
    });
    const stack = new cdk.Stack(app);

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tracing: lambda.Tracing.PASS_THROUGH,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'xray:PutTraceSegments',
              'xray:PutTelemetryRecords',
            ],
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
      Roles: [
        {
          Ref: 'MyLambdaServiceRole4539ECB6',
        },
      ],
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
        TracingConfig: {
          Mode: 'PassThrough',
        },
      },
      DependsOn: [
        'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('default function with Disabled tracing', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tracing: lambda.Tracing.DISABLED,
    });

    Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          ZipFile: 'foo',
        },
        Handler: 'index.handler',
        Role: {
          'Fn::GetAtt': [
            'MyLambdaServiceRole4539ECB6',
            'Arn',
          ],
        },
        Runtime: 'nodejs24.x',
      },
      DependsOn: [
        'MyLambdaServiceRole4539ECB6',
      ],
    });
  });

  test('runtime and handler set to FROM_IMAGE are set to undefined in CloudFormation', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: lambda.Code.fromAssetImage(dockerLambdaHandlerPath),
      handler: lambda.Handler.FROM_IMAGE,
      runtime: lambda.Runtime.FROM_IMAGE,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Runtime: Match.absent(),
      Handler: Match.absent(),
      PackageType: 'Image',
    });
  });

  describe('grantInvoke', () => {
    test('adds iam:InvokeFunction', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AccountPrincipal('1234'),
      });
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.grantInvoke(role);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: [
                { 'Fn::GetAtt': ['Function76856677', 'Arn'] },
                { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['Function76856677', 'Arn'] }, ':*']] },
              ],
            },
          ],
        },
      });
    });

    test('adds grantInvokeLatestVersion ', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AccountPrincipal('1234'),
      });
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.grantInvokeLatestVersion(role);

      // THEN function should have allow on both unqualified arn and arn:$LATEST
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: [
                { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['Function76856677', 'Arn'] }, ':$LATEST']] },
                { 'Fn::GetAtt': ['Function76856677', 'Arn'] },
              ],
            },
          ],
        },
      });
    });

    test('adds grantInvokeVersion ', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AccountPrincipal('1234'),
      });
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      const lv2 = new lambda.Version(stack, 'v2', {
        lambda: fn,
      });
      // WHEN
      fn.grantInvokeVersion(role, lv2);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['Function76856677', 'Arn'] }, ':', { 'Fn::GetAtt': ['v248F3DDCC', 'Version'] }]] },
            },
          ],
        },
      });
    });

    test('with a service principal', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      const service = new iam.ServicePrincipal('apigateway.amazonaws.com');

      // WHEN
      fn.grantInvoke(service);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: 'apigateway.amazonaws.com',
      });
    });

    test('with an account principal', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      const account = new iam.AccountPrincipal('123456789012');

      // WHEN
      fn.grantInvoke(account);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: '123456789012',
      });
    });

    test('with an arn principal', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      const account = new iam.ArnPrincipal('arn:aws:iam::123456789012:role/someRole');

      // WHEN
      fn.grantInvoke(account);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: 'arn:aws:iam::123456789012:role/someRole',
      });
    });

    test('with an organization principal', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      const org = new iam.OrganizationPrincipal('o-12345abcde');

      // WHEN
      fn.grantInvoke(org);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: '*',
        PrincipalOrgID: 'o-12345abcde',
      });
    });

    test('can be called twice for the same service principal', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      const service = new iam.ServicePrincipal('elasticloadbalancing.amazonaws.com');

      // WHEN
      fn.grantInvoke(service);
      fn.grantInvoke(service);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: 'elasticloadbalancing.amazonaws.com',
      });
    });

    test('with an imported role (in the same account)', () => {
      // GIVEN
      const stack = new cdk.Stack(undefined, undefined, {
        env: { account: '123456789012' },
      });
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.grantInvoke(iam.Role.fromRoleArn(stack, 'ForeignRole', 'arn:aws:iam::123456789012:role/someRole'));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: [
                { 'Fn::GetAtt': ['Function76856677', 'Arn'] },
                { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['Function76856677', 'Arn'] }, ':*']] },
              ],
            },
          ],
        },
        Roles: ['someRole'],
      });
    });

    test('with an imported role (from a different account)', () => {
      // GIVEN
      const stack = new cdk.Stack(undefined, undefined, {
        env: { account: '3333' },
      });
      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.grantInvoke(iam.Role.fromRoleArn(stack, 'ForeignRole', 'arn:aws:iam::123456789012:role/someRole'));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: 'arn:aws:iam::123456789012:role/someRole',
      });
    });

    test('on an imported function (same account)', () => {
      // GIVEN
      const stack = new cdk.Stack(undefined, undefined, {
        env: { account: '123456789012' },
      });
      const fn = lambda.Function.fromFunctionArn(stack, 'Function', 'arn:aws:lambda:us-east-1:123456789012:function:MyFn');

      // WHEN
      fn.grantInvoke(new iam.ServicePrincipal('elasticloadbalancing.amazonaws.com'));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: 'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
        Principal: 'elasticloadbalancing.amazonaws.com',
      });
    });

    test('on an imported function (unresolved account)', () => {
      const stack = new cdk.Stack();
      const fn = lambda.Function.fromFunctionArn(stack, 'Function', 'arn:aws:lambda:us-east-1:123456789012:function:MyFn');

      expect(
        () => fn.grantInvoke(new iam.ServicePrincipal('elasticloadbalancing.amazonaws.com')),
      ).toThrow(/Cannot modify permission to lambda function/);
    });

    test('on an imported function (unresolved account & w/ allowPermissions)', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = lambda.Function.fromFunctionAttributes(stack, 'Function', {
        functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
        sameEnvironment: true,
      });

      // WHEN
      fn.grantInvoke(new iam.ServicePrincipal('elasticloadbalancing.amazonaws.com'));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: 'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
        Principal: 'elasticloadbalancing.amazonaws.com',
      });
    });

    test('on an imported function (different account)', () => {
      // GIVEN
      const stack = new cdk.Stack(undefined, undefined, {
        env: { account: '111111111111' }, // Different account
      });
      const fn = lambda.Function.fromFunctionArn(stack, 'Function', 'arn:aws:lambda:us-east-1:123456789012:function:MyFn');

      // THEN
      expect(() => {
        fn.grantInvoke(new iam.ServicePrincipal('elasticloadbalancing.amazonaws.com'));
      }).toThrow(/Cannot modify permission to lambda function/);
    });

    test('on an imported function (different account & w/ skipPermissions', () => {
      // GIVEN
      const stack = new cdk.Stack(undefined, undefined, {
        env: { account: '111111111111' }, // Different account
      });
      const fn = lambda.Function.fromFunctionAttributes(stack, 'Function', {
        functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
        skipPermissions: true,
      });

      // THEN
      expect(() => {
        fn.grantInvoke(new iam.ServicePrincipal('elasticloadbalancing.amazonaws.com'));
      }).not.toThrow();
    });
  });

  describe('grantInvokeCompositePrincipal', () => {
    test('adds iam:InvokeFunction for a CompositePrincipal (two accounts)', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const compositePrincipal = new iam.CompositePrincipal(
        new iam.AccountPrincipal('1234'),
        new iam.AccountPrincipal('5678'),
      );

      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.grantInvokeCompositePrincipal(compositePrincipal);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: '1234',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: '5678',
      });
    });

    test('adds iam:InvokeFunction for a CompositePrincipal (multiple types)', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const compositePrincipal = new iam.CompositePrincipal(
        new iam.AccountPrincipal('1234'),
        new iam.ServicePrincipal('apigateway.amazonaws.com'),
        new iam.ArnPrincipal('arn:aws:iam::123456789012:role/someRole'),
        new iam.OrganizationPrincipal('o-12345abcde'),
      );

      const fn = new lambda.Function(stack, 'Function', {
        code: lambda.Code.fromInline('xxx'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.grantInvokeCompositePrincipal(compositePrincipal);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: '1234',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: 'apigateway.amazonaws.com',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: 'arn:aws:iam::123456789012:role/someRole',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Permission', {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: '*',
        PrincipalOrgID: 'o-12345abcde',
      });
    });
  });

  test('Can use metricErrors on a lambda Function', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Function', {
      code: lambda.Code.fromInline('xxx'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    // THEN
    expect(stack.resolve(fn.metricErrors())).toEqual({
      dimensions: { FunctionName: { Ref: 'Function76856677' } },
      namespace: 'AWS/Lambda',
      metricName: 'Errors',
      period: cdk.Duration.minutes(5),
      statistic: 'Sum',
    });
  });

  test('addEventSource calls bind', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Function', {
      code: lambda.Code.fromInline('xxx'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    let bindTarget;

    class EventSourceMock implements lambda.IEventSource {
      public bind(target: lambda.IFunction) {
        bindTarget = target;
      }
    }

    // WHEN
    fn.addEventSource(new EventSourceMock());

    // THEN
    expect(bindTarget).toEqual(fn);
  });

  test('multi-tenant function accepts ApiEventSource', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Function', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',
      code: lambda.Code.fromInline('exports.handler = async () => {}'),
      tenancyConfig: lambda.TenancyConfig.PER_TENANT,
    });

    let bindCalled = false;
    class MockApiEventSource implements lambda.IEventSource {
      bind(_target: lambda.IFunction): void {
        bindCalled = true;
      }
    }
    Object.defineProperty(MockApiEventSource, 'name', { value: 'ApiEventSource' });

    // WHEN & THEN
    expect(() => {
      fn.addEventSource(new MockApiEventSource());
    }).not.toThrow();
    expect(bindCalled).toBe(true);
  });

  test('multi-tenant function rejects non-API event sources', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Function', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',
      code: lambda.Code.fromInline('exports.handler = async () => {}'),
      tenancyConfig: lambda.TenancyConfig.PER_TENANT,
    });

    class MockSqsEventSource implements lambda.IEventSource {
      bind(_target: lambda.IFunction): void {}
    }
    Object.defineProperty(MockSqsEventSource, 'name', { value: 'SqsEventSource' });

    // WHEN & THEN
    expect(() => {
      fn.addEventSource(new MockSqsEventSource());
    }).toThrow('Event source SqsEventSource is not supported for functions with tenant isolation mode');
  });

  test('layer is baked into the function version', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'TestStack');
    const bucket = new s3.Bucket(stack, 'Bucket');
    const code = new lambda.S3Code(bucket, 'ObjectKey');

    const fn = new lambda.Function(stack, 'fn', {
      runtime: lambda.Runtime.NODEJS_22_X,
      code: lambda.Code.fromInline('exports.main = function() { console.log("DONE"); }'),
      handler: 'index.main',
    });

    const fnHash = calculateFunctionHash(fn);

    // WHEN
    const layer = new lambda.LayerVersion(stack, 'LayerVersion', {
      code,
      compatibleRuntimes: [lambda.Runtime.NODEJS_22_X],
    });

    fn.addLayers(layer);

    const newFnHash = calculateFunctionHash(fn);

    expect(fnHash).not.toEqual(newFnHash);
  });

  test('with feature flag, layer version is baked into function version', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.LAMBDA_RECOGNIZE_LAYER_VERSION]: true } });
    const stack = new cdk.Stack(app, 'TestStack');
    const bucket = new s3.Bucket(stack, 'Bucket');
    const code = new lambda.S3Code(bucket, 'ObjectKey');
    const layer = new lambda.LayerVersion(stack, 'LayerVersion', {
      code,
      compatibleRuntimes: [lambda.Runtime.NODEJS_22_X],
    });

    // function with layer
    const fn = new lambda.Function(stack, 'fn', {
      runtime: lambda.Runtime.NODEJS_22_X,
      code: lambda.Code.fromInline('exports.main = function() { console.log("DONE"); }'),
      handler: 'index.main',
      layers: [layer],
    });

    const fnHash = calculateFunctionHash(fn);

    // use escape hatch to change the content of the layer
    // this simulates updating the layer code which changes the version.
    const cfnLayer = layer.node.defaultChild as lambda.CfnLayerVersion;
    const newCode = (new lambda.S3Code(bucket, 'NewObjectKey')).bind(layer);
    cfnLayer.content = {
      s3Bucket: newCode.s3Location!.bucketName,
      s3Key: newCode.s3Location!.objectKey,
      s3ObjectVersion: newCode.s3Location!.objectVersion,
    };

    const newFnHash = calculateFunctionHash(fn);

    expect(fnHash).not.toEqual(newFnHash);
  });

  test('using an incompatible layer', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'TestStack');
    const layer = lambda.LayerVersion.fromLayerVersionAttributes(stack, 'TestLayer', {
      layerVersionArn: 'arn:aws:...',
      compatibleRuntimes: [new lambda.Runtime('something2')],
    });

    // THEN
    expect(() => new lambda.Function(stack, 'Function', {
      layers: [layer],
      runtime: new lambda.Runtime('something1', lambda.RuntimeFamily.NODEJS, {
        supportsInlineCode: true,
      }),
      code: lambda.Code.fromInline('exports.main = function() { console.log("DONE"); }'),
      handler: 'index.main',
    })).toThrow(/something1 is not in \[something2\]/);
  });

  test('using more than 5 layers', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'TestStack');
    const layers = new Array(6).fill(lambda.LayerVersion.fromLayerVersionAttributes(stack, 'TestLayer', {
      layerVersionArn: 'arn:aws:...',
      compatibleRuntimes: [lambda.Runtime.NODEJS_22_X],
    }));

    // THEN
    expect(() => new lambda.Function(stack, 'Function', {
      layers,
      runtime: lambda.Runtime.NODEJS_22_X,
      code: lambda.Code.fromInline('exports.main = function() { console.log("DONE"); }'),
      handler: 'index.main',
    })).toThrow(/Unable to add layer:/);
  });

  test('environment variables work in China', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, undefined, { env: { region: 'cn-north-1' } });

    // WHEN
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      environment: {
        SOME: 'Variable',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Environment: {
        Variables: {
          SOME: 'Variable',
        },
      },
    });
  });

  test('environment variables work in an unspecified region', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      environment: {
        SOME: 'Variable',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Environment: {
        Variables: {
          SOME: 'Variable',
        },
      },
    });
  });

  test('support reserved concurrent executions', () => {
    const stack = new cdk.Stack();

    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      reservedConcurrentExecutions: 10,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      ReservedConcurrentExecutions: 10,
    });
  });

  test('its possible to specify event sources upon creation', () => {
    // GIVEN
    const stack = new cdk.Stack();

    let bindCount = 0;

    class EventSource implements lambda.IEventSource {
      public bind(_fn: lambda.IFunction): void {
        bindCount++;
      }
    }

    // WHEN
    new lambda.Function(stack, 'fn', {
      code: lambda.Code.fromInline('boom'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.bam',
      events: [
        new EventSource(),
        new EventSource(),
      ],
    });

    // THEN
    expect(bindCount).toEqual(2);
  });

  test('Provided Runtime returns the right values', () => {
    const rt = lambda.Runtime.PROVIDED;

    expect(rt.name).toEqual('provided');
    expect(rt.supportsInlineCode).toEqual(false);
  });

  test('specify log retention', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      logRetention: logs.RetentionDays.ONE_MONTH,
      logRemovalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetention', {
      LogGroupName: {
        'Fn::Join': [
          '',
          [
            '/aws/lambda/',
            {
              Ref: 'MyLambdaCCE802FB',
            },
          ],
        ],
      },
      RetentionInDays: 30,
      RemovalPolicy: 'destroy',
    });
  });

  test('cannot use logRemovalPolicy and logGroup', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN/THEN
    expect(() => new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      logGroup: new logs.LogGroup(stack, 'CustomLogGroup'),
      logRemovalPolicy: cdk.RemovalPolicy.DESTROY,
    })).toThrow(/Cannot use `logRemovalPolicy` and `logGroup`/);
  });

  test('cannot use logRemovalPolicy and USE_CDK_MANAGED_LAMBDA_LOGGROUP', () => {
    // GIVEN
    const app = new cdk.App({ context: { [cxapi.USE_CDK_MANAGED_LAMBDA_LOGGROUP]: true } });
    const stack = new cdk.Stack(app, 'Stack');

    // WHEN/THEN
    expect(() => new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      logRemovalPolicy: cdk.RemovalPolicy.DESTROY,
    })).toThrow(/Cannot use `logRemovalPolicy` and `@aws-cdk\/aws-lambda:useCdkManagedLogGroup`/);
  });

  test('imported lambda with imported security group and allowAllOutbound set to false', () => {
    // GIVEN
    const stack = new cdk.Stack();

    const fn = lambda.Function.fromFunctionAttributes(stack, 'fn', {
      functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
      securityGroup: ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG', 'sg-123456789', {
        allowAllOutbound: false,
      }),
    });

    // WHEN
    fn.connections.allowToAnyIpv4(ec2.Port.tcp(443));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
      GroupId: 'sg-123456789',
    });
  });

  test('with event invoke config', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      onFailure: {
        bind: () => ({ destination: 'on-failure-arn' }),
      },
      onSuccess: {
        bind: () => ({ destination: 'on-success-arn' }),
      },
      maxEventAge: cdk.Duration.hours(1),
      retryAttempts: 0,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: {
        Ref: 'fn5FF616E3',
      },
      Qualifier: '$LATEST',
      DestinationConfig: {
        OnFailure: {
          Destination: 'on-failure-arn',
        },
        OnSuccess: {
          Destination: 'on-success-arn',
        },
      },
      MaximumEventAgeInSeconds: 3600,
      MaximumRetryAttempts: 0,
    });
  });

  test('event invoke config accepts tokens for maxEventAge and retryAttempts', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const maxEventAge = new cdk.CfnParameter(stack, 'MaxEventAge', { type: 'Number' });
    const retryAttempts = new cdk.CfnParameter(stack, 'RetryAttempts', { type: 'Number' });

    // WHEN
    new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      maxEventAge: cdk.Duration.seconds(maxEventAge.valueAsNumber),
      retryAttempts: retryAttempts.valueAsNumber,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: {
        Ref: 'fn5FF616E3',
      },
      Qualifier: '$LATEST',
      MaximumEventAgeInSeconds: { Ref: 'MaxEventAge' },
      MaximumRetryAttempts: { Ref: 'RetryAttempts' },
    });
  });

  test('throws when calling configureAsyncInvoke on already configured function', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      maxEventAge: cdk.Duration.hours(1),
    });

    // THEN
    expect(() => fn.configureAsyncInvoke({ retryAttempts: 0 })).toThrow(/An EventInvokeConfig has already been configured/);
  });

  test('event invoke config on imported lambda', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = lambda.Function.fromFunctionAttributes(stack, 'fn', {
      functionArn: 'arn:aws:lambda:us-east-1:123456789012:function:my-function',
    });

    // WHEN
    fn.configureAsyncInvoke({
      retryAttempts: 1,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: 'my-function',
      Qualifier: '$LATEST',
      MaximumRetryAttempts: 1,
    });
  });

  testDeprecated('add a version with event invoke config', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    // WHEN
    fn.addVersion('1', 'sha256', 'desc', undefined, {
      retryAttempts: 0,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::EventInvokeConfig', {
      FunctionName: {
        Ref: 'fn5FF616E3',
      },
      Qualifier: {
        'Fn::GetAtt': [
          'fnVersion197FA813F',
          'Version',
        ],
      },
      MaximumRetryAttempts: 0,
    });
  });

  test('check edge compatibility with env vars that can be removed', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    fn.addEnvironment('KEY', 'value', { removeInEdge: true });

    // WHEN
    fn._checkEdgeCompatibility();

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Environment: Match.absent(),
    });
  });

  test('check edge compatibility with env vars that cannot be removed', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      environment: {
        KEY: 'value',
      },
    });
    fn.addEnvironment('OTHER_KEY', 'other_value', { removeInEdge: true });

    // THEN
    expect(() => fn._checkEdgeCompatibility()).toThrow(/The function Default\/fn contains environment variables \[KEY\] and is not compatible with Lambda@Edge/);
  });

  test('add incompatible layer', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'TestStack');
    const bucket = new s3.Bucket(stack, 'Bucket');
    const code = new lambda.S3Code(bucket, 'ObjectKey');

    const func = new lambda.Function(stack, 'myFunc', {
      runtime: lambda.Runtime.PYTHON_3_7,
      handler: 'index.handler',
      code,
    });
    const layer = new lambda.LayerVersion(stack, 'myLayer', {
      code,
      compatibleRuntimes: [lambda.Runtime.NODEJS],
    });

    // THEN
    expect(() => func.addLayers(layer)).toThrow(
      /This lambda function uses a runtime that is incompatible with this layer/);
  });

  test('add compatible layer', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'TestStack');
    const bucket = new s3.Bucket(stack, 'Bucket');
    const code = new lambda.S3Code(bucket, 'ObjectKey');

    const func = new lambda.Function(stack, 'myFunc', {
      runtime: lambda.Runtime.PYTHON_3_7,
      handler: 'index.handler',
      code,
    });
    const layer = new lambda.LayerVersion(stack, 'myLayer', {
      code,
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_7],
    });

    // THEN
    // should not throw
    expect(() => func.addLayers(layer)).not.toThrow();
  });

  test('add compatible layer for deep clone', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'TestStack');
    const bucket = new s3.Bucket(stack, 'Bucket');
    const code = new lambda.S3Code(bucket, 'ObjectKey');

    const runtime = lambda.Runtime.PYTHON_3_7;
    const func = new lambda.Function(stack, 'myFunc', {
      runtime,
      handler: 'index.handler',
      code,
    });
    const clone = _.cloneDeep(runtime);
    const layer = new lambda.LayerVersion(stack, 'myLayer', {
      code,
      compatibleRuntimes: [clone],
    });

    // THEN
    // should not throw
    expect(() => func.addLayers(layer)).not.toThrow();
  });

  test('empty inline code is not allowed', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN/THEN
    expect(() => new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline(''),
    })).toThrow(/Lambda inline code cannot be empty/);
  });

  test('logGroup is correctly returned', () => {
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
    });
    const logGroup = fn.logGroup;
    expect(logGroup.logGroupName).toBeDefined();
    expect(logGroup.logGroupArn).toBeDefined();
  });

  test('dlq is returned when provided by user and is Queue', () => {
    const stack = new cdk.Stack();

    const dlQueue = new sqs.Queue(stack, 'DeadLetterQueue', {
      queueName: 'MyLambda_DLQ',
      retentionPeriod: cdk.Duration.days(14),
    });

    const fn = new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
      deadLetterQueue: dlQueue,
    });
    const deadLetterQueue = fn.deadLetterQueue;
    const deadLetterTopic = fn.deadLetterTopic;

    expect(deadLetterTopic).toBeUndefined();

    expect(deadLetterQueue).toBeDefined();
    expect(deadLetterQueue).toBeInstanceOf(sqs.Queue);
  });

  test('dlq is returned when provided by user and is Topic', () => {
    const stack = new cdk.Stack();

    const dlTopic = new sns.Topic(stack, 'DeadLetterQueue', {
      topicName: 'MyLambda_DLQ',
    });

    const fn = new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
      deadLetterTopic: dlTopic,
    });
    const deadLetterQueue = fn.deadLetterQueue;
    const deadLetterTopic = fn.deadLetterTopic;

    expect(deadLetterQueue).toBeUndefined();

    expect(deadLetterTopic).toBeDefined();
    expect(deadLetterTopic).toBeInstanceOf(sns.Topic);
  });

  test('dlq is returned when setup by cdk and is Queue', () => {
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
      deadLetterQueueEnabled: true,
    });
    const deadLetterQueue = fn.deadLetterQueue;
    const deadLetterTopic = fn.deadLetterTopic;

    expect(deadLetterTopic).toBeUndefined();

    expect(deadLetterQueue).toBeDefined();
    expect(deadLetterQueue).toBeInstanceOf(sqs.Queue);
  });

  test('dlq is undefined when not setup', () => {
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
    });
    const deadLetterQueue = fn.deadLetterQueue;
    const deadLetterTopic = fn.deadLetterTopic;

    expect(deadLetterQueue).toBeUndefined();
    expect(deadLetterTopic).toBeUndefined();
  });

  test('one and only one child LogRetention construct will be created', () => {
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
      logRetention: logs.RetentionDays.FIVE_DAYS,
    });

    // Call logGroup a few times. If more than one instance of LogRetention was created,
    // the second call will fail on duplicate constructs.
    fn.logGroup;
    fn.logGroup;
    fn.logGroup;
  });

  test('fails when inline code is specified on an incompatible runtime', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'fn', {
      handler: 'foo',
      runtime: lambda.Runtime.PROVIDED,
      code: lambda.Code.fromInline('foo'),
    })).toThrow(/Inline source not allowed for/);
  });

  test('multiple calls to latestVersion returns the same version', () => {
    const stack = new cdk.Stack();

    const fn = new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('hello()'),
      handler: 'index.hello',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    const version1 = fn.latestVersion;
    const version2 = fn.latestVersion;

    const expectedArn = {
      'Fn::Join': ['', [
        { 'Fn::GetAtt': ['MyLambdaCCE802FB', 'Arn'] },
        ':$LATEST',
      ]],
    };
    expect(version1).toEqual(version2);
    expect(stack.resolve(version1.functionArn)).toEqual(expectedArn);
    expect(stack.resolve(version2.functionArn)).toEqual(expectedArn);
  });

  test('latestVersion functionRef ARN is the version ARN, not the plain ARN', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const fn = new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('hello()'),
      handler: 'index.hello',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    // THEN
    expect(fn.latestVersion.functionRef.functionArn).toEqual(fn.latestVersion.functionArn);
  });

  test('default function with kmsKeyArn, environmentEncryption passed as props', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const key: kms.IKey = new kms.Key(stack, 'EnvVarEncryptKey', {
      description: 'sample key',
    });

    // WHEN
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      environment: {
        SOME: 'Variable',
      },
      environmentEncryption: key,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Environment: {
        Variables: {
          SOME: 'Variable',
        },
      },
      KmsKeyArn: {
        'Fn::GetAtt': [
          'EnvVarEncryptKey1A7CABDB',
          'Arn',
        ],
      },
    });
  });

  describe('profiling group', () => {
    test('default function with CDK created Profiling Group', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_9,
        profiling: true,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodeGuruProfiler::ProfilingGroup', {
        ProfilingGroupName: 'MyLambdaProfilingGroupC5B6CCD8',
        ComputePlatform: 'AWSLambda',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'codeguru-profiler:ConfigureAgent',
                'codeguru-profiler:PostAgentProfile',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['MyLambdaProfilingGroupEC6DE32F', 'Arn'],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        Roles: [
          {
            Ref: 'MyLambdaServiceRole4539ECB6',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Environment: {
          Variables: {
            AWS_CODEGURU_PROFILER_GROUP_NAME: { Ref: 'MyLambdaProfilingGroupEC6DE32F' },
            AWS_CODEGURU_PROFILER_GROUP_ARN: { 'Fn::GetAtt': ['MyLambdaProfilingGroupEC6DE32F', 'Arn'] },
            AWS_CODEGURU_PROFILER_TARGET_REGION: { Ref: 'AWS::Region' },
            AWS_CODEGURU_PROFILER_ENABLED: 'TRUE',
          },
        },
      });
    });

    test('default function with client provided Profiling Group', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_9,
        profilingGroup: new ProfilingGroup(stack, 'ProfilingGroup'),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'codeguru-profiler:ConfigureAgent',
                'codeguru-profiler:PostAgentProfile',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['ProfilingGroup26979FD7', 'Arn'],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        Roles: [
          {
            Ref: 'MyLambdaServiceRole4539ECB6',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Environment: {
          Variables: {
            AWS_CODEGURU_PROFILER_GROUP_NAME: { Ref: 'ProfilingGroup26979FD7' },
            AWS_CODEGURU_PROFILER_GROUP_ARN: {
              'Fn::GetAtt': [
                'ProfilingGroup26979FD7',
                'Arn',
              ],
            },
            AWS_CODEGURU_PROFILER_TARGET_REGION: { Ref: 'AWS::Region' },
            AWS_CODEGURU_PROFILER_ENABLED: 'TRUE',
          },
        },
      });
    });

    test('default function with client imported Profiling Group', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_9,
        profilingGroup: ProfilingGroup.fromProfilingGroupArn(stack, 'ProfilingGroup', 'arn:aws:codeguru-profiler:us-east-1:1234567890:profilingGroup/MyAwesomeProfilingGroup'),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'codeguru-profiler:ConfigureAgent',
                'codeguru-profiler:PostAgentProfile',
              ],
              Effect: 'Allow',
              Resource: 'arn:aws:codeguru-profiler:us-east-1:1234567890:profilingGroup/MyAwesomeProfilingGroup',
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        Roles: [
          {
            Ref: 'MyLambdaServiceRole4539ECB6',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Environment: {
          Variables: {
            AWS_CODEGURU_PROFILER_GROUP_NAME: 'MyAwesomeProfilingGroup',
            AWS_CODEGURU_PROFILER_GROUP_ARN: 'arn:aws:codeguru-profiler:us-east-1:1234567890:profilingGroup/MyAwesomeProfilingGroup',
            AWS_CODEGURU_PROFILER_TARGET_REGION: 'us-east-1',
          },
        },
      });
    });

    test('default function with client provided Profiling Group but profiling set to false', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_9,
        profiling: false,
        profilingGroup: new ProfilingGroup(stack, 'ProfilingGroup'),
      });

      Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', Match.not({
        Environment: {
          Variables: {
            AWS_CODEGURU_PROFILER_GROUP_ARN: {
              'Fn::Join': [
                '',
                [
                  'arn:', { Ref: 'AWS::Partition' }, ':codeguru-profiler:', { Ref: 'AWS::Region' },
                  ':', { Ref: 'AWS::AccountId' }, ':profilingGroup/', { Ref: 'ProfilingGroup26979FD7' },
                ],
              ],
            },
            AWS_CODEGURU_PROFILER_ENABLED: 'TRUE',
          },
        },
      }));
    });

    test('default function with profiling enabled and client provided env vars', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_9,
        profiling: true,
        environment: {
          AWS_CODEGURU_PROFILER_GROUP_NAME: 'profiler_group',
          AWS_CODEGURU_PROFILER_GROUP_ARN: 'profiler_group_arn',
          AWS_CODEGURU_PROFILER_TARGET_REGION: 'us-east-1',
          AWS_CODEGURU_PROFILER_ENABLED: 'yes',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'codeguru-profiler:ConfigureAgent',
                'codeguru-profiler:PostAgentProfile',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'MyLambdaProfilingGroupEC6DE32F',
                  'Arn',
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        Roles: [
          {
            Ref: 'MyLambdaServiceRole4539ECB6',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Environment: {
          Variables: {
            AWS_CODEGURU_PROFILER_GROUP_NAME: 'profiler_group',
            AWS_CODEGURU_PROFILER_GROUP_ARN: 'profiler_group_arn',
            AWS_CODEGURU_PROFILER_TARGET_REGION: 'us-east-1',
            AWS_CODEGURU_PROFILER_ENABLED: 'yes',
          },
        },
      });

      Annotations.fromStack(stack).hasWarning('/Default/MyLambda', Match.stringLikeRegexp('AWS_CODEGURU_PROFILER_GROUP_NAME, AWS_CODEGURU_PROFILER_GROUP_ARN, AWS_CODEGURU_PROFILER_TARGET_REGION, and AWS_CODEGURU_PROFILER_ENABLED should not be set when profiling options enabled'));
    });

    test('default function with client provided Profiling Group and client provided env vars', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_9,
        profilingGroup: new ProfilingGroup(stack, 'ProfilingGroup'),
        environment: {
          AWS_CODEGURU_PROFILER_GROUP_NAME: 'profiler_group',
          AWS_CODEGURU_PROFILER_GROUP_ARN: 'profiler_group_arn',
          AWS_CODEGURU_PROFILER_TARGET_REGION: 'us-east-1',
          AWS_CODEGURU_PROFILER_ENABLED: 'yes',
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'codeguru-profiler:ConfigureAgent',
                'codeguru-profiler:PostAgentProfile',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'ProfilingGroup26979FD7',
                  'Arn',
                ],
              },
            },
          ],
          Version: '2012-10-17',
        },
        PolicyName: 'MyLambdaServiceRoleDefaultPolicy5BBC6F68',
        Roles: [
          {
            Ref: 'MyLambdaServiceRole4539ECB6',
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Environment: {
          Variables: {
            AWS_CODEGURU_PROFILER_GROUP_NAME: 'profiler_group',
            AWS_CODEGURU_PROFILER_GROUP_ARN: 'profiler_group_arn',
            AWS_CODEGURU_PROFILER_TARGET_REGION: 'us-east-1',
            AWS_CODEGURU_PROFILER_ENABLED: 'yes',
          },
        },
      });

      Annotations.fromStack(stack).hasWarning('/Default/MyLambda', Match.stringLikeRegexp('AWS_CODEGURU_PROFILER_GROUP_NAME, AWS_CODEGURU_PROFILER_GROUP_ARN, AWS_CODEGURU_PROFILER_TARGET_REGION, and AWS_CODEGURU_PROFILER_ENABLED should not be set when profiling options enabled'));
    });

    test('throws an error when used with an unsupported runtime', () => {
      const stack = new cdk.Stack();
      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        profilingGroup: new ProfilingGroup(stack, 'ProfilingGroup'),
        environment: {
          AWS_CODEGURU_PROFILER_GROUP_ARN: 'profiler_group_arn',
          AWS_CODEGURU_PROFILER_ENABLED: 'yes',
        },
      })).toThrow(/not supported by runtime/);
    });
  });

  describe('lambda.Function timeout', () => {
    test('should be a cdk.Duration when defined', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const { timeout } = new lambda.Function(stack, 'MyFunction', {
        handler: 'foo',
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        timeout: cdk.Duration.minutes(2),
      });

      // THEN
      expect(timeout).toEqual(cdk.Duration.minutes(2));
    });

    test('should be optional', () => {
      // GIVEN
      const stack = new cdk.Stack();

      // WHEN
      const { timeout } = new lambda.Function(stack, 'MyFunction', {
        handler: 'foo',
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
      });

      // THEN
      expect(timeout).not.toBeDefined();
    });
  });

  describe('currentVersion', () => {
    // see test.function-hash.ts for more coverage for this
    test('logical id of version is based on the function hash', () => {
      // GIVEN
      const stack1 = new cdk.Stack();
      const fn1 = new lambda.Function(stack1, 'MyFunction', {
        handler: 'foo',
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        environment: {
          FOO: 'bar',
        },
      });
      const stack2 = new cdk.Stack();
      const fn2 = new lambda.Function(stack2, 'MyFunction', {
        handler: 'foo',
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        environment: {
          FOO: 'bear',
        },
      });

      // WHEN
      new cdk.CfnOutput(stack1, 'CurrentVersionArn', {
        value: fn1.currentVersion.functionArn,
      });
      new cdk.CfnOutput(stack2, 'CurrentVersionArn', {
        value: fn2.currentVersion.functionArn,
      });

      // THEN
      const template1 = Template.fromStack(stack1).toJSON();
      const template2 = Template.fromStack(stack2).toJSON();

      // these functions are different in their configuration but the original
      // logical ID of the version would be the same unless the logical ID
      // includes the hash of function's configuration.
      expect(template1.Outputs.CurrentVersionArn.Value).not.toEqual(template2.Outputs.CurrentVersionArn.Value);
    });
  });

  describe('filesystem', () => {
    test('mount efs filesystem', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');
      // WHEN
      new lambda.Function(stack, 'MyFunction', {
        vpc,
        handler: 'foo',
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        FileSystemConfigs: [
          {
            Arn: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':elasticfilesystem:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':access-point/',
                  {
                    Ref: 'EfsAccessPointE419FED9',
                  },
                ],
              ],
            },
            LocalMountPath: '/mnt/msg',
          },
        ],
      });
    });

    test('throw error mounting efs with no vpc', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      // THEN
      expect(() => {
        new lambda.Function(stack, 'MyFunction', {
          handler: 'foo',
          runtime: lambda.Runtime.NODEJS_LATEST,
          code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
          filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
        });
      }).toThrow();
    });

    test('verify deps when mounting efs', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });
      const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
        vpc,
        allowAllOutbound: false,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');
      // WHEN
      new lambda.Function(stack, 'MyFunction', {
        vpc,
        handler: 'foo',
        securityGroups: [securityGroup],
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
      });

      // THEN
      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        DependsOn: [
          'EfsEfsMountTarget195B2DD2E',
          'EfsEfsMountTarget2315C927F',
          'EfsEfsSecurityGroupfromLambdaSG20491B2F751D',
          'LambdaSGtoEfsEfsSecurityGroupFCE2954020499719694A',
          'MyFunctionServiceRoleDefaultPolicyB705ABD4',
          'MyFunctionServiceRole3C357FF2',
          'VpcPrivateSubnet1DefaultRouteBE02A9ED',
          'VpcPrivateSubnet1RouteTableAssociation70C59FA6',
          'VpcPrivateSubnet2DefaultRoute060D2087',
          'VpcPrivateSubnet2RouteTableAssociationA89CAD56',
        ],
      });
    });

    test('validate localMountPath format when mounting efs', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });
      const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
        vpc,
        allowAllOutbound: false,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      // THEN
      expect(() => {
        new lambda.Function(stack, 'MyFunction', {
          vpc,
          handler: 'foo',
          securityGroups: [securityGroup],
          runtime: lambda.Runtime.NODEJS_LATEST,
          code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
          filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/not-mnt/foo-bar'),
        });
      }).toThrow('Local mount path should match with ^/mnt/[a-zA-Z0-9-_.]+$ but given /not-mnt/foo-bar');
    });

    test('validate localMountPath length when mounting efs', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });
      const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
        vpc,
        allowAllOutbound: false,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      // THEN
      expect(() => {
        new lambda.Function(stack, 'MyFunction', {
          vpc,
          handler: 'foo',
          securityGroups: [securityGroup],
          runtime: lambda.Runtime.NODEJS_LATEST,
          code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
          filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, `/mnt/${'a'.repeat(160)}`),
        });
      }).toThrow('Local mount path can not be longer than 160 characters but has 165 characters');
    });

    test('No error when local mount path is Tokenized and Unresolved', () => {
      // GIVEN
      const realLocalMountPath = '/not-mnt/foo-bar';
      const tokenizedLocalMountPath = cdk.Token.asString(new cdk.Intrinsic(realLocalMountPath));

      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });
      const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
        vpc,
        allowAllOutbound: false,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      // THEN
      expect(() => {
        new lambda.Function(stack, 'MyFunction', {
          vpc,
          handler: 'foo',
          securityGroups: [securityGroup],
          runtime: lambda.Runtime.NODEJS_LATEST,
          code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
          filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, tokenizedLocalMountPath),
        });
      }).not.toThrow();
    });

    test('correct security group is created when deployed in separate stacks', () => {
      const app = new cdk.App();

      // EfsStack
      const efsStack = new cdk.Stack(app, 'EfsStack');
      const vpc = new ec2.Vpc(efsStack, 'Vpc');
      const fs = new efs.FileSystem(efsStack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      // LambdaStack
      const lambdaStack = new cdk.Stack(app, 'LambdaStack');
      new lambda.Function(lambdaStack, 'MyFunction', {
        vpc,
        handler: 'foo',
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
      });

      Template.fromStack(lambdaStack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            IpProtocol: '-1',
          },
        ],
      });

      Template.fromStack(lambdaStack).hasResourceProperties('AWS::EC2::SecurityGroupIngress', {
        FromPort: 2049,
        ToPort: 2049,
        IpProtocol: 'tcp',
      });
    });

    test('mount s3files filesystem', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 3, natGateways: 1 });
      const bucket = new s3.Bucket(stack, 'Bucket');

      const fileSystem = new s3files.CfnFileSystem(stack, 'S3FilesFs', {
        bucket: bucket.bucketArn,
        roleArn: 'arn:aws:iam::123456789012:role/S3FilesRole',
      });

      const sg = new ec2.SecurityGroup(stack, 'MountTargetSG', { vpc });

      new s3files.CfnMountTarget(stack, 'MountTarget0', {
        fileSystemId: fileSystem.attrFileSystemId,
        subnetId: vpc.privateSubnets[0].subnetId,
        securityGroups: [sg.securityGroupId],
      });

      const accessPoint = new s3files.CfnAccessPoint(stack, 'AccessPoint', {
        fileSystemId: fileSystem.attrFileSystemId,
      });

      // WHEN — reflection auto-resolves fileSystem, mountTargets, and securityGroups
      new lambda.Function(stack, 'MyFunction', {
        vpc,
        handler: 'index.handler',
        runtime: lambda.Runtime.PYTHON_3_12,
        code: lambda.Code.fromInline('def handler(event, context): pass'),
        filesystem: lambda.FileSystem.fromS3FilesAccessPoint(accessPoint, '/mnt/data'),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        FileSystemConfigs: [
          {
            Arn: {
              'Fn::GetAtt': ['AccessPoint', 'AccessPointArn'],
            },
            LocalMountPath: '/mnt/data',
          },
        ],
      });

      // Verify IAM policies for s3files
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: Match.arrayWith([
            Match.objectLike({
              Action: 's3files:ClientMount',
              Resource: {
                'Fn::GetAtt': ['AccessPoint', 'AccessPointArn'],
              },
            }),
            Match.objectLike({
              Action: ['s3files:ClientMount', 's3files:ClientWrite'],
              Resource: {
                Ref: 'S3FilesFs',
              },
            }),
          ]),
        },
      });
    });
  });

  describe('code config', () => {
    class MyCode extends lambda.Code {
      public readonly isInline: boolean;
      constructor(private readonly config: lambda.CodeConfig) {
        super();
        this.isInline = 'inlineCode' in config;
      }

      public bind(_scope: constructs.Construct): lambda.CodeConfig {
        return this.config;
      }
    }

    test('only one of inline, s3 or imageConfig are allowed', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.Function(stack, 'Fn1', {
        code: new MyCode({}),
        handler: 'index.handler',
        runtime: lambda.Runtime.GO_1_X,
      })).toThrow(/lambda.Code must specify exactly one of/);

      expect(() => new lambda.Function(stack, 'Fn2', {
        code: new MyCode({
          inlineCode: 'foo',
          image: { imageUri: 'bar' },
        }),
        handler: 'index.handler',
        runtime: lambda.Runtime.GO_1_X,
      })).toThrow(/lambda.Code must specify exactly one of/);

      expect(() => new lambda.Function(stack, 'Fn3', {
        code: new MyCode({
          image: { imageUri: 'baz' },
          s3Location: { bucketName: 's3foo', objectKey: 's3bar' },
        }),
        handler: 'index.handler',
        runtime: lambda.Runtime.GO_1_X,
      })).toThrow(/lambda.Code must specify exactly one of/);

      expect(() => new lambda.Function(stack, 'Fn4', {
        code: new MyCode({ inlineCode: 'baz', s3Location: { bucketName: 's3foo', objectKey: 's3bar' } }),
        handler: 'index.handler',
        runtime: lambda.Runtime.GO_1_X,
      })).toThrow(/lambda.Code must specify exactly one of/);
    });

    test('handler must be FROM_IMAGE when image asset is specified', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.Function(stack, 'Fn1', {
        code: lambda.Code.fromAssetImage(dockerLambdaHandlerPath),
        handler: lambda.Handler.FROM_IMAGE,
        runtime: lambda.Runtime.FROM_IMAGE,
      })).not.toThrow();

      expect(() => new lambda.Function(stack, 'Fn2', {
        code: lambda.Code.fromAssetImage(dockerLambdaHandlerPath),
        handler: 'index.handler',
        runtime: lambda.Runtime.FROM_IMAGE,
      })).toThrow(/handler must be.*FROM_IMAGE/);
    });

    test('runtime must be FROM_IMAGE when image asset is specified', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.Function(stack, 'Fn1', {
        code: lambda.Code.fromAssetImage(dockerLambdaHandlerPath),
        handler: lambda.Handler.FROM_IMAGE,
        runtime: lambda.Runtime.FROM_IMAGE,
      })).not.toThrow();

      expect(() => new lambda.Function(stack, 'Fn2', {
        code: lambda.Code.fromAssetImage(dockerLambdaHandlerPath),
        handler: lambda.Handler.FROM_IMAGE,
        runtime: lambda.Runtime.GO_1_X,
      })).toThrow(/runtime must be.*FROM_IMAGE/);
    });

    test('imageUri is correctly configured', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'Fn1', {
        code: new MyCode({
          image: {
            imageUri: 'ecr image uri',
          },
        }),
        handler: lambda.Handler.FROM_IMAGE,
        runtime: lambda.Runtime.FROM_IMAGE,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        Code: {
          ImageUri: 'ecr image uri',
        },
        ImageConfig: Match.absent(),
      });
    });

    test('imageConfig is correctly configured', () => {
      const stack = new cdk.Stack();

      new lambda.Function(stack, 'Fn1', {
        code: new MyCode({
          image: {
            imageUri: 'ecr image uri',
            cmd: ['cmd', 'param1'],
            entrypoint: ['entrypoint', 'param2'],
            workingDirectory: '/some/path',
          },
        }),
        handler: lambda.Handler.FROM_IMAGE,
        runtime: lambda.Runtime.FROM_IMAGE,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        ImageConfig: {
          Command: ['cmd', 'param1'],
          EntryPoint: ['entrypoint', 'param2'],
          WorkingDirectory: '/some/path',
        },
      });
    });
  });

  describe('code signing config', () => {
    test('default', () => {
      const stack = new cdk.Stack();

      const signingProfile = new signer.SigningProfile(stack, 'SigningProfile', {
        platform: signer.Platform.AWS_LAMBDA_SHA384_ECDSA,
      });

      const codeSigningConfig = new lambda.CodeSigningConfig(stack, 'CodeSigningConfig', {
        signingProfiles: [signingProfile],
      });

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        codeSigningConfig,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        CodeSigningConfigArn: {
          'Fn::GetAtt': [
            'CodeSigningConfigD8D41C10',
            'CodeSigningConfigArn',
          ],
        },
      });
    });
  });

  test('error when layers set in a container function', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'Bucket');
    const code = new lambda.S3Code(bucket, 'ObjectKey');

    const layer = new lambda.LayerVersion(stack, 'Layer', {
      code,
    });

    expect(() => new lambda.DockerImageFunction(stack, 'MyLambda', {
      code: lambda.DockerImageCode.fromImageAsset(dockerLambdaHandlerPath),
      layers: [layer],
    })).toThrow(/Layers are not supported for container image functions/);
  });

  testDeprecated('specified architectures is recognized', () => {
    const stack = new cdk.Stack();
    new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',

      architectures: [lambda.Architecture.ARM_64],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Architectures: ['arm64'],
    });
  });

  test('specified architecture is recognized', () => {
    const stack = new cdk.Stack();
    new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',

      architecture: lambda.Architecture.ARM_64,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Architectures: ['arm64'],
    });
  });

  testDeprecated('both architectures and architecture are not recognized', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',

      architecture: lambda.Architecture.ARM_64,
      architectures: [lambda.Architecture.X86_64],
    })).toThrow(/architecture or architectures must be specified/);
  });

  testDeprecated('Only one architecture allowed', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',

      architectures: [lambda.Architecture.X86_64, lambda.Architecture.ARM_64],
    })).toThrow(/one architecture must be specified/);
  });

  test('Architecture is properly readable from the function', () => {
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',
      architecture: lambda.Architecture.ARM_64,
    });
    expect(fn.architecture?.name).toEqual('arm64');
  });

  test('Error when function name is longer than 64 chars', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',
      functionName: 'a'.repeat(65),
    })).toThrow(/Function name can not be longer than 64 characters/);
  });

  test('Error when function name contains invalid characters', () => {
    const stack = new cdk.Stack();
    [' ', '\n', '\r', '[', ']', '<', '>', '$'].forEach(invalidChar => {
      expect(() => {
        new lambda.Function(stack, `foo${invalidChar}`, {
          code: new lambda.InlineCode('foo'),
          handler: 'index.handler',
          runtime: lambda.Runtime.NODEJS_LATEST,
          functionName: `foo${invalidChar}`,
        });
      }).toThrow(/can contain only letters, numbers, hyphens, or underscores with no spaces./);
    });
  });

  test('No error when function name is Tokenized and Unresolved', () => {
    const stack = new cdk.Stack();
    expect(() => {
      const realFunctionName = 'a'.repeat(141);
      const tokenizedFunctionName = cdk.Token.asString(new cdk.Intrinsic(realFunctionName));

      new lambda.Function(stack, 'foo', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        functionName: tokenizedFunctionName,
      });
    }).not.toThrow();
  });

  test('Error when function description is longer than 256 chars', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('foo'),
      runtime: lambda.Runtime.NODEJS_LATEST,
      handler: 'index.handler',
      description: 'a'.repeat(257),
    })).toThrow(/Function description can not be longer than 256 characters/);
  });

  test('No error when function name is Tokenized and Unresolved', () => {
    const stack = new cdk.Stack();
    expect(() => {
      const realFunctionDescription = 'a'.repeat(257);
      const tokenizedFunctionDescription = cdk.Token.asString(new cdk.Intrinsic(realFunctionDescription));

      new lambda.Function(stack, 'foo', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        description: tokenizedFunctionDescription,
      });
    }).not.toThrow();
  });

  describe('FunctionUrl', () => {
    test('addFunctionUrl creates a function url with default options', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('hello()'),
        handler: 'index.hello',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.addFunctionUrl();

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Url', {
        AuthType: 'AWS_IAM',
        TargetFunctionArn: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
      });
    });

    test('addFunctionUrl creates a function url with all options', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const fn = new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('hello()'),
        handler: 'index.hello',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      // WHEN
      fn.addFunctionUrl({
        authType: lambda.FunctionUrlAuthType.NONE,
        cors: {
          allowCredentials: true,
          allowedOrigins: ['https://example.com'],
          allowedMethods: [lambda.HttpMethod.GET],
          allowedHeaders: ['X-Custom-Header'],
          maxAge: cdk.Duration.seconds(300),
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Url', {
        AuthType: 'NONE',
        TargetFunctionArn: {
          'Fn::GetAtt': [
            'MyLambdaCCE802FB',
            'Arn',
          ],
        },
        Cors: {
          AllowCredentials: true,
          AllowHeaders: [
            'X-Custom-Header',
          ],
          AllowMethods: [
            'GET',
          ],
          AllowOrigins: [
            'https://example.com',
          ],
          MaxAge: 300,
        },
      });
    });

    test('grantInvokeUrl: adds appropriate permissions', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AccountPrincipal('1234'),
      });
      const fn = new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('hello()'),
        handler: 'index.hello',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });
      fn.addFunctionUrl();

      // WHEN
      fn.grantInvokeUrl(role);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 'lambda:InvokeFunctionUrl',
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'MyLambdaCCE802FB',
                  'Arn',
                ],
              },
            },
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'MyLambdaCCE802FB',
                  'Arn',
                ],
              },
              Condition: {
                Bool: {
                  'lambda:InvokedViaFunctionUrl': true,
                },
              },
            },
          ],
        },
      });
    });
  });

  describe('SnapStart', () => {
    test('set SnapStart to desired value', () => {
      const stack = new cdk.Stack();
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2530', reason: 'SnapStart is incomplete, we know' });

      new lambda.CfnFunction(stack, 'MyLambda', {
        code: {
          s3Bucket: 'my-bucket',
          s3Key: 'java11-test-function.zip',
        },
        functionName: 'MyCDK-SnapStart-Function',
        handler: 'example.Handler::handleRequest',
        role: 'testRole',
        runtime: 'java11',
        snapStart: { applyOn: 'PublishedVersions' },
      });

      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties:
        {
          Code: { S3Bucket: 'my-bucket', S3Key: 'java11-test-function.zip' },
          Handler: 'example.Handler::handleRequest',
          Runtime: 'java11',
          SnapStart: {
            ApplyOn: 'PublishedVersions',
          },
        },
      });
    });

    test('function using SnapStart', () => {
      const stack = new cdk.Stack();
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2530', reason: 'SnapStart is incomplete, we know' });

      // WHEN
      new lambda.Function(stack, 'MyLambda', {
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        handler: 'example.Handler::handleRequest',
        runtime: lambda.Runtime.JAVA_11,
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      });

      // THEN
      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties:
        {
          Handler: 'example.Handler::handleRequest',
          Runtime: 'java11',
          SnapStart: {
            ApplyOn: 'PublishedVersions',
          },
        },
      });
    });

    test('runtime validation for snapStart', () => {
      const stack = new cdk.Stack();
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2531', reason: 'Intentionally testing on old runtime' });
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2530', reason: 'SnapStart is incomplete, we know' });

      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'bar',
        runtime: lambda.Runtime.NODEJS_20_X,
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart currently not supported by runtime nodejs20.x');
    });

    test('arm64 function using snapStart', () => {
      const stack = new cdk.Stack();
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2530', reason: 'SnapStart is incomplete, we know' });

      // WHEN
      new lambda.Function(stack, 'MyLambda', {
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        handler: 'example.Handler::handleRequest',
        runtime: lambda.Runtime.JAVA_11,
        architecture: lambda.Architecture.ARM_64,
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      });

      // THEN
      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties:
        {
          Handler: 'example.Handler::handleRequest',
          Runtime: 'java11',
          Architectures: ['arm64'],
          SnapStart: {
            ApplyOn: 'PublishedVersions',
          },
        },
      });
    });

    test('EFS validation for snapStart', () => {
      const stack = new cdk.Stack();
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2530', reason: 'SnapStart is incomplete, we know' });

      const vpc = new ec2.Vpc(stack, 'Vpc', {
        maxAzs: 3,
        natGateways: 1,
      });

      const fs = new efs.FileSystem(stack, 'Efs', {
        vpc,
      });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      expect(() => new lambda.Function(stack, 'MyLambda', {
        vpc,
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        handler: 'example.Handler::handleRequest',
        runtime: lambda.Runtime.JAVA_11,
        filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart is currently not supported using EFS');
    });

    test('ephemeral storage limit validation for snapStart', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        handler: 'example.Handler::handleRequest',
        runtime: lambda.Runtime.JAVA_11,
        ephemeralStorageSize: Size.mebibytes(1024),
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart is currently not supported using more than 512 MiB Ephemeral Storage');
    });

    test('multi-tenant function with snapStart should throw error', () => {
      const stack = new cdk.Stack();
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W2530', reason: 'SnapStart is incomplete, we know' });

      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
        handler: 'example.Handler::handleRequest',
        runtime: lambda.Runtime.JAVA_11,
        tenancyConfig: lambda.TenancyConfig.PER_TENANT,
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart is not supported for functions with tenant isolation mode');
    });

    test('container image function using SnapStart', () => {
      const stack = new cdk.Stack();

      // WHEN
      const fn = new lambda.DockerImageFunction(stack, 'MyLambda', {
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-lambda-handler')),
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      });

      fn.currentVersion;

      // THEN
      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties: {
          PackageType: 'Image',
          SnapStart: {
            ApplyOn: 'PublishedVersions',
          },
        },
      });
    });

    test('container image function with SnapStart and EFS throws', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc', { maxAzs: 3, natGateways: 1 });
      const fs = new efs.FileSystem(stack, 'Efs', { vpc });
      const accessPoint = fs.addAccessPoint('AccessPoint');

      expect(() => new lambda.DockerImageFunction(stack, 'MyLambda', {
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-lambda-handler')),
        vpc,
        filesystem: lambda.FileSystem.fromEfsAccessPoint(accessPoint, '/mnt/msg'),
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart is currently not supported using EFS');
    });

    test('container image function with SnapStart and >512 MiB ephemeral storage throws', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.DockerImageFunction(stack, 'MyLambda', {
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-lambda-handler')),
        ephemeralStorageSize: Size.mebibytes(1024),
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart is currently not supported using more than 512 MiB Ephemeral Storage');
    });

    test('container image function with SnapStart and tenant isolation throws', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.DockerImageFunction(stack, 'MyLambda', {
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, 'docker-lambda-handler')),
        tenancyConfig: lambda.TenancyConfig.PER_TENANT,
        snapStart: lambda.SnapStartConf.ON_PUBLISHED_VERSIONS,
      })).toThrow('SnapStart is not supported for functions with tenant isolation mode');
    });
  });

  describe('Recursive Loop', () => {
    test('with recursive loop protection', () => {
      const stack = new cdk.Stack();
      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'bar',
        runtime: lambda.Runtime.NODEJS_LATEST,
        recursiveLoop: lambda.RecursiveLoop.TERMINATE,
      });

      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties:
        {
          Code: { ZipFile: 'foo' },
          Handler: 'bar',
          Runtime: 'nodejs24.x',
          RecursiveLoop: 'Terminate',
        },
      });
    });

    test('without recursive loop protection', () => {
      const stack = new cdk.Stack();
      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'bar',
        runtime: lambda.Runtime.NODEJS_LATEST,
        recursiveLoop: lambda.RecursiveLoop.ALLOW,
      });

      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties:
        {
          Code: { ZipFile: 'foo' },
          Handler: 'bar',
          Runtime: 'nodejs24.x',
          RecursiveLoop: 'Allow',
        },
      });
    });

    test('default recursive loop protection', () => {
      const stack = new cdk.Stack();
      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'bar',
        runtime: lambda.Runtime.NODEJS_LATEST,
      });

      Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
        Properties:
        {
          Code: { ZipFile: 'foo' },
          Handler: 'bar',
          Runtime: 'nodejs24.x',
          // for default, if the property is not set up in stack it doesn't show up in the template.
        },
      });
    });
  });

  test('called twice for the same service principal but with different conditions', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Function', {
      code: lambda.Code.fromInline('xxx'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    const sourceArnA = 'arn:aws:s3:::bucket-a';
    const sourceArnB = 'arn:aws:s3:::bucket-b';
    const service = 's3.amazonaws.com';
    const conditionalPrincipalA = new iam.PrincipalWithConditions(new iam.ServicePrincipal(service), {
      ArnLike: {
        'aws:SourceArn': sourceArnA,
      },
      StringEquals: {
        'aws:SourceAccount': stack.account,
      },
    });
    const conditionalPrincipalB = new iam.PrincipalWithConditions(new iam.ServicePrincipal(service), {
      ArnLike: {
        'aws:SourceArn': sourceArnB,
      },
      StringEquals: {
        'aws:SourceAccount': stack.account,
      },
    });

    // WHEN
    fn.grantInvoke(conditionalPrincipalA);
    fn.grantInvoke(conditionalPrincipalB);

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::Lambda::Permission', 2);
    Template.fromStack(stack).hasResource('AWS::Lambda::Permission', {
      Properties: {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: service,
        SourceAccount: {
          Ref: 'AWS::AccountId',
        },
        SourceArn: sourceArnA,
      },
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Permission', {
      Properties: {
        Action: 'lambda:InvokeFunction',
        FunctionName: {
          'Fn::GetAtt': [
            'Function76856677',
            'Arn',
          ],
        },
        Principal: service,
        SourceAccount: {
          Ref: 'AWS::AccountId',
        },
        SourceArn: sourceArnB,
      },
    });
  });

  test('adds ADOT instrumentation to a ZIP Lambda function', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Base', {
      env: { account: '111111111111', region: 'us-west-2' },
    });

    // WHEN
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      adotInstrumentation: {
        layerVersion: lambda.AdotLayerVersion.fromJavaSdkLayerVersion(AdotLambdaLayerJavaSdkVersion.V1_32_0_1),
        execWrapper: lambda.AdotLambdaExecWrapper.REGULAR_HANDLER,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Layers: ['arn:aws:lambda:us-west-2:901920570463:layer:aws-otel-java-wrapper-amd64-ver-1-32-0:4'],
      Environment: {
        Variables: {
          AWS_LAMBDA_EXEC_WRAPPER: '/opt/otel-handler',
        },
      },
    });
  });

  test('adds ADOT instrumentation to a ZIP Lambda function for instrumentation', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Base', {
      env: { account: '111111111111', region: 'us-west-2' },
    });

    // WHEN
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      adotInstrumentation: {
        layerVersion: lambda.AdotLayerVersion.fromPythonSdkLayerVersion(lambda.AdotLambdaLayerPythonSdkVersion.V1_29_0),
        execWrapper: lambda.AdotLambdaExecWrapper.INSTRUMENT_HANDLER,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Layers: ['arn:aws:lambda:us-west-2:901920570463:layer:aws-otel-python-amd64-ver-1-29-0:1'],
      Environment: {
        Variables: {
          AWS_LAMBDA_EXEC_WRAPPER: '/opt/otel-instrument',
        },
      },
    });
  });

  test('Adot Instrumentation errors out when not using INSTRUMENT_HANDLER', () => {
    const stack = new cdk.Stack();

    expect(() => new lambda.Function(stack, 'Fn1', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_10,
      adotInstrumentation: {
        layerVersion: lambda.AdotLayerVersion.fromPythonSdkLayerVersion(lambda.AdotLambdaLayerPythonSdkVersion.V1_29_0),
        execWrapper: lambda.AdotLambdaExecWrapper.REGULAR_HANDLER,
      },
    })).toThrow(/Python Adot Lambda layer requires AdotLambdaExecWrapper.INSTRUMENT_HANDLER/);
  });

  test('adds ADOT instrumentation to a container image Lambda function', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Base', {
      env: { account: '111111111111', region: 'us-west-2' },
    });

    // WHEN
    expect(
      () =>
        new lambda.DockerImageFunction(stack, 'MyLambda', {
          code: lambda.DockerImageCode.fromImageAsset(dockerLambdaHandlerPath),
          adotInstrumentation: {
            layerVersion: lambda.AdotLayerVersion.fromJavaSdkLayerVersion(AdotLambdaLayerJavaSdkVersion.V1_32_0_1),
            execWrapper: lambda.AdotLambdaExecWrapper.REGULAR_HANDLER,
          },
        }),
    ).toThrow(/ADOT Lambda layer can't be configured with container image package type/);
  });

  describe('allowAllIpv6Outbound', () => {
    test('allowAllIpv6Outbound set to true', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc');

      new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        allowAllIpv6Outbound: true,
        vpc,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroup', {
        SecurityGroupEgress: [
          {
            CidrIp: '0.0.0.0/0',
            Description: 'Allow all outbound traffic by default',
            IpProtocol: '-1',
          },
          {
            CidrIpv6: '::/0',
            Description: 'Allow all outbound ipv6 traffic by default',
            IpProtocol: '-1',
          },
        ],
      });
    });

    test('throws when allowAllIpv6Outbound is defined without vpc', () => {
      const stack = new cdk.Stack();

      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        allowAllIpv6Outbound: true,
      })).toThrow(/Cannot configure \'allowAllIpv6Outbound\' without configuring a VPC/);
    });

    test('throws when both allowAllIpv6Outbound and securityGroup are defined', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc: vpc });

      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        allowAllIpv6Outbound: true,
        vpc,
        securityGroup: securityGroup,
      })).toThrow(/Configure \'allowAllIpv6Outbound\' directly on the supplied SecurityGroup./);
    });

    test('throws when both allowAllIpv6Outbound and securityGroups are defined', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Vpc');
      const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc: vpc });

      expect(() => new lambda.Function(stack, 'MyLambda', {
        code: new lambda.InlineCode('foo'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        allowAllIpv6Outbound: true,
        vpc,
        securityGroups: [securityGroup],
      })).toThrow(/Configure \'allowAllIpv6Outbound\' directly on the supplied SecurityGroups./);
    });
  });

  test('function with tenancy config PER_TENANT', () => {
    const stack = new cdk.Stack();
    new lambda.Function(stack, 'Lambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      tenancyConfig: lambda.TenancyConfig.PER_TENANT,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      TenancyConfig: {
        TenantIsolationMode: 'PER_TENANT',
      },
    });
  });

  test('function without tenancy config has no tenancy properties', () => {
    const stack = new cdk.Stack();
    new lambda.Function(stack, 'Lambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      // No tenancyConfig specified
    });

    const template = Template.fromStack(stack);
    const functions = template.findResources('AWS::Lambda::Function');
    const functionResource = functions[Object.keys(functions)[0]];
    expect(functionResource.Properties).not.toHaveProperty('TenancyConfig');
  });
});

test('throws if ephemeral storage size is out of bound', () => {
  const stack = new cdk.Stack();
  expect(() => new lambda.Function(stack, 'MyLambda', {
    code: new lambda.InlineCode('foo'),
    handler: 'bar',
    runtime: lambda.Runtime.NODEJS_LATEST,
    ephemeralStorageSize: Size.mebibytes(511),
  })).toThrow(/Ephemeral storage size must be between 512 and 10240 MB/);
});

test('set ephemeral storage to desired size', () => {
  const stack = new cdk.Stack();
  new lambda.Function(stack, 'MyLambda', {
    code: new lambda.InlineCode('foo'),
    handler: 'bar',
    runtime: lambda.Runtime.NODEJS_LATEST,
    ephemeralStorageSize: Size.mebibytes(1024),
  });

  Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
    Properties:
    {
      Code: { ZipFile: 'foo' },
      Handler: 'bar',
      Runtime: 'nodejs24.x',
      EphemeralStorage: {
        Size: 1024,
      },
    },
  });
});

test('ephemeral storage allows unresolved tokens', () => {
  const stack = new cdk.Stack();
  expect(() => {
    new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'bar',
      runtime: lambda.Runtime.NODEJS_LATEST,
      ephemeralStorageSize: Size.mebibytes(Lazy.number({ produce: () => 1024 })),
    });
  }).not.toThrow();
});

test('FunctionVersionUpgrade adds new description to function', () => {
  const app = new cdk.App({ context: { [cxapi.LAMBDA_RECOGNIZE_LAYER_VERSION]: true } });
  const stack = new cdk.Stack(app);
  new lambda.Function(stack, 'MyLambda', {
    code: new lambda.InlineCode('foo'),
    handler: 'bar',
    runtime: lambda.Runtime.NODEJS_LATEST,
    description: 'my description',
  });

  Aspects.of(stack).add(new lambda.FunctionVersionUpgrade(cxapi.LAMBDA_RECOGNIZE_LAYER_VERSION));

  Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
    Properties:
    {
      Code: { ZipFile: 'foo' },
      Handler: 'bar',
      Runtime: 'nodejs24.x',
      Description: Match.stringLikeRegexp('my description version-hash'),
    },
  });
});

test('function using a reserved environment variable', () => {
  const stack = new cdk.Stack();
  expect(() => new lambda.Function(stack, 'MyLambda', {
    code: new lambda.InlineCode('foo'),
    handler: 'index.handler',
    runtime: lambda.Runtime.PYTHON_3_9,
    environment: {
      AWS_REGION: 'ap-southeast-2',
    },
  })).toThrow(/AWS_REGION environment variable is reserved/);
});

test('test 2.87.0 version hash stability', () => {
  const theRuntime = new lambda.Runtime('node99.x');
  // GIVEN
  const app = new cdk.App({
    context: {
      '@aws-cdk/aws-lambda:recognizeLayerVersion': true,
    },
  });
  cdk.Validations.of(app).acknowledge(
    { id: 'CloudFormation-Validate::W3030', reason: 'Node 99.x is not valid, sure' },
  );
  const stack = new cdk.Stack(app, 'Stack');

  // WHEN
  const layer = new lambda.LayerVersion(stack, 'MyLayer', {
    code: lambda.Code.fromAsset(path.join(__dirname, 'x.zip')),
    compatibleRuntimes: [
      theRuntime,
    ],
  });

  const role = new iam.Role(stack, 'MyRole', {
    assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    managedPolicies: [
      iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole'),
      iam.ManagedPolicy.fromAwsManagedPolicyName('AWSXRayDaemonWriteAccess'),
    ],
  });

  const lambdaFn = new lambda.Function(stack, 'MyLambda', {
    runtime: theRuntime,
    memorySize: 128,
    handler: 'index.handler',
    timeout: cdk.Duration.seconds(30),
    environment: {
      VARIABLE_1: 'ONE',
    },
    code: lambda.Code.fromAsset(path.join(__dirname, 'x.zip')),
    role,
    currentVersionOptions: {
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    },
    layers: [
      layer,
    ],
  });

  new lambda.Alias(stack, 'MyAlias', {
    aliasName: 'current',
    version: lambdaFn.currentVersion,
  });

  // THEN
  // Precalculated version hash using 2.87.0 version
  Template.fromStack(stack).hasResource('AWS::Lambda::Alias', {
    Properties: {
      FunctionVersion: {
        'Fn::GetAtt': [
          'MyLambdaCurrentVersionE7A382CC36ba24a5a2aa206296d3e383129fd83a',
          'Version',
        ],
      },
    },
  });
});

describe('VPC configuration', () => {
  test('with both securityGroup and securityGroups', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: false,
    });
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroup,
      securityGroups: [securityGroup],
    })).toThrow(/Only one of the function props, securityGroup or securityGroups, is allowed/);
  });

  test('with allowAllOutbound and no VPC', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      allowAllOutbound: true,
    })).toThrow(/Cannot configure 'allowAllOutbound' without configuring a VPC/);
  });

  test('with allowAllOutbound and no VPC', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      allowAllOutbound: true,
    })).toThrow(/Cannot configure 'allowAllOutbound' without configuring a VPC/);
  });

  test('with securityGroup and no VPC', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: false,
    });
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroup,
    })).toThrow(/Cannot configure 'securityGroup' without configuring a VPC/);
  });

  test('with securityGroups and no VPC', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: false,
    });
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroups: [securityGroup],
    })).toThrow(/Cannot configure 'securityGroups' without configuring a VPC/);
  });

  test('with vpcSubnets and no VPC', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    })).toThrow(/Cannot configure 'vpcSubnets' without configuring a VPC/);
  });

  test('with securityGroup and allowAllOutbound', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: false,
    });
    expect(() => new lambda.Function(stack, 'MyLambda', {
      vpc,
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroup,
      allowAllOutbound: false,
    })).toThrow(/Configure 'allowAllOutbound' directly on the supplied SecurityGroup./);
  });

  test('with securityGroups and allowAllOutbound', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: false,
    });
    expect(() => new lambda.Function(stack, 'MyLambda', {
      vpc,
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroups: [securityGroup],
      allowAllOutbound: false,
    })).toThrow(/Configure 'allowAllOutbound' directly on the supplied SecurityGroups./);
  });

  test('with VPC and empty securityGroups creates a default security group', () => {
    const stack = new cdk.Stack();

    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    new lambda.Function(stack, 'MyLambda', {
      vpc,
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroups: [],
    });

    Template.fromStack(stack).resourceCountIs('AWS::EC2::SecurityGroup', 1);
  });

  test('with no VPC and empty securityGroups', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroups: [],
    })).not.toThrow();
  });

  test('with empty securityGroups and allowAllOutbound', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    expect(() => new lambda.Function(stack, 'MyLambda', {
      vpc,
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      securityGroups: [],
      allowAllOutbound: false,
    })).not.toThrow();
  });

  test('with ipv6AllowedForDualStack and no VPC', () => {
    const stack = new cdk.Stack();
    expect(() => new lambda.Function(stack, 'MyLambda', {
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      ipv6AllowedForDualStack: true,
    })).toThrow(/Cannot configure 'ipv6AllowedForDualStack' without configuring a VPC/);
  });

  test('set ipv6AllowedForDualStack with VPC', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: true,
      allowAllIpv6Outbound: true,
    });
    new lambda.Function(stack, 'MyLambda', {
      vpc: vpc,
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      ipv6AllowedForDualStack: true,
      securityGroups: [securityGroup],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties:
      {
        Code: { ZipFile: 'foo' },
        Handler: 'index.handler',
        Runtime: 'python3.9',
        Role: { 'Fn::GetAtt': ['MyLambdaServiceRole4539ECB6', 'Arn'] },
        VpcConfig: {
          Ipv6AllowedForDualStack: true,
          SecurityGroupIds: [
            { 'Fn::GetAtt': ['LambdaSG9DBFCFB7', 'GroupId'] },
          ],
          SubnetIds: [
            { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
            { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
          ],
        },
      },
    });
  });

  test('set ipv6AllowedForDualStack to False with VPC', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc', {
      maxAzs: 3,
      natGateways: 1,
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'LambdaSG', {
      vpc,
      allowAllOutbound: true,
      allowAllIpv6Outbound: true,
    });
    new lambda.Function(stack, 'MyLambda', {
      vpc: vpc,
      code: new lambda.InlineCode('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
      ipv6AllowedForDualStack: false,
      securityGroups: [securityGroup],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties:
      {
        Code: { ZipFile: 'foo' },
        Handler: 'index.handler',
        Runtime: 'python3.9',
        Role: { 'Fn::GetAtt': ['MyLambdaServiceRole4539ECB6', 'Arn'] },
        VpcConfig: {
          Ipv6AllowedForDualStack: false,
          SecurityGroupIds: [
            { 'Fn::GetAtt': ['LambdaSG9DBFCFB7', 'GroupId'] },
          ],
          SubnetIds: [
            { Ref: 'VpcPrivateSubnet1Subnet536B997A' },
            { Ref: 'VpcPrivateSubnet2Subnet3788AAA1' },
          ],
        },
      },
    });
  });
});

describe('latest Lambda node runtime', () => {
  test('with region agnostic stack', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.determineLatestNodeRuntime(stack),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Runtime: 'nodejs24.x',
      },
    });
  });

  test('with stack in commercial region', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.determineLatestNodeRuntime(stack),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Runtime: 'nodejs24.x',
      },
    });
  });

  test('with stack in china region', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'Stack', { env: { region: 'cn-north-1' } });

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.determineLatestNodeRuntime(stack),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Runtime: 'nodejs24.x',
      },
    });
  });

  test('with stack in adc region', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'Stack', { env: { region: 'us-iso-east-1' } });

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.determineLatestNodeRuntime(stack),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Runtime: 'nodejs24.x',
      },
    });
  });

  test('with stack in govcloud region', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'Stack', { env: { region: 'us-gov-east-1' } });

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.determineLatestNodeRuntime(stack),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Runtime: 'nodejs24.x',
      },
    });
  });

  test('with stack in unsupported region', () => {
    // GIVEN
    const stack = new cdk.Stack(undefined, 'Stack', { env: { region: 'us-fake-1' } });

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.determineLatestNodeRuntime(stack),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Runtime: 'nodejs24.x',
      },
    });
  });
});

// Test sourceKMSKeyArn feature
describe('CMCMK', () => {
  test('set sourceKMSKeyArn using fromAsset', () => {
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'myImportedKey', {
      enableKeyRotation: true,
    });
    const option = {
      sourceKMSKey: key,
    };
    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip'), option),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: { 'Fn::GetAtt': ['myImportedKey10DE2890', 'Arn'] },
        },
        Runtime: 'python3.9',
        Handler: 'index.handler',
      },
    });
  });

  test('no sourceKMSKey provided using fromAsset', () => {
    const stack = new cdk.Stack();

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromAsset(path.join(__dirname, 'handler.zip')),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: Match.absent(),
        },
      },
    });
  });

  test('set sourceKMSKeyArn using fromBucket', () => {
    const stack = new cdk.Stack();
    // S3 Bucket
    const key = 'script';
    let bucket: s3.IBucket;
    bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'bucketname');

    // KMS
    const KMSkey = new kms.Key(stack, 'myImportedKey', {
      enableKeyRotation: true,
    });
    const option = {
      sourceKMSKey: KMSkey,
    };
    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromBucketV2(bucket, key, option),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: { 'Fn::GetAtt': ['myImportedKey10DE2890', 'Arn'] },
        },
        Runtime: 'python3.9',
        Handler: 'index.handler',
      },
    });
  });

  test('no sourceKMSKey provided using fromBucket', () => {
    const stack = new cdk.Stack();
    // S3 Bucket
    const key = 'script';
    let bucket: s3.IBucket;
    bucket = s3.Bucket.fromBucketName(stack, 'Bucket', 'bucketname');

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromBucketV2(bucket, key),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: Match.absent(),
        },
        Runtime: 'python3.9',
        Handler: 'index.handler',
      },
    });
  });

  test('set sourceKMSKeyArn using fromCfnParameters', () => {
    const stack = new cdk.Stack();
    // KMS
    const KMSkey = new kms.Key(stack, 'myImportedKey', {
      enableKeyRotation: true,
    });
    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromCfnParameters({
        sourceKMSKey: KMSkey,
      }),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: { 'Fn::GetAtt': ['myImportedKey10DE2890', 'Arn'] },
        },
        Runtime: 'python3.9',
        Handler: 'index.handler',
      },
    });
  });

  test('no sourceKMSKey provided using fromCfnParameters', () => {
    const stack = new cdk.Stack();

    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromCfnParameters(),
      handler: 'index.handler',
      runtime: lambda.Runtime.PYTHON_3_9,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: Match.absent(),
        },
        Runtime: 'python3.9',
        Handler: 'index.handler',
      },
    });
  });

  test('set sourceKMSKeyArn using fromCustomCommand', () => {
    const mockCallsites = jest.fn();
    jest.mock('../lib/util', () => ({
      ...jest.requireActual('../lib/util'),
      callsites: () => mockCallsites(),
    }));
    bockfs({
      '/home/project/function.test.handler7.zip': '// nothing',
    });
    using bockPath = bockfs.workingDirectory('/home/project');
    mockCallsites.mockImplementation(() => [
      { getFunctionName: () => 'NodejsFunction' },
      { getFileName: () => bockPath.translate`function.test.ts` },
    ]);

    const stack = new cdk.Stack();
    // KMS
    const KMSkey = new kms.Key(stack, 'myImportedKey', {
      enableKeyRotation: true,
    });
    // const command = ;
    const commandOptions = { sourceKMSKey: KMSkey };
    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromCustomCommand('function.test.handler7.zip', ['node'], commandOptions),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: { 'Fn::GetAtt': ['myImportedKey10DE2890', 'Arn'] },
        },
        Runtime: 'nodejs24.x',
        Handler: 'index.handler',
      },
    });
    bockfs.restore();
  });

  test('no sourceKMSKey provided using fromCustomCommand', () => {
    const mockCallsites = jest.fn();
    jest.mock('../lib/util', () => ({
      ...jest.requireActual('../lib/util'),
      callsites: () => mockCallsites(),
    }));
    bockfs({
      '/home/project/function.test.handler7.zip': '// nothing',
    });
    using bockPath = bockfs.workingDirectory('/home/project');
    mockCallsites.mockImplementation(() => [
      { getFunctionName: () => 'NodejsFunction' },
      { getFileName: () => bockPath.translate`function.test.ts` },
    ]);

    const stack = new cdk.Stack();
    // WHEN
    new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromCustomCommand('function.test.handler7.zip', ['node']),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    // THEN
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Code: {
          SourceKMSKeyArn: Match.absent(),
        },
        Runtime: 'nodejs24.x',
        Handler: 'index.handler',
      },
    });
    bockfs.restore();
  });
});

describe('tag propagation to logGroup on FF USE_CDK_MANAGED_LAMBDA_LOGGROUP enabled', () => {
  it('log group inherits tags from function when USE_CDK_MANAGED_LAMBDA_LOGGROUP is enabled', () => {
    const app = new cdk.App({ context: { [cxapi.USE_CDK_MANAGED_LAMBDA_LOGGROUP]: true } });
    const stack = new cdk.Stack(app, 'Stack');

    const fn = new lambda.Function(stack, 'Function', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    cdk.Tags.of(fn).add('Environment', 'Test');
    cdk.Tags.of(fn).add('Owner', 'CDKTeam');

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::Logs::LogGroup', {
      Tags: Match.arrayWith([
        Match.objectLike({ Key: 'Environment', Value: 'Test' }),
        Match.objectLike({ Key: 'Owner', Value: 'CDKTeam' }),
      ]),
    });
  });
});

describe('USE_CDK_MANAGED_LAMBDA_LOGGROUP defaults to false when not specified', () => {
  it('does not create a managed log group when context flag is not specified', () => {
    // GIVEN
    const app = new cdk.App(); // No context provided
    const stack = new cdk.Stack(app, 'Stack');

    // WHEN
    new lambda.Function(stack, 'Function', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    // THEN
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::Logs::LogGroup', 0); // No log group should be created
  });
});

describe('Lambda Function log group behavior', () => {
  it('throws if both logRetention and logGroup are set', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'TestStack');

    expect(() => {
      new lambda.Function(stack, 'TestFunction', {
        code: lambda.Code.fromInline('exports.handler = async () => {};'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        logRetention: logs.RetentionDays.ONE_WEEK,
        logGroup: new logs.LogGroup(stack, 'CustomLogGroup'),
      });
    }).toThrow('CDK does not support setting logRetention and logGroup');
  });

  it('does not throw if only logRetention is set', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'TestStack');

    expect(() => {
      new lambda.Function(stack, 'LogRetentionOnlyFunction', {
        code: lambda.Code.fromInline('exports.handler = async () => {};'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        logRetention: logs.RetentionDays.ONE_WEEK,
      });
    }).not.toThrow();
  });

  it('does not throw if only logGroup is set', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'TestStack');

    expect(() => {
      new lambda.Function(stack, 'LogGroupOnlyFunction', {
        code: lambda.Code.fromInline('exports.handler = async () => {};'),
        handler: 'index.handler',
        runtime: lambda.Runtime.NODEJS_LATEST,
        logGroup: new logs.LogGroup(stack, 'MyLogGroup'),
      });
    }).not.toThrow();
  });
});

describe('telemetry metadata', () => {
  beforeEach(() => {
    // In case we didn't compile using jsii
    if (!(lambda.Function as any).hasOwnProperty(JSII_RUNTIME_SYMBOL)) {
      (lambda.Function as any)[JSII_RUNTIME_SYMBOL] = {
        fqn: 'aws-cdk-lib.aws-lambda.Function',
      };
    }
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('redaction happens when feature flag is enabled', () => {
    const app = new cdk.App();
    app.node.setContext(cxapi.ENABLE_ADDITIONAL_METADATA_COLLECTION, true);
    const stack = new cdk.Stack(app);

    const fn = new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    fn.addEnvironment('foo', '1234567890', {
      removeInEdge: true,
    });

    expect(fn.node.metadata).toStrictEqual([
      {
        data: { code: '*', handler: '*', runtime: '*' },
        trace: undefined,
        type: 'aws:cdk:analytics:construct',
      },
      {
        data: { addEnvironment: ['*', '*', { removeInEdge: true }] },
        trace: undefined,
        type: 'aws:cdk:analytics:method',
      },
    ]);
  });

  it('redaction happens when feature flag is disabled', () => {
    const app = new cdk.App();
    app.node.setContext(cxapi.ENABLE_ADDITIONAL_METADATA_COLLECTION, false);
    const stack = new cdk.Stack(app);

    const fn = new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    expect(fn.node.metadata).toStrictEqual([]);
  });
});
describe('L1 Relationships', () => {
  it('simple union', () => {
    const stack = new cdk.Stack();
    const role = new iam.Role(stack, 'SomeRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
    new lambda.CfnFunction(stack, 'MyLambda', {
      code: { zipFile: 'foo' },
      handler: 'index.handler',
      runtime: 'python3.14',
      role: role, // Simple Union
    });
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Role: { 'Fn::GetAtt': ['SomeRole6DDC54DD', 'Arn'] },
      },
    });
  });

  it('array of unions', () => {
    const stack = new cdk.Stack();
    const role = new iam.Role(stack, 'SomeRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
    const layer1 = new lambda.LayerVersion(stack, 'LayerVersion1', {
      code: lambda.Code.fromAsset(path.join(__dirname, 'my-lambda-handler')),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_14],
    });
    const layer2 = new lambda.LayerVersion(stack, 'LayerVersion2', {
      code: lambda.Code.fromAsset(path.join(__dirname, 'my-lambda-handler')),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_14],
    });
    new lambda.CfnFunction(stack, 'MyLambda', {
      code: { zipFile: 'foo' },
      handler: 'index.handler',
      runtime: 'python3.14',
      role: role,
      layers: [layer1, layer2], // Array of Unions
    });
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Role: { 'Fn::GetAtt': ['SomeRole6DDC54DD', 'Arn'] },
        Layers: [{ Ref: 'LayerVersion139D4D7A8' }, { Ref: 'LayerVersion23E5F3CEA' }],
      },
    });
  });

  it('array reference should be valid', () => {
    const stack = new cdk.Stack();
    const role = new iam.Role(stack, 'SomeRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
    const layer1 = new lambda.LayerVersion(stack, 'LayerVersion1', {
      code: lambda.Code.fromAsset(path.join(__dirname, 'my-lambda-handler')),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_14],
    });
    const layerArray = [layer1, 'layer2Arn'];
    new lambda.CfnFunction(stack, 'MyLambda', {
      code: { zipFile: 'foo' },
      handler: 'index.handler',
      runtime: 'python3.14',
      role: role,
      layers: layerArray,
    });

    layerArray.push('layer3Arn');

    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Role: { 'Fn::GetAtt': ['SomeRole6DDC54DD', 'Arn'] },
        Layers: [{ Ref: 'LayerVersion139D4D7A8' }, 'layer2Arn', 'layer3Arn'],
      },
    });
  });

  it('tokens should be passed as is', () => {
    const stack = new cdk.Stack();
    const role = new iam.Role(stack, 'SomeRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
    const bucket = new s3.Bucket(stack, 'MyBucket');

    const codeToken = cdk.Token.asAny({
      resolve: () => ({ s3Bucket: bucket.bucketName }),
    });

    const fsConfigToken = cdk.Token.asAny({
      resolve: () => ([{ arn: 'TestArn', localMountPath: '/mnt' }]),
    });

    new lambda.CfnFunction(stack, 'MyLambda', {
      code: codeToken,
      role: role,
      fileSystemConfigs: fsConfigToken,
    });
    Template.fromStack(stack).hasResource('AWS::Lambda::Function', {
      Properties: {
        Role: { 'Fn::GetAtt': ['SomeRole6DDC54DD', 'Arn'] },
        Code: { S3Bucket: { Ref: 'MyBucketF68F3FF0' } },
        FileSystemConfigs: [{ Arn: 'TestArn', LocalMountPath: '/mnt' }],
      },
    });
  });
});

function newTestLambda(scope: constructs.Construct) {
  return new lambda.Function(scope, 'MyLambda', {
    code: new lambda.InlineCode('foo'),
    handler: 'bar',
    runtime: lambda.Runtime.PYTHON_3_14,
  });
}
