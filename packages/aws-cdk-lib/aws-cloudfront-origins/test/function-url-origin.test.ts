import { Template, Match } from '../../assertions';
import * as cloudfront from '../../aws-cloudfront';
import { OriginIpAddressType } from '../../aws-cloudfront';
import * as lambda from '../../aws-lambda';
import { Duration, Stack } from '../../core';
import { FunctionUrlOrigin } from '../lib';

let stack: Stack;

beforeEach(() => {
  stack = new Stack();
});

test('Correctly renders the origin for a Lambda Function URL', () => {
  const fn = new lambda.Function(stack, 'MyFunction', {
    code: lambda.Code.fromInline('exports.handler = async () => {};'),
    handler: 'index.handler',
    runtime: lambda.Runtime.NODEJS_20_X,
  });

  const fnUrl = fn.addFunctionUrl({
    authType: lambda.FunctionUrlAuthType.NONE,
  });

  const origin = new FunctionUrlOrigin(fnUrl);
  const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });

  expect(stack.resolve(originBindConfig.originProperty)).toEqual({
    id: 'StackOriginLambdaFunctionURL',
    domainName: {
      'Fn::Select': [
        2,
        {
          'Fn::Split': [
            '/',
            { 'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'] },
          ],
        },
      ],
    },
    customOriginConfig: {
      originProtocolPolicy: 'https-only',
      originSslProtocols: ['TLSv1.2'],
    },
  });
});

test('Correctly sets readTimeout and keepaliveTimeout', () => {
  const fn = new lambda.Function(stack, 'MyFunction', {
    code: lambda.Code.fromInline('exports.handler = async () => {};'),
    handler: 'index.handler',
    runtime: lambda.Runtime.NODEJS_20_X,
  });

  const fnUrl = fn.addFunctionUrl({
    authType: lambda.FunctionUrlAuthType.NONE,
  });

  const origin = new FunctionUrlOrigin(fnUrl, {
    readTimeout: Duration.seconds(120),
    keepaliveTimeout: Duration.seconds(60),
  });

  const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });

  expect(stack.resolve(originBindConfig.originProperty)).toMatchObject({
    customOriginConfig: {
      originReadTimeout: 120,
      originKeepaliveTimeout: 60,
    },
  });
});

describe('FunctionUrlOriginAccessControl', () => {
  test('Correctly adds permission to Lambda for CloudFront', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    new cloudfront.Distribution(stack, 'MyDistribution', {
      defaultBehavior: {
        origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {}),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Origins: Match.arrayWith([
          Match.objectLike({
            DomainName: {
              'Fn::Select': [
                2,
                {
                  'Fn::Split': [
                    '/',
                    {
                      'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'],
                    },
                  ],
                },
              ],
            },
            OriginAccessControlId: Match.objectLike({
              'Fn::GetAtt': [
                'MyDistributionOrigin1FunctionUrlOriginAccessControl0591AF75',
                'Id',
              ],
            }),
          }),
        ]),
      },
    });

    template.hasResourceProperties('AWS::Lambda::Permission', {
      Action: 'lambda:InvokeFunctionUrl',
      FunctionName: {
        'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionArn'],
      },
      Principal: 'cloudfront.amazonaws.com',
      SourceArn: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':cloudfront::',
            { Ref: 'AWS::AccountId' },
            ':distribution/',
            { Ref: 'MyDistribution6271DFB5' },
          ],
        ],
      },
    });
  });

  test('Creates Lambda Function URL origin with default Origin Access Control', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    new cloudfront.Distribution(stack, 'MyDistribution', {
      defaultBehavior: {
        origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
          originAccessControl: undefined,
        }),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Origins: Match.arrayWith([
          Match.objectLike({
            DomainName: {
              'Fn::Select': [
                2,
                {
                  'Fn::Split': [
                    '/',
                    {
                      'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'],
                    },
                  ],
                },
              ],
            },
            OriginAccessControlId: Match.objectLike({
              'Fn::GetAtt': [
                'MyDistributionOrigin1FunctionUrlOriginAccessControl0591AF75',
                'Id',
              ],
            }),
          }),
        ]),
      },
    });

    template.hasResourceProperties('AWS::CloudFront::OriginAccessControl', {
      OriginAccessControlConfig: {
        OriginAccessControlOriginType: 'lambda',
        SigningBehavior: 'always',
        SigningProtocol: 'sigv4',
      },
    });

    template.hasResourceProperties('AWS::Lambda::Permission', {
      Action: 'lambda:InvokeFunctionUrl',
      FunctionName: {
        'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionArn'],
      },
      Principal: 'cloudfront.amazonaws.com',
      SourceArn: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':cloudfront::',
            { Ref: 'AWS::AccountId' },
            ':distribution/',
            { Ref: 'MyDistribution6271DFB5' },
          ],
        ],
      },
    });
  });

  test('Correctly configures CloudFront Distribution with a custom Origin Access Control without signing prop defined', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    // Custom OAC configuration without defining `signing`
    const oac = new cloudfront.FunctionUrlOriginAccessControl(stack, 'CustomOAC', {
      originAccessControlName: 'CustomLambdaOAC',
    });

    new cloudfront.Distribution(stack, 'MyDistribution', {
      defaultBehavior: {
        origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
          originAccessControl: oac,
        }),
      },
    });

    const template = Template.fromStack(stack);

    // Validate the default signing behavior is SIGV4_ALWAYS
    template.hasResourceProperties('AWS::CloudFront::OriginAccessControl', {
      OriginAccessControlConfig: {
        Name: 'CustomLambdaOAC',
        OriginAccessControlOriginType: 'lambda',
        SigningBehavior: 'always',
        SigningProtocol: 'sigv4',
      },
    });
  });

  test('Correctly configures CloudFront Distribution with a custom Origin Access Control with signing prop defined', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    // Custom OAC configuration with signing defined as SIGV4_NO_OVERRIDE
    const oac = new cloudfront.FunctionUrlOriginAccessControl(stack, 'CustomOAC', {
      originAccessControlName: 'CustomLambdaOAC',
      signing: cloudfront.Signing.SIGV4_NO_OVERRIDE,
    });

    new cloudfront.Distribution(stack, 'MyDistribution', {
      defaultBehavior: {
        origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
          originAccessControl: oac,
        }),
      },
    });

    const template = Template.fromStack(stack);

    // Validate the signing behavior is correctly set to SIGV4_NO_OVERRIDE
    template.hasResourceProperties('AWS::CloudFront::OriginAccessControl', {
      OriginAccessControlConfig: {
        Name: 'CustomLambdaOAC',
        OriginAccessControlOriginType: 'lambda',
        SigningBehavior: 'no-override',
        SigningProtocol: 'sigv4',
      },
    });
  });

  test('Correctly adds permission for an imported Lambda Function', () => {
    const importedFn = lambda.Function.fromFunctionArn(stack, 'ImportedFunction', 'arn:aws:lambda:us-east-1:123456789012:function:my-imported-fn');

    const fnUrl = importedFn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    new cloudfront.Distribution(stack, 'MyDistribution', {
      defaultBehavior: {
        origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {}),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::Lambda::Permission', {
      Action: 'lambda:InvokeFunctionUrl',
      FunctionName: {
        'Fn::GetAtt': ['ImportedFunctionFunctionUrlB3FF8A17', 'FunctionArn'],
      },
      Principal: 'cloudfront.amazonaws.com',
      SourceArn: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':cloudfront::',
            { Ref: 'AWS::AccountId' },
            ':distribution/',
            { Ref: 'MyDistribution6271DFB5' },
          ],
        ],
      },
    });
  });
  test('Correctly creates a Lambda Function URL Origin with default properties', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    const origin = FunctionUrlOrigin.withOriginAccessControl(fnUrl);

    const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });

    expect(stack.resolve(originBindConfig.originProperty)).toEqual({
      id: 'StackOriginLambdaFunctionURL',
      domainName: {
        'Fn::Select': [
          2,
          {
            'Fn::Split': [
              '/',
              { 'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'] },
            ],
          },
        ],
      },
      customOriginConfig: {
        originProtocolPolicy: 'https-only',
        originSslProtocols: ['TLSv1.2'],
      },
      originAccessControlId: {
        'Fn::GetAtt': [
          'FunctionUrlOriginAccessControlC9E60518',
          'Id',
        ],
      },
    });
  });
  test('Creates Lambda Function URL origin with custom origin properties', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    const origin = FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
      readTimeout: Duration.seconds(120),
      keepaliveTimeout: Duration.seconds(60),
    });

    const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });

    expect(stack.resolve(originBindConfig.originProperty)).toEqual({
      id: 'StackOriginLambdaFunctionURL',
      domainName: {
        'Fn::Select': [
          2,
          {
            'Fn::Split': [
              '/',
              { 'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'] },
            ],
          },
        ],
      },
      customOriginConfig: {
        originProtocolPolicy: 'https-only',
        originSslProtocols: ['TLSv1.2'],
        originReadTimeout: 120,
        originKeepaliveTimeout: 60,
      },
      originAccessControlId: {
        'Fn::GetAtt': [
          'FunctionUrlOriginAccessControlC9E60518',
          'Id',
        ],
      },
    });
  });

  test('OAC SigV4 and AuthType is None', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
    });

    expect(() => {
      new cloudfront.Distribution(stack, 'MyDistribution', {
        defaultBehavior: {
          origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
            originAccessControl: new cloudfront.FunctionUrlOriginAccessControl(stack, 'OAC', {
              signing: cloudfront.Signing.SIGV4_ALWAYS,
            }),
          }),
        },
      });
    }).toThrow('The authType of the Function URL must be set to AWS_IAM when origin access control signing method is SIGV4_ALWAYS.');
  });

  test('renders responseCompletionTimeout in origin property', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
    });

    const origin = new FunctionUrlOrigin(fnUrl, {
      responseCompletionTimeout: Duration.seconds(120),
    });

    const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });

    expect(stack.resolve(originBindConfig.originProperty)).toEqual({
      id: 'StackOriginLambdaFunctionURL',
      domainName: {
        'Fn::Select': [
          2,
          {
            'Fn::Split': [
              '/',
              { 'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'] },
            ],
          },
        ],
      },
      responseCompletionTimeout: 120,
      customOriginConfig: {
        originProtocolPolicy: 'https-only',
        originSslProtocols: ['TLSv1.2'],
      },
    });
  });

  test('configure both responseCompletionTimeout and readTimeout', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
    });

    const origin = new FunctionUrlOrigin(fnUrl, {
      responseCompletionTimeout: Duration.seconds(60),
      readTimeout: Duration.seconds(60),
    });

    const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });

    expect(stack.resolve(originBindConfig.originProperty)).toEqual({
      id: 'StackOriginLambdaFunctionURL',
      domainName: {
        'Fn::Select': [
          2,
          {
            'Fn::Split': [
              '/',
              { 'Fn::GetAtt': ['MyFunctionFunctionUrlFF6DE78C', 'FunctionUrl'] },
            ],
          },
        ],
      },
      responseCompletionTimeout: 60,
      customOriginConfig: {
        originProtocolPolicy: 'https-only',
        originSslProtocols: ['TLSv1.2'],
        originReadTimeout: 60,
      },
    });
  });

  test('throw error for configuring readTimeout less than responseCompletionTimeout value', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });
    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
    });

    expect(() => {
      new FunctionUrlOrigin(fnUrl, {
        responseCompletionTimeout: Duration.seconds(30),
        readTimeout: Duration.seconds(60),
      });
    }).toThrow('responseCompletionTimeout must be equal to or greater than readTimeout (60s), got: 30s.');
  });
});

describe('ipAddressType', () => {
  test.each([OriginIpAddressType.DUALSTACK, OriginIpAddressType.IPV4, OriginIpAddressType.IPV6])('Correctly sets ipAddressType to %s', (ipAddressType) => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });
    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
    });
    const origin = new FunctionUrlOrigin(fnUrl, {
      ipAddressType,
    });
    const originBindConfig = origin.bind(stack, { originId: 'StackOriginLambdaFunctionURL' });
    expect(stack.resolve(originBindConfig.originProperty)).toMatchObject({
      customOriginConfig: {
        ipAddressType,
      },
    });
  });

  test('Correctly sets ipAddressType with OAC', () => {
    const fn = new lambda.Function(stack, 'MyFunction', {
      code: lambda.Code.fromInline('exports.handler = async () => {};'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_20_X,
    });

    const fnUrl = fn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    new cloudfront.Distribution(stack, 'MyDistribution', {
      defaultBehavior: {
        origin: FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
          ipAddressType: OriginIpAddressType.DUALSTACK,
        }),
      },
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Origins: Match.arrayWith([
          Match.objectLike({
            CustomOriginConfig: Match.objectLike({
              IpAddressType: OriginIpAddressType.DUALSTACK,
            }),
          }),
        ]),
      },
    });
  });
});

function testFunctionUrl(): lambda.IFunctionUrl {
  const fn = new lambda.Function(stack, 'TimeoutValidationFunction', {
    code: lambda.Code.fromInline('exports.handler = async () => {};'),
    handler: 'index.handler',
    runtime: lambda.Runtime.NODEJS_20_X,
  });
  return fn.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.NONE });
}

// Both construction paths validate independently: the plain constructor and the
// OAC variant returned by withOriginAccessControl.
interface TimeoutProps {
  readonly readTimeout?: Duration;
  readonly keepaliveTimeout?: Duration;
}

const originFactories: Array<[string, (fnUrl: lambda.IFunctionUrl, props: TimeoutProps) => unknown]> = [
  ['FunctionUrlOrigin', (fnUrl, props) => new FunctionUrlOrigin(fnUrl, props)],
  ['FunctionUrlOrigin.withOriginAccessControl', (fnUrl, props) => FunctionUrlOrigin.withOriginAccessControl(fnUrl, props)],
];

describe.each(originFactories)('%s timeout validation', (_name, createOrigin) => {
  test('validates readTimeout is at least 1 second', () => {
    expect(() => {
      createOrigin(testFunctionUrl(), { readTimeout: Duration.seconds(0) });
    }).toThrow('readTimeout: Must be an int 1 seconds or greater; received 0.');
  });

  test.each([
    Duration.seconds(121),
    Duration.minutes(5),
  ])('accepts readTimeout above the default quota, which the service validates at deploy time', (readTimeout) => {
    expect(() => {
      createOrigin(testFunctionUrl(), { readTimeout });
    }).not.toThrow();
  });

  test('validates keepaliveTimeout is at least 1 second', () => {
    expect(() => {
      createOrigin(testFunctionUrl(), { keepaliveTimeout: Duration.seconds(0) });
    }).toThrow('keepaliveTimeout: Must be an int 1 seconds or greater; received 0.');
  });

  test.each([
    Duration.seconds(301),
    Duration.minutes(10),
  ])('accepts keepaliveTimeout above the default quota, which the service validates at deploy time', (keepaliveTimeout) => {
    expect(() => {
      createOrigin(testFunctionUrl(), { keepaliveTimeout });
    }).not.toThrow();
  });
});
