import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Match, Template } from '../../assertions';
import * as certificatemanager from '../../aws-certificatemanager';
import * as iam from '../../aws-iam';
import * as lambda from '../../aws-lambda';
import * as s3 from '../../aws-s3';
import * as cdk from '../../core';
import {
  CfnDistribution,
  CloudFrontWebDistribution,
  Function,
  FunctionCode,
  FunctionEventType,
  GeoRestriction,
  KeyGroup,
  LambdaEdgeEventType,
  OriginAccessIdentity,
  PublicKey,
  SecurityPolicyProtocol,
  SSLMethod,
  ViewerCertificate,
  ViewerProtocolPolicy,
} from '../lib';

/* eslint-disable @stylistic/quote-props */

const publicKey = `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAudf8/iNkQgdvjEdm6xYS
JAyxd/kGTbJfQNg9YhInb7TSm0dGu0yx8yZ3fnpmxuRPqJIlaVr+fT4YRl71gEYa
dlhHmnVegyPNjP9dNqZ7zwNqMEPOPnS/NOHbJj1KYKpn1f8pPNycQ5MQCntKGnSj
6fc+nbcC0joDvGz80xuy1W4hLV9oC9c3GT26xfZb2jy9MVtA3cppNuTwqrFi3t6e
0iGpraxZlT5wewjZLpQkngqYr6s3aucPAZVsGTEYPo4nD5mswmtZOm+tgcOrivtD
/3sD/qZLQ6c5siqyS8aTraD6y+VXugujfarTU65IeZ6QAUbLMsWuZOIi5Jn8zAwx
NQIDAQAB
-----END PUBLIC KEY-----`;

describe('web distribution', () => {
  testDeprecated('distribution with custom origin adds custom origin', () => {
    const stack = new cdk.Stack();

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          originHeaders: {
            'X-Custom-Header': 'somevalue',
          },
          customOriginSource: {
            domainName: 'myorigin.com',
          },
          originShieldRegion: 'us-east-1',
          behaviors: [
            {
              isDefaultBehavior: true,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches(
      {
        'Resources': {
          'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
            'Type': 'AWS::CloudFront::Distribution',
            'Properties': {
              'DistributionConfig': {
                'DefaultCacheBehavior': {
                  'AllowedMethods': [
                    'GET',
                    'HEAD',
                  ],
                  'CachedMethods': [
                    'GET',
                    'HEAD',
                  ],
                  'ForwardedValues': {
                    'Cookies': {
                      'Forward': 'none',
                    },
                    'QueryString': false,
                  },
                  'TargetOriginId': 'origin1',
                  'ViewerProtocolPolicy': 'redirect-to-https',
                  'Compress': true,
                },
                'DefaultRootObject': 'index.html',
                'Enabled': true,
                'HttpVersion': 'http2',
                'IPV6Enabled': true,
                'Origins': [
                  {
                    'CustomOriginConfig': {
                      'HTTPPort': 80,
                      'HTTPSPort': 443,
                      'OriginKeepaliveTimeout': 5,
                      'OriginProtocolPolicy': 'https-only',
                      'OriginReadTimeout': 30,
                      'OriginSSLProtocols': [
                        'TLSv1.2',
                      ],
                    },
                    'ConnectionAttempts': 3,
                    'ConnectionTimeout': 10,
                    'DomainName': 'myorigin.com',
                    'Id': 'origin1',
                    'OriginCustomHeaders': [
                      {
                        'HeaderName': 'X-Custom-Header',
                        'HeaderValue': 'somevalue',
                      },
                    ],
                    'OriginShield': {
                      'Enabled': true,
                      'OriginShieldRegion': 'us-east-1',
                    },
                  },
                ],
                'PriceClass': 'PriceClass_100',
                'ViewerCertificate': {
                  'CloudFrontDefaultCertificate': true,
                },
              },
            },
          },
        },
      },
    );
  });

  test('most basic distribution', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    const dist = new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Bucket83908E77': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
          'Type': 'AWS::CloudFront::Distribution',
          'Properties': {
            'DistributionConfig': {
              'DefaultRootObject': 'index.html',
              'Origins': [
                {
                  'ConnectionAttempts': 3,
                  'ConnectionTimeout': 10,
                  'DomainName': {
                    'Fn::GetAtt': [
                      'Bucket83908E77',
                      'RegionalDomainName',
                    ],
                  },
                  'Id': 'origin1',
                  'S3OriginConfig': {},
                },
              ],
              'ViewerCertificate': {
                'CloudFrontDefaultCertificate': true,
              },
              'PriceClass': 'PriceClass_100',
              'DefaultCacheBehavior': {
                'AllowedMethods': [
                  'GET',
                  'HEAD',
                ],
                'CachedMethods': [
                  'GET',
                  'HEAD',
                ],
                'TargetOriginId': 'origin1',
                'ViewerProtocolPolicy': 'redirect-to-https',
                'ForwardedValues': {
                  'QueryString': false,
                  'Cookies': { 'Forward': 'none' },
                },
                'Compress': true,
              },
              'Enabled': true,
              'IPV6Enabled': true,
              'HttpVersion': 'http2',
            },
          },
        },
      },
    });

    expect(dist.distributionArn).toEqual(`arn:${cdk.Aws.PARTITION}:cloudfront::${cdk.Aws.ACCOUNT_ID}:distribution/${dist.distributionId}`);
  });

  test('can disable distribution', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      enabled: false,
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Bucket83908E77': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
          'Type': 'AWS::CloudFront::Distribution',
          'Properties': {
            'DistributionConfig': {
              'DefaultRootObject': 'index.html',
              'Origins': [
                {
                  'ConnectionAttempts': 3,
                  'ConnectionTimeout': 10,
                  'DomainName': {
                    'Fn::GetAtt': [
                      'Bucket83908E77',
                      'RegionalDomainName',
                    ],
                  },
                  'Id': 'origin1',
                  'S3OriginConfig': {},
                },
              ],
              'ViewerCertificate': {
                'CloudFrontDefaultCertificate': true,
              },
              'PriceClass': 'PriceClass_100',
              'DefaultCacheBehavior': {
                'AllowedMethods': [
                  'GET',
                  'HEAD',
                ],
                'CachedMethods': [
                  'GET',
                  'HEAD',
                ],
                'TargetOriginId': 'origin1',
                'ViewerProtocolPolicy': 'redirect-to-https',
                'ForwardedValues': {
                  'QueryString': false,
                  'Cookies': { 'Forward': 'none' },
                },
                'Compress': true,
              },
              'Enabled': false,
              'IPV6Enabled': true,
              'HttpVersion': 'http2',
            },
          },
        },
      },
    });
  });

  test('ensure long comments will not break the distribution', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      comment: `Adding a comment longer than 128 characters should be trimmed and
added the ellipsis so a user would know there was more to read and everything beyond this point should not show up`,
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      Resources: {
        Bucket83908E77: {
          Type: 'AWS::S3::Bucket',
          DeletionPolicy: 'Retain',
          UpdateReplacePolicy: 'Retain',
        },
        AnAmazingWebsiteProbablyCFDistribution47E3983B: {
          Type: 'AWS::CloudFront::Distribution',
          Properties: {
            DistributionConfig: {
              DefaultRootObject: 'index.html',
              Origins: [
                {
                  ConnectionAttempts: 3,
                  ConnectionTimeout: 10,
                  DomainName: {
                    'Fn::GetAtt': ['Bucket83908E77', 'RegionalDomainName'],
                  },
                  Id: 'origin1',
                  S3OriginConfig: {},
                },
              ],
              ViewerCertificate: {
                CloudFrontDefaultCertificate: true,
              },
              PriceClass: 'PriceClass_100',
              DefaultCacheBehavior: {
                AllowedMethods: ['GET', 'HEAD'],
                CachedMethods: ['GET', 'HEAD'],
                TargetOriginId: 'origin1',
                ViewerProtocolPolicy: 'redirect-to-https',
                ForwardedValues: {
                  QueryString: false,
                  Cookies: { Forward: 'none' },
                },
                Compress: true,
              },
              Comment: `Adding a comment longer than 128 characters should be trimmed and
added the ellipsis so a user would know there was more to r...`,
              Enabled: true,
              IPV6Enabled: true,
              HttpVersion: 'http2',
            },
          },
        },
      },
    });
  });

  test('distribution with bucket and OAI', () => {
    const stack = new cdk.Stack();
    const s3BucketSource = new s3.Bucket(stack, 'Bucket');
    const originAccessIdentity = new OriginAccessIdentity(stack, 'OAI');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [{
        s3OriginSource: { s3BucketSource, originAccessIdentity },
        behaviors: [{ isDefaultBehavior: true }],
      }],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Origins: [
          {
            ConnectionAttempts: 3,
            ConnectionTimeout: 10,
            DomainName: {
              'Fn::GetAtt': [
                'Bucket83908E77',
                'RegionalDomainName',
              ],
            },
            Id: 'origin1',
            S3OriginConfig: {
              OriginAccessIdentity: {
                'Fn::Join': ['', ['origin-access-identity/cloudfront/', { Ref: 'OAIE1EFC67F' }]],
              },
            },
          },
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Statement: [{
          Action: 's3:GetObject',
          Effect: 'Allow',
          Principal: {
            CanonicalUser: { 'Fn::GetAtt': ['OAIE1EFC67F', 'S3CanonicalUserId'] },
          },
          Resource: {
            'Fn::Join': ['', [{ 'Fn::GetAtt': ['Bucket83908E77', 'Arn'] }, '/*']],
          },
        }],
      },
    });
  });

  testDeprecated('distribution with trusted signers on default distribution', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');
    const pubKey = new PublicKey(stack, 'MyPubKey', {
      encodedKey: publicKey,
    });
    const keyGroup = new KeyGroup(stack, 'MyKeyGroup', {
      items: [
        pubKey,
      ],
    });

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              trustedSigners: ['1234'],
              trustedKeyGroups: [
                keyGroup,
              ],
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Bucket83908E77': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'MyPubKey6ADA4CF5': {
          'Type': 'AWS::CloudFront::PublicKey',
          'Properties': {
            'PublicKeyConfig': {
              'CallerReference': 'c8141e732ea37b19375d4cbef2b2d2c6f613f0649a',
              'EncodedKey': publicKey,
              'Name': 'MyPubKey',
            },
          },
        },
        'MyKeyGroupAF22FD35': {
          'Type': 'AWS::CloudFront::KeyGroup',
          'Properties': {
            'KeyGroupConfig': {
              'Items': [
                {
                  'Ref': 'MyPubKey6ADA4CF5',
                },
              ],
              'Name': 'MyKeyGroup',
            },
          },
        },
        'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
          'Type': 'AWS::CloudFront::Distribution',
          'Properties': {
            'DistributionConfig': {
              'DefaultRootObject': 'index.html',
              'Origins': [
                {
                  'ConnectionAttempts': 3,
                  'ConnectionTimeout': 10,
                  'DomainName': {
                    'Fn::GetAtt': [
                      'Bucket83908E77',
                      'RegionalDomainName',
                    ],
                  },
                  'Id': 'origin1',
                  'S3OriginConfig': {},
                },
              ],
              'ViewerCertificate': {
                'CloudFrontDefaultCertificate': true,
              },
              'PriceClass': 'PriceClass_100',
              'DefaultCacheBehavior': {
                'AllowedMethods': [
                  'GET',
                  'HEAD',
                ],
                'CachedMethods': [
                  'GET',
                  'HEAD',
                ],
                'TargetOriginId': 'origin1',
                'ViewerProtocolPolicy': 'redirect-to-https',
                'ForwardedValues': {
                  'QueryString': false,
                  'Cookies': { 'Forward': 'none' },
                },
                'TrustedKeyGroups': [
                  {
                    'Ref': 'MyKeyGroupAF22FD35',
                  },
                ],
                'TrustedSigners': ['1234'],
                'Compress': true,
              },
              'Enabled': true,
              'IPV6Enabled': true,
              'HttpVersion': 'http2',
            },
          },
        },
      },
    });
  });

  test('distribution with ViewerProtocolPolicy set to a non-default value', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      viewerProtocolPolicy: ViewerProtocolPolicy.ALLOW_ALL,
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Bucket83908E77': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
          'Type': 'AWS::CloudFront::Distribution',
          'Properties': {
            'DistributionConfig': {
              'DefaultRootObject': 'index.html',
              'Origins': [
                {
                  'ConnectionAttempts': 3,
                  'ConnectionTimeout': 10,
                  'DomainName': {
                    'Fn::GetAtt': [
                      'Bucket83908E77',
                      'RegionalDomainName',
                    ],
                  },
                  'Id': 'origin1',
                  'S3OriginConfig': {},
                },
              ],
              'ViewerCertificate': {
                'CloudFrontDefaultCertificate': true,
              },
              'PriceClass': 'PriceClass_100',
              'DefaultCacheBehavior': {
                'AllowedMethods': [
                  'GET',
                  'HEAD',
                ],
                'CachedMethods': [
                  'GET',
                  'HEAD',
                ],
                'TargetOriginId': 'origin1',
                'ViewerProtocolPolicy': 'allow-all',
                'ForwardedValues': {
                  'QueryString': false,
                  'Cookies': { 'Forward': 'none' },
                },
                'Compress': true,
              },
              'Enabled': true,
              'IPV6Enabled': true,
              'HttpVersion': 'http2',
            },
          },
        },
      },
    });
  });

  test('distribution with ViewerProtocolPolicy overridden in Behavior', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      viewerProtocolPolicy: ViewerProtocolPolicy.ALLOW_ALL,
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
            },
            {
              pathPattern: '/test/*',
              viewerProtocolPolicy: ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Bucket83908E77': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
          'Type': 'AWS::CloudFront::Distribution',
          'Properties': {
            'DistributionConfig': {
              'CacheBehaviors': [
                {
                  'AllowedMethods': [
                    'GET',
                    'HEAD',
                  ],
                  'CachedMethods': [
                    'GET',
                    'HEAD',
                  ],
                  'Compress': true,
                  'ForwardedValues': {
                    'Cookies': {
                      'Forward': 'none',
                    },
                    'QueryString': false,
                  },
                  'PathPattern': '/test/*',
                  'TargetOriginId': 'origin1',
                  'ViewerProtocolPolicy': 'redirect-to-https',
                },
              ],
              'DefaultRootObject': 'index.html',
              'Origins': [
                {
                  'ConnectionAttempts': 3,
                  'ConnectionTimeout': 10,
                  'DomainName': {
                    'Fn::GetAtt': [
                      'Bucket83908E77',
                      'RegionalDomainName',
                    ],
                  },
                  'Id': 'origin1',
                  'S3OriginConfig': {},
                },
              ],
              'ViewerCertificate': {
                'CloudFrontDefaultCertificate': true,
              },
              'PriceClass': 'PriceClass_100',
              'DefaultCacheBehavior': {
                'AllowedMethods': [
                  'GET',
                  'HEAD',
                ],
                'CachedMethods': [
                  'GET',
                  'HEAD',
                ],
                'TargetOriginId': 'origin1',
                'ViewerProtocolPolicy': 'allow-all',
                'ForwardedValues': {
                  'QueryString': false,
                  'Cookies': { 'Forward': 'none' },
                },
                'Compress': true,
              },
              'Enabled': true,
              'IPV6Enabled': true,
              'HttpVersion': 'http2',
            },
          },
        },
      },
    });
  });

  test('distribution with disabled compression', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              compress: false,
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'Bucket83908E77': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
        'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
          'Type': 'AWS::CloudFront::Distribution',
          'Properties': {
            'DistributionConfig': {
              'DefaultRootObject': 'index.html',
              'Origins': [
                {
                  'ConnectionAttempts': 3,
                  'ConnectionTimeout': 10,
                  'DomainName': {
                    'Fn::GetAtt': [
                      'Bucket83908E77',
                      'RegionalDomainName',
                    ],
                  },
                  'Id': 'origin1',
                  'S3OriginConfig': {},
                },
              ],
              'ViewerCertificate': {
                'CloudFrontDefaultCertificate': true,
              },
              'PriceClass': 'PriceClass_100',
              'DefaultCacheBehavior': {
                'AllowedMethods': [
                  'GET',
                  'HEAD',
                ],
                'CachedMethods': [
                  'GET',
                  'HEAD',
                ],
                'TargetOriginId': 'origin1',
                'ViewerProtocolPolicy': 'redirect-to-https',
                'ForwardedValues': {
                  'QueryString': false,
                  'Cookies': { 'Forward': 'none' },
                },
                'Compress': false,
              },
              'Enabled': true,
              'IPV6Enabled': true,
              'HttpVersion': 'http2',
            },
          },
        },
      },
    });
  });

  test('distribution with CloudFront function-association', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              functionAssociations: [{
                eventType: FunctionEventType.VIEWER_REQUEST,
                function: new Function(stack, 'TestFunction', {
                  code: FunctionCode.fromInline('foo'),
                }),
              }],
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      'DistributionConfig': {
        'DefaultCacheBehavior': {
          'FunctionAssociations': [
            {
              'EventType': 'viewer-request',
              'FunctionARN': {
                'Fn::GetAtt': [
                  'TestFunction22AD90FC',
                  'FunctionARN',
                ],
              },
            },
          ],
        },
      },
    });
  });

  test('distribution with resolvable lambda-association', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    const lambdaFunction = new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              lambdaFunctionAssociations: [{
                eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
                lambdaFunction: lambdaFunction.currentVersion,
                includeBody: true,
              }],
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      'DistributionConfig': {
        'DefaultCacheBehavior': {
          'LambdaFunctionAssociations': [
            {
              'EventType': 'origin-request',
              'IncludeBody': true,
              'LambdaFunctionARN': {
                'Ref': Match.stringLikeRegexp(stack.getLogicalId(lambdaFunction.currentVersion.node.defaultChild as lambda.CfnVersion)),
              },
            },
          ],
        },
      },
    });
  });

  test('associate a lambda with removable env vars', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    const lambdaFunction = new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    lambdaFunction.addEnvironment('KEY', 'value', { removeInEdge: true });

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              lambdaFunctionAssociations: [{
                eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
                lambdaFunction: lambdaFunction.currentVersion,
              }],
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Environment: Match.absent(),
    });
  });

  test('throws when associating a lambda with incompatible env vars', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    const lambdaFunction = new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
      environment: {
        KEY: 'value',
      },
    });

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [
            {
              isDefaultBehavior: true,
              lambdaFunctionAssociations: [{
                eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
                lambdaFunction: lambdaFunction.currentVersion,
              }],
            },
          ],
        },
      ],
    });

    expect(() => app.synth()).toThrow(/KEY/);
  });

  test('throws when associating a lambda with includeBody and a response event type', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'Stack');
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    const fnVersion = lambda.Version.fromVersionArn(stack, 'Version', 'arn:aws:lambda:testregion:111111111111:function:myTestFun:v1');

    expect(() => {
      new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
        originConfigs: [
          {
            s3OriginSource: {
              s3BucketSource: sourceBucket,
            },
            behaviors: [
              {
                isDefaultBehavior: true,
                lambdaFunctionAssociations: [{
                  eventType: LambdaEdgeEventType.VIEWER_RESPONSE,
                  includeBody: true,
                  lambdaFunction: fnVersion,
                }],
              },
            ],
          },
        ],
      });
    }).toThrow(/'includeBody' can only be true for ORIGIN_REQUEST or VIEWER_REQUEST event types./);
  });

  test('distribution has a defaultChild', () => {
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');

    const distribution = new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs: [
        {
          s3OriginSource: {
            s3BucketSource: sourceBucket,
          },
          behaviors: [{ isDefaultBehavior: true }],
        },
      ],
    });

    expect(distribution.node.defaultChild instanceof CfnDistribution).toEqual(true);
  });

  testDeprecated('allows multiple aliasConfiguration CloudFrontWebDistribution per stack', () => {
    const stack = new cdk.Stack();
    const s3BucketSource = new s3.Bucket(stack, 'Bucket');

    const originConfigs = [{
      s3OriginSource: { s3BucketSource },
      behaviors: [{ isDefaultBehavior: true }],
    }];

    new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
      originConfigs,
      aliasConfiguration: { acmCertRef: 'acm_ref', names: ['www.example.com'] },
    });
    new CloudFrontWebDistribution(stack, 'AnotherAmazingWebsiteProbably', {
      originConfigs,
      aliasConfiguration: { acmCertRef: 'another_acm_ref', names: ['ftp.example.com'] },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      'DistributionConfig': {
        'Aliases': ['www.example.com'],
        'ViewerCertificate': {
          'AcmCertificateArn': 'acm_ref',
          'SslSupportMethod': 'sni-only',
        },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      'DistributionConfig': {
        'Aliases': ['ftp.example.com'],
        'ViewerCertificate': {
          'AcmCertificateArn': 'another_acm_ref',
          'SslSupportMethod': 'sni-only',
        },
      },
    });
  });

  describe('viewerCertificate', () => {
    describe('acmCertificate', () => {
      test('base usage', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        const certificate = new certificatemanager.Certificate(stack, 'cert', {
          domainName: 'example.com',
        });

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromAcmCertificate(certificate),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': [],
            'ViewerCertificate': {
              'AcmCertificateArn': {
                'Ref': 'cert56CA94EB',
              },
              'SslSupportMethod': 'sni-only',
            },
          },
        });
      });
      test('imported certificate fromCertificateArn', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        const certificate = certificatemanager.Certificate.fromCertificateArn(
          stack, 'cert', 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d',
        );

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromAcmCertificate(certificate),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': [],
            'ViewerCertificate': {
              'AcmCertificateArn': 'arn:aws:acm:us-east-1:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d',
              'SslSupportMethod': 'sni-only',
            },
          },
        });
      });
      test('advanced usage', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        const certificate = new certificatemanager.Certificate(stack, 'cert', {
          domainName: 'example.com',
        });

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromAcmCertificate(certificate, {
            securityPolicy: SecurityPolicyProtocol.SSL_V3,
            sslMethod: SSLMethod.VIP,
            aliases: ['example.com', 'www.example.com'],
          }),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': ['example.com', 'www.example.com'],
            'ViewerCertificate': {
              'AcmCertificateArn': {
                'Ref': 'cert56CA94EB',
              },
              'MinimumProtocolVersion': 'SSLv3',
              'SslSupportMethod': 'vip',
            },
          },
        });
      });
    });
    describe('iamCertificate', () => {
      test('base usage', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromIamCertificate('test'),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': [],
            'ViewerCertificate': {
              'IamCertificateId': 'test',
              'SslSupportMethod': 'sni-only',
            },
          },
        });
      });
      test('advanced usage', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromIamCertificate('test', {
            securityPolicy: SecurityPolicyProtocol.TLS_V1,
            sslMethod: SSLMethod.VIP,
            aliases: ['example.com'],
          }),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': ['example.com'],
            'ViewerCertificate': {
              'IamCertificateId': 'test',
              'MinimumProtocolVersion': 'TLSv1',
              'SslSupportMethod': 'vip',
            },
          },
        });
      });
    });
    describe('cloudFrontDefaultCertificate', () => {
      test('base usage', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromCloudFrontDefaultCertificate(),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': [],
            'ViewerCertificate': {
              'CloudFrontDefaultCertificate': true,
            },
          },
        });
      });
      test('aliases are set', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromCloudFrontDefaultCertificate('example.com', 'www.example.com'),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': ['example.com', 'www.example.com'],
            'ViewerCertificate': {
              'CloudFrontDefaultCertificate': true,
            },
          },
        });
      });
    });
    describe('errors', () => {
      testDeprecated('throws if both deprecated aliasConfiguration and viewerCertificate', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        expect(() => {
          new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
            originConfigs: [{
              s3OriginSource: { s3BucketSource: sourceBucket },
              behaviors: [{ isDefaultBehavior: true }],
            }],
            aliasConfiguration: { acmCertRef: 'test', names: ['ftp.example.com'] },
            viewerCertificate: ViewerCertificate.fromCloudFrontDefaultCertificate('example.com', 'www.example.com'),
          });
        }).toThrow(/You cannot set both aliasConfiguration and viewerCertificate properties/);
      });
      test('throws if invalid security policy for SSL method', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        expect(() => {
          new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
            originConfigs: [{
              s3OriginSource: { s3BucketSource: sourceBucket },
              behaviors: [{ isDefaultBehavior: true }],
            }],
            viewerCertificate: ViewerCertificate.fromIamCertificate('test', {
              securityPolicy: SecurityPolicyProtocol.TLS_V1_1_2016,
              sslMethod: SSLMethod.VIP,
            }),
          });
        }).toThrow(/TLSv1.1_2016 is not compabtible with sslMethod vip./);
      });
      // FIXME https://github.com/aws/aws-cdk/issues/4724
      test('does not throw if acmCertificate explicitly not in us-east-1', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        const certificate = certificatemanager.Certificate.fromCertificateArn(
          stack, 'cert', 'arn:aws:acm:eu-west-3:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d',
        );

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          viewerCertificate: ViewerCertificate.fromAcmCertificate(certificate),
        });

        Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
          'DistributionConfig': {
            'Aliases': [],
            'ViewerCertificate': {
              'AcmCertificateArn': 'arn:aws:acm:eu-west-3:111111111111:certificate/11-3336f1-44483d-adc7-9cd375c5169d',
              'SslSupportMethod': 'sni-only',
            },
          },
        });
      });
    });
  });

  test('edgelambda.amazonaws.com is added to the trust policy of lambda', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');
    const fn = new lambda.Function(stack, 'Lambda', {
      code: lambda.Code.fromInline('foo'),
      handler: 'index.handler',
      runtime: lambda.Runtime.NODEJS_LATEST,
    });
    const lambdaVersion = new lambda.Version(stack, 'LambdaVersion', { lambda: fn });

    // WHEN
    new CloudFrontWebDistribution(stack, 'MyDistribution', {
      originConfigs: [
        {
          s3OriginSource: { s3BucketSource: sourceBucket },
          behaviors: [
            {
              isDefaultBehavior: true,
              lambdaFunctionAssociations: [
                {
                  eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
                  lambdaFunction: lambdaVersion,
                },
              ],
            },
          ],
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        'Statement': [
          {
            'Action': 'sts:AssumeRole',
            'Effect': 'Allow',
            'Principal': {
              'Service': 'lambda.amazonaws.com',
            },
          },
          {
            'Action': 'sts:AssumeRole',
            'Effect': 'Allow',
            'Principal': {
              'Service': 'edgelambda.amazonaws.com',
            },
          },
        ],
        'Version': '2012-10-17',
      },
    });
  });

  test('edgelambda.amazonaws.com is not added to lambda role for imported functions', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const sourceBucket = new s3.Bucket(stack, 'Bucket');
    const lambdaVersion = lambda.Version.fromVersionArn(stack, 'Version', 'arn:aws:lambda:function-region:111111111111:function:function-name');

    // WHEN
    new CloudFrontWebDistribution(stack, 'MyDistribution', {
      originConfigs: [
        {
          s3OriginSource: { s3BucketSource: sourceBucket },
          behaviors: [
            {
              isDefaultBehavior: true,
              lambdaFunctionAssociations: [
                {
                  eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
                  lambdaFunction: lambdaVersion,
                },
              ],
            },
          ],
        },
      ],
    });

    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 0);
  });

  describe('geo restriction', () => {
    describe('success', () => {
      test('allowlist', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          geoRestriction: GeoRestriction.allowlist('US', 'GB'),
        });

        Template.fromStack(stack).templateMatches({
          'Resources': {
            'Bucket83908E77': {
              'Type': 'AWS::S3::Bucket',
              'DeletionPolicy': 'Retain',
              'UpdateReplacePolicy': 'Retain',
            },
            'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
              'Type': 'AWS::CloudFront::Distribution',
              'Properties': {
                'DistributionConfig': {
                  'DefaultRootObject': 'index.html',
                  'Origins': [
                    {
                      'ConnectionAttempts': 3,
                      'ConnectionTimeout': 10,
                      'DomainName': {
                        'Fn::GetAtt': [
                          'Bucket83908E77',
                          'RegionalDomainName',
                        ],
                      },
                      'Id': 'origin1',
                      'S3OriginConfig': {},
                    },
                  ],
                  'ViewerCertificate': {
                    'CloudFrontDefaultCertificate': true,
                  },
                  'PriceClass': 'PriceClass_100',
                  'DefaultCacheBehavior': {
                    'AllowedMethods': [
                      'GET',
                      'HEAD',
                    ],
                    'CachedMethods': [
                      'GET',
                      'HEAD',
                    ],
                    'TargetOriginId': 'origin1',
                    'ViewerProtocolPolicy': 'redirect-to-https',
                    'ForwardedValues': {
                      'QueryString': false,
                      'Cookies': { 'Forward': 'none' },
                    },
                    'Compress': true,
                  },
                  'Enabled': true,
                  'IPV6Enabled': true,
                  'HttpVersion': 'http2',
                  'Restrictions': {
                    'GeoRestriction': {
                      'Locations': ['US', 'GB'],
                      'RestrictionType': 'whitelist',
                    },
                  },
                },
              },
            },
          },
        });
      });
      test('denylist', () => {
        const stack = new cdk.Stack();
        const sourceBucket = new s3.Bucket(stack, 'Bucket');

        new CloudFrontWebDistribution(stack, 'AnAmazingWebsiteProbably', {
          originConfigs: [{
            s3OriginSource: { s3BucketSource: sourceBucket },
            behaviors: [{ isDefaultBehavior: true }],
          }],
          geoRestriction: GeoRestriction.denylist('US'),
        });

        Template.fromStack(stack).templateMatches({
          'Resources': {
            'Bucket83908E77': {
              'Type': 'AWS::S3::Bucket',
              'DeletionPolicy': 'Retain',
              'UpdateReplacePolicy': 'Retain',
            },
            'AnAmazingWebsiteProbablyCFDistribution47E3983B': {
              'Type': 'AWS::CloudFront::Distribution',
              'Properties': {
                'DistributionConfig': {
                  'DefaultRootObject': 'index.html',
                  'Origins': [
                    {
                      'ConnectionAttempts': 3,
                      'ConnectionTimeout': 10,
                      'DomainName': {
                        'Fn::GetAtt': [
                          'Bucket83908E77',
                          'RegionalDomainName',
                        ],
                      },
                      'Id': 'origin1',
                      'S3OriginConfig': {},
                    },
                  ],
                  'ViewerCertificate': {
                    'CloudFrontDefaultCertificate': true,
                  },
                  'PriceClass': 'PriceClass_100',
                  'DefaultCacheBehavior': {
                    'AllowedMethods': [
                      'GET',
                      'HEAD',
                    ],
                    'CachedMethods': [
                      'GET',
                      'HEAD',
                    ],
                    'TargetOriginId': 'origin1',
                    'ViewerProtocolPolicy': 'redirect-to-https',
                    'ForwardedValues': {
                      'QueryString': false,
                      'Cookies': { 'Forward': 'none' },
                    },
                    'Compress': true,
                  },
                  'Enabled': true,
                  'IPV6Enabled': true,
                  'HttpVersion': 'http2',
                  'Restrictions': {
                    'GeoRestriction': {
                      'Locations': ['US'],
                      'RestrictionType': 'blacklist',
                    },
                  },
                },
              },
            },
          },
        });
      });
    });
    describe('error', () => {
      test('throws if locations is empty array', () => {
        expect(() => {
          GeoRestriction.allowlist();
        }).toThrow(/Should provide at least 1 location/);

        expect(() => {
          GeoRestriction.denylist();
        }).toThrow(/Should provide at least 1 location/);
      });
      test('throws if locations format is wrong', () => {
        expect(() => {
          GeoRestriction.allowlist('us');
        }).toThrow(/Invalid location format for location: us, location should be two-letter and uppercase country ISO 3166-1-alpha-2 code/);

        expect(() => {
          GeoRestriction.denylist('us');
        }).toThrow(/Invalid location format for location: us, location should be two-letter and uppercase country ISO 3166-1-alpha-2 code/);
      });
    });
  });

  describe('Connection behaviors between CloudFront and your origin', () => {
    describe('success', () => {
      test('connectionAttempts = 1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionAttempts: 1,
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).not.toThrow(/connectionAttempts: You can specify 1, 2, or 3 as the number of attempts./);
      });
      test('3 = connectionAttempts', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionAttempts: 3,
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).not.toThrow(/connectionAttempts: You can specify 1, 2, or 3 as the number of attempts./);
      });
      test('connectionTimeout = 1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionTimeout: cdk.Duration.seconds(1),
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).not.toThrow(/connectionTimeout: You can specify a number of seconds between 1 and 10 (inclusive)./);
      });
      test('10 = connectionTimeout', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionTimeout: cdk.Duration.seconds(10),
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).not.toThrow(/connectionTimeout: You can specify a number of seconds between 1 and 10 (inclusive)./);
      });
    });
    describe('errors', () => {
      test('connectionAttempts = 1.1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionAttempts: 1.1,
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/connectionAttempts: You can specify 1, 2, or 3 as the number of attempts./);
      });
      test('connectionAttempts = -1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionAttempts: -1,
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/connectionAttempts: You can specify 1, 2, or 3 as the number of attempts./);
      });
      test('connectionAttempts < 1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionAttempts: 0,
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/connectionAttempts: You can specify 1, 2, or 3 as the number of attempts./);
      });
      test('3 < connectionAttempts', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionAttempts: 4,
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/connectionAttempts: You can specify 1, 2, or 3 as the number of attempts./);
      });
      test('connectionTimeout = 1.1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionTimeout: cdk.Duration.seconds(1.1),
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/must be a whole number of/);
      });
      test('connectionTimeout < 1', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionTimeout: cdk.Duration.seconds(0),
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/connectionTimeout: You can specify a number of seconds between 1 and 10 \(inclusive\)./);
      });
      test('10 < connectionTimeout', () => {
        const stack = new cdk.Stack();
        expect(() => {
          new CloudFrontWebDistribution(stack, 'Distribution', {
            originConfigs: [{
              behaviors: [{ isDefaultBehavior: true }],
              connectionTimeout: cdk.Duration.seconds(11),
              customOriginSource: { domainName: 'myorigin.com' },
            }],
          });
        }).toThrow(/connectionTimeout: You can specify a number of seconds between 1 and 10 \(inclusive\)./);
      });
    });
  });

  test('existing distributions can be imported', () => {
    const stack = new cdk.Stack();
    const dist = CloudFrontWebDistribution.fromDistributionAttributes(stack, 'ImportedDist', {
      domainName: 'd111111abcdef8.cloudfront.net',
      distributionId: '012345ABCDEF',
    });

    expect(dist.distributionDomainName).toEqual('d111111abcdef8.cloudfront.net');
    expect(dist.distributionId).toEqual('012345ABCDEF');
    expect(dist.distributionArn).toEqual(`arn:${cdk.Aws.PARTITION}:cloudfront::${cdk.Aws.ACCOUNT_ID}:distribution/012345ABCDEF`);
  });
});

test('grants custom actions', () => {
  const stack = new cdk.Stack();
  const distribution = new CloudFrontWebDistribution(stack, 'Distribution', {
    originConfigs: [{
      customOriginSource: { domainName: 'myorigin.com' },
      behaviors: [{ isDefaultBehavior: true }],
    }],
  });
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.AccountRootPrincipal(),
  });
  distribution.grant(role, 'cloudfront:ListInvalidations', 'cloudfront:GetInvalidation');

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: [
            'cloudfront:ListInvalidations',
            'cloudfront:GetInvalidation',
          ],
          Resource: {
            'Fn::Join': [
              '', [
                'arn:', { Ref: 'AWS::Partition' }, ':cloudfront::', { Ref: 'AWS::AccountId' }, ':distribution/',
                { Ref: 'DistributionCFDistribution882A7313' },
              ],
            ],
          },
        },
      ],
    },
  });
});

test('grants createInvalidation', () => {
  const stack = new cdk.Stack();
  const distribution = new CloudFrontWebDistribution(stack, 'Distribution', {
    originConfigs: [{
      customOriginSource: { domainName: 'myorigin.com' },
      behaviors: [{ isDefaultBehavior: true }],
    }],
  });
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.AccountRootPrincipal(),
  });
  distribution.grantCreateInvalidation(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: 'cloudfront:CreateInvalidation',
          Resource: {
            'Fn::Join': [
              '', [
                'arn:', { Ref: 'AWS::Partition' }, ':cloudfront::', { Ref: 'AWS::AccountId' }, ':distribution/',
                { Ref: 'DistributionCFDistribution882A7313' },
              ],
            ],
          },
        },
      ],
    },
  });
});
