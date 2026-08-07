import { defaultOrigin, defaultOriginGroup, defaultOriginWithOriginAccessControl } from './test-origin';
import { Annotations, Match, Template } from '../../assertions';
import * as acm from '../../aws-certificatemanager';
import * as cloudwatch from '../../aws-cloudwatch';
import * as iam from '../../aws-iam';
import * as kinesis from '../../aws-kinesis';
import * as lambda from '../../aws-lambda';
import * as s3 from '../../aws-s3';
import { App, Aws, Duration, Stack, Token, Validations } from '../../core';
import type {
  CfnDistribution,
  IOrigin,
} from '../lib';
import {
  AllowedMethods,
  Distribution,
  Endpoint,
  Function,
  FunctionCode,
  FunctionEventType,
  GeoRestriction,
  HttpVersion,
  LambdaEdgeEventType,
  PriceClass,
  RealtimeLogConfig,
  SecurityPolicyProtocol,
  SSLMethod,
} from '../lib';
import { DistributionGrants } from '../lib/cloudfront-grants.generated';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'Stack', {
    env: { account: '1234', region: 'testregion' },
  });
});

test('minimal example renders correctly', () => {
  const origin = defaultOrigin();
  const dist = new Distribution(stack, 'MyDist', { defaultBehavior: { origin } });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Enabled: true,
      HttpVersion: 'http2',
      IPV6Enabled: true,
      Origins: [{
        DomainName: 'www.example.com',
        Id: 'StackMyDistOrigin1D6D5E535',
        CustomOriginConfig: {
          OriginProtocolPolicy: 'https-only',
        },
      }],
    },
  });

  expect(dist.distributionArn).toEqual(`arn:${Aws.PARTITION}:cloudfront::1234:distribution/${dist.distributionId}`);
});

test('distribution without additional behaviors or origin groups omits those properties', () => {
  const origin = defaultOrigin();
  new Distribution(stack, 'MyDist', { defaultBehavior: { origin } });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      CacheBehaviors: Match.absent(),
      OriginGroups: Match.absent(),
    },
  });
});

test('existing distributions can be imported', () => {
  const dist = Distribution.fromDistributionAttributes(stack, 'ImportedDist', {
    domainName: 'd111111abcdef8.cloudfront.net',
    distributionId: '012345ABCDEF',
  });

  expect(dist.distributionDomainName).toEqual('d111111abcdef8.cloudfront.net');
  expect(dist.distributionId).toEqual('012345ABCDEF');
  expect(dist.distributionArn).toEqual(`arn:${Aws.PARTITION}:cloudfront::1234:distribution/012345ABCDEF`);
});

test('exhaustive example of props renders correctly and SSL method sni-only', () => {
  const origin = defaultOrigin();
  const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

  new Distribution(stack, 'MyDist', {
    defaultBehavior: { origin },
    certificate,
    comment: 'a test',
    defaultRootObject: 'index.html',
    domainNames: ['example.com'],
    enabled: false,
    enableIpv6: false,
    enableLogging: true,
    geoRestriction: GeoRestriction.denylist('US', 'GB'),
    httpVersion: HttpVersion.HTTP1_1,
    logFilePrefix: 'logs/',
    logIncludesCookies: true,
    sslSupportMethod: SSLMethod.SNI,
    minimumProtocolVersion: SecurityPolicyProtocol.TLS_V1_2_2019,
    priceClass: PriceClass.PRICE_CLASS_100,
    webAclId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Aliases: ['example.com'],
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Comment: 'a test',
      DefaultRootObject: 'index.html',
      Enabled: false,
      HttpVersion: 'http1.1',
      IPV6Enabled: false,
      Logging: {
        Bucket: { 'Fn::GetAtt': ['MyDistLoggingBucket9B8976BC', 'RegionalDomainName'] },
        IncludeCookies: true,
        Prefix: 'logs/',
      },
      Origins: [{
        DomainName: 'www.example.com',
        Id: 'StackMyDistOrigin1D6D5E535',
        CustomOriginConfig: {
          OriginProtocolPolicy: 'https-only',
        },
      }],
      PriceClass: 'PriceClass_100',
      Restrictions: {
        GeoRestriction: {
          Locations: ['US', 'GB'],
          RestrictionType: 'blacklist',
        },
      },
      ViewerCertificate: {
        AcmCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012',
        SslSupportMethod: 'sni-only',
        MinimumProtocolVersion: 'TLSv1.2_2019',
      },
      WebACLId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
    },
  });
});

test('exhaustive example of props renders correctly and SSL method vip', () => {
  const origin = defaultOrigin();
  const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

  new Distribution(stack, 'MyDist', {
    defaultBehavior: { origin },
    certificate,
    comment: 'a test',
    defaultRootObject: 'index.html',
    domainNames: ['example.com'],
    enabled: false,
    enableIpv6: false,
    enableLogging: true,
    geoRestriction: GeoRestriction.denylist('US', 'GB'),
    httpVersion: HttpVersion.HTTP1_1,
    logFilePrefix: 'logs/',
    logIncludesCookies: true,
    sslSupportMethod: SSLMethod.VIP,
    minimumProtocolVersion: SecurityPolicyProtocol.TLS_V1_2_2019,
    priceClass: PriceClass.PRICE_CLASS_100,
    webAclId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Aliases: ['example.com'],
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Comment: 'a test',
      DefaultRootObject: 'index.html',
      Enabled: false,
      HttpVersion: 'http1.1',
      IPV6Enabled: false,
      Logging: {
        Bucket: { 'Fn::GetAtt': ['MyDistLoggingBucket9B8976BC', 'RegionalDomainName'] },
        IncludeCookies: true,
        Prefix: 'logs/',
      },
      Origins: [{
        DomainName: 'www.example.com',
        Id: 'StackMyDistOrigin1D6D5E535',
        CustomOriginConfig: {
          OriginProtocolPolicy: 'https-only',
        },
      }],
      PriceClass: 'PriceClass_100',
      Restrictions: {
        GeoRestriction: {
          Locations: ['US', 'GB'],
          RestrictionType: 'blacklist',
        },
      },
      ViewerCertificate: {
        AcmCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012',
        SslSupportMethod: 'vip',
        MinimumProtocolVersion: 'TLSv1.2_2019',
      },
      WebACLId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
    },
  });
});

test('exhaustive example of props renders correctly and SSL method default', () => {
  const origin = defaultOrigin();
  const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

  new Distribution(stack, 'MyDist', {
    defaultBehavior: { origin },
    certificate,
    comment: 'a test',
    defaultRootObject: 'index.html',
    domainNames: ['example.com'],
    enabled: false,
    enableIpv6: false,
    enableLogging: true,
    geoRestriction: GeoRestriction.denylist('US', 'GB'),
    httpVersion: HttpVersion.HTTP1_1,
    logFilePrefix: 'logs/',
    logIncludesCookies: true,
    minimumProtocolVersion: SecurityPolicyProtocol.TLS_V1_2_2019,
    priceClass: PriceClass.PRICE_CLASS_100,
    webAclId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Aliases: ['example.com'],
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Comment: 'a test',
      DefaultRootObject: 'index.html',
      Enabled: false,
      HttpVersion: 'http1.1',
      IPV6Enabled: false,
      Logging: {
        Bucket: { 'Fn::GetAtt': ['MyDistLoggingBucket9B8976BC', 'RegionalDomainName'] },
        IncludeCookies: true,
        Prefix: 'logs/',
      },
      Origins: [{
        DomainName: 'www.example.com',
        Id: 'StackMyDistOrigin1D6D5E535',
        CustomOriginConfig: {
          OriginProtocolPolicy: 'https-only',
        },
      }],
      PriceClass: 'PriceClass_100',
      Restrictions: {
        GeoRestriction: {
          Locations: ['US', 'GB'],
          RestrictionType: 'blacklist',
        },
      },
      ViewerCertificate: {
        AcmCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012',
        SslSupportMethod: 'sni-only',
        MinimumProtocolVersion: 'TLSv1.2_2019',
      },
      WebACLId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
    },
  });
});

test('ensure comment prop is not greater than max lenght', () => {
  const origin = defaultOrigin();
  new Distribution(stack, 'MyDist', {
    defaultBehavior: { origin },
    comment: `Adding a comment longer than 128 characters should be trimmed and added the\x20
ellipsis so a user would know there was more to read and everything beyond this point should not show up`,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Comment: `Adding a comment longer than 128 characters should be trimmed and added the\x20
ellipsis so a user would know there was more to ...`,
      Enabled: true,
      HttpVersion: 'http2',
      IPV6Enabled: true,
      Origins: [
        {
          DomainName: 'www.example.com',
          Id: 'StackMyDistOrigin1D6D5E535',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        },
      ],
    },
  });
});

describe('multiple behaviors', () => {
  test('a second behavior can\'t be specified with the catch-all path pattern', () => {
    const origin = defaultOrigin();

    expect(() => {
      new Distribution(stack, 'MyDist', {
        defaultBehavior: { origin },
        additionalBehaviors: {
          '*': { origin },
        },
      });
    }).toThrow(/Only the default behavior can have a path pattern of \'*\'/);
  });

  test('a second behavior can be added to the original origin', () => {
    const origin = defaultOrigin();
    new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      additionalBehaviors: {
        'api/*': { origin },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          TargetOriginId: 'StackMyDistOrigin1D6D5E535',
          ViewerProtocolPolicy: 'allow-all',
        },
        CacheBehaviors: [{
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          PathPattern: 'api/*',
          TargetOriginId: 'StackMyDistOrigin1D6D5E535',
          ViewerProtocolPolicy: 'allow-all',
        }],
        Enabled: true,
        HttpVersion: 'http2',
        IPV6Enabled: true,
        Origins: [{
          DomainName: 'www.example.com',
          Id: 'StackMyDistOrigin1D6D5E535',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        }],
      },
    });
  });

  test('a second behavior can be added to a secondary origin', () => {
    const origin = defaultOrigin();
    const origin2 = defaultOrigin('origin2.example.com');
    new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      additionalBehaviors: {
        'api/*': { origin: origin2 },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          TargetOriginId: 'StackMyDistOrigin1D6D5E535',
          ViewerProtocolPolicy: 'allow-all',
        },
        CacheBehaviors: [{
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          PathPattern: 'api/*',
          TargetOriginId: 'StackMyDistOrigin20B96F3AD',
          ViewerProtocolPolicy: 'allow-all',
        }],
        Enabled: true,
        HttpVersion: 'http2',
        IPV6Enabled: true,
        Origins: [{
          DomainName: 'www.example.com',
          Id: 'StackMyDistOrigin1D6D5E535',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        },
        {
          DomainName: 'origin2.example.com',
          Id: 'StackMyDistOrigin20B96F3AD',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        }],
      },
    });
  });

  test('behavior creation order is preserved', () => {
    const origin = defaultOrigin();
    const origin2 = defaultOrigin('origin2.example.com');
    const dist = new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      additionalBehaviors: {
        'api/1*': { origin: origin2 },
      },
    });
    dist.addBehavior('api/2*', origin);

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          TargetOriginId: 'StackMyDistOrigin1D6D5E535',
          ViewerProtocolPolicy: 'allow-all',
        },
        CacheBehaviors: [{
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          PathPattern: 'api/1*',
          TargetOriginId: 'StackMyDistOrigin20B96F3AD',
          ViewerProtocolPolicy: 'allow-all',
        },
        {
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          PathPattern: 'api/2*',
          TargetOriginId: 'StackMyDistOrigin1D6D5E535',
          ViewerProtocolPolicy: 'allow-all',
        }],
        Enabled: true,
        HttpVersion: 'http2',
        IPV6Enabled: true,
        Origins: [{
          DomainName: 'www.example.com',
          Id: 'StackMyDistOrigin1D6D5E535',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        },
        {
          DomainName: 'origin2.example.com',
          Id: 'StackMyDistOrigin20B96F3AD',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        }],
      },
    });
  });
});

describe('certificates', () => {
  test('should fail if using an imported certificate from outside of us-east-1', () => {
    const origin = defaultOrigin();
    const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:eu-west-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

    expect(() => {
      new Distribution(stack, 'Dist', {
        defaultBehavior: { origin },
        certificate,
      });
    }).toThrow(/Distribution certificates must be in the us-east-1 region and the certificate you provided is in eu-west-1./);
  });

  test('adding a certificate without a domain name', () => {
    const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

    new Distribution(stack, 'Dist1', {
      defaultBehavior: { origin: defaultOrigin() },
      certificate,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Aliases: Match.absent(),
        ViewerCertificate: {
          AcmCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012',
        },
      },
    });
    Annotations.fromStack(stack).hasWarning('/Stack/Dist1', 'No domain names are specified. You will need to specify it after running associate-alias CLI command manually. See the "Moving an alternate domain name to a different distribution" section of module\'s README for more info. [ack: @aws-cdk/aws-cloudfront:emptyDomainNames]');
  });

  test('use the TLSv1.2_2021 security policy by default', () => {
    const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

    new Distribution(stack, 'Dist', {
      defaultBehavior: { origin: defaultOrigin() },
      domainNames: ['example.com', 'www.example.com'],
      certificate,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Aliases: ['example.com', 'www.example.com'],
        ViewerCertificate: {
          AcmCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012',
          SslSupportMethod: 'sni-only',
          MinimumProtocolVersion: 'TLSv1.2_2021',
        },
      },
    });
  });

  test('adding a certificate with non default security policy protocol', () => {
    const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');
    new Distribution(stack, 'Dist', {
      defaultBehavior: { origin: defaultOrigin() },
      domainNames: ['www.example.com'],
      sslSupportMethod: SSLMethod.SNI,
      minimumProtocolVersion: SecurityPolicyProtocol.TLS_V1_2016,
      certificate: certificate,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        ViewerCertificate: {
          AcmCertificateArn: 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012',
          SslSupportMethod: 'sni-only',
          MinimumProtocolVersion: 'TLSv1_2016',
        },
      },
    });
  });

  test('warns when minimumProtocolVersion is set without a certificate', () => {
    new Distribution(stack, 'Dist', {
      defaultBehavior: { origin: defaultOrigin() },
      minimumProtocolVersion: SecurityPolicyProtocol.TLS_V1_2016,
    });

    Annotations.fromStack(stack).hasWarning('/Stack/Dist', Match.stringLikeRegexp('.*minimumProtocolVersion.*without.*certificate.*'));
  });

  test('warns when sslSupportMethod is set without a certificate', () => {
    new Distribution(stack, 'Dist', {
      defaultBehavior: { origin: defaultOrigin() },
      sslSupportMethod: SSLMethod.SNI,
    });

    Annotations.fromStack(stack).hasWarning('/Stack/Dist', Match.stringLikeRegexp('.*sslSupportMethod.*without.*certificate.*'));
  });

  test('does not warn about minimumProtocolVersion when certificate is provided', () => {
    const certificate = acm.Certificate.fromCertificateArn(stack, 'Cert', 'arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012');

    new Distribution(stack, 'Dist', {
      defaultBehavior: { origin: defaultOrigin() },
      domainNames: ['example.com'],
      minimumProtocolVersion: SecurityPolicyProtocol.TLS_V1_2016,
      certificate,
    });

    Annotations.fromStack(stack).hasNoWarning('/Stack/Dist', Match.stringLikeRegexp('.*minimumProtocolVersion.*'));
  });
});

describe('custom error responses', () => {
  test('should fail if only the error code is provided', () => {
    const origin = defaultOrigin();

    expect(() => {
      new Distribution(stack, 'Dist', {
        defaultBehavior: { origin },
        errorResponses: [{ httpStatus: 404 }],
      });
    }).toThrow(/A custom error response without either a \'responseHttpStatus\', \'ttl\' or \'responsePagePath\' is not valid./);
  });

  test('should render the array of error configs if provided', () => {
    const origin = defaultOrigin();
    new Distribution(stack, 'Dist', {
      defaultBehavior: { origin },
      errorResponses: [{
        // responseHttpStatus defaults to httpsStatus
        httpStatus: 404,
        responsePagePath: '/errors/404.html',
      },
      {
        // without responsePagePath
        httpStatus: 500,
        ttl: Duration.seconds(2),
      },
      {
        // with responseHttpStatus different from httpStatus
        httpStatus: 403,
        responseHttpStatus: 200,
        responsePagePath: '/index.html',
      }],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        CustomErrorResponses: [
          {
            ErrorCode: 404,
            ResponseCode: 404,
            ResponsePagePath: '/errors/404.html',
          },
          {
            ErrorCachingMinTTL: 2,
            ErrorCode: 500,
          },
          {
            ErrorCode: 403,
            ResponseCode: 200,
            ResponsePagePath: '/index.html',
          },
        ],
      },
    });
  });
});

describe('logging', () => {
  test('does not include logging if disabled and no bucket provided', () => {
    const origin = defaultOrigin();
    new Distribution(stack, 'MyDist', { defaultBehavior: { origin } });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Logging: Match.absent(),
      },
    });
  });

  test('throws error if logging disabled but bucket provided', () => {
    const origin = defaultOrigin();

    expect(() => {
      new Distribution(stack, 'MyDist', {
        defaultBehavior: { origin },
        enableLogging: false,
        logBucket: new s3.Bucket(stack, 'Bucket'),
      });
    }).toThrow(/Explicitly disabled logging but provided a logging bucket./);
  });

  test('creates bucket if none is provided', () => {
    const origin = defaultOrigin();
    new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      enableLogging: true,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Logging: {
          Bucket: { 'Fn::GetAtt': ['MyDistLoggingBucket9B8976BC', 'RegionalDomainName'] },
        },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      OwnershipControls: {
        Rules: [
          { ObjectOwnership: 'ObjectWriter' },
        ],
      },
    });
  });

  test('uses existing bucket if provided', () => {
    const origin = defaultOrigin();
    const loggingBucket = new s3.Bucket(stack, 'MyLoggingBucket');
    new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      logBucket: loggingBucket,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Logging: {
          Bucket: { 'Fn::GetAtt': ['MyLoggingBucket4382CD04', 'RegionalDomainName'] },
        },
      },
    });
  });

  test('can set prefix and cookies', () => {
    const origin = defaultOrigin();
    new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      enableLogging: true,
      logFilePrefix: 'logs/',
      logIncludesCookies: true,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Logging: {
          Bucket: { 'Fn::GetAtt': ['MyDistLoggingBucket9B8976BC', 'RegionalDomainName'] },
          IncludeCookies: true,
          Prefix: 'logs/',
        },
      },
    });
  });
});

describe('with Lambda@Edge functions', () => {
  let lambdaFunction: lambda.Function;
  let origin: IOrigin;

  beforeEach(() => {
    lambdaFunction = new lambda.Function(stack, 'Function', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('whatever'),
      handler: 'index.handler',
    });

    origin = defaultOrigin();
  });

  test('can add an edge lambdas to the default behavior', () => {
    new Distribution(stack, 'MyDist', {
      defaultBehavior: {
        origin,
        edgeLambdas: [
          {
            functionVersion: lambdaFunction.currentVersion,
            eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
            includeBody: true,
          },
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          LambdaFunctionAssociations: [
            {
              EventType: 'origin-request',
              IncludeBody: true,
              LambdaFunctionARN: {
                Ref: Match.stringLikeRegexp(stack.getLogicalId(lambdaFunction.currentVersion.node.defaultChild as lambda.CfnVersion)),
              },
            },
          ],
        },
      },
    });
  });

  test('edgelambda.amazonaws.com is added to the trust policy of lambda', () => {
    new Distribution(stack, 'MyDist', {
      defaultBehavior: {
        origin,
        edgeLambdas: [
          {
            functionVersion: lambdaFunction.currentVersion,
            eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
          },
        ],
      },
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
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'edgelambda.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('can add an edge lambdas to additional behaviors', () => {
    new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      additionalBehaviors: {
        'images/*': {
          origin,
          edgeLambdas: [
            {
              functionVersion: lambdaFunction.currentVersion,
              eventType: LambdaEdgeEventType.VIEWER_REQUEST,
            },
          ],
        },
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        CacheBehaviors: [
          Match.objectLike({
            PathPattern: 'images/*',
            LambdaFunctionAssociations: [
              {
                EventType: 'viewer-request',
                LambdaFunctionARN: {
                  Ref: Match.stringLikeRegexp(stack.getLogicalId(lambdaFunction.currentVersion.node.defaultChild as lambda.CfnVersion)),
                },
              },
            ],
          }),
        ],
      },
    });
  });

  test('fails creation when attempting to add the $LATEST function version as an edge Lambda to the default behavior', () => {
    expect(() => {
      new Distribution(stack, 'MyDist', {
        defaultBehavior: {
          origin,
          edgeLambdas: [
            {
              functionVersion: lambdaFunction.latestVersion,
              eventType: LambdaEdgeEventType.ORIGIN_RESPONSE,
            },
          ],
        },
      });
    }).toThrow(/\$LATEST function version cannot be used for Lambda@Edge/);
  });

  test('with removable env vars', () => {
    const envLambdaFunction = new lambda.Function(stack, 'EnvFunction', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('whateverwithenv'),
      handler: 'index.handler',
    });
    envLambdaFunction.addEnvironment('KEY', 'value', { removeInEdge: true });

    new Distribution(stack, 'MyDist', {
      defaultBehavior: {
        origin,
        edgeLambdas: [
          {
            functionVersion: envLambdaFunction.currentVersion,
            eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
          },
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
      Environment: Match.absent(),
      Code: {
        ZipFile: 'whateverwithenv',
      },
    });
  });

  test('with incompatible env vars', () => {
    const envLambdaFunction = new lambda.Function(stack, 'EnvFunction', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('whateverwithenv'),
      handler: 'index.handler',
      environment: {
        KEY: 'value',
      },
    });

    new Distribution(stack, 'MyDist', {
      defaultBehavior: {
        origin,
        edgeLambdas: [
          {
            functionVersion: envLambdaFunction.currentVersion,
            eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
          },
        ],
      },
    });

    expect(() => app.synth()).toThrow(/KEY/);
  });

  test('with singleton function', () => {
    const singleton = new lambda.SingletonFunction(stack, 'Singleton', {
      uuid: 'singleton-for-cloudfront',
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('code'),
      handler: 'index.handler',
    });

    new Distribution(stack, 'MyDist', {
      defaultBehavior: {
        origin,
        edgeLambdas: [
          {
            functionVersion: singleton.currentVersion,
            eventType: LambdaEdgeEventType.ORIGIN_REQUEST,
          },
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          LambdaFunctionAssociations: [
            {
              EventType: 'origin-request',
              LambdaFunctionARN: {
                Ref: Match.stringLikeRegexp(stack.getLogicalId(singleton.currentVersion.node.defaultChild as lambda.CfnVersion)),
              },
            },
          ],
        },
      },
    });
  });
});

describe('with CloudFront functions', () => {
  test('can add a CloudFront function to the default behavior', () => {
    new Distribution(stack, 'MyDist', {
      defaultBehavior: {
        origin: defaultOrigin(),
        functionAssociations: [
          {
            eventType: FunctionEventType.VIEWER_REQUEST,
            function: new Function(stack, 'TestFunction', {
              code: FunctionCode.fromInline('foo'),
            }),
          },
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          FunctionAssociations: [
            {
              EventType: 'viewer-request',
              FunctionARN: {
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
});

test('price class is included if provided', () => {
  const origin = defaultOrigin();
  new Distribution(stack, 'Dist', {
    defaultBehavior: { origin },
    priceClass: PriceClass.PRICE_CLASS_200,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      PriceClass: 'PriceClass_200',
    },
  });
});

test('escape hatches are supported', () => {
  Validations.of(stack).acknowledge({
    id: 'CloudFormation-Validate::F3003',
    reason: 'Missing required property in escape hatch',
  });
  const dist = new Distribution(stack, 'Dist', {
    defaultBehavior: { origin: defaultOrigin() },
  });
  const cfnDist = dist.node.defaultChild as CfnDistribution;
  cfnDist.addPropertyOverride('DistributionConfig.DefaultCacheBehavior.ForwardedValues.Headers', ['*']);

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      DefaultCacheBehavior: {
        ForwardedValues: {
          Headers: ['*'],
        },
      },
    },
  });
});

describe('origin IDs', () => {
  test('origin ID is limited to 128 characters', () => {
    const nestedStack = new Stack(stack, 'LongNameThatWillEndUpGeneratingAUniqueNodeIdThatIsLongerThanTheOneHundredAndTwentyEightCharacterLimit');

    new Distribution(nestedStack, 'AReallyAwesomeDistributionWithAMemorableNameThatIWillNeverForget', {
      defaultBehavior: { origin: defaultOrigin() },
    });

    Template.fromStack(nestedStack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        Origins: [Match.objectLike({
          Id: 'ngerThanTheOneHundredAndTwentyEightCharacterLimitAReallyAwesomeDistributionWithAMemorableNameThatIWillNeverForgetOrigin1D38031F9',
        })],
      },
    });
  });

  test('origin group ID is limited to 128 characters', () => {
    const nestedStack = new Stack(stack, 'LongNameThatWillEndUpGeneratingAUniqueNodeIdThatIsLongerThanTheOneHundredAndTwentyEightCharacterLimit');

    new Distribution(nestedStack, 'AReallyAwesomeDistributionWithAMemorableNameThatIWillNeverForget', {
      defaultBehavior: { origin: defaultOriginGroup() },
    });

    Template.fromStack(nestedStack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        OriginGroups: {
          Items: [Match.objectLike({
            Id: 'hanTheOneHundredAndTwentyEightCharacterLimitAReallyAwesomeDistributionWithAMemorableNameThatIWillNeverForgetOriginGroup1B5CE3FE6',
          })],
        },
      },
    });
  });
});

describe('custom origin ids', () => {
  test('test that originId param is respected', () => {
    const origin = defaultOrigin(undefined, 'custom-origin-id');

    const distribution = new Distribution(stack, 'Http1Distribution', {
      defaultBehavior: { origin },
      additionalBehaviors: {
        secondUsage: {
          origin,
        },
      },
    });
    distribution.addBehavior(
      'thirdUsage',
      origin,
    );

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          TargetOriginId: 'custom-origin-id',
          ViewerProtocolPolicy: 'allow-all',
        },
        CacheBehaviors: [{
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          PathPattern: 'secondUsage',
          TargetOriginId: 'custom-origin-id',
          ViewerProtocolPolicy: 'allow-all',
        },
        {
          CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
          Compress: true,
          PathPattern: 'thirdUsage',
          TargetOriginId: 'custom-origin-id',
          ViewerProtocolPolicy: 'allow-all',
        }],
        Enabled: true,
        HttpVersion: 'http2',
        IPV6Enabled: true,
        Origins: [{
          DomainName: 'www.example.com',
          Id: 'custom-origin-id',
          CustomOriginConfig: {
            OriginProtocolPolicy: 'https-only',
          },
        }],
      },
    });
  });
});

describe('supported HTTP versions', () => {
  test('setting HTTP/1.1 renders HttpVersion correctly', () => {
    new Distribution(stack, 'Http1Distribution', {
      httpVersion: HttpVersion.HTTP1_1,
      defaultBehavior: { origin: defaultOrigin() },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        HttpVersion: 'http1.1',
      },
    });
  });
  test('setting HTTP/2 renders HttpVersion correctly', () => {
    new Distribution(stack, 'Http1Distribution', {
      httpVersion: HttpVersion.HTTP2,
      defaultBehavior: { origin: defaultOrigin() },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        HttpVersion: 'http2',
      },
    });
  });
  test('setting HTTP/3 renders HttpVersion correctly', () => {
    new Distribution(stack, 'Http1Distribution', {
      httpVersion: HttpVersion.HTTP3,
      defaultBehavior: { origin: defaultOrigin() },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        HttpVersion: 'http3',
      },
    });
  });
  test('setting HTTP/2 and HTTP/3 renders HttpVersion correctly', () => {
    new Distribution(stack, 'Http1Distribution', {
      httpVersion: HttpVersion.HTTP2_AND_3,
      defaultBehavior: { origin: defaultOrigin() },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        HttpVersion: 'http2and3',
      },
    });
  });
});

test('grants custom actions', () => {
  const distribution = new Distribution(stack, 'Distribution', {
    defaultBehavior: { origin: defaultOrigin() },
  });
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.AccountRootPrincipal(),
  });
  distribution.grant(
    role,
    'cloudfront:ListInvalidations',
    'cloudfront:GetInvalidation',
    'cloudfront:CreateFieldLevelEncryptionConfig',
    'cloudfront:CreateFieldLevelEncryptionProfile',
    'cloudfront:CreateKeyGroup',
    'cloudfront:CreateMonitoringSubscription',
    'cloudfront:CreateOriginAccessControl',
    'cloudfront:CreatePublicKey',
    'cloudfront:CreateSavingsPlan',
    'cloudfront:DeleteKeyGroup',
    'cloudfront:DeleteMonitoringSubscription',
    'cloudfront:DeletePublicKey',
    'cloudfront:GetKeyGroup',
    'cloudfront:GetKeyGroupConfig',
    'cloudfront:GetMonitoringSubscription',
    'cloudfront:GetPublicKey',
    'cloudfront:GetPublicKeyConfig',
    'cloudfront:GetSavingsPlan',
    'cloudfront:ListAnycastIpLists',
    'cloudfront:ListCachePolicies',
    'cloudfront:ListCloudFrontOriginAccessIdentities',
    'cloudfront:ListContinuousDeploymentPolicies',
    'cloudfront:ListDistributions',
    'cloudfront:ListDistributionsByAnycastIpListId',
    'cloudfront:ListDistributionsByCachePolicyId',
    'cloudfront:ListDistributionsByKeyGroup',
    'cloudfront:ListDistributionsByLambdaFunction',
    'cloudfront:ListDistributionsByOriginRequestPolicyId',
    'cloudfront:ListDistributionsByRealtimeLogConfig',
    'cloudfront:ListDistributionsByResponseHeadersPolicyId',
    'cloudfront:ListDistributionsByVpcOriginId',
    'cloudfront:ListDistributionsByWebACLId',
    'cloudfront:ListFieldLevelEncryptionConfigs',
    'cloudfront:ListFieldLevelEncryptionProfiles',
    'cloudfront:ListFunctions',
    'cloudfront:ListKeyGroups',
    'cloudfront:ListKeyValueStores',
    'cloudfront:ListOriginAccessControls',
    'cloudfront:ListOriginRequestPolicies',
    'cloudfront:ListPublicKeys',
    'cloudfront:ListRateCards',
    'cloudfront:ListRealtimeLogConfigs',
    'cloudfront:ListResponseHeadersPolicies',
    'cloudfront:ListSavingsPlans',
    'cloudfront:ListStreamingDistributions',
    'cloudfront:ListUsages',
    'cloudfront:ListVpcOrigins',
    'cloudfront:UpdateFieldLevelEncryptionConfig',
    'cloudfront:UpdateKeyGroup',
    'cloudfront:UpdatePublicKey',
    'cloudfront:UpdateSavingsPlan',
  );

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: [
            'cloudfront:CreateFieldLevelEncryptionConfig',
            'cloudfront:CreateFieldLevelEncryptionProfile',
            'cloudfront:CreateKeyGroup',
            'cloudfront:CreateMonitoringSubscription',
            'cloudfront:CreateOriginAccessControl',
            'cloudfront:CreatePublicKey',
            'cloudfront:CreateSavingsPlan',
            'cloudfront:DeleteKeyGroup',
            'cloudfront:DeleteMonitoringSubscription',
            'cloudfront:DeletePublicKey',
            'cloudfront:GetKeyGroup',
            'cloudfront:GetKeyGroupConfig',
            'cloudfront:GetMonitoringSubscription',
            'cloudfront:GetPublicKey',
            'cloudfront:GetPublicKeyConfig',
            'cloudfront:GetSavingsPlan',
            'cloudfront:ListAnycastIpLists',
            'cloudfront:ListCachePolicies',
            'cloudfront:ListCloudFrontOriginAccessIdentities',
            'cloudfront:ListContinuousDeploymentPolicies',
            'cloudfront:ListDistributions',
            'cloudfront:ListDistributionsByAnycastIpListId',
            'cloudfront:ListDistributionsByCachePolicyId',
            'cloudfront:ListDistributionsByKeyGroup',
            'cloudfront:ListDistributionsByLambdaFunction',
            'cloudfront:ListDistributionsByOriginRequestPolicyId',
            'cloudfront:ListDistributionsByRealtimeLogConfig',
            'cloudfront:ListDistributionsByResponseHeadersPolicyId',
            'cloudfront:ListDistributionsByVpcOriginId',
            'cloudfront:ListDistributionsByWebACLId',
            'cloudfront:ListFieldLevelEncryptionConfigs',
            'cloudfront:ListFieldLevelEncryptionProfiles',
            'cloudfront:ListFunctions',
            'cloudfront:ListKeyGroups',
            'cloudfront:ListKeyValueStores',
            'cloudfront:ListOriginAccessControls',
            'cloudfront:ListOriginRequestPolicies',
            'cloudfront:ListPublicKeys',
            'cloudfront:ListRateCards',
            'cloudfront:ListRealtimeLogConfigs',
            'cloudfront:ListResponseHeadersPolicies',
            'cloudfront:ListSavingsPlans',
            'cloudfront:ListStreamingDistributions',
            'cloudfront:ListUsages',
            'cloudfront:ListVpcOrigins',
            'cloudfront:UpdateFieldLevelEncryptionConfig',
            'cloudfront:UpdateKeyGroup',
            'cloudfront:UpdatePublicKey',
            'cloudfront:UpdateSavingsPlan',
          ],
          Resource: '*',
        },
        {
          Action: [
            'cloudfront:ListInvalidations',
            'cloudfront:GetInvalidation',
          ],
          Resource: {
            'Fn::Join': [
              '', [
                'arn:', { Ref: 'AWS::Partition' }, ':cloudfront::1234:distribution/',
                { Ref: 'Distribution830FAC52' },
              ],
            ],
          },
        },
      ],
    },
  });
});

test('grants createInvalidation', () => {
  const distribution = new Distribution(stack, 'Distribution', {
    defaultBehavior: { origin: defaultOrigin() },
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
                'arn:', { Ref: 'AWS::Partition' }, ':cloudfront::1234:distribution/',
                { Ref: 'Distribution830FAC52' },
              ],
            ],
          },
        },
      ],
    },
  });
});

test('grants createInvalidation to L1', () => {
  const distribution = new Distribution(stack, 'Distribution', {
    defaultBehavior: { origin: defaultOrigin() },
  });

  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.AccountRootPrincipal(),
  });

  DistributionGrants.
    fromDistribution(distribution.node.defaultChild as CfnDistribution)
    .createInvalidation(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: 'cloudfront:CreateInvalidation',
          Resource: {
            'Fn::Join': [
              '', [
                'arn:', { Ref: 'AWS::Partition' }, ':cloudfront::1234:distribution/',
                { Ref: 'Distribution830FAC52' },
              ],
            ],
          },
        },
      ],
    },
  });
});

test('render distribution behavior with realtime log config', () => {
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('cloudfront.amazonaws.com'),
  });

  const stream = new kinesis.Stream(stack, 'stream', {
    streamMode: kinesis.StreamMode.ON_DEMAND,
    encryption: kinesis.StreamEncryption.MANAGED,
  });

  const realTimeConfig = new RealtimeLogConfig(stack, 'RealtimeConfig', {
    endPoints: [
      Endpoint.fromKinesisStream(stream, role),
    ],
    fields: ['timestamp'],
    realtimeLogConfigName: 'realtime-config',
    samplingRate: 50,
  });

  new Distribution(stack, 'MyDist', {
    defaultBehavior: {
      origin: defaultOrigin(),
      realtimeLogConfig: realTimeConfig,
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution',
    Match.objectLike({
      DistributionConfig: {
        DefaultCacheBehavior: {
          RealtimeLogConfigArn: {
            Ref: 'RealtimeConfigB6004E8E',
          },
        },
      },
    }));
});

test('render distribution behavior with realtime log config - multiple behaviors', () => {
  const role = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('cloudfront.amazonaws.com'),
  });

  const stream = new kinesis.Stream(stack, 'stream', {
    streamMode: kinesis.StreamMode.ON_DEMAND,
    encryption: kinesis.StreamEncryption.MANAGED,
  });

  const realTimeConfig = new RealtimeLogConfig(stack, 'RealtimeConfig', {
    endPoints: [
      Endpoint.fromKinesisStream(stream, role),
    ],
    fields: ['timestamp'],
    realtimeLogConfigName: 'realtime-config',
    samplingRate: 50,
  });

  const origin2 = defaultOrigin('origin2.example.com');

  new Distribution(stack, 'MyDist', {
    defaultBehavior: {
      origin: defaultOrigin(),
      realtimeLogConfig: realTimeConfig,
    },
    additionalBehaviors: {
      '/api/*': {
        origin: origin2,
        realtimeLogConfig: realTimeConfig,
      },
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution',
    Match.objectLike({
      DistributionConfig: {
        DefaultCacheBehavior: {
          RealtimeLogConfigArn: {
            Ref: 'RealtimeConfigB6004E8E',
          },
          TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        },
        CacheBehaviors: [{
          PathPattern: '/api/*',
          RealtimeLogConfigArn: {
            Ref: 'RealtimeConfigB6004E8E',
          },
          TargetOriginId: 'StackMyDistOrigin20B96F3AD',
        }],
      },
    }));
});

test('with publish additional metrics', () => {
  const origin = defaultOrigin();
  new Distribution(stack, 'MyDist', {
    defaultBehavior: { origin },
    publishAdditionalMetrics: true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Enabled: true,
      HttpVersion: 'http2',
      IPV6Enabled: true,
      Origins: [{
        DomainName: 'www.example.com',
        Id: 'StackMyDistOrigin1D6D5E535',
        CustomOriginConfig: {
          OriginProtocolPolicy: 'https-only',
        },
      }],
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::MonitoringSubscription', {
    DistributionId: {
      Ref: 'MyDistDB88FD9A',
    },
    MonitoringSubscription: {
      RealtimeMetricsSubscriptionConfig: {
        RealtimeMetricsSubscriptionStatus: 'Enabled',
      },
    },
  });
});

test('with origin access control id', () => {
  const origin = defaultOriginWithOriginAccessControl();
  new Distribution(stack, 'MyDist', {
    defaultBehavior: { origin },
    publishAdditionalMetrics: true,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      DefaultCacheBehavior: {
        CachePolicyId: '658327ea-f89d-4fab-a63d-7e88639e58f6',
        Compress: true,
        TargetOriginId: 'StackMyDistOrigin1D6D5E535',
        ViewerProtocolPolicy: 'allow-all',
      },
      Enabled: true,
      HttpVersion: 'http2',
      IPV6Enabled: true,
      Origins: [{
        DomainName: 'www.example.com',
        Id: 'StackMyDistOrigin1D6D5E535',
        CustomOriginConfig: {
          OriginProtocolPolicy: 'https-only',
        },
        OriginAccessControlId: 'test-origin-access-control-id',
      }],
    },
  });
});

describe('Distribution metrics tests', () => {
  const additionalMetrics = [
    { name: 'OriginLatency', method: 'metricOriginLatency', statistic: 'Average', additionalMetricsRequired: true, errorMetricName: 'Origin latency' },
    { name: 'CacheHitRate', method: 'metricCacheHitRate', statistic: 'Average', additionalMetricsRequired: true, errorMetricName: 'Cache hit rate' },
    ...['401', '403', '404', '502', '503', '504'].map(errorCode => ({
      name: `${errorCode}ErrorRate`,
      method: `metric${errorCode}ErrorRate`,
      statistic: 'Average',
      additionalMetricsRequired: true,
      errorMetricName: `${errorCode} error rate`,
    })),
  ] as const;

  const defaultMetrics = [
    { name: 'Requests', method: 'metricRequests', statistic: 'Sum', additionalMetricsRequired: false, errorMetricName: '' },
    { name: 'BytesDownloaded', method: 'metricBytesDownloaded', statistic: 'Sum', additionalMetricsRequired: false, errorMetricName: '' },
    { name: 'BytesUploaded', method: 'metricBytesUploaded', statistic: 'Sum', additionalMetricsRequired: false, errorMetricName: '' },
    { name: 'TotalErrorRate', method: 'metricTotalErrorRate', statistic: 'Average', additionalMetricsRequired: false, errorMetricName: '' },
    { name: '4xxErrorRate', method: 'metric4xxErrorRate', statistic: 'Average', additionalMetricsRequired: false, errorMetricName: '' },
    { name: '5xxErrorRate', method: 'metric5xxErrorRate', statistic: 'Average', additionalMetricsRequired: false, errorMetricName: '' },
  ] as const;

  test.each([... additionalMetrics, ...defaultMetrics])('get %s metric', (metric) => {
    const origin = defaultOrigin();
    const dist = new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      publishAdditionalMetrics: metric.additionalMetricsRequired,
    });

    const metricObj = (dist as any)[metric.method]();

    expect(metricObj).toEqual(new cloudwatch.Metric({
      namespace: 'AWS/CloudFront',
      metricName: metric.name,
      dimensions: { DistributionId: dist.distributionId },
      statistic: metric.statistic,
      period: Duration.minutes(5),
    }));
  });

  test.each(additionalMetrics)('throw error when trying to get %s metric without publishing additional metrics', (metric) => {
    const origin = defaultOrigin();
    const dist = new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      publishAdditionalMetrics: false,
    });

    expect(() => {
      (dist as any)[metric.method]();
    }).toThrow(new RegExp(`${metric.errorMetricName} metric is only available if 'publishAdditionalMetrics' is set 'true'`));
  });
});

describe('attachWebAclId', () => {
  test('can attach WebAcl to the distribution by the method', () => {
    const origin = defaultOrigin();

    const distribution = new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
    });

    distribution.attachWebAclId('473e64fd-f30b-4765-81a0-62ad96dd167a');

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        WebACLId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
      },
    });
  });

  test('throws if a WebAcl is already attached to the distribution', () => {
    const origin = defaultOrigin();

    const distribution = new Distribution(stack, 'MyDist', {
      defaultBehavior: { origin },
      webAclId: '473e64fd-f30b-4765-81a0-62ad96dd167a',
    });

    expect(() => {
      distribution.attachWebAclId('473e64fd-f30b-4765-81a0-62ad96dd167b');
    }).toThrow(/A WebACL has already been attached to this distribution/);
  });

  describe('throws if the WebAcl is not in us-east-1 region', () => {
    test('when try to attach WebACL using `attachWebAclId` method', () => {
      const origin = defaultOrigin();

      const distribution = new Distribution(stack, 'MyDist', {
        defaultBehavior: { origin },
      });

      expect(() => {
        distribution.attachWebAclId('arn:aws:wafv2:ap-northeast-1:123456789012:global/web-acl/MyWebAcl/473e64fd-f30b-4765-81a0-62ad96dd167a');
      }).toThrow(/WebACL for CloudFront distributions must be created in the us-east-1 region; received ap-northeast-1/);
    });

    test('when try to attach WebACL by specifying value for props', () => {
      const origin = defaultOrigin();

      expect(() => {
        new Distribution(stack, 'MyDist', {
          defaultBehavior: { origin },
          webAclId: 'arn:aws:wafv2:ap-northeast-1:123456789012:global/web-acl/MyWebAcl/473e64fd-f30b-4765-81a0-62ad96dd167a',
        });
      }).toThrow(/WebACL for CloudFront distributions must be created in the us-east-1 region; received ap-northeast-1/);
    });

    test('does not validate unresolved token webAclId', () => {
      const origin = defaultOrigin();

      new Distribution(stack, 'MyDist', {
        defaultBehavior: { origin },
        webAclId: Token.asString({ Ref: 'SomeWebAcl' }), // unresolved token
      });

      // Should synthesize without error
      Template.fromStack(stack);
    });
  });
});

describe('gRPC', () => {
  test.each([
    true,
    false,
    undefined,
  ])('set gRPC to %s in defaultBehavior', (enableGrpc) => {
    const origin = defaultOrigin();
    new Distribution(stack, 'MyDist', {
      httpVersion: HttpVersion.HTTP2,
      defaultBehavior: {
        origin,
        allowedMethods: AllowedMethods.ALLOW_ALL,
        enableGrpc,
      },
    });

    const grpcConfig = enableGrpc !== undefined ? {
      Enabled: enableGrpc,
    } : Match.absent();

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        DefaultCacheBehavior: {
          GrpcConfig: grpcConfig,
        },
      },
    });
  });

  test.each([
    true,
    false,
    undefined,
  ])('set gRPC to %s in additionalBehaviors', (enableGrpc) => {
    const origin = defaultOrigin();
    new Distribution(stack, 'MyDist', {
      httpVersion: HttpVersion.HTTP2,
      defaultBehavior: {
        origin,
      },
      additionalBehaviors: {
        '/second': {
          origin,
          allowedMethods: AllowedMethods.ALLOW_ALL,
          enableGrpc,
        },
      },
    });

    const grpcConfig = enableGrpc !== undefined ? {
      Enabled: enableGrpc,
    } : Match.absent();

    Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
      DistributionConfig: {
        CacheBehaviors: [{
          PathPattern: '/second',
          GrpcConfig: grpcConfig,
        }],
      },
    });
  });

  test.each([
    HttpVersion.HTTP1_1,
    HttpVersion.HTTP3,
  ])('throws if httpVersion is %s and enableGrpc in defaultBehavior is true', (httpVersion) => {
    const origin = defaultOrigin();
    const msg = `'httpVersion' must be http2 or http2and3 if 'enableGrpc' in 'defaultBehavior' or 'additionalBehaviors' is true, got ${httpVersion}`;

    expect(() => {
      new Distribution(stack, 'MyDist', {
        httpVersion,
        defaultBehavior: {
          origin,
          enableGrpc: true,
          allowedMethods: AllowedMethods.ALLOW_ALL,
        },
      });
    }).toThrow(msg);
  });

  test.each([
    HttpVersion.HTTP1_1,
    HttpVersion.HTTP3,
  ])('throws if httpVersion is %s and enableGrpc in additionalBehaviors is true', (httpVersion) => {
    const origin = defaultOrigin();
    const msg = `'httpVersion' must be http2 or http2and3 if 'enableGrpc' in 'defaultBehavior' or 'additionalBehaviors' is true, got ${httpVersion}`;

    expect(() => {
      new Distribution(stack, 'MyDist', {
        httpVersion,
        defaultBehavior: {
          origin,
          enableGrpc: false,
          allowedMethods: AllowedMethods.ALLOW_ALL,
        },
        additionalBehaviors: {
          '/second': {
            origin,
            enableGrpc: false,
            allowedMethods: AllowedMethods.ALLOW_ALL,
          },
          '/third': {
            origin,
            enableGrpc: true,
            allowedMethods: AllowedMethods.ALLOW_ALL,
          },
        },
      });
    }).toThrow(msg);
  });
});
