import { readdirSync, readFileSync, existsSync } from 'fs';
import * as path from 'path';
import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Match, Template } from '../../assertions';
import * as cloudfront from '../../aws-cloudfront';
import * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import * as logs from '../../aws-logs';
import * as s3 from '../../aws-s3';
import * as sns from '../../aws-sns';
import * as ssm from '../../aws-ssm';
import * as cdk from '../../core';
import { UnscopedValidationError } from '../../core/lib/errors';
import { lit } from '../../core/lib/private/literal-string';
import * as cxapi from '../../cx-api';
import * as s3deploy from '../lib';

/* eslint-disable max-len */

test('deploy from local directory asset', () => {
  // GIVEN
  const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
  const stack = new cdk.Stack(app);
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    ServiceToken: {
      'Fn::GetAtt': [
        'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756C81C01536',
        'Arn',
      ],
    },
    SourceBucketNames: [{
      Ref: 'AssetParametersfc4481abf279255619ff7418faa5d24456fef3432ea0da59c95542578ff0222eS3Bucket9CD8B20A',
    }],
    SourceObjectKeys: [{
      'Fn::Join': [
        '',
        [
          {
            'Fn::Select': [
              0,
              {
                'Fn::Split': [
                  '||',
                  {
                    Ref: 'AssetParametersfc4481abf279255619ff7418faa5d24456fef3432ea0da59c95542578ff0222eS3VersionKeyA58D380C',
                  },
                ],
              },
            ],
          },
          {
            'Fn::Select': [
              1,
              {
                'Fn::Split': [
                  '||',
                  {
                    Ref: 'AssetParametersfc4481abf279255619ff7418faa5d24456fef3432ea0da59c95542578ff0222eS3VersionKeyA58D380C',
                  },
                ],
              },
            ],
          },
        ],
      ],
    }],
    DestinationBucketName: {
      Ref: 'DestC383B82A',
    },
  });
});

test('empty sources array is preserved', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app);
  const bucket = new s3.Bucket(stack, 'Dest');

  new s3deploy.BucketDeployment(stack, 'EmptyDeployment', {
    destinationBucket: bucket,
    sources: [],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    SourceBucketNames: [],
  });
});

test('deploy with configured log retention', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    logRetention: logs.RetentionDays.ONE_WEEK,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::LogRetention', { RetentionInDays: 7 });
});

test('deploy with log group', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    logGroup: new logs.LogGroup(stack, 'LogGroup', {
      retention: logs.RetentionDays.ONE_WEEK,
    }),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Logs::LogGroup', { RetentionInDays: 7 });
});

test('deploy from local directory assets', () => {
  // GIVEN
  const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
  const stack = new cdk.Stack(app);
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [
      s3deploy.Source.asset(path.join(__dirname, 'my-website')),
      s3deploy.Source.asset(path.join(__dirname, 'my-website-second')),
    ],
    destinationBucket: bucket,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    ServiceToken: {
      'Fn::GetAtt': [
        'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756C81C01536',
        'Arn',
      ],
    },
    SourceBucketNames: [
      {
        Ref: 'AssetParametersfc4481abf279255619ff7418faa5d24456fef3432ea0da59c95542578ff0222eS3Bucket9CD8B20A',
      },
      {
        Ref: 'AssetParametersa94977ede0211fd3b45efa33d6d8d1d7bbe0c5a96d977139d8b16abfa96fe9cbS3Bucket99793559',
      },
    ],
    SourceObjectKeys: [
      {
        'Fn::Join': [
          '',
          [
            {
              'Fn::Select': [
                0,
                {
                  'Fn::Split': [
                    '||',
                    {
                      Ref: 'AssetParametersfc4481abf279255619ff7418faa5d24456fef3432ea0da59c95542578ff0222eS3VersionKeyA58D380C',
                    },
                  ],
                },
              ],
            },
            {
              'Fn::Select': [
                1,
                {
                  'Fn::Split': [
                    '||',
                    {
                      Ref: 'AssetParametersfc4481abf279255619ff7418faa5d24456fef3432ea0da59c95542578ff0222eS3VersionKeyA58D380C',
                    },
                  ],
                },
              ],
            },
          ],
        ],
      },
      {
        'Fn::Join': [
          '',
          [
            {
              'Fn::Select': [
                0,
                {
                  'Fn::Split': [
                    '||',
                    {
                      Ref: 'AssetParametersa94977ede0211fd3b45efa33d6d8d1d7bbe0c5a96d977139d8b16abfa96fe9cbS3VersionKeyD9ACE665',
                    },
                  ],
                },
              ],
            },
            {
              'Fn::Select': [
                1,
                {
                  'Fn::Split': [
                    '||',
                    {
                      Ref: 'AssetParametersa94977ede0211fd3b45efa33d6d8d1d7bbe0c5a96d977139d8b16abfa96fe9cbS3VersionKeyD9ACE665',
                    },
                  ],
                },
              ],
            },
          ],
        ],
      },
    ],
    DestinationBucketName: {
      Ref: 'DestC383B82A',
    },
  });
});

test('fails if local asset is a non-zip file', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // THEN
  expect(() => new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website', 'index.html'))],
    destinationBucket: bucket,
  })).toThrow(/Asset path must be either a \.zip file or a directory/);
});

test('deploy from a local .zip file', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
  });
});

test('AWS_CA_BUNDLE is set', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.data('test', 'test')],
    destinationBucket: bucket,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Environment: {
      Variables: {
        AWS_CA_BUNDLE: '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem',
      },
    },
  });
});

test('deploy from a local .zip file when efs is enabled', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    useEfs: true,
    vpc: new ec2.Vpc(stack, 'Vpc'),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Environment: {
      Variables: {
        MOUNT_PATH: '/mnt/lambda',
      },
    },
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
                Ref: 'BucketDeploymentEFSVPCc8fd940acb9a3f95ad0e87fb4c3a2482b1900ba175AccessPoint557A73A5',
              },
            ],
          ],
        },
        LocalMountPath: '/mnt/lambda',
      },
    ],
    Layers: [
      {
        Ref: 'DeployAwsCliLayer8445CB38',
      },
    ],
    VpcConfig: {
      SecurityGroupIds: [
        {
          'Fn::GetAtt': [
            'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756Cc8fd940acb9a3f95ad0e87fb4c3a2482b1900ba175SecurityGroup3E7AAF58',
            'GroupId',
          ],
        },
      ],
      SubnetIds: [
        {
          Ref: 'VpcPrivateSubnet1Subnet536B997A',
        },
        {
          Ref: 'VpcPrivateSubnet2Subnet3788AAA1',
        },
      ],
    },
  });
});

testDeprecated('honors passed asset options', () => {
  // The 'exclude' property is deprecated and not deprecated in AssetOptions interface.
  // The interface through a complex set of inheritance chain has a 'exclude' prop that is deprecated
  // and another 'exclude' prop that is not deprecated.
  // Using 'testDeprecated' block here since there's no way to work around this craziness.
  // When the deprecated property is removed from source, this block can be dropped.

  // GIVEN
  const app = new cdk.App({ context: { [cxapi.NEW_STYLE_STACK_SYNTHESIS_CONTEXT]: false } });
  const stack = new cdk.Stack(app);
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'), {
      exclude: ['*', '!index.html'],
    })],
    destinationBucket: bucket,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    ServiceToken: {
      'Fn::GetAtt': [
        'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756C81C01536',
        'Arn',
      ],
    },
    SourceBucketNames: [{
      Ref: 'AssetParametersa4d0f1d9c73aa029fd432ca3e640d46745f490023a241d0127f3351773a8938eS3Bucket02009982',
    }],
    SourceObjectKeys: [{
      'Fn::Join': [
        '',
        [
          {
            'Fn::Select': [
              0,
              {
                'Fn::Split': [
                  '||',
                  {
                    Ref: 'AssetParametersa4d0f1d9c73aa029fd432ca3e640d46745f490023a241d0127f3351773a8938eS3VersionKey07726F25',
                  },
                ],
              },
            ],
          },
          {
            'Fn::Select': [
              1,
              {
                'Fn::Split': [
                  '||',
                  {
                    Ref: 'AssetParametersa4d0f1d9c73aa029fd432ca3e640d46745f490023a241d0127f3351773a8938eS3VersionKey07726F25',
                  },
                ],
              },
            ],
          },
        ],
      ],
    }],
    DestinationBucketName: {
      Ref: 'DestC383B82A',
    },
  });
});

test('retainOnDelete can be used to retain files when resource is deleted', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    retainOnDelete: true,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    RetainOnDelete: true,
  });
});

test('user metadata is correctly transformed', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    metadata: {
      A: '1',
      B: '2',
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    UserMetadata: { a: '1', b: '2' },
  });
});

test('system metadata is correctly transformed', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const expiration = cdk.Expiration.after(cdk.Duration.hours(12));

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    contentType: 'text/html',
    contentLanguage: 'en',
    storageClass: s3deploy.StorageClass.INTELLIGENT_TIERING,
    contentDisposition: 'inline',
    serverSideEncryption: s3deploy.ServerSideEncryption.AWS_KMS,
    serverSideEncryptionAwsKmsKeyId: 'mykey',
    serverSideEncryptionCustomerAlgorithm: 'rot13',
    websiteRedirectLocation: 'example',
    cacheControl: [s3deploy.CacheControl.setPublic(), s3deploy.CacheControl.maxAge(cdk.Duration.hours(1))],
    expires: expiration,
    accessControl: s3.BucketAccessControl.BUCKET_OWNER_FULL_CONTROL,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    SystemMetadata: {
      'content-type': 'text/html',
      'content-language': 'en',
      'content-disposition': 'inline',
      'storage-class': 'INTELLIGENT_TIERING',
      'sse': 'aws:kms',
      'sse-kms-key-id': 'mykey',
      'cache-control': 'public, max-age=3600',
      'expires': expiration.date.toUTCString(),
      'sse-c-copy-source': 'rot13',
      'website-redirect': 'example',
      'acl': 'bucket-owner-full-control',
    },
  });
});

// type checking structure that forces to update it if BucketAccessControl changes
// see `--acl` here: https://docs.aws.amazon.com/cli/latest/reference/s3/sync.html
const accessControlMap: Record<s3.BucketAccessControl, string> = {
  [s3.BucketAccessControl.PRIVATE]: 'private',
  [s3.BucketAccessControl.PUBLIC_READ]: 'public-read',
  [s3.BucketAccessControl.PUBLIC_READ_WRITE]: 'public-read-write',
  [s3.BucketAccessControl.AUTHENTICATED_READ]: 'authenticated-read',
  [s3.BucketAccessControl.AWS_EXEC_READ]: 'aws-exec-read',
  [s3.BucketAccessControl.BUCKET_OWNER_READ]: 'bucket-owner-read',
  [s3.BucketAccessControl.BUCKET_OWNER_FULL_CONTROL]: 'bucket-owner-full-control',
  [s3.BucketAccessControl.LOG_DELIVERY_WRITE]: 'log-delivery-write',
};

test.each(Object.entries(accessControlMap) as [s3.BucketAccessControl, string][])(
  'system metadata acl %s is correctly transformed',
  (accessControl, systemMetadataKeyword) => {
    // GIVEN
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'Dest');

    // WHEN
    new s3deploy.BucketDeployment(stack, 'Deploy', {
      sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
      destinationBucket: bucket,
      accessControl: accessControl,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
      SystemMetadata: {
        acl: systemMetadataKeyword,
      },
    });
  },
);

test('expires type has correct values', () => {
  expect(cdk.Expiration.atDate(new Date('Sun, 26 Jan 2020 00:53:20 GMT')).date.toUTCString()).toEqual('Sun, 26 Jan 2020 00:53:20 GMT');
  expect(cdk.Expiration.atTimestamp(1580000000000).date.toUTCString()).toEqual('Sun, 26 Jan 2020 00:53:20 GMT');
  expect(Math.abs(new Date(cdk.Expiration.after(cdk.Duration.minutes(10)).date.toUTCString()).getTime() - (Date.now() + 600000)) < 15000).toBeTruthy();
  expect(cdk.Expiration.fromString('Tue, 04 Feb 2020 08:45:33 GMT').date.toUTCString()).toEqual('Tue, 04 Feb 2020 08:45:33 GMT');
});

test('cache control type has correct values', () => {
  expect(s3deploy.CacheControl.mustRevalidate().value).toEqual('must-revalidate');
  expect(s3deploy.CacheControl.noCache().value).toEqual('no-cache');
  expect(s3deploy.CacheControl.noTransform().value).toEqual('no-transform');
  expect(s3deploy.CacheControl.noStore().value).toEqual('no-store');
  expect(s3deploy.CacheControl.mustUnderstand().value).toEqual('must-understand');
  expect(s3deploy.CacheControl.setPublic().value).toEqual('public');
  expect(s3deploy.CacheControl.setPrivate().value).toEqual('private');
  expect(s3deploy.CacheControl.immutable().value).toEqual('immutable');
  expect(s3deploy.CacheControl.proxyRevalidate().value).toEqual('proxy-revalidate');
  expect(s3deploy.CacheControl.maxAge(cdk.Duration.minutes(1)).value).toEqual('max-age=60');
  expect(s3deploy.CacheControl.sMaxAge(cdk.Duration.minutes(1)).value).toEqual('s-maxage=60');
  expect(s3deploy.CacheControl.staleWhileRevalidate(cdk.Duration.minutes(1)).value).toEqual('stale-while-revalidate=60');
  expect(s3deploy.CacheControl.staleIfError(cdk.Duration.minutes(1)).value).toEqual('stale-if-error=60');
  expect(s3deploy.CacheControl.fromString('only-if-cached').value).toEqual('only-if-cached');
});

test('storage class type has correct values', () => {
  expect(s3deploy.StorageClass.STANDARD).toEqual('STANDARD');
  expect(s3deploy.StorageClass.REDUCED_REDUNDANCY).toEqual('REDUCED_REDUNDANCY');
  expect(s3deploy.StorageClass.STANDARD_IA).toEqual('STANDARD_IA');
  expect(s3deploy.StorageClass.ONEZONE_IA).toEqual('ONEZONE_IA');
  expect(s3deploy.StorageClass.INTELLIGENT_TIERING).toEqual('INTELLIGENT_TIERING');
  expect(s3deploy.StorageClass.GLACIER).toEqual('GLACIER');
  expect(s3deploy.StorageClass.DEEP_ARCHIVE).toEqual('DEEP_ARCHIVE');
});

test('server side encryption type has correct values', () => {
  expect(s3deploy.ServerSideEncryption.AES_256).toEqual('AES256');
  expect(s3deploy.ServerSideEncryption.AWS_KMS).toEqual('aws:kms');
});

test('distribution can be used to provide a CloudFront distribution for invalidation', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const distribution = new cloudfront.CloudFrontWebDistribution(stack, 'Distribution', {
    originConfigs: [
      {
        s3OriginSource: {
          s3BucketSource: bucket,
        },
        behaviors: [{ isDefaultBehavior: true }],
      },
    ],
  });

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    distribution,
    distributionPaths: ['/images/*'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    DistributionId: {
      Ref: 'DistributionCFDistribution882A7313',
    },
    DistributionPaths: ['/images/*'],
  });
});

test('invalidation can happen without distributionPaths provided', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const distribution = new cloudfront.CloudFrontWebDistribution(stack, 'Distribution', {
    originConfigs: [
      {
        s3OriginSource: {
          s3BucketSource: bucket,
        },
        behaviors: [{ isDefaultBehavior: true }],
      },
    ],
  });

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    distribution,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    DistributionId: {
      Ref: 'DistributionCFDistribution882A7313',
    },
  });
});

test('fails if distribution paths provided but not distribution ID', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // THEN
  expect(() => new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website', 'index.html'))],
    destinationBucket: bucket,
    distributionPaths: ['/images/*'],
  })).toThrow(/Distribution must be specified if distribution paths are specified/);
});

test('fails if distribution paths don\'t start with "/"', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const distribution = new cloudfront.CloudFrontWebDistribution(stack, 'Distribution', {
    originConfigs: [
      {
        s3OriginSource: {
          s3BucketSource: bucket,
        },
        behaviors: [{ isDefaultBehavior: true }],
      },
    ],
  });

  // THEN
  expect(() => new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    distribution,
    distributionPaths: ['images/*'],
  })).toThrow(/Distribution paths must start with "\/"/);
});

test('lambda execution role gets permissions to read from the source bucket and read/write in destination', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const source = new s3.Bucket(stack, 'Source');
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.bucket(source, 'file.zip')],
    destinationBucket: bucket,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [
        {
          Action: [
            's3:GetObject*',
            's3:GetBucket*',
            's3:List*',
          ],
          Effect: 'Allow',
          Resource: [
            {
              'Fn::GetAtt': [
                'Source71E471F1',
                'Arn',
              ],
            },
            {
              'Fn::Join': [
                '',
                [
                  {
                    'Fn::GetAtt': [
                      'Source71E471F1',
                      'Arn',
                    ],
                  },
                  '/*',
                ],
              ],
            },
          ],
        },
        {
          Action: [
            's3:GetObject*',
            's3:GetBucket*',
            's3:List*',
            's3:DeleteObject*',
            's3:PutObject',
            's3:PutObjectLegalHold',
            's3:PutObjectRetention',
            's3:PutObjectTagging',
            's3:PutObjectVersionTagging',
            's3:Abort*',
          ],
          Effect: 'Allow',
          Resource: [
            {
              'Fn::GetAtt': [
                'DestC383B82A',
                'Arn',
              ],
            },
            {
              'Fn::Join': [
                '',
                [
                  {
                    'Fn::GetAtt': [
                      'DestC383B82A',
                      'Arn',
                    ],
                  },
                  '/*',
                ],
              ],
            },
          ],
        },
      ],
      Version: '2012-10-17',
    },
    PolicyName: 'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756CServiceRoleDefaultPolicy88902FDF',
    Roles: [
      {
        Ref: 'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756CServiceRole89A01265',
      },
    ],
  });
});

test('lambda execution role gets putObjectAcl permission when deploying with accessControl', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const source = new s3.Bucket(stack, 'Source');
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.bucket(source, 'file.zip')],
    destinationBucket: bucket,
    accessControl: s3.BucketAccessControl.PUBLIC_READ,
  });

  // THEN
  const map = Template.fromStack(stack).findResources('AWS::IAM::Policy');
  expect(map).toBeDefined();
  const resource = map[Object.keys(map)[0]];
  expect(resource.Properties.PolicyDocument.Statement).toContainEqual({
    Action: [
      's3:PutObjectAcl',
      's3:PutObjectVersionAcl',
    ],
    Effect: 'Allow',
    Resource: {
      'Fn::Join': [
        '',
        [
          {
            'Fn::GetAtt': [
              'DestC383B82A',
              'Arn',
            ],
          },
          '/*',
        ],
      ],
    },
  });
});

test('default memory limit is 1024MB', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    // memoryLimit not specified - should default to 1024MB
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', { MemorySize: 1024 });
});

test('deployment handler runs on ARM_64', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Architectures: ['arm64'],
  });
});

test('the ARM_64 handler is still a singleton shared across deployments', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy1', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
  });
  new s3deploy.BucketDeployment(stack, 'Deploy2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
  });

  // THEN - a single handler, on arm64
  const template = Template.fromStack(stack);
  template.resourceCountIs('AWS::Lambda::Function', 1);
  template.hasResourceProperties('AWS::Lambda::Function', {
    Architectures: ['arm64'],
  });
});

test('memoryLimit can be used to specify the memory limit for the deployment resource handler', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  // we define 3 deployments with 2 different memory configurations
  new s3deploy.BucketDeployment(stack, 'Deploy256-1', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    memoryLimit: 256,
  });

  new s3deploy.BucketDeployment(stack, 'Deploy256-2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    memoryLimit: 256,
  });

  new s3deploy.BucketDeployment(stack, 'Deploy1024', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    memoryLimit: 1024,
  });

  // THEN
  // we expect to find only two handlers, one for each configuration
  Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 2);
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', { MemorySize: 256 });
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', { MemorySize: 1024 });
});

test('ephemeralStorageSize can be used to specify the storage size for the deployment resource handler', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  // we define 3 deployments with 2 different memory configurations
  new s3deploy.BucketDeployment(stack, 'Deploy256-1', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    ephemeralStorageSize: cdk.Size.mebibytes(512),
  });

  new s3deploy.BucketDeployment(stack, 'Deploy256-2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    ephemeralStorageSize: cdk.Size.mebibytes(512),
  });

  new s3deploy.BucketDeployment(stack, 'Deploy1024', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    ephemeralStorageSize: cdk.Size.mebibytes(1024),
  });

  // THEN
  // we expect to find only two handlers, one for each configuration
  Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 2);
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    EphemeralStorage: {
      Size: 512,
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    EphemeralStorage: {
      Size: 1024,
    },
  });
});

test('deployment allows custom role to be supplied', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const existingRole = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('lambda.amazon.com'),
  });

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithRole', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    role: existingRole,
  });

  // THEN
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);
  Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Role: {
      'Fn::GetAtt': [
        'Role1ABCC5F0',
        'Arn',
      ],
    },
  });
});

test('deploy without deleting missing files from destination', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    prune: false,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Prune: false,
  });
});

test('deploy with excluded files from destination', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    exclude: ['sample.js'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Exclude: ['sample.js'],
  });
});

test('deploy with included files from destination', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    include: ['sample.js'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Include: ['sample.js'],
  });
});

test('deploy with excluded and included files from destination', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    exclude: ['sample/*'],
    include: ['sample/include.json'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Exclude: ['sample/*'],
    Include: ['sample/include.json'],
  });
});

test('deploy with multiple exclude and include filters', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    exclude: ['sample/*', 'another/*'],
    include: ['sample/include.json', 'another/include.json'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Exclude: ['sample/*', 'another/*'],
    Include: ['sample/include.json', 'another/include.json'],
  });
});

test('deploy without extracting files in destination', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    extract: false,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Extract: false,
  });
});

test('deploy without extracting files in destination and get the object key', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  const deployment = new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website.zip'))],
    destinationBucket: bucket,
    extract: false,
  });

  // Tests object key retrieval.
  void(cdk.Fn.select(0, deployment.objectKeys));

  // THEN
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    Extract: false,
  });
});

test('given a source with markers and extract is false, BucketDeployment throws an error', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const topic: sns.Topic = new sns.Topic(stack, 'SomeTopic1', {});

  // WHEN
  const file = s3deploy.Source.jsonData('MyJsonObject', {
    'config.json': {
      Foo: {
        Bar: topic.topicArn, // marker
      },
    },
  });
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [file],
    destinationBucket: bucket,
    extract: false,
  });

  // THEN
  expect(() => {
    Template.fromStack(stack);
  }).toThrow('Some sources are incompatible with extract=false; sources with deploy-time values (such as \'snsTopic.topicArn\') must be extracted.');
});

test('deployment allows vpc to be implicitly supplied to lambda', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const vpc: ec2.IVpc = new ec2.Vpc(stack, 'SomeVpc1', {});

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc1', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    vpc,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    VpcConfig: {
      SecurityGroupIds: [
        {
          'Fn::GetAtt': [
            'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756Cc81cec990a9a5d64a5922e5708ad8067eeb95c53d1SecurityGroup881B9147',
            'GroupId',
          ],
        },
      ],
      SubnetIds: [
        {
          Ref: 'SomeVpc1PrivateSubnet1SubnetCBA5DD76',
        },
        {
          Ref: 'SomeVpc1PrivateSubnet2SubnetD4B3A566',
        },
      ],
    },
  });
});

test('deployment allows vpc and subnets to be implicitly supplied to lambda', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const vpc: ec2.IVpc = new ec2.Vpc(stack, 'SomeVpc2', {});
  new ec2.PrivateSubnet(stack, 'SomeSubnet', {
    vpcId: vpc.vpcId,
    availabilityZone: vpc.availabilityZones[0],
    cidrBlock: vpc.vpcCidrBlock,
  });

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    vpc,
    vpcSubnets: {
      availabilityZones: [vpc.availabilityZones[0]],
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    VpcConfig: {
      SecurityGroupIds: [
        {
          'Fn::GetAtt': [
            'CustomCDKBucketDeployment8693BB64968944B69AAFB0CC9EB8756Cc8a39596cb8641929fcf6a288bc9db5ab7b0f656adSecurityGroup11274779',
            'GroupId',
          ],
        },
      ],
      SubnetIds: [
        {
          Ref: 'SomeVpc2PrivateSubnet1SubnetB1DC76FF',
        },
      ],
    },
  });
});

test('deployment allows security groups to be implicitly supplied to lambda', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const vpc: ec2.IVpc = new ec2.Vpc(stack, 'SomeVpc', {});
  const securityGroups: ec2.SecurityGroup = new ec2.SecurityGroup(stack, 'SomeSecurityGroup', {
    vpc: vpc,
    securityGroupName: 'SomeSecurityGroup',
  });

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc1', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    vpc,
    securityGroups: [securityGroups],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    VpcConfig: {
      SecurityGroupIds: [
        {
          'Fn::GetAtt': [
            Match.stringLikeRegexp('SomeSecurityGroup'),
            'GroupId',
          ],
        },
      ],
      SubnetIds: Match.arrayWith([
        { Ref: Match.stringLikeRegexp('SomeVpc.*Subnet.*') },
        { Ref: Match.stringLikeRegexp('SomeVpc.*Subnet.*') },
      ]),
    },
  });
});

test('s3 deployment bucket is identical to destination bucket', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  const bd = new s3deploy.BucketDeployment(stack, 'Deployment', {
    destinationBucket: bucket,
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  });

  // Call this function
  void(bd.deployedBucket);

  // THEN
  const template = Template.fromStack(stack);
  template.hasResourceProperties('Custom::CDKBucketDeployment', {
    // Since this utilizes GetAtt, we know CFN will deploy the bucket first
    // before deploying other resources that rely call the destination bucket.
    DestinationBucketArn: { 'Fn::GetAtt': ['DestC383B82A', 'Arn'] },
  });
});

test('s3 deployed bucket in a different region has correct website url', () => {
  // GIVEN
  const stack = new cdk.Stack(undefined, undefined, {
    env: {
      region: 'us-east-1',
    },
  });
  const bucket = s3.Bucket.fromBucketAttributes(stack, 'Dest', {
    bucketName: 'my-bucket',
    // Bucket is in a different region than stack
    region: 'eu-central-1',
  });

  // WHEN
  const bd = new s3deploy.BucketDeployment(stack, 'Deployment', {
    destinationBucket: bucket,
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  });
  const websiteUrl = stack.resolve(bd.deployedBucket.bucketWebsiteUrl);

  // THEN
  // eu-central-1 uses website endpoint format with a `.`
  // see https://docs.aws.amazon.com/general/latest/gr/s3.html#s3_website_region_endpoints
  expect(JSON.stringify(websiteUrl)).toContain('.s3-website.eu-central-1.');
});

test('using deployment bucket references the destination bucket by means of the CustomResource', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const deployment = new s3deploy.BucketDeployment(stack, 'Deployment', {
    destinationBucket: bucket,
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
  });

  // WHEN
  new cdk.CfnOutput(stack, 'DestinationArn', {
    value: deployment.deployedBucket.bucketArn,
  });
  new cdk.CfnOutput(stack, 'DestinationName', {
    value: deployment.deployedBucket.bucketName,
  });

  // THEN

  const template = Template.fromStack(stack);
  expect(template.findOutputs('*')).toEqual({
    DestinationArn: {
      Value: { 'Fn::GetAtt': ['DeploymentCustomResource47E8B2E6', 'DestinationBucketArn'] },
    },
    DestinationName: {
      Value: {
        'Fn::Select': [0, {
          'Fn::Split': ['/', {
            'Fn::Select': [5, {
              'Fn::Split': [':',
                { 'Fn::GetAtt': ['DeploymentCustomResource47E8B2E6', 'DestinationBucketArn'] }],
            }],
          }],
        }],
      },
    },
  });
});

test('resource id includes memory and vpc', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const vpc: ec2.IVpc = new ec2.Vpc(stack, 'SomeVpc2', {});

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    vpc,
    memoryLimit: 256,
  });

  // THEN
  Template.fromStack(stack).templateMatches({
    Resources: Match.objectLike({
      DeployWithVpc2CustomResource256MiBc8a39596cb8641929fcf6a288bc9db5ab7b0f656ad3C5F6E78: Match.objectLike({
        Type: 'Custom::CDKBucketDeployment',
      }),
    }),
  });
});

test('bucket includes custom resource owner tag', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const vpc: ec2.IVpc = new ec2.Vpc(stack, 'SomeVpc2', {});

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    destinationKeyPrefix: '/a/b/c',
    vpc,
    memoryLimit: 256,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
    Tags: [{
      Key: 'aws-cdk:cr-owned:/a/b/c:971e1fa8',
      Value: 'true',
    }],
  });
});

test('throws if destinationKeyPrefix is too long', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  expect(() => new s3deploy.BucketDeployment(stack, 'DeployWithVpc2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    destinationKeyPrefix: '/this/is/a/random/key/prefix/that/is/a/lot/of/characters/do/we/think/that/it/will/ever/be/this/long??????',
    memoryLimit: 256,
  })).toThrow(/The BucketDeployment construct requires that/);
});

test('skips destinationKeyPrefix validation if token', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  // trick the cdk into creating a very long token
  const prefixToken = cdk.Token.asString(5, { displayHint: 'longlonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglonglong' });
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    destinationKeyPrefix: prefixToken,
    memoryLimit: 256,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    DestinationBucketKeyPrefix: 5,
  });
});

test('bucket has multiple deployments', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  const vpc: ec2.IVpc = new ec2.Vpc(stack, 'SomeVpc2', {});

  // WHEN
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc2', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    destinationKeyPrefix: '/a/b/c',
    vpc,
    memoryLimit: 256,
  });

  new s3deploy.BucketDeployment(stack, 'DeployWithVpc2Exclude', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'), {
      exclude: ['index.html'],
    })],
    destinationBucket: bucket,
    destinationKeyPrefix: '/a/b/c',
    vpc,
    memoryLimit: 256,
  });

  new s3deploy.BucketDeployment(stack, 'DeployWithVpc3', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    destinationKeyPrefix: '/x/z',
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
    Tags: [
      {
        Key: 'aws-cdk:cr-owned:/a/b/c:6da0a4ab',
        Value: 'true',
      },
      {
        Key: 'aws-cdk:cr-owned:/a/b/c:971e1fa8',
        Value: 'true',
      },
      {
        Key: 'aws-cdk:cr-owned:/x/z:2db04622',
        Value: 'true',
      },
    ],
  });
});

test('"SourceMarkers" is not included if none of the sources have markers', () => {
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Bucket');
  new s3deploy.BucketDeployment(stack, 'DeployWithVpc3', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
  });

  const map = Template.fromStack(stack).findResources('Custom::CDKBucketDeployment');
  expect(map).toBeDefined();
  const resource = map[Object.keys(map)[0]];
  expect(Object.keys(resource.Properties)).toStrictEqual([
    'ServiceToken',
    'SourceBucketNames',
    'SourceObjectKeys',
    'DestinationBucketName',
    'WaitForDistributionInvalidation',
    'Prune',
    'OutputObjectKeys',
  ]);
});

test('Source.data() can be used to create a file with string contents', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const source = s3deploy.Source.data('my/path.txt', 'hello, world');

  new s3deploy.BucketDeployment(stack, 'DeployWithVpc3', {
    sources: [source],
    destinationBucket: bucket,
    destinationKeyPrefix: '/x/z',
  });

  const result = app.synth();
  const content = readDataFile(result, 'my/path.txt');
  expect(content).toStrictEqual('hello, world');
});

test('Source.jsonData() can be used to create a file with a JSON object', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const config = {
    foo: 'bar',
    baz: null,
    sub: {
      hello: bucket.bucketArn,
    },
    [bucket.bucketName]: 'Token can be a key as well!',
  };

  new s3deploy.BucketDeployment(stack, 'DeployWithVpc3', {
    sources: [s3deploy.Source.jsonData('app-config.json', config)],
    destinationBucket: bucket,
  });

  const result = app.synth();
  expect(readDataFile(result, 'app-config.json')).toBe('{"foo":"bar","baz":null,"sub":{"hello":<<marker:0xbaba:0>>},"<<marker:0xbaba:1>>":"Token can be a key as well!"}');

  // verify marker is mapped to the bucket ARN in the resource props
  Template.fromJSON(result.stacks[0].template).hasResourceProperties('Custom::CDKBucketDeployment', {
    SourceMarkers: [
      {
        '<<marker:0xbaba:0>>': {
          'Fn::Join': ['', ['"',
            { 'Fn::GetAtt': ['Bucket83908E77', 'Arn'] },
            '"']],
        },
        '<<marker:0xbaba:1>>': {
          Ref: 'Bucket83908E77',
        },
      },
    ],
  });
});

test('Source.jsonData() can be used with list tokens', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');
  const readParam = ssm.StringListParameter.fromStringListParameterName(
    stack,
    'ReadParam',
    '/repro/subnets',
  );

  const config = {
    foo: 'bar',
    sub: {
      hello: readParam.stringListValue,
    },
  };

  new s3deploy.BucketDeployment(stack, 'DeployWithVpc3', {
    sources: [s3deploy.Source.jsonData('app-config.json', config)],
    destinationBucket: bucket,
  });

  const result = app.synth();
  expect(readDataFile(result, 'app-config.json')).toBe('{"foo":"bar","sub":{"hello":<<marker:0xbaba:0>>}}');

  // verify marker is mapped to the bucket ARN in the resource props
  Template.fromJSON(result.stacks[0].template).hasResourceProperties('Custom::CDKBucketDeployment', {
    SourceMarkers: [
      {
        '<<marker:0xbaba:0>>': {
          'Fn::GetAtt': ['CdkJsonStringifyFnSplitresolvessmreprosubnets67ED8B00', 'Value'],
        },
      },
    ],
  });
});

test('Source.yamlData() can be used to create a file with YAML content', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const config = {
    foo: 'bar',
    sub: {
      hello: bucket.bucketArn,
    },
  };

  new s3deploy.BucketDeployment(stack, 'DeployWithVpc3', {
    sources: [s3deploy.Source.yamlData('app-config.yaml', config)],
    destinationBucket: bucket,
  });

  const result = app.synth();
  const output = readDataFile(result, 'app-config.yaml');
  expect(output.trim()).toEqual([
    'foo: bar',
    'sub:',
    '  hello: <<marker:0xbaba:0>>',
  ].join('\n'));
});

test('can add sources with addSource', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');
  const deployment = new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.data('my/path.txt', 'helloWorld')],
    destinationBucket: bucket,
  });
  deployment.addSource(s3deploy.Source.data('my/other/path.txt', 'hello world'));

  const result = app.synth();
  const content = readDataFile(result, 'my/path.txt');
  const content2 = readDataFile(result, 'my/other/path.txt');
  expect(content).toStrictEqual('helloWorld');
  expect(content2).toStrictEqual('hello world');
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    SourceMarkers: [
      {},
      {},
    ],
  });
});

test('deploy with payload signing enabled', () => {
  // GIVEN
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');

  // WHEN
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    signContent: true,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    SignContent: true,
  });
});

test('if any source has markers then all sources have markers', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');
  const deployment = new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.data('my/path.txt', 'helloWorld')],
    destinationBucket: bucket,
  });
  deployment.addSource(s3deploy.Source.asset(path.join(__dirname, 'my-website')));

  const result = app.synth();
  const content = readDataFile(result, 'my/path.txt');
  expect(content).toStrictEqual('helloWorld');
  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    SourceMarkers: [
      {},
      {},
    ],
  });
});

test('DeployTimeSubstitutedFile can be used to add substitutions in a file', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const deployment = new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'),
    destinationBucket: bucket,
    substitutions: {
      testMethod: 'changedTestMethodSuccess',
      mock: 'changedMockTypeSuccess',
    },
  });

  const result = app.synth();
  const content = readDataFile(result, deployment.objectKey);
  expect(content).not.toContain('testMethod');
  expect(content).toContain('changedTestMethodSuccess');
  expect(content).not.toContain('mock');
  expect(content).toContain('changedMockTypeSuccess');
});

test('DeployTimeSubstitutedFile can be used to add substitutions in a file with a defined destination key', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');
  const destinationKey = 'helloworld.yaml';
  new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'),
    destinationKey: destinationKey,
    destinationBucket: bucket,
    substitutions: {
      testMethod: 'changedTestMethodSuccess',
      mock: 'changedMockTypeSuccess',
    },
  });

  const result = app.synth();
  const content = readDataFile(result, destinationKey);
  expect(content).not.toContain('testMethod');
  expect(content).toContain('changedTestMethodSuccess');
  expect(content).not.toContain('mock');
  expect(content).toContain('changedMockTypeSuccess');
});

test('DeployTimeSubstitutedFile throws error when source file path is invalid', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  expect(() => {
    new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
      source: path.join(__dirname, 'non-existent-file.yaml'),
      destinationBucket: bucket,
      substitutions: {
        testMethod: 'changedTestMethodSuccess',
        mock: 'changedMockTypeSuccess',
      },
    });
  }).toThrow(`No file found at 'source' path ${path.join(__dirname, 'non-existent-file.yaml')}`);
});

test('DeployTimeSubstitutedFile does not make substitutions when no substitutions are passed in', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const originalFileData = readFileSync(path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'), 'utf8');

  const deployment = new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'),
    destinationBucket: bucket,
    substitutions: { },
  });

  const result = app.synth();
  const assetFileFromOutput = readDataFile(result, deployment.objectKey);
  expect(originalFileData).toStrictEqual(assetFileFromOutput);
});

test('DeployTimeSubstitutedFile does not substitute nested variables', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const originalFileData = readFileSync(path.join(__dirname, 'file-substitution-test', 'sample-definition-nested-vars.yaml'), 'utf8');

  const deployment = new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition-nested-vars.yaml'),
    destinationBucket: bucket,
    substitutions: {
      foo: 'replacement1',
      bar: 'replacement2',
    },
  });

  const result = app.synth();
  const assetFileFromOutput = readDataFile(result, deployment.objectKey);
  expect(originalFileData).not.toStrictEqual(assetFileFromOutput);
  expect(assetFileFromOutput).toContain('{{ replacement1 }}');
  expect(assetFileFromOutput).toContain('{{ foo replacement2 }}');
});

test('DeployTimeSubstitutedFile does not substitute nested variables', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const originalFileData = readFileSync(path.join(__dirname, 'file-substitution-test', 'sample-definition-nested-vars.yaml'), 'utf8');

  const deployment = new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition-nested-vars.yaml'),
    destinationBucket: bucket,
    substitutions: {
      foo: 'replacement1',
      bar: 'replacement2',
    },
  });

  const result = app.synth();
  const assetFileFromOutput = readDataFile(result, deployment.objectKey);
  expect(originalFileData).not.toStrictEqual(assetFileFromOutput);
  expect(assetFileFromOutput).toContain('{{ replacement1 }}');
  expect(assetFileFromOutput).toContain('{{ foo replacement2 }}');
});

test('DeployTimeSubstitutedFile does not double substitute already replaced variables', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');

  const originalFileData = readFileSync(path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'), 'utf8');

  const deployment = new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'),
    destinationBucket: bucket,
    substitutions: {
      foo: 'bar',
      bar: 'zee',
    },
  });

  const result = app.synth();
  const assetFileFromOutput = readDataFile(result, deployment.objectKey);
  expect(originalFileData).not.toStrictEqual(assetFileFromOutput);
  expect(assetFileFromOutput).toContain('bar');
  expect(assetFileFromOutput).not.toContain('foo');
  expect(assetFileFromOutput).not.toContain('zee');
});

function readDataFile(casm: cxapi.CloudAssembly, relativePath: string): string {
  const assetDirs = readdirSync(casm.directory).filter(f => f.startsWith('asset.'));
  for (const dir of assetDirs) {
    const candidate = path.join(casm.directory, dir, relativePath);
    if (existsSync(candidate)) {
      return readFileSync(candidate, 'utf8');
    }
  }

  throw new UnscopedValidationError(lit`FileFoundAssetsAssembly`, `File ${relativePath} not found in any of the assets of the assembly`);
}

test('DeployTimeSubstitutedFile allows custom role to be supplied', () => {
  const app = new cdk.App();
  const stack = new cdk.Stack(app, 'Test');
  const bucket = new s3.Bucket(stack, 'Bucket');
  const existingRole = new iam.Role(stack, 'Role', {
    assumedBy: new iam.ServicePrincipal('lambda.amazon.com'),
  });

  new s3deploy.DeployTimeSubstitutedFile(stack, 'MyFile', {
    source: path.join(__dirname, 'file-substitution-test', 'sample-definition.yaml'),
    destinationBucket: bucket,
    substitutions: {
      testMethod: 'changedTestMethodSuccess',
      mock: 'changedMockTypeSuccess',
    },
    role: existingRole,
  });

  Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);
  Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Role: {
      'Fn::GetAtt': [
        'Role1ABCC5F0',
        'Arn',
      ],
    },
  });
});

test('outputObjectKeys to control SourceObjectKeys being sent back', () => {
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
    outputObjectKeys: false,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    OutputObjectKeys: false,
  });
});

test('outputObjectKeys default value is true', () => {
  const stack = new cdk.Stack();
  const bucket = new s3.Bucket(stack, 'Dest');
  new s3deploy.BucketDeployment(stack, 'Deploy', {
    sources: [s3deploy.Source.asset(path.join(__dirname, 'my-website'))],
    destinationBucket: bucket,
  });

  Template.fromStack(stack).hasResourceProperties('Custom::CDKBucketDeployment', {
    OutputObjectKeys: true,
  });
});
