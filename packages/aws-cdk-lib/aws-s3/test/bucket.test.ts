import { EOL } from 'os';
import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Annotations, Match, Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import { CfnKey } from '../../aws-kms';
import * as cdk from '../../core';
import { Tags } from '../../core';
import * as cxapi from '../../cx-api';
import * as s3 from '../lib';
import { BucketGrants, CfnBucket } from '../lib';

// to make it easy to copy & paste from output:
/* eslint-disable @stylistic/quote-props */

describe('bucket', () => {
  test('default bucket', () => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket');

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('CFN properties are type-validated during resolution', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      bucketName: cdk.Token.asString(5), // Oh no
    });

    expect(() => {
      Template.fromStack(stack).toJSON();
    }).toThrow(/bucketName: 5 should be a string/);
  });

  testDeprecated('bucket with UNENCRYPTED encryption', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.UNENCRYPTED,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with S3_MANAGED encryption', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'BucketEncryption': {
              'ServerSideEncryptionConfiguration': [
                {
                  'ServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'AES256',
                  },
                },
              ],
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with KMS_MANAGED encryption', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.KMS_MANAGED,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'BucketEncryption': {
              'ServerSideEncryptionConfiguration': [
                {
                  'ServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'aws:kms',
                  },
                },
              ],
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with DSSE_MANAGED encryption', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.DSSE_MANAGED,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'BucketEncryption': {
              'ServerSideEncryptionConfiguration': [
                {
                  'ServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'aws:kms:dsse',
                  },
                },
              ],
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('empty blockedEncryptionTypes not allowed', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.DSSE,
      blockedEncryptionTypes: [],
    })).toThrow(/At least one blocked encryption type must be specified/);
  });

  test('other blockedEncryptionTypes are not allowed with NONE', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.KMS_MANAGED,
      blockedEncryptionTypes: [
        s3.BlockedEncryptionType.NONE,
        s3.BlockedEncryptionType.SSE_C,
      ],
    })).toThrow(/If NONE is specified as the blocked encryption type, no other encryption types may be specified/);
  });

  test('bucket with no encryption by default and blockedEncryptionTypes', () => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.UNENCRYPTED,
      blockedEncryptionTypes: [s3.BlockedEncryptionType.SSE_C],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketEncryption: {
        ServerSideEncryptionConfiguration: [
          {
            BlockedEncryptionTypes: {
              EncryptionType: ['SSE-C'],
            },
          },
        ],
      },
    });
  });

  test('bucket with default encryption and blockedEncryptionTypes', () => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockedEncryptionTypes: [s3.BlockedEncryptionType.NONE],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketEncryption: {
        ServerSideEncryptionConfiguration: [
          {
            BlockedEncryptionTypes: {
              EncryptionType: ['NONE'],
            },
            ServerSideEncryptionByDefault: { SSEAlgorithm: 'AES256' },
          },
        ],
      },
    });
  });

  test('bucket with custom blockedEncryptionTypes', () => {
    const stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::W3030', reason: 'Bogus encryption types are only for tests' });

    new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockedEncryptionTypes: [
        s3.BlockedEncryptionType.custom('unknown'),
        s3.BlockedEncryptionType.custom('unsupported'),
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketEncryption: {
        ServerSideEncryptionConfiguration: [
          {
            BlockedEncryptionTypes: {
              EncryptionType: ['unknown', 'unsupported'],
            },
            ServerSideEncryptionByDefault: { SSEAlgorithm: 'AES256' },
          },
        ],
      },
    });
  });

  test('valid bucket names', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket1', {
      bucketName: 'abc.xyz-34ab',
    })).not.toThrow();

    expect(() => new s3.Bucket(stack, 'MyBucket2', {
      bucketName: '124.pp--33',
    })).not.toThrow();
  });

  test('creating bucket with underscore in name throws error', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'TestBucket', { bucketName: 'test_bucket_name' });
    }).toThrow(/Bucket name must only contain lowercase characters and the symbols, period \(\.\) and dash \(-\)/);
  });

  test('importing existing bucket with underscore using fromBucketName works with allowLegacyBucketNaming=true', () => {
    const stack = new cdk.Stack();
    expect(() => {
      s3.Bucket.fromBucketName(stack, 'TestBucket', 'test_bucket_name');
    }).not.toThrow();
  });

  test('importing existing bucket with underscore using fromBucketAttributes works with allowLegacyBucketNaming=true', () => {
    const stack = new cdk.Stack();
    expect(() => {
      s3.Bucket.fromBucketAttributes(stack, 'TestBucket', { bucketName: 'test_bucket_name' });
    }).not.toThrow();
  });

  test('importing existing bucket with underscore using fromBucketArn works with allowLegacyBucketNaming=true', () => {
    const stack = new cdk.Stack();
    expect(() => {
      s3.Bucket.fromBucketArn(stack, 'TestBucket', 'arn:aws:s3:::test_bucket_name');
    }).not.toThrow();
  });

  test('bucket validation skips tokenized values', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      bucketName: cdk.Lazy.string({ produce: () => '_BUCKET' }),
    })).not.toThrow();
  });

  test('fails with message on invalid bucket names', () => {
    const stack = new cdk.Stack();
    const bucket = `-buckEt.-${new Array(65).join('$')}`;
    const expectedErrors = [
      `Invalid S3 bucket name (value: ${bucket})`,
      'Bucket name must be at least 3 and no more than 63 characters',
      'Bucket name must only contain lowercase characters and the symbols, period (.) and dash (-) (offset: 5)',
      'Bucket name must start with a lowercase character or number (offset: 0)',
      `Bucket name must end with a lowercase character or number (offset: ${bucket.length - 1})`,
      'Bucket name must not have dash next to period, or period next to dash, or consecutive periods (offset: 7)',
    ].join(EOL);

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      bucketName: bucket,
    })).toThrow(expectedErrors);
  });

  test('validateBucketName allows underscore when allowLegacyBucketNaming=true', () => {
    expect(() => {
      s3.Bucket.validateBucketName('test_bucket_name', true);
    }).not.toThrow();
  });

  test('validateBucketName does not allow underscore when allowLegacyBucketNaming=false', () => {
    expect(() => {
      s3.Bucket.validateBucketName('test_bucket_name', false);
    }).toThrow(/Bucket name must only contain lowercase characters and the symbols, period \(\.\) and dash \(-\)/);
  });

  test('validateBucketName allows uppercase characters when allowLegacyBucketNaming=true', () => {
    expect(() => {
      s3.Bucket.validateBucketName('Test_Bucket_Name', true);
    }).not.toThrow();
  });

  test('validateBucketName does not allow uppercase characters when allowLegacyBucketNaming=false', () => {
    expect(() => {
      s3.Bucket.validateBucketName('Test_Bucket_Name', false);
    }).toThrow(/Bucket name must only contain lowercase characters and the symbols, period \(\.\) and dash \(-\)/);
  });

  test('validateBucketName does not allow underscore by default', () => {
    expect(() => {
      s3.Bucket.validateBucketName('test_bucket_name');
    }).toThrow(/Bucket name must only contain lowercase characters and the symbols, period \(\.\) and dash \(-\)/);
  });

  test('validateBucketName does not allow uppercase characters by default', () => {
    expect(() => {
      s3.Bucket.validateBucketName('TestBucketName');
    }).toThrow(/Bucket name must only contain lowercase characters and the symbols, period \(\.\) and dash \(-\)/);
  });

  test('fails if bucket name has less than 3 or more than 63 characters', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket1', {
      bucketName: 'a',
    })).toThrow(/at least 3/);

    expect(() => new s3.Bucket(stack, 'MyBucket2', {
      bucketName: new Array(65).join('x'),
    })).toThrow(/no more than 63/);
  });

  test('fails if bucket name has invalid characters', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket1', {
      bucketName: 'b@cket',
    })).toThrow(/offset: 1/);

    expect(() => new s3.Bucket(stack, 'MyBucket2', {
      bucketName: 'bucKet',
    })).toThrow(/offset: 3/);

    expect(() => new s3.Bucket(stack, 'MyBucket3', {
      bucketName: 'bučket',
    })).toThrow(/offset: 2/);
  });

  test('fails if bucket name does not start or end with lowercase character or number', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket1', {
      bucketName: '-ucket',
    })).toThrow(/offset: 0/);

    expect(() => new s3.Bucket(stack, 'MyBucket2', {
      bucketName: 'bucke.',
    })).toThrow(/offset: 5/);
  });

  test('fails only if bucket name has the consecutive symbols (..), (.-), (-.)', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket1', {
      bucketName: 'buc..ket',
    })).toThrow(/offset: 3/);

    expect(() => new s3.Bucket(stack, 'MyBucket2', {
      bucketName: 'buck.-et',
    })).toThrow(/offset: 4/);

    expect(() => new s3.Bucket(stack, 'MyBucket3', {
      bucketName: 'b-.ucket',
    })).toThrow(/offset: 1/);

    expect(() => new s3.Bucket(stack, 'MyBucket4', {
      bucketName: 'bu--cket',
    })).not.toThrow();
  });

  test('fails only if bucket name resembles IP address', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket1', {
      bucketName: '1.2.3.4',
    })).toThrow(/must not resemble an IP address/);

    expect(() => new s3.Bucket(stack, 'MyBucket2', {
      bucketName: '1.2.3',
    })).not.toThrow();

    expect(() => new s3.Bucket(stack, 'MyBucket3', {
      bucketName: '1.2.3.a',
    })).not.toThrow();

    expect(() => new s3.Bucket(stack, 'MyBucket4', {
      bucketName: '1000.2.3.4',
    })).not.toThrow();
  });

  test('fails if encryption key is used with managed encryption', () => {
    const stack = new cdk.Stack();
    const myKey = new kms.Key(stack, 'MyKey');

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.KMS_MANAGED,
      encryptionKey: myKey,
    })).toThrow(/encryptionKey is specified, so 'encryption' must be set to KMS or DSSE/);
  });

  test('fails if encryption key is used with dsse managed encryption', () => {
    const stack = new cdk.Stack();
    const myKey = new kms.Key(stack, 'MyKey');

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.DSSE_MANAGED,
      encryptionKey: myKey,
    })).toThrow(/encryptionKey is specified, so 'encryption' must be set to KMS or DSSE/);
  });

  testDeprecated('fails if encryption key is used with encryption set to UNENCRYPTED', () => {
    const stack = new cdk.Stack();
    const myKey = new kms.Key(stack, 'MyKey');

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.UNENCRYPTED,
      encryptionKey: myKey,
    })).toThrow(/encryptionKey is specified, so 'encryption' must be set to KMS or DSSE/);
  });

  test('fails if encryption key is used with encryption set to S3_MANAGED', () => {
    const stack = new cdk.Stack();
    const myKey = new kms.Key(stack, 'MyKey');

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      encryptionKey: myKey,
    })).toThrow(/encryptionKey is specified, so 'encryption' must be set to KMS or DSSE/);
  });

  test('encryptionKey can specify kms key', () => {
    const stack = new cdk.Stack();

    const encryptionKey = new kms.Key(stack, 'MyKey', { description: 'hello, world' });

    new s3.Bucket(stack, 'MyBucket', { encryptionKey, encryption: s3.BucketEncryption.KMS });

    Template.fromStack(stack).resourceCountIs('AWS::KMS::Key', 1);

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      'BucketEncryption': {
        'ServerSideEncryptionConfiguration': [
          {
            'ServerSideEncryptionByDefault': {
              'KMSMasterKeyID': {
                'Fn::GetAtt': [
                  'MyKey6AB29FA6',
                  'Arn',
                ],
              },
              'SSEAlgorithm': 'aws:kms',
            },
          },
        ],
      },
    });
  });

  test('dsse encryptionKey can specify kms key', () => {
    const stack = new cdk.Stack();

    const encryptionKey = new kms.Key(stack, 'MyKey', { description: 'hello, world' });

    new s3.Bucket(stack, 'MyBucket', { encryptionKey, encryption: s3.BucketEncryption.DSSE });

    Template.fromStack(stack).resourceCountIs('AWS::KMS::Key', 1);

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      'BucketEncryption': {
        'ServerSideEncryptionConfiguration': [
          {
            'ServerSideEncryptionByDefault': {
              'KMSMasterKeyID': {
                'Fn::GetAtt': [
                  'MyKey6AB29FA6',
                  'Arn',
                ],
              },
              'SSEAlgorithm': 'aws:kms:dsse',
            },
          },
        ],
      },
    });
  });

  test('KMS key is generated if encryption is KMS and no encryptionKey is specified', () => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.KMS });

    Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
      'Description': 'Created by Default/MyBucket',
      'EnableKeyRotation': true,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      'BucketEncryption': {
        'ServerSideEncryptionConfiguration': [
          {
            'ServerSideEncryptionByDefault': {
              'KMSMasterKeyID': {
                'Fn::GetAtt': [
                  'MyBucketKeyC17130CF',
                  'Arn',
                ],
              },
              'SSEAlgorithm': 'aws:kms',
            },
          },
        ],
      },
    });
  });

  test('enforceSsl can be enabled', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', { enforceSSL: true });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'UpdateReplacePolicy': 'Retain',
          'DeletionPolicy': 'Retain',
        },
        'MyBucketPolicyE7FBAC7B': {
          'Type': 'AWS::S3::BucketPolicy',
          'Properties': {
            'Bucket': {
              'Ref': 'MyBucketF68F3FF0',
            },
            'PolicyDocument': {
              'Statement': [
                {
                  Action: 's3:*',
                  'Condition': {
                    'Bool': {
                      'aws:SecureTransport': 'false',
                    },
                  },
                  'Effect': 'Deny',
                  'Principal': { AWS: '*' },
                  'Resource': [
                    {
                      'Fn::GetAtt': [
                        'MyBucketF68F3FF0',
                        'Arn',
                      ],
                    },
                    {
                      'Fn::Join': [
                        '',
                        [
                          {
                            'Fn::GetAtt': [
                              'MyBucketF68F3FF0',
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
              'Version': '2012-10-17',
            },
          },
        },
      },
    });
  });

  test('with minimumTLSVersion', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      enforceSSL: true,
      minimumTLSVersion: 1.2,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      'PolicyDocument': {
        'Statement': [
          {
            Action: 's3:*',
            'Condition': {
              'Bool': {
                'aws:SecureTransport': 'false',
              },
            },
            'Effect': 'Deny',
            'Principal': { AWS: '*' },
            'Resource': [
              {
                'Fn::GetAtt': [
                  'MyBucketF68F3FF0',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'MyBucketF68F3FF0',
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
            Action: 's3:*',
            'Condition': {
              'NumericLessThan': {
                's3:TlsVersion': 1.2,
              },
            },
            'Effect': 'Deny',
            'Principal': { AWS: '*' },
            'Resource': [
              {
                'Fn::GetAtt': [
                  'MyBucketF68F3FF0',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'MyBucketF68F3FF0',
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
        'Version': '2012-10-17',
      },
    });
  });

  test('enforceSSL must be enabled for minimumTLSVersion to work', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new s3.Bucket(stack, 'MyBucket1', {
        enforceSSL: false,
        minimumTLSVersion: 1.2,
      });
    }).toThrow(/'enforceSSL' must be enabled for 'minimumTLSVersion' to be applied/);

    expect(() => {
      new s3.Bucket(stack, 'MyBucket2', {
        minimumTLSVersion: 1.2,
      });
    }).toThrow(/'enforceSSL' must be enabled for 'minimumTLSVersion' to be applied/);
  });

  test.each([s3.BucketEncryption.KMS, s3.BucketEncryption.KMS_MANAGED])('bucketKeyEnabled can be enabled with %p encryption', (encryption) => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', { bucketKeyEnabled: true, encryption });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      'BucketEncryption': {
        'ServerSideEncryptionConfiguration': [
          {
            'BucketKeyEnabled': true,
            'ServerSideEncryptionByDefault': Match.objectLike({
              'SSEAlgorithm': 'aws:kms',
            }),
          },
        ],
      },
    });
  });

  test.each([s3.BucketEncryption.DSSE, s3.BucketEncryption.DSSE_MANAGED])('bucketKeyEnabled can be enabled with %p encryption', (encryption) => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', { bucketKeyEnabled: true, encryption });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      'BucketEncryption': {
        'ServerSideEncryptionConfiguration': [
          {
            'BucketKeyEnabled': true,
            'ServerSideEncryptionByDefault': Match.objectLike({
              'SSEAlgorithm': 'aws:kms:dsse',
            }),
          },
        ],
      },
    });
  });

  test('bucketKeyEnabled can be enabled with SSE-S3', () => {
    const stack = new cdk.Stack();

    // WHEN
    new s3.Bucket(stack, 'MyBucket', { bucketKeyEnabled: true, encryption: s3.BucketEncryption.S3_MANAGED });
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketEncryption: {
        ServerSideEncryptionConfiguration: [
          {
            ServerSideEncryptionByDefault: { SSEAlgorithm: 'AES256' },
            BucketKeyEnabled: true,
          },
        ],
      },
    });
  });
  test('bucketKeyEnabled can not be enabled with UNENCRYPTED', () => {
    const stack = new cdk.Stack();

    // WHEN
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketKeyEnabled: true,
        encryption: s3.BucketEncryption.UNENCRYPTED,
      });
    }).toThrow(/bucketKeyEnabled is specified, so 'encryption' must be set to KMS, DSSE or S3/);
  });

  test('bucketKeyEnabled can NOT be enabled with encryption undefined', () => {
    const stack = new cdk.Stack();

    expect(() => {
      new s3.Bucket(stack, 'MyBucket3', { bucketKeyEnabled: true });
    }).toThrow("bucketKeyEnabled is specified, so 'encryption' must be set to KMS, DSSE or S3 (value: UNENCRYPTED)");
  });

  testDeprecated('logs to self, UNENCRYPTED does not throw error', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.UNENCRYPTED, serverAccessLogsPrefix: 'test' });
    }).not.toThrow();
  });

  test('logs to self, S3_MANAGED encryption does not throw error', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.S3_MANAGED, serverAccessLogsPrefix: 'test' });
    }).not.toThrow();
  });

  test('logs to self, KMS_MANAGED encryption throws error', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.KMS_MANAGED, serverAccessLogsPrefix: 'test' });
    }).toThrow(/Default bucket encryption with KMS managed or DSSE managed key is not supported for Server Access Logging target buckets/);
  });

  test('logs to self, KMS encryption without key does not throw error', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.KMS, serverAccessLogsPrefix: 'test' });
    }).not.toThrow();
  });

  test('logs to self, KMS encryption with key does not throw error', () => {
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'TestKey');
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { encryptionKey: key, encryption: s3.BucketEncryption.KMS, serverAccessLogsPrefix: 'test' });
    }).not.toThrow();
  });

  test('logs to self, KMS key with no specific encryption specified does not throw error', () => {
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'TestKey');
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { encryptionKey: key, serverAccessLogsPrefix: 'test' });
    }).not.toThrow();
  });

  testDeprecated('logs to separate bucket, UNENCRYPTED does not throw error', () => {
    const stack = new cdk.Stack();
    const logBucket = new s3.Bucket(stack, 'testLogBucket', { encryption: s3.BucketEncryption.UNENCRYPTED });
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { serverAccessLogsBucket: logBucket });
    }).not.toThrow();
  });

  test('logs to separate bucket, S3_MANAGED encryption does not throw error', () => {
    const stack = new cdk.Stack();
    const logBucket = new s3.Bucket(stack, 'testLogBucket', { encryption: s3.BucketEncryption.S3_MANAGED });
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { serverAccessLogsBucket: logBucket });
    }).not.toThrow();
  });

  // When provided an external bucket (as an IBucket), we cannot detect KMS_MANAGED encryption. Since this
  // check is impossible, we skip this test.
  // eslint-disable-next-line jest/no-disabled-tests
  test.skip('logs to separate bucket, KMS_MANAGED encryption throws error', () => {
    const stack = new cdk.Stack();
    const logBucket = new s3.Bucket(stack, 'testLogBucket', { encryption: s3.BucketEncryption.KMS_MANAGED });
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { serverAccessLogsBucket: logBucket });
    }).toThrow(/Default bucket encryption with KMS managed key is not supported for Server Access Logging target buckets/);
  });

  test('logs to separate bucket, KMS encryption without key does not throw error', () => {
    const stack = new cdk.Stack();
    const logBucket = new s3.Bucket(stack, 'testLogBucket', { encryption: s3.BucketEncryption.KMS });
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { serverAccessLogsBucket: logBucket });
    }).not.toThrow();
  });

  test('logs to separate bucket, KMS encryption with key does not throw error', () => {
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'TestKey');
    const logBucket = new s3.Bucket(stack, 'testLogBucket', { encryptionKey: key, encryption: s3.BucketEncryption.KMS });
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { serverAccessLogsBucket: logBucket });
    }).not.toThrow();
  });

  test('logs to separate bucket, KMS key with no specific encryption specified does not throw error', () => {
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'TestKey');
    const logBucket = new s3.Bucket(stack, 'testLogBucket', { encryptionKey: key });
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', { serverAccessLogsBucket: logBucket });
    }).not.toThrow();
  });

  test('bucket with versioning turned on', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      versioned: true,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'VersioningConfiguration': {
              'Status': 'Enabled',
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test.each([
    [true, 'Enabled'],
    [false, 'Disabled'],
    [undefined, Match.absent()],
  ])('bucket with ABAC status %s', (abacStatus, expected) => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      abacStatus,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      AbacStatus: expected,
    });
  });

  test('bucket with bucketNamePrefix and bucketNamespace', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      bucketNamePrefix: 'my-app',
      bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketNamePrefix: 'my-app',
      BucketNamespace: 'account-regional',
    });
  });

  test('bucket with bucketNamespace GLOBAL and bucketNamePrefix throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketNamePrefix: 'my-app',
        bucketNamespace: s3.BucketNamespace.GLOBAL,
      });
    }).toThrow(/\'bucketNamePrefix\' requires \'bucketNamespace\' to be set to ACCOUNT_REGIONAL/);
  });

  test('bucket with bucketName and bucketNamePrefix throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketName: 'my-bucket',
        bucketNamePrefix: 'my-app',
        bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
      });
    }).toThrow(/\'bucketName\' and \'bucketNamePrefix\' cannot be used together/);
  });

  test('bucket with bucketName and bucketNamespace ACCOUNT_REGIONAL throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketName: 'my-bucket',
        bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
      });
    }).toThrow(/\'bucketName\' cannot be used with \'bucketNamespace\' \(except GLOBAL\)/);
  });

  test('bucket with bucketName and bucketNamespace GLOBAL is valid', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      bucketName: 'my-bucket',
      bucketNamespace: s3.BucketNamespace.GLOBAL,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'my-bucket',
      BucketNamespace: 'global',
    });
  });

  test('bucket with bucketNamespace ACCOUNT_REGIONAL but no bucketNamePrefix throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
      });
    }).toThrow(/\'bucketNamespace\' ACCOUNT_REGIONAL requires \'bucketNamePrefix\' to be specified/);
  });

  test('bucket with bucketNamespace GLOBAL without bucketNamePrefix is valid', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      bucketNamespace: s3.BucketNamespace.GLOBAL,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketNamePrefix: Match.absent(),
      BucketNamespace: 'global',
    });
  });

  test('bucket with bucketNamePrefix only (no namespace) throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketNamePrefix: 'my-app',
      });
    }).toThrow(/\'bucketNamePrefix\' requires \'bucketNamespace\' to be set to ACCOUNT_REGIONAL/);
  });

  test('bucket with invalid bucketNamePrefix throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketNamePrefix: 'My-App',
        bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
      });
    }).toThrow(/Invalid S3 bucket name prefix/);
  });

  test('bucket with bucketNamePrefix too long throws', () => {
    const stack = new cdk.Stack();
    expect(() => {
      new s3.Bucket(stack, 'MyBucket', {
        bucketNamePrefix: 'a'.repeat(38),
        bucketNamespace: s3.BucketNamespace.ACCOUNT_REGIONAL,
      });
    }).toThrow(/Bucket name prefix must be 37 characters or fewer/);
  });

  test('bucket with neither bucketNamePrefix nor bucketNamespace does not set properties', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket');

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketNamePrefix: Match.absent(),
      BucketNamespace: Match.absent(),
    });
  });

  test('bucket with object lock enabled but no retention', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'Bucket', {
      objectLockEnabled: true,
    });
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      ObjectLockEnabled: true,
      ObjectLockConfiguration: Match.absent(),
    });
  });

  test('object lock defaults to disabled', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'Bucket');
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      ObjectLockEnabled: Match.absent(),
    });
  });

  test('object lock defaults to enabled when default retention is specified', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'Bucket', {
      objectLockDefaultRetention: s3.ObjectLockRetention.governance(cdk.Duration.days(7 * 365)),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      ObjectLockEnabled: true,
      ObjectLockConfiguration: {
        ObjectLockEnabled: 'Enabled',
        Rule: {
          DefaultRetention: {
            Mode: 'GOVERNANCE',
            Days: 7 * 365,
          },
        },
      },
    });
  });

  test('bucket with object lock enabled with governance retention', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'Bucket', {
      objectLockEnabled: true,
      objectLockDefaultRetention: s3.ObjectLockRetention.governance(cdk.Duration.days(1)),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      ObjectLockEnabled: true,
      ObjectLockConfiguration: {
        ObjectLockEnabled: 'Enabled',
        Rule: {
          DefaultRetention: {
            Mode: 'GOVERNANCE',
            Days: 1,
          },
        },
      },
    });
  });

  test('bucket with object lock enabled with compliance retention', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'Bucket', {
      objectLockEnabled: true,
      objectLockDefaultRetention: s3.ObjectLockRetention.compliance(cdk.Duration.days(1)),
    });
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      ObjectLockEnabled: true,
      ObjectLockConfiguration: {
        ObjectLockEnabled: 'Enabled',
        Rule: {
          DefaultRetention: {
            Mode: 'COMPLIANCE',
            Days: 1,
          },
        },
      },
    });
  });

  test('bucket with object lock disabled throws error with retention set', () => {
    const stack = new cdk.Stack();
    expect(() => new s3.Bucket(stack, 'Bucket', {
      objectLockEnabled: false,
      objectLockDefaultRetention: s3.ObjectLockRetention.governance(cdk.Duration.days(1)),
    })).toThrow('Object Lock must be enabled to configure default retention settings');
  });

  test('bucket with object lock requires duration than one day', () => {
    const stack = new cdk.Stack();
    expect(() => new s3.Bucket(stack, 'Bucket', {
      objectLockEnabled: true,
      objectLockDefaultRetention: s3.ObjectLockRetention.governance(cdk.Duration.days(0)),
    })).toThrow('Object Lock retention duration must be at least 1 day');
  });

  // https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-managing.html#object-lock-managing-retention-limits
  test('bucket with object lock requires duration less than 100 years', () => {
    const stack = new cdk.Stack();
    expect(() => new s3.Bucket(stack, 'Bucket', {
      objectLockEnabled: true,
      objectLockDefaultRetention: s3.ObjectLockRetention.governance(cdk.Duration.days(365 * 101)),
    })).toThrow('Object Lock retention duration must be less than 100 years');
  });

  describe('bucket with custom block public access setting', () => {
    test('S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT Disabled', () => {
      const app = new cdk.App({
        context: {
          [cxapi.S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT]: false,
        },
      });
      const stack = new cdk.Stack(app);
      new s3.Bucket(stack, 'MyBucket', {
        blockPublicAccess: new s3.BlockPublicAccess({ restrictPublicBuckets: true }),
      });

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'Properties': {
              'PublicAccessBlockConfiguration': {
                'RestrictPublicBuckets': true,
              },
            },
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
        },
      });
    });

    test('S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT Enabled', () => {
      const app = new cdk.App({
        context: {
          [cxapi.S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT]: true,
        },
      });
      const stack = new cdk.Stack(app);
      new s3.Bucket(stack, 'MyBucket', {
        blockPublicAccess: new s3.BlockPublicAccess({ restrictPublicBuckets: false }),
      });

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'Properties': {
              'PublicAccessBlockConfiguration': {
                'BlockPublicAcls': true,
                'BlockPublicPolicy': true,
                'IgnorePublicAcls': true,
                'RestrictPublicBuckets': false,
              },
            },
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
        },
      });
    });
  });

  test('bucket with block public access set to BlockAcls', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ACLS,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'PublicAccessBlockConfiguration': {
              'BlockPublicAcls': true,
              'IgnorePublicAcls': true,
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with block public access set to BLOCK_ACLS_ONLY', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ACLS_ONLY,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'PublicAccessBlockConfiguration': {
              'BlockPublicAcls': true,
              'BlockPublicPolicy': false,
              'IgnorePublicAcls': true,
              'RestrictPublicBuckets': false,
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  describe('bucket with custom block public access setting', () => {
    test('S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT Disabled', () => {
      const app = new cdk.App({
        context: {
          [cxapi.S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT]: false,
        },
      });
      const stack = new cdk.Stack(app);
      new s3.Bucket(stack, 'MyBucket', {
        blockPublicAccess: new s3.BlockPublicAccess({ restrictPublicBuckets: true }),
      });

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'Properties': {
              'PublicAccessBlockConfiguration': {
                'RestrictPublicBuckets': true,
              },
            },
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
        },
      });
    });

    test('S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT Enabled', () => {
      const app = new cdk.App({
        context: {
          [cxapi.S3_PUBLIC_ACCESS_BLOCKED_BY_DEFAULT]: true,
        },
      });
      const stack = new cdk.Stack(app);
      new s3.Bucket(stack, 'MyBucket', {
        blockPublicAccess: new s3.BlockPublicAccess({ restrictPublicBuckets: false }),
      });

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'Properties': {
              'PublicAccessBlockConfiguration': {
                'BlockPublicAcls': true,
                'BlockPublicPolicy': true,
                'IgnorePublicAcls': true,
                'RestrictPublicBuckets': false,
              },
            },
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
        },
      });
    });
  });

  test('bucket with default block public access setting to throw error msg', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'Bucket', {
      publicReadAccess: true,
    })).toThrow('Cannot use \'publicReadAccess\' property on a bucket without allowing bucket-level public access through \'blockPublicAccess\' property.');
  });

  test('bucket with enabled block public access setting to throw error msg', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'Bucket', {
      publicReadAccess: true,
      blockPublicAccess: {
        blockPublicPolicy: true,
        blockPublicAcls: false,
        ignorePublicAcls: false,
        restrictPublicBuckets: false,
      },
    })).toThrow('Cannot grant public access when \'blockPublicPolicy\' is enabled');
  });

  test('bucket with custom canned access control', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      accessControl: s3.BucketAccessControl.LOG_DELIVERY_WRITE,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'AccessControl': 'LogDeliveryWrite',
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  describe('permissions', () => {
    testDeprecated('addPermission creates a bucket policy for an UNENCRYPTED bucket', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.UNENCRYPTED });

      bucket.addToResourcePolicy(new iam.PolicyStatement({
        resources: ['foo'],
        actions: ['bar:baz'],
        principals: [new iam.AnyPrincipal()],
      }));

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
          'MyBucketPolicyE7FBAC7B': {
            'Type': 'AWS::S3::BucketPolicy',
            'Properties': {
              'Bucket': {
                'Ref': 'MyBucketF68F3FF0',
              },
              'PolicyDocument': {
                'Statement': [
                  {
                    Action: 'bar:baz',
                    'Effect': 'Allow',
                    'Principal': { AWS: '*' },
                    'Resource': 'foo',
                  },
                ],
                'Version': '2012-10-17',
              },
            },
          },
        },
      });
    });

    test('L2 addToResourcePolicy and L1 ResourceWithPolicies.of share a single bucket policy', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');

      // Add a statement via the L2 API
      bucket.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [bucket.arnForObjects('*')],
        principals: [new iam.AnyPrincipal()],
      }));

      // Add a statement via the L1 default trait
      const cfnBucket = bucket.node.defaultChild as s3.CfnBucket;
      iam.ResourceWithPolicies.of(cfnBucket)?.addToResourcePolicy(new iam.PolicyStatement({
        actions: ['s3:PutObject'],
        resources: [bucket.arnForObjects('*')],
        principals: [new iam.AnyPrincipal()],
      }));

      // Should result in a single bucket policy with both statements
      const template = Template.fromStack(stack);
      template.resourceCountIs('AWS::S3::BucketPolicy', 1);
      template.hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Statement: Match.arrayWith([
            Match.objectLike({ Action: 's3:GetObject' }),
            Match.objectLike({ Action: 's3:PutObject' }),
          ]),
        },
      });
    });

    test('addPermission creates a bucket policy for an S3_MANAGED bucket', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.S3_MANAGED });

      bucket.addToResourcePolicy(new iam.PolicyStatement({
        resources: ['foo'],
        actions: ['bar:baz'],
        principals: [new iam.AnyPrincipal()],
      }));

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
            'Properties': {
              'BucketEncryption': {
                'ServerSideEncryptionConfiguration': [
                  {
                    'ServerSideEncryptionByDefault': {
                      'SSEAlgorithm': 'AES256',
                    },
                  },
                ],
              },
            },
          },
          'MyBucketPolicyE7FBAC7B': {
            'Type': 'AWS::S3::BucketPolicy',
            'Properties': {
              'Bucket': {
                'Ref': 'MyBucketF68F3FF0',
              },
              'PolicyDocument': {
                'Statement': [
                  {
                    Action: 'bar:baz',
                    'Effect': 'Allow',
                    'Principal': { AWS: '*' },
                    'Resource': 'foo',
                  },
                ],
                'Version': '2012-10-17',
              },
            },
          },
        },
      });
    });

    testDeprecated('forBucket returns a permission statement associated with an UNENCRYPTED bucket\'s ARN', () => {
      const stack = new cdk.Stack();

      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.UNENCRYPTED });

      const x = new iam.PolicyStatement({
        resources: [bucket.bucketArn],
        actions: ['s3:ListBucket'],
        principals: [new iam.AnyPrincipal()],
      });

      expect(stack.resolve(x.toStatementJson())).toEqual({
        Action: 's3:ListBucket',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
      });
    });

    test('forBucket returns a permission statement associated with an S3_MANAGED bucket\'s ARN', () => {
      const stack = new cdk.Stack();

      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.S3_MANAGED });

      const x = new iam.PolicyStatement({
        resources: [bucket.bucketArn],
        actions: ['s3:ListBucket'],
        principals: [new iam.AnyPrincipal()],
      });

      expect(stack.resolve(x.toStatementJson())).toEqual({
        Action: 's3:ListBucket',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
      });
    });

    testDeprecated('arnForObjects returns a permission statement associated with objects in an UNENCRYPTED bucket', () => {
      const stack = new cdk.Stack();

      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.UNENCRYPTED });

      const p = new iam.PolicyStatement({
        resources: [bucket.arnForObjects('hello/world')],
        actions: ['s3:GetObject'],
        principals: [new iam.AnyPrincipal()],
      });

      expect(stack.resolve(p.toStatementJson())).toEqual({
        Action: 's3:GetObject',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: {
          'Fn::Join': [
            '',
            [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/hello/world'],
          ],
        },
      });
    });

    test('arnForObjects returns a permission statement associated with objects in an S3_MANAGED bucket', () => {
      const stack = new cdk.Stack();

      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.S3_MANAGED });

      const p = new iam.PolicyStatement({
        resources: [bucket.arnForObjects('hello/world')],
        actions: ['s3:GetObject'],
        principals: [new iam.AnyPrincipal()],
      });

      expect(stack.resolve(p.toStatementJson())).toEqual({
        Action: 's3:GetObject',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: {
          'Fn::Join': [
            '',
            [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/hello/world'],
          ],
        },
      });
    });

    testDeprecated('arnForObjects accepts multiple arguments and FnConcats them for an UNENCRYPTED bucket', () => {
      const stack = new cdk.Stack();

      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.UNENCRYPTED });

      const user = new iam.User(stack, 'MyUser');
      const team = new iam.Group(stack, 'MyTeam');

      const resource = bucket.arnForObjects(`home/${team.groupName}/${user.userName}/*`);
      const p = new iam.PolicyStatement({
        resources: [resource],
        actions: ['s3:GetObject'],
        principals: [new iam.AnyPrincipal()],
      });

      expect(stack.resolve(p.toStatementJson())).toEqual({
        Action: 's3:GetObject',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: {
          'Fn::Join': [
            '',
            [
              { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
              '/home/',
              { Ref: 'MyTeam01DD6685' },
              '/',
              { Ref: 'MyUserDC45028B' },
              '/*',
            ],
          ],
        },
      });
    });

    test('arnForObjects accepts multiple arguments and FnConcats them an S3_MANAGED bucket', () => {
      const stack = new cdk.Stack();

      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.S3_MANAGED });

      const user = new iam.User(stack, 'MyUser');
      const team = new iam.Group(stack, 'MyTeam');

      const resource = bucket.arnForObjects(`home/${team.groupName}/${user.userName}/*`);
      const p = new iam.PolicyStatement({
        resources: [resource],
        actions: ['s3:GetObject'],
        principals: [new iam.AnyPrincipal()],
      });

      expect(stack.resolve(p.toStatementJson())).toEqual({
        Action: 's3:GetObject',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: {
          'Fn::Join': [
            '',
            [
              { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
              '/home/',
              { Ref: 'MyTeam01DD6685' },
              '/',
              { Ref: 'MyUserDC45028B' },
              '/*',
            ],
          ],
        },
      });
    });
  });

  testDeprecated('removal policy can be used to specify behavior upon delete for an UNENCRYPTED bucket', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.UNENCRYPTED,
    });

    Template.fromStack(stack).templateMatches({
      Resources: {
        MyBucketF68F3FF0: {
          Type: 'AWS::S3::Bucket',
          DeletionPolicy: 'Retain',
          UpdateReplacePolicy: 'Retain',
        },
      },
    });
  });

  test('removal policy can be used to specify behavior upon delete for an S3_MANAGED bucket', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
    });

    Template.fromStack(stack).templateMatches({
      Resources: {
        MyBucketF68F3FF0: {
          Type: 'AWS::S3::Bucket',
          DeletionPolicy: 'Retain',
          UpdateReplacePolicy: 'Retain',
        },
      },
    });
  });

  describe('import/export', () => {
    test('static import(ref) allows importing an external/existing bucket', () => {
      const stack = new cdk.Stack();

      const bucketArn = 'arn:aws:s3:::my-bucket';
      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', { bucketArn });

      // this is a no-op since the bucket is external
      bucket.addToResourcePolicy(new iam.PolicyStatement({
        resources: ['foo'],
        actions: ['bar:baz'],
        principals: [new iam.AnyPrincipal()],
      }));

      const p = new iam.PolicyStatement({
        resources: [bucket.bucketArn],
        actions: ['s3:ListBucket'],
        principals: [new iam.AnyPrincipal()],
      });

      // it is possible to obtain a permission statement for a ref
      expect(p.toStatementJson()).toEqual({
        Action: 's3:ListBucket',
        Effect: 'Allow',
        Principal: { AWS: '*' },
        Resource: 'arn:aws:s3:::my-bucket',
      });

      expect(bucket.bucketArn).toEqual(bucketArn);
      expect(stack.resolve(bucket.bucketName)).toEqual('my-bucket');

      Template.fromStack(stack).templateMatches({});
    });

    test('import does not create any resources', () => {
      const stack = new cdk.Stack();
      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', { bucketArn: 'arn:aws:s3:::my-bucket' });
      bucket.addToResourcePolicy(new iam.PolicyStatement({
        resources: ['*'],
        actions: ['*'],
        principals: [new iam.AnyPrincipal()],
      }));

      // at this point we technically didn't create any resources in the consuming stack.
      Template.fromStack(stack).templateMatches({});
    });

    test('import can also be used to import arbitrary ARNs', () => {
      const stack = new cdk.Stack();
      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', { bucketArn: 'arn:aws:s3:::my-bucket' });
      bucket.addToResourcePolicy(new iam.PolicyStatement({ resources: ['*'], actions: ['*'] }));

      // but now we can reference the bucket
      // you can even use the bucket name, which will be extracted from the arn provided.
      const user = new iam.User(stack, 'MyUser');
      user.addToPolicy(new iam.PolicyStatement({
        resources: [bucket.arnForObjects(`my/folder/${bucket.bucketName}`)],
        actions: ['s3:*'],
      }));

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyUserDC45028B': {
            'Type': 'AWS::IAM::User',
          },
          'MyUserDefaultPolicy7B897426': {
            'Type': 'AWS::IAM::Policy',
            'Properties': {
              'PolicyDocument': {
                'Statement': [
                  {
                    Action: 's3:*',
                    'Effect': 'Allow',
                    'Resource': 'arn:aws:s3:::my-bucket/my/folder/my-bucket',
                  },
                ],
                'Version': '2012-10-17',
              },
              'PolicyName': 'MyUserDefaultPolicy7B897426',
              'Users': [
                {
                  'Ref': 'MyUserDC45028B',
                },
              ],
            },
          },
        },
      });
    });

    test('import can explicitly set bucket region with different suffix than stack', () => {
      const stack = new cdk.Stack(undefined, undefined, {
        env: { region: 'cn-north-1' },
      });

      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', {
        bucketName: 'mybucket',
        region: 'eu-west-1',
      });

      expect(bucket.bucketRegionalDomainName).toEqual('mybucket.s3.eu-west-1.amazonaws.com');
      expect(bucket.bucketWebsiteDomainName).toEqual('mybucket.s3-website-eu-west-1.amazonaws.com');
    });

    test('new bucketWebsiteUrl format for specific region', () => {
      const stack = new cdk.Stack(undefined, undefined, {
        env: { region: 'us-east-2' },
      });

      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', {
        bucketName: 'mybucket',
      });

      expect(bucket.bucketWebsiteUrl).toEqual('http://mybucket.s3-website.us-east-2.amazonaws.com');
    });

    test('new bucketWebsiteUrl format for specific region with cn suffix', () => {
      const stack = new cdk.Stack(undefined, undefined, {
        env: { region: 'cn-north-1' },
      });

      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', {
        bucketName: 'mybucket',
      });

      expect(bucket.bucketWebsiteUrl).toEqual('http://mybucket.s3-website.cn-north-1.amazonaws.com.cn');
    });

    testDeprecated('new bucketWebsiteUrl format with explicit bucketWebsiteNewUrlFormat', () => {
      const stack = new cdk.Stack(undefined, undefined, {
        env: { region: 'us-east-1' },
      });

      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', {
        bucketName: 'mybucket',
        bucketWebsiteNewUrlFormat: true,
      });

      expect(bucket.bucketWebsiteUrl).toEqual('http://mybucket.s3-website.us-east-1.amazonaws.com');
    });

    testDeprecated('old bucketWebsiteUrl format with explicit bucketWebsiteNewUrlFormat', () => {
      const stack = new cdk.Stack(undefined, undefined, {
        env: { region: 'us-east-2' },
      });

      const bucket = s3.Bucket.fromBucketAttributes(stack, 'ImportedBucket', {
        bucketName: 'mybucket',
        bucketWebsiteNewUrlFormat: false,
      });

      expect(bucket.bucketWebsiteUrl).toEqual('http://mybucket.s3-website-us-east-2.amazonaws.com');
    });

    test('import needs to specify a valid bucket name', () => {
      const stack = new cdk.Stack(undefined, undefined, {
        env: { region: 'us-east-1' },
      });

      expect(() => s3.Bucket.fromBucketAttributes(stack, 'MyBucket3', {
        bucketName: 'arn:aws:s3:::example-com',
      })).toThrow();
    });
  });

  describe('fromCfnBucket()', () => {
    let stack: cdk.Stack;
    let cfnBucket: s3.CfnBucket;
    let bucket: s3.IBucket;

    beforeEach(() => {
      stack = new cdk.Stack();
      cfnBucket = new s3.CfnBucket(stack, 'CfnBucket');
      bucket = s3.Bucket.fromCfnBucket(cfnBucket);
    });

    test("correctly resolves the 'bucketName' property", () => {
      expect(stack.resolve(bucket.bucketName)).toStrictEqual({
        Ref: 'CfnBucket',
      });
    });

    test("correctly resolves the 'bucketArn' property", () => {
      expect(stack.resolve(bucket.bucketArn)).toStrictEqual({
        'Fn::GetAtt': ['CfnBucket', 'Arn'],
      });
    });

    test('allows setting the RemovalPolicy of the underlying resource', () => {
      bucket.applyRemovalPolicy(cdk.RemovalPolicy.RETAIN);

      Template.fromStack(stack).hasResource('AWS::S3::Bucket', {
        UpdateReplacePolicy: 'Retain',
        DeletionPolicy: 'Retain',
      });
    });

    test('allows overriding cross-stack reference strength', () => {
      const app = new cdk.App();
      const producerStack = new cdk.Stack(app, 'Producer', { env: { region: 'us-east-1', account: '111111111111' } });
      const consumerStack = new cdk.Stack(app, 'Consumer', { env: { region: 'us-east-1', account: '111111111111' } });

      const b = new s3.Bucket(producerStack, 'MyBucket');
      cdk.CrossStackReferences.of(b).produce(cdk.ReferenceStrength.WEAK);

      new cdk.CfnOutput(consumerStack, 'BucketName', { value: b.bucketName });

      const assembly = app.synth();
      const consumerTemplate = assembly.getStackByName(consumerStack.stackName).template;

      const output = consumerTemplate.Outputs.BucketName;
      expect(output.Value).toHaveProperty('Fn::GetStackOutput');
    });

    test('correctly sets the default child of the returned L2', () => {
      expect(bucket.node.defaultChild).toBe(cfnBucket);
    });

    test('allows granting permissions to Principals', () => {
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AccountRootPrincipal(),
      });
      bucket.grantRead(role);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: [
                's3:GetObject*',
                's3:GetBucket*',
                's3:List*',
              ],
              'Resource': [{
                'Fn::GetAtt': ['CfnBucket', 'Arn'],
              }, {
                'Fn::Join': ['', [
                  { 'Fn::GetAtt': ['CfnBucket', 'Arn'] },
                  '/*',
                ]],
              }],
            },
          ],
        },
      });
    });

    test("sets the isWebsite property to 'false' if 'websiteConfiguration' is 'undefined'", () => {
      expect(bucket.isWebsite).toBeFalsy();
    });

    test("sets the isWebsite property to 'true' if 'websiteConfiguration' is not 'undefined'", () => {
      cfnBucket = new s3.CfnBucket(stack, 'WebsiteCfnBucket', {
        websiteConfiguration: {
          indexDocument: 'index.html',
        },
      });
      bucket = s3.Bucket.fromCfnBucket(cfnBucket);

      expect(bucket.isWebsite).toBeTruthy();
    });

    test('allows granting public access by default', () => {
      expect(() => {
        bucket.grantPublicAccess();
      }).not.toThrow();
    });

    test('does not allow granting public access for a Bucket that blocks it', () => {
      cfnBucket = new s3.CfnBucket(stack, 'BlockedPublicAccessCfnBucket', {
        publicAccessBlockConfiguration: {
          blockPublicPolicy: true,
        },
      });
      bucket = s3.Bucket.fromCfnBucket(cfnBucket);

      expect(() => {
        bucket.grantPublicAccess();
      }).toThrow(/Cannot grant public access when 'blockPublicPolicy' is enabled/);
    });

    test('correctly fills the encryption key if the L1 references one', () => {
      const cfnKey = new kms.CfnKey(stack, 'CfnKey', {
        keyPolicy: {
          'Statement': [
            {
              Action: [
                'kms:*',
              ],
              'Effect': 'Allow',
              'Principal': {
                'AWS': {
                  'Fn::Join': ['', [
                    'arn:',
                    { 'Ref': 'AWS::Partition' },
                    ':iam::',
                    { 'Ref': 'AWS::AccountId' },
                    ':root',
                  ]],
                },
              },
              'Resource': '*',
            },
          ],
          'Version': '2012-10-17',
        },
      });
      cfnBucket = new s3.CfnBucket(stack, 'KmsEncryptedCfnBucket', {
        bucketEncryption: {
          serverSideEncryptionConfiguration: [
            {
              serverSideEncryptionByDefault: {
                kmsMasterKeyId: cfnKey.attrArn,
                sseAlgorithm: 'aws:kms',
              },
            },
          ],
        },
      });
      bucket = s3.Bucket.fromCfnBucket(cfnBucket);

      expect(bucket.encryptionKey).not.toBeUndefined();
    });

    test('allows importing a BucketPolicy that references a Bucket', () => {
      cdk.Validations.of(stack).acknowledge({
        id: 'CloudFormation-Validate::E3019',
        reason: 'This test is asserting something pointless',
      });

      new s3.CfnBucketPolicy(stack, 'CfnBucketPolicy', {
        policyDocument: {
          'Statement': [
            {
              Action: 's3:*',
              'Effect': 'Allow',
              'Principal': {
                'AWS': '*',
              },
              'Resource': ['*'],
            },
          ],
          'Version': '2012-10-17',
        },
        bucket: cfnBucket.ref,
      });
      bucket.addToResourcePolicy(new iam.PolicyStatement({
        resources: ['*'],
        actions: ['s3:*'],
        principals: [new iam.AccountRootPrincipal()],
      }));

      Template.fromStack(stack).resourceCountIs('AWS::S3::BucketPolicy', 2);
    });
  });

  test('can grant read permissions to a CfnBucket', () => {
    const stack = new cdk.Stack();
    const principal = new iam.ServicePrincipal('s3.amazonaws.com');

    const key = new CfnKey(stack, 'EncryptionKey');
    const cfnBucket = new CfnBucket(stack, 'CfnBucket', {
      bucketEncryption: {
        serverSideEncryptionConfiguration: [{
          bucketKeyEnabled: true,
          serverSideEncryptionByDefault: {
            kmsMasterKeyId: key.attrKeyId,
            sseAlgorithm: 'aws:kms',
          },
        }],
      },
    });

    BucketGrants.fromBucket(cfnBucket).read(principal);

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: { Ref: 'CfnBucket' },
      PolicyDocument: {
        Statement: [{
          Action: ['s3:GetObject*', 's3:GetBucket*', 's3:List*'],
          Effect: 'Allow',
          Principal: { 'Service': 's3.amazonaws.com' },
          Resource: [{
            'Fn::GetAtt': ['CfnBucket', 'Arn'],
          }, {
            'Fn::Join': ['', [{ 'Fn::GetAtt': ['CfnBucket', 'Arn'] }, '/*']],
          }],
        }],
      },
    });

    template.hasResourceProperties('AWS::KMS::Key', {
      KeyPolicy: {
        Statement: [
          {
            Action: [
              'kms:Decrypt',
              'kms:DescribeKey',
            ],
            Effect: 'Allow',
            Principal: {
              Service: 's3.amazonaws.com',
            },
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('BucketGrants.actionsOnObjectKeys grants only on objects ARN', () => {
    const stack = new cdk.Stack();
    const user = new iam.User(stack, 'User');
    const bucket = new s3.Bucket(stack, 'MyBucket');

    BucketGrants.fromBucket(bucket).actionsOnObjectKeys(user, '*', 's3:GetObject', 's3:PutObject');

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: ['s3:GetObject', 's3:PutObject'],
            Effect: 'Allow',
            Resource: { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/*']] },
          },
        ],
      },
    });
  });

  test('BucketGrants.actionsOnObjectKeys with custom key pattern restricts objects ARN', () => {
    const stack = new cdk.Stack();
    const user = new iam.User(stack, 'User');
    const bucket = new s3.Bucket(stack, 'MyBucket');

    BucketGrants.fromBucket(bucket).actionsOnObjectKeys(user, 'my/prefix/*', 's3:GetObject');

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 's3:GetObject',
            Effect: 'Allow',
            Resource: { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/my/prefix/*']] },
          },
        ],
      },
    });
  });

  test('BucketGrants.actionsOnObjectKeys works with CfnBucket', () => {
    const stack = new cdk.Stack();
    const principal = new iam.ServicePrincipal('lambda.amazonaws.com');
    const cfnBucket = new CfnBucket(stack, 'CfnBucket');

    BucketGrants.fromBucket(cfnBucket).actionsOnObjectKeys(principal, '*', 's3:GetObject');

    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: { Ref: 'CfnBucket' },
      PolicyDocument: {
        Statement: [{
          Action: 's3:GetObject',
          Effect: 'Allow',
          Principal: { Service: 'lambda.amazonaws.com' },
          Resource: { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['CfnBucket', 'Arn'] }, '/*']] },
        }],
      },
    });
  });

  test('BucketGrants.actionsOnBucketAndObjectKeys grants on both bucket and objects ARN', () => {
    const stack = new cdk.Stack();
    const user = new iam.User(stack, 'User');
    const bucket = new s3.Bucket(stack, 'MyBucket');

    BucketGrants.fromBucket(bucket).actionsOnBucketAndObjectKeys(user, '*', 's3:GetObject', 's3:PutObject');

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: ['s3:GetObject', 's3:PutObject'],
            Effect: 'Allow',
            Resource: [
              { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
              { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/*']] },
            ],
          },
        ],
      },
    });
  });

  test('BucketGrants.actionsOnBucketAndObjectKeys with custom key pattern', () => {
    const stack = new cdk.Stack();
    const user = new iam.User(stack, 'User');
    const bucket = new s3.Bucket(stack, 'MyBucket');

    BucketGrants.fromBucket(bucket).actionsOnBucketAndObjectKeys(user, 'my/prefix/*', 's3:GetObject');

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 's3:GetObject',
            Effect: 'Allow',
            Resource: [
              { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
              { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/my/prefix/*']] },
            ],
          },
        ],
      },
    });
  });

  test('BucketGrants.actionsOnBucketAndObjectKeys works with CfnBucket', () => {
    const stack = new cdk.Stack();
    const principal = new iam.ServicePrincipal('lambda.amazonaws.com');
    const cfnBucket = new CfnBucket(stack, 'CfnBucket');

    BucketGrants.fromBucket(cfnBucket).actionsOnBucketAndObjectKeys(principal, '*', 's3:GetObject');

    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: { Ref: 'CfnBucket' },
      PolicyDocument: {
        Statement: [{
          Action: 's3:GetObject',
          Effect: 'Allow',
          Principal: { Service: 'lambda.amazonaws.com' },
          Resource: [
            { 'Fn::GetAtt': ['CfnBucket', 'Arn'] },
            { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['CfnBucket', 'Arn'] }, '/*']] },
          ],
        }],
      },
    });
  });

  test('grantRead', () => {
    const stack = new cdk.Stack();
    const reader = new iam.User(stack, 'Reader');
    const bucket = new s3.Bucket(stack, 'MyBucket');
    bucket.grantRead(reader);
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'ReaderF7BF189D': {
          'Type': 'AWS::IAM::User',
        },
        'ReaderDefaultPolicy151F3818': {
          'Type': 'AWS::IAM::Policy',
          'Properties': {
            'PolicyDocument': {
              'Statement': [
                {
                  Action: [
                    's3:GetObject*',
                    's3:GetBucket*',
                    's3:List*',
                  ],
                  'Effect': 'Allow',
                  'Resource': [
                    {
                      'Fn::GetAtt': [
                        'MyBucketF68F3FF0',
                        'Arn',
                      ],
                    },
                    {
                      'Fn::Join': [
                        '',
                        [
                          {
                            'Fn::GetAtt': [
                              'MyBucketF68F3FF0',
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
              'Version': '2012-10-17',
            },
            'PolicyName': 'ReaderDefaultPolicy151F3818',
            'Users': [
              {
                'Ref': 'ReaderF7BF189D',
              },
            ],
          },
        },
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  describe('grantReadWrite', () => {
    test('can be used to grant reciprocal permissions to an identity', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      const user = new iam.User(stack, 'MyUser');
      bucket.grantReadWrite(user);

      Template.fromStack(stack).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
          'MyUserDC45028B': {
            'Type': 'AWS::IAM::User',
          },
          'MyUserDefaultPolicy7B897426': {
            'Type': 'AWS::IAM::Policy',
            'Properties': {
              'PolicyDocument': {
                'Statement': [
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
                    'Effect': 'Allow',
                    'Resource': [
                      {
                        'Fn::GetAtt': [
                          'MyBucketF68F3FF0',
                          'Arn',
                        ],
                      },
                      {
                        'Fn::Join': [
                          '',
                          [
                            {
                              'Fn::GetAtt': [
                                'MyBucketF68F3FF0',
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
                'Version': '2012-10-17',
              },
              'PolicyName': 'MyUserDefaultPolicy7B897426',
              'Users': [
                {
                  'Ref': 'MyUserDC45028B',
                },
              ],
            },
          },
        },
      });
    });

    test('grant permissions to non-identity principal', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.KMS });

      // WHEN
      bucket.grantRead(new iam.OrganizationPrincipal('o-12345abcde'));

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          'Version': '2012-10-17',
          'Statement': [
            {
              Action: ['s3:GetObject*', 's3:GetBucket*', 's3:List*'],
              'Condition': { 'StringEquals': { 'aws:PrincipalOrgID': 'o-12345abcde' } },
              'Effect': 'Allow',
              'Principal': { AWS: '*' },
              'Resource': [
                { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] }, '/*']] },
              ],
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
        'EnableKeyRotation': true,
        'KeyPolicy': {
          'Statement': Match.arrayWith([
            {
              Action: ['kms:Decrypt', 'kms:DescribeKey'],
              'Effect': 'Allow',
              'Resource': '*',
              'Principal': { AWS: '*' },
              'Condition': { 'StringEquals': { 'aws:PrincipalOrgID': 'o-12345abcde' } },
            },
          ]),
          'Version': '2012-10-17',
        },

      });
    });

    test('does not grant PutObjectAcl when the S3_GRANT_WRITE_WITHOUT_ACL feature is enabled', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      const user = new iam.User(stack, 'MyUser');

      bucket.grantReadWrite(user);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', Match.objectLike({
        'PolicyDocument': {
          'Statement': [
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
              'Effect': 'Allow',
              'Resource': [
                { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                {
                  'Fn::Join': ['', [
                    { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                    '/*',
                  ]],
                },
              ],
            },
          ],
        },
      }));
    });
  });

  describe('grantWrite', () => {
    test('with KMS key has appropriate permissions for multipart uploads', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.KMS });
      const user = new iam.User(stack, 'MyUser');
      bucket.grantWrite(user);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: [
                's3:DeleteObject*',
                's3:PutObject',
                's3:PutObjectLegalHold',
                's3:PutObjectRetention',
                's3:PutObjectTagging',
                's3:PutObjectVersionTagging',
                's3:Abort*',
              ],
              'Effect': 'Allow',
              'Resource': [
                {
                  'Fn::GetAtt': [
                    'MyBucketF68F3FF0',
                    'Arn',
                  ],
                },
                {
                  'Fn::Join': [
                    '',
                    [
                      {
                        'Fn::GetAtt': [
                          'MyBucketF68F3FF0',
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
                'kms:Encrypt',
                'kms:ReEncrypt*',
                'kms:GenerateDataKey*',
                'kms:Decrypt',
              ],
              'Effect': 'Allow',
              'Resource': {
                'Fn::GetAtt': [
                  'MyBucketKeyC17130CF',
                  'Arn',
                ],
              },
            },
          ],
          'Version': '2012-10-17',
        },
        'PolicyName': 'MyUserDefaultPolicy7B897426',
        'Users': [
          {
            'Ref': 'MyUserDC45028B',
          },
        ],
      });
    });

    test('does not grant PutObjectAcl when the S3_GRANT_WRITE_WITHOUT_ACL feature is enabled', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      const user = new iam.User(stack, 'MyUser');

      bucket.grantWrite(user);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: [
                's3:DeleteObject*',
                's3:PutObject',
                's3:PutObjectLegalHold',
                's3:PutObjectRetention',
                's3:PutObjectTagging',
                's3:PutObjectVersionTagging',
                's3:Abort*',
              ],
              'Effect': 'Allow',
              'Resource': [
                { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                {
                  'Fn::Join': ['', [
                    { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                    '/*',
                  ]],
                },
              ],
            },
          ],
        },
      });
    });

    test('grant only allowedActionPatterns when specified', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      const user = new iam.User(stack, 'MyUser');

      bucket.grantWrite(user, '*', ['s3:PutObject', 's3:DeleteObject*']);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: [
                's3:PutObject',
                's3:DeleteObject*',
              ],
              'Effect': 'Allow',
              'Resource': [
                { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                {
                  'Fn::Join': ['', [
                    { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                    '/*',
                  ]],
                },
              ],
            },
          ],
        },
      });
    });
  });

  describe('grantPut', () => {
    test('does not grant PutObjectAcl when the S3_GRANT_WRITE_WITHOUT_ACL feature is enabled', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket');
      const user = new iam.User(stack, 'MyUser');

      bucket.grantPut(user);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: [
                's3:PutObject',
                's3:PutObjectLegalHold',
                's3:PutObjectRetention',
                's3:PutObjectTagging',
                's3:PutObjectVersionTagging',
                's3:Abort*',
              ],
              'Effect': 'Allow',
              'Resource': {
                'Fn::Join': ['', [
                  { 'Fn::GetAtt': ['MyBucketF68F3FF0', 'Arn'] },
                  '/*',
                ]],
              },
            },
          ],
        },
      });
    });
  });

  test('more grants', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket', { encryption: s3.BucketEncryption.KMS });
    const putter = new iam.User(stack, 'Putter');
    const writer = new iam.User(stack, 'Writer');
    const deleter = new iam.User(stack, 'Deleter');

    bucket.grantPut(putter);
    bucket.grantWrite(writer);
    bucket.grantDelete(deleter);

    const resources = Template.fromStack(stack).toJSON().Resources;
    const actions = (id: string) => resources[id].Properties.PolicyDocument.Statement[0].Action;

    expect(actions('WriterDefaultPolicyDC585BCE')).toEqual([
      's3:DeleteObject*',
      's3:PutObject',
      's3:PutObjectLegalHold',
      's3:PutObjectRetention',
      's3:PutObjectTagging',
      's3:PutObjectVersionTagging',
      's3:Abort*',
    ]);
    expect(actions('PutterDefaultPolicyAB138DD3')).toEqual([
      's3:PutObject',
      's3:PutObjectLegalHold',
      's3:PutObjectRetention',
      's3:PutObjectTagging',
      's3:PutObjectVersionTagging',
      's3:Abort*',
    ]);
    expect(actions('DeleterDefaultPolicyCD33B8A0')).toEqual('s3:DeleteObject*');
  });

  test('grantDelete, with a KMS Key', () => {
    // given
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'MyKey');
    const deleter = new iam.User(stack, 'Deleter');
    const bucket = new s3.Bucket(stack, 'MyBucket', {
      bucketName: 'my-bucket-physical-name',
      encryptionKey: key,
      encryption: s3.BucketEncryption.KMS,
    });

    // when
    bucket.grantDelete(deleter);

    // then
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            Action: 's3:DeleteObject*',
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': [
                '',
                [
                  {
                    'Fn::GetAtt': [
                      'MyBucketF68F3FF0',
                      'Arn',
                    ],
                  },
                  '/*',
                ],
              ],
            },
          },
        ],
        'Version': '2012-10-17',
      },
    });
  });

  describe('cross-stack permissions', () => {
    test('in the same account and region', () => {
      const app = new cdk.App();
      const stackA = new cdk.Stack(app, 'stackA');
      const bucketFromStackA = new s3.Bucket(stackA, 'MyBucket');

      const stackB = new cdk.Stack(app, 'stackB');
      const user = new iam.User(stackB, 'UserWhoNeedsAccess');
      bucketFromStackA.grantRead(user);

      Template.fromStack(stackA).templateMatches({
        'Resources': {
          'MyBucketF68F3FF0': {
            'Type': 'AWS::S3::Bucket',
            'DeletionPolicy': 'Retain',
            'UpdateReplacePolicy': 'Retain',
          },
        },
        'Outputs': {
          'ExportsOutputFnGetAttMyBucketF68F3FF0Arn0F7E8E58': {
            'Value': {
              'Fn::GetAtt': [
                'MyBucketF68F3FF0',
                'Arn',
              ],
            },
            'Export': {
              'Name': 'stackA:ExportsOutputFnGetAttMyBucketF68F3FF0Arn0F7E8E58',
            },
          },
        },
      });

      Template.fromStack(stackB).templateMatches({
        'Resources': {
          'UserWhoNeedsAccessF8959C3D': {
            'Type': 'AWS::IAM::User',
          },
          'UserWhoNeedsAccessDefaultPolicy6A9EB530': {
            'Type': 'AWS::IAM::Policy',
            'Properties': {
              'PolicyDocument': {
                'Statement': [
                  {
                    Action: [
                      's3:GetObject*',
                      's3:GetBucket*',
                      's3:List*',
                    ],
                    'Effect': 'Allow',
                    'Resource': [
                      {
                        'Fn::ImportValue': 'stackA:ExportsOutputFnGetAttMyBucketF68F3FF0Arn0F7E8E58',
                      },
                      {
                        'Fn::Join': [
                          '',
                          [
                            {
                              'Fn::ImportValue': 'stackA:ExportsOutputFnGetAttMyBucketF68F3FF0Arn0F7E8E58',
                            },
                            '/*',
                          ],
                        ],
                      },
                    ],
                  },
                ],
                'Version': '2012-10-17',
              },
              'PolicyName': 'UserWhoNeedsAccessDefaultPolicy6A9EB530',
              'Users': [
                {
                  'Ref': 'UserWhoNeedsAccessF8959C3D',
                },
              ],
            },
          },
        },
      });
    });

    test('in different accounts', () => {
      // GIVEN
      const stackA = new cdk.Stack(undefined, 'StackA', { env: { account: '123456789012' } });
      const bucketFromStackA = new s3.Bucket(stackA, 'MyBucket', {
        bucketName: 'my-bucket-physical-name',
      });

      const stackB = new cdk.Stack(undefined, 'StackB', { env: { account: '234567890123' } });
      const roleFromStackB = new iam.Role(stackB, 'MyRole', {
        assumedBy: new iam.AccountPrincipal('234567890123'),
        roleName: 'MyRolePhysicalName',
      });

      // WHEN
      bucketFromStackA.grantRead(roleFromStackB);

      // THEN
      Template.fromStack(stackA).hasResourceProperties('AWS::S3::BucketPolicy', {
        'PolicyDocument': {
          'Statement': [
            Match.objectLike({
              Action: [
                's3:GetObject*',
                's3:GetBucket*',
                's3:List*',
              ],
              'Effect': 'Allow',
              'Principal': {
                'AWS': {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::234567890123:role/MyRolePhysicalName',
                    ],
                  ],
                },
              },
            }),
          ],
        },
      });

      Template.fromStack(stackB).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: [
                's3:GetObject*',
                's3:GetBucket*',
                's3:List*',
              ],
              'Effect': 'Allow',
              'Resource': [
                {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':s3:::my-bucket-physical-name',
                    ],
                  ],
                },
                {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':s3:::my-bucket-physical-name/*',
                    ],
                  ],
                },
              ],
            },
          ],
        },
      });
    });

    test('in different accounts, with a KMS Key', () => {
      // GIVEN
      const stackA = new cdk.Stack(undefined, 'StackA', { env: { account: '123456789012' } });
      const key = new kms.Key(stackA, 'MyKey');
      const bucketFromStackA = new s3.Bucket(stackA, 'MyBucket', {
        bucketName: 'my-bucket-physical-name',
        encryptionKey: key,
        encryption: s3.BucketEncryption.KMS,
      });

      const stackB = new cdk.Stack(undefined, 'StackB', { env: { account: '234567890123' } });
      const roleFromStackB = new iam.Role(stackB, 'MyRole', {
        assumedBy: new iam.AccountPrincipal('234567890123'),
        roleName: 'MyRolePhysicalName',
      });

      // WHEN
      bucketFromStackA.grantRead(roleFromStackB);

      // THEN
      Template.fromStack(stackA).hasResourceProperties('AWS::KMS::Key', {
        'KeyPolicy': {
          'Statement': Match.arrayWith([
            Match.objectLike({
              Action: [
                'kms:Decrypt',
                'kms:DescribeKey',
              ],
              'Effect': 'Allow',
              'Principal': {
                'AWS': {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::234567890123:role/MyRolePhysicalName',
                    ],
                  ],
                },
              },
            }),
          ]),
        },
      });

      Template.fromStack(stackB).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([
            Match.objectLike({
              Action: [
                'kms:Decrypt',
                'kms:DescribeKey',
              ],
              'Effect': 'Allow',
              'Resource': '*',
            }),
          ]),
        },
      });
    });
  });

  test('urlForObject returns a token with the S3 URL of the token', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const bucketWithRegion = s3.Bucket.fromBucketAttributes(stack, 'RegionalBucket', {
      bucketArn: 'arn:aws:s3:::explicit-region-bucket',
      region: 'us-west-2',
    });

    new cdk.CfnOutput(stack, 'BucketURL', { value: bucket.urlForObject() });
    new cdk.CfnOutput(stack, 'MyFileURL', { value: bucket.urlForObject('my/file.txt') });
    new cdk.CfnOutput(stack, 'YourFileURL', { value: bucket.urlForObject('/your/file.txt') }); // "/" is optional
    new cdk.CfnOutput(stack, 'RegionBucketURL', { value: bucketWithRegion.urlForObject() });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
      'Outputs': {
        'BucketURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://s3.',
                {
                  'Ref': 'AWS::Region',
                },
                '.',
                {
                  'Ref': 'AWS::URLSuffix',
                },
                '/',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
              ],
            ],
          },
        },
        'MyFileURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://s3.',
                {
                  'Ref': 'AWS::Region',
                },
                '.',
                {
                  'Ref': 'AWS::URLSuffix',
                },
                '/',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '/my/file.txt',
              ],
            ],
          },
        },
        'YourFileURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://s3.',
                {
                  'Ref': 'AWS::Region',
                },
                '.',
                {
                  'Ref': 'AWS::URLSuffix',
                },
                '/',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '/your/file.txt',
              ],
            ],
          },
        },
        'RegionBucketURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://s3.us-west-2.',
                {
                  'Ref': 'AWS::URLSuffix',
                },
                '/explicit-region-bucket',
              ],
            ],
          },
        },
      },
    });
  });

  test('s3UrlForObject returns a token with the S3 URL of the token', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');

    new cdk.CfnOutput(stack, 'BucketS3URL', { value: bucket.s3UrlForObject() });
    new cdk.CfnOutput(stack, 'MyFileS3URL', { value: bucket.s3UrlForObject('my/file.txt') });
    new cdk.CfnOutput(stack, 'YourFileS3URL', { value: bucket.s3UrlForObject('/your/file.txt') }); // "/" is optional

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
      'Outputs': {
        'BucketS3URL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                's3://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
              ],
            ],
          },
        },
        'MyFileS3URL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                's3://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '/my/file.txt',
              ],
            ],
          },
        },
        'YourFileS3URL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                's3://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '/your/file.txt',
              ],
            ],
          },
        },
      },
    });
  });

  describe('grantPublicAccess', () => {
    test('by default, grants s3:GetObject to all objects', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'b');

      // WHEN
      bucket.grantPublicAccess();

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: 's3:GetObject',
              'Effect': 'Allow',
              'Principal': { AWS: '*' },
              'Resource': { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['bC3BBCC65', 'Arn'] }, '/*']] },
            },
          ],
          'Version': '2012-10-17',
        },
      });
    });

    test('"keyPrefix" can be used to only grant access to certain objects', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'b');

      // WHEN
      bucket.grantPublicAccess('only/access/these/*');

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: 's3:GetObject',
              'Effect': 'Allow',
              'Principal': { AWS: '*' },
              'Resource': { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['bC3BBCC65', 'Arn'] }, '/only/access/these/*']] },
            },
          ],
          'Version': '2012-10-17',
        },
      });
    });

    test('"allowedActions" can be used to specify actions explicitly', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'b');

      // WHEN
      bucket.grantPublicAccess('*', 's3:GetObject', 's3:PutObject');

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: ['s3:GetObject', 's3:PutObject'],
              'Effect': 'Allow',
              'Principal': { AWS: '*' },
              'Resource': { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['bC3BBCC65', 'Arn'] }, '/*']] },
            },
          ],
          'Version': '2012-10-17',
        },
      });
    });

    test('returns the PolicyStatement which can be then customized', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'b');

      // WHEN
      const result = bucket.grantPublicAccess();
      result.resourceStatement!.addCondition('IpAddress', { 'aws:SourceIp': '54.240.143.0/24' });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        'PolicyDocument': {
          'Statement': [
            {
              Action: 's3:GetObject',
              'Effect': 'Allow',
              'Principal': { AWS: '*' },
              'Resource': { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['bC3BBCC65', 'Arn'] }, '/*']] },
              'Condition': {
                'IpAddress': { 'aws:SourceIp': '54.240.143.0/24' },
              },
            },
          ],
          'Version': '2012-10-17',
        },
      });
    });

    test('throws when blockPublicPolicy is set to true', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'MyBucket', {
        blockPublicAccess: new s3.BlockPublicAccess({ blockPublicPolicy: true }),
      });

      // THEN
      expect(() => bucket.grantPublicAccess()).toThrow(/blockPublicPolicy/);
    });
  });

  describe('website configuration', () => {
    test('only index doc', () => {
      const stack = new cdk.Stack();
      new s3.Bucket(stack, 'Website', {
        websiteIndexDocument: 'index2.html',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        WebsiteConfiguration: {
          IndexDocument: 'index2.html',
        },
      });
    });
    test('fails if only error doc is specified', () => {
      const stack = new cdk.Stack();
      expect(() => {
        new s3.Bucket(stack, 'Website', {
          websiteErrorDocument: 'error.html',
        });
      }).toThrow(/"websiteIndexDocument" is required if "websiteErrorDocument" is set/);
    });
    test('error and index docs', () => {
      const stack = new cdk.Stack();
      new s3.Bucket(stack, 'Website', {
        websiteIndexDocument: 'index2.html',
        websiteErrorDocument: 'error.html',
      });
      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        WebsiteConfiguration: {
          IndexDocument: 'index2.html',
          ErrorDocument: 'error.html',
        },
      });
    });
    test('exports the WebsiteURL', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'Website', {
        websiteIndexDocument: 'index.html',
      });
      expect(stack.resolve(bucket.bucketWebsiteUrl)).toEqual({ 'Fn::GetAtt': ['Website32962D0B', 'WebsiteURL'] });
    });
    test('exports the WebsiteDomain', () => {
      const stack = new cdk.Stack();
      const bucket = new s3.Bucket(stack, 'Website', {
        websiteIndexDocument: 'index.html',
      });
      expect(stack.resolve(bucket.bucketWebsiteDomainName)).toEqual({
        'Fn::Select': [
          2,
          {
            'Fn::Split': ['/', { 'Fn::GetAtt': ['Website32962D0B', 'WebsiteURL'] }],
          },
        ],
      });
    });
    test('exports the WebsiteURL for imported buckets', () => {
      const stack = new cdk.Stack();
      const bucket = s3.Bucket.fromBucketName(stack, 'Website', 'my-test-bucket');
      expect(stack.resolve(bucket.bucketWebsiteUrl)).toEqual({
        'Fn::Join': [
          '',
          [
            'http://my-test-bucket.',
            {
              'Fn::FindInMap': [
                'S3staticwebsiteMap',
                { Ref: 'AWS::Region' },
                'endpoint',
              ],
            },
          ],
        ],
      });
      expect(stack.resolve(bucket.bucketWebsiteDomainName)).toEqual({
        'Fn::Join': [
          '',
          [
            'my-test-bucket.',
            {
              'Fn::FindInMap': [
                'S3staticwebsiteMap',
                { Ref: 'AWS::Region' },
                'endpoint',
              ],
            },
          ],
        ],
      });
    });
    test('exports the WebsiteURL for imported buckets with url', () => {
      const stack = new cdk.Stack();
      const bucket = s3.Bucket.fromBucketAttributes(stack, 'Website', {
        bucketName: 'my-test-bucket',
        bucketWebsiteUrl: 'http://my-test-bucket.my-test.suffix',
      });
      expect(stack.resolve(bucket.bucketWebsiteUrl)).toEqual('http://my-test-bucket.my-test.suffix');
      expect(stack.resolve(bucket.bucketWebsiteDomainName)).toEqual('my-test-bucket.my-test.suffix');
    });
    test('adds RedirectAllRequestsTo property', () => {
      const stack = new cdk.Stack();
      new s3.Bucket(stack, 'Website', {
        websiteRedirect: {
          hostName: 'www.example.com',
          protocol: s3.RedirectProtocol.HTTPS,
        },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        WebsiteConfiguration: {
          RedirectAllRequestsTo: {
            HostName: 'www.example.com',
            Protocol: 'https',
          },
        },
      });
    });
    test('fails if websiteRedirect and websiteIndex and websiteError are specified', () => {
      const stack = new cdk.Stack();
      expect(() => {
        new s3.Bucket(stack, 'Website', {
          websiteIndexDocument: 'index.html',
          websiteErrorDocument: 'error.html',
          websiteRedirect: {
            hostName: 'www.example.com',
          },
        });
      }).toThrow(/"websiteIndexDocument", "websiteErrorDocument" and, "websiteRoutingRules" cannot be set if "websiteRedirect" is used/);
    });
    test('fails if websiteRedirect and websiteRoutingRules are specified', () => {
      const stack = new cdk.Stack();
      expect(() => {
        new s3.Bucket(stack, 'Website', {
          websiteRoutingRules: [],
          websiteRedirect: {
            hostName: 'www.example.com',
          },
        });
      }).toThrow(/"websiteIndexDocument", "websiteErrorDocument" and, "websiteRoutingRules" cannot be set if "websiteRedirect" is used/);
    });
    test('adds RedirectRules property', () => {
      const stack = new cdk.Stack();
      new s3.Bucket(stack, 'Website', {
        websiteRoutingRules: [{
          hostName: 'www.example.com',
          httpRedirectCode: '302',
          protocol: s3.RedirectProtocol.HTTPS,
          replaceKey: s3.ReplaceKey.prefixWith('test/'),
          condition: {
            httpErrorCodeReturnedEquals: '200',
            keyPrefixEquals: 'prefix',
          },
        }],
      });
      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        WebsiteConfiguration: {
          RoutingRules: [{
            RedirectRule: {
              HostName: 'www.example.com',
              HttpRedirectCode: '302',
              Protocol: 'https',
              ReplaceKeyPrefixWith: 'test/',
            },
            RoutingRuleCondition: {
              HttpErrorCodeReturnedEquals: '200',
              KeyPrefixEquals: 'prefix',
            },
          }],
        },
      });
    });
    test('adds RedirectRules property with empty keyPrefixEquals condition', () => {
      const stack = new cdk.Stack();
      new s3.Bucket(stack, 'Website', {
        websiteRoutingRules: [{
          hostName: 'www.example.com',
          httpRedirectCode: '302',
          protocol: s3.RedirectProtocol.HTTPS,
          replaceKey: s3.ReplaceKey.prefixWith('test/'),
          condition: {
            httpErrorCodeReturnedEquals: '200',
            keyPrefixEquals: '',
          },
        }],
      });
      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        WebsiteConfiguration: {
          RoutingRules: [{
            RedirectRule: {
              HostName: 'www.example.com',
              HttpRedirectCode: '302',
              Protocol: 'https',
              ReplaceKeyPrefixWith: 'test/',
            },
            RoutingRuleCondition: {
              HttpErrorCodeReturnedEquals: '200',
              KeyPrefixEquals: '',
            },
          }],
        },
      });
    });
    test('fails if routingRule condition object is empty', () => {
      const stack = new cdk.Stack();
      expect(() => {
        new s3.Bucket(stack, 'Website', {
          websiteRoutingRules: [{
            httpRedirectCode: '303',
            condition: {},
          }],
        });
      }).toThrow(/The condition property cannot be an empty object/);
    });
    describe('isWebsite set properly with', () => {
      test('only index doc', () => {
        const stack = new cdk.Stack();
        const bucket = new s3.Bucket(stack, 'Website', {
          websiteIndexDocument: 'index2.html',
        });
        expect(bucket.isWebsite).toEqual(true);
      });
      test('error and index docs', () => {
        const stack = new cdk.Stack();
        const bucket = new s3.Bucket(stack, 'Website', {
          websiteIndexDocument: 'index2.html',
          websiteErrorDocument: 'error.html',
        });
        expect(bucket.isWebsite).toEqual(true);
      });
      test('redirects', () => {
        const stack = new cdk.Stack();
        const bucket = new s3.Bucket(stack, 'Website', {
          websiteRedirect: {
            hostName: 'www.example.com',
            protocol: s3.RedirectProtocol.HTTPS,
          },
        });
        expect(bucket.isWebsite).toEqual(true);
      });
      test('no website properties set', () => {
        const stack = new cdk.Stack();
        const bucket = new s3.Bucket(stack, 'Website');
        expect(bucket.isWebsite).toEqual(false);
      });
      test('imported website buckets', () => {
        const stack = new cdk.Stack();
        const bucket = s3.Bucket.fromBucketAttributes(stack, 'Website', {
          bucketArn: 'arn:aws:s3:::my-bucket',
          isWebsite: true,
        });
        expect(bucket.isWebsite).toEqual(true);
      });
      test('imported buckets', () => {
        const stack = new cdk.Stack();
        const bucket = s3.Bucket.fromBucketAttributes(stack, 'NotWebsite', {
          bucketArn: 'arn:aws:s3:::my-bucket',
        });
        expect(bucket.isWebsite).toEqual(false);
      });
    });
  });

  test('Bucket.fromBucketArn', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const bucket = s3.Bucket.fromBucketArn(stack, 'my-bucket', 'arn:aws:s3:::my-corporate-bucket');

    // THEN
    expect(bucket.bucketName).toEqual('my-corporate-bucket');
    expect(bucket.bucketArn).toEqual('arn:aws:s3:::my-corporate-bucket');
  });

  test('Bucket.fromBucketName', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const bucket = s3.Bucket.fromBucketName(stack, 'imported-bucket', 'my-bucket-name');

    // THEN
    expect(bucket.bucketName).toEqual('my-bucket-name');
    expect(stack.resolve(bucket.bucketArn)).toEqual({
      'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':s3:::my-bucket-name']],
    });
  });

  test('if a kms key is specified, it implies bucket is encrypted with kms (dah)', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const key = new kms.Key(stack, 'MyKey');
    // WHEN
    new s3.Bucket(stack, 'MyBucket', { encryptionKey: key });
    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      'BucketEncryption': {
        'ServerSideEncryptionConfiguration': [
          {
            'ServerSideEncryptionByDefault': {
              'KMSMasterKeyID': {
                'Fn::GetAtt': [
                  'MyKey6AB29FA6',
                  'Arn',
                ],
              },
              'SSEAlgorithm': 'aws:kms',
            },
          },
        ],
      },
    });
  });

  test('Bucket with Server Access Logs', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = new s3.Bucket(stack, 'AccessLogs');
    new s3.Bucket(stack, 'MyBucket', {
      serverAccessLogsBucket: accessLogBucket,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        DestinationBucketName: {
          Ref: 'AccessLogs8B620ECA',
        },
        TargetObjectKeyFormat: Match.absent(),
      },
    });
  });

  test('Bucket with Server Access Logs with Prefix', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = new s3.Bucket(stack, 'AccessLogs');
    new s3.Bucket(stack, 'MyBucket', {
      serverAccessLogsBucket: accessLogBucket,
      serverAccessLogsPrefix: 'hello',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        DestinationBucketName: {
          Ref: 'AccessLogs8B620ECA',
        },
        LogFilePrefix: 'hello',
        TargetObjectKeyFormat: Match.absent(),
      },
    });
  });

  test('Access log prefix given without bucket', () => {
    // GIVEN
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', {
      serverAccessLogsPrefix: 'hello',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        LogFilePrefix: 'hello',
        TargetObjectKeyFormat: Match.absent(),
      },
    });
  });

  test('Use simple prefix for log objects', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = new s3.Bucket(stack, 'AccessLogs');
    new s3.Bucket(stack, 'MyBucket', {
      serverAccessLogsBucket: accessLogBucket,
      targetObjectKeyFormat: s3.TargetObjectKeyFormat.simplePrefix(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        DestinationBucketName: {
          Ref: 'AccessLogs8B620ECA',
        },
        TargetObjectKeyFormat: {
          SimplePrefix: {},
          PartitionedPrefix: Match.absent(),
        },
      },
    });
  });

  test('Use partitioned prefix for log objects', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = new s3.Bucket(stack, 'AccessLogs');
    new s3.Bucket(stack, 'MyBucket', {
      serverAccessLogsBucket: accessLogBucket,
      targetObjectKeyFormat: s3.TargetObjectKeyFormat.partitionedPrefix(s3.PartitionDateSource.EVENT_TIME),
    });
    new s3.Bucket(stack, 'MyBucket2', {
      serverAccessLogsBucket: accessLogBucket,
      targetObjectKeyFormat: s3.TargetObjectKeyFormat.partitionedPrefix(s3.PartitionDateSource.DELIVERY_TIME),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        DestinationBucketName: {
          Ref: 'AccessLogs8B620ECA',
        },
        TargetObjectKeyFormat: {
          SimplePrefix: Match.absent(),
          PartitionedPrefix: {
            PartitionDateSource: 'EventTime',
          },
        },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        DestinationBucketName: {
          Ref: 'AccessLogs8B620ECA',
        },
        TargetObjectKeyFormat: {
          SimplePrefix: Match.absent(),
          PartitionedPrefix: {
            PartitionDateSource: 'DeliveryTime',
          },
        },
      },
    });
  });

  test('Bucket Allow Log delivery changes bucket Access Control should fail', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = new s3.Bucket(stack, 'AccessLogs', {
      accessControl: s3.BucketAccessControl.AUTHENTICATED_READ,
    });
    expect(() =>
      new s3.Bucket(stack, 'MyBucket', {
        serverAccessLogsBucket: accessLogBucket,
        serverAccessLogsPrefix: 'hello',
        accessControl: s3.BucketAccessControl.AUTHENTICATED_READ,
      }),
    ).toThrow(/Cannot enable log delivery to this bucket because the bucket's ACL has been set and can't be changed/);
  });

  test('Bucket skips setting up access log ACL but configures delivery for an imported target bucket', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = s3.Bucket.fromBucketName(stack, 'TargetBucket', 'target-logs-bucket');
    new s3.Bucket(stack, 'TestBucket', {
      serverAccessLogsBucket: accessLogBucket,
      serverAccessLogsPrefix: 'test/',
    });

    // THEN
    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::S3::Bucket', {
      LoggingConfiguration: {
        DestinationBucketName: stack.resolve(accessLogBucket.bucketName),
        LogFilePrefix: 'test/',
      },
    });
    template.allResourcesProperties('AWS::S3::Bucket', {
      AccessControl: Match.absent(),
    });
    Annotations.fromStack(stack).hasWarning('*', Match.stringLikeRegexp('Unable to add necessary logging permissions to imported target bucket'));
  });

  test('Bucket Allow Log delivery should use the recommended policy when flag enabled', () => {
    // GIVEN
    const stack = new cdk.Stack();
    stack.node.setContext('@aws-cdk/aws-s3:serverAccessLogsUseBucketPolicy', true);

    // WHEN
    const bucket = new s3.Bucket(stack, 'TestBucket', { serverAccessLogsPrefix: 'test' });

    // THEN
    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::S3::Bucket', {
      AccessControl: Match.absent(),
    });
    template.hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: stack.resolve(bucket.bucketName),
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([Match.objectLike({
          Effect: 'Allow',
          Principal: { Service: 'logging.s3.amazonaws.com' },
          Action: 's3:PutObject',
          Resource: stack.resolve(`${bucket.bucketArn}/test*`),
          Condition: {
            ArnLike: {
              'aws:SourceArn': stack.resolve(bucket.bucketArn),
            },
            StringEquals: {
              'aws:SourceAccount': { 'Ref': 'AWS::AccountId' },
            },
          },
        })]),
      }),
    });
  });

  test('Log Delivery bucket policy should properly set source bucket ARN/Account', () => {
    // GIVEN
    const stack = new cdk.Stack();
    stack.node.setContext('@aws-cdk/aws-s3:serverAccessLogsUseBucketPolicy', true);

    // WHEN
    const targetBucket = new s3.Bucket(stack, 'TargetBucket');
    const sourceBucket = new s3.Bucket(stack, 'SourceBucket', { serverAccessLogsBucket: targetBucket });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: stack.resolve(targetBucket.bucketName),
      PolicyDocument: Match.objectLike({
        Statement: Match.arrayWith([Match.objectLike({
          Effect: 'Allow',
          Principal: { Service: 'logging.s3.amazonaws.com' },
          Action: 's3:PutObject',
          Resource: stack.resolve(`${targetBucket.bucketArn}/*`),
          Condition: {
            ArnLike: {
              'aws:SourceArn': stack.resolve(sourceBucket.bucketArn),
            },
            StringEquals: {
              'aws:SourceAccount': stack.resolve(sourceBucket.env.account),
            },
          },
        })]),
      }),
    });
  });

  test('Log bucket has ACL enabled when feature flag is disabled', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const accessLogBucket = new s3.Bucket(stack, 'AccessLogs', {
      bucketName: 'mylogbucket',
    });

    new s3.Bucket(stack, 'MyBucket', {
      serverAccessLogsBucket: accessLogBucket,
    });

    // Logging bucket has ACL enabled when feature flag is not set
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'mylogbucket',
      OwnershipControls: {
        Rules: [{ ObjectOwnership: 'ObjectWriter' }],
      },
    });
  });

  test('ObjectOwnership is configured when AccessControl is set', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    new s3.Bucket(stack, 'AccessLogs', {
      bucketName: 'mylogbucket',
      accessControl: s3.BucketAccessControl.LOG_DELIVERY_WRITE,
    });

    // Logging bucket has ACL enabled when feature flag is not set
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'mylogbucket',
      AccessControl: 'LogDeliveryWrite',
      OwnershipControls: {
        Rules: [{ ObjectOwnership: 'ObjectWriter' }],
      },
    });
  });

  test('ObjectOwnership is not configured when AccessControl="Private"', () => {
    // GIVEN
    const stack = new cdk.Stack();

    // WHEN
    const bucket = new s3.Bucket(stack, 'AccessLogs', {
      bucketName: 'mylogbucket',
      accessControl: s3.BucketAccessControl.PRIVATE,
    });

    Tags.of(bucket).add('foo', 'bar');

    // Logging bucket has ACL enabled when feature flag is not set
    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'mylogbucket',
      AccessControl: 'Private',
      OwnershipControls: Match.absent(),
    });
  });

  test('Throws if ObjectOwnership and AccessControl do not match', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);

    // Error is thrown at construction time because `ownershipControls` is a
    // separate eagerly-assigned Box (not derived from `accessControl`). It
    // must be a separate Box because Computed.resolve() short-circuits when
    // the source resolves to `undefined` — if it were derived from
    // `accessControl`, the derive function would never run when
    // `accessControl` is undefined, which would break `allowLogDelivery`
    // (it needs to set ownershipControls even when accessControl is unset).
    expect(() => {
      new s3.Bucket(stack, 'AccessLogs', {
        bucketName: 'mylogbucket',
        accessControl: s3.BucketAccessControl.LOG_DELIVERY_WRITE,
        objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
      });
    }).toThrow(/objectOwnership must be set to \"ObjectWriter\" when accessControl is \"LogDeliveryWrite\"/);
  });

  test('Defaults for an inventory bucket', () => {
    // Given
    const stack = new cdk.Stack();

    const inventoryBucket = new s3.Bucket(stack, 'InventoryBucket');
    new s3.Bucket(stack, 'MyBucket', {
      inventories: [
        {
          destination: {
            bucket: inventoryBucket,
          },
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      InventoryConfigurations: [
        {
          Enabled: true,
          IncludedObjectVersions: 'All',
          ScheduleFrequency: 'Weekly',
          Destination: {
            Format: 'CSV',
            BucketArn: { 'Fn::GetAtt': ['InventoryBucketA869B8CB', 'Arn'] },
          },
          Id: 'MyBucketInventory0',
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: { Ref: 'InventoryBucketA869B8CB' },
      PolicyDocument: {
        Statement: Match.arrayWith([Match.objectLike({
          Action: 's3:PutObject',
          Principal: { Service: 's3.amazonaws.com' },
          Resource: [
            {
              'Fn::GetAtt': ['InventoryBucketA869B8CB', 'Arn'],
            },
            {
              'Fn::Join': ['', [{ 'Fn::GetAtt': ['InventoryBucketA869B8CB', 'Arn'] }, '/*']],
            },
          ],
        })]),
      },
    });
  });

  test('Inventory Ids are shortened to 64 characters', () => {
    // Given
    const stack = new cdk.Stack();

    const inventoryBucket = new s3.Bucket(stack, 'InventoryBucket');
    new s3.Bucket(stack, 'AVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVery@#$+:;?!&LongNodeIdName', {
      inventories: [
        {
          destination: {
            bucket: inventoryBucket,
          },
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      InventoryConfigurations: Match.arrayWith([
        Match.objectLike({
          Id: 'VeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongNodeIdNameInventory0',
        }),
      ]),
    });
  });

  test('throws when inventoryid is invalid', () => {
    // Given
    const stack = new cdk.Stack();

    const inventoryBucket = new s3.Bucket(stack, 'InventoryBucket');
    new s3.Bucket(stack, 'MyBucket2', {
      inventories: [
        {
          destination: {
            bucket: inventoryBucket,
          },
          inventoryId: 'InvalidId&123',
        },
      ],
    });

    expect(() => Template.fromStack(stack)).toThrow(/inventoryId should not exceed 64 characters and should not contain special characters except . and -, got InvalidId&123/);
  });

  test('Bucket with objectOwnership set to BUCKET_OWNER_ENFORCED', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
    });
    const template1 = Template.fromStack(stack);
    template1.templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'OwnershipControls': {
              'Rules': [
                {
                  'ObjectOwnership': 'BucketOwnerEnforced',
                },
              ],
            },
          },
          'UpdateReplacePolicy': 'Retain',
          'DeletionPolicy': 'Retain',
        },
      },
    });
  });

  test('Bucket with objectOwnership set to BUCKET_OWNER_PREFERRED', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_PREFERRED,
    });
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'OwnershipControls': {
              'Rules': [
                {
                  'ObjectOwnership': 'BucketOwnerPreferred',
                },
              ],
            },
          },
          'UpdateReplacePolicy': 'Retain',
          'DeletionPolicy': 'Retain',
        },
      },
    });
  });

  test('Bucket with objectOwnership set to OBJECT_WRITER', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      objectOwnership: s3.ObjectOwnership.OBJECT_WRITER,
    });
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'OwnershipControls': {
              'Rules': [
                {
                  'ObjectOwnership': 'ObjectWriter',
                },
              ],
            },
          },
          'UpdateReplacePolicy': 'Retain',
          'DeletionPolicy': 'Retain',
        },
      },
    });
  });

  test('Bucket with objectOwnerships set to undefined', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      objectOwnership: undefined,
    });
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'UpdateReplacePolicy': 'Retain',
          'DeletionPolicy': 'Retain',
        },
      },
    });
  });

  test('with autoDeleteObjects', () => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'MyBucket', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    Template.fromStack(stack).hasResource('AWS::S3::Bucket', {
      UpdateReplacePolicy: 'Delete',
      DeletionPolicy: 'Delete',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
      Bucket: {
        Ref: 'MyBucketF68F3FF0',
      },
      'PolicyDocument': {
        'Statement': [
          {
            Action: [
              's3:PutBucketPolicy',
              's3:GetBucket*',
              's3:List*',
              's3:DeleteObject*',
            ],
            'Effect': 'Allow',
            'Principal': {
              'AWS': {
                'Fn::GetAtt': [
                  'CustomS3AutoDeleteObjectsCustomResourceProviderRole3B1BD092',
                  'Arn',
                ],
              },
            },
            'Resource': [
              {
                'Fn::GetAtt': [
                  'MyBucketF68F3FF0',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'MyBucketF68F3FF0',
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
        'Version': '2012-10-17',
      },
    });

    Template.fromStack(stack).hasResource('Custom::S3AutoDeleteObjects', {
      'Properties': {
        'ServiceToken': {
          'Fn::GetAtt': [
            'CustomS3AutoDeleteObjectsCustomResourceProviderHandler9D90184F',
            'Arn',
          ],
        },
        'BucketName': {
          'Ref': 'MyBucketF68F3FF0',
        },
      },
      'DependsOn': [
        'MyBucketPolicyE7FBAC7B',
      ],
    });
  });

  test('with autoDeleteObjects on multiple buckets', () => {
    const stack = new cdk.Stack();

    new s3.Bucket(stack, 'Bucket1', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    new s3.Bucket(stack, 'Bucket2', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
  });

  test('autoDeleteObjects throws if RemovalPolicy is not DESTROY', () => {
    const stack = new cdk.Stack();

    expect(() => new s3.Bucket(stack, 'MyBucket', {
      autoDeleteObjects: true,
    })).toThrow(/Cannot use \'autoDeleteObjects\' property on a bucket without setting removal policy to \'DESTROY\'/);
  });

  test('bucket with transfer acceleration turned on', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      transferAcceleration: true,
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'AccelerateConfiguration': {
              'AccelerationStatus': 'Enabled',
            },
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('transferAccelerationUrlForObject returns a token with the S3 URL of the token', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const bucketWithRegion = s3.Bucket.fromBucketAttributes(stack, 'RegionalBucket', {
      bucketArn: 'arn:aws:s3:::explicit-region-bucket',
      region: 'us-west-2',
    });

    new cdk.CfnOutput(stack, 'BucketURL', { value: bucket.transferAccelerationUrlForObject() });
    new cdk.CfnOutput(stack, 'MyFileURL', { value: bucket.transferAccelerationUrlForObject('my/file.txt') });
    new cdk.CfnOutput(stack, 'YourFileURL', { value: bucket.transferAccelerationUrlForObject('/your/file.txt') }); // "/" is optional
    new cdk.CfnOutput(stack, 'RegionBucketURL', { value: bucketWithRegion.transferAccelerationUrlForObject() });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
      'Outputs': {
        'BucketURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '.s3-accelerate.amazonaws.com/',
              ],
            ],
          },
        },
        'MyFileURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '.s3-accelerate.amazonaws.com/my/file.txt',
              ],
            ],
          },
        },
        'YourFileURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '.s3-accelerate.amazonaws.com/your/file.txt',
              ],
            ],
          },
        },
        'RegionBucketURL': {
          'Value': 'https://explicit-region-bucket.s3-accelerate.amazonaws.com/',
        },
      },
    });
  });

  test('transferAccelerationUrlForObject with dual stack option returns a token with the S3 URL of the token', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    const bucketWithRegion = s3.Bucket.fromBucketAttributes(stack, 'RegionalBucket', {
      bucketArn: 'arn:aws:s3:::explicit-region-bucket',
      region: 'us-west-2',
    });

    new cdk.CfnOutput(stack, 'BucketURL', { value: bucket.transferAccelerationUrlForObject(undefined, { dualStack: true }) });
    new cdk.CfnOutput(stack, 'MyFileURL', { value: bucket.transferAccelerationUrlForObject('my/file.txt', { dualStack: true }) });
    new cdk.CfnOutput(stack, 'YourFileURL', { value: bucket.transferAccelerationUrlForObject('/your/file.txt', { dualStack: true }) }); // "/" is optional
    new cdk.CfnOutput(stack, 'RegionBucketURL', { value: bucketWithRegion.transferAccelerationUrlForObject(undefined, { dualStack: true }) });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
      'Outputs': {
        'BucketURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '.s3-accelerate.dualstack.amazonaws.com/',
              ],
            ],
          },
        },
        'MyFileURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '.s3-accelerate.dualstack.amazonaws.com/my/file.txt',
              ],
            ],
          },
        },
        'YourFileURL': {
          'Value': {
            'Fn::Join': [
              '',
              [
                'https://',
                {
                  'Ref': 'MyBucketF68F3FF0',
                },
                '.s3-accelerate.dualstack.amazonaws.com/your/file.txt',
              ],
            ],
          },
        },
        'RegionBucketURL': {
          'Value': 'https://explicit-region-bucket.s3-accelerate.dualstack.amazonaws.com/',
        },
      },
    });
  });

  test('bucket with intelligent tiering turned on', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      intelligentTieringConfigurations: [{
        name: 'foo',
      }],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'IntelligentTieringConfigurations': [
              {
                'Id': 'foo',
                'Status': 'Enabled',
                'Tierings': [],
              },
            ],
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with intelligent tiering turned on with archive access', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      intelligentTieringConfigurations: [{
        name: 'foo',
        archiveAccessTierTime: cdk.Duration.days(90),
      }],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'IntelligentTieringConfigurations': [
              {
                'Id': 'foo',
                'Status': 'Enabled',
                'Tierings': [{
                  'AccessTier': 'ARCHIVE_ACCESS',
                  'Days': 90,
                }],
              },
            ],
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with intelligent tiering turned on with deep archive access', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      intelligentTieringConfigurations: [{
        name: 'foo',
        deepArchiveAccessTierTime: cdk.Duration.days(180),
      }],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'IntelligentTieringConfigurations': [
              {
                'Id': 'foo',
                'Status': 'Enabled',
                'Tierings': [{
                  'AccessTier': 'DEEP_ARCHIVE_ACCESS',
                  'Days': 180,
                }],
              },
            ],
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('bucket with intelligent tiering turned on with all properties', () => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      intelligentTieringConfigurations: [{
        name: 'foo',
        prefix: 'bar',
        archiveAccessTierTime: cdk.Duration.days(90),
        deepArchiveAccessTierTime: cdk.Duration.days(180),
        tags: [{ key: 'test', value: 'bazz' }],
      }],
    });

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'MyBucketF68F3FF0': {
          'Type': 'AWS::S3::Bucket',
          'Properties': {
            'IntelligentTieringConfigurations': [
              {
                'Id': 'foo',
                'Prefix': 'bar',
                'Status': 'Enabled',
                'TagFilters': [
                  {
                    'Key': 'test',
                    'Value': 'bazz',
                  },
                ],
                'Tierings': [{
                  'AccessTier': 'ARCHIVE_ACCESS',
                  'Days': 90,
                },
                {
                  'AccessTier': 'DEEP_ARCHIVE_ACCESS',
                  'Days': 180,
                }],
              },
            ],
          },
          'DeletionPolicy': 'Retain',
          'UpdateReplacePolicy': 'Retain',
        },
      },
    });
  });

  test('Event Bridge notification can be enabled after the bucket is created', () => {
    const stack = new cdk.Stack();
    const bucket = new s3.Bucket(stack, 'MyBucket');
    bucket.enableEventBridgeNotification();

    Template.fromStack(stack).hasResourceProperties('Custom::S3BucketNotifications', {
      NotificationConfiguration: {
        EventBridgeConfiguration: {},
      },
    });
  });

  describe('replication', () => {
    test('default settings', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstBucket = new s3.Bucket(stack, 'DstBucket');
      new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRules: [
          { destination: dstBucket },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
          },
          Rules: [
            {
              Destination: {
                Bucket: {
                  'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                },
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
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
                Service: 's3.amazonaws.com',
              },
            },
          ],
          Version: '2012-10-17',
        },
      });
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: ['s3:GetReplicationConfiguration', 's3:ListBucket'],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['SrcBucket613E28A1', 'Arn'],
              },
            },
            {
              Action: [
                's3:GetObjectVersionForReplication',
                's3:GetObjectVersionAcl',
                's3:GetObjectVersionTagging',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'SrcBucket613E28A1',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: [
                's3:ReplicateObject',
                's3:ReplicateDelete',
                's3:ReplicateTags',
                's3:ObjectOwnerOverrideToBucketOwner',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'DstBucket3E241BF2',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
          ],
          'Version': '2012-10-17',
        },
        Roles: [
          {
            'Ref': 'SrcBucketReplicationRole5B31865A',
          },
        ],
      });
    });

    test('full settings', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstBucket = new s3.Bucket(stack, 'DstBucket');
      const kmsKey = new kms.Key(stack, 'Key');

      new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRules: [
          {
            destination: dstBucket,
            replicationTimeControl: s3.ReplicationTimeValue.FIFTEEN_MINUTES,
            metrics: s3.ReplicationTimeValue.FIFTEEN_MINUTES,
            kmsKey,
            storageClass: s3.StorageClass.GLACIER,
            sseKmsEncryptedObjects: true,
            replicaModifications: true,
            id: 'rule1',
            priority: 1,
            deleteMarkerReplication: false,
            filter: {
              prefix: 'filterWord',
              tags: [{ key: 'filterKey', value: 'filterValue' }],
            },
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
          },
          Rules: [
            {
              Id: 'rule1',
              Priority: 1,
              Status: 'Enabled',
              Destination: {
                Bucket: {
                  'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                },
                StorageClass: 'GLACIER',
                EncryptionConfiguration: {
                  ReplicaKmsKeyID: {
                    'Fn::GetAtt': ['Key961B73FD', 'Arn'],
                  },
                },
                ReplicationTime: {
                  Status: 'Enabled',
                  Time: {
                    Minutes: 15,
                  },
                },
                Metrics: {
                  Status: 'Enabled',
                  EventThreshold: {
                    Minutes: 15,
                  },
                },
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
              SourceSelectionCriteria: {
                ReplicaModifications: {
                  Status: 'Enabled',
                },
                SseKmsEncryptedObjects: {
                  Status: 'Enabled',
                },
              },
              Filter: {
                And: {
                  Prefix: 'filterWord',
                  TagFilters: [{
                    Key: 'filterKey',
                    Value: 'filterValue',
                  }],
                },
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: ['s3:GetReplicationConfiguration', 's3:ListBucket'],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['SrcBucket613E28A1', 'Arn'],
              },
            },
            {
              Action: [
                's3:GetObjectVersionForReplication',
                's3:GetObjectVersionAcl',
                's3:GetObjectVersionTagging',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'SrcBucket613E28A1',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: [
                's3:ReplicateObject',
                's3:ReplicateDelete',
                's3:ReplicateTags',
                's3:ObjectOwnerOverrideToBucketOwner',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'DstBucket3E241BF2',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: [
                'kms:Encrypt',
                'kms:ReEncrypt*',
                'kms:GenerateDataKey*',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'Key961B73FD',
                  'Arn',
                ],
              },
            },
          ],
          'Version': '2012-10-17',
        },
        Roles: [
          {
            'Ref': 'SrcBucketReplicationRole5B31865A',
          },
        ],
      });
    });

    test('use custom replication role', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstBucket = new s3.Bucket(stack, 'DstBucket');
      const replicationRole = new iam.Role(stack, 'ReplicationRole', {
        assumedBy: new iam.ServicePrincipal('s3.amazonaws.com'),
      });
      (replicationRole.node.defaultChild as iam.CfnRole).overrideLogicalId('CustomReplicationRole');
      new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRole,
        replicationRules: [
          { destination: dstBucket },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['CustomReplicationRole', 'Arn'],
          },
          Rules: [
            {
              Destination: {
                Bucket: {
                  'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                },
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
            },
          ],
        },
      });
      // assert that the built-in replication role is not created
      Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);
    });

    test('grant permissions to custom replication role', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstEncryptionKey = new kms.Key(stack, 'DstEncryptionKey');
      const srcEncryptionKey = new kms.Key(stack, 'SrcEncryptionKey');
      const dstBucket = new s3.Bucket(stack, 'DstBucket', {
        encryptionKey: dstEncryptionKey,
      });
      const dstBucketNoEncryption = new s3.Bucket(stack, 'DstBucketNoEncryption');
      const replicationRole = new iam.Role(stack, 'ReplicationRole', {
        assumedBy: new iam.ServicePrincipal('s3.amazonaws.com'),
      });

      const bucket = new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRole,
        encryptionKey: srcEncryptionKey,
        replicationRules: [
          { destination: dstBucket, kmsKey: dstEncryptionKey, priority: 1 },
          { destination: dstBucketNoEncryption, priority: 2 },
        ],
      });
      const grant = bucket.grantReplicationPermission(replicationRole, {
        sourceDecryptionKey: srcEncryptionKey,
        destinations: [{
          bucket: dstBucket,
          encryptionKey: dstEncryptionKey,
        }],
      });

      grant.assertSuccess();
      // 5 because of the 3 actions for `s3:*`, the 1 kms decrypt action for source bucket, and the 1 kms encrypt action for destination bucket
      expect(grant.principalStatements).toHaveLength(5);

      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['ReplicationRoleCE149CEC', 'Arn'],
          },
          Rules: [
            {
              Priority: 1,
              Destination: {
                Bucket: {
                  'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                },
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
            },
            {
              Priority: 2,
              Destination: {
                Bucket: {
                  'Fn::GetAtt': ['DstBucketNoEncryptionB500C798', 'Arn'],
                },
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
            },
          ],
        },
      });

      // assert that the built-in replication role is not created
      Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: ['s3:GetReplicationConfiguration', 's3:ListBucket'],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['SrcBucket613E28A1', 'Arn'],
              },
            },
            {
              Action: [
                's3:GetObjectVersionForReplication',
                's3:GetObjectVersionAcl',
                's3:GetObjectVersionTagging',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'SrcBucket613E28A1',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: [
                's3:ReplicateObject',
                's3:ReplicateDelete',
                's3:ReplicateTags',
                's3:ObjectOwnerOverrideToBucketOwner',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'DstBucket3E241BF2',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: [
                'kms:Encrypt',
                'kms:ReEncrypt*',
                'kms:GenerateDataKey*',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'DstEncryptionKey8C28AFFA',
                  'Arn',
                ],
              },
            },
            {
              Action: 'kms:Decrypt',
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['SrcEncryptionKeyCCB1A334', 'Arn'],
              },
            },
          ],
          'Version': '2012-10-17',
        },
        Roles: [
          {
            'Ref': 'ReplicationRoleCE149CEC',
          },
        ],
      });
    });

    test('throw error when attempting to grant permissions with no destinations', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstBucket = new s3.Bucket(stack, 'DstBucket');
      new s3.Bucket(stack, 'DstBucketNoEncryption');
      const replicationRole = new iam.Role(stack, 'ReplicationRole', {
        assumedBy: new iam.ServicePrincipal('s3.amazonaws.com'),
      });

      const bucket = new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRole,
        replicationRules: [
          { destination: dstBucket, priority: 1 },
        ],
      });

      expect(() => {
        bucket.grantReplicationPermission(replicationRole, {
          destinations: [],
        });
      }).toThrow('At least one destination bucket must be specified in the destinations array');
    });

    test('cross account', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '111111111111',
          // To avoid generating replication role name error, we need to set the region explicitly
          region: 'us-east-1',
        },
      });
      const dstStack = new cdk.Stack(app, 'DstStack', {
        env: {
          account: '222222222222',
        },
      });
      const dstBucket = new s3.Bucket(dstStack, 'DstBucket', {
        bucketName: 'another-account-dst-bucket',
      });

      const sourcebucket = new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRules: [
          {
            destination: dstBucket,
          },
        ],
      });

      if (sourcebucket.replicationRoleArn) {
        dstBucket.addReplicationPolicy(sourcebucket.replicationRoleArn);
      }

      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
          },
          Rules: [
            {
              Destination: {
                Bucket: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':s3:::another-account-dst-bucket',
                    ],
                  ],
                },
                Account: '222222222222',
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
            },
          ],
        },
      });

      Template.fromStack(dstStack).hasResourceProperties('AWS::S3::BucketPolicy', {
        Bucket: {
          Ref: 'DstBucket3E241BF2',
        },
        PolicyDocument: {
          Statement: [
            {
              Action: ['s3:GetBucketVersioning', 's3:PutBucketVersioning'],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'DstBucket3E241BF2',
                  'Arn',
                ],
              },
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::111111111111:role/CDKReplicationRole',
                    ],
                  ],
                },
              },
            },
            {
              Action: ['s3:ReplicateObject', 's3:ReplicateDelete'],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'DstBucket3E241BF2',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::111111111111:role/CDKReplicationRole',
                    ],
                  ],
                },
              },
            },
          ],
        },
      });
    });

    test('cross account with accessControlTransition', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '111111111111',
          // To avoid generating replication role name error, we need to set the region explicitly
          region: 'us-east-1',
        },
      });
      const dstStack = new cdk.Stack(app, 'DstStack', {
        env: {
          account: '222222222222',
        },
      });
      const dstBucket = new s3.Bucket(dstStack, 'DstBucket', {
        bucketName: 'another-account-dst-bucket',
      });

      const sourcebucket = new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        replicationRules: [
          {
            destination: dstBucket,
            accessControlTransition: true,
          },
        ],
      });

      if (sourcebucket.replicationRoleArn) {
        dstBucket.addReplicationPolicy(sourcebucket.replicationRoleArn, true, '111111111111');
      }
      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
          },
          Rules: [
            {
              Destination: {
                Bucket: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':s3:::another-account-dst-bucket',
                    ],
                  ],
                },
                Account: '222222222222',
                AccessControlTranslation: {
                  Owner: 'Destination',
                },
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
            },
          ],
        },
      });

      Template.fromStack(dstStack).hasResourceProperties('AWS::S3::BucketPolicy', {
        Bucket: {
          Ref: 'DstBucket3E241BF2',
        },
        PolicyDocument: {
          Statement: [
            {
              Action: ['s3:GetBucketVersioning', 's3:PutBucketVersioning'],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'DstBucket3E241BF2',
                  'Arn',
                ],
              },
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::111111111111:role/CDKReplicationRole',
                    ],
                  ],
                },
              },
            },
            {
              Action: ['s3:ReplicateObject', 's3:ReplicateDelete'],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'DstBucket3E241BF2',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::111111111111:role/CDKReplicationRole',
                    ],
                  ],
                },
              },
            },
            {
              Action: 's3:ObjectOwnerOverrideToBucketOwner',
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'DstBucket3E241BF2',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
              Principal: {
                AWS: {
                  'Fn::Join': [
                    '',
                    [
                      'arn:',
                      {
                        'Ref': 'AWS::Partition',
                      },
                      ':iam::111111111111:root',
                    ],
                  ],
                },
              },
            },
          ],
        },
      });
    });

    test('throw error for enabled accessControlTransition with same account replication', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstBucket = new s3.Bucket(stack, 'DstBucket');

      expect(() => {
        new s3.Bucket(stack, 'SrcBucket', {
          versioned: true,
          replicationRules: [
            {
              destination: dstBucket,
              accessControlTransition: true,
            },
          ],
        });
      }).toThrow('accessControlTranslation is only supported for cross-account replication');
    });

    describe('throw error when replicationRole is provided without valid replicationRules', () => {
      test('fails when replicationRules is not specified', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');

        expect(() => {
          new s3.Bucket(stack, 'SrcBucket', {
            versioned: true,
            replicationRole: new iam.Role(stack, 'ReplicationRole', {
              assumedBy: new iam.ServicePrincipal('s3.amazonaws.com'),
            }),
          });
        }).toThrow('cannot specify replicationRole when replicationRules is empty');
      });

      test('fails when replicationRules is an empty array', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');

        expect(() => {
          new s3.Bucket(stack, 'SrcBucket', {
            versioned: true,
            replicationRole: new iam.Role(stack, 'ReplicationRole', {
              assumedBy: new iam.ServicePrincipal('s3.amazonaws.com'),
            }),
            replicationRules: [],
          });
        }).toThrow('cannot specify replicationRole when replicationRules is empty');
      });
    });

    test('source object encryption', () => {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const dstBucket = new s3.Bucket(stack, 'DstBucket');
      const kmsKey = new kms.Key(stack, 'Key');

      new s3.Bucket(stack, 'SrcBucket', {
        versioned: true,
        encryption: s3.BucketEncryption.KMS,
        encryptionKey: kmsKey,
        replicationRules: [
          { destination: dstBucket },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
        VersioningConfiguration: { Status: 'Enabled' },
        ReplicationConfiguration: {
          Role: {
            'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
          },
          Rules: [
            {
              Destination: {
                Bucket: {
                  'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                },
              },
              Status: 'Enabled',
              Filter: {
                Prefix: '',
              },
              DeleteMarkerReplication: {
                Status: 'Disabled',
              },
            },
          ],
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: ['s3:GetReplicationConfiguration', 's3:ListBucket'],
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': ['SrcBucket613E28A1', 'Arn'],
              },
            },
            {
              Action: [
                's3:GetObjectVersionForReplication',
                's3:GetObjectVersionAcl',
                's3:GetObjectVersionTagging',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'SrcBucket613E28A1',
                        'Arn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: [
                's3:ReplicateObject',
                's3:ReplicateDelete',
                's3:ReplicateTags',
                's3:ObjectOwnerOverrideToBucketOwner',
              ],
              Effect: 'Allow',
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    { 'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'] },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: 'kms:Decrypt',
              Effect: 'Allow',
              Resource: {
                'Fn::GetAtt': [
                  'Key961B73FD',
                  'Arn',
                ],
              },
            },
          ],
          'Version': '2012-10-17',
        },
        Roles: [
          {
            'Ref': 'SrcBucketReplicationRole5B31865A',
          },
        ],
      });
    });

    describe('filter', () => {
      test('specify only prefix filter', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');
        const dstBucket = new s3.Bucket(stack, 'DstBucket');

        new s3.Bucket(stack, 'SrcBucket', {
          versioned: true,
          replicationRules: [
            {
              destination: dstBucket,
              filter: { prefix: 'filterWord' },
            },
          ],
        });

        Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
          VersioningConfiguration: { Status: 'Enabled' },
          ReplicationConfiguration: {
            Role: {
              'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
            },
            Rules: [
              {
                Status: 'Enabled',
                Destination: {
                  Bucket: {
                    'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                  },
                },
                Filter: {
                  Prefix: 'filterWord',
                },
                DeleteMarkerReplication: {
                  Status: 'Disabled',
                },
              },
            ],
          },
        });
      });

      test('specify only tag filter', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');
        const dstBucket = new s3.Bucket(stack, 'DstBucket');

        new s3.Bucket(stack, 'SrcBucket', {
          versioned: true,
          replicationRules: [
            {
              destination: dstBucket,
              filter: {
                tags: [{ key: 'filterKey', value: 'filterValue' }],
              },
            },
          ],
        });

        Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
          VersioningConfiguration: { Status: 'Enabled' },
          ReplicationConfiguration: {
            Role: {
              'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
            },
            Rules: [
              {
                Status: 'Enabled',
                Destination: {
                  Bucket: {
                    'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                  },
                },
                Filter: {
                  And: {
                    TagFilters: [{ Key: 'filterKey', Value: 'filterValue' }],
                    Prefix: '',
                  },
                },
                DeleteMarkerReplication: {
                  Status: 'Disabled',
                },
              },
            ],
          },
        });
      });

      test('specify multiple tag filters', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');
        const dstBucket = new s3.Bucket(stack, 'DstBucket');

        new s3.Bucket(stack, 'SrcBucket', {
          versioned: true,
          replicationRules: [
            {
              destination: dstBucket,
              filter: {
                tags: [
                  { key: 'filterKey1', value: 'filterValue1' },
                  { key: 'filterKey2', value: 'filterValue2' },
                ],
              },
            },
          ],
        });

        Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
          VersioningConfiguration: { Status: 'Enabled' },
          ReplicationConfiguration: {
            Role: {
              'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
            },
            Rules: [
              {
                Status: 'Enabled',
                Destination: {
                  Bucket: {
                    'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                  },
                },
                Filter: {
                  And: {
                    TagFilters: [
                      { Key: 'filterKey1', Value: 'filterValue1' },
                      { Key: 'filterKey2', Value: 'filterValue2' },
                    ],
                    Prefix: '',
                  },
                },
                DeleteMarkerReplication: {
                  Status: 'Disabled',
                },
              },
            ],
          },
        });
      });

      test('specify both prefix and tag filters', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');
        const dstBucket = new s3.Bucket(stack, 'DstBucket');

        new s3.Bucket(stack, 'SrcBucket', {
          versioned: true,
          replicationRules: [
            {
              destination: dstBucket,
              filter: {
                prefix: 'filterWord',
                tags: [{ key: 'filterKey', value: 'filterValue' }],
              },
            },
          ],
        });

        Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
          VersioningConfiguration: { Status: 'Enabled' },
          ReplicationConfiguration: {
            Role: {
              'Fn::GetAtt': ['SrcBucketReplicationRole5B31865A', 'Arn'],
            },
            Rules: [
              {
                Status: 'Enabled',
                Destination: {
                  Bucket: {
                    'Fn::GetAtt': ['DstBucket3E241BF2', 'Arn'],
                  },
                },
                Filter: {
                  And: {
                    Prefix: 'filterWord',
                    TagFilters: [{ Key: 'filterKey', Value: 'filterValue' }],
                  },
                },
                DeleteMarkerReplication: {
                  Status: 'Disabled',
                },
              },
            ],
          },
        });
      });

      test('throw error for specifying tag filter when delete markter replication is enabled', () => {
        const app = new cdk.App();
        const stack = new cdk.Stack(app, 'stack');
        const dstBucket = new s3.Bucket(stack, 'DstBucket');

        expect(() => {
          new s3.Bucket(stack, 'SrcBucket', {
            versioned: true,
            replicationRules: [
              {
                destination: dstBucket,
                deleteMarkerReplication: true,
                filter: {
                  tags: [{ key: 'filterKey', value: 'filterValue' }],
                },
              },
            ],
          });
        }).toThrow('tag filter cannot be specified when \'deleteMarkerReplication\' is enabled.');
      });
    });
  });

  test.each([
    [s3.TransitionDefaultMinimumObjectSize.ALL_STORAGE_CLASSES_128_K, 'all_storage_classes_128K'],
    [s3.TransitionDefaultMinimumObjectSize.VARIES_BY_STORAGE_CLASS, 'varies_by_storage_class'],
  ])('transitionDefaultMinimumObjectSize %s can be specified', (key, value) => {
    const stack = new cdk.Stack();
    new s3.Bucket(stack, 'MyBucket', {
      transitionDefaultMinimumObjectSize: key,
      lifecycleRules: [
        {
          expiration: cdk.Duration.days(365),
        },
      ],
    });

    Template.fromStack(stack).hasResourceProperties('AWS::S3::Bucket', {
      LifecycleConfiguration: {
        TransitionDefaultMinimumObjectSize: value,
        Rules: [{
          ExpirationInDays: 365,
        }],
      },
    });
  });

  describe('property injection', () => {
    test('Compare AccessLog Bucket Injector Templates', () => {
      // This case is complex because we are creating a bucket within Bucket constructor.
      // GIVEN - with Injector
      const app = new cdk.App({
        propertyInjectors: [
          accessLogBucketInjector,
        ],
      });
      const stack = new cdk.Stack(app, 'MyStack', {});
      new s3.Bucket(stack, 'my-bucket-1', {});
      const template = Template.fromStack(stack).toJSON();

      // WHEN - no Injector, but props
      const app2 = new cdk.App({});
      const stack2 = new cdk.Stack(app2, 'MyStack', {});
      const accessLog = new s3.Bucket(stack2, 'access-logging-12345', {
        accessControl: undefined,
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        encryption: s3.BucketEncryption.KMS,
        enforceSSL: true,
        publicReadAccess: false,
        lifecycleRules: [],
        removalPolicy: cdk.RemovalPolicy.RETAIN,
      });
      new s3.Bucket(stack2, 'my-bucket-1', {
        accessControl: undefined,
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        encryption: s3.BucketEncryption.KMS,
        enforceSSL: true,
        publicReadAccess: false,
        lifecycleRules: [],
        serverAccessLogsBucket: accessLog,
      });

      // THEN
      Template.fromStack(stack2).templateMatches(
        Match.exact(template),
      );
    });

    test('Compare Bucket Injector Templates', () => {
      // GIVEN - with Injector
      const app = new cdk.App({
        propertyInjectors: [
          bucketInjector,
        ],
      });
      const stack = new cdk.Stack(app, 'MyStack', {});
      new s3.Bucket(stack, 'my-bucket-1', {});
      const template = Template.fromStack(stack).toJSON();

      // WHEN - no Injector, but props
      const app2 = new cdk.App({});
      const stack2 = new cdk.Stack(app2, 'MyStack', {});
      new s3.Bucket(stack2, 'my-bucket-1', {
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        enforceSSL: true,
      });

      // THEN
      Template.fromStack(stack2).templateMatches(
        Match.exact(template),
      );
    });
  });
});

class AccessBucketInjector implements cdk.IPropertyInjector {
  public readonly constructUniqueId: string;

  // Skip property injection if this class attribute is set to true
  private _skip: boolean;

  constructor() {
    this._skip = false;
    this.constructUniqueId = s3.Bucket.PROPERTY_INJECTION_ID;
  }

  inject(originalProps: s3.BucketProps, context: cdk.InjectionContext): s3.BucketProps {
    const commonInjectionValues = {
      accessControl: undefined,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.KMS,
      enforceSSL: true,
      publicReadAccess: false,
      lifecycleRules: [],
    };

    // Don't set up access logging bucket if this._skip=true
    if (this._skip) {
      return commonInjectionValues;
    }

    let accessLoggingBucket = originalProps.serverAccessLogsBucket;
    if (!accessLoggingBucket) {
      // Set the _skip flag to disable indefinite access log bucket creation loop
      this._skip = true;

      accessLoggingBucket = new s3.Bucket(context.scope, 'access-logging-12345', {
        ...commonInjectionValues,
        removalPolicy: originalProps.removalPolicy ?? cdk.RemovalPolicy.RETAIN,
      });

      // reset the _skip flag
      this._skip = false;
    }
    return {
      ...commonInjectionValues,
      serverAccessLogsBucket: accessLoggingBucket,
      ...originalProps,
    };
  }
}

const accessLogBucketInjector = new AccessBucketInjector();

class MyBucketPropsInjector implements cdk.IPropertyInjector {
  public readonly constructUniqueId: string;

  constructor() {
    this.constructUniqueId = s3.Bucket.PROPERTY_INJECTION_ID;
  }

  inject(originalProps: any, _context: cdk.InjectionContext): any {
    const newProps = {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      ...originalProps,
    };
    return newProps;
  }
}

const bucketInjector = new MyBucketPropsInjector();
