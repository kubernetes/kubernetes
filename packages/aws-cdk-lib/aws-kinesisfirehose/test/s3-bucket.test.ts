import { Match, Template } from '../../assertions';
import * as glue from '../../aws-glue';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import * as lambda from '../../aws-lambda';
import * as logs from '../../aws-logs';
import * as s3 from '../../aws-s3';
import * as cdk from '../../core';
import * as firehose from '../lib';

describe('S3 destination', () => {
  let stack: cdk.Stack;
  let bucket: s3.IBucket;
  let destinationRole: iam.IRole;

  beforeEach(() => {
    stack = new cdk.Stack();
    bucket = new s3.Bucket(stack, 'Bucket');
    destinationRole = new iam.Role(stack, 'Destination Role', {
      assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
    });
  });

  it('provides defaults when no configuration is provided', () => {
    new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket, { role: destinationRole }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
      ExtendedS3DestinationConfiguration: {
        BucketARN: stack.resolve(bucket.bucketArn),
        CloudWatchLoggingOptions: {
          Enabled: true,
        },
        RoleARN: stack.resolve(destinationRole.roleArn),
      },
    });
    Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 1);
    Template.fromStack(stack).resourceCountIs('AWS::Logs::LogStream', 1);
  });

  it('creates a role when none is provided', () => {
    new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.S3Bucket(bucket),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
      ExtendedS3DestinationConfiguration: {
        RoleARN: {
          'Fn::GetAtt': [
            'DeliveryStreamS3DestinationRoleD96B8345',
            'Arn',
          ],
        },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Effect: 'Allow',
            Principal: {
              Service: 'firehose.amazonaws.com',
            },
            Action: 'sts:AssumeRole',
          },
        ],
      },
    });
  });

  it('grants read/write access to the bucket', () => {
    const destination = new firehose.S3Bucket(bucket, { role: destinationRole });

    new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: destination,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      Roles: [stack.resolve(destinationRole.roleName)],
      PolicyDocument: {
        Statement: Match.arrayWith([
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
              stack.resolve(bucket.bucketArn),
              { 'Fn::Join': ['', [stack.resolve(bucket.bucketArn), '/*']] },
            ],
          },
        ]),
      },
    });
  });

  it('bucket and log group grants are depended on by delivery stream', () => {
    const logGroup = logs.LogGroup.fromLogGroupName(stack, 'Log Group', 'evergreen');
    const destination = new firehose.S3Bucket(bucket, {
      role: destinationRole, loggingConfig: new firehose.EnableLogging(logGroup),
    });
    new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: destination,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyName: 'DestinationRoleDefaultPolicy1185C75D',
      Roles: [stack.resolve(destinationRole.roleName)],
      PolicyDocument: {
        Statement: [
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
              stack.resolve(bucket.bucketArn),
              { 'Fn::Join': ['', [stack.resolve(bucket.bucketArn), '/*']] },
            ],
          },
          {
            Action: [
              'logs:CreateLogStream',
              'logs:PutLogEvents',
            ],
            Effect: 'Allow',
            Resource: stack.resolve(logGroup.logGroupArn),
          },
        ],
      },
    });
    Template.fromStack(stack).hasResource('AWS::KinesisFirehose::DeliveryStream', {
      DependsOn: ['DestinationRoleDefaultPolicy1185C75D'],
    });
  });

  describe('logging', () => {
    it('creates resources and configuration by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket),
      });

      Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 1);
      Template.fromStack(stack).resourceCountIs('AWS::Logs::LogStream', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CloudWatchLoggingOptions: {
            Enabled: true,
          },
        },
      });
    });

    it('does not create resources or configuration if disabled', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, { loggingConfig: new firehose.DisableLogging() }),
      });

      Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 0);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CloudWatchLoggingOptions: Match.absent(),
        },
      });
    });

    it('uses provided log group', () => {
      const logGroup = new logs.LogGroup(stack, 'Log Group');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, { loggingConfig: new firehose.EnableLogging(logGroup) }),
      });

      Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CloudWatchLoggingOptions: {
            Enabled: true,
            LogGroupName: stack.resolve(logGroup.logGroupName),
          },
        },
      });
    });

    it('grants log group write permissions to destination role', () => {
      const logGroup = new logs.LogGroup(stack, 'Log Group');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, {
          loggingConfig: new firehose.EnableLogging(logGroup), role: destinationRole,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        Roles: [stack.resolve(destinationRole.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([
            {
              Action: [
                'logs:CreateLogStream',
                'logs:PutLogEvents',
              ],
              Effect: 'Allow',
              Resource: stack.resolve(logGroup.logGroupArn),
            },
          ]),
        },
      });
    });
  });

  describe('processing configuration with deprecated processor prop', () => {
    let lambdaFunction: lambda.IFunction;
    let basicLambdaProcessor: firehose.LambdaFunctionProcessor;
    let destinationWithBasicLambdaProcessor: firehose.S3Bucket;

    beforeEach(() => {
      lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      basicLambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction);
      destinationWithBasicLambdaProcessor = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        processor: basicLambdaProcessor,
      });
    });

    it('creates configuration for LambdaFunctionProcessor', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destinationWithBasicLambdaProcessor,
      });

      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'Lambda',
              Parameters: [
                {
                  ParameterName: 'RoleArn',
                  ParameterValue: stack.resolve(destinationRole.roleArn),
                },
                {
                  ParameterName: 'LambdaArn',
                  ParameterValue: stack.resolve(lambdaFunction.functionArn),
                },
              ],
            }],
          },
        },
      });
    });

    it('set all optional parameters', () => {
      const processor = new firehose.LambdaFunctionProcessor(lambdaFunction, {
        bufferInterval: cdk.Duration.minutes(1),
        bufferSize: cdk.Size.mebibytes(1),
        retries: 5,
      });
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        processor: processor,
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'Lambda',
              Parameters: [
                {
                  ParameterName: 'RoleArn',
                  ParameterValue: stack.resolve(destinationRole.roleArn),
                },
                {
                  ParameterName: 'LambdaArn',
                  ParameterValue: stack.resolve(lambdaFunction.functionArn),
                },
                {
                  ParameterName: 'BufferIntervalInSeconds',
                  ParameterValue: '60',
                },
                {
                  ParameterName: 'BufferSizeInMBs',
                  ParameterValue: '1',
                },
                {
                  ParameterName: 'NumberOfRetries',
                  ParameterValue: '5',
                },
              ],
            }],
          },
        },
      });
    });

    it('grants invoke access to the lambda function and delivery stream depends on grant', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destinationWithBasicLambdaProcessor,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyName: 'DestinationRoleDefaultPolicy1185C75D',
        Roles: [stack.resolve(destinationRole.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: [
                stack.resolve(lambdaFunction.functionArn),
                { 'Fn::Join': ['', [stack.resolve(lambdaFunction.functionArn), ':*']] },
              ],
            },
          ]),
        },
      });
      Template.fromStack(stack).hasResource('AWS::KinesisFirehose::DeliveryStream', {
        DependsOn: ['DestinationRoleDefaultPolicy1185C75D'],
      });
    });
  });

  describe('processing configuration with processors prop', () => {
    it('creates configuration for LambdaFunctionProcessor', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const lambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction);
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        processors: [lambdaProcessor],
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });

      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'Lambda',
              Parameters: [
                {
                  ParameterName: 'RoleArn',
                  ParameterValue: stack.resolve(destinationRole.roleArn),
                },
                {
                  ParameterName: 'LambdaArn',
                  ParameterValue: stack.resolve(lambdaFunction.functionArn),
                },
              ],
            }],
          },
        },
      });
    });

    it('set all optional parameters', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const lambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction, {
        bufferInterval: cdk.Duration.minutes(1),
        bufferSize: cdk.Size.mebibytes(1),
        retries: 5,
      });
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        processors: [lambdaProcessor],
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });

      Template.fromStack(stack).resourceCountIs('AWS::Lambda::Function', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'Lambda',
              Parameters: [
                {
                  ParameterName: 'RoleArn',
                  ParameterValue: stack.resolve(destinationRole.roleArn),
                },
                {
                  ParameterName: 'LambdaArn',
                  ParameterValue: stack.resolve(lambdaFunction.functionArn),
                },
                {
                  ParameterName: 'BufferIntervalInSeconds',
                  ParameterValue: '60',
                },
                {
                  ParameterName: 'BufferSizeInMBs',
                  ParameterValue: '1',
                },
                {
                  ParameterName: 'NumberOfRetries',
                  ParameterValue: '5',
                },
              ],
            }],
          },
        },
      });
    });

    it('grants invoke access to the lambda function and delivery stream depends on grant', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const lambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction);
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        processors: [lambdaProcessor],
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyName: 'DestinationRoleDefaultPolicy1185C75D',
        Roles: [stack.resolve(destinationRole.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([
            {
              Action: 'lambda:InvokeFunction',
              Effect: 'Allow',
              Resource: [
                stack.resolve(lambdaFunction.functionArn),
                { 'Fn::Join': ['', [stack.resolve(lambdaFunction.functionArn), ':*']] },
              ],
            },
          ]),
        },
      });
      Template.fromStack(stack).hasResource('AWS::KinesisFirehose::DeliveryStream', {
        DependsOn: ['DestinationRoleDefaultPolicy1185C75D'],
      });
    });

    it('creates configuration with built-in processors', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const lambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction);
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        processors: [
          new firehose.DecompressionProcessor(),
          new firehose.CloudWatchLogProcessor({ dataMessageExtraction: true }),
          lambdaProcessor,
          new firehose.AppendDelimiterToRecordProcessor(),
        ],
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'Decompression',
              Parameters: [
                { ParameterName: 'CompressionFormat', ParameterValue: 'GZIP' },
              ],
            }, {
              Type: 'CloudWatchLogProcessing',
              Parameters: [
                { ParameterName: 'DataMessageExtraction', ParameterValue: 'true' },
              ],
            }, {
              Type: 'Lambda',
              Parameters: [
                { ParameterName: 'RoleArn', ParameterValue: stack.resolve(destinationRole.roleArn) },
                { ParameterName: 'LambdaArn', ParameterValue: stack.resolve(lambdaFunction.functionArn) },
              ],
            }, {
              Type: 'AppendDelimiterToRecord',
              Parameters: [],
            }],
          },
        },
      });
    });

    test('CloudWatchLogProcessor throws when dataMessageExtraction is false', () => {
      expect(() => {
        new firehose.CloudWatchLogProcessor({ dataMessageExtraction: false });
      }).toThrow('dataMessageExtraction must be true.');
    });
  });

  it('throws when specified both processor and processors', () => {
    const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
      runtime: lambda.Runtime.NODEJS_LATEST,
      code: lambda.Code.fromInline('foo'),
      handler: 'bar',
    });
    const lambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction);
    const destination = new firehose.S3Bucket(bucket, {
      role: destinationRole,
      processor: lambdaProcessor,
      processors: [lambdaProcessor],
    });

    expect(() => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });
    }).toThrow("You can specify either 'processors' or 'processor', not both.");
  });

  describe('compression', () => {
    it('configures when specified', () => {
      const destination = new firehose.S3Bucket(bucket, {
        compression: firehose.Compression.GZIP,
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CompressionFormat: 'GZIP',
        },
      });
    });

    it('allows custom compression types', () => {
      const destination = new firehose.S3Bucket(bucket, {
        compression: firehose.Compression.of('HADOOP_SNAPPY'),
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CompressionFormat: 'HADOOP_SNAPPY',
        },
      });
    });
  });

  describe('buffering', () => {
    it('creates configuration when interval and size provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, {
          bufferingInterval: cdk.Duration.minutes(1),
          bufferingSize: cdk.Size.mebibytes(1),
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          BufferingHints: {
            IntervalInSeconds: 60,
            SizeInMBs: 1,
          },
        },
      });
    });

    it('validates bufferingInterval', () => {
      expect(() => new firehose.DeliveryStream(stack, 'DeliveryStream2', {
        destination: new firehose.S3Bucket(bucket, {
          bufferingInterval: cdk.Duration.minutes(16),
          bufferingSize: cdk.Size.mebibytes(1),
        }),
      })).toThrow('Buffering interval must be less than 900 seconds, got 960 seconds.');
    });

    it('validates bufferingSize', () => {
      expect(() => new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, {
          bufferingInterval: cdk.Duration.minutes(1),
          bufferingSize: cdk.Size.mebibytes(0),

        }),
      })).toThrow('Buffering size must be at least 1 MiB, got 0 MiBs.');

      expect(() => new firehose.DeliveryStream(stack, 'DeliveryStream2', {
        destination: new firehose.S3Bucket(bucket, {
          bufferingInterval: cdk.Duration.minutes(1),
          bufferingSize: cdk.Size.mebibytes(256),
        }),
      })).toThrow('Buffering size must be at most 128 MiBs, got 256 MiBs.');
    });
  });

  describe('destination encryption', () => {
    it('creates configuration', () => {
      const key = new kms.Key(stack, 'Key');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, {
          encryptionKey: key,
          role: destinationRole,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          EncryptionConfiguration: {
            KMSEncryptionConfig: {
              AWSKMSKeyARN: stack.resolve(key.keyArn),
            },
          },
        },
      });
    });

    it('grants encrypt/decrypt access to the destination encryptionKey', () => {
      const key = new kms.Key(stack, 'Key');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, {
          encryptionKey: key,
          role: destinationRole,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        Roles: [stack.resolve(destinationRole.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([{
            Action: [
              'kms:Decrypt',
              'kms:Encrypt',
              'kms:ReEncrypt*',
              'kms:GenerateDataKey*',
            ],
            Effect: 'Allow',
            Resource: stack.resolve(key.keyArn),
          }]),
        },
      });
    });
  });

  describe('s3 backup configuration', () => {
    it('set backupMode to ALL creates resources', () => {
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        s3Backup: {
          mode: firehose.BackupMode.ALL,
        },
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          S3BackupConfiguration: {
            CloudWatchLoggingOptions: {
              Enabled: true,
            },
            RoleARN: stack.resolve(destinationRole.roleArn),
          },
          S3BackupMode: 'Enabled',
        },
      });
    });

    it('sets backup configuration if backup bucket provided', () => {
      const backupBucket = new s3.Bucket(stack, 'MyBackupBucket');
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        s3Backup: {
          bucket: backupBucket,
        },
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          S3BackupConfiguration: {
            BucketARN: stack.resolve(backupBucket.bucketArn),
            CloudWatchLoggingOptions: {
              Enabled: true,
            },
            RoleARN: stack.resolve(destinationRole.roleArn),
          },
          S3BackupMode: 'Enabled',
        },
      });
    });

    it('throws error if backupMode set to FAILED', () => {
      expect(() => new firehose.S3Bucket(bucket, {
        role: destinationRole,
        s3Backup: {
          mode: firehose.BackupMode.FAILED,
        },
      })).toThrow('S3 destinations do not support BackupMode.FAILED');
    });

    it('by default does not create resources', () => {
      const destination = new firehose.S3Bucket(bucket);
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).resourceCountIs('AWS::S3::Bucket', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          S3BackupConfiguration: Match.absent(),
        },
      });
    });

    it('sets full backup configuration', () => {
      const backupBucket = new s3.Bucket(stack, 'MyBackupBucket');
      const key = new kms.Key(stack, 'Key');
      const logGroup = new logs.LogGroup(stack, 'BackupLogGroup');
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        s3Backup: {
          mode: firehose.BackupMode.ALL,
          bucket: backupBucket,
          dataOutputPrefix: 'myBackupPrefix',
          errorOutputPrefix: 'myBackupErrorPrefix',
          bufferingSize: cdk.Size.mebibytes(1),
          bufferingInterval: cdk.Duration.minutes(1),
          compression: firehose.Compression.ZIP,
          encryptionKey: key,
          loggingConfig: new firehose.EnableLogging(logGroup),
        },
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          S3BackupConfiguration: {
            BucketARN: stack.resolve(backupBucket.bucketArn),
            CloudWatchLoggingOptions: {
              Enabled: true,
              LogGroupName: stack.resolve(logGroup.logGroupName),
            },
            RoleARN: stack.resolve(destinationRole.roleArn),
            EncryptionConfiguration: {
              KMSEncryptionConfig: {
                AWSKMSKeyARN: stack.resolve(key.keyArn),
              },
            },
            Prefix: 'myBackupPrefix',
            ErrorOutputPrefix: 'myBackupErrorPrefix',
            BufferingHints: {},
            CompressionFormat: 'ZIP',
          },
          S3BackupMode: 'Enabled',
        },
      });
    });
  });

  describe('output prefix', () => {
    it.each(['!{', '!{blah}', '!{blah:blah}'])('throws when dataOutputPrefix contains invalid expression %s', (prefix) => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          dataOutputPrefix: prefix,
        });
      }).toThrow('The expression must be of the form !{namespace:value} and include a valid namespace at dataOutputPrefix.');
    });

    it.each(['!{', '!{firehose}', '!{blah:blah}'])('throws when errorOutputPrefix contains invalid expression %s', (prefix) => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          errorOutputPrefix: prefix,
        });
      }).toThrow('The expression must be of the form !{namespace:value} and include a valid namespace at errorOutputPrefix.');
    });

    it('throws when errorOutputPrefix is empty with dataOutputPrefix expressions', () => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          dataOutputPrefix: '!{timestamp:YYYY}/',
        });
      }).toThrow('Specify the errorOutputPrefix in order to use expressions in the dataOutputPrefix.');
    });

    it('throws when errorOutputOrefix does not contain !{firehose:error-output-type}', () => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          errorOutputPrefix: '!{timestamp:YYYY}/',
        });
      }).toThrow('The errorOutputPrefix expression must include at least one instance of !{firehose:error-output-type}.');
    });

    it('throws when dataOutputPrefix contains !{firehose:error-output-type}', () => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          dataOutputPrefix: '!{firehose:error-output-type}/',
          errorOutputPrefix: 'ERROR/',
        });
      }).toThrow('The dataOutputPrefix cannot contain !{firehose:error-output-type}.');
    });

    it.each(['partitionKeyFromQuery', 'partitionKeyFromLambda'])('throws when errorOutputPrefix contains %s', (ns) => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          errorOutputPrefix: `!{firehose:error-output-type}/!{${ns}:foo}/`,
        });
      }).toThrow('You cannot use partitionKeyFromLambda and partitionKeyFromQuery namespaces in errorOutputPreix.');
    });

    it.each(['partitionKeyFromQuery', 'partitionKeyFromLambda'])('throws when dataOutputPrefix contains %s without dynamic partitioning', (ns) => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          dataOutputPrefix: `!{${ns}:foo}/`,
          errorOutputPrefix: '!{firehose:error-output-type}/',
        });
      }).toThrow('When dynamic partitioning is not enabled, the dataOutputPrefix cannot contain neither partitionKeyFromLambda nor partitionKeyFromQuery.');
    });
  });

  describe('file extension', () => {
    it('sets fileExtension', () => {
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        fileExtension: '.json',
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          FileExtension: '.json',
        },
      });
    });

    it('sets fileExtension from a token', () => {
      const fileExtension = new cdk.CfnParameter(stack, 'FileExtension');
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        fileExtension: fileExtension.valueAsString,
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          FileExtension: { Ref: 'FileExtension' },
        },
      });
    });

    it('throws when fileExtension does not start with a period', () => {
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        fileExtension: 'json',
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow("fileExtension must start with '.'");
    });

    it('throws when fileExtension contains unallowed characters', () => {
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        fileExtension: '.json?',
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow("fileExtension can contain allowed characters: 0-9a-z!-_.*'()");
    });
  });

  describe('customTimeZone', () => {
    test.each([
      cdk.TimeZone.ASIA_TOKYO,
      cdk.TimeZone.AMERICA_LOS_ANGELES,
      cdk.TimeZone.of('UTC'),
    ])('sets customTimeZone: %s', (timezone: cdk.TimeZone) => {
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        timeZone: timezone,
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CustomTimeZone: timezone.timezoneName,
        },
      });
    });

    it('sets customTimeZone from a token', () => {
      const timeZone = new cdk.CfnParameter(stack, 'TimeZone');
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        timeZone: cdk.TimeZone.of(timeZone.valueAsString),
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          CustomTimeZone: { Ref: 'TimeZone' },
        },
      });
    });

    test.each([
      cdk.TimeZone.EST,
      cdk.TimeZone.ETC_UTC,
      cdk.TimeZone.ETC_GMT,
      cdk.TimeZone.EST5EDT,
      cdk.TimeZone.ETC_GMT_MINUS_1,
      cdk.TimeZone.FACTORY,
      cdk.TimeZone.AMERICA_PORT_MINUS_AU_MINUS_PRINCE,
      cdk.TimeZone.of(''),
    ])('throws when customTimeZone is not a standard IANA timezone: %s', (timezone: cdk.TimeZone) => {
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        timeZone: timezone,
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow(`Invalid timezone format '${timezone.timezoneName}'. Use standard IANA timezone identifiers ` +
      '(e.g., \'America/New_York\', \'Europe/London\'). ' +
      'See https://docs.aws.amazon.com/firehose/latest/dev/s3-object-name.html for more details');
    });
  });

  describe('data format conversion', () => {
    let database: glue.CfnDatabase;
    let table: glue.CfnTable;

    beforeEach(() => {
      database = new glue.CfnDatabase(stack, 'Database', {
        databaseInput: {
          description: 'A database',
        },
        catalogId: stack.account,
      });

      // Create a dummy table to use in the Delivery stream
      table = new glue.CfnTable(stack, 'Table', {
        databaseName: database.ref,
        catalogId: database.catalogId,
        tableInput: {
          description: 'A table',
        },
      });
    });

    it('sets data format conversion', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.S3Bucket(bucket, {
          dataFormatConversion: {
            schemaConfiguration: firehose.SchemaConfiguration.fromCfnTable(table),
            inputFormat: firehose.InputFormat.OPENX_JSON,
            outputFormat: firehose.OutputFormat.PARQUET,
          },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          DataFormatConversionConfiguration: {
            Enabled: true,
            SchemaConfiguration: {},
            InputFormatConfiguration: {},
            OutputFormatConfiguration: {},
          },
        },
      });
    });

    it('throws when compression is set on the destination', () => {
      expect(() => {
        new firehose.DeliveryStream(stack, 'Delivery Stream', {
          destination: new firehose.S3Bucket(bucket, {
            compression: firehose.Compression.SNAPPY,
            dataFormatConversion: {
              schemaConfiguration: firehose.SchemaConfiguration.fromCfnTable(table),
              inputFormat: firehose.InputFormat.OPENX_JSON,
              outputFormat: firehose.OutputFormat.ORC,
            },
          }),
        });
      }).toThrow(cdk.UnscopedValidationError);
    });

    it('throws when buffer size is less than the minimum', () => {
      expect(() => {
        new firehose.DeliveryStream(stack, 'Delivery Stream', {
          destination: new firehose.S3Bucket(bucket, {
            bufferingSize: cdk.Size.mebibytes(63),
            dataFormatConversion: {
              schemaConfiguration: firehose.SchemaConfiguration.fromCfnTable(table),
              inputFormat: firehose.InputFormat.OPENX_JSON,
              outputFormat: firehose.OutputFormat.ORC,
            },
          }),
        });
      }).toThrow(cdk.ValidationError);
    });

    it('default buffer size is set to 128 MiB', () => {
      new firehose.DeliveryStream(stack, 'Delivery Stream', {
        destination: new firehose.S3Bucket(bucket, {
          bufferingInterval: cdk.Duration.seconds(5),
          dataFormatConversion: {
            schemaConfiguration: firehose.SchemaConfiguration.fromCfnTable(table),
            inputFormat: firehose.InputFormat.OPENX_JSON,
            outputFormat: firehose.OutputFormat.ORC,
          },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          BufferingHints: {
            IntervalInSeconds: 5,
            SizeInMBs: 128,
          },
          DataFormatConversionConfiguration: {
            Enabled: true,
            SchemaConfiguration: {},
            InputFormatConfiguration: {},
            OutputFormatConfiguration: {},
          },
        },
      });
    });
  });

  describe('dynamic partitioning', () => {
    it('configures dynamic partitioning with inline parsing (JQ 1.6)', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          firehose.MetadataExtractionProcessor.jq16({
            customer_id: '.customer_id',
            device: '.type.device',
            year: '.event_timestamp|strftime("%Y")',
          }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:year}/!{partitionKeyFromQuery:device}/!{partitionKeyFromQuery:customer_id}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          DynamicPartitioningConfiguration: {
            Enabled: true,
          },
          BufferingHints: {
            SizeInMBs: 128,
            IntervalInSeconds: 300,
          },
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'MetadataExtraction',
              Parameters: [
                { ParameterName: 'MetadataExtractionQuery', ParameterValue: '{"customer_id":.customer_id,"device":.type.device,"year":.event_timestamp|strftime("%Y")}' },
                { ParameterName: 'JsonParsingEngine', ParameterValue: 'JQ-1.6' },
              ],
            }],
          },
          Prefix: '!{partitionKeyFromQuery:year}/!{partitionKeyFromQuery:device}/!{partitionKeyFromQuery:customer_id}/',
          ErrorOutputPrefix: '!{firehose:error-output-type}/',
        },
      });
    });

    it('configures dynamic partitioning with lambda', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const destination = new firehose.S3Bucket(bucket, {
        role: destinationRole,
        dynamicPartitioning: { enabled: true },
        processors: [
          new firehose.LambdaFunctionProcessor(lambdaFunction),
        ],
        dataOutputPrefix: '!{partitionKeyFromLambda:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          DynamicPartitioningConfiguration: {
            Enabled: true,
          },
          BufferingHints: {
            SizeInMBs: 128,
            IntervalInSeconds: 300,
          },
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'Lambda',
              Parameters: [
                { ParameterName: 'RoleArn', ParameterValue: stack.resolve(destinationRole.roleArn) },
                { ParameterName: 'LambdaArn', ParameterValue: stack.resolve(lambdaFunction.functionArn) },
              ],
            }],
          },
          Prefix: '!{partitionKeyFromLambda:foo}/',
          ErrorOutputPrefix: '!{firehose:error-output-type}/',
        },
      });
    });

    it('throws MetadataExtractionProcessor without dynamic partitioning', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: false },
        processors: [firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' })],
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('MetadataExtractionProcessor can only be present when Dynamic Partitioning is enabled.');
    });

    it('throws RecordDeAggregationProcessor without dynamic partitioning', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: false },
        processors: [firehose.RecordDeAggregationProcessor.json()],
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('RecordDeAggregationProcessor can only be present when Dynamic Partitioning is enabled.');
    });

    it('configures dynamic partitioning with retry options', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true, retryDuration: cdk.Duration.minutes(5) },
        processors: [firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' })],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          DynamicPartitioningConfiguration: {
            Enabled: true,
            RetryOptions: {
              DurationInSeconds: 300,
            },
          },
          BufferingHints: {
            SizeInMBs: 128,
            IntervalInSeconds: 300,
          },
        },
      });
    });

    it('validates retryDuration', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true, retryDuration: cdk.Duration.minutes(121) },
        processors: [firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' })],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('Retry duration must be less than 7200 seconds, got 7260 seconds.');
    });

    it('validates bufferingSize', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        bufferingSize: cdk.Size.mebibytes(63),
        processors: [firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' })],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('When data format conversion or dynamic partitioning is enabled, buffering size must be at least 64 MiBs, got 63 MiBs.');
    });

    it('validates bufferingInterval', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        bufferingInterval: cdk.Duration.seconds(50),
        processors: [firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' })],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('When dynamic partitioning is enabled, buffering interval must be at least 60 seconds, got 50 seconds.');
    });

    it('configures multi record deaggregation with JSON', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          firehose.RecordDeAggregationProcessor.json(),
          firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          DynamicPartitioningConfiguration: {
            Enabled: true,
          },
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'RecordDeAggregation',
              Parameters: [
                { ParameterName: 'SubRecordType', ParameterValue: 'JSON' },
              ],
            }, {
              Type: 'MetadataExtraction',
            }],
          },
        },
      });
    });

    it('configures multi record deaggregation with custom delimiter', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          firehose.RecordDeAggregationProcessor.delimited('####'),
          firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: destination,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        ExtendedS3DestinationConfiguration: {
          DynamicPartitioningConfiguration: {
            Enabled: true,
          },
          ProcessingConfiguration: {
            Enabled: true,
            Processors: [{
              Type: 'RecordDeAggregation',
              Parameters: [
                { ParameterName: 'SubRecordType', ParameterValue: 'DELIMITED' },
                { ParameterName: 'Delimiter', ParameterValue: 'IyMjIw==' },
              ],
            }, {
              Type: 'MetadataExtraction',
            }],
          },
        },
      });
    });

    it('throws when multi record deaggregation delimiter is not specified', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          new firehose.RecordDeAggregationProcessor({ subRecordType: firehose.SubRecordType.DELIMITED }),
          firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('delimiter must be specified when subRecordType is DELIMITED.');
    });

    it('throws when the JQ query is empty', () => {
      expect(() => {
        new firehose.S3Bucket(bucket, {
          dynamicPartitioning: { enabled: true },
          processors: [
            firehose.MetadataExtractionProcessor.jq16({}),
          ],
          dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
          errorOutputPrefix: '!{firehose:error-output-type}/',
        });
      }).toThrow('The query for MetadataExtractionProcessor should not be empty.');
    });

    it('throws when the JQ query has more than 50 keys', () => {
      expect(() => {
        const query = Object.fromEntries(Array.from({ length: 51 }, (_, i) => [i, `${i}`]));
        new firehose.S3Bucket(bucket, {
          dynamicPartitioning: { enabled: true },
          processors: [
            firehose.MetadataExtractionProcessor.jq16(query),
          ],
          dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
          errorOutputPrefix: '!{firehose:error-output-type}/',
        });
      }).toThrow('The query for MetadataExtractionProcessor cannot exceed the limit of 50 keys.');
    });

    it('throws when dataOutputPrefix does not contain all partition keys', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          firehose.MetadataExtractionProcessor.jq16({ foo: '.foo', bar: '.bar' }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('When dynamic partitioning via inline parsing is enabled, you must use all specified dynamic partitioning key values for partitioning your data source.');
    });

    it('throws when dataOutputPrefix contains extra partition keys', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:bar}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('When dynamic partitioning via inline parsing is enabled, you must use all specified dynamic partitioning key values for partitioning your data source.');
    });

    it('throws when dataOutputPrefix does not contain partitionKeyFromLambda with only lambda processor', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          new firehose.LambdaFunctionProcessor(lambdaFunction),
        ],
        dataOutputPrefix: 'YYYY/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('When dynamic partitioning is enabled and the only LambdaFunctionProcessor is specified, you must specify at least one instance of !{partitionKeyFromLambda:keyID}.');
    });

    it('throws when dataOutputPrefix contains partitionKeyFromLambda without lambda processor', () => {
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          firehose.MetadataExtractionProcessor.jq16({ foo: '.foo' }),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/!{partitionKeyFromLambda:bar}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('The dataOutputPrefix cannot contain !{partitionKeyFromLambda:keyID} when LambdaFunctionProcessor is not specified.');
    });

    it('throws when dataOutputPrefix contains partitionKeyFromQuery without inline parsing', () => {
      const lambdaFunction = new lambda.Function(stack, 'DataProcessorFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        code: lambda.Code.fromInline('foo'),
        handler: 'bar',
      });
      const destination = new firehose.S3Bucket(bucket, {
        dynamicPartitioning: { enabled: true },
        processors: [
          new firehose.LambdaFunctionProcessor(lambdaFunction),
        ],
        dataOutputPrefix: '!{partitionKeyFromQuery:foo}/!{partitionKeyFromLambda:bar}/',
        errorOutputPrefix: '!{firehose:error-output-type}/',
      });
      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: destination,
        });
      }).toThrow('The dataOutputPrefix cannot contain !{partitionKeyFromQuery:keyID} when MetadataExtractionProcessor is not specified.');
    });
  });
});
