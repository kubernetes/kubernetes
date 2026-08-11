import * as cdk from '../..';
import { Match, Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as logs from '../../aws-logs';
import * as s3 from '../../aws-s3';
import * as secretsmanager from '../../aws-secretsmanager';
import * as firehose from '../lib';

describe('HTTP destination', () => {
  let stack: cdk.Stack;

  const baseEndpointConfig = { url: 'https://test-endpoint.example.com' };

  beforeEach(() => {
    stack = new cdk.Stack();
  });

  it('provides defaults when only required configuration is given', () => {
    new firehose.DeliveryStream(stack, 'DeliveryStream', {
      destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
      HttpEndpointDestinationConfiguration: {
        EndpointConfiguration: { Url: baseEndpointConfig.url },
        RequestConfiguration: { ContentEncoding: 'NONE', CommonAttributes: Match.absent() },
        S3BackupMode: 'FailedDataOnly',
        S3Configuration: { BucketARN: Match.anyValue() },
        BufferingHints: Match.absent(),
        RetryOptions: Match.absent(),
        SecretsManagerConfiguration: Match.absent(),
        CloudWatchLoggingOptions: { Enabled: true },
      },
    });
  });

  describe('role', () => {
    it('creates a role when none is provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [{
            Effect: 'Allow',
            Principal: { Service: 'firehose.amazonaws.com' },
            Action: 'sts:AssumeRole',
          }],
        },
      });
    });

    it('scopes the role trust policy to Firehose in this account (confused-deputy protection)', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [Match.objectLike({
            Principal: { Service: 'firehose.amazonaws.com' },
            Action: 'sts:AssumeRole',
            Condition: {
              StringEquals: { 'aws:SourceAccount': { Ref: 'AWS::AccountId' } },
              ArnLike: { 'aws:SourceArn': Match.anyValue() },
            },
          })],
        },
      });
    });

    it('uses the provided role', () => {
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
      });

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig, role }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RoleARN: stack.resolve(role.roleArn),
        },
      });
    });
  });

  describe('logging', () => {
    it('creates a log group and enables logging by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          CloudWatchLoggingOptions: { Enabled: true },
        },
      });
    });

    it('disables logging when configured', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          loggingConfig: new firehose.DisableLogging(),
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          CloudWatchLoggingOptions: Match.absent(),
        },
      });
    });

    it('uses a provided log group', () => {
      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          loggingConfig: new firehose.EnableLogging(logGroup),
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          CloudWatchLoggingOptions: {
            Enabled: true,
            LogGroupName: stack.resolve(logGroup.logGroupName),
          },
        },
      });
    });
  });

  describe('backup', () => {
    it('auto-creates a backup bucket with FailedDataOnly mode by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).resourceCountIs('AWS::S3::Bucket', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          S3BackupMode: 'FailedDataOnly',
          S3Configuration: { BucketARN: Match.anyValue() },
        },
      });
    });

    it('sets S3BackupMode to AllData when backupMode is ALL', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          backupMode: firehose.HttpBackupMode.ALL,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          S3BackupMode: 'AllData',
        },
      });
    });

    it('uses the provided backup bucket from s3Backup', () => {
      const backupBucket = new s3.Bucket(stack, 'ExistingBucket');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          s3Backup: { bucket: backupBucket },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          S3Configuration: { BucketARN: stack.resolve(backupBucket.bucketArn) },
        },
      });
      // Only the user's bucket — no extra auto-created bucket
      Template.fromStack(stack).resourceCountIs('AWS::S3::Bucket', 1);
    });

    it('grants the role read/write access to the backup bucket', () => {
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
      });
      const backupBucket = new s3.Bucket(stack, 'BackupBucket');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          role,
          s3Backup: { bucket: backupBucket },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        Roles: [stack.resolve(role.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([{
            Effect: 'Allow',
            Action: Match.arrayWith(['s3:GetObject*', 's3:PutObject']),
            Resource: Match.arrayWith([stack.resolve(backupBucket.bucketArn)]),
          }]),
        },
      });
    });
  });

  describe('endpoint configuration', () => {
    it('renders the endpoint URL', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: { url: 'https://my.endpoint.example.com' },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { Url: 'https://my.endpoint.example.com' },
        },
      });
    });

    it('includes name when provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: { ...baseEndpointConfig, name: 'My Endpoint' },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { Name: 'My Endpoint' },
        },
      });
    });

    it('omits name when not provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { Name: Match.absent() },
        },
      });
    });

    it('includes access key when provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: {
            ...baseEndpointConfig,
            accessKey: cdk.SecretValue.unsafePlainText('my-access-key'),
          },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { AccessKey: 'my-access-key' },
        },
      });
    });

    it('omits access key when not provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { AccessKey: Match.absent() },
        },
      });
    });

    it('configures secrets manager when a secret is provided', () => {
      const secret = new secretsmanager.Secret(stack, 'Secret');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: { ...baseEndpointConfig, secret },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          SecretsManagerConfiguration: {
            SecretARN: stack.resolve(secret.secretArn),
            Enabled: true,
          },
        },
      });
    });

    it('grants the role read access to the secret', () => {
      const secret = new secretsmanager.Secret(stack, 'Secret');
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
      });

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: { ...baseEndpointConfig, secret },
          role,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        Roles: [stack.resolve(role.roleName)],
        PolicyDocument: {
          Statement: Match.arrayWith([{
            Effect: 'Allow',
            Action: Match.arrayWith(['secretsmanager:GetSecretValue', 'secretsmanager:DescribeSecret']),
            Resource: stack.resolve(secret.secretArn),
          }]),
        },
      });
    });

    it('omits secrets manager configuration when no secret is provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          SecretsManagerConfiguration: Match.absent(),
        },
      });
    });
  });

  describe('request configuration', () => {
    it('defaults content encoding to NONE', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: { ContentEncoding: 'NONE' },
        },
      });
    });

    it('sets content encoding to GZIP when requested', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          requestCompression: firehose.HttpCompression.GZIP,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: { ContentEncoding: 'GZIP' },
        },
      });
    });

    it('renders common attributes when provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          attributes: [
            { name: 'env', value: 'prod' },
            { name: 'region', value: 'us-east-1' },
          ],
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: {
            CommonAttributes: [
              { AttributeName: 'env', AttributeValue: 'prod' },
              { AttributeName: 'region', AttributeValue: 'us-east-1' },
            ],
          },
        },
      });
    });

    it('omits common attributes when not provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: { CommonAttributes: Match.absent() },
        },
      });
    });
  });

  describe('buffering hints', () => {
    it('renders buffering hints when provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          bufferingHints: {
            interval: cdk.Duration.seconds(120),
            size: cdk.Size.mebibytes(8),
          },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          BufferingHints: { IntervalInSeconds: 120, SizeInMBs: 8 },
        },
      });
    });

    it('omits buffering hints when not provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: { BufferingHints: Match.absent() },
      });
    });

    it('fails when interval exceeds 900 seconds', () => {
      const destination = new firehose.HttpEndpoint({
        endpointConfig: baseEndpointConfig,
        bufferingHints: { interval: cdk.Duration.seconds(901) },
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });
      }).toThrow(/must be less than 900 seconds/);
    });

    it('fails when size exceeds 128 MiB', () => {
      const destination = new firehose.HttpEndpoint({
        endpointConfig: baseEndpointConfig,
        bufferingHints: { size: cdk.Size.mebibytes(129) },
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });
      }).toThrow(/must be at most 128 MiBs/);
    });

    it('fails when size is less than 1 MiB', () => {
      const destination = new firehose.HttpEndpoint({
        endpointConfig: baseEndpointConfig,
        bufferingHints: { size: cdk.Size.mebibytes(0) },
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });
      }).toThrow(/must be at least 1 MiB/);
    });

    it('skips interval validation for tokenized values', () => {
      const param = new cdk.CfnParameter(stack, 'IntervalParam', { type: 'Number' });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: new firehose.HttpEndpoint({
            endpointConfig: baseEndpointConfig,
            bufferingHints: { interval: cdk.Duration.seconds(param.valueAsNumber) },
          }),
        });
      }).not.toThrow();
    });

    it('skips size validation for tokenized values', () => {
      const param = new cdk.CfnParameter(stack, 'SizeParam', { type: 'Number' });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: new firehose.HttpEndpoint({
            endpointConfig: baseEndpointConfig,
            bufferingHints: { size: cdk.Size.mebibytes(param.valueAsNumber) },
          }),
        });
      }).not.toThrow();
    });
  });

  describe('retry options', () => {
    it('renders retry options when provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({
          endpointConfig: baseEndpointConfig,
          retryOptions: { duration: cdk.Duration.seconds(300) },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RetryOptions: { DurationInSeconds: 300 },
        },
      });
    });

    it('omits retry options when not provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.HttpEndpoint({ endpointConfig: baseEndpointConfig }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: { RetryOptions: Match.absent() },
      });
    });

    it('fails when retry duration exceeds 7200 seconds', () => {
      const destination = new firehose.HttpEndpoint({
        endpointConfig: baseEndpointConfig,
        retryOptions: { duration: cdk.Duration.seconds(7201) },
      });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', { destination });
      }).toThrow(/must be at most 7200 seconds/);
    });

    it('skips retry validation for tokenized values', () => {
      const param = new cdk.CfnParameter(stack, 'RetryParam', { type: 'Number' });

      expect(() => {
        new firehose.DeliveryStream(stack, 'DeliveryStream', {
          destination: new firehose.HttpEndpoint({
            endpointConfig: baseEndpointConfig,
            retryOptions: { duration: cdk.Duration.seconds(param.valueAsNumber) },
          }),
        });
      }).not.toThrow();
    });
  });
});
