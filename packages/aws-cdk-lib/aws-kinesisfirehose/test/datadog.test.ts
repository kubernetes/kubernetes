import * as cdk from '../..';
import { Match, Template } from '../../assertions';
import * as iam from '../../aws-iam';
import * as s3 from '../../aws-s3';
import * as secretsmanager from '../../aws-secretsmanager';
import * as firehose from '../lib';

describe('Datadog destination', () => {
  let stack: cdk.Stack;
  let apiKey: secretsmanager.Secret;

  beforeEach(() => {
    stack = new cdk.Stack();
    apiKey = new secretsmanager.Secret(stack, 'ApiKey');
  });

  const minimalProps = () => ({
    apiKey,
    endpoint: firehose.DatadogEndpoint.LOGS_US1,
  });

  describe('defaults', () => {
    it('uses GZIP compression by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: { ContentEncoding: 'GZIP' },
        },
      });
    });

    it('uses 60-second / 4-MiB buffering by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          BufferingHints: { IntervalInSeconds: 60, SizeInMBs: 4 },
        },
      });
    });

    it('uses 60-second retry duration by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RetryOptions: { DurationInSeconds: 60 },
        },
      });
    });

    it('uses FailedDataOnly backup mode by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          S3BackupMode: 'FailedDataOnly',
        },
      });
    });

    it('authenticates with the API key secret via secrets manager', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          SecretsManagerConfiguration: {
            SecretARN: stack.resolve(apiKey.secretArn),
            Enabled: true,
          },
        },
      });
    });

    it('enables CloudWatch logging by default', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).resourceCountIs('AWS::Logs::LogGroup', 1);
      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          CloudWatchLoggingOptions: { Enabled: true },
        },
      });
    });

    it('produces empty CommonAttributes when no tags are provided', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog(minimalProps()),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: { CommonAttributes: [] },
        },
      });
    });
  });

  describe('DatadogEndpoint', () => {
    test.each([
      ['LOGS_US1', firehose.DatadogEndpoint.LOGS_US1, 'https://aws-kinesis-http-intake.logs.datadoghq.com/v1/input'],
      ['LOGS_US3', firehose.DatadogEndpoint.LOGS_US3, 'https://aws-kinesis-http-intake.logs.us3.datadoghq.com/api/v2/logs?dd-protocol=aws-kinesis-firehose'],
      ['LOGS_US5', firehose.DatadogEndpoint.LOGS_US5, 'https://aws-kinesis-http-intake.logs.us5.datadoghq.com/api/v2/logs?dd-protocol=aws-kinesis-firehose'],
      ['LOGS_AP1', firehose.DatadogEndpoint.LOGS_AP1, 'https://aws-kinesis-http-intake.logs.ap1.datadoghq.com/api/v2/logs?dd-protocol=aws-kinesis-firehose'],
      ['LOGS_EU', firehose.DatadogEndpoint.LOGS_EU, 'https://aws-kinesis-http-intake.logs.datadoghq.eu/v1/input'],
      ['LOGS_GOV', firehose.DatadogEndpoint.LOGS_GOV, 'https://aws-kinesis-http-intake.logs.ddog-gov.com/v1/input'],
      ['METRICS_US', firehose.DatadogEndpoint.METRICS_US, 'https://awsmetrics-intake.datadoghq.com/v1/input'],
      ['METRICS_US5', firehose.DatadogEndpoint.METRICS_US5, 'https://event-platform-intake.us5.datadoghq.com/api/v2/awsmetrics?dd-protocol=aws-kinesis-firehose'],
      ['METRICS_AP1', firehose.DatadogEndpoint.METRICS_AP1, 'https://event-platform-intake.ap1.datadoghq.com/api/v2/awsmetrics?dd-protocol=aws-kinesis-firehose'],
      ['METRICS_EU', firehose.DatadogEndpoint.METRICS_EU, 'https://awsmetrics-intake.datadoghq.eu/v1/input'],
      ['CONFIGURATION_US1', firehose.DatadogEndpoint.CONFIGURATION_US1, 'https://cloudplatform-intake.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose'],
      ['CONFIGURATION_US3', firehose.DatadogEndpoint.CONFIGURATION_US3, 'https://cloudplatform-intake.us3.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose'],
      ['CONFIGURATION_US5', firehose.DatadogEndpoint.CONFIGURATION_US5, 'https://cloudplatform-intake.us5.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose'],
      ['CONFIGURATION_AP1', firehose.DatadogEndpoint.CONFIGURATION_AP1, 'https://cloudplatform-intake.ap1.datadoghq.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose'],
      ['CONFIGURATION_EU', firehose.DatadogEndpoint.CONFIGURATION_EU, 'https://cloudplatform-intake.datadoghq.eu/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose'],
      ['CONFIGURATION_US_GOV', firehose.DatadogEndpoint.CONFIGURATION_US_GOV, 'https://cloudplatform-intake.ddog-gov.com/api/v2/cloudchanges?dd-protocol=aws-kinesis-firehose'],
    ])('%s maps to the correct URL', (_name, endpoint, expectedUrl) => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({ apiKey, endpoint }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { Url: expectedUrl },
        },
      });
    });

    it('accepts a custom URL via DatadogEndpoint.of()', () => {
      const customEndpoint = firehose.DatadogEndpoint.of('https://custom.datadog-proxy.example.com');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({ apiKey, endpoint: customEndpoint }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          EndpointConfiguration: { Url: 'https://custom.datadog-proxy.example.com' },
        },
      });
    });
  });

  describe('tags', () => {
    it('renders tags as CommonAttributes', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          tags: [
            { name: 'source', value: 'cloudfront' },
            { name: 'env', value: 'prod' },
          ],
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: {
            CommonAttributes: [
              { AttributeName: 'source', AttributeValue: 'cloudfront' },
              { AttributeName: 'env', AttributeValue: 'prod' },
            ],
          },
        },
      });
    });
  });

  describe('overrides', () => {
    it('overrides backup mode to ALL', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          backupMode: firehose.HttpBackupMode.ALL,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          S3BackupMode: 'AllData',
        },
      });
    });

    it('overrides buffering hints', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          bufferingHints: { interval: cdk.Duration.seconds(300), size: cdk.Size.mebibytes(16) },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          BufferingHints: { IntervalInSeconds: 300, SizeInMBs: 16 },
        },
      });
    });

    it('overrides retry duration', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          retryOptions: { duration: cdk.Duration.seconds(120) },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RetryOptions: { DurationInSeconds: 120 },
        },
      });
    });

    it('overrides request compression to NONE', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          requestCompression: firehose.HttpCompression.NONE,
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RequestConfiguration: { ContentEncoding: 'NONE' },
        },
      });
    });
  });

  describe('inherited CommonDestinationProps', () => {
    it('uses the provided IAM role', () => {
      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
      });

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({ ...minimalProps(), role }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          RoleARN: stack.resolve(role.roleArn),
        },
      });
    });

    it('uses a provided backup bucket from s3Backup', () => {
      const backupBucket = new s3.Bucket(stack, 'BackupBucket');

      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          s3Backup: { bucket: backupBucket },
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          S3Configuration: { BucketARN: stack.resolve(backupBucket.bucketArn) },
        },
      });
    });

    it('disables logging when configured', () => {
      new firehose.DeliveryStream(stack, 'DeliveryStream', {
        destination: new firehose.Datadog({
          ...minimalProps(),
          loggingConfig: new firehose.DisableLogging(),
        }),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
        HttpEndpointDestinationConfiguration: {
          CloudWatchLoggingOptions: Match.absent(),
        },
      });
    });
  });
});
