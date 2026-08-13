import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import * as iam from 'aws-cdk-lib/aws-iam';
import { LogGroup } from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from '../lib';

describe('Job', () => {
  let stack: cdk.Stack;
  let role: iam.IRole;
  let script: glue.Code;
  let codeBucket: s3.IBucket;
  let job: glue.IJob;

  beforeEach(() => {
    stack = new cdk.Stack();
    cdk.Validations.of(stack).acknowledge({
      id: 'CloudFormation-Validate::E1155',
      reason: 'Syntactically incorrect log group name',
    });
    role = iam.Role.fromRoleArn(stack, 'Role', 'arn:aws:iam::123456789012:role/TestRole');
    codeBucket = s3.Bucket.fromBucketName(stack, 'CodeBucket', 'bucketname');
    script = glue.Code.fromBucket(codeBucket, 'script');
  });

  describe('Create new PySpark ETL Flex Job with default parameters', () => {
    beforeEach(() => {
      job = new glue.PySparkFlexEtlJob(stack, 'ImportedJob', { role, script });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 5.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '5.0',
      });
    });

    test('Default WorkerType should be G.1X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'G.1X',
      });
    });

    test('ExecutionClass should be Flex', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        ExecutionClass: 'FLEX',
      });
    });

    test('Has Continuous Logging Enabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--job-language': 'python',
          '--enable-continuous-cloudwatch-log': 'true',
        }),
      });
    });

    test('Default numberOfWorkers should be 10', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 10,
      });
    });

    test('Job run queuing must be disabled for flex jobs', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        JobRunQueuingEnabled: false,
      });
    });
  });

  describe('Create new PySpark ETL Job with log override parameters', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        jobName: 'PySparkETLJob',
        role,
        script,
        continuousLogging: {
          enabled: true,
          quiet: true,
          logGroup: new LogGroup(stack, 'logGroup', {
            logGroupName: '/aws-glue/jobs/${job.jobName}',
          }),
          logStreamPrefix: 'logStreamPrefix',
          conversionPattern: 'convert',
        },
      });
    });

    test('Has Continuous Logging enabled with optional args', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--job-language': 'python',
          '--continuous-log-logGroup': Match.objectLike({
            Ref: Match.anyValue(),
          }),
          '--enable-continuous-cloudwatch-log': 'true',
          '--enable-continuous-log-filter': 'true',
          '--continuous-log-logStreamPrefix': 'logStreamPrefix',
          '--continuous-log-conversionPattern': 'convert',
        }),
      });
    });
  });

  describe('Create new PySpark ETL Flex Job with logging explicitly disabled', () => {
    beforeEach(() => {
      job = new glue.PySparkFlexEtlJob(stack, 'PySparkFlexETLJob', {
        jobName: 'PySparkFlexETLJob',
        role,
        script,
        continuousLogging: {
          enabled: false,
        },
      });
    });

    test('Has Continuous Logging Disabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: {
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--job-language': 'python',
        },
      });
    });
  });

  describe('Create pySpark ETL Job with optional properties', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'pySparkEtlJob', {
        jobName: 'pySparkEtlJob',
        description: 'This is a description',
        role,
        script,
        glueVersion: glue.GlueVersion.V3_0,
        continuousLogging: { enabled: false },
        workerType: glue.WorkerType.G_2X,
        maxConcurrentRuns: 100,
        timeout: cdk.Duration.hours(2),
        connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
        securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
        tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
        numberOfWorkers: 2,
        maxRetries: 2,
      });
    });

    test('Test job attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Custom Job Name and Description', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Name: 'pySparkEtlJob',
        Description: 'This is a description',
      });
    });

    test('Overridden Glue Version should be 3.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '3.0',
      });
    });

    test('Verify Default Arguments', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--job-language': 'python',
        }),
      });
    });

    test('Overridden numberOfWorkers should be 2', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 2,
      });
    });

    test('Overridden WorkerType should be G.2X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: glue.WorkerType.G_2X,
      });
    });

    test('Overridden max retries should be 2', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        MaxRetries: 2,
      });
    });

    test('Overridden max concurrent runs should be 100', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        ExecutionProperty: {
          MaxConcurrentRuns: 100,
        },
      });
    });

    test('Overridden timeout should be 2 hours', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Timeout: 120,
      });
    });

    test('Overridden connections should be 100', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Connections: {
          Connections: ['connectionName'],
        },
      });
    });

    test('Overridden security configuration should be set', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        SecurityConfiguration: 'securityConfigName',
      });
    });

    test('Should have tags', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
      });
    });
  });
});
