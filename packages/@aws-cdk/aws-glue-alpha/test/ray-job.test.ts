
import { testDeprecated } from '@aws-cdk/cdk-build-tools';
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

  describe('Create new Ray Job with default parameters', () => {
    beforeEach(() => {
      job = new glue.RayJob(stack, 'ImportedJob', { role, script });
    });

    testDeprecated('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    testDeprecated('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
      });
    });

    testDeprecated('Default number of workers should be 3', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 3,
      });
    });

    testDeprecated('Default worker type should be Z.2X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'Z.2X',
      });
    });

    testDeprecated('Has Continuous Logging Enabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--enable-continuous-cloudwatch-log': 'true',
        }),
      });
    });

    testDeprecated('Default job run queuing should be diabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        JobRunQueuingEnabled: false,
      });
    });
  });

  describe('Create new Ray Job with log override parameters', () => {
    beforeEach(() => {
      job = new glue.RayJob(stack, 'RayJob', {
        jobName: 'RayJob',
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

    testDeprecated('Has Continuous Logging enabled with optional args', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
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

  describe('Create new Ray Job with logging explicitly disabled', () => {
    beforeEach(() => {
      job = new glue.RayJob(stack, 'RayJob', {
        jobName: 'RayJob',
        role,
        script,
        continuousLogging: {
          enabled: false,
        },
      });
    });

    testDeprecated('Has Continuous Logging Disabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: {
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
        },
      });
    });
  });

  describe('Create Ray Job with optional override parameters', () => {
    beforeEach(() => {
      job = new glue.RayJob(stack, 'ImportedJob', {
        role,
        script,
        jobName: 'RayCustomJobName',
        description: 'This is a description',
        numberOfWorkers: 5,
        runtime: glue.Runtime.RAY_TWO_FOUR,
        maxRetries: 3,
        maxConcurrentRuns: 100,
        timeout: cdk.Duration.hours(2),
        connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
        securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
        tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
      });
    });

    testDeprecated('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    testDeprecated('Cannot override Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
      });
    });

    testDeprecated('Overridden number of workers should be 5', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 5,
      });
    });

    testDeprecated('Cannot override worker type should be Z.2X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'Z.2X',
      });
    });

    testDeprecated('Has Continuous Logging Enabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--enable-continuous-cloudwatch-log': 'true',
        }),
      });
    });

    testDeprecated('Custom Job Name and Description', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Name: 'RayCustomJobName',
        Description: 'This is a description',
      });
    });

    testDeprecated('Verify Default Arguemnts', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
        }),
      });
    });

    testDeprecated('Overridden max retries should be 3', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        MaxRetries: 3,
      });
    });

    testDeprecated('Overridden max concurrent runs should be 100', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        ExecutionProperty: {
          MaxConcurrentRuns: 100,
        },
      });
    });

    testDeprecated('Overridden timeout should be 2 hours', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Timeout: 120,
      });
    });

    testDeprecated('Overridden connections should be 100', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Connections: {
          Connections: ['connectionName'],
        },
      });
    });

    testDeprecated('Overridden security configuration should be set', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        SecurityConfiguration: 'securityConfigName',
      });
    });

    testDeprecated('Should have tags', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
      });
    });
  });

  describe('Create Ray Job with job run queuing enabled', () => {
    beforeEach(() => {
      job = new glue.RayJob(stack, 'ImportedJob', {
        role,
        script,
        jobName: 'RayCustomJobName',
        description: 'This is a description',
        numberOfWorkers: 5,
        runtime: glue.Runtime.RAY_TWO_FOUR,
        maxRetries: 3,
        maxConcurrentRuns: 100,
        timeout: cdk.Duration.hours(2),
        connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
        securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
        tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
        jobRunQueuingEnabled: true,
      });
    });

    testDeprecated('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    testDeprecated('Cannot override Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
      });
    });

    testDeprecated('Overridden number of workers should be 5', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 5,
      });
    });

    testDeprecated('Cannot override worker type should be Z.2X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'Z.2X',
      });
    });

    testDeprecated('Has Continuous Logging Enabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--enable-continuous-cloudwatch-log': 'true',
        }),
      });
    });

    testDeprecated('Custom Job Name and Description', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Name: 'RayCustomJobName',
        Description: 'This is a description',
      });
    });

    testDeprecated('Verify Default Arguemnts', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
        }),
      });
    });

    testDeprecated('Overridden job run queuing should be enabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        JobRunQueuingEnabled: true,
      });
    });

    testDeprecated('Default max retries with job run queuing enabled should be 0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        MaxRetries: 0,
      });
    });

    testDeprecated('Overridden max concurrent runs should be 100', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        ExecutionProperty: {
          MaxConcurrentRuns: 100,
        },
      });
    });

    testDeprecated('Overridden timeout should be 2 hours', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Timeout: 120,
      });
    });

    testDeprecated('Overridden connections should be 100', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Connections: {
          Connections: ['connectionName'],
        },
      });
    });

    testDeprecated('Overridden security configuration should be set', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        SecurityConfiguration: 'securityConfigName',
      });
    });

    testDeprecated('Should have tags', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
      });
    });
  });

  describe('Create new Ray Job with metrics control', () => {
    testDeprecated('Default behavior should include metrics (backward compatibility)', () => {
      new glue.RayJob(stack, 'RayJob', {
        role,
        script,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
        }),
      });
    });

    testDeprecated('Should exclude metrics when enableMetrics is false', () => {
      new glue.RayJob(stack, 'RayJob', {
        role,
        script,
        enableMetrics: false,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-observability-metrics': 'true',
        }),
      });

      // Verify that --enable-metrics is NOT present
      const template = Template.fromStack(stack);
      const jobs = template.findResources('AWS::Glue::Job');
      const jobResource = Object.values(jobs)[0] as any;
      expect(jobResource.Properties.DefaultArguments).not.toHaveProperty('--enable-metrics');
    });

    testDeprecated('Should exclude both metrics when both are disabled', () => {
      new glue.RayJob(stack, 'RayJob', {
        role,
        script,
        enableMetrics: false,
        enableObservabilityMetrics: false,
      });

      // Verify that neither metrics argument is present
      const template = Template.fromStack(stack);
      const jobs = template.findResources('AWS::Glue::Job');
      const jobResource = Object.values(jobs)[0] as any;
      expect(jobResource.Properties.DefaultArguments).not.toHaveProperty('--enable-metrics');
      expect(jobResource.Properties.DefaultArguments).not.toHaveProperty('--enable-observability-metrics');
    });
  });
});
