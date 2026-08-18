import * as cdk from 'aws-cdk-lib';
import { Template, Match, Annotations } from 'aws-cdk-lib/assertions';
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
  let sparkUIBucket: s3.Bucket;

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

  describe('Create new PySpark ETL Job with default parameters', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
      });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
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

    test('Default WorkerType should be G.1X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'G.1X',
      });
    });

    test('Default Python version should be 3', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        Command: {
          Name: glue.JobType.ETL,
          ScriptLocation: 's3://bucketname/script',
          PythonVersion: glue.PythonVersion.THREE,
        },
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

    test('Default job run queuing should be disabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        JobRunQueuingEnabled: false,
      });
    });
  });

  describe('Create new PySpark ETL Job with logging explicitly disabled', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        jobName: 'PySparkETLJob',
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

  describe('Create PySpark ETL Job with G2 worker type with 2 workers', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        workerConfiguration: { workerType: glue.WorkerType.G_2X, numberOfWorkers: 2 },
      });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
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

    test('Overridden numberOfWorkers should be 2', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 2,
      });
    });

    test('Overridden WorkerType should be G.1X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'G.2X',
      });
    });
  });

  describe('Create PySpark ETL Job with G4 worker type with 4 workers', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        workerConfiguration: { workerType: glue.WorkerType.G_4X, numberOfWorkers: 4 },
      });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
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

    test('Overridden numberOfWorkers should be 2', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 4,
      });
    });

    test('Overridden WorkerType should be G.4X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'G.4X',
      });
    });
  });

  describe('Create PySpark ETL Job with G8 worker type and 8 workers', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        workerConfiguration: { workerType: glue.WorkerType.G_8X, numberOfWorkers: 8 },
      });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: '4.0',
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

    test('Overridden numberOfWorkers should be 8', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 8,
      });
    });

    test('Overridden WorkerType should be G.8X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'G.8X',
      });
    });
  });

  describe('Override SparkUI properties for PySpark ETL Job', () => {
    beforeEach(() => {
      sparkUIBucket = new s3.Bucket(stack, 'SparkUIBucket', { bucketName: 'bucket-name' });
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        sparkUI: {
          bucket: sparkUIBucket,
          prefix: '/prefix',
        },
      });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: glue.GlueVersion.V4_0,
      });
    });

    test('Has Continuous Logging and SparkUIEnabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--job-language': 'python',
          '--enable-continuous-cloudwatch-log': 'true',
          '--enable-spark-ui': 'true',
          '--spark-event-logs-path': Match.objectLike({
            'Fn::Join': [
              '',
              [
                's3://',
                { Ref: Match.anyValue() },
                '/prefix/',
              ],
            ],
          }),
        }),
      });
    });
  });

  describe('Invalid overrides should cause errors', () => {
    test('Invalid SparkUI prefix should throw an error', () => {
      expect(() => {
        sparkUIBucket = new s3.Bucket(stack, 'sparkUIbucket', { bucketName: 'bucket-name' });
        job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
          role,
          script,
          jobName: 'PySparkETLJob',
          sparkUI: {
            bucket: sparkUIBucket,
            prefix: 'prefix',
          },
          workerConfiguration: { workerType: glue.WorkerType.G_8X, numberOfWorkers: 8 },
          continuousLogging: { enabled: false },
        });
      }).toThrow('Invalid prefix format (value: prefix)');
    });
  });

  describe('Create PySpark ETL Job with extraPythonFiles, extraFiles, extraJars and extraJarsFirst', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        extraPythonFiles: [
          glue.Code.fromBucket(
            s3.Bucket.fromBucketName(stack, 'extraPythonFilesBucket', 'extra-python-files-bucket'),
            'prefix/file.py'),
        ],
        extraFiles: [
          glue.Code.fromBucket(
            s3.Bucket.fromBucketName(stack, 'extraFilesBucket', 'extra-files-bucket'),
            'prefix/file.txt'),
        ],
        extraJars: [
          glue.Code.fromBucket(
            s3.Bucket.fromBucketName(stack, 'extraJarsBucket', 'extra-jars-bucket'),
            'prefix/file.jar'),
        ],
        extraJarsFirst: true,
      });
    });

    test('Test default attributes', () => {
      expect(job.jobArn).toEqual(stack.formatArn({
        service: 'glue',
        resource: 'job',
        resourceName: job.jobName,
      }));
      expect(job.grantPrincipal).toEqual(role);
    });

    test('Default Glue Version should be 4.0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        GlueVersion: glue.GlueVersion.V4_0,
      });
    });

    test('Verify Default Arguments', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
          '--job-language': 'python',
          '--enable-continuous-cloudwatch-log': 'true',
          '--extra-py-files': 's3://extra-python-files-bucket/prefix/file.py',
          '--extra-files': 's3://extra-files-bucket/prefix/file.txt',
          '--extra-jars': 's3://extra-jars-bucket/prefix/file.jar',
          '--user-jars-first': 'true',
        }),
      });
    });

    test('Default numberOfWorkers should be 10', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NumberOfWorkers: 10,
      });
    });

    test('Default WorkerType should be G.1X', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        WorkerType: 'G.1X',
      });
    });
  });

  describe('Create PySpark ETL Job with optional properties', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        jobName: 'PySparkETLJobCustomName',
        description: 'This is a description',
        role,
        script,
        glueVersion: glue.GlueVersion.V3_0,
        continuousLogging: { enabled: false },
        workerConfiguration: { workerType: glue.WorkerType.G_2X, numberOfWorkers: 2 },
        maxConcurrentRuns: 100,
        timeout: cdk.Duration.hours(2),
        connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
        securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
        tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
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
        Name: 'PySparkETLJobCustomName',
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

  describe('Create PySpark ETL Job with overridden job run queueing', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        jobName: 'PySparkETLJobCustomName',
        description: 'This is a description',
        role,
        script,
        glueVersion: glue.GlueVersion.V3_0,
        continuousLogging: { enabled: false },
        workerConfiguration: { workerType: glue.WorkerType.G_2X, numberOfWorkers: 2 },
        maxConcurrentRuns: 100,
        timeout: cdk.Duration.hours(2),
        connections: [glue.Connection.fromConnectionName(stack, 'Connection', 'connectionName')],
        securityConfiguration: glue.SecurityConfiguration.fromSecurityConfigurationName(stack, 'SecurityConfig', 'securityConfigName'),
        tags: {
          FirstTagName: 'FirstTagValue',
          SecondTagName: 'SecondTagValue',
          XTagName: 'XTagValue',
        },
        maxRetries: 2,
        jobRunQueuingEnabled: true,
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
        Name: 'PySparkETLJobCustomName',
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

    test('Overridden job run queuing should be enabled', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        JobRunQueuingEnabled: true,
      });
    });

    test('Default max retries with job run queuing enabled should be 0', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        MaxRetries: 0,
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

  describe('Create new PySpark ETL Job with metrics control', () => {
    test('Default behavior should include metrics (backward compatibility)', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
        }),
      });
    });

    test('Should exclude metrics when enableMetrics is false', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
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

    test('Should exclude observability metrics when enableObservabilityMetrics is false', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        enableObservabilityMetrics: false,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
        }),
      });

      // Verify that --enable-observability-metrics is NOT present
      const template = Template.fromStack(stack);
      const jobs = template.findResources('AWS::Glue::Job');
      const jobResource = Object.values(jobs)[0] as any;
      expect(jobResource.Properties.DefaultArguments).not.toHaveProperty('--enable-observability-metrics');
    });

    test('Should exclude both metrics when both are disabled', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
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

    test('Should include metrics when explicitly enabled', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        enableMetrics: true,
        enableObservabilityMetrics: true,
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        DefaultArguments: Match.objectLike({
          '--enable-metrics': '',
          '--enable-observability-metrics': 'true',
        }),
      });
    });
  });

  describe('maxRetries warning with job run queuing enabled', () => {
    const WARNING = Match.stringLikeRegexp('.*Overriding it to 0 with since job run queuing is enabled.*');

    test('warns when job run queuing is enabled and maxRetries is greater than 0', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        jobRunQueuingEnabled: true,
        maxRetries: 2,
      });

      Annotations.fromStack(stack).hasWarning('/Default/PySparkETLJob', WARNING);
    });

    test('does not warn when job run queuing is enabled and maxRetries is 0', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        jobRunQueuingEnabled: true,
        maxRetries: 0,
      });

      Annotations.fromStack(stack).hasNoWarning('/Default/PySparkETLJob', WARNING);
    });

    test('does not warn when job run queuing is enabled and maxRetries is not set', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        jobRunQueuingEnabled: true,
      });

      Annotations.fromStack(stack).hasNoWarning('/Default/PySparkETLJob', WARNING);
    });

    test('does not warn when maxRetries is greater than 0 but job run queuing is disabled', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        jobRunQueuingEnabled: false,
        maxRetries: 2,
      });

      Annotations.fromStack(stack).hasNoWarning('/Default/PySparkETLJob', WARNING);
    });

    test('does not warn when maxRetries is greater than 0 but job run queuing is not set', () => {
      new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        maxRetries: 2,
      });

      Annotations.fromStack(stack).hasNoWarning('/Default/PySparkETLJob', WARNING);
    });
  });

  describe('Create PySpark ETL Job with notifyDelayAfter', () => {
    beforeEach(() => {
      job = new glue.PySparkEtlJob(stack, 'PySparkETLJob', {
        role,
        script,
        jobName: 'PySparkETLJob',
        notifyDelayAfter: cdk.Duration.minutes(10),
      });
    });

    test('NotificationProperty is set', () => {
      Template.fromStack(stack).hasResourceProperties('AWS::Glue::Job', {
        NotificationProperty: {
          NotifyDelayAfter: 10,
        },
      });
    });
  });
});
