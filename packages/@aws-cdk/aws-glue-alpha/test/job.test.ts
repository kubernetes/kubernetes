import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import type * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from '../lib';

/**
 * Tests for the CloudWatch event/metric convenience API on `JobBase` and for
 * importing an existing job via `Job.fromJobAttributes`. A concrete job
 * (`PySparkEtlJob`) is used to exercise the abstract base's shared behavior.
 */
describe('JobBase events and metrics', () => {
  let stack: cdk.Stack;
  let role: iam.IRole;
  let script: glue.Code;
  let codeBucket: s3.IBucket;
  let job: glue.IJob;

  beforeEach(() => {
    stack = new cdk.Stack();
    role = iam.Role.fromRoleArn(stack, 'Role', 'arn:aws:iam::123456789012:role/TestRole');
    codeBucket = s3.Bucket.fromBucketName(stack, 'CodeBucket', 'bucketname');
    script = glue.Code.fromBucket(codeBucket, 'script');
    job = new glue.PySparkEtlJob(stack, 'Job', { role, script, jobName: 'MyJob' });
  });

  describe('onEvent', () => {
    test('creates a rule matching any Glue job state change, scoped to this job and with no state filter', () => {
      job.onEvent('AnyStateRule');

      const rules = Template.fromStack(stack).findResources('AWS::Events::Rule');
      expect(Object.keys(rules)).toHaveLength(1);

      const pattern = Object.values(rules)[0].Properties.EventPattern;
      expect(pattern.source).toEqual(['aws.glue']);
      expect(pattern['detail-type']).toEqual(['Glue Job State Change', 'Glue Job Run Status']);
      // Routed to this job (a single Ref to the job resource) and not filtered by state.
      expect(pattern.detail.jobName).toHaveLength(1);
      expect(pattern.detail.state).toBeUndefined();
    });
  });

  describe.each([
    ['onSuccess', 'SUCCEEDED'],
    ['onFailure', 'FAILED'],
    ['onTimeout', 'TIMEOUT'],
  ] as const)('%s', (method, state) => {
    test(`filters on the ${state} state and describes the rule`, () => {
      (job as any)[method]('StateRule');

      Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
        EventPattern: Match.objectLike({
          source: ['aws.glue'],
          detail: Match.objectLike({
            state: [state],
          }),
        }),
      });

      // onStateChange sets a human-readable description mentioning the state.
      // jobName is a token, so the value renders as an Fn::Join rather than a
      // plain string; assert only that a description is present.
      const rule = Object.values(Template.fromStack(stack).findResources('AWS::Events::Rule'))[0];
      expect(rule.Properties.Description).toBeDefined();
    });
  });

  describe('metric', () => {
    test('builds a Glue-namespaced metric scoped to the job across all runs', () => {
      const metric = job.metric('glue.driver.aggregate.elapsedTime', glue.MetricType.GAUGE);

      expect(metric.namespace).toEqual('Glue');
      expect(metric.metricName).toEqual('glue.driver.aggregate.elapsedTime');
      expect(metric.dimensions).toEqual({
        JobName: job.jobName,
        JobRunId: 'ALL',
        Type: 'gauge',
      });
    });

    test('uses the count type for count metrics', () => {
      const metric = job.metric('glue.driver.aggregate.numFailedTasks', glue.MetricType.COUNT);

      expect(metric.dimensions).toMatchObject({ Type: 'count' });
    });

    test('lets caller-supplied options override the defaults', () => {
      const metric = job.metric('glue.driver.aggregate.elapsedTime', glue.MetricType.GAUGE, {
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
      });

      expect(metric.statistic).toEqual('Average');
      expect(metric.period).toEqual(cdk.Duration.minutes(5));
    });
  });

  describe.each([
    ['metricSuccess', 'SUCCEEDED'],
    ['metricFailure', 'FAILED'],
    ['metricTimeout', 'TIMEOUT'],
  ] as const)('%s', (method, state) => {
    test('returns a TriggeredRules metric based on a job-state rule', () => {
      const metric = (job as any)[method]() as cloudwatch.Metric;

      expect(metric.namespace).toEqual('AWS/Events');
      expect(metric.metricName).toEqual('TriggeredRules');
      expect(metric.statistic).toEqual('Sum');

      Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
        EventPattern: Match.objectLike({
          detail: Match.objectLike({ state: [state] }),
        }),
      });
    });

    test('reuses a single event rule when called repeatedly', () => {
      (job as any)[method]();
      (job as any)[method]();

      Template.fromStack(stack).resourceCountIs('AWS::Events::Rule', 1);
    });
  });
});

describe('Job.fromJobAttributes', () => {
  let stack: cdk.Stack;

  beforeEach(() => {
    stack = new cdk.Stack();
  });

  test('imports a job and derives its ARN from the name', () => {
    const imported = glue.Job.fromJobAttributes(stack, 'Imported', { jobName: 'ExistingJob' });

    expect(imported.jobName).toEqual('ExistingJob');
    expect(imported.jobArn).toEqual(stack.formatArn({
      service: 'glue',
      resource: 'job',
      resourceName: 'ExistingJob',
    }));
  });

  test('wires the provided role as the grant principal', () => {
    const role = iam.Role.fromRoleArn(stack, 'Role', 'arn:aws:iam::123456789012:role/TestRole');
    const imported = glue.Job.fromJobAttributes(stack, 'Imported', { jobName: 'ExistingJob', role });

    expect(imported.grantPrincipal).toBe(role);
  });

  test('falls back to an unknown principal when no role is provided', () => {
    const imported = glue.Job.fromJobAttributes(stack, 'Imported', { jobName: 'ExistingJob' });

    expect(imported.grantPrincipal).toBeInstanceOf(iam.UnknownPrincipal);
  });
});
