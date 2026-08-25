import * as integ from '@aws-cdk/integ-tests-alpha';
import { App, Duration, Stack } from 'aws-cdk-lib';
import { Queue } from 'aws-cdk-lib/aws-sqs';

const app = new App();

const stack = new Stack(app, 'aws-cdk-sqs-metrics');

const queue = new Queue(stack, 'Queue');

// `ApproximateNumberOfMessagesVisible + ApproximateNumberOfMessagesNotVisible`, rendered as a
// CloudWatch metric math alarm. Deploying this asserts that CloudWatch accepts the expression.
const alarm = queue
  .metricApproximateNumberOfMessagesOutstanding({ period: Duration.minutes(1) })
  .createAlarm(stack, 'OutstandingMessages', {
    threshold: 100,
    evaluationPeriods: 3,
  });

const test = new integ.IntegTest(app, 'SqsMetricsTest', {
  testCases: [stack],
});

test.assertions
  .awsApiCall('CloudWatch', 'describeAlarms', { AlarmNames: [alarm.alarmName] })
  .expect(integ.ExpectedResult.objectLike({
    MetricAlarms: [{
      Threshold: 100,
      EvaluationPeriods: 3,
    }],
  }));
