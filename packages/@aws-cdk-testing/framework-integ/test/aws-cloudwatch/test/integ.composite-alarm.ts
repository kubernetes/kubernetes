import type { StackProps } from 'aws-cdk-lib';
import { App, CfnOutput, CfnParameter, Stack } from 'aws-cdk-lib';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import { Alarm, AlarmRule, AlarmState, AtLeastThreshold, CompositeAlarm, Metric } from 'aws-cdk-lib/aws-cloudwatch';

class CompositeAlarmIntegrationTest extends Stack {
  constructor(scope: App, id: string, props?: StackProps) {
    super(scope, id, props);

    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const alarm1 = new Alarm(this, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarm2 = new Alarm(this, 'Alarm2', {
      metric: testMetric,
      threshold: 1000,
      evaluationPeriods: 3,
    });

    const alarm3 = new Alarm(this, 'Alarm3', {
      metric: testMetric,
      threshold: 10000,
      evaluationPeriods: 3,
    });

    const alarm4 = new Alarm(this, 'Alarm4', {
      metric: testMetric,
      threshold: 100000,
      evaluationPeriods: 3,
    });

    const alarm5 = new Alarm(this, 'Alarm5', {
      alarmName: 'Alarm with space in name',
      metric: testMetric,
      threshold: 100000,
      evaluationPeriods: 3,
    });

    const alarmRule = AlarmRule.anyOf(
      AlarmRule.allOf(
        AlarmRule.anyOf(
          alarm1,
          AlarmRule.fromAlarm(alarm2, AlarmState.OK),
          alarm3,
          alarm5,
        ),
        AlarmRule.not(AlarmRule.fromAlarm(alarm4, AlarmState.INSUFFICIENT_DATA)),
        AlarmRule.atLeast(AlarmState.ALARM, {
          operands: [alarm1, alarm2, alarm3],
          threshold: AtLeastThreshold.count(2),
        }),
        AlarmRule.atLeastNot(AlarmState.OK, {
          operands: [alarm1, alarm2, alarm3],
          threshold: AtLeastThreshold.percentage(60),
        }),
      ),
      AlarmRule.fromBoolean(false),
    );

    new CompositeAlarm(this, 'CompositeAlarm', {
      alarmRule,
      actionsSuppressor: alarm5,
    });
  }
}

class CompositeAlarmImportIntegrationTest extends Stack {
  constructor(scope: App, id: string, props?: StackProps) {
    super(scope, id, props);

    const alarm = CompositeAlarm.fromCompositeAlarmName(this, 'alarm', 'TestAlarm');

    new CfnOutput(this, 'AlarmName', { value: alarm.alarmName });
    new CfnOutput(this, 'AlarmArn', { value: alarm.alarmArn });
  }
}

class CompositeAlarmTokenThresholdIntegrationTest extends Stack {
  constructor(scope: App, id: string, props?: StackProps) {
    super(scope, id, props);

    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const alarm1 = new Alarm(this, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarm2 = new Alarm(this, 'Alarm2', {
      metric: testMetric,
      threshold: 1000,
      evaluationPeriods: 3,
    });

    const alarm3 = new Alarm(this, 'Alarm3', {
      metric: testMetric,
      threshold: 10000,
      evaluationPeriods: 3,
    });

    const atLeastCount = new CfnParameter(this, 'AtLeastCount', {
      type: 'Number',
      default: 2,
      description: 'Minimum number of alarms that must be in ALARM state',
    });

    const alarmRule = AlarmRule.atLeast(AlarmState.ALARM, {
      operands: [alarm1, alarm2, alarm3],
      threshold: AtLeastThreshold.count(atLeastCount.valueAsNumber),
    });

    new CompositeAlarm(this, 'CompositeAlarmWithTokenThreshold', {
      alarmRule,
    });
  }
}

const app = new App();

new IntegTest(app, 'cdk-integ-composite-alarm', {
  testCases: [
    new CompositeAlarmIntegrationTest(app, 'CompositeAlarmIntegrationTest'),
    new CompositeAlarmImportIntegrationTest(app, 'CompositeAlarmImportIntegrationTest'),
    new CompositeAlarmTokenThresholdIntegrationTest(app, 'CompositeAlarmTokenThresholdIntegrationTest'),
  ],
});
