import type { StackProps } from 'aws-cdk-lib';
import { App, Duration, Stack, TimeZone } from 'aws-cdk-lib';
import { ExpectedResult, IntegTest, Match } from '@aws-cdk/integ-tests-alpha';
import { Alarm, AlarmMuteRule, Metric, ScheduleExpression } from 'aws-cdk-lib/aws-cloudwatch';

class AlarmMuteRuleIntegrationTest extends Stack {
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
      threshold: 100,
      evaluationPeriods: 3,
    });

    const muteRule1 = alarm1.addAlarmMuteRule('MuteRule1', {
      alarmMuteRuleName: 'AlarmMuteRuleIntegTest',
      description: 'Alarm mute rule integration test',
      schedule: ScheduleExpression.cron({ minute: '0', hour: '0' }),
      duration: Duration.minutes(1),
      start: { year: 2030, month: 1, day: 1, hour: 0, minute: 0 },
      expire: { year: 2030, month: 12, day: 31, hour: 0, minute: 0 },
    });
    muteRule1.addAlarm(alarm2);

    new AlarmMuteRule(this, 'MuteRule2', {
      alarms: [alarm1, alarm2],
      schedule: ScheduleExpression.at({ year: 2030, month: 2, day: 1, hour: 0, minute: 0 }, TimeZone.ASIA_TOKYO),
      duration: Duration.days(15),
    });
  }
}

const app = new App();

const integ = new IntegTest(app, 'cdk-cloudwatch-alarms-mute-rules-integ-test', {
  testCases: [new AlarmMuteRuleIntegrationTest(app, 'AlarmMuteRuleIntegrationTest')],
});

integ.assertions
  .awsApiCall('CloudWatch', 'getAlarmMuteRule', {
    AlarmMuteRuleName: 'AlarmMuteRuleIntegTest',
  }).expect(ExpectedResult.objectLike({
    Name: 'AlarmMuteRuleIntegTest',
    Description: 'Alarm mute rule integration test',
    Rule: {
      Schedule: {
        Expression: 'cron(0 0 * * *)',
        Duration: 'PT1M',
      },
    },
    MuteTargets: {
      AlarmNames: Match.arrayWith([
        Match.stringLikeRegexp('Alarm1'),
        Match.stringLikeRegexp('Alarm2'),
      ]),
    },
  }));
