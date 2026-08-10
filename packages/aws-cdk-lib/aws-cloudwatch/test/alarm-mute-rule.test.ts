import { Match, Template } from '../../assertions';
import * as cdk from '../../core';
import * as cloudwatch from '../lib';

const testMetric = new cloudwatch.Metric({
  namespace: 'CDK/Test',
  metricName: 'Metric',
});

describe('Alarm mute rule', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let alarm: cloudwatch.Alarm;

  beforeEach(() => {
    app = new cdk.App;
    stack = new cdk.Stack(app);
    alarm = new cloudwatch.Alarm(stack, 'Alarm', {
      metric: testMetric,
      threshold: 1,
      evaluationPeriods: 1,
    });
  });

  test('full configurations', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarmMuteRuleName: 'RuleName',
      description: 'RuleDescription',
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', timeZone: cdk.TimeZone.ASIA_TOKYO }),
      duration: cdk.Duration.hours(1),
      start: { year: 2026, month: 1, day: 1, hour: 0, minute: 0 },
      expire: { year: 2026, month: 12, day: 31, hour: 23, minute: 59 },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Name: 'RuleName',
      Description: 'RuleDescription',
      MuteTargets: {
        AlarmNames: [{ Ref: 'Alarm7103F465' }],
      },
      Rule: {
        Schedule: {
          Duration: 'PT1H',
          Expression: 'cron(0 * * * *)',
          Timezone: 'Asia/Tokyo',
        },
      },
      StartDate: '2026-01-01T00:00',
      ExpireDate: '2026-12-31T23:59',
    });
  });

  test('addAlarm', () => {
    // GIVEN
    const alarmMuteRule = new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
      duration: cdk.Duration.hours(1),
    });

    // WHEN
    alarmMuteRule.addAlarm(alarm);
    alarmMuteRule.addAlarm(new cloudwatch.Alarm(stack, 'Alarm2', { metric: testMetric, threshold: 1, evaluationPeriods: 1 }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      MuteTargets: {
        AlarmNames: [{ Ref: 'Alarm7103F465' }, { Ref: 'Alarm2A7122E13' }],
      },
    });
  });

  test('throws when alarmMuteRuleName is blank', () => {
    expect(() => {
      new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
        alarmMuteRuleName: '',
        schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
        duration: cdk.Duration.hours(1),
      });
    }).toThrow('Alarm mute rule name must be between 1 and 255 characters');
  });

  test('throws when length of alarmMuteRuleName > 255', () => {
    expect(() => {
      new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
        alarmMuteRuleName: Array.from({ length: 256 }, () => 'x').join(''),
        schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
        duration: cdk.Duration.hours(1),
      });
    }).toThrow('Alarm mute rule name must be between 1 and 255 characters');
  });

  test('throws when number of target alarms > 100', () => {
    // GIVEN
    const alarmMuteRule = new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
      duration: cdk.Duration.hours(1),
    });

    // WHEN
    for (let i = 1; i <= 101; ++i) {
      alarmMuteRule.addAlarm(new cloudwatch.Alarm(stack, `Alarm${i}`, { metric: testMetric, threshold: 1, evaluationPeriods: 1 }));
    }

    // THEN
    expect(() => app.synth()).toThrow('The maximum number of target alarms is 100.');
  });

  test('cron schedule without time zone', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', hour: '0' }),
      duration: cdk.Duration.hours(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Expression: 'cron(0 0 * * *)',
          Timezone: Match.absent(),
        },
      },
    });
  });

  test('cron schedule with time zone', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', hour: '0', timeZone: cdk.TimeZone.ASIA_TOKYO }),
      duration: cdk.Duration.hours(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Expression: 'cron(0 0 * * *)',
          Timezone: 'Asia/Tokyo',
        },
      },
    });
  });

  test('cron schedule with day and month', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', hour: '0', day: '1', month: 'JAN' }),
      duration: cdk.Duration.hours(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Expression: 'cron(0 0 1 JAN *)',
        },
      },
    });
  });

  test('cron schedule with weekday', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', hour: '0', weekDay: 'SUN' }),
      duration: cdk.Duration.hours(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Expression: 'cron(0 0 * * SUN)',
        },
      },
    });
  });

  test('throws when both day and weekDay are specified', () => {
    expect(() => {
      new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
        alarms: [alarm],
        schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', day: '1', weekDay: 'SUN' }),
        duration: cdk.Duration.hours(1),
      });
    }).toThrow("Cannot supply both 'day' and 'weekDay', use at most one");
  });

  test('at schedule without time zone', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.at({ year: 2026, month: 1, day: 2, hour: 3, minute: 4 }),
      duration: cdk.Duration.hours(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Expression: 'at(2026-01-02T03:04)',
          Timezone: Match.absent(),
        },
      },
    });
  });

  test('at schedule with time zone', () => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.at({ year: 2026, month: 1, day: 2, hour: 3, minute: 4 }, cdk.TimeZone.ASIA_TOKYO),
      duration: cdk.Duration.hours(1),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Expression: 'at(2026-01-02T03:04)',
          Timezone: 'Asia/Tokyo',
        },
      },
    });
  });

  test.each([
    [{ year: 2026, month: 0, day: 1, hour: 0, minute: 0 }],
    [{ year: 2026, month: 13, day: 1, hour: 0, minute: 0 }],
    [{ year: 2026, month: 1, day: 0, hour: 0, minute: 0 }],
    [{ year: 2026, month: 1, day: 32, hour: 0, minute: 0 }],
    [{ year: 2026, month: 1, day: 1, hour: -1, minute: 0 }],
    [{ year: 2026, month: 1, day: 1, hour: 24, minute: 0 }],
    [{ year: 2026, month: 1, day: 1, hour: 0, minute: -1 }],
    [{ year: 2026, month: 1, day: 1, hour: 0, minute: 60 }],
  ])('at schedule throws from %s', (date) => {
    expect(() => cloudwatch.ScheduleExpression.at(date)).toThrow('The specified date is invalid.');
  });

  test.each([
    [cdk.Duration.minutes(10), 'PT10M'],
    [cdk.Duration.hours(10), 'PT10H'],
    [cdk.Duration.days(10), 'P10D'],
  ])('configures duration %s', (duration, durationString) => {
    // WHEN
    new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
      alarms: [alarm],
      schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
      duration,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::AlarmMuteRule', {
      Rule: {
        Schedule: {
          Duration: durationString,
        },
      },
    });
  });

  test('throws when duration is less than 1 minute', () => {
    expect(() => {
      new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
        alarms: [alarm],
        schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
        duration: cdk.Duration.minutes(0),
      });
    }).toThrow('Duration must be greater than or equal to 1 minute');
  });

  test('throws when duration is greater than 15 days', () => {
    expect(() => {
      new cloudwatch.AlarmMuteRule(stack, 'AlarmMuteRule', {
        alarms: [alarm],
        schedule: cloudwatch.ScheduleExpression.cron({ minute: '0' }),
        duration: cdk.Duration.hours(24 * 15 + 1),
      });
    }).toThrow('Duration must be less than or equal to 15 days');
  });

  test('fromAlarmMuteRuleName', () => {
    const alarmMuteRule = cloudwatch.AlarmMuteRule.fromAlarmMuteRuleName(stack, 'AlarmMuteRule', 'MyAlarmMuteRule');
    expect(alarmMuteRule.alarmMuteRuleName).toEqual('MyAlarmMuteRule');
    expect(stack.resolve(alarmMuteRule.alarmMuteRuleArn)).toEqual({ 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':cloudwatch:', { Ref: 'AWS::Region' }, ':', { Ref: 'AWS::AccountId' }, ':alarm-mute-rule:MyAlarmMuteRule']] });
  });

  test('fromAlarmMuteRuleArn', () => {
    const alarmMuteRule = cloudwatch.AlarmMuteRule.fromAlarmMuteRuleArn(stack, 'AlarmMuteRule', 'arn:aws:cloudwatch:us-east-1:123456789012:alarm-mute-rule:MyAlarmMuteRule');
    expect(alarmMuteRule.alarmMuteRuleName).toEqual('MyAlarmMuteRule');
    expect(stack.resolve(alarmMuteRule.alarmMuteRuleArn)).toEqual('arn:aws:cloudwatch:us-east-1:123456789012:alarm-mute-rule:MyAlarmMuteRule');
  });
});
