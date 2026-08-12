import { Template } from '../../assertions';
import { Duration, Stack } from '../../core';
import { Alarm, AlarmRule, AlarmState, AtLeastThreshold, CompositeAlarm, Metric } from '../lib';

describe('CompositeAlarm', () => {
  test('test alarm rule expression builder', () => {
    const stack = new Stack();

    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const alarm1 = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarm2 = new Alarm(stack, 'Alarm2', {
      metric: testMetric,
      threshold: 1000,
      evaluationPeriods: 3,
    });

    const alarm3 = new Alarm(stack, 'Alarm3', {
      metric: testMetric,
      threshold: 10000,
      evaluationPeriods: 3,
    });

    const alarm4 = new Alarm(stack, 'Alarm4', {
      metric: testMetric,
      threshold: 100000,
      evaluationPeriods: 3,
    });

    const alarm5 = new Alarm(stack, 'Alarm5', {
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

    new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::CompositeAlarm', {
      AlarmName: 'CompositeAlarm',
      AlarmRule: {
        'Fn::Join': [
          '',
          [
            '(((ALARM("',
            { 'Fn::GetAtt': ['Alarm1F9009D71', 'Arn'] },
            '") OR OK("',
            { 'Fn::GetAtt': ['Alarm2A7122E13', 'Arn'] },
            '") OR ALARM("',
            { 'Fn::GetAtt': ['Alarm32341D8D9', 'Arn'] },
            '") OR ALARM("',
            { 'Fn::GetAtt': ['Alarm548383B2F', 'Arn'] },
            '")) AND (NOT (INSUFFICIENT_DATA("',
            { 'Fn::GetAtt': ['Alarm4671832C8', 'Arn'] },
            '"))) AND AT_LEAST(2, ALARM, ("',
            { 'Fn::GetAtt': ['Alarm1F9009D71', 'Arn'] },
            '", "',
            { 'Fn::GetAtt': ['Alarm2A7122E13', 'Arn'] },
            '", "',
            { 'Fn::GetAtt': ['Alarm32341D8D9', 'Arn'] },
            '")) AND AT_LEAST(60%, NOT OK, ("',
            { 'Fn::GetAtt': ['Alarm1F9009D71', 'Arn'] },
            '", "',
            { 'Fn::GetAtt': ['Alarm2A7122E13', 'Arn'] },
            '", "',
            { 'Fn::GetAtt': ['Alarm32341D8D9', 'Arn'] },
            '"))) OR FALSE)',
          ],
        ],
      },
    });
  });

  test('test action suppressor translates to a correct CFN properties', () => {
    const stack = new Stack();

    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const actionsSuppressor = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarmRule = AlarmRule.fromBoolean(true);

    new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule,
      actionsSuppressor,
      actionsSuppressorExtensionPeriod: Duration.minutes(2),
      actionsSuppressorWaitPeriod: Duration.minutes(5),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::CompositeAlarm', {
      AlarmName: 'CompositeAlarm',
      ActionsSuppressor: {
        'Fn::GetAtt': [
          'Alarm1F9009D71',
          'Arn',
        ],
      },
      ActionsSuppressorExtensionPeriod: 120,
      ActionsSuppressorWaitPeriod: 300,
    });
  });

  test('test wait and extension periods set without action suppressor', () => {
    const stack = new Stack();

    const alarmRule = AlarmRule.fromBoolean(true);

    var createAlarm = () => new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule,
      actionsSuppressorExtensionPeriod: Duration.minutes(2),
      actionsSuppressorWaitPeriod: Duration.minutes(5),
    });

    expect(createAlarm).toThrow('ActionsSuppressor Extension/Wait Periods require an ActionsSuppressor to be set.');
  });

  test('test action suppressor has correct defaults set', () => {
    const stack = new Stack();

    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const actionsSuppressor = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarmRule = AlarmRule.fromBoolean(true);

    new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule,
      actionsSuppressor,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::CompositeAlarm', {
      AlarmName: 'CompositeAlarm',
      ActionsSuppressor: {
        'Fn::GetAtt': [
          'Alarm1F9009D71',
          'Arn',
        ],
      },
      ActionsSuppressorExtensionPeriod: 60,
      ActionsSuppressorWaitPeriod: 60,
    });
  });

  test('imported alarm arn and name generated correctly', () => {
    const stack = new Stack();

    const alarmFromArn = CompositeAlarm.fromCompositeAlarmArn(stack, 'AlarmFromArn', 'arn:aws:cloudwatch:us-west-2:123456789012:alarm:TestAlarmName');

    expect(alarmFromArn.alarmName).toEqual('TestAlarmName');
    expect(alarmFromArn.alarmArn).toMatch(/:alarm:TestAlarmName$/);

    const alarmFromName = CompositeAlarm.fromCompositeAlarmName(stack, 'AlarmFromName', 'TestAlarmName');

    expect(alarmFromName.alarmName).toEqual('TestAlarmName');
    expect(alarmFromName.alarmArn).toMatch(/:alarm:TestAlarmName$/);
  });

  test('empty anyOf', () => {
    expect(() => new CompositeAlarm(new Stack(), 'alarm', {
      alarmRule: AlarmRule.anyOf(),
    })).toThrow('Did not detect any operands for AlarmRule.anyOf');
  });

  test('empty allOf', () => {
    expect(() => new CompositeAlarm(new Stack(), 'alarm', {
      alarmRule: AlarmRule.allOf(),
    })).toThrow('Did not detect any operands for AlarmRule.allOf');
  });

  test('atLeast renders alarm ARNs with quotes', () => {
    const stack = new Stack();
    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const alarm1 = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarm2 = new Alarm(stack, 'Alarm2', {
      metric: testMetric,
      threshold: 1000,
      evaluationPeriods: 3,
    });

    new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule: AlarmRule.atLeast(AlarmState.ALARM, {
        operands: [alarm1, alarm2],
        threshold: AtLeastThreshold.count(1),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::CompositeAlarm', {
      AlarmRule: {
        'Fn::Join': [
          '',
          [
            'AT_LEAST(1, ALARM, ("',
            { 'Fn::GetAtt': ['Alarm1F9009D71', 'Arn'] },
            '", "',
            { 'Fn::GetAtt': ['Alarm2A7122E13', 'Arn'] },
            '"))',
          ],
        ],
      },
    });
  });

  test('atLeast renders single operand with quotes', () => {
    const stack = new Stack();
    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const alarm1 = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule: AlarmRule.atLeast(AlarmState.ALARM, {
        operands: [alarm1],
        threshold: AtLeastThreshold.count(1),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::CompositeAlarm', {
      AlarmRule: {
        'Fn::Join': [
          '',
          [
            'AT_LEAST(1, ALARM, ("',
            { 'Fn::GetAtt': ['Alarm1F9009D71', 'Arn'] },
            '"))',
          ],
        ],
      },
    });
  });

  test('atLeast quotes alarm ARNs that contain spaces in alarm name', () => {
    const stack = new Stack();
    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });

    const alarm1 = new Alarm(stack, 'Alarm1', {
      alarmName: 'My Web Server CPU',
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    const alarm2 = new Alarm(stack, 'Alarm2', {
      alarmName: 'My DB Connection',
      metric: testMetric,
      threshold: 1000,
      evaluationPeriods: 3,
    });

    new CompositeAlarm(stack, 'CompositeAlarm', {
      alarmRule: AlarmRule.atLeast(AlarmState.ALARM, {
        operands: [alarm1, alarm2],
        threshold: AtLeastThreshold.count(1),
      }),
    });

    // ARNs contain the alarm name (which has spaces), so quotes are required
    // ARN format: arn:aws:cloudwatch:<region>:<account>:alarm:My Web Server CPU
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::CompositeAlarm', {
      AlarmRule: {
        'Fn::Join': [
          '',
          [
            'AT_LEAST(1, ALARM, ("',
            { 'Fn::GetAtt': ['Alarm1F9009D71', 'Arn'] },
            '", "',
            { 'Fn::GetAtt': ['Alarm2A7122E13', 'Arn'] },
            '"))',
          ],
        ],
      },
    });
  });

  test('empty operands for atLeast', () => {
    expect(() => new CompositeAlarm(new Stack(), 'alarm', {
      alarmRule: AlarmRule.atLeast(AlarmState.ALARM, {
        operands: [],
        threshold: AtLeastThreshold.count(1),
      }),
    })).toThrow('Did not detect any operands for AT_LEAST ALARM');
  });

  test('empty operands for atLeastNot', () => {
    expect(() => new CompositeAlarm(new Stack(), 'alarm', {
      alarmRule: AlarmRule.atLeastNot(AlarmState.OK, {
        operands: [],
        threshold: AtLeastThreshold.count(1),
      }),
    })).toThrow('Did not detect any operands for AT_LEAST NOT OK');
  });

  test.each([0, -1, 3, 1.5])('invalid count for atLeast: %s', (count: number) => {
    const stack = new Stack();
    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });
    const alarm = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    expect(() => new CompositeAlarm(new Stack(), 'alarm', {
      alarmRule: AlarmRule.atLeast(AlarmState.OK, {
        operands: [alarm],
        threshold: AtLeastThreshold.count(count),
      }),
    })).toThrow(`count must be an integer between 1 and the number of operands (1), got ${count}`);
  });

  test.each([0, -1, 101, 1.5])('invalid percentage for atLeast: %s%%', (percentage: number) => {
    const stack = new Stack();
    const testMetric = new Metric({
      namespace: 'CDK/Test',
      metricName: 'Metric',
    });
    const alarm = new Alarm(stack, 'Alarm1', {
      metric: testMetric,
      threshold: 100,
      evaluationPeriods: 3,
    });

    expect(() => new CompositeAlarm(new Stack(), 'alarm', {
      alarmRule: AlarmRule.atLeast(AlarmState.OK, {
        operands: [alarm],
        threshold: AtLeastThreshold.percentage(percentage),
      }),
    })).toThrow(`percentage must be an integer between 1 and 100, got ${percentage}`);
  });
});
