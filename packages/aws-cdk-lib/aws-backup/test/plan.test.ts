import { Match, Template } from '../../assertions';
import * as events from '../../aws-events';
import { App, CfnParameter, Duration, Lazy, Stack, TimeZone } from '../../core';
import { BackupPlan, IndexActionResourceType, BackupPlanRule, BackupVault } from '../lib';

let stack: Stack;
beforeEach(() => {
  stack = new Stack();
});

test('create a plan and add rules', () => {
  // GIVEN
  const vault = new BackupVault(stack, 'Vault');
  const otherVault = new BackupVault(stack, 'OtherVault');

  // WHEN
  const plan = new BackupPlan(stack, 'Plan', {
    backupVault: vault,
    backupPlanRules: [
      new BackupPlanRule({
        completionWindow: Duration.hours(2),
        startWindow: Duration.hours(1),
        scheduleExpression: events.Schedule.cron({
          day: '15',
          hour: '3',
          minute: '30',
        }),
        scheduleExpressionTimezone: TimeZone.ETC_UTC,
        moveToColdStorageAfter: Duration.days(30),
      }),
    ],
  });
  plan.addRule(BackupPlanRule.monthly5Year(otherVault));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          CompletionWindowMinutes: 120,
          Lifecycle: {
            MoveToColdStorageAfterDays: 30,
          },
          RuleName: 'PlanRule0',
          ScheduleExpression: 'cron(30 3 15 * ? *)',
          ScheduleExpressionTimezone: 'Etc/UTC',
          StartWindowMinutes: 60,
          TargetBackupVault: {
            'Fn::GetAtt': [
              'Vault23237E5B',
              'BackupVaultName',
            ],
          },
        },
        {
          Lifecycle: {
            DeleteAfterDays: 1825,
            MoveToColdStorageAfterDays: 90,
          },
          RuleName: 'Monthly5Year',
          ScheduleExpression: 'cron(0 5 1 * ? *)',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'OtherVault3C99BCE2',
              'BackupVaultName',
            ],
          },
        },
      ],
    },
  });
});

test('create a plan with continuous backup option', () => {
  // GIVEN
  const vault = new BackupVault(stack, 'Vault');

  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupVault: vault,
    backupPlanRules: [
      new BackupPlanRule({
        enableContinuousBackup: true,
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          EnableContinuousBackup: true,
          Lifecycle: {
            DeleteAfterDays: 35,
          },
          RuleName: 'PlanRule0',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'Vault23237E5B',
              'BackupVaultName',
            ],
          },
        },
      ],
    },
  });
});

test('create a plan with continuous backup option and specify deleteAfter', () => {
  // GIVEN
  const vault = new BackupVault(stack, 'Vault');

  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupVault: vault,
    backupPlanRules: [
      new BackupPlanRule({
        enableContinuousBackup: true,
        deleteAfter: Duration.days(1),
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          EnableContinuousBackup: true,
          Lifecycle: {
            DeleteAfterDays: 1,
          },
          RuleName: 'PlanRule0',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'Vault23237E5B',
              'BackupVaultName',
            ],
          },
        },
      ],
    },
  });
});

test('create a plan and add rules - add BackupPlan.AdvancedBackupSettings.BackupOptions', () => {
  const vault = new BackupVault(stack, 'Vault');
  const otherVault = new BackupVault(stack, 'OtherVault');

  // WHEN
  const plan = new BackupPlan(stack, 'Plan', {
    windowsVss: true,
    backupVault: vault,
    backupPlanRules: [
      new BackupPlanRule({
        completionWindow: Duration.hours(2),
        startWindow: Duration.hours(1),
        scheduleExpression: events.Schedule.cron({
          day: '15',
          hour: '3',
          minute: '30',
        }),
        moveToColdStorageAfter: Duration.days(30),
      }),
    ],
  });
  plan.addRule(BackupPlanRule.monthly5Year(otherVault));

  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      AdvancedBackupSettings: [{ BackupOptions: { WindowsVSS: 'enabled' }, ResourceType: 'EC2' }],
    },
  });
});

test('daily35DayRetention', () => {
  // WHEN
  BackupPlan.daily35DayRetention(stack, 'D35');

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'D35',
      BackupPlanRule: [
        {
          Lifecycle: {
            DeleteAfterDays: 35,
          },
          RuleName: 'Daily',
          ScheduleExpression: 'cron(0 5 * * ? *)',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'D35Vault2A9EB06F',
              'BackupVaultName',
            ],
          },
        },
      ],
    },
  });
});

test('dailyWeeklyMonthly7YearRetention', () => {
  // WHEN
  BackupPlan.dailyWeeklyMonthly7YearRetention(stack, 'DWM7');

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'DWM7',
      BackupPlanRule: [
        {
          Lifecycle: {
            DeleteAfterDays: 35,
          },
          RuleName: 'Daily',
          ScheduleExpression: 'cron(0 5 * * ? *)',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'DWM7Vault21F17E61',
              'BackupVaultName',
            ],
          },
        },
        {
          Lifecycle: {
            DeleteAfterDays: 90,
          },
          RuleName: 'Weekly',
          ScheduleExpression: 'cron(0 5 ? * SAT *)',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'DWM7Vault21F17E61',
              'BackupVaultName',
            ],
          },
        },
        {
          Lifecycle: {
            DeleteAfterDays: 2555,
            MoveToColdStorageAfterDays: 90,
          },
          RuleName: 'Monthly7Year',
          ScheduleExpression: 'cron(0 5 1 * ? *)',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'DWM7Vault21F17E61',
              'BackupVaultName',
            ],
          },
        },
      ],
    },
  });
});

test('automatically creates a new vault', () => {
  // GIVEN
  const plan = new BackupPlan(stack, 'Plan');

  // WHEN
  plan.addRule(BackupPlanRule.daily());

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          Lifecycle: {
            DeleteAfterDays: 35,
          },
          RuleName: 'Daily',
          ScheduleExpression: 'cron(0 5 * * ? *)',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'PlanVault0284B0C2',
              'BackupVaultName',
            ],
          },
        },
      ],
    },
  });
});

test('create a plan and add rule to copy to a different vault', () => {
  // GIVEN
  const primaryVault = new BackupVault(stack, 'PrimaryVault');
  const secondaryVault = new BackupVault(stack, 'SecondaryVault');

  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupVault: primaryVault,
    backupPlanRules: [
      new BackupPlanRule({
        copyActions: [{
          destinationBackupVault: secondaryVault,
          deleteAfter: Duration.days(120),
          moveToColdStorageAfter: Duration.days(30),
        }],
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          RuleName: 'PlanRule0',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'PrimaryVault9BBEBB0D',
              'BackupVaultName',
            ],
          },
          CopyActions: [{
            DestinationBackupVaultArn: {
              'Fn::GetAtt': [
                'SecondaryVault67665B5E',
                'BackupVaultArn',
              ],
            },
            Lifecycle: {
              DeleteAfterDays: 120,
              MoveToColdStorageAfterDays: 30,
            },
          }],
        },
      ],
    },
  });
});

test('create a plan and add rule with recoveryPointTags', () => {
  // GIVEN
  const tags = {
    key: 'value',
  };

  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupPlanRules: [
      new BackupPlanRule({
        recoveryPointTags: tags,
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          RuleName: 'PlanRule0',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'PlanVault0284B0C2',
              'BackupVaultName',
            ],
          },
          RecoveryPointTags: {
            key: 'value',
          },
        },
      ],
    },
  });
});

test('create a plan and add rule with indexActions', () => {
  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupPlanRules: [
      new BackupPlanRule({
        indexActions: [{ resourceTypes: [IndexActionResourceType.S3] }],
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanName: 'Plan',
      BackupPlanRule: [
        {
          RuleName: 'PlanRule0',
          TargetBackupVault: {
            'Fn::GetAtt': [
              'PlanVault0284B0C2',
              'BackupVaultName',
            ],
          },
          IndexActions: [
            {
              ResourceTypes: [
                'S3',
              ],
            },
          ],
        },
      ],
    },
  });
});

test('create a plan and add rule with EBS indexActions', () => {
  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupPlanRules: [
      new BackupPlanRule({
        indexActions: [{ resourceTypes: [IndexActionResourceType.EBS] }],
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanRule: [
        Match.objectLike({
          IndexActions: [
            {
              ResourceTypes: [
                'EBS',
              ],
            },
          ],
        }),
      ],
    },
  });
});

test('supports a custom index action resource type not yet in CDK', () => {
  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupPlanRules: [
      new BackupPlanRule({
        indexActions: [{ resourceTypes: [new IndexActionResourceType('EFS')] }],
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanRule: [
        Match.objectLike({
          IndexActions: [
            {
              ResourceTypes: [
                'EFS',
              ],
            },
          ],
        }),
      ],
    },
  });
});

test.each([
  [IndexActionResourceType.S3, 'S3'],
  [IndexActionResourceType.EBS, 'EBS'],
])('IndexActionResourceType %s has the expected value', (resourceType, expected) => {
  expect(resourceType.value).toEqual(expected);
});

test('throws when deleteAfter is not greater than moveToColdStorageAfter', () => {
  expect(() => new BackupPlanRule({
    deleteAfter: Duration.days(5),
    moveToColdStorageAfter: Duration.days(6),
  })).toThrow(/`deleteAfter` must be greater than `moveToColdStorageAfter`/);
});

test('throws when scheduleExpression is not of type cron', () => {
  expect(() => new BackupPlanRule({
    scheduleExpression: events.Schedule.rate(Duration.hours(5)),
  })).toThrow(/`scheduleExpression` must be of type `cron`/);
});

test('synth fails when plan has no rules', () => {
  // GIVEN
  const app = new App();
  const myStack = new Stack(app, 'Stack');

  // WHEN
  new BackupPlan(myStack, 'Plan');

  expect(() => app.synth()).toThrow(/A backup plan must have at least 1 rule/);
});

test('throws when moveToColdStorageAfter is used with enableContinuousBackup', () => {
  expect(() => new BackupPlanRule({
    enableContinuousBackup: true,
    deleteAfter: Duration.days(30),
    moveToColdStorageAfter: Duration.days(10),
  })).toThrow(/`moveToColdStorageAfter` must not be specified if `enableContinuousBackup` is enabled/);
});

test('throws when deleteAfter is less than 1 in combination with enableContinuousBackup', () => {
  expect(() => new BackupPlanRule({
    enableContinuousBackup: true,
    deleteAfter: Duration.days(0),
  })).toThrow(/'deleteAfter' must be between 1 and 35 days if 'enableContinuousBackup' is enabled, but got 0 days/);
});

test('throws when deleteAfter is greater than 35 in combination with enableContinuousBackup', () => {
  expect(() => new BackupPlanRule({
    enableContinuousBackup: true,
    deleteAfter: Duration.days(36),
  })).toThrow(/'deleteAfter' must be between 1 and 35 days if 'enableContinuousBackup' is enabled, but got 36 days/);
});

test('throws when deleteAfter is not greater than moveToColdStorageAfter in a copy action', () => {
  expect(() => new BackupPlanRule({
    copyActions: [{
      destinationBackupVault: new BackupVault(stack, 'Vault'),
      deleteAfter: Duration.days(5),
      moveToColdStorageAfter: Duration.days(6),
    }],
  })).toThrow(/deleteAfter' must at least 90 days later than corresponding 'moveToColdStorageAfter'\nreceived 'deleteAfter: 5' and 'moveToColdStorageAfter: 6'/);
});

test('throws when deleteAfter is not greater than 90 days past moveToColdStorageAfter parameter in a copy action', () => {
  expect(() => new BackupPlanRule({
    copyActions: [{
      destinationBackupVault: new BackupVault(stack, 'Vault'),
      deleteAfter: Duration.days(45),
      moveToColdStorageAfter: Duration.days(30),
    }],
  })).toThrow(/'deleteAfter' must at least 90 days later than corresponding 'moveToColdStorageAfter'\nreceived 'deleteAfter: 45' and 'moveToColdStorageAfter: 30'/);
});

test('does not throw when deleteAfter is a token and moveToColdStorageAfter is set', () => {
  expect(() => new BackupPlanRule({
    deleteAfter: Duration.days(Lazy.number({ produce: () => 365 })),
    moveToColdStorageAfter: Duration.days(30),
  })).not.toThrow();
});

test('does not throw when moveToColdStorageAfter is a token', () => {
  expect(() => new BackupPlanRule({
    deleteAfter: Duration.days(365),
    moveToColdStorageAfter: Duration.days(Lazy.number({ produce: () => 30 })),
  })).not.toThrow();
});

test('does not throw when deleteAfter is a token in combination with enableContinuousBackup', () => {
  expect(() => new BackupPlanRule({
    enableContinuousBackup: true,
    deleteAfter: Duration.days(Lazy.number({ produce: () => 14 })),
  })).not.toThrow();
});

test('does not throw when copy action durations are tokens, regardless of token creation order', () => {
  const moveToColdStorageAfter = Duration.days(Lazy.number({ produce: () => 30 }));
  const deleteAfter = Duration.days(Lazy.number({ produce: () => 365 }));
  expect(() => new BackupPlanRule({
    copyActions: [{
      destinationBackupVault: new BackupVault(stack, 'Vault'),
      deleteAfter,
      moveToColdStorageAfter,
    }],
  })).not.toThrow();
});

test('renders a lifecycle that references a CloudFormation parameter', () => {
  // GIVEN
  const retentionDays = new CfnParameter(stack, 'RetentionDays', { type: 'Number', default: 365 });

  // WHEN
  new BackupPlan(stack, 'Plan', {
    backupPlanRules: [
      new BackupPlanRule({
        deleteAfter: Duration.days(retentionDays.valueAsNumber),
        moveToColdStorageAfter: Duration.days(30),
        scheduleExpression: events.Schedule.cron({ hour: '5', minute: '0' }),
      }),
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::Backup::BackupPlan', {
    BackupPlan: {
      BackupPlanRule: [
        Match.objectLike({
          Lifecycle: {
            DeleteAfterDays: { Ref: 'RetentionDays' },
            MoveToColdStorageAfterDays: 30,
          },
        }),
      ],
    },
  });
});
