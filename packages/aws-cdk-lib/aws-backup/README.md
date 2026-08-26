# AWS Backup Construct Library


AWS Backup is a fully managed backup service that makes it easy to centralize and automate the
backup of data across AWS services in the cloud and on premises. Using AWS Backup, you can
configure backup policies and monitor backup activity for your AWS resources in one place.

## Backup plan and selection

In AWS Backup, a *backup plan* is a policy expression that defines when and how you want to back up
 your AWS resources, such as Amazon DynamoDB tables or Amazon Elastic File System (Amazon EFS) file
 systems. You can assign resources to backup plans, and AWS Backup automatically backs up and retains
 backups for those resources according to the backup plan. You can create multiple backup plans if you
 have workloads with different backup requirements.

This module provides ready-made backup plans (similar to the console experience):

```ts
// Daily, weekly and monthly with 5 year retention
const plan = backup.BackupPlan.dailyWeeklyMonthly5YearRetention(this, 'Plan');
```

Assigning resources to a plan can be done with `addSelection()`:

```ts
declare const plan: backup.BackupPlan;
declare const vpc: ec2.Vpc;
const myTable = dynamodb.Table.fromTableName(this, 'Table', 'myTableName');
const myDatabaseInstance = new rds.DatabaseInstance(this, 'DatabaseInstance', {
  engine: rds.DatabaseInstanceEngine.mysql({ version: rds.MysqlEngineVersion.VER_8_0_26 }),
  vpc,
});
const myDatabaseCluster = new rds.DatabaseCluster(this, 'DatabaseCluster', {
  engine: rds.DatabaseClusterEngine.auroraMysql({ version: rds.AuroraMysqlEngineVersion.VER_2_08_1 }),
  credentials: rds.Credentials.fromGeneratedSecret('clusteradmin'),
  instanceProps: {
    vpc,
  },
});
const myServerlessCluster = new rds.ServerlessCluster(this, 'ServerlessCluster', {
  engine: rds.DatabaseClusterEngine.AURORA_POSTGRESQL,
  parameterGroup: rds.ParameterGroup.fromParameterGroupName(this, 'ParameterGroup', 'default.aurora-postgresql11'),
  vpc,
});
const myCoolConstruct = new Construct(this, 'MyCoolConstruct');

plan.addSelection('Selection', {
  resources: [
    backup.BackupResource.fromDynamoDbTable(myTable), // A DynamoDB table
    backup.BackupResource.fromRdsDatabaseInstance(myDatabaseInstance), // A RDS instance
    backup.BackupResource.fromRdsDatabaseCluster(myDatabaseCluster), // A RDS database cluster
    backup.BackupResource.fromRdsServerlessCluster(myServerlessCluster), // An Aurora Serverless cluster
    backup.BackupResource.fromTag('stage', 'prod'), // All resources that are tagged stage=prod in the region/account
    backup.BackupResource.fromConstruct(myCoolConstruct), // All backupable resources in `myCoolConstruct`
  ]
})
```

If not specified, a new IAM role with a managed policy for backup will be
created for the selection. The `BackupSelection` implements `IGrantable`.

To disable the plan from assigning the default `AWSBackupServiceRolePolicyForBackup` backup policy use the `disableDefaultBackupPolicy` property.

This is useful if you want to avoid granting unnecessary permissions to the role.

```ts
declare const plan: backup.BackupPlan;

const role = new iam.Role(this, 'BackupRole', {
  assumedBy: new iam.ServicePrincipal('backup.amazonaws.com'),
});
// Assign S3-specific backup policy
role.addManagedPolicy(iam.ManagedPolicy.fromAwsManagedPolicyName('AWSBackupServiceRolePolicyForS3Backup'));

plan.addSelection('Selection', {
  resources: [
    backup.BackupResource.fromTag('stage', 'prod'),
  ],
  role,
  disableDefaultBackupPolicy: true,
});
```

To add rules to a plan, use `addRule()`:

```ts
import { TimeZone } from 'aws-cdk-lib';
declare const plan: backup.BackupPlan;
plan.addRule(new backup.BackupPlanRule({
  completionWindow: Duration.hours(2),
  startWindow: Duration.hours(1),
  scheduleExpression: events.Schedule.cron({ // Only cron expressions are supported
    day: '15',
    hour: '3',
    minute: '30',
  }),
  scheduleExpressionTimezone: TimeZone.ETC_UTC,
  moveToColdStorageAfter: Duration.days(30),
}));
```

Continuous backup and point-in-time restores (PITR) can be configured.
Property `deleteAfter` defines the retention period for the backup. It is mandatory if PITR is enabled.
If no value is specified, the retention period is set to 35 days which is the maximum retention period supported by PITR.
Property `moveToColdStorageAfter` must not be specified because PITR does not support this option.
This example defines an AWS Backup rule with PITR and a retention period set to 14 days:

```ts
declare const plan: backup.BackupPlan;
plan.addRule(new backup.BackupPlanRule({
  enableContinuousBackup: true,
  deleteAfter: Duration.days(14),
}));
```

Rules can also specify to copy recovery points to another Backup Vault using `copyActions`. Copied recovery points can
optionally have `moveToColdStorageAfter` and `deleteAfter` configured.

```ts
declare const plan: backup.BackupPlan;
declare const secondaryVault: backup.BackupVault;
plan.addRule(new backup.BackupPlanRule({
  copyActions: [{
    destinationBackupVault: secondaryVault,
    moveToColdStorageAfter: Duration.days(30),
    deleteAfter: Duration.days(120),
  }]
}));
```

You can assign your own metadata to the resources that are associated with the rule when restored from backup using `recoveryPointTags`. Each tag is a key-value pair.

```ts
declare const plan: backup.BackupPlan;
plan.addRule(new backup.BackupPlanRule({
  recoveryPointTags: {
    key: 'value',
  },
}));
```

You can enable Backup indexes for specific resource types in the rule by using `indexActions`.
Each rule can have at most one index action, and its `resourceTypes` must list at least one
resource type to index.

```ts
declare const plan: backup.BackupPlan;
plan.addRule(new backup.BackupPlanRule({
  indexActions: [{
      resourceTypes: [backup.IndexActionResourceType.S3],
  }],
}));
```

Ready-made rules are also available:

```ts
declare const plan: backup.BackupPlan;
plan.addRule(backup.BackupPlanRule.daily());
plan.addRule(backup.BackupPlanRule.weekly());
```

By default a new [vault](#Backup-vault) is created when creating a plan.
It is also possible to specify a vault either at the plan level or at the
rule level.

```ts
const myVault = backup.BackupVault.fromBackupVaultName(this, 'Vault1', 'myVault');
const otherVault = backup.BackupVault.fromBackupVaultName(this, 'Vault2', 'otherVault');

const plan = backup.BackupPlan.daily35DayRetention(this, 'Plan', myVault); // Use `myVault` for all plan rules
plan.addRule(backup.BackupPlanRule.monthly1Year(otherVault)); // Use `otherVault` for this specific rule
```

You can [backup](https://docs.aws.amazon.com/aws-backup/latest/devguide/windows-backups.html)
VSS-enabled Windows applications running on Amazon EC2 instances by setting the `windowsVss`
parameter to `true`. If the application has VSS writer registered with Windows VSS,
then AWS Backup creates a snapshot that will be consistent for that application.

```ts
const plan = new backup.BackupPlan(this, 'Plan', {
  windowsVss: true,
});
```

## Backup vault

In AWS Backup, a *backup vault* is a container that you organize your backups in. You can use backup
vaults to set the AWS Key Management Service (AWS KMS) encryption key that is used to encrypt backups
in the backup vault and to control access to the backups in the backup vault. If you require different
encryption keys or access policies for different groups of backups, you can optionally create multiple
backup vaults.

```ts
const myKey = kms.Key.fromKeyArn(this, 'MyKey', 'aaa');
const myTopic = sns.Topic.fromTopicArn(this, 'MyTopic', 'bbb');

const vault = new backup.BackupVault(this, 'Vault', {
  encryptionKey: myKey, // Custom encryption key
  notificationTopic: myTopic, // Send all vault events to this SNS topic
});
```

A vault has a default `RemovalPolicy` set to `RETAIN`. Note that removing a vault
that contains recovery points will fail.

You can assign policies to backup vaults and the resources they contain. Assigning policies allows
you to do things like grant access to users to create backup plans and on-demand backups, but limit
their ability to delete recovery points after they're created.

Use the `accessPolicy` property to create a backup vault policy:

```ts
const vault = new backup.BackupVault(this, 'Vault', {
  accessPolicy: new iam.PolicyDocument({
    statements: [
      new iam.PolicyStatement({
        effect: iam.Effect.DENY,
        principals: [new iam.AnyPrincipal()],
        actions: ['backup:DeleteRecoveryPoint'],
        resources: ['*'],
        conditions: {
          StringNotLike: {
            'aws:userId': [
              'user1',
              'user2',
            ],
          },
        },
      }),
    ],
  }),
})
```

Alternativately statements can be added to the vault policy using `addToAccessPolicy()`.

Use the `blockRecoveryPointDeletion` property or the `blockRecoveryPointDeletion()` method to add
a statement to the vault access policy that prevents recovery point deletions in your vault:

```ts
new backup.BackupVault(this, 'Vault', {
  blockRecoveryPointDeletion: true,
});

declare const backupVault: backup.BackupVault;
backupVault.blockRecoveryPointDeletion();
```

By default access is not restricted.

Use the `lockConfiguration` property to enable [AWS Backup Vault Lock](https://docs.aws.amazon.com/aws-backup/latest/devguide/vault-lock.html):

```ts
new backup.BackupVault(this, 'Vault', {
  lockConfiguration: {
    minRetention: Duration.days(30),
  },
});
```

## Importing existing backup vault

To import an existing backup vault into your CDK application, use the `BackupVault.fromBackupVaultArn` or `BackupVault.fromBackupVaultName`
static method. Here is an example of giving an IAM Role permission to start a backup job:

```ts
const importedVault = backup.BackupVault.fromBackupVaultName(this, 'Vault', 'myVaultName');

const role = new iam.Role(this, 'Access Role', { assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com') });

importedVault.grant(role, 'backup:StartBackupJob');
```
