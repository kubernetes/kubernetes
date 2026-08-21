import { Annotations, Match, Template } from '../../assertions';
import { ArnPrincipal, PolicyDocument, PolicyStatement } from '../../aws-iam';
import * as iam from '../../aws-iam';
import { Stream } from '../../aws-kinesis';
import { Key } from '../../aws-kms';
import { App, CfnDeletionPolicy, Fn, Lazy, RemovalPolicy, Stack, Tags } from '../../core';
import type {
  GlobalSecondaryIndexPropsV2,
  LocalSecondaryIndexProps,
} from '../lib';
import {
  AttributeType,
  Billing,
  Capacity,
  TableV2,
  ProjectionType,
  StreamViewType,
  TableClass,
  TableEncryptionV2,
  MultiRegionConsistency,
  ContributorInsightsMode,
  GlobalTableSettingsReplicationMode,
  TableV2MultiAccountReplica,
} from '../lib';

describe('table', () => {
  test('with default properties', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
      ],
      BillingMode: 'PAY_PER_REQUEST',
      StreamSpecification: Match.absent(),
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
        },
      ],
    });
    Template.fromStack(stack).hasResource('AWS::DynamoDB::GlobalTable', { DeletionPolicy: CfnDeletionPolicy.RETAIN });
  });

  test('with dynamo stream', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      dynamoStream: StreamViewType.NEW_IMAGE,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      StreamSpecification: {
        StreamViewType: 'NEW_IMAGE',
      },
    });
  });

  test('with sort key', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.NUMBER },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'N' },
      ],
    });
  });

  test('with contributor insights', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      contributorInsights: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ContributorInsightsSpecification: {
            Enabled: true,
          },
        },
      ],
    });
  });

  test('with deletion protection', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      deletionProtection: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          DeletionProtectionEnabled: true,
        },
      ],
    });
  });

  test('with point-in-time recovery', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      pointInTimeRecovery: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          PointInTimeRecoverySpecification: {
            PointInTimeRecoveryEnabled: true,
          },
        },
      ],
    });
  });

  test('with point-in-time-recovery-specification', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true,
        recoveryPeriodInDays: 4,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          PointInTimeRecoverySpecification: {
            PointInTimeRecoveryEnabled: true,
            RecoveryPeriodInDays: 4,
          },
        },
      ],
    });
  });

  test('both point-in-time-recovery-specification and point-in-time-recovery set', () => {
    const stack = new Stack(undefined, 'Stack');
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        pointInTimeRecovery: true,
        pointInTimeRecoverySpecification: {
          pointInTimeRecoveryEnabled: true,
          recoveryPeriodInDays: 5,
        },
      });
    }).toThrow('`pointInTimeRecoverySpecification` and `pointInTimeRecovery` are set. Use `pointInTimeRecoverySpecification` only.');
  });

  test('recoveryPeriodInDays set out of bounds', () => {
    const stack = new Stack(undefined, 'Stack');
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        pointInTimeRecoverySpecification: {
          pointInTimeRecoveryEnabled: true,
          recoveryPeriodInDays: 36,
        },
      });
    }).toThrow('`recoveryPeriodInDays` must be a value between `1` and `35`.');
  });

  test('recoveryPeriodInDays set but pitr ENABLED', () => {
    const stack = new Stack(undefined, 'Stack');
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        pointInTimeRecoverySpecification: {
          pointInTimeRecoveryEnabled: false,
          recoveryPeriodInDays: 35,
        },
      });
    }).toThrow('Cannot set `recoveryPeriodInDays` while `pointInTimeRecoveryEnabled` is set to false.');
  });

  test('with STANDARD table class', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      tableClass: TableClass.STANDARD,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          TableClass: 'STANDARD',
        },
      ],
    });
  });

  test('with STANDARD_INFREQUENT_ACCESS table class', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      tableClass: TableClass.STANDARD_INFREQUENT_ACCESS,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          TableClass: 'STANDARD_INFREQUENT_ACCESS',
        },
      ],
    });
  });

  test('with kinesis stream', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');
    const kinesisStream = new Stream(stack, 'Stream');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      kinesisStream,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          KinesisStreamSpecification: {
            StreamArn: {
              'Fn::GetAtt': [
                'Stream790BDEE4',
                'Arn',
              ],
            },
          },
        },
      ],
    });
  });

  test('with table name', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      tableName: 'my-table',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      TableName: 'my-table',
    });
  });

  test('with TTL attribute', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      timeToLiveAttribute: 'attribute',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      TimeToLiveSpecification: {
        AttributeName: 'attribute',
        Enabled: true,
      },
    });
  });

  test('with removal policy as DESTROY', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::DynamoDB::GlobalTable', { DeletionPolicy: CfnDeletionPolicy.DELETE });
  });

  test('with on-demand billing', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.onDemand(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      BillingMode: 'PAY_PER_REQUEST',
    });
  });

  test('with provisioned billing and fixed read capacity', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(10),
        writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      BillingMode: 'PROVISIONED',
      WriteProvisionedThroughputSettings: {
        WriteCapacityAutoScalingSettings: {
          MinCapacity: 1,
          MaxCapacity: 10,
          TargetTrackingScalingPolicyConfiguration: {
            TargetValue: 70,
          },
        },
      },
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
        },
      ],
    });
  });

  test('with provisioned billing and autoscaled read capacity', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
        writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      BillingMode: 'PROVISIONED',
      WriteProvisionedThroughputSettings: {
        WriteCapacityAutoScalingSettings: {
          MinCapacity: 1,
          MaxCapacity: 10,
          TargetTrackingScalingPolicyConfiguration: {
            TargetValue: 70,
          },
        },
      },
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ReadProvisionedThroughputSettings: {
            ReadCapacityAutoScalingSettings: {
              MinCapacity: 1,
              MaxCapacity: 10,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
        },
      ],
    });
  });

  test('with non-default replica table', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{ region: 'us-east-1' }],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        { Region: 'us-east-1' },
        { Region: 'us-west-2' },
      ],
    });
  });

  test('with global secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.BINARY },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'B' },
        { AttributeName: 'gsi-pk', AttributeType: 'N' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
            },
          ],
        },
      ],
    });
  });

  test('with local secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.BINARY },
      localSecondaryIndexes: [
        {
          indexName: 'lsi',
          sortKey: { name: 'lsi-sk', type: AttributeType.NUMBER },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'B' },
        { AttributeName: 'lsi-sk', AttributeType: 'N' },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
    });
  });

  test('with encryption via dynamodb owned key', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      encryption: TableEncryptionV2.dynamoOwnedKey(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      SSESpecification: {
        SSEEnabled: false,
      },
    });
  });

  test('with encryption via aws managed key', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      encryption: TableEncryptionV2.awsManagedKey(),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      SSESpecification: {
        SSEEnabled: true,
        SSEType: 'KMS',
      },
    });
  });

  test('with encryption via customer managed key', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    const tableKey = new Key(stack, 'Key');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      encryption: TableEncryptionV2.customerManagedKey(tableKey),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      SSESpecification: {
        SSEEnabled: true,
        SSEType: 'KMS',
      },
    });
  });

  test('with tags', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{ region: 'us-west-1' }],
      tags: [{ key: 'tagKey', value: 'tagValue' }],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-west-1',
        },
        {
          Region: 'us-east-1',
          Tags: [{ Key: 'tagKey', Value: 'tagValue' }],
        },
      ],
    });
  });

  test('with all properties configured', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    const stream = new Stream(stack, 'Stream');

    const tableKey = new Key(stack, 'Key');
    const replicaKeyArns = {
      'us-east-1': 'arn:aws:kms:us-east-1:123456789012:key/g24efbna-az9b-42ro-m3bp-cq249l94fca6',
      'us-east-2': 'arn:aws:kms:us-east-2:123456789012:key/g24efbna-az9b-42ro-m3bp-cq249l94fca6',
    };

    // WHEN
    new TableV2(stack, 'Table', {
      tableName: 'my-table',
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.NUMBER },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(10),
        writeCapacity: Capacity.autoscaled({ maxCapacity: 20 }),
      }),
      encryption: TableEncryptionV2.customerManagedKey(tableKey, replicaKeyArns),
      contributorInsights: true,
      deletionProtection: true,
      pointInTimeRecovery: true,
      tableClass: TableClass.STANDARD_INFREQUENT_ACCESS,
      kinesisStream: stream,
      timeToLiveAttribute: 'attribute',
      removalPolicy: RemovalPolicy.DESTROY,
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
          readCapacity: Capacity.fixed(10),
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
        },
      ],
      localSecondaryIndexes: [
        {
          indexName: 'lsi',
          sortKey: { name: 'sk', type: AttributeType.NUMBER },
        },
      ],
      replicas: [
        {
          region: 'us-east-1',
          deletionProtection: false,
          readCapacity: Capacity.autoscaled({
            minCapacity: 5,
            maxCapacity: 25,
          }),
          globalSecondaryIndexOptions: {
            gsi2: {
              contributorInsights: false,
            },
          },
          tags: [{ key: 'USE1Key', value: 'USE1Value' }],
        },
        {
          region: 'us-east-2',
          tableClass: TableClass.STANDARD,
          contributorInsights: false,
          globalSecondaryIndexOptions: {
            gsi1: {
              readCapacity: Capacity.fixed(15),
            },
          },
          tags: [{ key: 'USE2Key', value: 'USE2Value' }],
        },
      ],
      tags: [{ key: 'USW2Key', value: 'USW2Value' }],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      AttributeDefinitions: [
        {
          AttributeName: 'pk',
          AttributeType: 'S',
        },
        {
          AttributeName: 'sk',
          AttributeType: 'N',
        },
      ],
      BillingMode: 'PROVISIONED',
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi1',
          KeySchema: [
            {
              AttributeName: 'pk',
              KeyType: 'HASH',
            },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
          WriteProvisionedThroughputSettings: {
            WriteCapacityAutoScalingSettings: {
              MaxCapacity: 20,
              MinCapacity: 1,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
        },
        {
          IndexName: 'gsi2',
          KeySchema: [
            {
              AttributeName: 'pk',
              KeyType: 'HASH',
            },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
          WriteProvisionedThroughputSettings: {
            WriteCapacityAutoScalingSettings: {
              MaxCapacity: 20,
              MinCapacity: 1,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
        },
      ],
      KeySchema: [
        {
          AttributeName: 'pk',
          KeyType: 'HASH',
        },
        {
          AttributeName: 'sk',
          KeyType: 'RANGE',
        },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi',
          KeySchema: [
            {
              AttributeName: 'pk',
              KeyType: 'HASH',
            },
            {
              AttributeName: 'sk',
              KeyType: 'RANGE',
            },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          ContributorInsightsSpecification: {
            Enabled: true,
          },
          DeletionProtectionEnabled: false,
          GlobalSecondaryIndexes: [
            {
              ContributorInsightsSpecification: {
                Enabled: true,
              },
              IndexName: 'gsi1',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
            {
              ContributorInsightsSpecification: {
                Enabled: false,
              },
              IndexName: 'gsi2',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
          ],
          KinesisStreamSpecification: Match.absent(),
          PointInTimeRecoverySpecification: {
            PointInTimeRecoveryEnabled: true,
          },
          ReadProvisionedThroughputSettings: {
            ReadCapacityAutoScalingSettings: {
              MaxCapacity: 25,
              MinCapacity: 5,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
          Region: 'us-east-1',
          SSESpecification: {
            KMSMasterKeyId: 'arn:aws:kms:us-east-1:123456789012:key/g24efbna-az9b-42ro-m3bp-cq249l94fca6',
          },
          TableClass: 'STANDARD_INFREQUENT_ACCESS',
          Tags: [{ Key: 'USE1Key', Value: 'USE1Value' }],
        },
        {
          ContributorInsightsSpecification: {
            Enabled: false,
          },
          DeletionProtectionEnabled: true,
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 15,
              },
            },
            {
              ContributorInsightsSpecification: {
                Enabled: true,
              },
              IndexName: 'gsi2',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
          ],
          KinesisStreamSpecification: Match.absent(),
          PointInTimeRecoverySpecification: {
            PointInTimeRecoveryEnabled: true,
          },
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
          Region: 'us-east-2',
          SSESpecification: {
            KMSMasterKeyId: 'arn:aws:kms:us-east-2:123456789012:key/g24efbna-az9b-42ro-m3bp-cq249l94fca6',
          },
          TableClass: 'STANDARD',
          Tags: [{ Key: 'USE2Key', Value: 'USE2Value' }],
        },
        {
          ContributorInsightsSpecification: {
            Enabled: true,
          },
          DeletionProtectionEnabled: true,
          GlobalSecondaryIndexes: [
            {
              ContributorInsightsSpecification: {
                Enabled: true,
              },
              IndexName: 'gsi1',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
            {
              ContributorInsightsSpecification: {
                Enabled: true,
              },
              IndexName: 'gsi2',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
          ],
          KinesisStreamSpecification: {
            StreamArn: {
              'Fn::GetAtt': [
                'Stream790BDEE4',
                'Arn',
              ],
            },
          },
          PointInTimeRecoverySpecification: {
            PointInTimeRecoveryEnabled: true,
          },
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
          Region: 'us-west-2',
          SSESpecification: {
            KMSMasterKeyId: {
              'Fn::GetAtt': [
                'Key961B73FD',
                'Arn',
              ],
            },
          },
          TableClass: 'STANDARD_INFREQUENT_ACCESS',
          Tags: [{ Key: 'USW2Key', Value: 'USW2Value' }],
        },
      ],
      SSESpecification: {
        SSEEnabled: true,
        SSEType: 'KMS',
      },
      StreamSpecification: {
        StreamViewType: 'NEW_AND_OLD_IMAGES',
      },
      TableName: 'my-table',
      TimeToLiveSpecification: {
        AttributeName: 'attribute',
        Enabled: true,
      },
      WriteProvisionedThroughputSettings: {
        WriteCapacityAutoScalingSettings: {
          MaxCapacity: 20,
          MinCapacity: 1,
          TargetTrackingScalingPolicyConfiguration: {
            TargetValue: 70,
          },
        },
      },
    });
  });

  test('can add global secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.BINARY },
    });

    const globalSecondaryIndex: GlobalSecondaryIndexPropsV2 = {
      indexName: 'gsi',
      partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
    };

    // WHEN
    table.addGlobalSecondaryIndex(globalSecondaryIndex);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'B' },
        { AttributeName: 'gsi-pk', AttributeType: 'N' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
            },
          ],
        },
      ],
    });
  });

  test('can add local secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.BINARY },
    });
    const localSecondaryIndex: LocalSecondaryIndexProps = {
      indexName: 'lsi',
      sortKey: { name: 'lsi-sk', type: AttributeType.NUMBER },
    };

    // WHEN
    table.addLocalSecondaryIndex(localSecondaryIndex);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'B' },
        { AttributeName: 'lsi-sk', AttributeType: 'N' },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
    });
  });

  test('multiple tables', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table1', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });
    new TableV2(stack, 'Table2', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });
    new TableV2(stack, 'Table3', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::DynamoDB::GlobalTable', 3);
  });

  test('throws if defining non-default replica table in region agnostic stack', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        replicas: [{ region: 'us-east-1' }],
      });
    }).toThrow('Replica tables are not supported in a region agnostic stack');
  });

  test('throws if getting replica table in region agnostic stack', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });

    // WHEN / THEN
    expect(() => {
      table.replica('us-west-2');
    }).toThrow('Replica tables are not supported in a region agnostic stack');
  });

  test('with on-demand maximum throughput', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.onDemand({
        maxReadRequestUnits: 10,
        maxWriteRequestUnits: 10,
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
      ],
      WriteOnDemandThroughputSettings: {
        MaxWriteRequestUnits: 10,
      },
      BillingMode: 'PAY_PER_REQUEST',
      StreamSpecification: Match.absent(),
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ReadOnDemandThroughputSettings: {
            MaxReadRequestUnits: 10,
          },
        },
      ],
    });
    Template.fromStack(stack).hasResource('AWS::DynamoDB::GlobalTable', { DeletionPolicy: CfnDeletionPolicy.RETAIN });
  });

  test('with on-demand maximum throughput - read only', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.onDemand({
        maxReadRequestUnits: 10,
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
      ],
      BillingMode: 'PAY_PER_REQUEST',
      StreamSpecification: Match.absent(),
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ReadOnDemandThroughputSettings: {
            MaxReadRequestUnits: 10,
          },
        },
      ],
    });
    Template.fromStack(stack).hasResource('AWS::DynamoDB::GlobalTable', { DeletionPolicy: CfnDeletionPolicy.RETAIN });
  });

  test('with on-demand maximum throughput - index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
          maxReadRequestUnits: 100,
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
          maxReadRequestUnits: 1,
          maxWriteRequestUnits: 1,
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi1',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'gsi2',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
          WriteOnDemandThroughputSettings: {
            MaxWriteRequestUnits: 1,
          },
        },
      ],
      BillingMode: 'PAY_PER_REQUEST',
      StreamSpecification: Match.absent(),
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [{
            IndexName: 'gsi1',
            ReadOnDemandThroughputSettings: {
              MaxReadRequestUnits: 100,
            },
          },
          {
            IndexName: 'gsi2',
            ReadOnDemandThroughputSettings: {
              MaxReadRequestUnits: 1,
            },
          }],
        },
      ],
    });
    Template.fromStack(stack).hasResource('AWS::DynamoDB::GlobalTable', { DeletionPolicy: CfnDeletionPolicy.RETAIN });
  });
});

describe('grants', () => {
  test('grants.readData includes index ARN when a GSI is added after construction', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });
    const role = new iam.Role(stack, 'Role', { assumedBy: new iam.AccountRootPrincipal() });

    // WHEN
    table.addGlobalSecondaryIndex({
      indexName: 'gsi1',
      partitionKey: { name: 'gsiPk', type: AttributeType.STRING },
    });
    table.grants.readData(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Resource: Match.arrayWith([
              Match.objectLike({
                'Fn::Join': ['', Match.arrayWith(['/index/*'])],
              }),
            ]),
          }),
        ]),
      },
    });
  });

  test('grants.readData includes index ARN when a GSI is provided via props', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [{
        indexName: 'gsi1',
        partitionKey: { name: 'gsiPk', type: AttributeType.STRING },
      }],
    });
    const role = new iam.Role(stack, 'Role', { assumedBy: new iam.AccountRootPrincipal() });

    // WHEN
    table.grants.readData(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Resource: Match.arrayWith([
              Match.objectLike({
                'Fn::Join': ['', Match.arrayWith(['/index/*'])],
              }),
            ]),
          }),
        ]),
      },
    });
  });

  test('grants.readData includes index ARN when an LSI is added after construction', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
    });
    const role = new iam.Role(stack, 'Role', { assumedBy: new iam.AccountRootPrincipal() });

    // WHEN
    table.addLocalSecondaryIndex({
      indexName: 'lsi1',
      sortKey: { name: 'lsiSk', type: AttributeType.STRING },
    });
    table.grants.readData(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Resource: Match.arrayWith([
              Match.objectLike({
                'Fn::Join': ['', Match.arrayWith(['/index/*'])],
              }),
            ]),
          }),
        ]),
      },
    });
  });

  test('grants.readData omits index ARN when the table has no indexes', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });
    const role = new iam.Role(stack, 'Role', { assumedBy: new iam.AccountRootPrincipal() });

    // WHEN
    table.grants.readData(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Resource: Match.not(Match.arrayWith([
              Match.objectLike({
                'Fn::Join': ['', Match.arrayWith(['/index/*'])],
              }),
            ])),
          }),
        ]),
      },
    });
  });

  test('grantReadData with AccountRootPrincipal uses wildcard resources', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });

    // WHEN
    table.grantReadData(new iam.AccountRootPrincipal());

    // THEN - Should create resource policy with wildcard to avoid circular dependency
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: Match.arrayWith([
        Match.objectLike({
          ResourcePolicy: {
            PolicyDocument: {
              Statement: Match.arrayWith([
                Match.objectLike({
                  Action: [
                    'dynamodb:BatchGetItem',
                    'dynamodb:Query',
                    'dynamodb:GetItem',
                    'dynamodb:Scan',
                    'dynamodb:ConditionCheckItem',
                    'dynamodb:DescribeTable',
                  ],
                  Effect: 'Allow',
                  Resource: '*', // Wildcard to avoid circular dependency
                  Principal: Match.anyValue(), // AccountRootPrincipal
                }),
              ]),
            },
          },
        }),
      ]),
    });
  });

  test('grant* with ServicePrincipal throws error', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(() => table.grantReadWriteData(new iam.ServicePrincipal('bedrock.amazonaws.com')))
      .toThrow(/DynamoDB grant\* methods do not support ServicePrincipal grantees/);
  });

  test('grant with ServicePrincipal throws error', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(() => table.grant(new iam.ServicePrincipal('bedrock.amazonaws.com'), 'dynamodb:GetItem'))
      .toThrow(/DynamoDB grant\* methods do not support ServicePrincipal grantees/);
  });

  test('grant* with wrapped ServicePrincipal (withConditions) throws error', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // WHEN
    const principal = new iam.ServicePrincipal('bedrock.amazonaws.com').withConditions({
      StringEquals: { 'aws:SourceAccount': '123456789012' },
    });

    // THEN
    expect(() => table.grantReadData(principal))
      .toThrow(/DynamoDB grant\* methods do not support ServicePrincipal grantees/);
  });

  test.each([
    'redshift.amazonaws.com',
    'replication.dynamodb.amazonaws.com',
    'glue.amazonaws.com',
  ])('grant* with allowlisted ServicePrincipal %s succeeds', (serviceName) => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // WHEN
    const grant = table.grantReadWriteData(new iam.ServicePrincipal(serviceName));

    // THEN
    expect(grant.success).toBe(true);
  });

  test('grant* with wrapped allowlisted ServicePrincipal succeeds', () => {
    // GIVEN
    const stack = new Stack();
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // WHEN
    const principal = new iam.ServicePrincipal('redshift.amazonaws.com').withConditions({
      StringEquals: { 'aws:SourceAccount': '123456789012' },
    });
    const grant = table.grantReadWriteData(principal);

    // THEN
    expect(grant.success).toBe(true);
  });
});

describe('replica tables', () => {
  test('with fixed read capacity', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(5),
        writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
      }),
      replicas: [
        { region: 'us-east-1', readCapacity: Capacity.fixed(10) },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-east-1',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
        },
        {
          Region: 'us-west-2',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 5,
          },
        },
      ],
    });
  });

  test('with autoscaled read capacity', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(5),
        writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
      }),
      replicas: [
        {
          region: 'us-east-1',
          readCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-east-1',
          ReadProvisionedThroughputSettings: {
            ReadCapacityAutoScalingSettings: {
              MinCapacity: 1,
              MaxCapacity: 10,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
        },
        {
          Region: 'us-west-2',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 5,
          },
        },
      ],
    });
  });

  test('with tags', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{
        region: 'us-west-1',
        tags: [{ key: 'tagKey', value: 'tagValue' }],
      }],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-west-1',
          Tags: [{ Key: 'tagKey', Value: 'tagValue' }],
        },
        {
          Region: 'us-east-1',
        },
      ],
    });
  });

  test('with TagAspect', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{
        region: 'us-west-1',
      }],
    });

    Tags.of(table).add('tagKey', 'tagValue');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-west-1',
          Tags: [{ Key: 'tagKey', Value: 'tagValue' }],
        },
        {
          Region: 'us-east-1',
          Tags: [{ Key: 'tagKey', Value: 'tagValue' }],
        },
      ],
    });
  });

  test('with TagAspect on parent scope', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{
        region: 'us-west-1',
      }],
    });

    Tags.of(stack).add('stage', 'Prod');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-west-1',
          Tags: [{ Key: 'stage', Value: 'Prod' }],
        },
        {
          Region: 'us-east-1',
          Tags: [{ Key: 'stage', Value: 'Prod' }],
        },
      ],
    });
  });

  test('replica tags override tag aspect tags', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

    // WHEN
    const table = new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{
        region: 'us-west-1',
        tags: [{ key: 'tableTagProperty', value: 'replicaW1TagPropertyValue' }],
      }, {
        region: 'us-west-2',
      }],
      tags: [{ key: 'tableTagProperty', value: 'defaultReplicaTagPropertyValue' }],
    });

    Tags.of(table).add('tableTagProperty', 'tagAspectValue');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-west-1',
          Tags: [
            { Key: 'tableTagProperty', Value: 'replicaW1TagPropertyValue' },
          ],
        },
        {
          Region: 'us-west-2',
          Tags: [
            { Key: 'tableTagProperty', Value: 'tagAspectValue' },
          ],
        },
        {
          Region: 'us-east-1',
          Tags: [
            { Key: 'tableTagProperty', Value: 'defaultReplicaTagPropertyValue' },
          ],
        },
      ],
    });
  });

  test('with per-replica kinesis stream', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    const kinesisStream1 = new Stream(stack, 'Stream1');
    const kinesisStream2 = Stream.fromStreamArn(stack, 'Stream2', 'arn:aws:kinesis:us-east-1:123456789012:stream/my-stream');

    // WHEN
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      kinesisStream: kinesisStream1,
      replicas: [
        {
          region: 'us-east-1',
          kinesisStream: kinesisStream2,
        },
        {
          region: 'us-east-2',
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-east-1',
          KinesisStreamSpecification: {
            StreamArn: 'arn:aws:kinesis:us-east-1:123456789012:stream/my-stream',
          },
        },
        {
          Region: 'us-east-2',
          KinesisStreamSpecification: Match.absent(),
        },
        {
          Region: 'us-west-2',
          KinesisStreamSpecification: {
            StreamArn: {
              'Fn::GetAtt': [
                'Stream16C8F97AF',
                'Arn',
              ],
            },
          },
        },
      ],
    });
  });

  test('with per-replica contributor insights on global secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      contributorInsights: true,
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
        },
      ],
      replicas: [
        {
          region: 'us-east-2',
          globalSecondaryIndexOptions: {
            gsi2: {
              contributorInsights: false,
            },
          },
        },
        {
          region: 'us-east-1',
          globalSecondaryIndexOptions: {
            gsi1: {
              contributorInsights: false,
            },
          },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-east-2',
          ContributorInsightsSpecification: {
            Enabled: true,
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ContributorInsightsSpecification: {
                Enabled: true,
              },
            },
            {
              IndexName: 'gsi2',
              ContributorInsightsSpecification: {
                Enabled: false,
              },
            },
          ],
        },
        {
          Region: 'us-east-1',
          ContributorInsightsSpecification: {
            Enabled: true,
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ContributorInsightsSpecification: {
                Enabled: false,
              },
            },
            {
              IndexName: 'gsi2',
              ContributorInsightsSpecification: {
                Enabled: true,
              },
            },
          ],
        },
        {
          Region: 'us-west-2',
          ContributorInsightsSpecification: {
            Enabled: true,
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ContributorInsightsSpecification: {
                Enabled: true,
              },
            },
            {
              IndexName: 'gsi2',
              ContributorInsightsSpecification: {
                Enabled: true,
              },
            },
          ],
        },
      ],
    });
  });

  test('with per-replica read capacity on global secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(10),
        writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
      }),
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
          readCapacity: Capacity.fixed(10),
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
          readCapacity: Capacity.fixed(10),
        },
      ],
      replicas: [
        {
          region: 'us-east-2',
          globalSecondaryIndexOptions: {
            gsi2: {
              readCapacity: Capacity.fixed(15),
            },
          },
        },
        {
          region: 'us-east-1',
          globalSecondaryIndexOptions: {
            gsi1: {
              readCapacity: Capacity.autoscaled({ minCapacity: 5, maxCapacity: 15 }),
            },
          },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        {
          Region: 'us-east-2',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
            {
              IndexName: 'gsi2',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 15,
              },
            },
          ],
        },
        {
          Region: 'us-east-1',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ReadProvisionedThroughputSettings: {
                ReadCapacityAutoScalingSettings: {
                  MinCapacity: 5,
                  MaxCapacity: 15,
                  TargetTrackingScalingPolicyConfiguration: {
                    TargetValue: 70,
                  },
                },
              },
            },
            {
              IndexName: 'gsi2',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
          ],
        },
        {
          Region: 'us-west-2',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
            {
              IndexName: 'gsi2',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
          ],
        },
      ],
    });
  });

  test('throws if replica table region is a token', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    const table = new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });

    // WHEN / THEN
    expect(() => {
      table.addReplica({ region: Lazy.string({ produce: () => 'us-east-1' }) });
    }).toThrow('Replica table region must not be a token');
  });

  test('throws if adding replica table in deployment region', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    const table = new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });

    // WHEN / THEN
    expect(() => {
      table.addReplica({ region: 'us-west-2' });
    }).toThrow('You cannot add a replica table in the same region as the primary table - the primary table region is us-west-2');
  });

  test('throws if adding duplicate replica table', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    const table = new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{ region: 'us-east-1' }],
    });

    // WHEN / THEN
    expect(() => {
      table.addReplica({ region: 'us-east-1' });
    }).toThrow('Duplicate replica table region, us-east-1, is not allowed');
  });

  test('throws if read capacity is configured on replica table when billing mode is PAY_PER_REQUEST', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'GlobalTable', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        replicas: [
          {
            region: 'us-east-1',
            readCapacity: Capacity.fixed(10),
          },
        ],
      });
    }).toThrow("You cannot provide 'readCapacity' on a replica table when the billing mode is PAY_PER_REQUEST");
  });

  test('throws if configuring options for non-existent global secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
        },
      ],
      replicas: [
        {
          region: 'us-east-1',
          globalSecondaryIndexOptions: {
            global: {
              readCapacity: Capacity.fixed(10),
            },
          },
        },
      ],
    });

    // WHEN / THEN
    expect(() => {
      Template.fromStack(stack);
    }).toThrow('Cannot configure replica global secondary index, global, because it is not defined on the primary table');
  });

  test('throws if read capacity is configured as global secondary index options when billing mode is PAY_PER_REQUEST', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });
    new TableV2(stack, 'GlobalTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'pk', type: AttributeType.STRING },
        },
      ],
      replicas: [
        {
          region: 'us-east-1',
          globalSecondaryIndexOptions: {
            gsi: {
              readCapacity: Capacity.fixed(10),
            },
          },
        },
      ],
    });

    // WHEN / THEN
    expect(() => {
      Template.fromStack(stack);
    }).toThrow("Cannot configure 'readCapacity' for replica global secondary index, gsi, because billing mode is PAY_PER_REQUEST");
  });
});

describe('secondary indexes', () => {
  test('with multiple global secondary indexes with different partition keys', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'gsi-pk-1', type: AttributeType.NUMBER },
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'gsi-pk-2', type: AttributeType.NUMBER },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk-1', AttributeType: 'N' },
        { AttributeName: 'gsi-pk-2', AttributeType: 'N' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi1',
          KeySchema: [
            { AttributeName: 'gsi-pk-1', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'gsi2',
          KeySchema: [
            { AttributeName: 'gsi-pk-2', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
            },
            {
              IndexName: 'gsi2',
            },
          ],
        },
      ],
    });
  });

  test('with multiple global secondary indexes with the same partition keys', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk', AttributeType: 'N' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi1',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'gsi2',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
            },
            {
              IndexName: 'gsi2',
            },
          ],
        },
      ],
    });
  });

  test('with multiple global secondary indexes with different sort keys', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
          sortKey: { name: 'gsi-sk-1', type: AttributeType.STRING },
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
          sortKey: { name: 'gsi-sk-2', type: AttributeType.STRING },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk', AttributeType: 'N' },
        { AttributeName: 'gsi-sk-1', AttributeType: 'S' },
        { AttributeName: 'gsi-sk-2', AttributeType: 'S' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi1',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
            { AttributeName: 'gsi-sk-1', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'gsi2',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
            { AttributeName: 'gsi-sk-2', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
            },
            {
              IndexName: 'gsi2',
            },
          ],
        },
      ],
    });
  });

  test('with multiple global secondary indexes with the same sort keys', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi1',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
          sortKey: { name: 'gsi-sk', type: AttributeType.STRING },
        },
        {
          indexName: 'gsi2',
          partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
          sortKey: { name: 'gsi-sk', type: AttributeType.STRING },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk', AttributeType: 'N' },
        { AttributeName: 'gsi-sk', AttributeType: 'S' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi1',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
            { AttributeName: 'gsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'gsi2',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
            { AttributeName: 'gsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi1',
            },
            {
              IndexName: 'gsi2',
            },
          ],
        },
      ],
    });
  });

  test('with multiple local secondary indexes with different sort keys', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      localSecondaryIndexes: [
        {
          indexName: 'lsi1',
          sortKey: { name: 'lsi-sk-1', type: AttributeType.STRING },
        },
        {
          indexName: 'lsi2',
          sortKey: { name: 'lsi-sk-2', type: AttributeType.NUMBER },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
        { AttributeName: 'lsi-sk-1', AttributeType: 'S' },
        { AttributeName: 'lsi-sk-2', AttributeType: 'N' },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi1',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk-1', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'lsi2',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk-2', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
    });
  });

  test('with multiple local secondary indexes with the same sort keys', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      localSecondaryIndexes: [
        {
          indexName: 'lsi1',
          sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
        },
        {
          indexName: 'lsi2',
          sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
        { AttributeName: 'lsi-sk', AttributeType: 'S' },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi1',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
        {
          IndexName: 'lsi2',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
    });
  });

  test('with global secondary index and local secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
        },
      ],
      localSecondaryIndexes: [
        {
          indexName: 'lsi',
          sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk', AttributeType: 'S' },
        { AttributeName: 'lsi-sk', AttributeType: 'S' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
            },
          ],
        },
      ],
    });
  });

  test('with global secondary index read capacity', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(10),
        writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
      }),
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
          readCapacity: Capacity.fixed(15),
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      WriteProvisionedThroughputSettings: {
        WriteCapacityAutoScalingSettings: {
          MinCapacity: 1,
          MaxCapacity: 10,
          TargetTrackingScalingPolicyConfiguration: {
            TargetValue: 70,
          },
        },
      },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
          WriteProvisionedThroughputSettings: {
            WriteCapacityAutoScalingSettings: {
              MinCapacity: 1,
              MaxCapacity: 10,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 15,
              },
            },
          ],
        },
      ],
    });
  });

  test('with global secondary index without read capacity inherits from table when billing mode is provisioned', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(10),
        writeCapacity: Capacity.autoscaled({ maxCapacity: 10 }),
      }),
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      WriteProvisionedThroughputSettings: {
        WriteCapacityAutoScalingSettings: {
          MinCapacity: 1,
          MaxCapacity: 10,
          TargetTrackingScalingPolicyConfiguration: {
            TargetValue: 70,
          },
        },
      },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'ALL',
          },
          WriteProvisionedThroughputSettings: {
            WriteCapacityAutoScalingSettings: {
              MinCapacity: 1,
              MaxCapacity: 10,
              TargetTrackingScalingPolicyConfiguration: {
                TargetValue: 70,
              },
            },
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
              ReadProvisionedThroughputSettings: {
                ReadCapacityUnits: 10,
              },
            },
          ],
        },
      ],
    });
  });

  test('with global secondary index and KEYS_ONLY projection type', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
          projectionType: ProjectionType.KEYS_ONLY,
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk', AttributeType: 'S' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'KEYS_ONLY',
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
            },
          ],
        },
      ],
    });
  });

  test('with global secondary index and INCLUDE projection type', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      globalSecondaryIndexes: [
        {
          indexName: 'gsi',
          partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
          projectionType: ProjectionType.INCLUDE,
          nonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'gsi-pk', AttributeType: 'S' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'gsi',
          KeySchema: [
            { AttributeName: 'gsi-pk', KeyType: 'HASH' },
          ],
          Projection: {
            ProjectionType: 'INCLUDE',
            NonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
          },
        },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          GlobalSecondaryIndexes: [
            {
              IndexName: 'gsi',
            },
          ],
        },
      ],
    });
  });

  test('with local secondary index and KEYS_ONLY projection type', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      localSecondaryIndexes: [
        {
          indexName: 'lsi',
          sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
          projectionType: ProjectionType.KEYS_ONLY,
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
        { AttributeName: 'lsi-sk', AttributeType: 'S' },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'KEYS_ONLY',
          },
        },
      ],
    });
  });

  test('with local secondary index and INCLUDE projection type', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      localSecondaryIndexes: [
        {
          indexName: 'lsi',
          sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
          projectionType: ProjectionType.INCLUDE,
          nonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
        { AttributeName: 'lsi-sk', AttributeType: 'S' },
      ],
      LocalSecondaryIndexes: [
        {
          IndexName: 'lsi',
          KeySchema: [
            { AttributeName: 'pk', KeyType: 'HASH' },
            { AttributeName: 'lsi-sk', KeyType: 'RANGE' },
          ],
          Projection: {
            ProjectionType: 'INCLUDE',
            NonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
          },
        },
      ],
    });
  });

  test('throws for duplicate global secondary index names', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'gsi-pk-1', type: AttributeType.STRING },
          },
          {
            indexName: 'gsi',
            partitionKey: { name: 'gsi-pk-2', type: AttributeType.STRING },
          },
        ],
      });
    }).toThrow('Duplicate secondary index name, gsi, is not allowed');
  });

  test('throws for duplicate local secondary index names', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'lsi-sk-1', type: AttributeType.STRING },
          },
          {
            indexName: 'lsi',
            sortKey: { name: 'lsi-sk-2', type: AttributeType.STRING },
          },
        ],
      });
    }).toThrow('Duplicate secondary index name, lsi, is not allowed');
  });

  test('throws for duplicate index name in global secondary index and local secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'secondary-index',
            partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
          },
        ],
        localSecondaryIndexes: [
          {
            indexName: 'secondary-index',
            sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
          },
        ],
      });
    }).toThrow('Duplicate secondary index name, secondary-index, is not allowed');
  });

  test('throws if attribute definition is re-defined in global secondary indexes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi1',
            partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
          },
          {
            indexName: 'gsi2',
            partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
          },
        ],
      });
    }).toThrow('Unable to specify gsi-pk as N because it was already defined as S');
  });

  test('throws if attribute definition is re-defined in local secondary indexes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi1',
            sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
          },
          {
            indexName: 'lsi2',
            sortKey: { name: 'lsi-sk', type: AttributeType.NUMBER },
          },
        ],
      });
    }).toThrow('Unable to specify lsi-sk as N because it was already defined as S');
  });

  test('throws if attribute definition is re-defined across global secondary index and local secondary index', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'key', type: AttributeType.STRING },
          },
        ],
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'key', type: AttributeType.NUMBER },
          },
        ],
      });
    }).toThrow('Unable to specify key as N because it was already defined as S');
  });

  test('throws if attribute definition is re-defined across global secondary index and global table', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'pk', type: AttributeType.NUMBER },
          },
        ],
      });
    }).toThrow('Unable to specify pk as N because it was already defined as S');
  });

  test('throws if attribute definition is re-defined across local secondary index and global table', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'key', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'key', type: AttributeType.NUMBER },
          },
        ],
      });
    }).toThrow('Unable to specify key as N because it was already defined as S');
  });

  test('throws if global secondary index has read capacity when billing mode is PAY_PER_REQUEST', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'pk', type: AttributeType.STRING },
            readCapacity: Capacity.fixed(10),
          },
        ],
      });
    }).toThrow("You cannot configure 'readCapacity' or 'writeCapacity' on a global secondary index when the billing mode is PAY_PER_REQUEST");
  });

  test('throws if global secondary index has write capacity when billing mode is PAY_PER_REQUEST', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'pk', type: AttributeType.STRING },
            writeCapacity: Capacity.autoscaled({ minCapacity: 1, maxCapacity: 10 }),
          },
        ],
      });
    }).toThrow("You cannot configure 'readCapacity' or 'writeCapacity' on a global secondary index when the billing mode is PAY_PER_REQUEST");
  });

  test('throws if global secondary index count is greater than 20', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    const globalSecondaryIndexes: GlobalSecondaryIndexPropsV2[] = [];
    for (let count = 0; count <= 20; count++) {
      globalSecondaryIndexes.push({
        indexName: `gsi${count}`,
        partitionKey: { name: 'gsi-pk', type: AttributeType.NUMBER },
      });
    }

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes,
      });
    }).toThrow('You may not provide more than 20 global secondary indexes');
  });

  test('throws if local secondary index count is greater than 5', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    const localSecondaryIndexes: LocalSecondaryIndexProps[] = [];
    for (let count = 0; count <= 5; count++) {
      localSecondaryIndexes.push({
        indexName: `lsi${count}`,
        sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
      });
    }

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes,
      });
    }).toThrow('You may not provide more than 5 local secondary indexes');
  });

  test('throws if global secondary index has INCLUDE projection type and no non-key attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
            projectionType: ProjectionType.INCLUDE,
          },
        ],
      });
    }).toThrow('Non-key attributes should be specified when using INCLUDE projection type');
  });

  test('throws if global secondary index has ALL projection type and non-key attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
            projectionType: ProjectionType.ALL,
            nonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
          },
        ],
      });
    }).toThrow('Non-key attributes should not be specified when not using INCLUDE projection type');
  });

  test('throws if global secondary index has KEYS_ONLY projection type and non-key attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        globalSecondaryIndexes: [
          {
            indexName: 'gsi',
            partitionKey: { name: 'gsi-pk', type: AttributeType.STRING },
            projectionType: ProjectionType.KEYS_ONLY,
            nonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
          },
        ],
      });
    }).toThrow('Non-key attributes should not be specified when not using INCLUDE projection type');
  });

  test('throws if local secondary index has INCLUDE projection type and no non-key attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
            projectionType: ProjectionType.INCLUDE,
          },
        ],
      });
    }).toThrow('Non-key attributes should be specified when using INCLUDE projection type');
  });

  test('throws if local secondary index has ALL projection type and non-key attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
            projectionType: ProjectionType.ALL,
            nonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
          },
        ],
      });
    }).toThrow('Non-key attributes should not be specified when not using INCLUDE projection type');
  });

  test('throws if local secondary index has KEYS_ONLY projection type and non-key attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        sortKey: { name: 'sk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'lsi-sk', type: AttributeType.STRING },
            projectionType: ProjectionType.KEYS_ONLY,
            nonKeyAttributes: ['nonKeyAttr1', 'nonKeyAttr2'],
          },
        ],
      });
    }).toThrow('Non-key attributes should not be specified when not using INCLUDE projection type');
  });

  test('throws if local secondary index is specified without sort key', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack');

    // WHEN / THEN
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        localSecondaryIndexes: [
          {
            indexName: 'lsi',
            sortKey: { name: 'sk', type: AttributeType.NUMBER },
          },
        ],
      });
    }).toThrow('The table must have a sort key in order to add a local secondary index');
  });
});

describe('imports', () => {
  test('can import a table by name', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2', account: '123456789012' } });

    // WHEN
    const table = TableV2.fromTableName(stack, 'Table', 'my-table');

    // THEN
    expect(table.tableName).toEqual('my-table');
    expect(stack.resolve(table.tableArn)).toEqual({
      'Fn::Join': [
        '',
        [
          'arn:',
          {
            Ref: 'AWS::Partition',
          },
          ':dynamodb:us-west-2:123456789012:table/my-table',
        ],
      ],
    });
  });

  test('can import a table by arn', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2', account: '123456789012' } });

    // WHEN
    const table = TableV2.fromTableArn(stack, 'Table', 'arn:aws:dynamodb:us-east-2:123456789012:table/my-table');

    // THEN
    expect(table.tableArn).toEqual('arn:aws:dynamodb:us-east-2:123456789012:table/my-table');
    expect(table.tableName).toEqual('my-table');
  });

  test('can import a table with attributes', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2', account: '123456789012' } });
    const tableKey = new Key(stack, 'Key');

    // WHEN
    const table = TableV2.fromTableAttributes(stack, 'Table', {
      tableArn: 'arn:aws:dynamodb:us-east-2:123456789012:table/my-table',
      tableStreamArn: 'arn:aws:dynamodb:us-east-2:123456789012:table/my-table/stream/*',
      tableId: 'a123b456-01ab-23cd-123a-111222aaabbb',
      encryptionKey: tableKey,
    });

    // THEN
    expect(table.tableStreamArn).toEqual('arn:aws:dynamodb:us-east-2:123456789012:table/my-table/stream/*');
    expect(table.encryptionKey?.keyArn).toEqual(tableKey.keyArn);
    expect(table.tableId).toEqual('a123b456-01ab-23cd-123a-111222aaabbb');
  });

  test('throws if name or arn are not provided', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN / THEN
    expect(() => {
      TableV2.fromTableAttributes(stack, 'Table', {
        tableStreamArn: 'arn:aws:dynamodb:us-east-2:123456789012:table/my-table/stream/*',
      });
    }).toThrow('At least one of `tableArn` or `tableName` must be provided');
  });

  test('throws if name and arn are both provided', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN / THEN
    expect(() => {
      TableV2.fromTableAttributes(stack, 'Table', {
        tableName: 'my-table',
        tableArn: 'arn:aws:dynamodb:us-east-2:123456789012:table/my-table',
      });
    }).toThrow('Only one of `tableArn` or `tableName` can be provided, but not both');
  });

  test('throws for invalid arn format', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN / THEN
    expect(() => {
      TableV2.fromTableAttributes(stack, 'Table', {
        tableArn: 'arn:aws:dynamodb:us-east-2:123456789012:table/',
      });
    }).toThrow('Table ARN must be of the form: arn:<partition>:dynamodb:<region>:<account>:table/<table-name>');
  });
});

test('Resource policy test', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack');

  const doc = new PolicyDocument({
    statements: [
      new PolicyStatement({
        actions: ['dynamodb:GetItem'],
        principals: [new ArnPrincipal('arn:aws:iam::111122223333:user/foobar')],
        resources: ['*'],
      }),
    ],
  });

  // WHEN
  new TableV2(stack, 'Table', {
    partitionKey: { name: 'metric', type: AttributeType.STRING },
    resourcePolicy: doc,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: {
          Ref: 'AWS::Region',
        },
        ResourcePolicy: {
          PolicyDocument: {
            Statement: [
              {
                Action: 'dynamodb:GetItem',
                Effect: 'Allow',
                Principal: {
                  AWS: 'arn:aws:iam::111122223333:user/foobar',
                },
                Resource: '*',
              },
            ],
            Version: '2012-10-17',
          },
        },
      },
    ],
  });
});

test('Resource policy is scoped to primary region only when resourcePolicyPerReplica feature flag is enabled', () => {
  // GIVEN
  const app = new App({
    postCliContext: {
      '@aws-cdk/aws-dynamodb:resourcePolicyPerReplica': true,
    },
  });
  const stack = new Stack(app, 'Stack', { env: { region: 'eu-west-1' } });

  const doc = new PolicyDocument({
    statements: [
      new PolicyStatement({
        actions: ['dynamodb:*'],
        principals: [new iam.AccountRootPrincipal()],
        resources: ['*'],
      }),
    ],
  });

  // WHEN
  new TableV2(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    resourcePolicy: doc,
    replicas: [{
      region: 'eu-west-2',
    }],
  });

  // THEN
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: Match.arrayWith([
      Match.objectLike({
        Region: 'eu-west-2',
        ResourcePolicy: Match.absent(),
      }),
      Match.objectLike({
        Region: 'eu-west-1',
        ResourcePolicy: Match.objectLike({
          PolicyDocument: Match.objectLike({
            Statement: Match.arrayWith([
              Match.objectLike({ Action: 'dynamodb:*' }),
            ]),
          }),
        }),
      }),
    ]),
  });
});

test('Warm Throughput test on-demand', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack', { env: { region: 'eu-west-1' } });

  // WHEN
  new TableV2(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    warmThroughput: {
      readUnitsPerSecond: 13000,
      writeUnitsPerSecond: 5000,
    },
    replicas: [{
      region: 'us-west-2',
    }],
    globalSecondaryIndexes: [{
      indexName: 'my-index-1',
      partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
      warmThroughput: {
        readUnitsPerSecond: 15000,
        writeUnitsPerSecond: 6000,
      },
    },
    {
      indexName: 'my-index-2',
      partitionKey: { name: 'gsi2pk', type: AttributeType.STRING },
    }],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    KeySchema: [
      { AttributeName: 'id', KeyType: 'HASH' },
    ],
    AttributeDefinitions: [
      { AttributeName: 'id', AttributeType: 'S' },
      { AttributeName: 'gsi1pk', AttributeType: 'S' },
      { AttributeName: 'gsi2pk', AttributeType: 'S' },
    ],
    WarmThroughput: {
      ReadUnitsPerSecond: 13000,
      WriteUnitsPerSecond: 5000,
    },
    GlobalSecondaryIndexes: [
      {
        IndexName: 'my-index-1',
        KeySchema: [
          { AttributeName: 'gsi1pk', KeyType: 'HASH' },
        ],
        Projection: { ProjectionType: 'ALL' },
        WarmThroughput: {
          ReadUnitsPerSecond: 15000,
          WriteUnitsPerSecond: 6000,
        },
      },
      {
        IndexName: 'my-index-2',
        KeySchema: [
          { AttributeName: 'gsi2pk', KeyType: 'HASH' },
        ],
        Projection: { ProjectionType: 'ALL' },
      },
    ],
  });
});

describe('MRSC global tables', () => {
  test('with witness region', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{ region: 'us-east-1' }],
      witnessRegion: 'us-east-2',
      multiRegionConsistency: MultiRegionConsistency.STRONG,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        { Region: 'us-east-1' },
        { Region: 'us-west-2' },
      ],
      GlobalTableWitnesses: [
        { Region: 'us-east-2' },
      ],
    });
  });

  test('without witness region should not have GlobalTableWitnesses property', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{ region: 'us-east-1' }],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      Replicas: [
        { Region: 'us-east-1' },
        { Region: 'us-west-2' },
      ],
    });
    // Verify that GlobalTableWitnesses is not present in the template
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable',
      Match.not(Match.objectLike({
        GlobalTableWitnesses: Match.anyValue(),
      })),
    );
  });

  test('with witness region and strong consistency requirements', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      replicas: [{ region: 'us-east-1' }],
      witnessRegion: 'us-east-2',
      multiRegionConsistency: MultiRegionConsistency.STRONG,
      billing: Billing.provisioned({
        readCapacity: Capacity.fixed(10),
        writeCapacity: Capacity.autoscaled({ minCapacity: 10, maxCapacity: 100 }),
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      KeySchema: [
        { AttributeName: 'pk', KeyType: 'HASH' },
        { AttributeName: 'sk', KeyType: 'RANGE' },
      ],
      AttributeDefinitions: [
        { AttributeName: 'pk', AttributeType: 'S' },
        { AttributeName: 'sk', AttributeType: 'S' },
      ],
      BillingMode: 'PROVISIONED',
      Replicas: [
        {
          Region: 'us-east-1',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
        },
        {
          Region: 'us-west-2',
          ReadProvisionedThroughputSettings: {
            ReadCapacityUnits: 10,
          },
        },
      ],
      GlobalTableWitnesses: [
        { Region: 'us-east-2' },
      ],
    });
  });
});

describe('MRSC global tables validation', () => {
  test('throws when witness region is used with eventual consistency', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN / THEN - Error should be thrown during construction
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        replicas: [{ region: 'us-east-1' }],
        witnessRegion: 'us-east-2',
        // multiRegionConsistency defaults to EVENTUAL
      });
    }).toThrow('Witness region cannot be specified for a Multi-Region Eventual Consistency (MREC) Global Table - Witness regions are only supported for Multi-Region Strong Consistency (MRSC) Global Tables.');
  });

  test('validates regions are in same region set for STRONG consistency', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN / THEN - Error should be thrown during construction
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        replicas: [{ region: 'eu-west-1' }],
        witnessRegion: 'us-east-2',
        multiRegionConsistency: MultiRegionConsistency.STRONG,
      });
    }).toThrow("Region 'eu-west-1' is not in the same region set (US) as the primary region 'us-west-2'. All regions must be within the same region set for MRSC global tables with STRONG consistency. Supported US regions: us-east-1, us-east-2, us-west-2");
  });

  test('validates exactly 2 replicas with witness for STRONG consistency', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'eu-west-1' } });

    // WHEN / THEN - Error should be thrown during construction
    expect(() => {
      new TableV2(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        replicas: [{ region: 'eu-west-2' }, { region: 'eu-west-3' }], // Too many replicas
        witnessRegion: 'eu-central-1', // Use same region set
        multiRegionConsistency: MultiRegionConsistency.STRONG,
      });
    }).toThrow("MRSC global table with witness region must have exactly 2 replicas (including primary), but found 3. Current configuration: primary region 'eu-west-1', replica regions [eu-west-2, eu-west-3], witness region 'eu-central-1'");
  });

  test('allows valid STRONG consistency configuration with witness', () => {
    // GIVEN
    const stack = new Stack(undefined, 'Stack', { env: { region: 'us-west-2' } });

    // WHEN
    new TableV2(stack, 'Table', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      replicas: [{ region: 'us-east-1' }],
      witnessRegion: 'us-east-2',
      multiRegionConsistency: MultiRegionConsistency.STRONG,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
      MultiRegionConsistency: 'STRONG',
      Replicas: [
        { Region: 'us-east-1' },
        { Region: 'us-west-2' },
      ],
      GlobalTableWitnesses: [
        { Region: 'us-east-2' },
      ],
    });
  });
});

test('TableV2 addToResourcePolicy works with wildcard resources', () => {
  // GIVEN
  const stack = new Stack();

  // WHEN
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  table.addToResourcePolicy(new iam.PolicyStatement({
    actions: ['dynamodb:GetItem', 'dynamodb:PutItem'],
    principals: [new iam.AccountRootPrincipal()],
    resources: ['*'], // Wildcard avoids circular dependency - same pattern as KMS
  }));

  // THEN
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: {
          Ref: 'AWS::Region',
        },
        ResourcePolicy: {
          PolicyDocument: {
            Version: '2012-10-17',
            Statement: [
              {
                Effect: 'Allow',
                Principal: {
                  AWS: Match.anyValue(),
                },
                Action: ['dynamodb:GetItem', 'dynamodb:PutItem'],
                Resource: '*',
              },
            ],
          },
        },
      },
    ],
  });
});

test('TableV2 addToResourcePolicy allows scoped ARN resources when table has explicit name', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack');

  // WHEN - Create table with explicit name (enables scoped resource policies)
  const table = new TableV2(stack, 'Table', {
    tableName: 'my-explicit-table-name', // Explicit name enables scoped ARN construction
    partitionKey: { name: 'id', type: AttributeType.STRING },
  });

  // With explicit table name, we can use scoped resources without circular dependency
  table.addToResourcePolicy(new iam.PolicyStatement({
    actions: ['dynamodb:GetItem', 'dynamodb:Query'],
    principals: [new iam.AccountRootPrincipal()],
    resources: [
      // This works because table name is known at synthesis time
      Fn.sub('arn:aws:dynamodb:${AWS::Region}:${AWS::AccountId}:table/my-explicit-table-name'),
    ],
  }));

  // THEN
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: {
          Ref: 'AWS::Region',
        },
        ResourcePolicy: {
          PolicyDocument: {
            Version: '2012-10-17',
            Statement: [
              {
                Effect: 'Allow',
                Principal: {
                  AWS: Match.anyValue(),
                },
                Action: ['dynamodb:GetItem', 'dynamodb:Query'],
                Resource: {
                  'Fn::Sub': 'arn:aws:dynamodb:${AWS::Region}:${AWS::AccountId}:table/my-explicit-table-name',
                },
              },
            ],
          },
        },
      },
    ],
  });
});

test('Contributor Insights Specification - tableV2', () => {
  const stack = new Stack();

  new TableV2(stack, 'TableV2', {
    partitionKey: { name: 'hashKey', type: AttributeType.STRING },
    sortKey: { name: 'sortKey', type: AttributeType.NUMBER },
    contributorInsightsSpecification: {
      enabled: true,
      mode: ContributorInsightsMode.ACCESSED_AND_THROTTLED_KEYS,
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ContributorInsightsSpecification: {
            Enabled: true,
            Mode: 'ACCESSED_AND_THROTTLED_KEYS',
          },
        },
      ],
    },
  );
});

test('Contributor Insights Specification - tableV2 - without mode', () => {
  const stack = new Stack();

  new TableV2(stack, 'TableV2', {
    partitionKey: { name: 'hashKey', type: AttributeType.STRING },
    sortKey: { name: 'sortKey', type: AttributeType.NUMBER },
    contributorInsightsSpecification: {
      enabled: true,
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      Replicas: [
        {
          Region: {
            Ref: 'AWS::Region',
          },
          ContributorInsightsSpecification: {
            Enabled: true,
          },
        },
      ],
    },
  );
});

test('Contributor Insights Specification - index', () => {
  const stack = new Stack(undefined, 'Stack', { env: { region: 'eu-west-1' } });

  new TableV2(stack, 'TableV2', {
    partitionKey: { name: 'hashKey', type: AttributeType.STRING },
    sortKey: { name: 'sortKey', type: AttributeType.NUMBER },
    globalSecondaryIndexes: [
      {
        indexName: 'gsi1',
        partitionKey: { name: 'gsiHashKey', type: AttributeType.STRING },
      },
    ],
    contributorInsightsSpecification: {
      enabled: true,
      mode: ContributorInsightsMode.ACCESSED_AND_THROTTLED_KEYS,
    },
    replicas: [
      {
        region: 'eu-west-2',
        contributorInsightsSpecification: {
          enabled: false,
        },
        globalSecondaryIndexOptions: {
          gsi1: {
            contributorInsightsSpecification: {
              enabled: true,
              mode: ContributorInsightsMode.THROTTLED_KEYS,
            },
          },
        },
      },
    ],
  });

  Template.fromStack(stack).hasResource('AWS::DynamoDB::GlobalTable', {
    Properties: Match.objectLike({
      Replicas: Match.arrayWith([
        Match.objectLike({
          Region: 'eu-west-2',
          ContributorInsightsSpecification: {
            Enabled: false,
          },
          GlobalSecondaryIndexes: Match.arrayWith([
            Match.objectLike({
              IndexName: 'gsi1',
              ContributorInsightsSpecification: {
                Enabled: true,
                Mode: 'THROTTLED_KEYS',
              },
            }),
          ]),
        }),
        Match.objectLike({
          Region: 'eu-west-1',
          ContributorInsightsSpecification: {
            Enabled: true,
            Mode: 'ACCESSED_AND_THROTTLED_KEYS',
          },
          GlobalSecondaryIndexes: Match.arrayWith([
            Match.objectLike({
              IndexName: 'gsi1',
              ContributorInsightsSpecification: {
                Enabled: true,
                Mode: 'ACCESSED_AND_THROTTLED_KEYS',
              },
            }),
          ]),
        }),
      ]),
    }),
  });
});

test('ContributorInsightsSpecification && ContributorInsights - v2', () => {
  const stack = new Stack();

  expect(() => {
    new TableV2(stack, 'Tablev2', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
      sortKey: { name: 'sk', type: AttributeType.STRING },
      contributorInsights: true,
      contributorInsightsSpecification: {
        enabled: true,
        mode: ContributorInsightsMode.ACCESSED_AND_THROTTLED_KEYS,
      },
    });

    Template.fromStack(stack);
  }).toThrow('`contributorInsightsSpecification` and `contributorInsights` are set. Use `contributorInsightsSpecification` only.');
});

test('grantMultiAccountReplicationTo adds required resource policy statements', () => {
  const stack = new Stack(undefined, 'Stack', { env: { account: '111111111111', region: 'us-east-2' } });
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  table.grants.multiAccountReplicationTo('arn:aws:dynamodb:us-east-1:222222222222:table/RemoteTable');

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        ResourcePolicy: {
          PolicyDocument: {
            Statement: [
              {
                Sid: 'AllowMultiAccountReplicaAssociation222222222222',
                Effect: 'Allow',
                Action: 'dynamodb:AssociateTableReplica',
                Resource: '*',
                Principal: {
                  AWS: {
                    'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::222222222222:root']],
                  },
                },
              },
              {
                Sid: 'AllowReplicationServiceReadWrite222222222222',
                Effect: 'Allow',
                Action: [
                  'dynamodb:ReadDataForReplication',
                  'dynamodb:WriteDataForReplication',
                  'dynamodb:ReplicateSettings',
                ],
                Resource: '*',
                Principal: {
                  Service: 'replication.dynamodb.amazonaws.com',
                },
                Condition: {
                  StringEquals: {
                    'aws:SourceAccount': ['111111111111', '222222222222'],
                  },
                },
              },
            ],
          },
        },
      },
    ],
  });
});

test('grantMultiAccountReplicationTo grants KMS permissions for encrypted tables', () => {
  const stack = new Stack(undefined, 'Stack', { env: { account: '111111111111', region: 'us-east-2' } });
  const key = new Key(stack, 'Key');
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
    encryption: TableEncryptionV2.customerManagedKey(key),
  });

  table.grants.multiAccountReplicationTo('arn:aws:dynamodb:us-east-1:222222222222:table/RemoteTable');

  Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
    KeyPolicy: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: [
            'kms:Decrypt',
            'kms:DescribeKey',
            'kms:Encrypt',
            'kms:ReEncrypt*',
            'kms:GenerateDataKey*',
          ],
          Effect: 'Allow',
          Principal: {
            Service: 'replication.dynamodb.amazonaws.com',
          },
        }),
      ]),
    },
  });
});

test('grantMultiAccountReplicationFrom adds required resource policy statements', () => {
  const stack = new Stack(undefined, 'Stack', { env: { account: '222222222222', region: 'us-east-1' } });
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  table.grants.multiAccountReplicationFrom('arn:aws:dynamodb:us-east-2:111111111111:table/SourceTable');

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        ResourcePolicy: {
          PolicyDocument: {
            Statement: [
              {
                Sid: 'AllowReplicationService',
                Effect: 'Allow',
                Action: [
                  'dynamodb:ReadDataForReplication',
                  'dynamodb:WriteDataForReplication',
                  'dynamodb:ReplicateSettings',
                ],
                Resource: '*',
                Principal: {
                  Service: 'replication.dynamodb.amazonaws.com',
                },
                Condition: {
                  StringEquals: {
                    'aws:SourceAccount': ['222222222222', '111111111111'],
                  },
                },
              },
            ],
          },
        },
      },
    ],
  });
});

test('grantMultiAccountReplicationFrom grants KMS permissions for encrypted tables', () => {
  const stack = new Stack(undefined, 'Stack', { env: { account: '222222222222', region: 'us-east-1' } });
  const key = new Key(stack, 'Key');
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
    encryption: TableEncryptionV2.customerManagedKey(key),
  });

  table.grants.multiAccountReplicationFrom('arn:aws:dynamodb:us-east-2:111111111111:table/SourceTable');

  Template.fromStack(stack).hasResourceProperties('AWS::KMS::Key', {
    KeyPolicy: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: [
            'kms:Decrypt',
            'kms:DescribeKey',
            'kms:Encrypt',
            'kms:ReEncrypt*',
            'kms:GenerateDataKey*',
          ],
          Effect: 'Allow',
          Principal: {
            Service: 'replication.dynamodb.amazonaws.com',
          },
        }),
      ]),
    },
  });
});

test('TableV2MultiAccountReplica creates replica with permissions', () => {
  const app = new App();
  const sourceStack = new Stack(app, 'SourceStack', { env: { account: '111111111111', region: 'us-east-2' } });
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  const sourceTable = new TableV2(sourceStack, 'SourceTable', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
    replicaSourceTable: sourceTable,
    globalTableSettingsReplicationMode: GlobalTableSettingsReplicationMode.ALL,
  });

  // Grants are automatically set up - no manual call needed

  // Source table should have resource policy with account principal
  Template.fromStack(sourceStack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        ResourcePolicy: {
          PolicyDocument: {
            Statement: Match.arrayWith([
              Match.objectLike({
                Action: 'dynamodb:AssociateTableReplica',
              }),
              Match.objectLike({
                Action: [
                  'dynamodb:ReadDataForReplication',
                  'dynamodb:WriteDataForReplication',
                  'dynamodb:ReplicateSettings',
                ],
              }),
            ]),
          },
        },
      },
    ],
  });

  // Replica table should be created with source ARN and replication mode in replicas
  Template.fromStack(replicaStack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: Stack.of(replicaStack).region,
        GlobalTableSettingsReplicationMode: 'ENABLED',
        ResourcePolicy: {
          PolicyDocument: {
            Statement: Match.arrayWith([
              Match.objectLike({
                Sid: 'AllowReplicationService',
                Action: [
                  'dynamodb:ReadDataForReplication',
                  'dynamodb:WriteDataForReplication',
                  'dynamodb:ReplicateSettings',
                ],
                Principal: {
                  Service: 'replication.dynamodb.amazonaws.com',
                },
              }),
            ]),
          },
        },
      },
    ],
  });

  // Verify replica has source ARN reference (using Fn::Join for cross-stack reference)
  const replicaTemplate = Template.fromStack(replicaStack);
  const resources = replicaTemplate.findResources('AWS::DynamoDB::GlobalTable');
  const replicaTable = Object.values(resources)[0];
  expect(replicaTable.Properties.GlobalTableSourceArn).toBeDefined();
});

test('TableV2MultiAccountReplica throws when same account', () => {
  const app = new App();
  const stack = new Stack(app, 'Stack', { env: { account: '111111111111', region: 'us-east-2' } });

  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    new TableV2MultiAccountReplica(stack, 'Replica', {
      replicaSourceTable: table,
    });
  }).toThrow('Multi-account replica must be in a different account than the source table. For same-account replication, use addReplica() instead.');
});

test('TableV2MultiAccountReplica on imported table does not throw', () => {
  const app = new App();
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  const importedTable = TableV2.fromTableArn(
    replicaStack,
    'ImportedTable',
    'arn:aws:dynamodb:us-east-2:111111111111:table/SourceTable',
  );

  new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
    replicaSourceTable: importedTable,
  });

  // Should issue a warning about missing resource policy
  const warnings = Annotations.fromStack(replicaStack).findWarning('*', Match.stringLikeRegexp('.*imported without a resource policy.*'));
  expect(warnings.length).toBe(1);
  expect(warnings[0].entry.data).toContain('manually configure multi-account replication permissions');
});

test('TableV2MultiAccountReplica works with fromTableArn without key schema', () => {
  const app = new App();
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  // Import using fromTableArn - no key schema needed
  const importedTable = TableV2.fromTableArn(
    replicaStack,
    'ImportedTable',
    'arn:aws:dynamodb:us-east-2:111111111111:table/SourceTable',
  );

  new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
    replicaSourceTable: importedTable,
  });

  // Verify replica is created with globalTableSourceArn
  Template.fromStack(replicaStack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    GlobalTableSourceArn: 'arn:aws:dynamodb:us-east-2:111111111111:table/SourceTable',
  });
});

test('grantMultiAccountReplicationTo validates ARN has account', () => {
  const stack = new Stack(undefined, 'Stack', { env: { account: '111111111111', region: 'us-east-2' } });
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.grants.multiAccountReplicationTo('arn:aws:dynamodb:us-east-1::table/RemoteTable');
  }).toThrow('Invalid table ARN');
});

test('grantMultiAccountReplicationFrom validates ARN has account', () => {
  const stack = new Stack(undefined, 'Stack', { env: { account: '222222222222', region: 'us-east-1' } });
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.grants.multiAccountReplicationFrom('arn:aws:dynamodb:us-east-2::table/SourceTable');
  }).toThrow('Invalid table ARN');
});

test('TableV2MultiAccountReplica throws error when replica is in same region', () => {
  const app = new App();
  const sourceStack = new Stack(app, 'SourceStack', { env: { account: '111111111111', region: 'us-east-1' } });
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  const table = new TableV2(sourceStack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
      replicaSourceTable: table,
    });
  }).toThrow(/Multi-account replica must be in a different region/);
});

test('TableV2MultiAccountReplica with all optional parameters', () => {
  const app = new App();
  const sourceStack = new Stack(app, 'SourceStack', { env: { account: '111111111111', region: 'us-east-2' } });
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  const sourceTable = new TableV2(sourceStack, 'SourceTable', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  const kinesisStream = {
    streamArn: 'arn:aws:kinesis:us-east-1:222222222222:stream/MyStream',
  } as any;

  new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
    replicaSourceTable: sourceTable,
    globalTableSettingsReplicationMode: GlobalTableSettingsReplicationMode.ALL,
    deletionProtection: true,
    tableClass: TableClass.STANDARD_INFREQUENT_ACCESS,
    kinesisStream,
    contributorInsightsSpecification: { enabled: true },
    pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
    tags: [{ key: 'Environment', value: 'Test' }],
    removalPolicy: RemovalPolicy.DESTROY,
  });

  // Grants are automatically set up - no manual call needed

  Template.fromStack(replicaStack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      Match.objectLike({
        Region: 'us-east-1',
        GlobalTableSettingsReplicationMode: 'ENABLED',
        DeletionProtectionEnabled: true,
        TableClass: 'STANDARD_INFREQUENT_ACCESS',
        KinesisStreamSpecification: {
          StreamArn: 'arn:aws:kinesis:us-east-1:222222222222:stream/MyStream',
        },
        ContributorInsightsSpecification: { Enabled: true },
        PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true },
        Tags: [{ Key: 'Environment', Value: 'Test' }],
      }),
    ],
  });
});

test('TableV2MultiAccountReplica throws when replicaSourceTable is missing', () => {
  const app = new App();
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  expect(() => {
    new TableV2MultiAccountReplica(replicaStack, 'Replica', {});
  }).toThrow('replicaSourceTable is required for TableV2MultiAccountReplica');
});

test('TableV2MultiAccountReplica with custom encryption', () => {
  const app = new App();
  const sourceStack = new Stack(app, 'SourceStack', { env: { account: '111111111111', region: 'us-east-2' } });
  const replicaStack = new Stack(app, 'ReplicaStack', { env: { account: '222222222222', region: 'us-east-1' } });

  const sourceTable = new TableV2(sourceStack, 'SourceTable', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  const key = new Key(replicaStack, 'Key');
  new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
    replicaSourceTable: sourceTable,
    encryption: TableEncryptionV2.customerManagedKey(key),
  });

  Template.fromStack(replicaStack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      Match.objectLike({
        SSESpecification: {
          KMSMasterKeyId: {
            'Fn::GetAtt': [Match.stringLikeRegexp('Key'), 'Arn'],
          },
        },
      }),
    ],
  });
});

test('TableV2MultiAccountReplica does not throw when account/region are tokens', () => {
  const app = new App();
  const sourceStack = new Stack(app, 'SourceStack');
  const replicaStack = new Stack(app, 'ReplicaStack');

  const sourceTable = new TableV2(sourceStack, 'SourceTable', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  // Should not throw when accounts/regions are unresolved tokens
  expect(() => {
    new TableV2MultiAccountReplica(replicaStack, 'ReplicaTable', {
      replicaSourceTable: sourceTable,
    });
  }).not.toThrow();
});

test('can add GSI with compound partition keys', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'GSI1',
    partitionKeys: [
      { name: 'gsi1pk1', type: AttributeType.STRING },
      { name: 'gsi1pk2', type: AttributeType.NUMBER },
    ],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    AttributeDefinitions: [
      { AttributeName: 'pk', AttributeType: 'S' },
      { AttributeName: 'gsi1pk1', AttributeType: 'S' },
      { AttributeName: 'gsi1pk2', AttributeType: 'N' },
    ],
    GlobalSecondaryIndexes: [
      {
        IndexName: 'GSI1',
        KeySchema: [
          { AttributeName: 'gsi1pk1', KeyType: 'HASH' },
          { AttributeName: 'gsi1pk2', KeyType: 'HASH' },
        ],
      },
    ],
  });
});

test('can add GSI with multi-attribute sort keys', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'GSI1',
    partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
    sortKeys: [
      { name: 'gsi1sk1', type: AttributeType.STRING },
      { name: 'gsi1sk2', type: AttributeType.NUMBER },
    ],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    GlobalSecondaryIndexes: [
      {
        IndexName: 'GSI1',
        KeySchema: [
          { AttributeName: 'gsi1pk', KeyType: 'HASH' },
          { AttributeName: 'gsi1sk1', KeyType: 'RANGE' },
          { AttributeName: 'gsi1sk2', KeyType: 'RANGE' },
        ],
      },
    ],
  });
});

test('throws when both partitionKey and partitionKeys defined', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
      partitionKeys: [{ name: 'gsi1pk2', type: AttributeType.NUMBER }],
    });
  }).toThrow('Exactly one of \'partitionKey\', \'partitionKeys\' must be specified');
});

test('throws when both sortKey and sortKeys defined', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
      sortKey: { name: 'gsi1sk', type: AttributeType.STRING },
      sortKeys: [{ name: 'gsi1sk2', type: AttributeType.NUMBER }],
    });
  }).toThrow('At most one of \'sortKey\', \'sortKeys\' may be specified');
});
test('throws when more than 4 partition keys', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKeys: [
        { name: 'pk1', type: AttributeType.STRING },
        { name: 'pk2', type: AttributeType.STRING },
        { name: 'pk3', type: AttributeType.STRING },
        { name: 'pk4', type: AttributeType.STRING },
        { name: 'pk5', type: AttributeType.STRING },
      ],
    });
  }).toThrow('Maximum of 4 partition keys allowed');
});

test('throws when more than 4 sort keys', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
      sortKeys: [
        { name: 'sk1', type: AttributeType.STRING },
        { name: 'sk2', type: AttributeType.STRING },
        { name: 'sk3', type: AttributeType.STRING },
        { name: 'sk4', type: AttributeType.STRING },
        { name: 'sk5', type: AttributeType.STRING },
      ],
    });
  }).toThrow('Maximum of 4 sort keys allowed');
});

test('throws when no partition key specified', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  expect(() => {
    table.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      sortKey: { name: 'sk', type: AttributeType.STRING },
    });
  }).toThrow('Exactly one of \'partitionKey\', \'partitionKeys\' must be specified');
});

test('can add GSI with both multi-attribute partition and sort keys', () => {
  const stack = new Stack();
  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'GSI1',
    partitionKeys: [
      { name: 'gsi1pk1', type: AttributeType.STRING },
      { name: 'gsi1pk2', type: AttributeType.NUMBER },
    ],
    sortKeys: [
      { name: 'gsi1sk1', type: AttributeType.STRING },
      { name: 'gsi1sk2', type: AttributeType.BINARY },
    ],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    GlobalSecondaryIndexes: [
      {
        IndexName: 'GSI1',
        KeySchema: [
          { AttributeName: 'gsi1pk1', KeyType: 'HASH' },
          { AttributeName: 'gsi1pk2', KeyType: 'HASH' },
          { AttributeName: 'gsi1sk1', KeyType: 'RANGE' },
          { AttributeName: 'gsi1sk2', KeyType: 'RANGE' },
        ],
      },
    ],
  });
});

test('stream resource policy on primary table', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack');

  const doc = new PolicyDocument({
    statements: [
      new PolicyStatement({
        actions: ['dynamodb:DescribeStream', 'dynamodb:GetRecords', 'dynamodb:GetShardIterator'],
        principals: [new ArnPrincipal('arn:aws:iam::111122223333:user/foobar')],
        resources: ['*'],
      }),
    ],
  });

  // WHEN
  new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
    dynamoStream: StreamViewType.NEW_AND_OLD_IMAGES,
    streamResourcePolicy: doc,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: {
          Ref: 'AWS::Region',
        },
        ReplicaStreamSpecification: {
          ResourcePolicy: {
            PolicyDocument: {
              Statement: [
                {
                  Action: [
                    'dynamodb:DescribeStream',
                    'dynamodb:GetRecords',
                    'dynamodb:GetShardIterator',
                  ],
                  Effect: 'Allow',
                  Principal: {
                    AWS: 'arn:aws:iam::111122223333:user/foobar',
                  },
                  Resource: '*',
                },
              ],
              Version: '2012-10-17',
            },
          },
        },
      },
    ],
  });
});

test('stream resource policy on replica table', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack', { env: { region: 'us-east-1' } });

  const doc = new PolicyDocument({
    statements: [
      new PolicyStatement({
        actions: ['dynamodb:GetRecords'],
        principals: [new ArnPrincipal('arn:aws:iam::111122223333:user/foobar')],
        resources: ['*'],
      }),
    ],
  });

  // WHEN
  new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
    dynamoStream: StreamViewType.NEW_AND_OLD_IMAGES,
    replicas: [
      {
        region: 'us-west-2',
        streamResourcePolicy: doc,
      },
    ],
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: Match.arrayWith([
      Match.objectLike({
        Region: 'us-west-2',
        ReplicaStreamSpecification: {
          ResourcePolicy: {
            PolicyDocument: {
              Statement: [
                {
                  Action: 'dynamodb:GetRecords',
                  Effect: 'Allow',
                  Principal: {
                    AWS: 'arn:aws:iam::111122223333:user/foobar',
                  },
                  Resource: '*',
                },
              ],
              Version: '2012-10-17',
            },
          },
        },
      }),
      Match.objectLike({
        Region: 'us-east-1',
        ReplicaStreamSpecification: Match.absent(),
      }),
    ]),
  });
});

test('addToStreamResourcePolicy on primary table', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack');

  const table = new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
    dynamoStream: StreamViewType.NEW_AND_OLD_IMAGES,
  });

  // WHEN
  table.addToStreamResourcePolicy(new PolicyStatement({
    actions: ['dynamodb:GetRecords'],
    principals: [new ArnPrincipal('arn:aws:iam::111122223333:user/foobar')],
    resources: ['*'],
  }));

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: {
          Ref: 'AWS::Region',
        },
        ReplicaStreamSpecification: {
          ResourcePolicy: {
            PolicyDocument: {
              Statement: [
                {
                  Action: 'dynamodb:GetRecords',
                  Effect: 'Allow',
                  Principal: {
                    AWS: 'arn:aws:iam::111122223333:user/foobar',
                  },
                  Resource: '*',
                },
              ],
              Version: '2012-10-17',
            },
          },
        },
      },
    ],
  });
});

test('no stream resource policy by default', () => {
  // GIVEN
  const stack = new Stack(undefined, 'Stack');

  // WHEN
  new TableV2(stack, 'Table', {
    partitionKey: { name: 'pk', type: AttributeType.STRING },
    dynamoStream: StreamViewType.NEW_AND_OLD_IMAGES,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::GlobalTable', {
    Replicas: [
      {
        Region: {
          Ref: 'AWS::Region',
        },
        ReplicaStreamSpecification: Match.absent(),
      },
    ],
  });
});
