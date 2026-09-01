import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import type { Construct } from 'constructs';
import { Annotations, Capture, Match, Template } from '../../assertions';
import * as appscaling from '../../aws-applicationautoscaling';
import * as cloudwatch from '../../aws-cloudwatch';
import * as iam from '../../aws-iam';
import * as kinesis from '../../aws-kinesis';
import * as kms from '../../aws-kms';
import * as s3 from '../../aws-s3';
import {
  App,
  ArnFormat,
  Aws,
  CfnDeletionPolicy,
  Duration,
  Fn,
  PhysicalName,
  RemovalPolicy,
  Resource,
  Stack,
  Tags,
} from '../../core';
import * as cr from '../../custom-resources';
import * as cxapi from '../../cx-api';
import type { Attribute, GlobalSecondaryIndexProps, LocalSecondaryIndexProps } from '../lib';
import {
  ApproximateCreationDateTimePrecision,
  AttributeType,
  BillingMode,
  CfnTable,
  ContributorInsightsMode,
  InputCompressionType,
  InputFormat,
  Operation,
  ProjectionType,
  StreamViewType,
  Table,
  TableClass,
  TableEncryption,
  TableGrants,
} from '../lib';
import { ReplicaProvider } from '../lib/replica-provider';

jest.mock('../../custom-resources', () => {
  const autoMock = jest.createMockFromModule('../../custom-resources');
  const { builtInCustomResourceNodeRuntime } = jest.requireActual('../../custom-resources');
  return {
    // @ts-ignore
    ...autoMock,
    builtInCustomResourceNodeRuntime,
  };
});

const tableStreamArn = 'arn:aws:dynamodb:us-east-1:111111111111:table/TableName/stream/StreamLabel';

/* eslint-disable @stylistic/quote-props */

// CDK parameters
const CONSTRUCT_NAME = 'MyTable';

// DynamoDB table parameters
const TABLE_NAME = 'MyTable';
const TABLE_PARTITION_KEY: Attribute = { name: 'hashKey', type: AttributeType.STRING };
const TABLE_SORT_KEY: Attribute = { name: 'sortKey', type: AttributeType.NUMBER };

// DynamoDB global secondary index parameters
const GSI_NAME = 'MyGSI';
const GSI_PARTITION_KEY: Attribute = { name: 'gsiHashKey', type: AttributeType.STRING };
const GSI_PARTITION_KEY_TWO: Attribute = { name: 'gsiHaskKeyTwo', type: AttributeType.NUMBER };
const GSI_SORT_KEY: Attribute = { name: 'gsiSortKey', type: AttributeType.BINARY };
const GSI_SORT_KEY_TWO: Attribute = { name: 'gsiSortKeyTwo', type: AttributeType.STRING };
const GSI_NON_KEY = 'gsiNonKey';
function* GSI_GENERATOR(): Generator<GlobalSecondaryIndexProps, never> {
  let n = 0;
  while (true) {
    const globalSecondaryIndexProps: GlobalSecondaryIndexProps = {
      indexName: `${GSI_NAME}${n}`,
      partitionKey: { name: `${GSI_PARTITION_KEY.name}${n}`, type: GSI_PARTITION_KEY.type },
    };
    yield globalSecondaryIndexProps;
    n++;
  }
}
function* NON_KEY_ATTRIBUTE_GENERATOR(nonKeyPrefix: string): Generator<string, never> {
  let n = 0;
  while (true) {
    yield `${nonKeyPrefix}${n}`;
    n++;
  }
}

// DynamoDB local secondary index parameters
const LSI_NAME = 'MyLSI';
const LSI_SORT_KEY: Attribute = { name: 'lsiSortKey', type: AttributeType.NUMBER };
const LSI_NON_KEY = 'lsiNonKey';
function* LSI_GENERATOR(): Generator<LocalSecondaryIndexProps, never> {
  let n = 0;
  while (true) {
    const localSecondaryIndexProps: LocalSecondaryIndexProps = {
      indexName: `${LSI_NAME}${n}`,
      sortKey: { name: `${LSI_SORT_KEY.name}${n}`, type: LSI_SORT_KEY.type },
    };
    yield localSecondaryIndexProps;
    n++;
  }
}

describe('default properties', () => {
  let stack: Stack;
  beforeEach(() => {
    stack = new Stack();
  });

  test('hash key only', () => {
    new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      AttributeDefinitions: [{ AttributeName: 'hashKey', AttributeType: 'S' }],
      KeySchema: [{ AttributeName: 'hashKey', KeyType: 'HASH' }],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
    });

    Template.fromStack(stack).hasResource('AWS::DynamoDB::Table', { DeletionPolicy: CfnDeletionPolicy.RETAIN });
  });

  test('table without indexes omits GSI and LSI properties', () => {
    new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      GlobalSecondaryIndexes: Match.absent(),
      LocalSecondaryIndexes: Match.absent(),
    });
  });

  test('removalPolicy is DESTROY', () => {
    new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, removalPolicy: RemovalPolicy.DESTROY });

    Template.fromStack(stack).hasResource('AWS::DynamoDB::Table', { DeletionPolicy: CfnDeletionPolicy.DELETE });
  });

  test('hash + range key', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
    });
  });

  test('hash + range key can also be specified in props', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      });
  });

  test('point-in-time recovery is not enabled', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      },
    );
  });

  test('point-in-time-recovery-specification enabled', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true,
        recoveryPeriodInDays: 5,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        PointInTimeRecoverySpecification: {
          PointInTimeRecoveryEnabled: true,
          RecoveryPeriodInDays: 5,
        },
      },
    );
  });

  test('both point-in-time-recovery-specification and point-in-time-recovery set', () => {
    expect(() => new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
      pointInTimeRecovery: true,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true,
        recoveryPeriodInDays: 5,
      },
    })).toThrow('`pointInTimeRecoverySpecification` and `pointInTimeRecovery` are set. Use `pointInTimeRecoverySpecification` only.');
  });

  test('recoveryPeriodInDays set out of bounds', () => {
    expect(() => {
      new Table(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        pointInTimeRecoverySpecification: {
          pointInTimeRecoveryEnabled: true,
          recoveryPeriodInDays: 36,
        },
      });
    }).toThrow('`recoveryPeriodInDays` must be a value between `1` and `35`.');
  });

  test('recoveryPeriodInDays set but pitr disabled', () => {
    expect(() => {
      new Table(stack, 'Table', {
        partitionKey: { name: 'pk', type: AttributeType.STRING },
        pointInTimeRecoverySpecification: {
          pointInTimeRecoveryEnabled: false,
          recoveryPeriodInDays: 35,
        },
      });
    }).toThrow('Cannot set `recoveryPeriodInDays` while `pointInTimeRecoveryEnabled` is set to false.');
  });

  test('server-side encryption is not enabled', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      },
    );
  });

  test('stream is not enabled', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      },
    );
  });

  test('ttl is not enabled', () => {
    new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      },
    );
  });

  test('can specify new and old images', () => {
    new Table(stack, CONSTRUCT_NAME, {
      tableName: TABLE_NAME,
      readCapacity: 42,
      writeCapacity: 1337,
      stream: StreamViewType.NEW_AND_OLD_IMAGES,
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        StreamSpecification: { StreamViewType: 'NEW_AND_OLD_IMAGES' },
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 42, WriteCapacityUnits: 1337 },
        TableName: 'MyTable',
      },
    );
  });

  test('can specify new images only', () => {
    new Table(stack, CONSTRUCT_NAME, {
      tableName: TABLE_NAME,
      readCapacity: 42,
      writeCapacity: 1337,
      stream: StreamViewType.NEW_IMAGE,
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 42, WriteCapacityUnits: 1337 },
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        StreamSpecification: { StreamViewType: 'NEW_IMAGE' },
        TableName: 'MyTable',
      },
    );
  });

  test('can specify old images only', () => {
    new Table(stack, CONSTRUCT_NAME, {
      tableName: TABLE_NAME,
      readCapacity: 42,
      writeCapacity: 1337,
      stream: StreamViewType.OLD_IMAGE,
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
          { AttributeName: 'sortKey', KeyType: 'RANGE' },
        ],
        ProvisionedThroughput: { ReadCapacityUnits: 42, WriteCapacityUnits: 1337 },
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'sortKey', AttributeType: 'N' },
        ],
        StreamSpecification: { StreamViewType: 'OLD_IMAGE' },
        TableName: 'MyTable',
      },
    );
  });

  test('can use PhysicalName.GENERATE_IF_NEEDED as the Table name', () => {
    new Table(stack, CONSTRUCT_NAME, {
      tableName: PhysicalName.GENERATE_IF_NEEDED,
      partitionKey: TABLE_PARTITION_KEY,
    });

    // since the resource has not been used in a cross-environment manner,
    // so the name should not be filled
    Template.fromStack(stack).hasResource('AWS::DynamoDB::Table', {
      TableName: Match.absent(),
    });
  });
});

describe('L1 static factory methods', () => {
  test('fromTableArn', () => {
    const stack = new Stack();
    const table = CfnTable.fromTableArn(stack, 'MyBucket', 'arn:aws:dynamodb:eu-west-1:123456789012:table/MyTable');
    expect(table.tableRef.tableName).toEqual('MyTable');
    expect(table.tableRef.tableArn).toEqual('arn:aws:dynamodb:eu-west-1:123456789012:table/MyTable');

    const env = stack.resolve((table as unknown as Resource).env);
    expect(env).toEqual({
      region: 'eu-west-1',
      account: '123456789012',
    });
  });

  test('fromTableName', () => {
    const app = new App();
    const stack = new Stack(app, 'MyStack', {
      env: { account: '23432424', region: 'us-east-1' },
    });

    const table = CfnTable.fromTableName(stack, 'Table', 'MyTable');
    const arnComponents = stack.splitArn(table.tableRef.tableArn, ArnFormat.SLASH_RESOURCE_NAME);

    expect(table.tableRef.tableName).toEqual('MyTable');
    expect(arnComponents).toMatchObject({
      account: '23432424',
      region: 'us-east-1',
      resource: 'table',
      resourceName: 'MyTable',
      service: 'dynamodb',
    });

    expect(stack.resolve(arnComponents.partition)).toEqual({
      Ref: 'AWS::Partition',
    });

    const env = stack.resolve((table as unknown as Resource).env);
    expect(env).toEqual({
      region: 'us-east-1',
      account: '23432424',
    });
  });

  test('arnForTable created with fromTableName', () => {
    const app = new App();
    const stack = new Stack(app, 'MyStack', {
      env: { account: '23432424', region: 'us-east-1' },
    });

    const table = CfnTable.fromTableName(stack, 'Table', 'MyTable');
    const arn = CfnTable.arnForTable(table);

    expect(stack.resolve(arn)).toEqual(stack.resolve(table.tableRef.tableArn));
  });

  test('arnForTable output matches input ARN', () => {
    const app = new App();
    const stack = new Stack(app, 'MyStack', {
      env: { account: '23432424', region: 'us-east-1' },
    });

    const inputArn = 'arn:aws:dynamodb:us-east-2:123456789012:table/myDynamoDBTable';
    const table = CfnTable.fromTableArn(stack, 'Table', 'arn:aws:dynamodb:us-east-2:123456789012:table/myDynamoDBTable');
    const outputArn = CfnTable.arnForTable(table);

    expect(stack.resolve(outputArn)).toEqual(stack.resolve(inputArn));
  });
});

testDeprecated('when specifying every property', () => {
  const stack = new Stack();
  const stream = new kinesis.Stream(stack, 'MyStream');
  const table = new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    readCapacity: 42,
    writeCapacity: 1337,
    pointInTimeRecovery: true,
    serverSideEncryption: true,
    billingMode: BillingMode.PROVISIONED,
    stream: StreamViewType.KEYS_ONLY,
    timeToLiveAttribute: 'timeToLive',
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
    contributorInsightsEnabled: true,
    kinesisStream: stream,
  });
  Tags.of(table).add('Environment', 'Production');

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: {
        ReadCapacityUnits: 42,
        WriteCapacityUnits: 1337,
      },
      PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true },
      SSESpecification: { SSEEnabled: true },
      StreamSpecification: { StreamViewType: 'KEYS_ONLY' },
      TableName: 'MyTable',
      Tags: [{ Key: 'Environment', Value: 'Production' }],
      TimeToLiveSpecification: { AttributeName: 'timeToLive', Enabled: true },
      ContributorInsightsSpecification: { Enabled: true },
      KinesisStreamSpecification: {
        StreamArn: {
          'Fn::GetAtt': ['MyStream5C050E93', 'Arn'],
        },
      },
    },
  );
});

test('when replica removal policy is not specified', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    removalPolicy: RemovalPolicy.RETAIN,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': true,
  });
});

test('when replica removal policy is not specified', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  const table = new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });
  table.applyRemovalPolicy(RemovalPolicy.DESTROY);

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': false,
  });
});

test('when replica and table removal policy is not specified', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': true,
  });
});

test('when replica and table removal policy is not specified with feature flag true', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': true,
  });
});

test('when table removal policy is specified with feature flag true', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    removalPolicy: RemovalPolicy.DESTROY,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': false,
  });
});

test('when replica and table removal policy is not specified with feature flag false', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: false,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': Match.absent(),
  });
});

test('when replica is retain and table is destroy', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    removalPolicy: RemovalPolicy.DESTROY,
    replicaRemovalPolicy: RemovalPolicy.RETAIN,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': true,
  });
});

test('when replica is destory and table is retain', () => {
  const app = new App({
    context: {
      [cxapi.DYNAMODB_TABLE_RETAIN_TABLE_REPLICA]: true,
    },
  });
  const stack = new Stack(app);
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    removalPolicy: RemovalPolicy.RETAIN,
    replicaRemovalPolicy: RemovalPolicy.DESTROY,
    replicationRegions: ['eu-west-2', 'eu-west-3'],
  });

  Template.fromStack(stack).hasResourceProperties('Custom::DynamoDBReplica', {
    'SkipReplicaDeletion': false,
  });
});

test('when specifying sse with customer managed CMK', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    encryption: TableEncryption.CUSTOMER_MANAGED,
    partitionKey: TABLE_PARTITION_KEY,
  });
  Tags.of(table).add('Environment', 'Production');

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    'SSESpecification': {
      'KMSMasterKeyId': {
        'Fn::GetAtt': [
          'MyTableKey8597C7A6',
          'Arn',
        ],
      },
      'SSEEnabled': true,
      'SSEType': 'KMS',
    },
  });
});

test('when specifying only encryptionKey', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key', {
    enableKeyRotation: true,
  });
  const table = new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    encryptionKey,
    partitionKey: TABLE_PARTITION_KEY,
  });
  Tags.of(table).add('Environment', 'Production');

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    'SSESpecification': {
      'KMSMasterKeyId': {
        'Fn::GetAtt': [
          'Key961B73FD',
          'Arn',
        ],
      },
      'SSEEnabled': true,
      'SSEType': 'KMS',
    },
  });
});

test('when specifying sse with customer managed CMK with encryptionKey provided by user', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key', {
    enableKeyRotation: true,
  });
  const table = new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    encryption: TableEncryption.CUSTOMER_MANAGED,
    encryptionKey,
    partitionKey: TABLE_PARTITION_KEY,
  });
  Tags.of(table).add('Environment', 'Production');

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    'SSESpecification': {
      'KMSMasterKeyId': {
        'Fn::GetAtt': [
          'Key961B73FD',
          'Arn',
        ],
      },
      'SSEEnabled': true,
      'SSEType': 'KMS',
    },
  });
});

test('fails if encryption key is used with AWS managed CMK', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key', {
    enableKeyRotation: true,
  });
  expect(() => new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    encryption: TableEncryption.AWS_MANAGED,
    encryptionKey,
  })).toThrow(`encryptionKey cannot be specified unless encryption is set to TableEncryption.CUSTOMER_MANAGED (it was set to ${TableEncryption.AWS_MANAGED})`);
});

test('fails if encryption key is used with default encryption', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key', {
    enableKeyRotation: true,
  });
  expect(() => new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    encryption: TableEncryption.DEFAULT,
    encryptionKey,
  })).toThrow(`encryptionKey cannot be specified unless encryption is set to TableEncryption.CUSTOMER_MANAGED (it was set to ${TableEncryption.DEFAULT})`);
});

testDeprecated('fails if encryption key is used with serverSideEncryption', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key', {
    enableKeyRotation: true,
  });
  expect(() => new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    serverSideEncryption: true,
    encryptionKey,
  })).toThrow(/encryptionKey cannot be specified when serverSideEncryption is specified. Use encryption instead/);
});

testDeprecated('fails if both encryption and serverSideEncryption is specified', () => {
  const stack = new Stack();
  expect(() => new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    encryption: TableEncryption.DEFAULT,
    serverSideEncryption: true,
  })).toThrow(/Only one of encryption and serverSideEncryption can be specified, but both were provided/);
});

test('fails if both replication regions used with customer managed CMK', () => {
  const stack = new Stack();
  expect(() => new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    replicationRegions: ['us-east-1', 'us-east-2', 'us-west-2'],
    encryption: TableEncryption.CUSTOMER_MANAGED,
  })).toThrow('TableEncryption.CUSTOMER_MANAGED is not supported by DynamoDB Global Tables (where replicationRegions was set)');
});

test('if an encryption key is included, encrypt/decrypt permissions are added to the principal', () => {
  const stack = new Stack();
  const table = new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    encryption: TableEncryption.CUSTOMER_MANAGED,
  });
  const user = new iam.User(stack, 'MyUser');
  table.grantReadWriteData(user);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: [
          'kms:Decrypt',
          'kms:DescribeKey',
          'kms:Encrypt',
          'kms:ReEncrypt*',
          'kms:GenerateDataKey*',
        ],
        Effect: 'Allow',
        Resource: {
          'Fn::GetAtt': [
            'TableAKey07CC09EC',
            'Arn',
          ],
        },
      }]),
    },
  });
});

test('if an encryption key is included, encrypt/decrypt permissions are added to the principal for grantWriteData', () => {
  const stack = new Stack();
  const table = new Table(stack, 'Table A', {
    tableName: TABLE_NAME,
    partitionKey: TABLE_PARTITION_KEY,
    encryption: TableEncryption.CUSTOMER_MANAGED,
  });
  const user = new iam.User(stack, 'MyUser');
  table.grantWriteData(user);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: [
          'kms:Decrypt',
          'kms:DescribeKey',
          'kms:Encrypt',
          'kms:ReEncrypt*',
          'kms:GenerateDataKey*',
        ],
        Effect: 'Allow',
        Resource: {
          'Fn::GetAtt': [
            'TableAKey07CC09EC',
            'Arn',
          ],
        },
      }]),
    },
  });
});

test('replica-handler permission check @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy enabled', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: true,
    },
  });
  const stack = new Stack(app, 'Stack');

  // WHEN
  const provider = ReplicaProvider.getOrCreate(stack, {
    tableName: 'test',
    regions: ['eu-central-1', 'eu-west-1'],
  });

  // THEN
  Template.fromStack(provider).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': 'iam:CreateServiceLinkedRole',
          'Effect': 'Allow',
          'Resource': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  Ref: 'AWS::Partition',
                },
                ':iam::',
                {
                  Ref: 'AWS::AccountId',
                },
                ':role/aws-service-role/replication.dynamodb.amazonaws.com/AWSServiceRoleForDynamoDBReplication',
              ],
            ],
          },
        },
      ],
    },
  });
  Template.fromStack(provider).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': 'dynamodb:DescribeLimits',
          'Effect': 'Allow',
          'Resource': '*',
        },
      ],
    },
  });
  Template.fromStack(provider).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': [
            'dynamodb:DeleteTable',
            'dynamodb:DeleteTableReplica',
          ],
          'Effect': 'Allow',
          'Resource': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':dynamodb:eu-central-1:',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':table/test',
                ],
              ],
            },
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':dynamodb:eu-west-1:',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':table/test',
                ],
              ],
            },
          ],
        },
      ],
    },
  });
});

test('replica-handler permission check @aws-cdk/aws-lambda:createNewPoliciesWithAddToRolePolicy disabled', () => {
  // GIVEN
  const app = new App({
    context: {
      [cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY]: false,
    },
  });
  const stack = new Stack(app, 'Stack');

  // WHEN
  const provider = ReplicaProvider.getOrCreate(stack, {
    tableName: 'test',
    regions: ['eu-central-1', 'eu-west-1'],
  });

  // THEN
  Template.fromStack(provider).hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': [
        {
          'Action': 'iam:CreateServiceLinkedRole',
          'Effect': 'Allow',
          'Resource': {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  Ref: 'AWS::Partition',
                },
                ':iam::',
                {
                  Ref: 'AWS::AccountId',
                },
                ':role/aws-service-role/replication.dynamodb.amazonaws.com/AWSServiceRoleForDynamoDBReplication',
              ],
            ],
          },
        },
        {
          'Action': 'dynamodb:DescribeLimits',
          'Effect': 'Allow',
          'Resource': '*',
        },
        {
          'Action': [
            'dynamodb:DeleteTable',
            'dynamodb:DeleteTableReplica',
          ],
          'Effect': 'Allow',
          'Resource': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':dynamodb:eu-central-1:',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':table/test',
                ],
              ],
            },
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':dynamodb:eu-west-1:',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':table/test',
                ],
              ],
            },
          ],
        },
      ],
    },
  });
});

test('when specifying STANDARD_INFREQUENT_ACCESS table class', () => {
  const stack = new Stack();
  new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    tableClass: TableClass.STANDARD_INFREQUENT_ACCESS,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      TableClass: 'STANDARD_INFREQUENT_ACCESS',
    },
  );
});

test('when specifying STANDARD table class', () => {
  const stack = new Stack();
  new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    tableClass: TableClass.STANDARD,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      TableClass: 'STANDARD',
    },
  );
});

test('when specifying no table class', () => {
  const stack = new Stack();
  new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      TableClass: Match.absent(),
    },
  );
});

test('when specifying PAY_PER_REQUEST billing mode', () => {
  const stack = new Stack();
  new Table(stack, CONSTRUCT_NAME, {
    tableName: TABLE_NAME,
    billingMode: BillingMode.PAY_PER_REQUEST,
    partitionKey: TABLE_PARTITION_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
      ],
      BillingMode: 'PAY_PER_REQUEST',
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
      ],
      TableName: 'MyTable',
    },
  );
});

describe('when billing mode is PAY_PER_REQUEST', () => {
  let stack: Stack;

  beforeEach(() => {
    stack = new Stack();
  });

  test('creating the Table fails when readCapacity is specified', () => {
    expect(() => new Table(stack, 'Table A', {
      tableName: TABLE_NAME,
      partitionKey: TABLE_PARTITION_KEY,
      billingMode: BillingMode.PAY_PER_REQUEST,
      readCapacity: 1,
    })).toThrow(/PAY_PER_REQUEST/);
  });

  test('creating the Table fails when writeCapacity is specified', () => {
    expect(() => new Table(stack, 'Table B', {
      tableName: TABLE_NAME,
      partitionKey: TABLE_PARTITION_KEY,
      billingMode: BillingMode.PAY_PER_REQUEST,
      writeCapacity: 1,
    })).toThrow(/PAY_PER_REQUEST/);
  });

  test('creating the Table fails when both readCapacity and writeCapacity are specified', () => {
    expect(() => new Table(stack, 'Table C', {
      tableName: TABLE_NAME,
      partitionKey: TABLE_PARTITION_KEY,
      billingMode: BillingMode.PAY_PER_REQUEST,
      readCapacity: 1,
      writeCapacity: 1,
    })).toThrow(/PAY_PER_REQUEST/);
  });

  test('when specifying maximum throughput for on-demand', () => {
    stack = new Stack();
    new Table(stack, CONSTRUCT_NAME, {
      tableName: TABLE_NAME,
      billingMode: BillingMode.PAY_PER_REQUEST,
      partitionKey: TABLE_PARTITION_KEY,
      maxReadRequestUnits: 10,
      maxWriteRequestUnits: 5,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        KeySchema: [
          { AttributeName: 'hashKey', KeyType: 'HASH' },
        ],
        BillingMode: 'PAY_PER_REQUEST',
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
        ],
        TableName: 'MyTable',
        OnDemandThroughput: {
          MaxReadRequestUnits: 10,
          MaxWriteRequestUnits: 5,
        },
      },
    );
  });

  test('when specifying maximum throughput for on-demand-indexes', () => {
    stack = new Stack();
    const table = new Table(stack, CONSTRUCT_NAME, {
      tableName: TABLE_NAME,
      billingMode: BillingMode.PAY_PER_REQUEST,
      partitionKey: TABLE_PARTITION_KEY,
      maxReadRequestUnits: 10,
      maxWriteRequestUnits: 5,
    });
    table.addGlobalSecondaryIndex({
      maxReadRequestUnits: 10,
      maxWriteRequestUnits: 20,
      indexName: 'gsi1',
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
      {
        KeySchema: [{ AttributeName: 'hashKey', KeyType: 'HASH' }],
        BillingMode: 'PAY_PER_REQUEST',
        AttributeDefinitions: [
          { AttributeName: 'hashKey', AttributeType: 'S' },
          { AttributeName: 'pk', AttributeType: 'S' },
        ],
        TableName: 'MyTable',
        OnDemandThroughput: {
          MaxReadRequestUnits: 10,
          MaxWriteRequestUnits: 5,
        },
        GlobalSecondaryIndexes: [{
          IndexName: 'gsi1',
          KeySchema: [{ AttributeName: 'pk', KeyType: 'HASH' }],
          OnDemandThroughput: {
            MaxReadRequestUnits: 10,
            MaxWriteRequestUnits: 20,
          },
        }],
      },
    );
  });
});

describe('schema details', () => {
  let stack: Stack;
  let table: Table;

  beforeEach(() => {
    stack = new Stack();
    table = new Table(stack, 'Table A', {
      tableName: TABLE_NAME,
      partitionKey: TABLE_PARTITION_KEY,
    });
  });

  test('get schema for table with hash key only', () => {
    expect(table.schema()).toEqual({
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: undefined,
    });
  });

  test('get schema for table with hash key + range key', () => {
    table = new Table(stack, 'TableB', {
      tableName: TABLE_NAME,
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    expect(table.schema()).toEqual({
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });
  });

  test('get schema for GSI with hash key', () => {
    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKey: GSI_PARTITION_KEY,
    });

    expect(table.schema(GSI_NAME)).toEqual({
      partitionKey: GSI_PARTITION_KEY,
      sortKey: undefined,
    });
  });

  test('get schema for GSI with hash key + range key', () => {
    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKey: GSI_PARTITION_KEY,
      sortKey: GSI_SORT_KEY,
    });

    expect(table.schema(GSI_NAME)).toEqual({
      partitionKey: GSI_PARTITION_KEY,
      sortKey: GSI_SORT_KEY,
    });
  });

  test('get schema for LSI', () => {
    table.addLocalSecondaryIndex({
      indexName: LSI_NAME,
      sortKey: LSI_SORT_KEY,
    });

    expect(table.schema(LSI_NAME)).toEqual({
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: LSI_SORT_KEY,
    });
  });

  test('get schema for multiple secondary indexes', () => {
    table.addLocalSecondaryIndex({
      indexName: LSI_NAME,
      sortKey: LSI_SORT_KEY,
    });

    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKey: GSI_PARTITION_KEY,
      sortKey: GSI_SORT_KEY,
    });

    expect(table.schema(LSI_NAME)).toEqual({
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: LSI_SORT_KEY,
    });

    expect(table.schema(GSI_NAME)).toEqual({
      partitionKey: GSI_PARTITION_KEY,
      sortKey: GSI_SORT_KEY,
    });
  });

  test('get schema for unknown secondary index', () => {
    expect(() => table.schema(GSI_NAME))
      .toThrow(/Cannot find schema for index: MyGSI. Use 'addGlobalSecondaryIndex' or 'addLocalSecondaryIndex' to add index/);
  });

  describe('schemaV2', () => {
    test('get normalized schema for table with hash key only', () => {
      expect(table.schemaV2()).toEqual({
        partitionKeys: [TABLE_PARTITION_KEY],
        sortKeys: [],
      });
    });

    test('get normalized schema for table with hash key + range key', () => {
      table = new Table(stack, 'TableB', {
        tableName: TABLE_NAME,
        partitionKey: TABLE_PARTITION_KEY,
        sortKey: TABLE_SORT_KEY,
      });

      expect(table.schemaV2()).toEqual({
        partitionKeys: [TABLE_PARTITION_KEY],
        sortKeys: [TABLE_SORT_KEY],
      });
    });

    test('get normalized schema for GSI with single partition key', () => {
      table.addGlobalSecondaryIndex({
        indexName: GSI_NAME,
        partitionKey: GSI_PARTITION_KEY,
      });

      expect(table.schemaV2(GSI_NAME)).toEqual({
        partitionKeys: [GSI_PARTITION_KEY],
        sortKeys: [],
      });
    });

    test('get normalized schema for GSI with multi-attribute partition keys', () => {
      const pk1: Attribute = { name: 'pk1', type: AttributeType.STRING };
      const pk2: Attribute = { name: 'pk2', type: AttributeType.STRING };

      table.addGlobalSecondaryIndex({
        indexName: GSI_NAME,
        partitionKeys: [pk1, pk2],
      });

      expect(table.schemaV2(GSI_NAME)).toEqual({
        partitionKeys: [pk1, pk2],
        sortKeys: [],
      });
    });

    test('get normalized schema for GSI with multi-attribute sort keys', () => {
      const sk1: Attribute = { name: 'sk1', type: AttributeType.STRING };
      const sk2: Attribute = { name: 'sk2', type: AttributeType.STRING };

      table.addGlobalSecondaryIndex({
        indexName: GSI_NAME,
        partitionKey: GSI_PARTITION_KEY,
        sortKeys: [sk1, sk2],
      });

      expect(table.schemaV2(GSI_NAME)).toEqual({
        partitionKeys: [GSI_PARTITION_KEY],
        sortKeys: [sk1, sk2],
      });
    });

    test('get normalized schema for LSI', () => {
      table.addLocalSecondaryIndex({
        indexName: LSI_NAME,
        sortKey: LSI_SORT_KEY,
      });

      expect(table.schemaV2(LSI_NAME)).toEqual({
        partitionKeys: [TABLE_PARTITION_KEY],
        sortKeys: [LSI_SORT_KEY],
      });
    });
  });
});

test('when adding a global secondary index with hash key only', () => {
  const stack = new Stack();

  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    readCapacity: 42,
    writeCapacity: 1337,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 42, WriteCapacityUnits: 1337 },
        },
      ],
    },
  );
});

test('when adding a global secondary index with hash + range key', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    projectionType: ProjectionType.ALL,
    readCapacity: 42,
    writeCapacity: 1337,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
        { AttributeName: 'gsiSortKey', AttributeType: 'B' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
            { AttributeName: 'gsiSortKey', KeyType: 'RANGE' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 42, WriteCapacityUnits: 1337 },
        },
      ],
    },
  );
});

test('when adding a global secondary index with projection type KEYS_ONLY', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    projectionType: ProjectionType.KEYS_ONLY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
        { AttributeName: 'gsiSortKey', AttributeType: 'B' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
            { AttributeName: 'gsiSortKey', KeyType: 'RANGE' },
          ],
          Projection: { ProjectionType: 'KEYS_ONLY' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
      ],
    },
  );
});

test('when adding a global secondary index with projection type INCLUDE', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const gsiNonKeyAttributeGenerator = NON_KEY_ATTRIBUTE_GENERATOR(GSI_NON_KEY);
  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    projectionType: ProjectionType.INCLUDE,
    nonKeyAttributes: [gsiNonKeyAttributeGenerator.next().value, gsiNonKeyAttributeGenerator.next().value],
    readCapacity: 42,
    writeCapacity: 1337,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
        { AttributeName: 'gsiSortKey', AttributeType: 'B' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
            { AttributeName: 'gsiSortKey', KeyType: 'RANGE' },
          ],
          Projection: { NonKeyAttributes: ['gsiNonKey0', 'gsiNonKey1'], ProjectionType: 'INCLUDE' },
          ProvisionedThroughput: { ReadCapacityUnits: 42, WriteCapacityUnits: 1337 },
        },
      ],
    },
  );
});

test('when adding a global secondary index on a table with PAY_PER_REQUEST billing mode', () => {
  const stack = new Stack();
  new Table(stack, CONSTRUCT_NAME, {
    billingMode: BillingMode.PAY_PER_REQUEST,
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  }).addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
      ],
      BillingMode: 'PAY_PER_REQUEST',
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
        },
      ],
    },
  );
});

test('error when adding a global secondary index with projection type INCLUDE, but without specifying non-key attributes', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    projectionType: ProjectionType.INCLUDE,
  })).toThrow(/non-key attributes should be specified when using INCLUDE projection type/);
});

test('error when adding a global secondary index with projection type ALL, but with non-key attributes', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const gsiNonKeyAttributeGenerator = NON_KEY_ATTRIBUTE_GENERATOR(GSI_NON_KEY);

  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    nonKeyAttributes: [gsiNonKeyAttributeGenerator.next().value],
  })).toThrow(/non-key attributes should not be specified when not using INCLUDE projection type/);
});

test('error when adding a global secondary index with projection type KEYS_ONLY, but with non-key attributes', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const gsiNonKeyAttributeGenerator = NON_KEY_ATTRIBUTE_GENERATOR(GSI_NON_KEY);

  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    projectionType: ProjectionType.KEYS_ONLY,
    nonKeyAttributes: [gsiNonKeyAttributeGenerator.next().value],
  })).toThrow(/non-key attributes should not be specified when not using INCLUDE projection type/);
});

test('error when adding a global secondary index with projection type INCLUDE, but with more than 100 non-key attributes', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const gsiNonKeyAttributeGenerator = NON_KEY_ATTRIBUTE_GENERATOR(GSI_NON_KEY);
  const gsiNonKeyAttributes: string[] = [];
  for (let i = 0; i < 101; i++) {
    gsiNonKeyAttributes.push(gsiNonKeyAttributeGenerator.next().value);
  }

  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    projectionType: ProjectionType.INCLUDE,
    nonKeyAttributes: gsiNonKeyAttributes,
  })).toThrow(/a maximum number of nonKeyAttributes across all of secondary indexes is 100/);
});

test('error when adding a global secondary index with read or write capacity on a PAY_PER_REQUEST table', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    billingMode: BillingMode.PAY_PER_REQUEST,
  });

  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    readCapacity: 1,
  })).toThrow(/PAY_PER_REQUEST/);
  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    writeCapacity: 1,
  })).toThrow(/PAY_PER_REQUEST/);
  expect(() => table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    sortKey: GSI_SORT_KEY,
    readCapacity: 1,
    writeCapacity: 1,
  })).toThrow(/PAY_PER_REQUEST/);
});

test('when adding multiple global secondary indexes', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const gsiGenerator = GSI_GENERATOR();
  for (let i = 0; i < 5; i++) {
    table.addGlobalSecondaryIndex(gsiGenerator.next().value);
  }

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey0', AttributeType: 'S' },
        { AttributeName: 'gsiHashKey1', AttributeType: 'S' },
        { AttributeName: 'gsiHashKey2', AttributeType: 'S' },
        { AttributeName: 'gsiHashKey3', AttributeType: 'S' },
        { AttributeName: 'gsiHashKey4', AttributeType: 'S' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI0',
          KeySchema: [
            { AttributeName: 'gsiHashKey0', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
        {
          IndexName: 'MyGSI1',
          KeySchema: [
            { AttributeName: 'gsiHashKey1', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
        {
          IndexName: 'MyGSI2',
          KeySchema: [
            { AttributeName: 'gsiHashKey2', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
        {
          IndexName: 'MyGSI3',
          KeySchema: [
            { AttributeName: 'gsiHashKey3', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
        {
          IndexName: 'MyGSI4',
          KeySchema: [
            { AttributeName: 'gsiHashKey4', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
      ],
    },
  );
});

test('when adding a global secondary index without specifying read and write capacity', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
          ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
        },
      ],
    },
  );
});

test.each([true, false])('when adding a global secondary index with contributorInsightsEnabled %s', (contributorInsightsEnabled: boolean) => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  });

  table.addGlobalSecondaryIndex({
    contributorInsightsEnabled,
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          ContributorInsightsSpecification: {
            Enabled: contributorInsightsEnabled,
          },
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
          ],
          Projection: { ProjectionType: 'ALL' },
        },
      ],
    },
  );
});

test('when adding a local secondary index with hash + range key', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });

  table.addLocalSecondaryIndex({
    indexName: LSI_NAME,
    sortKey: LSI_SORT_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'lsiSortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      LocalSecondaryIndexes: [
        {
          IndexName: 'MyLSI',
          KeySchema: [
            { AttributeName: 'hashKey', KeyType: 'HASH' },
            { AttributeName: 'lsiSortKey', KeyType: 'RANGE' },
          ],
          Projection: { ProjectionType: 'ALL' },
        },
      ],
    },
  );
});

test('when adding a local secondary index with projection type KEYS_ONLY', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  table.addLocalSecondaryIndex({
    indexName: LSI_NAME,
    sortKey: LSI_SORT_KEY,
    projectionType: ProjectionType.KEYS_ONLY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'lsiSortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      LocalSecondaryIndexes: [
        {
          IndexName: 'MyLSI',
          KeySchema: [
            { AttributeName: 'hashKey', KeyType: 'HASH' },
            { AttributeName: 'lsiSortKey', KeyType: 'RANGE' },
          ],
          Projection: { ProjectionType: 'KEYS_ONLY' },
        },
      ],
    },
  );
});

test('when adding a local secondary index with projection type INCLUDE', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const lsiNonKeyAttributeGenerator = NON_KEY_ATTRIBUTE_GENERATOR(LSI_NON_KEY);
  table.addLocalSecondaryIndex({
    indexName: LSI_NAME,
    sortKey: LSI_SORT_KEY,
    projectionType: ProjectionType.INCLUDE,
    nonKeyAttributes: [lsiNonKeyAttributeGenerator.next().value, lsiNonKeyAttributeGenerator.next().value],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'lsiSortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      LocalSecondaryIndexes: [
        {
          IndexName: 'MyLSI',
          KeySchema: [
            { AttributeName: 'hashKey', KeyType: 'HASH' },
            { AttributeName: 'lsiSortKey', KeyType: 'RANGE' },
          ],
          Projection: { NonKeyAttributes: ['lsiNonKey0', 'lsiNonKey1'], ProjectionType: 'INCLUDE' },
        },
      ],
    },
  );
});

test('error when adding more than 5 local secondary indexes', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  const lsiGenerator = LSI_GENERATOR();
  for (let i = 0; i < 5; i++) {
    table.addLocalSecondaryIndex(lsiGenerator.next().value);
  }

  expect(() => table.addLocalSecondaryIndex(lsiGenerator.next().value))
    .toThrow(/a maximum number of local secondary index per table is 5/);
});

test('error when adding a local secondary index with the name of a global secondary index', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY, sortKey: TABLE_SORT_KEY });
  table.addGlobalSecondaryIndex({
    indexName: 'SecondaryIndex',
    partitionKey: GSI_PARTITION_KEY,
  });

  expect(() => table.addLocalSecondaryIndex({
    indexName: 'SecondaryIndex',
    sortKey: LSI_SORT_KEY,
  })).toThrow(/a duplicate index name, SecondaryIndex, is not allowed/);
});

test('error when validating construct if a local secondary index exists without a sort key of the table', () => {
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { partitionKey: TABLE_PARTITION_KEY });

  table.addLocalSecondaryIndex({
    indexName: LSI_NAME,
    sortKey: LSI_SORT_KEY,
  });

  const errors = table.node.validate();

  expect(errors.length).toBe(1);
  expect(errors[0]).toBe('a sort key of the table must be specified to add local secondary indexes');
});

test('can enable Read AutoScaling', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { readCapacity: 42, writeCapacity: 1337, partitionKey: TABLE_PARTITION_KEY });

  // WHEN
  table.autoScaleReadCapacity({ minCapacity: 50, maxCapacity: 500 }).scaleOnUtilization({ targetUtilizationPercent: 75 });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalableTarget', {
    MaxCapacity: 500,
    MinCapacity: 50,
    ScalableDimension: 'dynamodb:table:ReadCapacityUnits',
    ServiceNamespace: 'dynamodb',
  });
  Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
    PolicyType: 'TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration: {
      PredefinedMetricSpecification: { PredefinedMetricType: 'DynamoDBReadCapacityUtilization' },
      TargetValue: 75,
    },
  });
});

test('can enable Write AutoScaling', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { readCapacity: 42, writeCapacity: 1337, partitionKey: TABLE_PARTITION_KEY });

  // WHEN
  table.autoScaleWriteCapacity({ minCapacity: 50, maxCapacity: 500 }).scaleOnUtilization({ targetUtilizationPercent: 75 });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalableTarget', {
    MaxCapacity: 500,
    MinCapacity: 50,
    ScalableDimension: 'dynamodb:table:WriteCapacityUnits',
    ServiceNamespace: 'dynamodb',
  });
  Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalingPolicy', {
    PolicyType: 'TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration: {
      PredefinedMetricSpecification: { PredefinedMetricType: 'DynamoDBWriteCapacityUtilization' },
      TargetValue: 75,
    },
  });
});

test('cannot enable AutoScaling twice on the same property', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { readCapacity: 42, writeCapacity: 1337, partitionKey: TABLE_PARTITION_KEY });
  table.autoScaleReadCapacity({ minCapacity: 50, maxCapacity: 500 }).scaleOnUtilization({ targetUtilizationPercent: 75 });

  // WHEN
  expect(() => {
    table.autoScaleReadCapacity({ minCapacity: 50, maxCapacity: 500 });
  }).toThrow(/Read AutoScaling already enabled for this table/);
});

test('error when enabling AutoScaling on the PAY_PER_REQUEST table', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { billingMode: BillingMode.PAY_PER_REQUEST, partitionKey: TABLE_PARTITION_KEY });
  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
  });

  // WHEN
  expect(() => {
    table.autoScaleReadCapacity({ minCapacity: 50, maxCapacity: 500 });
  }).toThrow(/PAY_PER_REQUEST/);
  expect(() => {
    table.autoScaleWriteCapacity({ minCapacity: 50, maxCapacity: 500 });
  }).toThrow(/PAY_PER_REQUEST/);
  expect(() => table.autoScaleGlobalSecondaryIndexReadCapacity(GSI_NAME, {
    minCapacity: 1,
    maxCapacity: 5,
  })).toThrow(/PAY_PER_REQUEST/);
});

test('error when specifying Read Auto Scaling with invalid scalingTargetValue < 10', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { readCapacity: 42, writeCapacity: 1337, partitionKey: TABLE_PARTITION_KEY });

  // THEN
  expect(() => {
    table.autoScaleReadCapacity({ minCapacity: 50, maxCapacity: 500 }).scaleOnUtilization({ targetUtilizationPercent: 5 });
  }).toThrow(/targetUtilizationPercent for DynamoDB scaling must be between 10 and 90 percent, got: 5/);
});

test('error when specifying Read Auto Scaling with invalid minimumCapacity', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, { readCapacity: 42, writeCapacity: 1337, partitionKey: TABLE_PARTITION_KEY });

  // THEN
  expect(() => table.autoScaleReadCapacity({ minCapacity: 10, maxCapacity: 5 }))
    .toThrow(/minCapacity \(10\) should be lower than maxCapacity \(5\)/);
});

test('can autoscale on a schedule', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    readCapacity: 42,
    writeCapacity: 1337,
    partitionKey: { name: 'Hash', type: AttributeType.STRING },
  });

  // WHEN
  const scaling = table.autoScaleReadCapacity({ minCapacity: 1, maxCapacity: 100 });
  scaling.scaleOnSchedule('SaveMoneyByNotScalingUp', {
    schedule: appscaling.Schedule.cron({}),
    maxCapacity: 10,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::ApplicationAutoScaling::ScalableTarget', {
    ScheduledActions: [
      {
        ScalableTargetAction: { 'MaxCapacity': 10 },
        Schedule: 'cron(* * * * ? *)',
        ScheduledActionName: 'SaveMoneyByNotScalingUp',
      },
    ],
  });
});

test('scheduled scaling shows warning when minute is not defined in cron', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    readCapacity: 42,
    writeCapacity: 1337,
    partitionKey: { name: 'Hash', type: AttributeType.STRING },
  });

  // WHEN
  const scaling = table.autoScaleReadCapacity({ minCapacity: 1, maxCapacity: 100 });
  scaling.scaleOnSchedule('SaveMoneyByNotScalingUp', {
    schedule: appscaling.Schedule.cron({}),
    maxCapacity: 10,
  });

  // THEN
  Annotations.fromStack(stack).hasWarning('/Default/MyTable/ReadScaling/Target', "cron: If you don't pass 'minute', by default the event runs every minute. Pass 'minute: '*'' if that's what you intend, or 'minute: 0' to run once per hour instead. [ack: @aws-cdk/aws-applicationautoscaling:defaultRunEveryMinute]");
});

test('scheduled scaling shows no warning when minute is * in cron', () => {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, CONSTRUCT_NAME, {
    readCapacity: 42,
    writeCapacity: 1337,
    partitionKey: { name: 'Hash', type: AttributeType.STRING },
  });

  // WHEN
  const scaling = table.autoScaleReadCapacity({ minCapacity: 1, maxCapacity: 100 });
  scaling.scaleOnSchedule('SaveMoneyByNotScalingUp', {
    schedule: appscaling.Schedule.cron({ minute: '*' }),
    maxCapacity: 10,
  });

  // THEN
  const annotations = Annotations.fromStack(stack).findWarning('*', Match.anyValue());
  expect(annotations.length).toBe(0);
});

describe('metrics', () => {
  test('Can use metricConsumedReadCapacityUnits on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricConsumedReadCapacityUnits())).toEqual({
      period: Duration.minutes(5),
      dimensions: { TableName: { Ref: 'TableCD117FA1' } },
      namespace: 'AWS/DynamoDB',
      metricName: 'ConsumedReadCapacityUnits',
      statistic: 'Sum',
    });
  });

  test('Can use metricConsumedWriteCapacityUnits on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricConsumedWriteCapacityUnits())).toEqual({
      period: Duration.minutes(5),
      dimensions: { TableName: { Ref: 'TableCD117FA1' } },
      namespace: 'AWS/DynamoDB',
      metricName: 'ConsumedWriteCapacityUnits',
      statistic: 'Sum',
    });
  });

  test('Using metricSystemErrorsForOperations with no operations will default to all', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    expect(Object.keys(table.metricSystemErrorsForOperations().toMetricConfig().mathExpression!.usingMetrics)).toEqual([
      'getitem',
      'batchgetitem',
      'scan',
      'query',
      'getrecords',
      'putitem',
      'deleteitem',
      'updateitem',
      'batchwriteitem',
      'transactwriteitems',
      'transactgetitems',
      'executetransaction',
      'batchexecutestatement',
      'executestatement',
    ]);
  });

  testDeprecated('Can use metricSystemErrors without the TableName dimension', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    expect(table.metricSystemErrors({ dimensions: { Operation: 'GetItem' } }).dimensions).toEqual({
      TableName: table.tableName,
      Operation: 'GetItem',
    });
  });

  testDeprecated('Using metricSystemErrors without the Operation dimension will fail', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    expect(() => table.metricSystemErrors({ dimensions: { TableName: table.tableName } }))
      .toThrow(/'Operation' dimension must be passed for the 'SystemErrors' metric./);
  });

  test('Can use metricSystemErrorsForOperations on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricSystemErrorsForOperations({ operations: [Operation.GET_ITEM, Operation.PUT_ITEM] }))).toEqual({
      expression: 'getitem + putitem',
      label: 'Sum of errors across all operations',
      period: Duration.minutes(5),
      usingMetrics: {
        getitem: {
          dimensions: {
            Operation: 'GetItem',
            TableName: {
              Ref: 'TableCD117FA1',
            },
          },
          metricName: 'SystemErrors',
          namespace: 'AWS/DynamoDB',
          period: Duration.minutes(5),
          statistic: 'Sum',
        },
        putitem: {
          dimensions: {
            Operation: 'PutItem',
            TableName: {
              Ref: 'TableCD117FA1',
            },
          },
          metricName: 'SystemErrors',
          namespace: 'AWS/DynamoDB',
          period: Duration.minutes(5),
          statistic: 'Sum',
        },
      },
    });
  });

  testDeprecated('Can use metricSystemErrors on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricSystemErrors({ dimensionsMap: { TableName: table.tableName, Operation: 'GetItem' } }))).toEqual({
      period: Duration.minutes(5),
      dimensions: { TableName: { Ref: 'TableCD117FA1' }, Operation: 'GetItem' },
      namespace: 'AWS/DynamoDB',
      metricName: 'SystemErrors',
      statistic: 'Sum',
    });
  });

  test('Using metricUserErrors with dimensions will fail', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    expect(() => table.metricUserErrors({ dimensions: { TableName: table.tableName } })).toThrow(/'dimensions' is not supported for the 'UserErrors' metric/);
  });

  test('Can use metricUserErrors on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricUserErrors())).toEqual({
      period: Duration.minutes(5),
      dimensions: {},
      namespace: 'AWS/DynamoDB',
      metricName: 'UserErrors',
      statistic: 'Sum',
    });
  });

  test('Can use metricConditionalCheckFailedRequests on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricConditionalCheckFailedRequests())).toEqual({
      period: Duration.minutes(5),
      dimensions: { TableName: { Ref: 'TableCD117FA1' } },
      namespace: 'AWS/DynamoDB',
      metricName: 'ConditionalCheckFailedRequests',
      statistic: 'Sum',
    });
  });

  test('Can use metricSuccessfulRequestLatency without the TableName dimension', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    expect(table.metricSuccessfulRequestLatency({ dimensionsMap: { Operation: 'GetItem' } }).dimensions).toEqual({
      TableName: table.tableName,
      Operation: 'GetItem',
    });
  });

  test('Using metricSuccessfulRequestLatency without the Operation dimension will fail', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    expect(() => table.metricSuccessfulRequestLatency({ dimensionsMap: { TableName: table.tableName } }))
      .toThrow(/'Operation' dimension must be passed for the 'SuccessfulRequestLatency' metric./);
  });

  test('Can use metricSuccessfulRequestLatency on a Dynamodb Table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(stack.resolve(table.metricSuccessfulRequestLatency({
      dimensionsMap: {
        TableName: table.tableName,
        Operation: 'GetItem',
      },
    }))).toEqual({
      period: Duration.minutes(5),
      dimensions: { TableName: { Ref: 'TableCD117FA1' }, Operation: 'GetItem' },
      namespace: 'AWS/DynamoDB',
      metricName: 'SuccessfulRequestLatency',
      statistic: 'Average',
    });
  });
});

describe('grants', () => {
  test('"grant" allows adding arbitrary actions associated with this table resource', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'my-table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });
    const user = new iam.User(stack, 'user');

    // WHEN
    table.grant(user, 'dynamodb:action1', 'dynamodb:action2');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': [
              'dynamodb:action1',
              'dynamodb:action2',
            ],
            'Effect': 'Allow',
            'Resource': [{
              'Fn::GetAtt': [
                'mytable0324D45C',
                'Arn',
              ],
            }],
          },
        ],
        'Version': '2012-10-17',
      },
      'PolicyName': 'userDefaultPolicy083DF682',
      'Users': [
        {
          'Ref': 'user2C2B57AE',
        },
      ],
    });
  });

  test('"grant" allows adding arbitrary actions associated with this table resource (via testGrant)', () => {
    testGrant(
      ['action1', 'action2'], (p, t) => t.grant(p, 'dynamodb:action1', 'dynamodb:action2'));
  });

  test('"grantReadData" allows the principal to read data from the table', () => {
    testGrant(
      ['BatchGetItem', 'GetRecords', 'GetShardIterator', 'Query', 'GetItem', 'Scan', 'ConditionCheckItem', 'DescribeTable'], (p, t) => t.grantReadData(p));
  });

  test('"grantWriteData" allows the principal to write data to the table', () => {
    testGrant(
      ['BatchWriteItem', 'PutItem', 'UpdateItem', 'DeleteItem', 'DescribeTable'], (p, t) => t.grantWriteData(p));
  });

  test('"grantReadWriteData" allows the principal to read/write data', () => {
    testGrant([
      'BatchGetItem', 'GetRecords', 'GetShardIterator', 'Query', 'GetItem', 'Scan',
      'ConditionCheckItem', 'BatchWriteItem', 'PutItem', 'UpdateItem', 'DeleteItem', 'DescribeTable',
    ], (p, t) => t.grantReadWriteData(p));
  });

  test('"grantFullAccess" allows the principal to perform any action on the table ("*")', () => {
    testGrant(['*'], (p, t) => t.grantFullAccess(p));
  });

  test('grant* with ServicePrincipal throws error', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'id', type: AttributeType.STRING },
    });

    // THEN
    expect(() => table.grantReadWriteData(new iam.ServicePrincipal('bedrock.amazonaws.com')))
      .toThrow(/DynamoDB grant\* methods do not support ServicePrincipal grantees/);
  });

  test('grant* with wrapped ServicePrincipal (withConditions) throws error', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
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
    const table = new Table(stack, 'Table', {
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
    const table = new Table(stack, 'Table', {
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

  testDeprecated('"Table.grantListStreams" allows principal to list all streams', () => {
    // GIVEN
    const stack = new Stack();
    const user = new iam.User(stack, 'user');

    // WHEN
    Table.grantListStreams(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': 'dynamodb:ListStreams',
            'Effect': 'Allow',
            'Resource': '*',
          },
        ],
        'Version': '2012-10-17',
      },
      'Users': [{ 'Ref': 'user2C2B57AE' }],
    });
  });

  test('"grantTableListStreams" should fail if streaming is not enabled on table"', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'my-table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });
    const user = new iam.User(stack, 'user');

    // WHEN
    expect(() => table.grantTableListStreams(user)).toThrow(/DynamoDB Streams must be enabled on the table Default\/my-table/);
  });

  test('"grantTableListStreams" allows principal to list all streams for this table', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'my-table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      stream: StreamViewType.NEW_IMAGE,
    });
    const user = new iam.User(stack, 'user');

    // WHEN
    table.grantTableListStreams(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': 'dynamodb:ListStreams',
            'Effect': 'Allow',
            'Resource': '*',
          },
        ],
        'Version': '2012-10-17',
      },
      'Users': [{ 'Ref': 'user2C2B57AE' }],
    });
  });

  test('"grantStreamRead" should fail if streaming is not enabled on table"', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'my-table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });
    const user = new iam.User(stack, 'user');

    // WHEN
    expect(() => table.grantStreamRead(user)).toThrow(/DynamoDB Streams must be enabled on the table Default\/my-table/);
  });

  test('"grantStreamRead" allows principal to read and describe the table stream"', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'my-table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      stream: StreamViewType.NEW_IMAGE,
    });
    const user = new iam.User(stack, 'user');

    // WHEN
    table.grantStreamRead(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': 'dynamodb:ListStreams',
            'Effect': 'Allow',
            'Resource': '*',
          },
          {
            'Action': [
              'dynamodb:DescribeStream',
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            'Effect': 'Allow',
            'Resource': {
              'Fn::GetAtt': [
                'mytable0324D45C',
                'StreamArn',
              ],
            },
          },
        ],
        'Version': '2012-10-17',
      },
      'Users': [{ 'Ref': 'user2C2B57AE' }],
    });
  });

  test('if table has an index grant gives access to the index', () => {
    // GIVEN
    const stack = new Stack();

    const table = new Table(stack, 'my-table', { partitionKey: { name: 'ID', type: AttributeType.STRING } });
    table.addGlobalSecondaryIndex({ indexName: 'MyIndex', partitionKey: { name: 'Age', type: AttributeType.NUMBER } });
    const user = new iam.User(stack, 'user');

    // WHEN
    table.grantReadData(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': [
              'dynamodb:BatchGetItem',
              'dynamodb:Query',
              'dynamodb:GetItem',
              'dynamodb:Scan',
              'dynamodb:ConditionCheckItem',
              'dynamodb:DescribeTable',
            ],
            'Effect': 'Allow',
            'Resource': [
              {
                'Fn::GetAtt': [
                  'mytable0324D45C',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'mytable0324D45C',
                        'Arn',
                      ],
                    },
                    '/index/*',
                  ],
                ],
              },
            ],
          },
          {
            'Action': [
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            'Effect': 'Allow',
            'Resource': [
              {
                'Fn::GetAtt': [
                  'mytable0324D45C',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'mytable0324D45C',
                        'Arn',
                      ],
                    },
                    '/index/*',
                  ],
                ],
              },
            ],
          },
        ],
        'Version': '2012-10-17',
      },
      'PolicyName': 'userDefaultPolicy083DF682',
      'Users': [
        {
          'Ref': 'user2C2B57AE',
        },
      ],
    });
  });

  test('grant for an imported table', () => {
    // GIVEN
    const stack = new Stack();
    const table = Table.fromTableName(stack, 'MyTable', 'my-table');
    const user = new iam.User(stack, 'user');

    // WHEN
    table.grant(user, 'dynamodb:*');

    // THEN
    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'dynamodb:*',
            Effect: 'Allow',
            Resource: [{
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':dynamodb:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':table/my-table',
                ],
              ],
            }],
          },
        ],
        Version: '2012-10-17',
      },
      Users: [
        {
          Ref: 'user2C2B57AE',
        },
      ],
    });
  });
});

describe('secondary indexes', () => {
  // See https://github.com/aws/aws-cdk/issues/4398
  test('attribute can be used as key attribute in one index, and non-key in another', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: { name: 'pkey', type: AttributeType.NUMBER },
    });

    // WHEN
    table.addGlobalSecondaryIndex({
      indexName: 'IndexA',
      partitionKey: { name: 'foo', type: AttributeType.STRING },
      projectionType: ProjectionType.INCLUDE,
      nonKeyAttributes: ['bar'],
    });

    // THEN
    expect(() => table.addGlobalSecondaryIndex({
      indexName: 'IndexB',
      partitionKey: { name: 'baz', type: AttributeType.STRING },
      sortKey: { name: 'bar', type: AttributeType.STRING },
      projectionType: ProjectionType.INCLUDE,
      nonKeyAttributes: ['blah'],
    })).not.toThrow();
  });
});

describe('import', () => {
  test('report error when importing an external/existing table from invalid arn missing resource name', () => {
    const stack = new Stack();

    const tableArn = 'arn:aws:dynamodb:us-east-1::table/';
    // WHEN
    expect(() => Table.fromTableArn(stack, 'ImportedTable', tableArn)).toThrow(/ARN for DynamoDB table must be in the form: .../);
  });

  test('static fromTableArn(arn) allows importing an external/existing table from arn', () => {
    const stack = new Stack();

    const tableArn = 'arn:aws:dynamodb:us-east-1:111111111111:table/MyTable';
    const table = Table.fromTableArn(stack, 'ImportedTable', tableArn);

    const role = new iam.Role(stack, 'NewRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });
    table.grantReadData(role);

    // it is possible to obtain a permission statement for a ref
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': [
              'dynamodb:BatchGetItem',
              'dynamodb:Query',
              'dynamodb:GetItem',
              'dynamodb:Scan',
              'dynamodb:ConditionCheckItem',
              'dynamodb:DescribeTable',
            ],
            'Effect': 'Allow',
            'Resource': [tableArn],
          },
          {
            'Action': [
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            'Effect': 'Allow',
            'Resource': [tableArn],
          },
        ],
        'Version': '2012-10-17',
      },
      'PolicyName': 'NewRoleDefaultPolicy90E8F49D',
      'Roles': [{ 'Ref': 'NewRole99763075' }],
    });

    expect(table.tableArn).toBe(tableArn);
    expect(stack.resolve(table.tableName)).toBe('MyTable');
  });

  test('static fromTableName(name) allows importing an external/existing table from table name', () => {
    const stack = new Stack();

    const tableName = 'MyTable';
    const table = Table.fromTableName(stack, 'ImportedTable', tableName);

    const role = new iam.Role(stack, 'NewRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });
    table.grantReadWriteData(role);

    // it is possible to obtain a permission statement for a ref
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyDocument': {
        'Statement': [
          {
            'Action': [
              'dynamodb:BatchGetItem',
              'dynamodb:Query',
              'dynamodb:GetItem',
              'dynamodb:Scan',
              'dynamodb:ConditionCheckItem',
              'dynamodb:BatchWriteItem',
              'dynamodb:PutItem',
              'dynamodb:UpdateItem',
              'dynamodb:DeleteItem',
              'dynamodb:DescribeTable',
            ],
            'Effect': 'Allow',
            'Resource': [{
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    'Ref': 'AWS::Partition',
                  },
                  ':dynamodb:',
                  {
                    'Ref': 'AWS::Region',
                  },
                  ':',
                  {
                    'Ref': 'AWS::AccountId',
                  },
                  ':table/MyTable',
                ],
              ],
            }],
          },
          {
            'Action': [
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            'Effect': 'Allow',
            'Resource': [{
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    'Ref': 'AWS::Partition',
                  },
                  ':dynamodb:',
                  {
                    'Ref': 'AWS::Region',
                  },
                  ':',
                  {
                    'Ref': 'AWS::AccountId',
                  },
                  ':table/MyTable',
                ],
              ],
            }],
          },
        ],
        'Version': '2012-10-17',
      },
      'PolicyName': 'NewRoleDefaultPolicy90E8F49D',
      'Roles': [{ 'Ref': 'NewRole99763075' }],
    });

    expect(table.tableArn).toBe(`arn:${Aws.PARTITION}:dynamodb:${Aws.REGION}:${Aws.ACCOUNT_ID}:table/MyTable`);
    expect(stack.resolve(table.tableName)).toBe(tableName);
  });

  describe('stream permissions on imported tables', () => {
    test('throw if no tableStreamArn is specified', () => {
      const stack = new Stack();

      const tableName = 'MyTable';
      const table = Table.fromTableAttributes(stack, 'ImportedTable', { tableName });

      const role = new iam.Role(stack, 'NewRole', {
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      });

      expect(() => table.grantTableListStreams(role)).toThrow(/DynamoDB Streams must be enabled on the table/);
      expect(() => table.grantStreamRead(role)).toThrow(/DynamoDB Streams must be enabled on the table/);
    });

    test('creates the correct list streams grant', () => {
      const stack = new Stack();

      const tableName = 'MyTable';
      const table = Table.fromTableAttributes(stack, 'ImportedTable', { tableName, tableStreamArn });

      const role = new iam.Role(stack, 'NewRole', {
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      });

      expect(table.grantTableListStreams(role)).toBeDefined();

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: 'dynamodb:ListStreams',
              Effect: 'Allow',
              Resource: '*',
            },
          ],
          Version: '2012-10-17',
        },
        Roles: [stack.resolve(role.roleName)],
      });
    });

    test('creates the correct stream read grant', () => {
      const stack = new Stack();

      const tableName = 'MyTable';
      const table = Table.fromTableAttributes(stack, 'ImportedTable', { tableName, tableStreamArn });

      const role = new iam.Role(stack, 'NewRole', {
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      });

      expect(table.grantStreamRead(role)).toBeDefined();

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: 'dynamodb:ListStreams',
              Effect: 'Allow',
              Resource: '*',
            },
            {
              Action: ['dynamodb:DescribeStream', 'dynamodb:GetRecords', 'dynamodb:GetShardIterator'],
              Effect: 'Allow',
              Resource: tableStreamArn,
            },
          ],
          Version: '2012-10-17',
        },
        Roles: [stack.resolve(role.roleName)],
      });
    });

    test('if an encryption key is included, encrypt/decrypt permissions are added to the principal for grantStreamRead', () => {
      const stack = new Stack();

      const tableName = 'MyTable';
      const encryptionKey = new kms.Key(stack, 'Key', {
        enableKeyRotation: true,
      });

      const table = Table.fromTableAttributes(stack, 'ImportedTable', { tableName, tableStreamArn, encryptionKey });

      const role = new iam.Role(stack, 'NewRole', {
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      });

      expect(table.grantStreamRead(role)).toBeDefined();

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              'Action': 'dynamodb:ListStreams',
              'Effect': 'Allow',
              'Resource': '*',
            },
            {
              'Action': [
                'kms:Decrypt',
                'kms:DescribeKey',
              ],
              'Effect': 'Allow',
              'Resource': {
                'Fn::GetAtt': [
                  'Key961B73FD',
                  'Arn',
                ],
              },
            },
            {
              'Action': [
                'dynamodb:DescribeStream',
                'dynamodb:GetRecords',
                'dynamodb:GetShardIterator',
              ],
              'Effect': 'Allow',
              'Resource': tableStreamArn,
            },
          ],
          Version: '2012-10-17',
        },
        Roles: [stack.resolve(role.roleName)],
      });
    });

    test('creates the correct index grant if indexes have been provided when importing', () => {
      const stack = new Stack();

      const table = Table.fromTableAttributes(stack, 'ImportedTable', {
        tableName: 'MyTableName',
        globalIndexes: ['global'],
        localIndexes: ['local'],
      });

      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AnyPrincipal(),
      });

      table.grantReadData(role);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'dynamodb:BatchGetItem',
                'dynamodb:Query',
                'dynamodb:GetItem',
                'dynamodb:Scan',
                'dynamodb:ConditionCheckItem',
                'dynamodb:DescribeTable',
              ],
              Resource: [
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName',
                  ]],
                },
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName/index/*',
                  ]],
                },
              ],
            },
            {
              Action: [
                'dynamodb:GetRecords',
                'dynamodb:GetShardIterator',
              ],
              Resource: [
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName',
                  ]],
                },
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName/index/*',
                  ]],
                },
              ],
            },
          ],
        },
      });
    });

    test('creates the index permissions if grantIndexPermissions is provided', () => {
      const stack = new Stack();

      const table = Table.fromTableAttributes(stack, 'ImportedTable', {
        tableName: 'MyTableName',
        grantIndexPermissions: true,
      });

      const role = new iam.Role(stack, 'Role', {
        assumedBy: new iam.AnyPrincipal(),
      });

      table.grantReadData(role);

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: [
            {
              Action: [
                'dynamodb:BatchGetItem',
                'dynamodb:Query',
                'dynamodb:GetItem',
                'dynamodb:Scan',
                'dynamodb:ConditionCheckItem',
                'dynamodb:DescribeTable',
              ],
              Resource: [
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName',
                  ]],
                },
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName/index/*',
                  ]],
                },
              ],
            },
            {
              Action: [
                'dynamodb:GetRecords',
                'dynamodb:GetShardIterator',
              ],
              Resource: [
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName',
                  ]],
                },
                {
                  'Fn::Join': ['', [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':dynamodb:',
                    { Ref: 'AWS::Region' },
                    ':',
                    { Ref: 'AWS::AccountId' },
                    ':table/MyTableName/index/*',
                  ]],
                },
              ],
            },
          ],
        },
      });
    });
  });
});

describe('global', () => {
  test('create replicas', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
    });

    // THEN
    Template.fromStack(stack).hasResource('Custom::DynamoDBReplica', {
      Properties: {
        TableName: {
          Ref: 'TableCD117FA1',
        },
        Region: 'eu-west-2',
      },
      Condition: 'TableStackRegionNotEqualseuwest2A03859E7',
    });

    Template.fromStack(stack).hasResource('Custom::DynamoDBReplica', {
      Properties: {
        TableName: {
          Ref: 'TableCD117FA1',
        },
        Region: 'eu-central-1',
      },
      Condition: 'TableStackRegionNotEqualseucentral199D46FC0',
    });

    Template.fromStack(stack).hasCondition('TableStackRegionNotEqualseuwest2A03859E7', {
      'Fn::Not': [
        { 'Fn::Equals': ['eu-west-2', { Ref: 'AWS::Region' }] },
      ],
    });

    Template.fromStack(stack).hasCondition('TableStackRegionNotEqualseucentral199D46FC0', {
      'Fn::Not': [
        { 'Fn::Equals': ['eu-central-1', { Ref: 'AWS::Region' }] },
      ],
    });
  });

  test('create replicas without waiting to finish replication', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
      waitForReplicationToFinish: false,
    });

    // THEN
    Template.fromStack(stack).hasResource('Custom::DynamoDBReplica', {
      Properties: {
        TableName: {
          Ref: 'TableCD117FA1',
        },
        Region: 'eu-west-2',
        SkipReplicationCompletedWait: 'true',
      },
      Condition: 'TableStackRegionNotEqualseuwest2A03859E7',
    });

    Template.fromStack(stack).hasResource('Custom::DynamoDBReplica', {
      Properties: {
        TableName: {
          Ref: 'TableCD117FA1',
        },
        Region: 'eu-central-1',
        SkipReplicationCompletedWait: 'true',
      },
      Condition: 'TableStackRegionNotEqualseucentral199D46FC0',
    });

    Template.fromStack(stack).hasCondition('TableStackRegionNotEqualseuwest2A03859E7', {
      'Fn::Not': [
        { 'Fn::Equals': ['eu-west-2', { Ref: 'AWS::Region' }] },
      ],
    });

    Template.fromStack(stack).hasCondition('TableStackRegionNotEqualseucentral199D46FC0', {
      'Fn::Not': [
        { 'Fn::Equals': ['eu-central-1', { Ref: 'AWS::Region' }] },
      ],
    });
  });

  test('grantReadData', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
    });
    table.addGlobalSecondaryIndex({
      indexName: 'my-index',
      partitionKey: {
        name: 'key',
        type: AttributeType.STRING,
      },
    });
    const user = new iam.User(stack, 'User');

    // WHEN
    table.grantReadData(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'dynamodb:BatchGetItem',
              'dynamodb:Query',
              'dynamodb:GetItem',
              'dynamodb:Scan',
              'dynamodb:ConditionCheckItem',
              'dynamodb:DescribeTable',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  'TableCD117FA1',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'TableCD117FA1',
                        'Arn',
                      ],
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
            ],
          },
          {
            Action: [
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  'TableCD117FA1',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'TableCD117FA1',
                        'Arn',
                      ],
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
            ],
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('grantReadData - global secondary index added after granting', () => {
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
    });
    const user = new iam.User(stack, 'User');

    // WHEN
    table.grantReadData(user);
    table.addGlobalSecondaryIndex({
      indexName: 'my-index',
      partitionKey: {
        name: 'key',
        type: AttributeType.STRING,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'dynamodb:BatchGetItem',
              'dynamodb:Query',
              'dynamodb:GetItem',
              'dynamodb:Scan',
              'dynamodb:ConditionCheckItem',
              'dynamodb:DescribeTable',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  'TableCD117FA1',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'TableCD117FA1',
                        'Arn',
                      ],
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
            ],
          },
          {
            Action: [
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  'TableCD117FA1',
                  'Arn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        'TableCD117FA1',
                        'Arn',
                      ],
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/',
                    {
                      Ref: 'TableCD117FA1',
                    },
                    '/index/*',
                  ],
                ],
              },
            ],
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('grantReadData with AccountRootPrincipal uses wildcard resources', () => {
    // GIVEN
    const stack = new Stack();
    const table = new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });

    // WHEN
    table.grantReadData(new iam.AccountRootPrincipal());

    // THEN - Should create resource policy with wildcard to avoid circular dependency
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
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
    });
  });

  test('grantReadData across regions', () => {
    // GIVEN
    const app = new App();
    const stack1 = new Stack(app, 'Stack1', {
      env: { region: 'us-east-1' },
    });
    const table = new Table(stack1, 'Table', {
      tableName: 'my-table',
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
    });
    table.addGlobalSecondaryIndex({
      indexName: 'my-index',
      partitionKey: {
        name: 'key',
        type: AttributeType.STRING,
      },
    });
    const stack2 = new Stack(app, 'Stack2', {
      env: { region: 'eu-west-2' },
    });
    const user = new iam.User(stack2, 'User');

    // WHEN
    table.grantReadData(user);

    // THEN
    Template.fromStack(stack2).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'dynamodb:BatchGetItem',
              'dynamodb:Query',
              'dynamodb:GetItem',
              'dynamodb:Scan',
              'dynamodb:ConditionCheckItem',
              'dynamodb:DescribeTable',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:us-east-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:us-east-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table/index/*',
                  ],
                ],
              },
            ],
          },
          {
            Action: [
              'dynamodb:GetRecords',
              'dynamodb:GetShardIterator',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:us-east-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:us-east-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-west-2:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table/index/*',
                  ],
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    {
                      Ref: 'AWS::Partition',
                    },
                    ':dynamodb:eu-central-1:',
                    {
                      Ref: 'AWS::AccountId',
                    },
                    ':table/my-table/index/*',
                  ],
                ],
              },
            ],
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('grantTableListStreams across regions', () => {
    // GIVEN
    const app = new App();
    const stack1 = new Stack(app, 'Stack1', {
      env: { region: 'us-east-1' },
    });
    const table = new Table(stack1, 'Table', {
      tableName: 'my-table',
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
    });
    const stack2 = new Stack(app, 'Stack2', {
      env: { region: 'eu-west-2' },
    });
    const user = new iam.User(stack2, 'User');

    // WHEN
    table.grantTableListStreams(user);

    // THEN
    Template.fromStack(stack2).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'dynamodb:ListStreams',
            Effect: 'Allow',
            Resource: '*',
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('throws when PROVISIONED billing mode is used without auto-scaled writes', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
      billingMode: BillingMode.PROVISIONED,
    });

    // THEN
    expect(() => {
      Template.fromStack(stack);
    }).toThrow(/A global Table that uses PROVISIONED as the billing mode needs auto-scaled write capacity/);
  });

  test('throws when PROVISIONED billing mode is used with auto-scaled writes, but without a policy', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const table = new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
      billingMode: BillingMode.PROVISIONED,
    });
    table.autoScaleWriteCapacity({
      minCapacity: 1,
      maxCapacity: 10,
    });

    // THEN
    expect(() => {
      Template.fromStack(stack);
    }).toThrow(/A global Table that uses PROVISIONED as the billing mode needs auto-scaled write capacity with a policy/);
  });

  test('allows PROVISIONED billing mode when auto-scaled writes with a policy are specified', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const table = new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
      billingMode: BillingMode.PROVISIONED,
    });
    table.autoScaleWriteCapacity({
      minCapacity: 1,
      maxCapacity: 10,
    }).scaleOnUtilization({ targetUtilizationPercent: 75 });

    Template.fromStack(stack).hasResource('AWS::DynamoDB::Table', {
      BillingMode: Match.absent(), // PROVISIONED is the default
    });
  });

  test('throws when stream is set and not set to NEW_AND_OLD_IMAGES', () => {
    // GIVEN
    const stack = new Stack();

    // THEN
    expect(() => new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
      stream: StreamViewType.OLD_IMAGE,
    })).toThrow(/`NEW_AND_OLD_IMAGES`/);
  });

  test('throws with replica in same region as stack', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack', {
      env: { region: 'us-east-1' },
    });

    // THEN
    expect(() => new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-1',
        'us-east-1',
        'eu-west-2',
      ],
    })).toThrow(/`replicationRegions` cannot include the region where this stack is deployed/);
  });

  test('no conditions when region is known', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'Stack', {
      env: { region: 'eu-west-1' },
    });

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: [
        'eu-west-2',
        'eu-central-1',
      ],
    });

    // THEN
    const conditions = Template.fromStack(stack).findConditions('*');
    expect(Object.keys(conditions).length).toEqual(0);
  });

  test('can configure timeout', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      replicationRegions: ['eu-central-1'],
      replicationTimeout: Duration.hours(1),
    });

    // THEN
    expect(cr.Provider).toHaveBeenCalledWith(expect.anything(), expect.any(String), expect.objectContaining({
      totalTimeout: Duration.hours(1),
    }));
  });
});

test('L1 inside L2 expects removalpolicy to have been set', () => {
  // Check that the "stateful L1 validation generation" works. Do it here
  // because we know DDB tables are stateful.
  const app = new App();
  const stack = new Stack(app, 'Stack');

  class FakeTableL2 extends Resource {
    constructor(scope: Construct, id: string) {
      super(scope, id);

      new CfnTable(this, 'Resource', {
        keySchema: [{ attributeName: 'hash', keyType: 'S' }],
      });
    }
  }

  new FakeTableL2(stack, 'Table');

  expect(() => {
    Template.fromStack(stack);
  }).toThrow(/is a stateful resource type/);
});

test('System errors metrics', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  // WHEN
  const table = new Table(stack, 'Table', {
    partitionKey: { name: 'metric', type: AttributeType.STRING },
  });
  const metricTableThrottled = table.metricSystemErrorsForOperations({
    operations: [Operation.SCAN],
    period: Duration.minutes(1),
  });
  new cloudwatch.Alarm(stack, 'TableErrorAlarm', {
    metric: metricTableThrottled,
    evaluationPeriods: 1,
    threshold: 1,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
    Metrics: Match.arrayWith([
      Match.objectLike({
        Expression: 'scan',
      }),
      Match.objectLike({
        MetricStat: Match.objectLike({
          Metric: Match.objectLike({
            Dimensions: Match.arrayWith([
              Match.objectLike({
                Name: 'Operation',
              }),
              Match.objectLike({
                Name: 'TableName',
              }),
            ]),
            MetricName: 'SystemErrors',
            Namespace: 'AWS/DynamoDB',
          }),
        }),
      }),
    ]),
  });
});

test('Throttled requests metrics', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  // WHEN
  const table = new Table(stack, 'Table', {
    partitionKey: { name: 'metric', type: AttributeType.STRING },
  });
  const metricTableThrottled = table.metricThrottledRequestsForOperations({
    operations: [Operation.PUT_ITEM],
    period: Duration.minutes(1),
  });
  new cloudwatch.Alarm(stack, 'TableThrottleAlarm', {
    metric: metricTableThrottled,
    evaluationPeriods: 1,
    threshold: 1,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Alarm', {
    Metrics: Match.arrayWith([
      Match.objectLike({
        Expression: 'putitem',
      }),
      Match.objectLike({
        MetricStat: Match.objectLike({
          Metric: Match.objectLike({
            Dimensions: Match.arrayWith([
              Match.objectLike({
                Name: 'Operation',
              }),
              Match.objectLike({
                Name: 'TableName',
              }),
            ]),
            MetricName: 'ThrottledRequests',
            Namespace: 'AWS/DynamoDB',
          }),
        }),
      }),
    ]),
  });
});

function testGrant(expectedActions: string[], invocation: (user: iam.IPrincipal, table: Table) => void) {
  // GIVEN
  const stack = new Stack();
  const table = new Table(stack, 'my-table', { partitionKey: { name: 'ID', type: AttributeType.STRING } });
  const user = new iam.User(stack, 'user');

  // WHEN
  invocation(user, table);

  // THEN
  const template = Template.fromStack(stack);
  const capture = new Capture();

  template.hasResourceProperties('AWS::IAM::Policy', {
    'PolicyDocument': {
      'Statement': capture,
    },
    'PolicyName': 'userDefaultPolicy083DF682',
    'Users': [
      {
        'Ref': 'user2C2B57AE',
      },
    ],
  });

  // Collect all actions from statements that target the table
  const tableResource = { 'Fn::GetAtt': ['mytable0324D45C', 'Arn'] };
  const allActions: string[] = [];

  for (const statement of capture.asArray()) {
    if (statement.Effect === 'Allow' &&
        JSON.stringify(statement.Resource) === JSON.stringify([tableResource])) {
      if (Array.isArray(statement.Action)) {
        allActions.push(...statement.Action);
      } else {
        allActions.push(statement.Action);
      }
    }
  }

  // Check that all expected actions are present
  const expectedDynamoActions = expectedActions.map(a => `dynamodb:${a}`);
  for (const expectedAction of expectedDynamoActions) {
    expect(allActions).toContain(expectedAction);
  }
}

describe('deletionProtectionEnabled', () => {
  test.each([
    [true],
    [false],
  ])('gets passed to table', (state) => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      deletionProtection: state,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      DeletionProtectionEnabled: state,
    });
  });

  test('is not passed when not set', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', Match.objectLike({
      DeletionProtectionEnabled: Match.absent(),
    }));
  });
});

describe('import source', () => {
  let stack: Stack;
  let bucket: s3.IBucket;

  beforeEach(() => {
    stack = new Stack();
    bucket = new s3.Bucket(stack, 'Bucket');
  });

  test('by default ImportSource property is not set', () => {
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      ImportSourceSpecification: Match.absent(),
    });
  });

  test('import DynamoDBJson format', () => {
    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      importSource: {
        compressionType: InputCompressionType.GZIP,
        inputFormat: InputFormat.dynamoDBJson(),
        bucket,
        bucketOwner: '111111111111',
        keyPrefix: 'prefix',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      ImportSourceSpecification: {
        InputCompressionType: 'GZIP',
        InputFormat: 'DYNAMODB_JSON',
        S3BucketSource: {
          S3Bucket: {
            'Ref': 'Bucket83908E77',
          },
          S3BucketOwner: '111111111111',
          S3KeyPrefix: 'prefix',
        },
      },
    });
  });

  test('import Amazon ION format', () => {
    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      importSource: {
        compressionType: InputCompressionType.ZSTD,
        inputFormat: InputFormat.ion(),
        bucket,
        bucketOwner: '111111111111',
        keyPrefix: 'prefix',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      ImportSourceSpecification: {
        InputCompressionType: 'ZSTD',
        InputFormat: 'ION',
        S3BucketSource: {
          S3Bucket: {
            'Ref': 'Bucket83908E77',
          },
          S3BucketOwner: '111111111111',
          S3KeyPrefix: 'prefix',
        },
      },
    });
  });

  test('import CSV format', () => {
    // WHEN
    new Table(stack, 'Table', {
      partitionKey: {
        name: 'id',
        type: AttributeType.STRING,
      },
      importSource: {
        compressionType: InputCompressionType.NONE,
        inputFormat: InputFormat.csv({
          delimiter: ',',
          headerList: ['id', 'name'],
        }),
        bucket,
        bucketOwner: '111111111111',
        keyPrefix: 'prefix',
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
      ImportSourceSpecification: {
        InputCompressionType: 'NONE',
        InputFormat: 'CSV',
        InputFormatOptions: {
          Csv: {
            Delimiter: ',',
            HeaderList: ['id', 'name'],
          },
        },
        S3BucketSource: {
          S3Bucket: {
            'Ref': 'Bucket83908E77',
          },
          S3BucketOwner: '111111111111',
          S3KeyPrefix: 'prefix',
        },
      },
    });
  });

  test.each([
    [',,'], ['a'], ['1'], ['/'], ['+'], ['!'], ['@'],
  ])('throw error when invalid delimiter is specified', (delimiter) => {
    expect(() => {
      new Table(stack, 'Table', {
        partitionKey: {
          name: 'id',
          type: AttributeType.STRING,
        },
        importSource: {
          compressionType: InputCompressionType.NONE,
          inputFormat: InputFormat.csv({
            delimiter,
            headerList: ['id', 'name'],
          }),
          bucket,
          bucketOwner: '111111111111',
          keyPrefix: 'prefix',
        },
      });
    }).toThrow(`Delimiter must be a single character and one of the following: comma (,), tab (\\t), colon (:), semicolon (;), pipe (|), space ( ), got '${delimiter}'`);
  });
});

test('Resource policy test', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  const doc = new iam.PolicyDocument({
    statements: [
      new iam.PolicyStatement({
        actions: ['dynamodb:GetItem'],
        principals: [new iam.ArnPrincipal('arn:aws:iam::111122223333:user/foobar')],
        resources: ['*'],
      }),
    ],
  });

  // WHEN
  new Table(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    resourcePolicy: doc,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    KeySchema: [
      { AttributeName: 'id', KeyType: 'HASH' },
    ],
    AttributeDefinitions: [
      { AttributeName: 'id', AttributeType: 'S' },
    ],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    'ResourcePolicy': {
      'PolicyDocument': {
        'Version': '2012-10-17',
        'Statement': [
          {
            'Principal': {
              'AWS': 'arn:aws:iam::111122223333:user/foobar',
            },
            'Effect': 'Allow',
            'Action': 'dynamodb:GetItem',
            'Resource': '*',
          },
        ],
      },
    },
  });
});

test('addToResourcePolicy allows scoped ARN resources when table has explicit name', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  // WHEN - Create table with explicit name (enables scoped resource policies)
  const table = new Table(stack, 'Table', {
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
  template.hasResourceProperties('AWS::DynamoDB::Table', {
    TableName: 'my-explicit-table-name',
    ResourcePolicy: {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Principal: Match.anyValue(),
            Action: ['dynamodb:GetItem', 'dynamodb:Query'],
            Resource: {
              'Fn::Sub': 'arn:aws:dynamodb:${AWS::Region}:${AWS::AccountId}:table/my-explicit-table-name',
            },
          },
        ],
      },
    },
  });
});

test('addToResourcePolicy requires wildcard resources with auto-generated table names to prevent circular dependencies', () => {
  // This test documents the fundamental limitation of resource policies with auto-generated names

  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  const table = new Table(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    // No explicit tableName - CDK will generate unique name
  });

  // LIMITATION: Cannot use table.tableArn or construct scoped ARN because it creates circular dependency
  // This would fail: resources: [table.tableArn]
  // This would also fail: resources: [Fn.sub('arn:aws:dynamodb:${AWS::Region}:${AWS::AccountId}:table/${TableRef}', { TableRef: cfnTable.ref })]

  // WORKAROUND: Must use wildcard resource (same pattern as KMS)
  table.addToResourcePolicy(new iam.PolicyStatement({
    actions: ['dynamodb:GetItem'],
    principals: [new iam.AccountRootPrincipal()],
    resources: ['*'], // Only option for auto-generated table names
  }));

  // THEN - Verify wildcard is preserved
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::DynamoDB::Table', {
    ResourcePolicy: {
      PolicyDocument: {
        Statement: [
          {
            Resource: '*', // Wildcard is the only way to avoid circular dependency
          },
        ],
      },
    },
  });
});

test('addToResourcePolicy supports multiple statements with wildcard resources to avoid circular dependencies', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  // WHEN
  const table = new Table(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
  });

  // Test multiple policy statements with different principals and actions
  table.addToResourcePolicy(new iam.PolicyStatement({
    actions: ['dynamodb:GetItem', 'dynamodb:PutItem'],
    principals: [new iam.AccountRootPrincipal()],
    resources: ['*'], // Wildcard avoids circular dependency - same pattern as KMS
  }));

  table.addToResourcePolicy(new iam.PolicyStatement({
    actions: ['dynamodb:Query'],
    principals: [new iam.ArnPrincipal('arn:aws:iam::111122223333:user/testuser')],
    resources: ['*'], // Wildcard avoids circular dependency
  }));

  // THEN
  const template = Template.fromStack(stack);
  template.hasResourceProperties('AWS::DynamoDB::Table', {
    ResourcePolicy: {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Principal: {
              AWS: Match.anyValue(), // Principal format can vary
            },
            Action: ['dynamodb:GetItem', 'dynamodb:PutItem'],
            Resource: '*', // Wildcard resource to avoid circular dependency
          },
          {
            Effect: 'Allow',
            Principal: {
              AWS: 'arn:aws:iam::111122223333:user/testuser',
            },
            Action: 'dynamodb:Query',
            Resource: '*', // Wildcard resource
          },
        ],
      },
    },
  });
});

test('Warm Throughput test on-demand', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  // WHEN
  const table = new Table(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    warmThroughput: {
      readUnitsPerSecond: 13000,
      writeUnitsPerSecond: 5000,
    },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'my-index-1',
    partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
    warmThroughput: {
      readUnitsPerSecond: 15000,
      writeUnitsPerSecond: 6000,
    },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'my-index-2',
    partitionKey: { name: 'gsi2pk', type: AttributeType.STRING },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
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

test('Warm Throughput test provisioned', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  // WHEN
  const table = new Table(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    readCapacity: 5,
    writeCapacity: 6,
    warmThroughput: {
      readUnitsPerSecond: 2000,
      writeUnitsPerSecond: 1000,
    },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'my-index-1',
    partitionKey: { name: 'gsi1pk', type: AttributeType.STRING },
    readCapacity: 7,
    writeCapacity: 8,
    warmThroughput: {
      readUnitsPerSecond: 3000,
      writeUnitsPerSecond: 4000,
    },
  });

  table.addGlobalSecondaryIndex({
    indexName: 'my-index-2',
    partitionKey: { name: 'gsi2pk', type: AttributeType.STRING },
    readCapacity: 9,
    writeCapacity: 10,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    KeySchema: [
      { AttributeName: 'id', KeyType: 'HASH' },
    ],
    AttributeDefinitions: [
      { AttributeName: 'id', AttributeType: 'S' },
      { AttributeName: 'gsi1pk', AttributeType: 'S' },
      { AttributeName: 'gsi2pk', AttributeType: 'S' },
    ],
    WarmThroughput: {
      ReadUnitsPerSecond: 2000,
      WriteUnitsPerSecond: 1000,
    },
    ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 6 },
    GlobalSecondaryIndexes: [
      {
        IndexName: 'my-index-1',
        KeySchema: [
          { AttributeName: 'gsi1pk', KeyType: 'HASH' },
        ],
        Projection: { ProjectionType: 'ALL' },
        WarmThroughput: {
          ReadUnitsPerSecond: 3000,
          WriteUnitsPerSecond: 4000,
        },
        ProvisionedThroughput: { ReadCapacityUnits: 7, WriteCapacityUnits: 8 },
      },
      {
        IndexName: 'my-index-2',
        KeySchema: [
          { AttributeName: 'gsi2pk', KeyType: 'HASH' },
        ],
        Projection: { ProjectionType: 'ALL' },
        ProvisionedThroughput: { ReadCapacityUnits: 9, WriteCapacityUnits: 10 },
      },
    ],
  });
});

test('Kinesis Stream - precision timestamp', () => {
  // GIVEN
  const app = new App();
  const stack = new Stack(app, 'Stack');

  const stream = new kinesis.Stream(stack, 'Stream');

  // WHEN
  new Table(stack, 'Table', {
    partitionKey: { name: 'id', type: AttributeType.STRING },
    kinesisStream: stream,
    kinesisPrecisionTimestamp: ApproximateCreationDateTimePrecision.MILLISECOND,
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table', {
    KeySchema: [
      { AttributeName: 'id', KeyType: 'HASH' },
    ],
    AttributeDefinitions: [
      { AttributeName: 'id', AttributeType: 'S' },
    ],
    KinesisStreamSpecification: {
      StreamArn: {
        'Fn::GetAtt': ['Stream790BDEE4', 'Arn'],
      },
      ApproximateCreationDateTimePrecision: 'MILLISECOND',
    },
  });
});

test('Contributor Insights Specification - table', () => {
  const stack = new Stack();

  new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
    contributorInsightsSpecification: {
      enabled: true,
      mode: ContributorInsightsMode.ACCESSED_AND_THROTTLED_KEYS,
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      ContributorInsightsSpecification: {
        Enabled: true,
        Mode: 'ACCESSED_AND_THROTTLED_KEYS',
      },
    },
  );
});

test('Contributor Insights Specification - table - without mode', () => {
  const stack = new Stack();

  new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
    contributorInsightsSpecification: {
      enabled: true,
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      ContributorInsightsSpecification: {
        Enabled: true,
      },
    },
  );
});

test('Contributor Insights Specification - index', () => {
  const stack = new Stack();

  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
    contributorInsightsSpecification: {
      enabled: true,
      mode: ContributorInsightsMode.ACCESSED_AND_THROTTLED_KEYS,
    },
  });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKey: GSI_PARTITION_KEY,
    contributorInsightsSpecification: {
      enabled: true,
      mode: ContributorInsightsMode.THROTTLED_KEYS,
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      ContributorInsightsSpecification: {
        Enabled: true,
        Mode: 'ACCESSED_AND_THROTTLED_KEYS',
      },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
          ],
          ContributorInsightsSpecification: {
            Enabled: true,
            Mode: 'THROTTLED_KEYS',
          },
        },
      ],
    },
  );
});

test('ContributorInsightsSpecification && ContributorInsightsEnabled', () => {
  const stack = new Stack();

  expect(() => {
    new Table(stack, 'Table', {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
      contributorInsightsEnabled: true,
      contributorInsightsSpecification: {
        enabled: true,
        mode: ContributorInsightsMode.ACCESSED_AND_THROTTLED_KEYS,
      },
    });
  }).toThrow('`contributorInsightsSpecification` and `contributorInsightsEnabled` are set. Use `contributorInsightsSpecification` only.');
});

test('Multi-attribute partition keys for global secondary index', () => {
  const stack = new Stack();

  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKeys: [GSI_PARTITION_KEY, GSI_PARTITION_KEY_TWO],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
        { AttributeName: 'gsiHaskKeyTwo', AttributeType: 'N' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
            { AttributeName: 'gsiHaskKeyTwo', KeyType: 'HASH' },
          ],
        },
      ],
    },
  );
});

test('Multi-attribute partition keys and standard sort key for global secondary index', () => {
  const stack = new Stack();

  const table = new Table(stack, CONSTRUCT_NAME, {
    partitionKey: TABLE_PARTITION_KEY,
    sortKey: TABLE_SORT_KEY,
  });

  table.addGlobalSecondaryIndex({
    indexName: GSI_NAME,
    partitionKeys: [GSI_PARTITION_KEY, GSI_PARTITION_KEY_TWO],
    sortKey: GSI_SORT_KEY,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::DynamoDB::Table',
    {
      AttributeDefinitions: [
        { AttributeName: 'hashKey', AttributeType: 'S' },
        { AttributeName: 'sortKey', AttributeType: 'N' },
        { AttributeName: 'gsiHashKey', AttributeType: 'S' },
        { AttributeName: 'gsiHaskKeyTwo', AttributeType: 'N' },
        { AttributeName: 'gsiSortKey', AttributeType: 'B' },
      ],
      KeySchema: [
        { AttributeName: 'hashKey', KeyType: 'HASH' },
        { AttributeName: 'sortKey', KeyType: 'RANGE' },
      ],
      ProvisionedThroughput: { ReadCapacityUnits: 5, WriteCapacityUnits: 5 },
      GlobalSecondaryIndexes: [
        {
          IndexName: 'MyGSI',
          KeySchema: [
            { AttributeName: 'gsiHashKey', KeyType: 'HASH' },
            { AttributeName: 'gsiHaskKeyTwo', KeyType: 'HASH' },
            { AttributeName: 'gsiSortKey', KeyType: 'RANGE' },
          ],
        },
      ],
    },
  );
});

test('Throws when multi-attribute partitionKeys and partitionKey are specified', () => {
  const stack = new Stack();
  expect(() => {
    const table = new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKeys: [GSI_PARTITION_KEY, GSI_PARTITION_KEY_TWO],
      partitionKey: { name: 'gsiHashKeyThree', type: AttributeType.STRING },
      sortKey: GSI_SORT_KEY,
    });
  }).toThrow('Exactly one of \'partitionKey\', \'partitionKeys\' must be specified');
});

test('Throws when multi-attribute sortKeys and sortKey are specified', () => {
  const stack = new Stack();
  expect(() => {
    const table = new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKeys: [GSI_PARTITION_KEY, GSI_PARTITION_KEY_TWO],
      sortKey: GSI_SORT_KEY,
      sortKeys: [GSI_SORT_KEY_TWO, { name: 'gsiSortKeyThree', type: AttributeType.BINARY }],
    });
  }).toThrow('At most one of \'sortKey\', \'sortKeys\' may be specified');
});

test('Throws when more than four multi-attribute partition keys are specified', () => {
  const stack = new Stack();
  expect(() => {
    const table = new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKeys: [GSI_PARTITION_KEY, GSI_PARTITION_KEY_TWO,
        { name: 'gsiPartitionKeyThree', type: AttributeType.BINARY },
        { name: 'gsiPartitionKeyFour', type: AttributeType.BINARY },
        { name: 'gsiPartitionKeyFive', type: AttributeType.BINARY }],
      sortKeys: [GSI_SORT_KEY, GSI_SORT_KEY_TWO],
    });
  }).toThrow('Maximum of 4 partition keys allowed');
});

test('Throws when more than four multi-attribute sort keys are specified', () => {
  const stack = new Stack();
  expect(() => {
    const table = new Table(stack, CONSTRUCT_NAME, {
      partitionKey: TABLE_PARTITION_KEY,
      sortKey: TABLE_SORT_KEY,
    });

    table.addGlobalSecondaryIndex({
      indexName: GSI_NAME,
      partitionKeys: [GSI_PARTITION_KEY, GSI_PARTITION_KEY_TWO],
      sortKeys: [GSI_SORT_KEY, GSI_SORT_KEY_TWO,
        { name: 'gsiSortKeyThree', type: AttributeType.BINARY },
        { name: 'gsiSortKeyFour', type: AttributeType.BINARY },
        { name: 'gsiSortKeyFive', type: AttributeType.BINARY }],
    });
  }).toThrow('Maximum of 4 sort keys allowed');
});

describe('L1 table grants', () => {
  test('grant read permission to service principal (L1) throws error', () => {
    const stack = new Stack();
    const table = new CfnTable(stack, 'Table', {
      keySchema: [{ attributeName: 'id', keyType: 'HASH' }],
      attributeDefinitions: [{ attributeName: 'id', attributeType: 'S' }],
    });
    const principal = new iam.ServicePrincipal('lambda.amazonaws.com');

    expect(() => TableGrants.fromTable(table).readData(principal))
      .toThrow(/DynamoDB grant\* methods do not support ServicePrincipal grantees/);
  });
});

test('grant read permission to CfnTable with encryption adds KMS permissions', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key');
  const table = new CfnTable(stack, 'Table', {
    keySchema: [{ attributeName: 'id', keyType: 'HASH' }],
    attributeDefinitions: [{ attributeName: 'id', attributeType: 'S' }],
    sseSpecification: {
      sseEnabled: true,
      sseType: 'KMS',
      kmsMasterKeyId: encryptionKey.keyArn,
    },
    billingMode: 'PAY_PER_REQUEST',
  });
  const user = new iam.User(stack, 'User');

  TableGrants.fromTable(table).readData(user);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['kms:Decrypt', 'kms:DescribeKey'],
        Effect: 'Allow',
        Resource: { 'Fn::GetAtt': ['Key961B73FD', 'Arn'] },
      }]),
    },
  });
});

test('grant write permission to CfnTable with encryption adds KMS permissions', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key');
  const table = new CfnTable(stack, 'Table', {
    keySchema: [{ attributeName: 'id', keyType: 'HASH' }],
    attributeDefinitions: [{ attributeName: 'id', attributeType: 'S' }],
    sseSpecification: {
      sseEnabled: true,
      sseType: 'KMS',
      kmsMasterKeyId: encryptionKey.keyArn,
    },
    billingMode: 'PAY_PER_REQUEST',
  });
  const user = new iam.User(stack, 'User');

  TableGrants.fromTable(table).writeData(user);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['kms:Decrypt', 'kms:DescribeKey', 'kms:Encrypt', 'kms:ReEncrypt*', 'kms:GenerateDataKey*'],
        Effect: 'Allow',
        Resource: { 'Fn::GetAtt': ['Key961B73FD', 'Arn'] },
      }]),
    },
  });
});

test('grant readWrite permission to CfnTable with encryption adds KMS permissions', () => {
  const stack = new Stack();
  const encryptionKey = new kms.Key(stack, 'Key');
  const table = new CfnTable(stack, 'Table', {
    keySchema: [{ attributeName: 'id', keyType: 'HASH' }],
    attributeDefinitions: [{ attributeName: 'id', attributeType: 'S' }],
    sseSpecification: {
      sseEnabled: true,
      sseType: 'KMS',
      kmsMasterKeyId: encryptionKey.keyArn,
    },
    billingMode: 'PAY_PER_REQUEST',
  });
  const user = new iam.User(stack, 'User');

  TableGrants.fromTable(table).readWriteData(user);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([{
        Action: ['kms:Decrypt', 'kms:DescribeKey', 'kms:Encrypt', 'kms:ReEncrypt*', 'kms:GenerateDataKey*'],
        Effect: 'Allow',
        Resource: { 'Fn::GetAtt': ['Key961B73FD', 'Arn'] },
      }]),
    },
  });
});
