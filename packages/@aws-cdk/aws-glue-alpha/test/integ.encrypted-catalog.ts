#!/usr/bin/env node
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import { Key } from 'aws-cdk-lib/aws-kms';
import { Catalog, Database, DataCatalogEncryptionAtRest, DataFormat, S3Table, Schema } from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-glue');

const catalogKey = new Key(stack, 'CatalogKey', {
  enableKeyRotation: true,
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});

// Encrypt the account-wide catalog. This must be configured before the account
// catalog is first used (e.g. by the Database below, which uses it by default).
// All Data Catalog metadata - including partition indexes - is now encrypted
// with this customer-managed key.
Catalog.encryptAccount(stack, {
  encryptionAtRest: DataCatalogEncryptionAtRest.kms(catalogKey),
});

const database = new Database(stack, 'Database', {
  databaseName: 'testdb',
});

const table = new S3Table(stack, 'MyTable', {
  database: database,
  tableName: 'my_table',
  columns: [
    {
      name: 'col1',
      type: Schema.STRING,
    },
  ],
  partitionKeys: [
    {
      name: 'year',
      type: Schema.SMALL_INT,
    },
  ],
  dataFormat: DataFormat.JSON,
  enablePartitionFiltering: true,
  partitionIndexes: [
    {
      indexName: 'test',
      keyNames: [
        'year',
      ],
    },
  ],
});

const integTest = new integ.IntegTest(app, 'aws-cdk-glue-table-integ', {
  testCases: [stack],
});

// The partition index above is created at deploy time by the custom-resource
// handlers, which call CreatePartitionIndex/UpdateTable against the encrypted
// catalog. That write only succeeds if the handlers hold the KMS permissions
// this construct grants them (kms:Decrypt on the catalog key) - otherwise the
// custom resource fails and the deployment fails. Asserting the index reached
// ACTIVE confirms the grant let the handlers write the encrypted metadata; a
// missing grant would surface here as a failed deployment.
const indexes = integTest.assertions.awsApiCall('Glue', 'getPartitionIndexes', {
  CatalogId: stack.account,
  DatabaseName: database.databaseName,
  TableName: table.tableName,
});

// The assertion Lambda is a separate principal that only reads the index, so it
// needs the Glue read actions plus kms:Decrypt to read the encrypted metadata.
// It is not the subject of the test - the handlers are - it just observes the
// result they produced at deploy time.
indexes.provider.addToRolePolicy({
  Effect: 'Allow',
  Action: ['glue:GetPartitionIndexes', 'glue:GetTable'],
  Resource: ['*'],
});
indexes.provider.addToRolePolicy({
  Effect: 'Allow',
  Action: ['kms:Decrypt'],
  Resource: [catalogKey.keyArn],
});

indexes.expect(integ.ExpectedResult.objectLike({
  PartitionIndexDescriptorList: [
    { IndexName: 'test', IndexStatus: 'ACTIVE' },
  ],
}));
