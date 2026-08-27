#!/usr/bin/env node
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-glue');
const bucket = new s3.Bucket(stack, 'DataBucket');
const database = new glue.Database(stack, 'MyDatabase', {
  databaseName: 'database',
});

const columns = [{
  name: 'col1',
  type: glue.Schema.STRING,
}, {
  name: 'col2',
  type: glue.Schema.STRING,
}, {
  name: 'col3',
  type: glue.Schema.STRING,
}];

const partitionKeys = [{
  name: 'year',
  type: glue.Schema.SMALL_INT,
}, {
  name: 'month',
  type: glue.Schema.BIG_INT,
}, {
  name: 'day',
  type: glue.Schema.BIG_INT,
}];

const csvTable = new glue.S3Table(stack, 'CSVTable', {
  database,
  storage: glue.S3TableStorage.fromBucket(bucket),
  tableName: 'csv_table',
  columns,
  partitionKeys,
  partitionIndexes: [{
    indexName: 'index1',
    keyNames: ['month'],
  }],
  dataFormat: glue.DataFormat.CSV,
});

csvTable.addPartitionIndex({
  indexName: 'index2',
  keyNames: ['month', 'year'],
});

csvTable.addPartitionIndex({
  indexName: 'index3',
  keyNames: ['day'],
});

const jsonTable = new glue.S3Table(stack, 'JSONTable', {
  database,
  storage: glue.S3TableStorage.fromBucket(bucket),
  tableName: 'json_table',
  columns,
  partitionKeys,
  dataFormat: glue.DataFormat.JSON,
});

jsonTable.addPartitionIndex({
  keyNames: ['year', 'month'],
});

const integTest = new integ.IntegTest(app, 'glue-partition-index-integ', {
  testCases: [stack],
});

// Verify partition indexes were created on csv_table
const csvIndexes = integTest.assertions.awsApiCall('Glue', 'getPartitionIndexes', {
  CatalogId: stack.account,
  DatabaseName: 'database',
  TableName: 'csv_table',
});

csvIndexes.provider.addToRolePolicy({
  Effect: 'Allow',
  Action: ['glue:GetPartitionIndexes', 'glue:GetTable'],
  Resource: ['*'],
});

csvIndexes.expect(integ.ExpectedResult.objectLike({
  PartitionIndexDescriptorList: [
    { IndexName: 'index1', IndexStatus: 'ACTIVE' },
    { IndexName: 'index2', IndexStatus: 'ACTIVE' },
    { IndexName: 'index3', IndexStatus: 'ACTIVE' },
  ],
}));

app.synth();
