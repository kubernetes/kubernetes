import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as glue from '../lib';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-glue-data-quality-ruleset');

const bucket = new s3.Bucket(stack, 'DataBucket');
const database = new glue.Database(stack, 'MyDatabase', {
  databaseName: 'my_database',
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
}];

const csvTable = new glue.S3Table(stack, 'CSVTable', {
  database,
  storage: glue.S3TableStorage.fromBucket(bucket),
  tableName: 'csv_table',
  columns,
  partitionKeys,
  dataFormat: glue.DataFormat.CSV,
});

new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
  clientToken: 'client_token',
  description: 'my description',
  rulesetName: 'my_ruleset',
  dqdl: glue.Dqdl.fromString('Rules = [RowCount > 10]'),
  tags: {
    key1: 'value1',
    key2: 'value2',
  },
  targetTable: new glue.DataQualityTargetTable(database.databaseName, csvTable.tableName),
});

new IntegTest(app, 'glue-data-quality-ruleset', {
  testCases: [stack],
});
