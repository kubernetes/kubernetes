import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import * as glue from '../lib';

test('a data quality ruleset', () => {
  const stack = new cdk.Stack();
  new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
    description: 'description',
    rulesetName: 'ruleset_name',
    rulesetDqdl: 'ruleset_dqdl',
    targetTable: new glue.DataQualityTargetTable('database_name', 'table_name'),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::DataQualityRuleset', {
    Description: 'description',
    Name: 'ruleset_name',
    Ruleset: 'ruleset_dqdl',
    TargetTable: {
      DatabaseName: 'database_name',
      TableName: 'table_name',
    },
  });
});

test('a data quality ruleset with a client token', () => {
  const stack = new cdk.Stack();
  new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
    clientToken: 'client_token',
    description: 'description',
    rulesetName: 'ruleset_name',
    rulesetDqdl: 'ruleset_dqdl',
    targetTable: new glue.DataQualityTargetTable('database_name', 'table_name'),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::DataQualityRuleset', {
    ClientToken: 'client_token',
    Description: 'description',
    Name: 'ruleset_name',
    Ruleset: 'ruleset_dqdl',
    TargetTable: {
      DatabaseName: 'database_name',
      TableName: 'table_name',
    },
  });
});

test('a data quality ruleset with tags', () => {
  const stack = new cdk.Stack();
  new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
    clientToken: 'client_token',
    description: 'description',
    rulesetName: 'ruleset_name',
    rulesetDqdl: 'ruleset_dqdl',
    tags: {
      key1: 'value1',
      key2: 'value2',
    },
    targetTable: new glue.DataQualityTargetTable('database_name', 'table_name'),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Glue::DataQualityRuleset', {
    ClientToken: 'client_token',
    Description: 'description',
    Name: 'ruleset_name',
    Ruleset: 'ruleset_dqdl',
    Tags: {
      key1: 'value1',
      key2: 'value2',
    },
    TargetTable: {
      DatabaseName: 'database_name',
      TableName: 'table_name',
    },
  });
});

test('removalPolicy can be overridden to DESTROY', () => {
  const stack = new cdk.Stack();
  new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
    rulesetDqdl: 'ruleset_dqdl',
    targetTable: new glue.DataQualityTargetTable('database_name', 'table_name'),
    removalPolicy: cdk.RemovalPolicy.DESTROY,
  });

  Template.fromStack(stack).hasResource('AWS::Glue::DataQualityRuleset', {
    DeletionPolicy: 'Delete',
    UpdateReplacePolicy: 'Delete',
  });
});
