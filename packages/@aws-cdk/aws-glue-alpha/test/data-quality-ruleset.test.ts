import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import * as glue from '../lib';

test('a data quality ruleset', () => {
  const stack = new cdk.Stack();
  new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
    description: 'description',
    rulesetName: 'ruleset_name',
    dqdl: glue.Dqdl.fromString('ruleset_dqdl'),
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
    dqdl: glue.Dqdl.fromString('ruleset_dqdl'),
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
    dqdl: glue.Dqdl.fromString('ruleset_dqdl'),
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
    dqdl: glue.Dqdl.fromString('ruleset_dqdl'),
    targetTable: new glue.DataQualityTargetTable('database_name', 'table_name'),
    removalPolicy: cdk.RemovalPolicy.DESTROY,
  });

  Template.fromStack(stack).hasResource('AWS::Glue::DataQualityRuleset', {
    DeletionPolicy: 'Delete',
    UpdateReplacePolicy: 'Delete',
  });
});

describe('import', () => {
  test('fromRulesetName derives the ARN from the name', () => {
    const stack = new cdk.Stack();

    const imported = glue.DataQualityRuleset.fromRulesetName(stack, 'Imported', 'my_ruleset');

    expect(imported.rulesetName).toEqual('my_ruleset');
    expect(imported.rulesetArn).toEqual(stack.formatArn({
      service: 'glue',
      resource: 'dataqualityruleset',
      resourceName: 'my_ruleset',
    }));
  });

  test('fromRulesetArn extracts the name from the ARN', () => {
    const stack = new cdk.Stack();
    const rulesetArn = 'arn:aws:glue:us-east-1:123456789012:dataqualityruleset/my_ruleset';

    const imported = glue.DataQualityRuleset.fromRulesetArn(stack, 'Imported', rulesetArn);

    expect(imported.rulesetArn).toEqual(rulesetArn);
    expect(imported.rulesetName).toEqual('my_ruleset');
  });
});

test('exposes the ruleset name and ARN of a created ruleset', () => {
  const stack = new cdk.Stack();
  const ruleset = new glue.DataQualityRuleset(stack, 'DataQualityRuleset', {
    rulesetName: 'ruleset_name',
    dqdl: glue.Dqdl.fromString('ruleset_dqdl'),
    targetTable: new glue.DataQualityTargetTable('database_name', 'table_name'),
  });

  // The name getter returns an environment-sensitive token, so the ARN getter
  // renders as an Fn::Join. Assert it is built under the glue dataqualityruleset
  // resource and equals buildRulesetArn(this, rulesetName).
  expect(ruleset.rulesetName).toBeDefined();
  expect(stack.resolve(ruleset.rulesetArn)).toEqual(stack.resolve(stack.formatArn({
    service: 'glue',
    resource: 'dataqualityruleset',
    resourceName: ruleset.rulesetName,
  })));
});
