import { Template } from '../../assertions';
import { Stack } from '../../core';
import { ReceiptRule, ReceiptRuleSet, TlsPolicy } from '../lib';

/* eslint-disable @stylistic/quote-props */

describe('receipt rule', () => {
  test('can create receipt rules with second after first', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new ReceiptRuleSet(stack, 'RuleSet', {
      rules: [
        {
          receiptRuleName: 'FirstRule',
        },
        {
          enabled: false,
          receiptRuleName: 'SecondRule',
          recipients: ['hello@aws.com'],
          scanEnabled: true,
          tlsPolicy: TlsPolicy.REQUIRE,
        },
      ],
    });

    // THEN
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'RuleSetE30C6C48': {
          'Type': 'AWS::SES::ReceiptRuleSet',
        },
        'RuleSetRule023C3B8E1': {
          'Type': 'AWS::SES::ReceiptRule',
          'Properties': {
            'Rule': {
              'Name': 'FirstRule',
              'Enabled': true,
            },
            'RuleSetName': {
              'Ref': 'RuleSetE30C6C48',
            },
          },
        },
        'RuleSetRule117041B57': {
          'Type': 'AWS::SES::ReceiptRule',
          'Properties': {
            'Rule': {
              'Enabled': false,
              'Name': 'SecondRule',
              'Recipients': [
                'hello@aws.com',
              ],
              'ScanEnabled': true,
              'TlsPolicy': 'Require',
            },
            'RuleSetName': {
              'Ref': 'RuleSetE30C6C48',
            },
            'After': {
              'Ref': 'RuleSetRule023C3B8E1',
            },
          },
        },
      },
    });
  });

  test('import receipt rule', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const receiptRule = ReceiptRule.fromReceiptRuleName(stack, 'ImportedRule', 'MyRule');
    const receiptRuleSet = new ReceiptRuleSet(stack, 'RuleSet');

    receiptRuleSet.addRule('MyRule', {
      after: receiptRule,
    });

    // THEN
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'RuleSetE30C6C48': {
          'Type': 'AWS::SES::ReceiptRuleSet',
        },
        'RuleSetMyRule60B1D107': {
          'Type': 'AWS::SES::ReceiptRule',
          'Properties': {
            'Rule': {
              'Enabled': true,
            },
            'RuleSetName': {
              'Ref': 'RuleSetE30C6C48',
            },
            'After': 'MyRule',
          },
        },
      },
    });
  });

  test('fails to read ruleSetName of a rule imported by name', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const rule = ReceiptRule.fromReceiptRuleName(stack, 'Rule', 'MyRule');

    // THEN
    expect(() => rule.receiptRuleRef.ruleSetName).toThrow(
      'ruleSetName is not available on a ReceiptRule imported by rule name; use ReceiptRule.fromReceiptRuleAttributes() to get a complete reference',
    );
  });

  test('receiptRuleRef of a rule imported by attributes is fully readable', () => {
    // GIVEN
    const stack = new Stack();
    const ruleSet = ReceiptRuleSet.fromReceiptRuleSetName(stack, 'RuleSet', 'MyRuleSet');

    // WHEN
    const rule = ReceiptRule.fromReceiptRuleAttributes(stack, 'Rule', {
      ruleSet,
      receiptRuleName: 'MyRule',
    });

    // THEN
    expect(rule.receiptRuleRef.ruleName).toEqual('MyRule');
    expect(rule.receiptRuleRef.ruleSetName).toEqual('MyRuleSet');
  });

  test('can add actions in rule props', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const ruleSet = new ReceiptRuleSet(stack, 'RuleSet');
    ruleSet.addRule('Rule', {
      actions: [
        {
          bind: () => ({
            stopAction: {
              scope: 'RuleSet',
            },
          }),
        },
      ],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SES::ReceiptRule', {
      'Rule': {
        'Actions': [
          {
            'StopAction': {
              'Scope': 'RuleSet',
            },
          },
        ],
        'Enabled': true,
      },
      'RuleSetName': {
        'Ref': 'RuleSetE30C6C48',
      },
    });
  });

  test('can add action with addAction', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const ruleSet = new ReceiptRuleSet(stack, 'RuleSet');
    const rule = ruleSet.addRule('Rule');
    rule.addAction({
      bind: () => ({
        stopAction: {
          scope: 'RuleSet',
        },
      }),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SES::ReceiptRule', {
      'Rule': {
        'Actions': [
          {
            'StopAction': {
              'Scope': 'RuleSet',
            },
          },
        ],
        'Enabled': true,
      },
      'RuleSetName': {
        'Ref': 'RuleSetE30C6C48',
      },
    });
  });
});
