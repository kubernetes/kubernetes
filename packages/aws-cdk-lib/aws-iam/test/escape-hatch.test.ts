// tests for the L1 escape hatches (overrides). those are in the IAM module
// because we want to verify them end-to-end, as a complement to the unit
// tests in the @aws-cdk/core module
import { Template } from '../../assertions';
import { Stack, Validations } from '../../core';
import * as iam from '../lib';

/* eslint-disable @stylistic/quote-props */

function makeStackAndUser() {
  const stack = new Stack();
  Validations.of(stack).acknowledge(
    { id: 'CloudFormation-Validate::F3002', reason: "For these tests, we don't care about property names being valid" },
  );
  const user = new iam.User(stack, 'user', { userName: 'MyUserName' });
  return { stack, user };
}

describe('IAM escape hatches', () => {
  test('addPropertyOverride should allow overriding supported properties', () => {
    const { stack, user } = makeStackAndUser();
    const cfn = user.node.findChild('Resource') as iam.CfnUser;
    cfn.addPropertyOverride('UserName', 'OverriddenUserName');

    Template.fromStack(stack).templateMatches({
      'Resources': {
        'user2C2B57AE': {
          'Type': 'AWS::IAM::User',
          'Properties': {
            'UserName': 'OverriddenUserName',
          },
        },
      },
    });
  });

  test('addPropertyOverrides should allow specifying arbitrary properties', () => {
    // GIVEN
    const { stack, user } = makeStackAndUser();
    const cfn = user.node.findChild('Resource') as iam.CfnUser;

    // WHEN
    cfn.addPropertyOverride('Hello.World', 'Boom');

    // THEN
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'user2C2B57AE': {
          'Type': 'AWS::IAM::User',
          'Properties': {
            'UserName': 'MyUserName',
            'Hello': {
              'World': 'Boom',
            },
          },
        },
      },
    });
  });

  test('addOverride should allow overriding properties', () => {
    // GIVEN
    const { stack, user } = makeStackAndUser();

    Validations.of(stack).acknowledge(
      { id: 'CloudFormation-Validate::E3001', reason: "For these tests, we don't care about property names being valid" },
      { id: 'CloudFormation-Validate::E3016', reason: "For these tests, we don't care about property names being valid" },
    );

    const cfn = user.node.findChild('Resource') as iam.CfnUser;
    cfn.cfnOptions.updatePolicy = { useOnlineResharding: true };

    // WHEN
    cfn.addOverride('Properties.Hello.World', 'Bam');
    cfn.addOverride('Properties.UserName', 'HA!');
    cfn.addOverride('Joob.Jab', 'Jib');
    cfn.addOverride('Joob.Jab', 'Jib');
    cfn.addOverride('UpdatePolicy.UseOnlineResharding.Type', 'None');

    // THEN
    Template.fromStack(stack).templateMatches({
      'Resources': {
        'user2C2B57AE': {
          'Type': 'AWS::IAM::User',
          'Properties': {
            'UserName': 'HA!',
            'Hello': {
              'World': 'Bam',
            },
          },
          'Joob': {
            'Jab': 'Jib',
          },
          'UpdatePolicy': {
            'UseOnlineResharding': {
              'Type': 'None',
            },
          },
        },
      },
    });
  });
});
