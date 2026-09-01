import { Capture, Template } from '../../assertions';
import type { CfnElement } from '../../core';
import { App, Aws, CfnResource, Lazy, Stack } from '../../core';
import {
  IAM_IMPORTED_ROLE_STACK_SAFE_DEFAULT_POLICY_NAME,
} from '../../cx-api';
import type { IRole } from '../lib';
import { AnyPrincipal, ArnPrincipal, Grant, Policy, PolicyStatement, Role } from '../lib';

/* eslint-disable @stylistic/quote-props */

const roleAccount = '123456789012';
const notRoleAccount = '012345678901';

describe('IAM Role.fromRoleArn', () => {
  let app: App;

  beforeEach(() => {
    app = new App();
  });

  let roleStack: Stack;
  let importedRole: IRole;

  describe('imported with a static ARN', () => {
    const roleName = 'MyRole';

    describe('into an env-agnostic stack', () => {
      beforeEach(() => {
        roleStack = new Stack(app, 'RoleStack');
        importedRole = Role.fromRoleArn(roleStack, 'ImportedRole',
          `arn:aws:iam::${roleAccount}:role/${roleName}`);
      });

      test('correctly parses the imported role ARN', () => {
        expect(importedRole.roleArn).toBe(`arn:aws:iam::${roleAccount}:role/${roleName}`);
      });

      test('correctly parses the imported role name', () => {
        expect(importedRole.roleName).toBe(roleName);
      });

      describe('then adding a PolicyStatement to it', () => {
        let addToPolicyResult: boolean;

        beforeEach(() => {
          addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
        });

        test('returns true', () => {
          expect(addToPolicyResult).toBe(true);
        });

        test("generates a default Policy resource pointing at the imported role's physical name", () => {
          assertRoleHasDefaultPolicy(roleStack, roleName);
        });
      });

      describe('future flag: @aws-cdk/aws-iam:importedRoleStackSafeDefaultPolicyName', () => {
        test('the same role imported in different stacks has different default policy names', () => {
          const appWithFeatureFlag = new App({ context: { [IAM_IMPORTED_ROLE_STACK_SAFE_DEFAULT_POLICY_NAME]: true } });
          const roleStack1 = new Stack(appWithFeatureFlag, 'RoleStack1');
          const importedRoleInStack1 = Role.fromRoleArn(roleStack1, 'ImportedRole',
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
          importedRoleInStack1.addToPrincipalPolicy(somePolicyStatement());

          const roleStack2 = new Stack(appWithFeatureFlag, 'RoleStack2');
          const importedRoleInStack2 = Role.fromRoleArn(roleStack2, 'ImportedRole',
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
          importedRoleInStack2.addToPrincipalPolicy(somePolicyStatement());

          const stack1PolicyNameCapture = new Capture();
          Template.fromStack(roleStack1).hasResourceProperties('AWS::IAM::Policy', { PolicyName: stack1PolicyNameCapture });

          const stack2PolicyNameCapture = new Capture();
          Template.fromStack(roleStack2).hasResourceProperties('AWS::IAM::Policy', { PolicyName: stack2PolicyNameCapture });

          expect(stack1PolicyNameCapture.asString()).not.toBe(stack2PolicyNameCapture.asString());
          expect(stack1PolicyNameCapture.asString()).toMatch(/PolicyRoleStack1ImportedRole.*/);
          expect(stack2PolicyNameCapture.asString()).toMatch(/PolicyRoleStack2ImportedRole.*/);
        });

        test('Policy name is truncated to a maximum length of 128 characters when the original name exceeds this limit', () => {
          const appWithFeatureFlag = new App({ context: { [IAM_IMPORTED_ROLE_STACK_SAFE_DEFAULT_POLICY_NAME]: true } });
          const roleStack1 = new Stack(appWithFeatureFlag, 'RoleStack1');

          const tooLongString = 'x'.repeat(150);
          const importedRoleInStack = Role.fromRoleArn(roleStack1, `ImportedRole${tooLongString}`,
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
          importedRoleInStack.addToPrincipalPolicy(somePolicyStatement());

          const stackPolicyNameCapture = new Capture();
          Template.fromStack(roleStack1).hasResourceProperties('AWS::IAM::Policy', { PolicyName: stackPolicyNameCapture });

          expect(stackPolicyNameCapture.asString()).toHaveLength(128);
        });

        test('the same role imported in different stacks has the same default policy name without flag', () => {
          const appWithoutFeatureFlag = new App({ context: { [IAM_IMPORTED_ROLE_STACK_SAFE_DEFAULT_POLICY_NAME]: false } });
          const roleStack1 = new Stack(appWithoutFeatureFlag, 'RoleStack1');
          const importedRoleInStack1 = Role.fromRoleArn(roleStack1, 'ImportedRole',
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
          importedRoleInStack1.addToPrincipalPolicy(somePolicyStatement());

          const roleStack2 = new Stack(appWithoutFeatureFlag, 'RoleStack2');
          const importedRoleInStack2 = Role.fromRoleArn(roleStack2, 'ImportedRole',
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
          importedRoleInStack2.addToPrincipalPolicy(somePolicyStatement());

          const stack1PolicyNameCapture = new Capture();
          Template.fromStack(roleStack1).hasResourceProperties('AWS::IAM::Policy', { PolicyName: stack1PolicyNameCapture });

          const stack2PolicyNameCapture = new Capture();
          Template.fromStack(roleStack2).hasResourceProperties('AWS::IAM::Policy', { PolicyName: stack2PolicyNameCapture });

          expect(stack1PolicyNameCapture.asString()).toBe(stack2PolicyNameCapture.asString());
          expect(stack1PolicyNameCapture.asString()).toMatch(/ImportedRolePolicy.{8}/);
        });
      });

      describe('then attaching a Policy to it', () => {
        let policyStack: Stack;

        describe('that belongs to the same stack as the imported role', () => {
          beforeEach(() => {
            importedRole.attachInlinePolicy(somePolicy(roleStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(roleStack, roleName, 'MyPolicy');
          });
        });

        describe('that belongs to a different env-agnostic stack', () => {
          beforeEach(() => {
            policyStack = new Stack(app, 'PolicyStack');
            importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
          });
        });

        describe('that belongs to a targeted stack, with account set to', () => {
          describe('the same account as in the ARN of the imported role', () => {
            beforeEach(() => {
              policyStack = new Stack(app, 'PolicyStack', { env: { account: roleAccount } });
              importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
            });

            test('correctly attaches the Policy to the imported role', () => {
              assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
            });
          });

          describe('a different account than in the ARN of the imported role', () => {
            beforeEach(() => {
              policyStack = new Stack(app, 'PolicyStack', { env: { account: notRoleAccount } });
              importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
            });

            test('does NOT attach the Policy to the imported role', () => {
              assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
            });
          });
        });
      });
    });

    describe('into a targeted stack with account set to', () => {
      describe('the same account as in the ARN the role was imported with', () => {
        beforeEach(() => {
          roleStack = new Stack(app, 'RoleStack', { env: { account: roleAccount } });
          importedRole = Role.fromRoleArn(roleStack, 'ImportedRole',
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
        });

        describe('then adding a PolicyStatement to it', () => {
          let addToPolicyResult: boolean;

          beforeEach(() => {
            addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
          });

          test('returns true', () => {
            expect(addToPolicyResult).toBe(true);
          });

          test("generates a default Policy resource pointing at the imported role's physical name", () => {
            assertRoleHasDefaultPolicy(roleStack, roleName);
          });
        });

        describe('then attaching a Policy to it', () => {
          describe('that belongs to the same stack as the imported role', () => {
            beforeEach(() => {
              importedRole.attachInlinePolicy(somePolicy(roleStack, 'MyPolicy'));
            });

            test('correctly attaches the Policy to the imported role', () => {
              assertRoleHasAttachedPolicy(roleStack, roleName, 'MyPolicy');
            });
          });

          describe('that belongs to an env-agnostic stack', () => {
            let policyStack: Stack;

            beforeEach(() => {
              policyStack = new Stack(app, 'PolicyStack');
              importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
            });

            test('correctly attaches the Policy to the imported role', () => {
              assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
            });
          });

          describe('that belongs to a targeted stack, with account set to', () => {
            let policyStack: Stack;

            describe('the same account as in the imported role ARN and in the stack the imported role belongs to', () => {
              beforeEach(() => {
                policyStack = new Stack(app, 'PolicyStack', { env: { account: roleAccount } });
                importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
              });

              test('correctly attaches the Policy to the imported role', () => {
                assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
              });
            });

            describe('a different account than in the imported role ARN and in the stack the imported role belongs to', () => {
              beforeEach(() => {
                policyStack = new Stack(app, 'PolicyStack', { env: { account: notRoleAccount } });
                importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
              });

              test('does NOT attach the Policy to the imported role', () => {
                assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
              });
            });
          });
        });
      });

      describe('a different account than in the ARN the role was imported with', () => {
        beforeEach(() => {
          roleStack = new Stack(app, 'RoleStack', { env: { account: notRoleAccount } });
          importedRole = Role.fromRoleArn(roleStack, 'ImportedRole',
            `arn:aws:iam::${roleAccount}:role/${roleName}`);
        });

        describe('then adding a PolicyStatement to it', () => {
          let addToPolicyResult: boolean;

          beforeEach(() => {
            addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
          });

          test('pretends to succeed', () => {
            expect(addToPolicyResult).toBe(true);
          });

          test("does NOT generate a default Policy resource pointing at the imported role's physical name", () => {
            Template.fromStack(roleStack).resourceCountIs('AWS::IAM::Policy', 0);
          });
        });

        describe('then attaching a Policy to it', () => {
          describe('that belongs to the same stack as the imported role', () => {
            beforeEach(() => {
              importedRole.attachInlinePolicy(somePolicy(roleStack, 'MyPolicy'));
            });

            test('does NOT attach the Policy to the imported role', () => {
              assertPolicyDidNotAttachToRole(roleStack, 'MyPolicy');
            });
          });

          describe('that belongs to an env-agnostic stack', () => {
            let policyStack: Stack;

            beforeEach(() => {
              policyStack = new Stack(app, 'PolicyStack');
              importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
            });

            test('does NOT attach the Policy to the imported role', () => {
              assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
            });
          });

          describe('that belongs to a different targeted stack, with account set to', () => {
            let policyStack: Stack;

            describe('the same account as in the ARN of the imported role', () => {
              beforeEach(() => {
                policyStack = new Stack(app, 'PolicyStack', { env: { account: roleAccount } });
                importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
              });

              test('does NOT attach the Policy to the imported role', () => {
                assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
              });
            });

            describe('the same account as in the stack the imported role belongs to', () => {
              beforeEach(() => {
                policyStack = new Stack(app, 'PolicyStack', { env: { account: notRoleAccount } });
                importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
              });

              test('does NOT attach the Policy to the imported role', () => {
                assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
              });
            });

            describe('a third account, different from both the role and scope stack accounts', () => {
              beforeEach(() => {
                policyStack = new Stack(app, 'PolicyStack', { env: { account: 'some-random-account' } });
                importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
              });

              test('does NOT attach the Policy to the imported role', () => {
                assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
              });
            });
          });
        });
      });
    });

    describe('and with mutable=false', () => {
      beforeEach(() => {
        roleStack = new Stack(app, 'RoleStack');
        importedRole = Role.fromRoleArn(roleStack, 'ImportedRole',
          `arn:aws:iam::${roleAccount}:role/${roleName}`, { mutable: false });
      });

      describe('then adding a PolicyStatement to it', () => {
        let addToPolicyResult: boolean;

        beforeEach(() => {
          addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
        });

        test('pretends to succeed', () => {
          expect(addToPolicyResult).toBe(true);
        });

        test("does NOT generate a default Policy resource pointing at the imported role's physical name", () => {
          Template.fromStack(roleStack).resourceCountIs('AWS::IAM::Policy', 0);
        });
      });

      describe('then attaching a Policy to it', () => {
        let policyStack: Stack;

        describe('that belongs to a stack with account equal to the account in the imported role ARN', () => {
          beforeEach(() => {
            policyStack = new Stack(app, 'PolicyStack', { env: { account: roleAccount } });
            importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
          });

          test('does NOT attach the Policy to the imported role', () => {
            assertPolicyDidNotAttachToRole(policyStack, 'MyPolicy');
          });
        });
      });
    });

    describe('and with mutable=false and addGrantsToResources=true', () => {
      beforeEach(() => {
        roleStack = new Stack(app, 'RoleStack');
        importedRole = Role.fromRoleArn(roleStack, 'ImportedRole',
          `arn:aws:iam::${roleAccount}:role/${roleName}`, { mutable: false, addGrantsToResources: true });
      });

      describe('then adding a PolicyStatement to it', () => {
        let addToPolicyResult: boolean;

        beforeEach(() => {
          addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
        });

        test('pretends to fail', () => {
          expect(addToPolicyResult).toBe(false);
        });

        test("does NOT generate a default Policy resource pointing at the imported role's physical name", () => {
          Template.fromStack(roleStack).resourceCountIs('AWS::IAM::Policy', 0);
        });
      });
    });

    describe('imported with a user specified default policy name', () => {
      test('user specified default policy is used when fromRoleArn() creates a default policy', () => {
        roleStack = new Stack(app, 'RoleStack');
        new CfnResource(roleStack, 'SomeResource', {
          type: 'CDK::Test::SomeResource',
        });
        importedRole = Role.fromRoleArn(roleStack, 'ImportedRole',
          `arn:aws:iam::${roleAccount}:role/${roleName}`, { defaultPolicyName: 'UserSpecifiedDefaultPolicy' });

        Grant.addToPrincipal({
          actions: ['service:DoAThing'],
          grantee: importedRole,
          resourceArns: ['*'],
        });

        Template.fromStack(roleStack).templateMatches({
          Resources: {
            ImportedRoleUserSpecifiedDefaultPolicy7CBF6E85: {
              Type: 'AWS::IAM::Policy',
              Properties: {
                PolicyName: 'ImportedRoleUserSpecifiedDefaultPolicy7CBF6E85',
              },
            },
          },
        });
      });
    });

    test('`fromRoleName()` with options matches behavior of `fromRoleArn()`', () => {
      roleStack = new Stack(app, 'RoleStack');
      new CfnResource(roleStack, 'SomeResource', {
        type: 'CDK::Test::SomeResource',
      });
      importedRole = Role.fromRoleName(roleStack, 'ImportedRole',
        `${roleName}`, { defaultPolicyName: 'UserSpecifiedDefaultPolicy' });

      Grant.addToPrincipal({
        actions: ['service:DoAThing'],
        grantee: importedRole,
        resourceArns: ['*'],
      });

      Template.fromStack(roleStack).templateMatches({
        Resources: {
          ImportedRoleUserSpecifiedDefaultPolicy7CBF6E85: {
            Type: 'AWS::IAM::Policy',
            Properties: {
              PolicyName: 'ImportedRoleUserSpecifiedDefaultPolicy7CBF6E85',
            },
          },
        },
      });
    });
  });

  describe('imported with a dynamic ARN', () => {
    const dynamicValue = Lazy.string({ produce: () => 'role-arn' });
    const roleName: any = {
      'Fn::Select': [1,
        {
          'Fn::Split': ['/',
            {
              'Fn::Select': [5,
                { 'Fn::Split': [':', 'role-arn'] }],
            }],
        }],
    };

    describe('into an env-agnostic stack', () => {
      beforeEach(() => {
        roleStack = new Stack(app, 'RoleStack');
        importedRole = Role.fromRoleArn(roleStack, 'ImportedRole', dynamicValue);
      });

      test('correctly parses the imported role ARN', () => {
        expect(importedRole.roleArn).toBe(dynamicValue);
      });

      test('correctly parses the imported role name', () => {
        new Role(roleStack, 'AnyRole', {
          roleName: 'AnyRole',
          assumedBy: new ArnPrincipal(importedRole.roleName),
        });

        Template.fromStack(roleStack).hasResourceProperties('AWS::IAM::Role', {
          'AssumeRolePolicyDocument': {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'AWS': roleName,
                },
              },
            ],
          },
        });
      });

      describe('then adding a PolicyStatement to it', () => {
        let addToPolicyResult: boolean;

        beforeEach(() => {
          addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
        });

        test('returns true', () => {
          expect(addToPolicyResult).toBe(true);
        });

        test("generates a default Policy resource pointing at the imported role's physical name", () => {
          assertRoleHasDefaultPolicy(roleStack, roleName);
        });
      });

      describe('then attaching a Policy to it', () => {
        let policyStack: Stack;

        describe('that belongs to the same stack as the imported role', () => {
          beforeEach(() => {
            importedRole.attachInlinePolicy(somePolicy(roleStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(roleStack, roleName, 'MyPolicy');
          });
        });

        describe('that belongs to a different env-agnostic stack', () => {
          beforeEach(() => {
            policyStack = new Stack(app, 'PolicyStack');
            importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
          });
        });

        describe('that belongs to a targeted stack', () => {
          beforeEach(() => {
            policyStack = new Stack(app, 'PolicyStack', { env: { account: roleAccount } });
            importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
          });
        });
      });
    });

    describe('into a targeted stack with account set', () => {
      beforeEach(() => {
        roleStack = new Stack(app, 'RoleStack', { env: { account: roleAccount } });
        importedRole = Role.fromRoleArn(roleStack, 'ImportedRole', dynamicValue);
      });

      describe('then adding a PolicyStatement to it', () => {
        let addToPolicyResult: boolean;

        beforeEach(() => {
          addToPolicyResult = importedRole.addToPolicy(somePolicyStatement());
        });

        test('returns true', () => {
          expect(addToPolicyResult).toBe(true);
        });

        test("generates a default Policy resource pointing at the imported role's physical name", () => {
          assertRoleHasDefaultPolicy(roleStack, roleName);
        });
      });

      describe('then attaching a Policy to it', () => {
        let policyStack: Stack;

        describe('that belongs to the same stack as the imported role', () => {
          beforeEach(() => {
            importedRole.attachInlinePolicy(somePolicy(roleStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(roleStack, roleName, 'MyPolicy');
          });
        });

        describe('that belongs to an env-agnostic stack', () => {
          beforeEach(() => {
            policyStack = new Stack(app, 'PolicyStack');
            importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
          });

          test('correctly attaches the Policy to the imported role', () => {
            assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
          });
        });

        describe('that belongs to a different targeted stack, with account set to', () => {
          describe('the same account as the stack the role was imported into', () => {
            beforeEach(() => {
              policyStack = new Stack(app, 'PolicyStack', { env: { account: roleAccount } });
              importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
            });

            test('correctly attaches the Policy to the imported role', () => {
              assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
            });
          });

          describe('a different account than the stack the role was imported into', () => {
            beforeEach(() => {
              policyStack = new Stack(app, 'PolicyStack', { env: { account: notRoleAccount } });
              importedRole.attachInlinePolicy(somePolicy(policyStack, 'MyPolicy'));
            });

            test('correctly attaches the Policy to the imported role', () => {
              assertRoleHasAttachedPolicy(policyStack, roleName, 'MyPolicy');
            });
          });
        });
      });
    });
  });

  describe('imported with the ARN of a service role', () => {
    beforeEach(() => {
      roleStack = new Stack();
    });

    describe('without a service principal in the role name', () => {
      beforeEach(() => {
        importedRole = Role.fromRoleArn(roleStack, 'Role',
          `arn:aws:iam::${roleAccount}:role/service-role/codebuild-role`);
      });

      it("correctly strips the 'service-role' prefix from the role name", () => {
        new Policy(roleStack, 'Policy', {
          statements: [somePolicyStatement()],
          roles: [importedRole],
        });

        Template.fromStack(roleStack).hasResourceProperties('AWS::IAM::Policy', {
          'Roles': [
            'codebuild-role',
          ],
        });
      });
    });

    describe('with a service principal in the role name', () => {
      beforeEach(() => {
        importedRole = Role.fromRoleArn(roleStack, 'Role',
          `arn:aws:iam::${roleAccount}:role/aws-service-role/anyservice.amazonaws.com/codebuild-role`);
      });

      it("correctly strips both the 'aws-service-role' prefix and the service principal from the role name", () => {
        new Policy(roleStack, 'Policy', {
          statements: [somePolicyStatement()],
          roles: [importedRole],
        });

        Template.fromStack(roleStack).hasResourceProperties('AWS::IAM::Policy', {
          'Roles': [
            'codebuild-role',
          ],
        });
      });
    });
  });

  describe('for an incorrect ARN', () => {
    beforeEach(() => {
      roleStack = new Stack(app, 'RoleStack');
    });

    describe("that accidentally skipped the 'region' fragment of the ARN", () => {
      test('throws an exception, indicating that error', () => {
        expect(() => {
          Role.fromRoleArn(roleStack, 'Role',
            `arn:${Aws.PARTITION}:iam:${Aws.ACCOUNT_ID}:role/AwsCicd-${Aws.REGION}-CodeBuildRole`);
        }).toThrow(/The `resource` component \(6th component\) of an ARN is required:/);
      });
    });
  });
});

test('Role.fromRoleName with no options ', () => {
  const app = new App();
  const stack = new Stack(app, 'Stack', { env: { region: 'asdf', account: '1234' } });
  const role = Role.fromRoleName(stack, 'MyRole', 'MyRole');

  expect(stack.resolve(role.roleArn)).toEqual({ 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::1234:role/MyRole']] });
});

function somePolicyStatement() {
  return new PolicyStatement({
    actions: ['s3:*'],
    resources: ['arn:aws:s3:::my-bucket'],
  });
}

function somePolicy(policyStack: Stack, policyName: string) {
  const someRole = new Role(policyStack, 'SomeExampleRole', {
    assumedBy: new AnyPrincipal(),
  });
  const roleResource = someRole.node.defaultChild as CfnElement;
  roleResource.overrideLogicalId('SomeRole'); // force a particular logical ID in the Ref expression

  return new Policy(policyStack, 'MyPolicy', {
    policyName,
    statements: [somePolicyStatement()],
    // need at least one of user/group/role, otherwise validation fails
    roles: [someRole],
  });
}

function assertRoleHasDefaultPolicy(stack: Stack, roleName: string) {
  _assertStackContainsPolicyResource(stack, [roleName], undefined);
}

function assertRoleHasAttachedPolicy(stack: Stack, roleName: string, attachedPolicyName: string) {
  _assertStackContainsPolicyResource(stack, [{ Ref: 'SomeRole' }, roleName], attachedPolicyName);
}

function assertPolicyDidNotAttachToRole(stack: Stack, policyName: string) {
  _assertStackContainsPolicyResource(stack, [{ Ref: 'SomeRole' }], policyName);
}

function _assertStackContainsPolicyResource(stack: Stack, roleNames: any[], nameOfPolicy: string | undefined) {
  const expected: any = {
    PolicyDocument: {
      Statement: [
        {
          Action: 's3:*',
          Effect: 'Allow',
          Resource: 'arn:aws:s3:::my-bucket',
        },
      ],
    },
    Roles: roleNames,
  };
  if (nameOfPolicy) {
    expected.PolicyName = nameOfPolicy;
  }

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', expected);
}
