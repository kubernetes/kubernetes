import { Template } from '../../assertions';
import * as lambda from '../../aws-lambda';
import * as s3 from '../../aws-s3';
import { App, CfnResource, Resource, Stack } from '../../core';
import type { AddToPrincipalPolicyResult, CfnPolicy, IResourceWithPolicy } from '../lib';
import { AnyPrincipal, Grant, Group, Policy, PolicyDocument, PolicyStatement, Role, ServicePrincipal, User } from '../lib';

/* eslint-disable @stylistic/quote-props */

describe('IAM policy', () => {
  let app: App;
  let stack: Stack;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app, 'MyStack');
  });

  test('fails when "forced" policy is empty', () => {
    new Policy(stack, 'MyPolicy', { force: true });

    expect(() => app.synth()).toThrow(/is empty/);
  });

  test('policy with statements', () => {
    const policy = new Policy(stack, 'MyPolicy', { policyName: 'MyPolicyName' });
    policy.addStatements(new PolicyStatement({ resources: ['*'], actions: ['sqs:SendMessage'] }));
    policy.addStatements(new PolicyStatement({ resources: ['arn:aws:sns:us-east-1:123456789012:my-topic'], actions: ['sns:Subscribe'] }));

    const group = new Group(stack, 'MyGroup');
    group.attachInlinePolicy(policy);

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        MyPolicy39D66CF6:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            Groups: [{ Ref: 'MyGroupCBA54B1B' }],
            PolicyDocument:
           {
             Statement:
            [{ Action: 'sqs:SendMessage', Effect: 'Allow', Resource: '*' },
              { Action: 'sns:Subscribe', Effect: 'Allow', Resource: 'arn:aws:sns:us-east-1:123456789012:my-topic' }],
             Version: '2012-10-17',
           },
            PolicyName: 'MyPolicyName',
          },
         },
        MyGroupCBA54B1B: { Type: 'AWS::IAM::Group' },
      },
    });
  });

  test('policy from policy document alone', () => {
    const policy = new Policy(stack, 'MyPolicy', {
      policyName: 'MyPolicyName',
      document: PolicyDocument.fromJson({
        Statement: [
          {
            Action: 'sqs:SendMessage',
            Effect: 'Allow',
            Resource: '*',
          },
        ],
      }),
    });

    const group = new Group(stack, 'MyGroup');
    group.attachInlinePolicy(policy);

    Template.fromStack(stack).templateMatches({
      Resources: {
        MyPolicy39D66CF6: {
          Type: 'AWS::IAM::Policy',
          Properties: {
            PolicyName: 'MyPolicyName',
            Groups: [{ Ref: 'MyGroupCBA54B1B' }],
            PolicyDocument: {
              Statement: [
                { Action: 'sqs:SendMessage', Effect: 'Allow', Resource: '*' },
              ],
              Version: '2012-10-17',
            },
          },
        },
        MyGroupCBA54B1B: { Type: 'AWS::IAM::Group' },
      },
    });
  });
  test('policy name can be omitted, in which case the logical id will be used', () => {
    const policy = new Policy(stack, 'MyPolicy');
    policy.addStatements(new PolicyStatement({ resources: ['*'], actions: ['sqs:SendMessage'] }));
    policy.addStatements(new PolicyStatement({ resources: ['arn:aws:sns:us-east-1:123456789012:my-topic'], actions: ['sns:Subscribe'] }));

    const user = new User(stack, 'MyUser');
    user.attachInlinePolicy(policy);

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        MyPolicy39D66CF6:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            PolicyDocument:
           {
             Statement:
            [{ Action: 'sqs:SendMessage', Effect: 'Allow', Resource: '*' },
              { Action: 'sns:Subscribe', Effect: 'Allow', Resource: 'arn:aws:sns:us-east-1:123456789012:my-topic' }],
             Version: '2012-10-17',
           },
            PolicyName: 'MyPolicy39D66CF6',
            Users: [{ Ref: 'MyUserDC45028B' }],
          },
         },
        MyUserDC45028B: { Type: 'AWS::IAM::User' },
      },
    });
  });

  test('policy can be attached users, groups and roles and added permissions via props', () => {
    const user1 = new User(stack, 'User1');
    const group1 = new Group(stack, 'Group1');
    const role1 = new Role(stack, 'Role1', {
      assumedBy: new ServicePrincipal('test.service'),
    });

    new Policy(stack, 'MyTestPolicy', {
      policyName: 'Foo',
      users: [user1],
      groups: [group1],
      roles: [role1],
      statements: [new PolicyStatement({ resources: ['*'], actions: ['dynamodb:PutItem'] })],
    });

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        User1E278A736: { Type: 'AWS::IAM::User' },
        Group1BEBD4686: { Type: 'AWS::IAM::Group' },
        Role13A5C70C1:
         {
           Type: 'AWS::IAM::Role',
           Properties:
          {
            AssumeRolePolicyDocument:
           {
             Statement:
            [{
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'test.service' },
            }],
             Version: '2012-10-17',
           },
          },
         },
        MyTestPolicy316BDB50:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            Groups: [{ Ref: 'Group1BEBD4686' }],
            PolicyDocument:
           {
             Statement:
            [{ Action: 'dynamodb:PutItem', Effect: 'Allow', Resource: '*' }],
             Version: '2012-10-17',
           },
            PolicyName: 'Foo',
            Roles: [{ Ref: 'Role13A5C70C1' }],
            Users: [{ Ref: 'User1E278A736' }],
          },
         },
      },
    });
  });

  test('idempotent if a principal (user/group/role) is attached twice', () => {
    const p = new Policy(stack, 'MyPolicy');
    p.addStatements(new PolicyStatement({ actions: ['*'], resources: ['*'] }));

    const user = new User(stack, 'MyUser');
    p.attachToUser(user);
    p.attachToUser(user);

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        MyPolicy39D66CF6:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            PolicyDocument:
           {
             Statement: [{ Action: '*', Effect: 'Allow', Resource: '*' }],
             Version: '2012-10-17',
           },
            PolicyName: 'MyPolicy39D66CF6',
            Users: [{ Ref: 'MyUserDC45028B' }],
          },
         },
        MyUserDC45028B: { Type: 'AWS::IAM::User' },
      },
    });
  });

  test('users, groups, roles and permissions can be added using methods', () => {
    const p = new Policy(stack, 'MyTestPolicy', {
      policyName: 'Foo',
    });

    p.attachToUser(new User(stack, 'User1'));
    p.attachToUser(new User(stack, 'User2'));
    p.attachToGroup(new Group(stack, 'Group1'));
    p.attachToRole(new Role(stack, 'Role1', { assumedBy: new ServicePrincipal('test.service') }));
    p.addStatements(new PolicyStatement({ resources: ['*'], actions: ['dynamodb:GetItem'] }));

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        MyTestPolicy316BDB50:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            Groups: [{ Ref: 'Group1BEBD4686' }],
            PolicyDocument:
           {
             Statement:
            [{ Action: 'dynamodb:GetItem', Effect: 'Allow', Resource: '*' }],
             Version: '2012-10-17',
           },
            PolicyName: 'Foo',
            Roles: [{ Ref: 'Role13A5C70C1' }],
            Users: [{ Ref: 'User1E278A736' }, { Ref: 'User21F1486D1' }],
          },
         },
        User1E278A736: { Type: 'AWS::IAM::User' },
        User21F1486D1: { Type: 'AWS::IAM::User' },
        Group1BEBD4686: { Type: 'AWS::IAM::Group' },
        Role13A5C70C1:
         {
           Type: 'AWS::IAM::Role',
           Properties:
          {
            AssumeRolePolicyDocument:
           {
             Statement:
            [{
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'test.service' },
            }],
             Version: '2012-10-17',
           },
          },
         },
      },
    });
  });

  test('policy can be attached to users, groups or role via methods on the principal', () => {
    const policy = new Policy(stack, 'MyPolicy');
    const user = new User(stack, 'MyUser');
    const group = new Group(stack, 'MyGroup');
    const role = new Role(stack, 'MyRole', { assumedBy: new ServicePrincipal('test.service') });

    user.attachInlinePolicy(policy);
    group.attachInlinePolicy(policy);
    role.attachInlinePolicy(policy);

    policy.addStatements(new PolicyStatement({ resources: ['*'], actions: ['*'] }));

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        MyPolicy39D66CF6:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            Groups: [{ Ref: 'MyGroupCBA54B1B' }],
            PolicyDocument:
           {
             Statement: [{ Action: '*', Effect: 'Allow', Resource: '*' }],
             Version: '2012-10-17',
           },
            PolicyName: 'MyPolicy39D66CF6',
            Roles: [{ Ref: 'MyRoleF48FFE04' }],
            Users: [{ Ref: 'MyUserDC45028B' }],
          },
         },
        MyUserDC45028B: { Type: 'AWS::IAM::User' },
        MyGroupCBA54B1B: { Type: 'AWS::IAM::Group' },
        MyRoleF48FFE04:
         {
           Type: 'AWS::IAM::Role',
           Properties:
          {
            AssumeRolePolicyDocument:
           {
             Statement:
            [{
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'test.service' },
            }],
             Version: '2012-10-17',
           },
          },
         },
      },
    });
  });

  test('fails if policy name is not unique within a user/group/role', () => {
    // create two policies named Foo and attach them both to the same user/group/role
    const p1 = new Policy(stack, 'P1', { policyName: 'Foo' });
    const p2 = new Policy(stack, 'P2', { policyName: 'Foo' });
    const p3 = new Policy(stack, 'P3'); // uses logicalID as name

    const user = new User(stack, 'MyUser');
    const group = new Group(stack, 'MyGroup');
    const role = new Role(stack, 'MyRole', { assumedBy: new ServicePrincipal('sns.amazonaws.com') });

    p1.attachToUser(user);
    p1.attachToGroup(group);
    p1.attachToRole(role);

    // try to attach p2 to all of these and expect to fail
    expect(() => p2.attachToUser(user)).toThrow(/A policy named "Foo" is already attached/);
    expect(() => p2.attachToGroup(group)).toThrow(/A policy named "Foo" is already attached/);
    expect(() => p2.attachToRole(role)).toThrow(/A policy named "Foo" is already attached/);

    p3.attachToUser(user);
    p3.attachToGroup(group);
    p3.attachToRole(role);
  });

  test('idempotent if a principal (user/group/role) is attached twice', () => {
    const p = new Policy(stack, 'Policy');
    p.addStatements(new PolicyStatement({ resources: ['*'], actions: ['*'] }));

    const user = new User(stack, 'MyUser');
    p.attachToUser(user);
    p.attachToUser(user);

    const group = new Group(stack, 'MyGroup');
    p.attachToGroup(group);
    p.attachToGroup(group);

    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('test.service'),
    });
    p.attachToRole(role);
    p.attachToRole(role);

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        Policy23B91518:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            Groups: [{ Ref: 'MyGroupCBA54B1B' }],
            PolicyDocument:
           {
             Statement: [{ Action: '*', Effect: 'Allow', Resource: '*' }],
             Version: '2012-10-17',
           },
            PolicyName: 'Policy23B91518',
            Roles: [{ Ref: 'MyRoleF48FFE04' }],
            Users: [{ Ref: 'MyUserDC45028B' }],
          },
         },
        MyUserDC45028B: { Type: 'AWS::IAM::User' },
        MyGroupCBA54B1B: { Type: 'AWS::IAM::Group' },
        MyRoleF48FFE04:
         {
           Type: 'AWS::IAM::Role',
           Properties:
          {
            AssumeRolePolicyDocument:
           {
             Statement:
            [{
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'test.service' },
            }],
             Version: '2012-10-17',
           },
          },
         },
      },
    });
  });

  test('idempotent if an imported principal (user/group/role) is attached twice', () => {
    const p = new Policy(stack, 'Policy');
    p.addStatements(new PolicyStatement({ resources: ['*'], actions: ['*'] }));

    const user = new User(stack, 'MyUser');
    const importedUser = User.fromUserArn(stack, 'MyImportedUser', user.userArn);
    p.attachToUser(user);
    p.attachToUser(importedUser);

    const group = new Group(stack, 'MyGroup');
    const importedGroup = Group.fromGroupArn(stack, 'MyImportedGroup', group.groupArn);
    p.attachToGroup(group);
    p.attachToGroup(importedGroup);

    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('test.service'),
    });
    const importedRole = Role.fromRoleArn(stack, 'MyImportedRole', role.roleArn);
    p.attachToRole(role);
    p.attachToRole(importedRole);

    Template.fromStack(stack).templateMatches({
      Resources:
      {
        Policy23B91518:
         {
           Type: 'AWS::IAM::Policy',
           Properties:
          {
            Groups: [{ Ref: 'MyGroupCBA54B1B' }],
            PolicyDocument:
           {
             Statement: [{ Action: '*', Effect: 'Allow', Resource: '*' }],
             Version: '2012-10-17',
           },
            PolicyName: 'Policy23B91518',
            Roles: [{ Ref: 'MyRoleF48FFE04' }],
            Users: [{ Ref: 'MyUserDC45028B' }],
          },
         },
        MyUserDC45028B: { Type: 'AWS::IAM::User' },
        MyGroupCBA54B1B: { Type: 'AWS::IAM::Group' },
        MyRoleF48FFE04:
         {
           Type: 'AWS::IAM::Role',
           Properties:
          {
            AssumeRolePolicyDocument:
           {
             Statement:
            [{
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'test.service' },
            }],
             Version: '2012-10-17',
           },
          },
         },
      },
    });
  });

  test('fails if "forced" policy is not attached to a principal', () => {
    new Policy(stack, 'MyPolicy', { force: true });
    expect(() => app.synth()).toThrow(/attached to at least one principal: user, group or role/);
  });

  test("generated policy name is the same as the logical id if it's shorter than 128 characters", () => {
    createPolicyWithLogicalId(stack, 'Foo');

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyName': 'Foo',
    });
  });

  test('generated policy name only uses the last 128 characters of the logical id', () => {
    const logicalId128 = 'a' + dup(128 - 2) + 'a';
    const logicalIdOver128 = 'PREFIX' + logicalId128;

    createPolicyWithLogicalId(stack, logicalIdOver128);

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      'PolicyName': logicalId128,
    });

    function dup(count: number) {
      let r = '';
      for (let i = 0; i < count; ++i) {
        r += 'x';
      }
      return r;
    }
  });

  test('force=false, dependency on empty Policy never materializes', () => {
    // GIVEN
    const pol = new Policy(stack, 'Pol', { force: false });

    const res = new CfnResource(stack, 'Resource', {
      type: 'Some::Resource',
    });

    // WHEN
    res.node.addDependency(pol);

    // THEN
    Template.fromStack(stack).templateMatches({
      Resources: {
        Resource: {
          Type: 'Some::Resource',
        },
      },
    });
  });

  test('force=false, dependency on attached and non-empty Policy can be taken', () => {
    // GIVEN
    const pol = new Policy(stack, 'Pol', { force: false });
    pol.addStatements(new PolicyStatement({
      actions: ['s3:*'],
      resources: ['*'],
    }));
    pol.attachToUser(new User(stack, 'User'));

    const res = new CfnResource(stack, 'Resource', {
      type: 'Some::Resource',
    });

    // WHEN
    res.node.addDependency(pol);

    // THEN
    Template.fromStack(stack).hasResource('Some::Resource', {
      Type: 'Some::Resource',
      DependsOn: ['Pol0FE9AD5D'],
    });
  });

  test('empty policy is OK if force=false', () => {
    new Policy(stack, 'Pol', { force: false });

    app.synth();
    // If we got here, all OK
  });

  test('reading policyName forces a Policy to materialize', () => {
    const pol = new Policy(stack, 'Pol', { force: false });
    Array.isArray(pol.policyName);

    expect(() => app.synth()).toThrow(/must contain at least one statement/);
  });

  test('fails if policy document is invalid', () => {
    new Policy(stack, 'MyRole', {
      statements: [new PolicyStatement({
        actions: ['*'],
        principals: [new ServicePrincipal('test.service')],
      })],
    });

    expect(() => app.synth()).toThrow(/A PolicyStatement used in an identity-based policy cannot specify any IAM principals/);
  });

  test('Policy can be granted principal permissions', () => {
    const pol = new Policy(stack, 'Policy', {
      policyName: 'MyPolicyName',
    });
    Grant.addToPrincipal({ actions: ['dummy:Action'], grantee: pol, resourceArns: ['*'] });
    pol.attachToUser(new User(stack, 'User'));

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyName: 'MyPolicyName',
      PolicyDocument: {
        Statement: [
          { Action: 'dummy:Action', Effect: 'Allow', Resource: '*' },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('Policy can be granted via lambda.Function.grantInvoke()', () => {
    const func = new lambda.Function(stack, 'Function', { code: lambda.Code.fromInline('export const handler = () => {}'), runtime: lambda.Runtime.NODEJS_LATEST, handler: 'index.handler' });
    const pol = new Policy(stack, 'Policy', {
      policyName: 'MyPolicyName',
    });
    func.grantInvoke(pol);
    pol.attachToUser(new User(stack, 'User'));

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyName: 'MyPolicyName',
      PolicyDocument: {
        Statement: [
          {
            Effect: 'Allow',
            Action: 'lambda:InvokeFunction',
            Resource: [
              { 'Fn::GetAtt': ['Function76856677', 'Arn'] },
              { 'Fn::Join': ['', [{ 'Fn::GetAtt': ['Function76856677', 'Arn'] }, ':*']] },
            ],
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('addPrincipalOrResource() correctly grants Policy permissions', () => {
    const pol = new Policy(stack, 'Policy', {
      policyName: 'MyPolicyName',
    });
    pol.attachToUser(new User(stack, 'User'));

    class DummyResource extends Resource implements IResourceWithPolicy {
      addToResourcePolicy(_statement: PolicyStatement): AddToPrincipalPolicyResult {
        throw new Error('should not be called.');
      }
    }
    const resource = new DummyResource(stack, 'Dummy');
    Grant.addToPrincipalOrResource({ actions: ['dummy:Action'], grantee: pol, resource, resourceArns: ['*'] });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyName: 'MyPolicyName',
      PolicyDocument: {
        Statement: [
          { Action: 'dummy:Action', Effect: 'Allow', Resource: '*' },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('Policy cannot be granted principal permissions across accounts', () => {
    const pol = new Policy(stack, 'Policy', {
      policyName: 'MyPolicyName',
    });
    const crossAccountStack = new Stack(app, 'CrossAccountStack', { env: { account: '5678' } });
    const bucket = new s3.Bucket(crossAccountStack, 'CrossAccountBucket', {
      bucketName: 'cross-account-bucket',
    });
    bucket.grantRead(pol);

    expect(() => app.synth()).toThrow('This grant operation needs to add a resource policy so needs access to a principal.');
  });

  test('Policies cannot be granted resource permissions', () => {
    const pol = new Policy(stack, 'Policy', {
      policyName: 'MyPolicyName',
    });
    const crossAccountStack = new Stack(app, 'CrossAccountStack', { env: { account: '5678' } });
    const bucket = new s3.Bucket(crossAccountStack, 'CrossAccountBucket', {
      bucketName: 'cross-account-bucket',
    });
    Grant.addToPrincipalAndResource({ actions: ['dummy:Action'], grantee: pol, resourceArns: ['*'], resource: bucket });

    expect(() => app.synth()).toThrow('This grant operation needs to add a resource policy so needs access to a principal.');
  });
});

function createPolicyWithLogicalId(stack: Stack, logicalId: string): void {
  const policy = new Policy(stack, logicalId);
  const cfnPolicy = policy.node.defaultChild as CfnPolicy;
  cfnPolicy.overrideLogicalId(logicalId); // force a particular logical ID

  // add statements & principal to satisfy validation
  policy.addStatements(new PolicyStatement({
    actions: ['*'],
    resources: ['*'],
  }));
  policy.attachToRole(new Role(stack, 'Role', { assumedBy: new AnyPrincipal() }));
}
