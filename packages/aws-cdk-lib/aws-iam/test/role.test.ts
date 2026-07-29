import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import { Construct } from 'constructs';
import { Template, Match, Annotations } from '../../assertions';
import { Duration, Stack, App, CfnResource, RemovalPolicy, Lazy, Stage, DefaultStackSynthesizer, CliCredentialsStackSynthesizer, PERMISSIONS_BOUNDARY_CONTEXT_KEY, PermissionsBoundary, Token } from '../../core';
import { AccountPrincipal, AnyPrincipal, ArnPrincipal, CompositePrincipal, FederatedPrincipal, ManagedPolicy, PolicyStatement, Role, ServicePrincipal, User, Policy, PolicyDocument, Effect } from '../lib';

describe('isRole() returns', () => {
  test('true if given Role instance', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');
    // WHEN
    const pureRole = new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    expect(Role.isRole(pureRole)).toBe(true);
  });

  test('false if given imported role instance', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');
    // WHEN
    const importedRole = Role.fromRoleName(stack, 'ImportedRole', 'ImportedRole');
    // THEN
    expect(Role.isRole(importedRole)).toBe(false);
  });

  test('false if given undefined', () => {
    // THEN
    expect(Role.isRole(undefined)).toBe(false);
  });
});

describe('customizeRoles', () => {
  test('throws if precreatedRoles is not used', () => {
    // GIVEN
    const app = new App();
    Role.customizeRoles(app);
    const stack = new Stack(app, 'MyStack');

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Annotations.fromStack(stack).hasError('/MyStack/Role', Match.stringLikeRegexp('IAM Role is being created at path "MyStack/Role"'));
  });

  test('throws if precreatedRoles has a token', () => {
    // GIVEN
    const app = new App();
    Role.customizeRoles(app, {
      usePrecreatedRoles: {
        'MyStack/Role': Lazy.string({ produce: () => 'RoleName' }),
      },
    });
    const stack = new Stack(app, 'MyStack');

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Annotations.fromStack(stack).hasError('/MyStack/Role', Match.stringLikeRegexp('Cannot resolve precreated role name at path "MyStack/Role"'));
  });

  test('does not throw if preventSynthesis=false', () => {
    // GIVEN
    const app = new App();
    Role.customizeRoles(app, {
      preventSynthesis: false,
    });
    const stack = new Stack(app, 'MyStack');

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    // no annotations
    const annotations = Annotations.fromStack(stack);
    annotations.hasNoError('MyStack/Role', Match.stringLikeRegexp('*'));

    // and resource is still synthesized
    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1);
  });

  test('works at the app scope', () => {
    // GIVEN
    const app = new App();
    Role.customizeRoles(app, {
      usePrecreatedRoles: {
        'MyStack/Role': 'SomeRoleName',
      },
    });
    const stack = new Stack(app, 'MyStack');

    // WHEN
    const role = new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    new CfnResource(stack, 'MyResource', {
      type: 'AWS::Custom',
      properties: {
        Role: role.roleName,
      },
    });

    // THEN
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Role', 0);
    template.hasResourceProperties('AWS::Custom', {
      Role: 'SomeRoleName',
    });
  });

  test('works at a relative scope', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');
    const subScope = new Construct(stack, 'SomeConstruct');
    Role.customizeRoles(subScope, {
      usePrecreatedRoles: {
        Role: 'SomeRoleName',
      },
    });

    // WHEN
    const role = new Role(subScope, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    new CfnResource(subScope, 'MyResource', {
      type: 'AWS::Custom',
      properties: {
        Role: role.roleName,
      },
    });

    // THEN
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Role', 0);
    template.hasResourceProperties('AWS::Custom', {
      Role: 'SomeRoleName',
    });
  });

  test('works at an absolute scope', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');
    const subScope = new Construct(stack, 'SomeConstruct');
    Role.customizeRoles(subScope, {
      usePrecreatedRoles: {
        'MyStack/SomeConstruct/Role': 'SomeRoleName',
      },
    });

    // WHEN
    const role = new Role(subScope, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    new CfnResource(subScope, 'MyResource', {
      type: 'AWS::Custom',
      properties: {
        Role: role.roleName,
      },
    });

    // THEN
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Role', 0);
    template.hasResourceProperties('AWS::Custom', {
      Role: 'SomeRoleName',
    });
  });

  test('does not create policies', () => {
    // GIVEN
    const app = new App();
    Role.customizeRoles(app, {
      usePrecreatedRoles: {
        'MyStack/Role': 'SomeRoleName',
      },
    });
    const stack = new Stack(app, 'MyStack');

    // WHEN
    const role = new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    const importedRole = Role.fromRoleName(stack, 'ImportedRole', 'ImportedRole');
    importedRole.grant(role, 'sts:AssumeRole');
    role.addToPolicy(new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ['*'],
      resources: ['*'],
    }));

    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Policy', 0);
  });

  test('can use all role properties', () => {
    // GIVEN
    const app = new App();
    Role.customizeRoles(app, {
      usePrecreatedRoles: {
        'MyStack/Role': 'SomeRoleName',
      },
    });
    const stack = new Stack(app, 'MyStack');

    // WHEN
    const role = new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    const principal = Role.fromRoleName(stack, 'OtherRole', 'OtherRole');
    role.grant(principal, 'sts:AssumeRole');
    role.grantPassRole(principal);
    role.grantAssumeRole(principal);
    role.addManagedPolicy(ManagedPolicy.fromAwsManagedPolicyName('ReadOnlyAccess'));
    role.addManagedPolicy(new ManagedPolicy(stack, 'MyPolicy', {
      statements: [new PolicyStatement({
        effect: Effect.ALLOW,
        actions: ['sns:Publish'],
        resources: ['*'],
      })],
    }));
    role.withoutPolicyUpdates();
    new CfnResource(stack, 'MyResource', {
      type: 'AWS::Custom',
      properties: {
        RoleName: role.roleName,
        RoleArn: role.roleName,
        PolicyFragment: role.policyFragment,
        AssumeRoleAction: role.assumeRoleAction,
        PrincipalAccount: role.principalAccount,
        AssumeRolePolicy: role.assumeRolePolicy,
        PermissionsBoundary: role.permissionsBoundary,
      },
    });

    // THEN
    expect(() => {
      role.applyRemovalPolicy(RemovalPolicy.DESTROY);
    }).not.toThrow(/Cannot apply RemovalPolicy/);
    expect(() => {
      new CfnResource(stack, 'MyResource2', {
        type: 'AWS::Custom',
        properties: {
          RoleId: role.roleId,
        },
      });
    }).toThrow(/"roleId" is not available on precreated roles/);
    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Role', 0);
    template.resourceCountIs('AWS::IAM::Policy', 0);
    template.resourceCountIs('AWS::IAM::ManagedPolicy', 0);
    template.hasResourceProperties('AWS::Custom', {
      RoleName: 'SomeRoleName',
      RoleArn: 'SomeRoleName',
      PolicyFragment: {
        principalJson: {
          AWS: [
            {
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
                  ':role/SomeRoleName',
                ],
              ],
            },
          ],
        },
        conditions: {},
      },
      AssumeRoleAction: 'sts:AssumeRole',
      PrincipalAccount: {
        Ref: 'AWS::AccountId',
      },
      AssumeRolePolicy: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'sns.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });
});

describe('IAM role', () => {
  test('default role', () => {
    const stack = new Stack();

    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    Template.fromStack(stack).templateMatches({
      Resources:
      {
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
              Principal: { Service: 'sns.amazonaws.com' },
            }],
             Version: '2012-10-17',
           },
          },
         },
      },
    });
  });

  test('a role can grant PassRole permissions', () => {
    // GIVEN
    const stack = new Stack();
    const role = new Role(stack, 'Role', { assumedBy: new ServicePrincipal('henk.amazonaws.com') });
    const user = new User(stack, 'User');

    // WHEN
    role.grantPassRole(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'iam:PassRole',
            Effect: 'Allow',
            Resource: { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('a role can grant AssumeRole permissions', () => {
    // GIVEN
    const stack = new Stack();
    const role = new Role(stack, 'Role', { assumedBy: new ServicePrincipal('henk.amazonaws.com') });
    const user = new User(stack, 'User');

    // WHEN
    role.grantAssumeRole(user);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Resource: { 'Fn::GetAtt': ['Role1ABCC5F0', 'Arn'] },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('a role cannot grant AssumeRole permission to a Service Principal', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const user = new User(stack, 'User');
    const role = new Role(stack, 'MyRole', {
      assumedBy: user,
    });

    // THEN
    expect(() => role.grantAssumeRole(new ServicePrincipal('beep-boop.amazonaws.com')))
      .toThrow('Cannot use a service or account principal with grantAssumeRole, use assumeRolePolicy instead.');
  });

  test('a role cannot grant AssumeRole permission to an Account Principal', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const user = new User(stack, 'User');
    const role = new Role(stack, 'MyRole', {
      assumedBy: user,
    });

    // THEN
    expect(() => role.grantAssumeRole(new AccountPrincipal('123456789')))
      .toThrow('Cannot use a service or account principal with grantAssumeRole, use assumeRolePolicy instead.');
  });

  testDeprecated('can supply externalId', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      externalId: 'SomeSecret',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Condition: {
              StringEquals: { 'sts:ExternalId': 'SomeSecret' },
            },
            Effect: 'Allow',
            Principal: { Service: 'sns.amazonaws.com' },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('can supply single externalIds', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      externalIds: ['SomeSecret'],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Condition: {
              StringEquals: { 'sts:ExternalId': 'SomeSecret' },
            },
            Effect: 'Allow',
            Principal: { Service: 'sns.amazonaws.com' },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('can supply multiple externalIds', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      externalIds: ['SomeSecret', 'AnotherSecret'],
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Condition: {
              StringEquals: { 'sts:ExternalId': ['SomeSecret', 'AnotherSecret'] },
            },
            Effect: 'Allow',
            Principal: { Service: 'sns.amazonaws.com' },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('policy is created automatically when permissions are added', () => {
    // by default we don't expect a role policy
    const before = new Stack();
    new Role(before, 'MyRole', { assumedBy: new ServicePrincipal('sns.amazonaws.com') });
    Template.fromStack(before).resourceCountIs('AWS::IAM::Policy', 0);

    // add a policy to the role
    const after = new Stack();
    const afterRole = new Role(after, 'MyRole', { assumedBy: new ServicePrincipal('sns.amazonaws.com') });
    afterRole.addToPolicy(new PolicyStatement({ resources: ['myresource'], actions: ['service:myaction'] }));
    Template.fromStack(after).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'service:myaction',
            Effect: 'Allow',
            Resource: 'myresource',
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'MyRoleDefaultPolicyA36BE1DD',
      Roles: [
        {
          Ref: 'MyRoleF48FFE04',
        },
      ],
    });
  });

  test('managed policy arns can be supplied upon initialization and also added later', () => {
    const stack = new Stack();

    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('test.service'),
      managedPolicies: [{ managedPolicyArn: 'managed1' } as any, { managedPolicyArn: 'managed2' } as any],
    });

    role.addManagedPolicy({ managedPolicyArn: 'managed3' } as any);
    Template.fromStack(stack).templateMatches({
      Resources:
      {
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
            ManagedPolicyArns: ['managed1', 'managed2', 'managed3'],
          },
         },
      },
    });
  });

  test('federated principal can change AssumeRoleAction', () => {
    const stack = new Stack();
    const cognitoPrincipal = new FederatedPrincipal(
      'foo',
      { StringEquals: { key: 'value' } },
      'sts:AssumeSomething');

    new Role(stack, 'MyRole', { assumedBy: cognitoPrincipal });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Principal: { Federated: 'foo' },
            Condition: {
              StringEquals: { key: 'value' },
            },
            Action: 'sts:AssumeSomething',
            Effect: 'Allow',
          },
        ],
      },
    });
  });

  test('role path can be used to specify the path', () => {
    const stack = new Stack();

    new Role(stack, 'MyRole', { path: '/', assumedBy: new ServicePrincipal('sns.amazonaws.com') });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      Path: '/',
    });
  });

  test('role path can be 1 character', () => {
    const stack = new Stack();

    const assumedBy = new ServicePrincipal('bla');

    expect(() => new Role(stack, 'MyRole', { assumedBy, path: '/' })).not.toThrow();
  });

  test('role path cannot be empty', () => {
    const stack = new Stack();

    const assumedBy = new ServicePrincipal('bla');

    expect(() => new Role(stack, 'MyRole', { assumedBy, path: '' }))
      .toThrow('Role path must be between 1 and 512 characters. The provided role path is 0 characters.');
  });

  test('role path must be less than or equal to 512', () => {
    const stack = new Stack();

    const assumedBy = new ServicePrincipal('bla');

    expect(() => new Role(stack, 'MyRole', { assumedBy, path: '/' + Array(512).join('a') + '/' }))
      .toThrow('Role path must be between 1 and 512 characters. The provided role path is 513 characters.');
  });

  test('role path must start with a forward slash', () => {
    const stack = new Stack();

    const assumedBy = new ServicePrincipal('bla');

    const expected = (val: any) => 'Role path must be either a slash or valid characters (alphanumerics and symbols) surrounded by slashes. '
    + `Valid characters are unicode characters in [\\u0021-\\u007F]. However, ${val} is provided.`;
    expect(() => new Role(stack, 'MyRole', { assumedBy, path: 'aaa' })).toThrow(expected('aaa'));
  });

  test('role path must end with a forward slash', () => {
    const stack = new Stack();

    const assumedBy = new ServicePrincipal('bla');

    const expected = (val: any) => 'Role path must be either a slash or valid characters (alphanumerics and symbols) surrounded by slashes. '
    + `Valid characters are unicode characters in [\\u0021-\\u007F]. However, ${val} is provided.`;
    expect(() => new Role(stack, 'MyRole', { assumedBy, path: '/a' })).toThrow(expected('/a'));
  });

  test('role path must contain unicode chars within [\\u0021-\\u007F]', () => {
    const stack = new Stack();

    const assumedBy = new ServicePrincipal('bla');

    const expected = (val: any) => 'Role path must be either a slash or valid characters (alphanumerics and symbols) surrounded by slashes. '
    + `Valid characters are unicode characters in [\\u0021-\\u007F]. However, ${val} is provided.`;
    expect(() => new Role(stack, 'MyRole', { assumedBy, path: '/\u0020\u0080/' })).toThrow(expected('/\u0020\u0080/'));
  });

  describe('maxSessionDuration', () => {
    test('is not specified by default', () => {
      const stack = new Stack();
      new Role(stack, 'MyRole', { assumedBy: new ServicePrincipal('sns.amazonaws.com') });
      Template.fromStack(stack).templateMatches({
        Resources: {
          MyRoleF48FFE04: {
            Type: 'AWS::IAM::Role',
            Properties: {
              AssumeRolePolicyDocument: {
                Statement: [
                  {
                    Action: 'sts:AssumeRole',
                    Effect: 'Allow',
                    Principal: {
                      Service: 'sns.amazonaws.com',
                    },
                  },
                ],
                Version: '2012-10-17',
              },
            },
          },
        },
      });
    });

    test('can be used to specify the maximum session duration for assuming the role', () => {
      const stack = new Stack();

      new Role(stack, 'MyRole', { maxSessionDuration: Duration.seconds(3700), assumedBy: new ServicePrincipal('sns.amazonaws.com') });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        MaxSessionDuration: 3700,
      });
    });

    test('must be between 3600 and 43200', () => {
      const stack = new Stack();

      const assumedBy = new ServicePrincipal('bla');

      new Role(stack, 'MyRole1', { assumedBy, maxSessionDuration: Duration.hours(1) });
      new Role(stack, 'MyRole2', { assumedBy, maxSessionDuration: Duration.hours(12) });

      const expected = (val: any) => `maxSessionDuration is set to ${val}, but must be >= 3600sec (1hr) and <= 43200sec (12hrs)`;
      expect(() => new Role(stack, 'MyRole3', { assumedBy, maxSessionDuration: Duration.minutes(1) }))
        .toThrow(expected(60));
      expect(() => new Role(stack, 'MyRole4', { assumedBy, maxSessionDuration: Duration.seconds(3599) }))
        .toThrow(expected(3599));
      expect(() => new Role(stack, 'MyRole5', { assumedBy, maxSessionDuration: Duration.seconds(43201) }))
        .toThrow(expected(43201));
    });
  });

  test('allow role with multiple principals', () => {
    const stack = new Stack();

    new Role(stack, 'MyRole', {
      assumedBy: new CompositePrincipal(
        new ServicePrincipal('boom.amazonaws.test'),
        new ArnPrincipal('1111111'),
      ),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'boom.amazonaws.test',
            },
          },
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              AWS: '1111111',
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('can supply permissions boundary managed policy', () => {
    // GIVEN
    const stack = new Stack();

    const permissionsBoundary = ManagedPolicy.fromAwsManagedPolicyName('managed-policy');

    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      permissionsBoundary,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':iam::aws:policy/managed-policy',
          ],
        ],
      },
    });
  });

  test('Principal-* in an AssumeRolePolicyDocument gets translated to { "AWS": "*" }', () => {
    // The docs say that "Principal: *" and "Principal: { AWS: * }" are equivalent
    // (https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html)
    // but in practice CreateRole errors out if you use "Principal: *" in an AssumeRolePolicyDocument:
    // An error occurred (MalformedPolicyDocument) when calling the CreateRole operation: AssumeRolepolicy contained an invalid principal: "STAR":"*".

    // Make sure that we handle this case specially.
    const stack = new Stack();
    new Role(stack, 'Role', {
      assumedBy: new AnyPrincipal(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: { AWS: '*' },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('can have a description', () => {
    const stack = new Stack();

    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      description: 'This is a role description.',
    });

    Template.fromStack(stack).templateMatches({
      Resources:
      {
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
              Principal: { Service: 'sns.amazonaws.com' },
            }],
             Version: '2012-10-17',
           },
            Description: 'This is a role description.',
          },
         },
      },
    });
  });

  test('should not have an empty description', () => {
    const stack = new Stack();

    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      description: '',
    });

    Template.fromStack(stack).templateMatches({
      Resources:
      {
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
              Principal: { Service: 'sns.amazonaws.com' },
            }],
             Version: '2012-10-17',
           },
          },
         },
      },
    });
  });

  test('description can only be 1000 characters long', () => {
    const stack = new Stack();

    expect(() => {
      new Role(stack, 'MyRole', {
        assumedBy: new ServicePrincipal('sns.amazonaws.com'),
        description: '1000+ character long description: Lorem ipsum dolor sit amet, consectetuer adipiscing elit. \
        Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, \
        nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat \
        massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. In enim justo, rhoncus ut, \
        imperdiet a, venenatis vitae, justo. Nullam dictum felis eu pede mollis pretium. Integer tincidunt. Cras dapibus. \
        Vivamus elementum semper nisi. Aenean vulputate eleifend tellus. Aenean leo ligula, porttitor eu, consequat vitae, \
        eleifend ac, enim. Aliquam lorem ante, dapibus in, viverra quis, feugiat a, tellus. Phasellus viverra nulla ut metus \
        varius laoreet. Quisque rutrum. Aenean imperdiet. Etiam ultricies nisi vel augue. Curabitur ullamcorper ultricies nisi. \
        Nam eget dui. Etiam rhoncus. Maecenas tempus, tellus eget condimentum rhoncus, sem quam semper libero, sit amet adipiscing \
        sem neque sed ipsum.',
      });
    }).toThrow(/Role description must be no longer than 1000 characters./);
  });

  test('fails if managed policy is invalid', () => {
    const app = new App();
    const stack = new Stack(app, 'my-stack');
    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      managedPolicies: [new ManagedPolicy(stack, 'MyManagedPolicy', {
        statements: [new PolicyStatement({
          resources: ['*'],
          actions: ['*'],
          principals: [new ServicePrincipal('sns.amazonaws.com')],
        })],
      })],
    });

    expect(() => app.synth()).toThrow(/A PolicyStatement used in an identity-based policy cannot specify any IAM principals/);
  });

  test('fails if default role policy is invalid', () => {
    const app = new App();
    const stack = new Stack(app, 'my-stack');
    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    role.addToPrincipalPolicy(new PolicyStatement({
      resources: ['*'],
      actions: ['*'],
      principals: [new ServicePrincipal('sns.amazonaws.com')],
    }));

    expect(() => app.synth()).toThrow(/A PolicyStatement used in an identity-based policy cannot specify any IAM principals/);
  });

  test('fails if inline policy from props is invalid', () => {
    const app = new App();
    const stack = new Stack(app, 'my-stack');
    new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      inlinePolicies: {
        testPolicy: new PolicyDocument({
          statements: [new PolicyStatement({
            resources: ['*'],
            actions: ['*'],
            principals: [new ServicePrincipal('sns.amazonaws.com')],
          })],
        }),
      },
    });

    expect(() => app.synth()).toThrow(/A PolicyStatement used in an identity-based policy cannot specify any IAM principals/);
  });

  test('fails if attached inline policy is invalid', () => {
    const app = new App();
    const stack = new Stack(app, 'my-stack');
    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });
    role.attachInlinePolicy(new Policy(stack, 'MyPolicy', {
      statements: [new PolicyStatement({
        resources: ['*'],
        actions: ['*'],
        principals: [new ServicePrincipal('sns.amazonaws.com')],
      })],
    }));

    expect(() => app.synth()).toThrow(/A PolicyStatement used in an identity-based policy cannot specify any IAM principals/);
  });

  test('fails if assumeRolePolicy is invalid', () => {
    const app = new App();
    const stack = new Stack(app, 'my-stack');
    const role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      managedPolicies: [new ManagedPolicy(stack, 'MyManagedPolicy')],
    });
    role.assumeRolePolicy?.addStatements(new PolicyStatement({ actions: ['*'] }));

    expect(() => app.synth()).toThrow(/A PolicyStatement used in a trust policy must specify at least one IAM principal/);
  });
});

describe('permissions boundary', () => {
  test('can be applied to an app', () => {
    // GIVEN
    const app = new App({
      context: {
        [PERMISSIONS_BOUNDARY_CONTEXT_KEY]: {
          name: 'cdk-${Qualifier}-PermissionsBoundary',
        },
      },
    });
    const stack = new Stack(app);

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
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
            ':policy/cdk-hnb659fds-PermissionsBoundary',
          ],
        ],
      },
    });
  });

  test('can be applied to a stage', () => {
    // GIVEN
    const app = new App();
    const stage = new Stage(app, 'Stage', {
      permissionsBoundary: PermissionsBoundary.fromName('cdk-${Qualifier}-PermissionsBoundary'),
    });
    const stack = new Stack(stage);

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
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
            ':policy/cdk-hnb659fds-PermissionsBoundary',
          ],
        ],
      },
    });
  });

  test('can be applied to a stage, and will replace placeholders', () => {
    // GIVEN
    const app = new App();
    const stage = new Stage(app, 'Stage', {
      env: {
        region: 'test-region',
        account: '123456789012',
      },
      permissionsBoundary: PermissionsBoundary.fromName('cdk-${Qualifier}-PermissionsBoundary-${AWS::AccountId}-${AWS::Region}'),
    });
    const stack = new Stack(stage);

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':iam::123456789012:policy/cdk-hnb659fds-PermissionsBoundary-123456789012-test-region',
          ],
        ],
      },
    });
  });

  test('with a custom qualifier', () => {
    // GIVEN
    const app = new App();
    const stage = new Stage(app, 'Stage', {
      permissionsBoundary: PermissionsBoundary.fromName('cdk-${Qualifier}-PermissionsBoundary'),
    });
    const stack = new Stack(stage, 'MyStack', {
      synthesizer: new DefaultStackSynthesizer({
        qualifier: 'custom',
      }),
    });

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
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
            ':policy/cdk-custom-PermissionsBoundary',
          ],
        ],
      },
    });
  });

  test('with a custom permissions boundary', () => {
    // GIVEN
    const app = new App();
    const stage = new Stage(app, 'Stage', {
      permissionsBoundary: PermissionsBoundary.fromName('my-permissions-boundary'),
    });
    const stack = new Stack(stage);

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
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
            ':policy/my-permissions-boundary',
          ],
        ],
      },
    });
  });

  test('with a custom permissions boundary and qualifier', () => {
    // GIVEN
    const app = new App();
    const stage = new Stage(app, 'Stage', {
      permissionsBoundary: PermissionsBoundary.fromName('my-${Qualifier}-permissions-boundary'),
    });
    const stack = new Stack(stage, 'MyStack', {
      synthesizer: new CliCredentialsStackSynthesizer({
        qualifier: 'custom',
      }),
    });

    // WHEN
    new Role(stack, 'Role', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
      PermissionsBoundary: {
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
            ':policy/my-custom-permissions-boundary',
          ],
        ],
      },
    });
  });
});

test('managed policy ARNs are deduplicated', () => {
  const app = new App();
  const stack = new Stack(app, 'my-stack');
  const role = new Role(stack, 'MyRole', {
    assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    managedPolicies: [
      ManagedPolicy.fromAwsManagedPolicyName('SuperDeveloper'),
      ManagedPolicy.fromAwsManagedPolicyName('SuperDeveloper'),
    ],
  });
  role.addToPrincipalPolicy(
    new PolicyStatement({
      actions: ['s3:*'],
      resources: ['*'],
    }),
  );

  for (let i = 0; i < 20; i++) {
    role.addManagedPolicy(ManagedPolicy.fromAwsManagedPolicyName('SuperDeveloper'));
  }

  Annotations.fromStack(stack).hasNoWarning('/my-stack/MyRole', Match.stringLikeRegexp('.*'));

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
    ManagedPolicyArns: [
      {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':iam::aws:policy/SuperDeveloper',
          ],
        ],
      },
    ],
  });
});

test('too many managed policies warning', () => {
  const app = new App();
  const stack = new Stack(app, 'my-stack');
  const role = new Role(stack, 'MyRole', {
    assumedBy: new ServicePrincipal('sns.amazonaws.com'),
  });
  role.addToPrincipalPolicy(
    new PolicyStatement({
      actions: ['s3:*'],
      resources: ['*'],
    }),
  );

  for (let i = 0; i < 20; i++) {
    role.addManagedPolicy(ManagedPolicy.fromAwsManagedPolicyName(`SuperDeveloper${i}`));
  }

  Annotations.fromStack(stack).hasWarning('/my-stack/MyRole', Match.stringLikeRegexp('.*'));
});

describe('role with too large inline policy', () => {
  const N = 100;

  let app: App;
  let stack: Stack;
  let role: Role;
  beforeEach(() => {
    app = new App();
    stack = new Stack(app, 'my-stack');
    role = new Role(stack, 'MyRole', {
      assumedBy: new ServicePrincipal('service.amazonaws.com'),
    });

    for (let i = 0; i < N; i++) {
      role.addToPrincipalPolicy(new PolicyStatement({
        actions: ['aws:DoAThing'],
        resources: [`arn:aws:service:us-east-1:111122223333:someResource/SomeSpecificResource${i}`],
      }));
    }
  });

  test('excess gets split off into ManagedPolicies', () => {
    // THEN
    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::ManagedPolicy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Resource: `arn:aws:service:us-east-1:111122223333:someResource/SomeSpecificResource${N - 1}`,
          }),
        ]),
      },
      Roles: [{ Ref: 'MyRoleF48FFE04' }],
    });
  });

  test('Dependables track the final declaring construct', () => {
    // WHEN
    const result = role.addToPrincipalPolicy(new PolicyStatement({
      actions: ['aws:DoAThing'],
      resources: [`arn:aws:service:us-east-1:111122223333:someResource/SomeSpecificResource${N}`],
    }));

    const res = new CfnResource(stack, 'Depender', {
      type: 'AWS::Some::Resource',
    });

    expect(result.policyDependable).toBeTruthy();
    res.node.addDependency(result.policyDependable!);

    // THEN
    const template = Template.fromStack(stack);
    template.hasResource('AWS::Some::Resource', {
      DependsOn: [
        'MyRoleOverflowPolicy13EF5596A',
      ],
    });
  });
});

test('many copies of the same statement do not result in overflow policies', () => {
  const N = 100;

  const app = new App();
  const stack = new Stack(app, 'my-stack');
  const role = new Role(stack, 'MyRole', {
    assumedBy: new ServicePrincipal('service.amazonaws.com'),
  });

  for (let i = 0; i < N; i++) {
    role.addToPrincipalPolicy(new PolicyStatement({
      actions: ['aws:DoAThing'],
      resources: ['arn:aws:service:us-east-1:111122223333:someResource/SomeSpecificResource'],
    }));
  }

  // THEN
  const template = Template.fromStack(stack);
  template.resourceCountIs('AWS::IAM::ManagedPolicy', 0);
});

test('cross-env role ARNs include path', () => {
  const app = new App();
  const roleStack = new Stack(app, 'role-stack', { env: { account: '123456789012', region: 'us-east-1' } });
  const referencerStack = new Stack(app, 'referencer-stack', { env: { region: 'us-east-2' } });
  const role = new Role(roleStack, 'Role', {
    assumedBy: new ServicePrincipal('sns.amazonaws.com'),
    path: '/sample/path/',
    roleName: 'sample-name',
  });
  new CfnResource(referencerStack, 'Referencer', {
    type: 'Custom::RoleReferencer',
    properties: { RoleArn: role.roleArn },
  });

  Template.fromStack(referencerStack).hasResourceProperties('Custom::RoleReferencer', {
    RoleArn: {
      'Fn::Join': [
        '',
        [
          'arn:',
          {
            Ref: 'AWS::Partition',
          },
          ':iam::123456789012:role/sample/path/sample-name',
        ],
      ],
    },
  });
});

test('doesn\'t throw with roleName of 64 chars', () => {
  const app = new App();
  const stack = new Stack(app, 'MyStack');
  const valdName = 'a'.repeat(64);

  expect(() => {
    new Role(stack, 'Test', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      roleName: valdName,
    });
  }).not.toThrow('Invalid roleName');
});

test('throws with roleName over 64 chars', () => {
  const app = new App();
  const stack = new Stack(app, 'MyStack');
  const longName = 'a'.repeat(65);

  expect(() => {
    new Role(stack, 'Test', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      roleName: longName,
    });
  }).toThrow('Invalid roleName');
});

describe('roleName validation', () => {
  const app = new App();
  const stack = new Stack(app, 'MyStack');
  const invalidChars = '!#$%^&*()';

  it('rejects names with spaces', () => {
    expect(() => {
      new Role(stack, 'test spaces', {
        assumedBy: new ServicePrincipal('sns.amazonaws.com'),
        roleName: 'invalid name',
      });
    }).toThrow('Invalid roleName');
  });

  invalidChars.split('').forEach(char => {
    it(`rejects name with ${char}`, () => {
      expect(() => {
        new Role(stack, `test ${char}`, {
          assumedBy: new ServicePrincipal('sns.amazonaws.com'),
          roleName: `invalid${char}`,
        });
      }).toThrow('Invalid roleName');
    });
  });
});

test('roleName validation with Tokens', () =>{
  const app = new App();
  const stack = new Stack(app, 'MyStack');
  const token = Lazy.string({ produce: () => 'token' });

  // Mock isUnresolved to return false
  jest.spyOn(Token, 'isUnresolved').mockReturnValue(false);

  expect(() => {
    new Role(stack, 'Valid', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      roleName: token,
    });
  }).toThrow('Invalid roleName');

  // Mock isUnresolved to return true
  jest.spyOn(Token, 'isUnresolved').mockReturnValue(true);

  expect(() => {
    new Role(stack, 'Invalid', {
      assumedBy: new ServicePrincipal('sns.amazonaws.com'),
      roleName: token,
    });
  }).not.toThrow('Invalid roleName');

  jest.clearAllMocks();
});
