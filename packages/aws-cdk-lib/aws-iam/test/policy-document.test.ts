import { Template } from '../../assertions';
import { Lazy, Stack, Token } from '../../core';
import type { IPrincipal } from '../lib';
import {
  AccountPrincipal, Anyone, AnyPrincipal, ArnPrincipal, CanonicalUserPrincipal, CompositePrincipal,
  Effect, FederatedPrincipal, PolicyDocument, PolicyStatement, PrincipalPolicyFragment, ServicePrincipal, Role,
} from '../lib';

describe('IAM policy document', () => {
  test('the Permission class is a programming model for iam', () => {
    const stack = new Stack();

    const p = new PolicyStatement();
    p.addActions('sqs:SendMessage');
    p.addActions('dynamodb:CreateTable', 'dynamodb:DeleteTable');
    p.addResources('myQueue');
    p.addResources('yourQueue');

    p.addAllResources();
    p.addAwsAccountPrincipal(`my${Token.asString({ account: 'account' })}name`);
    p.addAccountCondition('12221121221');

    expect(stack.resolve(p.toStatementJson())).toEqual({
      Action:
        ['sqs:SendMessage',
          'dynamodb:CreateTable',
          'dynamodb:DeleteTable'],
      Resource: ['myQueue', 'yourQueue', '*'],
      Effect: 'Allow',
      Principal:
      {
        AWS:
        {
          'Fn::Join':
            ['',
              ['arn:',
                { Ref: 'AWS::Partition' },
                ':iam::my',
                { account: 'account' },
                'name:root']],
        },
      },
      Condition: { StringEquals: { 'sts:ExternalId': '12221121221' } },
    });
  });

  test('addSourceAccountCondition and addSourceArnCondition for cross-service resource access', () => {
    const stack = new Stack();

    const p = new PolicyStatement();
    p.addActions('sns:Publish');
    p.addResources('myTopic');
    p.addAllResources();
    p.addServicePrincipal('s3.amazonaws.com');
    p.addSourceAccountCondition('12221121221');
    p.addSourceArnCondition('bucketArn');

    expect(stack.resolve(p.toStatementJson())).toEqual({
      Action: 'sns:Publish',
      Resource: ['myTopic', '*'],
      Effect: 'Allow',
      Principal: { Service: 's3.amazonaws.com' },
      Condition: {
        StringEquals: { 'aws:SourceAccount': '12221121221' },
        ArnEquals: { 'aws:SourceArn': 'bucketArn' },
      },
    });
  });

  test('the PolicyDocument class is a dom for iam policy documents', () => {
    const stack = new Stack();
    const doc = new PolicyDocument();
    const p1 = new PolicyStatement();
    p1.addActions('sqs:SendMessage');
    p1.addNotResources('arn:aws:sqs:us-east-1:123456789012:forbidden_queue');

    const p2 = new PolicyStatement();
    p2.effect = Effect.DENY;
    p2.addActions('cloudformation:CreateStack');

    const p3 = new PolicyStatement();
    p3.effect = Effect.ALLOW;
    p3.addNotActions('cloudformation:UpdateTerminationProtection');

    const p4 = new PolicyStatement();
    p4.effect = Effect.DENY;
    p4.addNotPrincipals(new CanonicalUserPrincipal('OnlyAuthorizedUser'));

    doc.addStatements(p1);
    doc.addStatements(p2);
    doc.addStatements(p3);
    doc.addStatements(p4);

    expect(stack.resolve(doc)).toEqual({
      Version: '2012-10-17',
      Statement:
        [{ Effect: 'Allow', Action: 'sqs:SendMessage', NotResource: 'arn:aws:sqs:us-east-1:123456789012:forbidden_queue' },
          { Effect: 'Deny', Action: 'cloudformation:CreateStack' },
          { Effect: 'Allow', NotAction: 'cloudformation:UpdateTerminationProtection' },
          { Effect: 'Deny', NotPrincipal: { CanonicalUser: 'OnlyAuthorizedUser' } }],
    });
  });

  test('Cannot combine Actions and NotActions', () => {
    expect(() => {
      new PolicyStatement({
        actions: ['abc:def'],
        notActions: ['abc:def'],
      });
    }).toThrow(/Cannot add 'NotActions' to policy statement if 'Actions' have been added/);
  });

  test('Throws with invalid actions', () => {
    expect(() => {
      new PolicyStatement({
        actions: ['service:action', '*', 'service:acti*', 'in:val:id'],
      });
    }).toThrow(/Action 'in:val:id' is invalid/);
  });

  test('Throws with invalid not actions', () => {
    expect(() => {
      new PolicyStatement({
        notActions: ['service:action', '*', 'service:acti*', 'in:val:id'],
      });
    }).toThrow(/Action 'in:val:id' is invalid/);
  });

  // https://github.com/aws/aws-cdk/issues/13479
  test('Does not validate unresolved tokens', () => {
    const stack = new Stack();
    const perm = new PolicyStatement({
      actions: [`${Lazy.string({ produce: () => 'sqs:sendMessage' })}`],
    });

    expect(stack.resolve(perm.toStatementJson())).toEqual({
      Effect: 'Allow',
      Action: 'sqs:sendMessage',
    });
  });

  test('Cannot combine Resources and NotResources', () => {
    expect(() => {
      new PolicyStatement({
        resources: ['abc'],
        notResources: ['def'],
      });
    }).toThrow(/Cannot add 'NotResources' to policy statement if 'Resources' have been added/);
  });

  test('Cannot add NotPrincipals when Principals exist', () => {
    const stmt = new PolicyStatement({
      principals: [new CanonicalUserPrincipal('abc')],
    });
    expect(() => {
      stmt.addNotPrincipals(new CanonicalUserPrincipal('def'));
    }).toThrow(/Cannot add 'NotPrincipals' to policy statement if 'Principals' have been added/);
  });

  test('Cannot add Principals when NotPrincipals exist', () => {
    const stmt = new PolicyStatement({
      notPrincipals: [new CanonicalUserPrincipal('abc')],
    });
    expect(() => {
      stmt.addPrincipals(new CanonicalUserPrincipal('def'));
    }).toThrow(/Cannot add 'Principals' to policy statement if 'NotPrincipals' have been added/);
  });

  test('Permission allows specifying multiple actions upon construction', () => {
    const stack = new Stack();
    const perm = new PolicyStatement();
    perm.addResources('MyResource');
    perm.addActions('service:Action1', 'service:Action2', 'service:Action3');

    expect(stack.resolve(perm.toStatementJson())).toEqual({
      Effect: 'Allow',
      Action: ['service:Action1', 'service:Action2', 'service:Action3'],
      Resource: 'MyResource',
    });
  });

  test('PolicyDoc resolves to undefined if there are no permissions', () => {
    const stack = new Stack();
    const p = new PolicyDocument();
    expect(stack.resolve(p)).toBeUndefined();
  });

  test('canonicalUserPrincipal adds a principal to a policy with the passed canonical user id', () => {
    const stack = new Stack();
    const p = new PolicyStatement();
    const canoncialUser = 'averysuperduperlongstringfor';
    p.addPrincipals(new CanonicalUserPrincipal(canoncialUser));
    expect(stack.resolve(p.toStatementJson())).toEqual({
      Effect: 'Allow',
      Principal: {
        CanonicalUser: canoncialUser,
      },
    });
  });

  test('addAccountRootPrincipal adds a principal with the current account root', () => {
    const stack = new Stack();

    const p = new PolicyStatement();
    p.addAccountRootPrincipal();
    expect(stack.resolve(p.toStatementJson())).toEqual({
      Effect: 'Allow',
      Principal: {
        AWS: {
          'Fn::Join': [
            '',
            [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':iam::',
              { Ref: 'AWS::AccountId' },
              ':root',
            ],
          ],
        },
      },
    });
  });

  test('addFederatedPrincipal adds a Federated principal with the passed value', () => {
    const stack = new Stack();
    const p = new PolicyStatement();
    p.addFederatedPrincipal('com.amazon.cognito', { StringEquals: { key: 'value' } });
    expect(stack.resolve(p.toStatementJson())).toEqual({
      Effect: 'Allow',
      Principal: {
        Federated: 'com.amazon.cognito',
      },
      Condition: {
        StringEquals: { key: 'value' },
      },
    });
  });

  test('addAwsAccountPrincipal can be used multiple times', () => {
    const stack = new Stack();

    const p = new PolicyStatement();
    p.addAwsAccountPrincipal('1234');
    p.addAwsAccountPrincipal('5678');
    expect(stack.resolve(p.toStatementJson())).toEqual({
      Effect: 'Allow',
      Principal: {
        AWS: [
          { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::1234:root']] },
          { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::5678:root']] },
        ],
      },
    });
  });

  describe('hasResource', () => {
    test('false if there are no resources', () => {
      expect(new PolicyStatement().hasResource).toEqual(false);
    });

    test('true if there is one resource', () => {
      expect(new PolicyStatement({ resources: ['one-resource'] }).hasResource).toEqual(true);
    });

    test('true for multiple resources', () => {
      const p = new PolicyStatement();
      p.addResources('r1');
      p.addResources('r2');
      expect(p.hasResource).toEqual(true);
    });
  });

  describe('hasPrincipal', () => {
    test('false if there is no principal', () => {
      expect(new PolicyStatement().hasPrincipal).toEqual(false);
    });

    test('true if there is a principal', () => {
      const p = new PolicyStatement();
      p.addArnPrincipal('bla');
      expect(p.hasPrincipal).toEqual(true);
    });

    test('true if there is a notPrincipal', () => {
      const p = new PolicyStatement();
      p.addNotPrincipals(new CanonicalUserPrincipal('test'));
      expect(p.hasPrincipal).toEqual(true);
    });
  });

  test('statementCount returns the number of statement in the policy document', () => {
    const p = new PolicyDocument();
    expect(p.statementCount).toEqual(0);
    p.addStatements(new PolicyStatement({ actions: ['service:action1'] }));
    expect(p.statementCount).toEqual(1);
    p.addStatements(new PolicyStatement({ actions: ['service:action2'] }));
    expect(p.statementCount).toEqual(2);
  });

  describe('{ AWS: "*" } principal', () => {
    test('is represented as `Anyone`', () => {
      const stack = new Stack();
      const p = new PolicyDocument();

      p.addStatements(new PolicyStatement({ principals: [new Anyone()] }));

      expect(stack.resolve(p)).toEqual({
        Statement: [
          { Effect: 'Allow', Principal: { AWS: '*' } },
        ],
        Version: '2012-10-17',
      });
    });

    test('is represented as `AnyPrincipal`', () => {
      const stack = new Stack();
      const p = new PolicyDocument();

      p.addStatements(new PolicyStatement({ principals: [new AnyPrincipal()] }));

      expect(stack.resolve(p)).toEqual({
        Statement: [
          { Effect: 'Allow', Principal: { AWS: '*' } },
        ],
        Version: '2012-10-17',
      });
    });

    test('is represented as `addAnyPrincipal`', () => {
      const stack = new Stack();
      const p = new PolicyDocument();

      const s = new PolicyStatement();
      s.addAnyPrincipal();
      p.addStatements(s);

      expect(stack.resolve(p)).toEqual({
        Statement: [
          { Effect: 'Allow', Principal: { AWS: '*' } },
        ],
        Version: '2012-10-17',
      });
    });
  });

  test('addResources() will not break a list-encoded Token', () => {
    const stack = new Stack();

    const statement = new PolicyStatement();
    statement.addActions(...Lazy.list({ produce: () => ['service:a', 'service:b', 'service:c'] }));
    statement.addResources(...Lazy.list({ produce: () => ['x', 'y', 'z'] }));

    expect(stack.resolve(statement.toStatementJson())).toEqual({
      Effect: 'Allow',
      Action: ['service:a', 'service:b', 'service:c'],
      Resource: ['x', 'y', 'z'],
    });
  });

  test('addResources()/addActions() will not add duplicates', () => {
    const stack = new Stack();

    const statement = new PolicyStatement();
    statement.addActions('service:a');
    statement.addActions('service:a');

    statement.addResources('x');
    statement.addResources('x');

    expect(stack.resolve(statement.toStatementJson())).toEqual({
      Effect: 'Allow',
      Action: 'service:a',
      Resource: 'x',
    });
  });

  test('addNotResources()/addNotActions() will not add duplicates', () => {
    const stack = new Stack();

    const statement = new PolicyStatement();
    statement.addNotActions('service:a');
    statement.addNotActions('service:a');

    statement.addNotResources('x');
    statement.addNotResources('x');

    expect(stack.resolve(statement.toStatementJson())).toEqual({
      Effect: 'Allow',
      NotAction: 'service:a',
      NotResource: 'x',
    });
  });

  test('addCanonicalUserPrincipal can be used to add cannonical user principals', () => {
    const stack = new Stack();
    const p = new PolicyDocument();

    const s1 = new PolicyStatement();
    s1.addCanonicalUserPrincipal('cannonical-user-1');

    const s2 = new PolicyStatement();
    s2.addPrincipals(new CanonicalUserPrincipal('cannonical-user-2'));

    p.addStatements(s1);
    p.addStatements(s2);

    expect(stack.resolve(p)).toEqual({
      Statement: [
        { Effect: 'Allow', Principal: { CanonicalUser: 'cannonical-user-1' } },
        { Effect: 'Allow', Principal: { CanonicalUser: 'cannonical-user-2' } },
      ],
      Version: '2012-10-17',
    });
  });

  test('addPrincipal correctly merges array in', () => {
    const stack = new Stack();
    const arrayPrincipal: IPrincipal = {
      get grantPrincipal() { return this; },
      assumeRoleAction: 'sts:AssumeRole',
      policyFragment: new PrincipalPolicyFragment({ AWS: ['foo', 'bar'] }),
      addToPolicy() { return false; },
      addToPrincipalPolicy() { return { statementAdded: false }; },
    };
    const s = new PolicyStatement();
    s.addAccountRootPrincipal();
    s.addPrincipals(arrayPrincipal);
    expect(stack.resolve(s.toStatementJson())).toEqual({
      Effect: 'Allow',
      Principal: {
        AWS: [
          { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::', { Ref: 'AWS::AccountId' }, ':root']] },
          'foo', 'bar',
        ],
      },
    });
  });

  // https://github.com/aws/aws-cdk/issues/1201
  test('policy statements with multiple principal types can be created using multiple addPrincipal calls', () => {
    const stack = new Stack();
    const s = new PolicyStatement();
    s.addArnPrincipal('349494949494');
    s.addServicePrincipal('test.service');
    s.addResources('resource');
    s.addActions('service:action');

    expect(stack.resolve(s.toStatementJson())).toEqual({
      Action: 'service:action',
      Effect: 'Allow',
      Principal: { AWS: '349494949494', Service: 'test.service' },
      Resource: 'resource',
    });
  });

  describe('Service principals', () => {
    test('regional service principals resolve appropriately', () => {
      const stack = new Stack(undefined, undefined, { env: { region: 'cn-north-1' } });
      const s = new PolicyStatement();
      s.addActions('test:Action');
      s.addServicePrincipal('codedeploy.amazonaws.com');

      expect(stack.resolve(s.toStatementJson())).toEqual({
        Effect: 'Allow',
        Action: 'test:Action',
        Principal: { Service: 'codedeploy.amazonaws.com' },
      });
    });

    test('obscure service principals resolve to the user-provided value', () => {
      const stack = new Stack(undefined, undefined, { env: { region: 'cn-north-1' } });
      const s = new PolicyStatement();
      s.addActions('test:Action');
      s.addServicePrincipal('test.service-principal.dev');

      expect(stack.resolve(s.toStatementJson())).toEqual({
        Effect: 'Allow',
        Action: 'test:Action',
        Principal: { Service: 'test.service-principal.dev' },
      });
    });
  });

  describe('CompositePrincipal can be used to represent a principal that has multiple types', () => {
    test('with a single principal', () => {
      const stack = new Stack();
      const p = new CompositePrincipal(new ArnPrincipal('i:am:an:arn'));
      const statement = new PolicyStatement();
      statement.addPrincipals(p);
      expect(stack.resolve(statement.toStatementJson())).toEqual({ Effect: 'Allow', Principal: { AWS: 'i:am:an:arn' } });
    });

    test('conditions are allowed in an assumerolepolicydocument', () => {
      const stack = new Stack();
      new Role(stack, 'Role', {
        assumedBy: new CompositePrincipal(
          new ArnPrincipal('i:am'),
          new FederatedPrincipal('federated', { StringEquals: { 'aws:some-key': 'some-value' } }),
        ),
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { AWS: 'i:am' },
            },
            {
              Action: 'sts:AssumeRole',
              Condition: {
                StringEquals: { 'aws:some-key': 'some-value' },
              },
              Effect: 'Allow',
              Principal: { Federated: 'federated' },
            },
          ],
        },
      });
    });

    test('conditions are not allowed when used in a single statement', () => {
      expect(() => {
        new PolicyStatement({
          actions: ['s3:test'],
          principals: [new CompositePrincipal(
            new ArnPrincipal('i:am'),
            new FederatedPrincipal('federated', { StringEquals: { 'aws:some-key': 'some-value' } }))],
        });
      }).toThrow(/Components of a CompositePrincipal must not have conditions/);
    });

    test('principals and conditions are a big nice merge', () => {
      const stack = new Stack();
      // add via ctor
      const p = new CompositePrincipal(
        new ArnPrincipal('i:am:an:arn'),
        new ServicePrincipal('amazon.com'));

      // add via `addPrincipals` (with condition)
      p.addPrincipals(
        new Anyone(),
        new ServicePrincipal('another.service'),
      );

      const statement = new PolicyStatement();
      statement.addPrincipals(p);

      // add via policy statement
      statement.addArnPrincipal('aws-principal-3');
      statement.addCondition('cond2', { boom: '123' });

      expect(stack.resolve(statement.toStatementJson())).toEqual({
        Condition: {
          cond2: { boom: '123' },
        },
        Effect: 'Allow',
        Principal: {
          AWS: ['i:am:an:arn', '*', 'aws-principal-3'],
          Service: ['amazon.com', 'another.service'],
        },
      });
    });

    test('can mix types of assumeRoleAction in a single composite', () => {
      const stack = new Stack();

      // WHEN
      new Role(stack, 'Role', {
        assumedBy: new CompositePrincipal(
          new ArnPrincipal('arn'),
          new FederatedPrincipal('fed', {}, 'sts:Boom')),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { AWS: 'arn' },
            },
            {
              Action: 'sts:Boom',
              Effect: 'Allow',
              Principal: { Federated: 'fed' },
            },
          ],
        },
      });
    });
  });

  describe('PrincipalWithConditions can be used to add a principal with conditions', () => {
    test('includes conditions from both the wrapped principal and the wrapper', () => {
      const stack = new Stack();
      const principalOpts = {
        conditions: {
          BinaryEquals: {
            'principal-key': 'SGV5LCBmcmllbmQh',
          },
        },
      };
      const p = new ServicePrincipal('s3.amazonaws.com', principalOpts)
        .withConditions({ StringEquals: { 'wrapper-key': ['val-1', 'val-2'] } });
      const statement = new PolicyStatement();
      statement.addPrincipals(p);
      expect(stack.resolve(statement.toStatementJson())).toEqual({
        Condition: {
          BinaryEquals: { 'principal-key': 'SGV5LCBmcmllbmQh' },
          StringEquals: { 'wrapper-key': ['val-1', 'val-2'] },
        },
        Effect: 'Allow',
        Principal: {
          Service: 's3.amazonaws.com',
        },
      });
    });

    test('conditions from addCondition are merged with those from the principal', () => {
      const stack = new Stack();
      const p = new AccountPrincipal('012345678900').withConditions({ StringEquals: { key: 'val' } });
      const statement = new PolicyStatement();
      statement.addPrincipals(p);
      statement.addCondition('Null', { 'banned-key': 'true' });
      expect(stack.resolve(statement.toStatementJson())).toEqual({
        Effect: 'Allow',
        Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::012345678900:root']] } },
        Condition: { StringEquals: { key: 'val' }, Null: { 'banned-key': 'true' } },
      });
    });

    test('adding conditions via `withConditions` does not affect the original principal', () => {
      const originalPrincipal = new ArnPrincipal('iam:an:arn');
      const principalWithConditions = originalPrincipal.withConditions({ StringEquals: { key: 'val' } });
      expect(originalPrincipal.policyFragment.conditions).toEqual({});
      expect(principalWithConditions.policyFragment.conditions).toEqual({ StringEquals: { key: 'val' } });
    });

    test('conditions are merged when operators conflict', () => {
      const p = new FederatedPrincipal('fed', {
        OperatorOne: { 'fed-key': 'fed-val' },
        OperatorTwo: { 'fed-key': 'fed-val' },
        OperatorThree: { 'fed-key': 'fed-val' },
      }).withConditions({
        OperatorTwo: { 'with-key': 'with-val' },
        OperatorThree: { 'with-key': 'with-val' },
      });
      const statement = new PolicyStatement();
      statement.addCondition('OperatorThree', { 'add-key': 'add-val' });
      statement.addPrincipals(p);
      expect(statement.toStatementJson()).toEqual({
        Effect: 'Allow',
        Principal: { Federated: 'fed' },
        Condition: {
          OperatorOne: { 'fed-key': 'fed-val' },
          OperatorTwo: { 'fed-key': 'fed-val', 'with-key': 'with-val' },
          OperatorThree: { 'fed-key': 'fed-val', 'with-key': 'with-val', 'add-key': 'add-val' },
        },
      });
    });

    test('tokens can be used in conditions', () => {
      // GIVEN
      const stack = new Stack();
      const statement = new PolicyStatement();

      // WHEN
      const p = new ArnPrincipal('arn:of:principal').withConditions({
        StringEquals: Lazy.any({ produce: () => ({ goo: 'zar' }) }),
      });

      statement.addPrincipals(p);

      // THEN
      const resolved = stack.resolve(statement.toStatementJson());
      expect(resolved).toEqual({
        Condition: {
          StringEquals: {
            goo: 'zar',
          },
        },
        Effect: 'Allow',
        Principal: {
          AWS: 'arn:of:principal',
        },
      });
    });

    test('conditions cannot be merged if they include tokens', () => {
      const p = new FederatedPrincipal('fed', {
        StringEquals: { foo: 'bar' },
      }).withConditions({
        StringEquals: Lazy.any({ produce: () => ({ goo: 'zar' }) }),
      });

      const statement = new PolicyStatement();

      expect(() => statement.addPrincipals(p)).toThrow(/multiple "StringEquals" conditions cannot be merged if one of them contains an unresolved token/);
    });

    test('values passed to `withConditions` overwrite values from the wrapped principal ' +
      'when keys conflict within an operator', () => {
      const p = new FederatedPrincipal('fed', {
        Operator: { key: 'p-val' },
      }).withConditions({
        Operator: { key: 'with-val' },
      });
      const statement = new PolicyStatement();
      statement.addPrincipals(p);
      expect(statement.toStatementJson()).toEqual({
        Effect: 'Allow',
        Principal: { Federated: 'fed' },
        Condition: {
          Operator: { key: 'with-val' },
        },
      });
    });
  });

  describe('duplicate statements', () => {
    test('without tokens', () => {
      // GIVEN
      const stack = new Stack();
      const p = new PolicyDocument();

      const statement = new PolicyStatement();
      statement.addResources('resource1', 'resource2');
      statement.addActions('service:action1', 'service:action2');
      statement.addServicePrincipal('service');
      statement.addConditions({
        a: {
          b: 'c',
        },
        d: {
          e: 'f',
        },
      });

      // WHEN
      p.addStatements(statement);
      p.addStatements(statement);
      p.addStatements(statement);

      // THEN
      expect(stack.resolve(p).Statement).toHaveLength(1);
    });

    test('with tokens', () => {
      // GIVEN
      const stack = new Stack();
      const p = new PolicyDocument();

      const statement1 = new PolicyStatement();
      statement1.addResources(Lazy.string({ produce: () => 'resource' }));
      statement1.addActions(Lazy.string({ produce: () => 'service:action' }));

      const statement2 = new PolicyStatement();
      statement2.addResources(Lazy.string({ produce: () => 'resource' }));
      statement2.addActions(Lazy.string({ produce: () => 'service:action' }));

      // WHEN
      p.addStatements(statement1);
      p.addStatements(statement2);

      // THEN
      expect(stack.resolve(p).Statement).toHaveLength(1);
    });
  });

  test('autoAssignSids enables auto-assignment of a unique SID for each statement', () => {
    // GIVEN
    const doc = new PolicyDocument({
      assignSids: true,
    });

    // WHEN
    doc.addStatements(new PolicyStatement({ actions: ['service:action1'], resources: ['resource1'] }));
    doc.addStatements(new PolicyStatement({ actions: ['service:action1'], resources: ['resource1'] }));
    doc.addStatements(new PolicyStatement({ actions: ['service:action1'], resources: ['resource1'] }));
    doc.addStatements(new PolicyStatement({ actions: ['service:action1'], resources: ['resource1'] }));
    doc.addStatements(new PolicyStatement({ actions: ['service:action2'], resources: ['resource2'] }));

    // THEN
    const stack = new Stack();
    expect(stack.resolve(doc)).toEqual({
      Version: '2012-10-17',
      Statement: [
        { Action: 'service:action1', Effect: 'Allow', Resource: 'resource1', Sid: '0' },
        { Action: 'service:action2', Effect: 'Allow', Resource: 'resource2', Sid: '1' },
      ],
    });
  });

  test('constructor args are equivalent to mutating in-place', () => {
    const stack = new Stack();

    const s = new PolicyStatement();
    s.addActions('service:action1', 'service:action2');
    s.addAllResources();
    s.addArnPrincipal('arn');
    s.addCondition('key', { equals: 'value' });

    const doc1 = new PolicyDocument();
    doc1.addStatements(s);

    const doc2 = new PolicyDocument();
    doc2.addStatements(new PolicyStatement({
      actions: ['service:action1', 'service:action2'],
      resources: ['*'],
      principals: [new ArnPrincipal('arn')],
      conditions: {
        key: { equals: 'value' },
      },
    }));

    expect(stack.resolve(doc1)).toEqual(stack.resolve(doc2));
  });

  describe('fromJson', () => {
    test("throws error when Statement isn't an array", () => {
      expect(() => {
        PolicyDocument.fromJson({
          Statement: 'asdf',
        });
      }).toThrow(/Statement must be an array/);
    });
  });

  test('adding another condition with the same operator does not delete the original', () => {
    const stack = new Stack();

    const p = new PolicyStatement();

    p.addCondition('StringEquals', { 'kms:ViaService': 'service' });

    p.addAccountCondition('12221121221');

    expect(stack.resolve(p.toStatementJson())).toEqual({
      Effect: 'Allow',
      Condition: { StringEquals: { 'kms:ViaService': 'service', 'sts:ExternalId': '12221121221' } },
    });
  });

  test('validation error if policy statement has no actions', () => {
    const policyStatement = new PolicyStatement({
      principals: [new AnyPrincipal()],
    });

    // THEN
    const validationErrorsForResourcePolicy: string[] = policyStatement.validateForResourcePolicy();
    // const validationErrorsForIdentityPolicy: string[] = policyStatement.validateForIdentityPolicy();
    expect(validationErrorsForResourcePolicy).toEqual([
      'A PolicyStatement must specify at least one \'action\' or \'notAction\'.',
      'A PolicyStatement used in a resource-based policy must specify at least one resource.',
    ]);
  });

  test('validation error if policy statement for resource-based policy has no principals specified', () => {
    const policyStatement = new PolicyStatement({
      actions: ['*'],
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy();
    expect(validationErrors).toEqual([
      'A PolicyStatement used in a resource-based policy must specify at least one IAM principal.',
      'A PolicyStatement used in a resource-based policy must specify at least one resource.',
    ]);
  });

  test('validation error if policy statement for resource-based policy has no resources specified', () => {
    const policyStatement = new PolicyStatement({
      actions: ['s3:GetObject'],
      principals: [new AnyPrincipal()],
      // Missing: resources
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy();
    expect(validationErrors).toEqual(['A PolicyStatement used in a resource-based policy must specify at least one resource.']);
  });

  test('validation passes when both principals and resources are specified', () => {
    const policyStatement = new PolicyStatement({
      actions: ['s3:GetObject'],
      principals: [new AnyPrincipal()],
      resources: ['arn:aws:s3:::my-bucket/*'],
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy();
    expect(validationErrors).toEqual([]);
  });

  test('validation passes with NotResource instead of Resource', () => {
    const policyStatement = new PolicyStatement({
      actions: ['s3:GetObject'],
      principals: [new AnyPrincipal()],
      notResources: ['arn:aws:s3:::excluded-bucket/*'],
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy();
    expect(validationErrors).toEqual([]);
  });

  test('validation error if policy statement for resource-based policy has neither principals nor resources', () => {
    const policyStatement = new PolicyStatement({
      actions: ['s3:GetObject'],
      // Missing: both principals and resources
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy();
    expect(validationErrors).toEqual([
      'A PolicyStatement used in a resource-based policy must specify at least one IAM principal.',
      'A PolicyStatement used in a resource-based policy must specify at least one resource.',
    ]);
  });

  test('validation error if policy statement for trust policy has no principals specified', () => {
    const policyStatement = new PolicyStatement({
      actions: ['sts:AssumeRole'],
      // Missing: principals (resources not needed for trust policies)
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForTrustPolicy();
    expect(validationErrors).toEqual(['A PolicyStatement used in a trust policy must specify at least one IAM principal.']);
  });

  test('validation passes for trust policy with principals but no resources', () => {
    const policyStatement = new PolicyStatement({
      actions: ['sts:AssumeRole'],
      principals: [new AnyPrincipal()],
      // No resources needed for trust policies
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForTrustPolicy();
    expect(validationErrors).toEqual([]);
  });

  test('validation passes for resource-based policy with skipResourceValidation option', () => {
    const policyStatement = new PolicyStatement({
      actions: ['ecr:GetDownloadUrlForLayer'],
      principals: [new AnyPrincipal()],
      // Missing: resources (but validation is skipped)
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy({ skipResourceValidation: true });
    expect(validationErrors).toEqual([]);
  });

  test('validation still requires principals even with skipResourceValidation', () => {
    const policyStatement = new PolicyStatement({
      actions: ['ecr:GetDownloadUrlForLayer'],
      // Missing: both principals and resources
    });

    // THEN
    const validationErrors: string[] = policyStatement.validateForResourcePolicy({ skipResourceValidation: true });
    expect(validationErrors).toEqual(['A PolicyStatement used in a resource-based policy must specify at least one IAM principal.']);
  });
});
