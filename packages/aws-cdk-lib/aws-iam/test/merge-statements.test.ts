import { App, Lazy, Stack } from '../../core';
import * as iam from '../lib';
import { PolicyStatement } from '../lib';

const PRINCIPAL_ARN1 = 'arn:aws:iam::111111111111:user/user-name';
const principal1 = new iam.ArnPrincipal(PRINCIPAL_ARN1);

const PRINCIPAL_ARN2 = 'arn:aws:iam::111111111111:role/role-name';
const principal2 = new iam.ArnPrincipal(PRINCIPAL_ARN2);

// Check that 'resource' statements are merged, and that 'notResource' statements are not,
// if the statements are otherwise the same.
test.each([
  ['resources', true],
  ['notResources', false],
] as Array<['resources' | 'notResources', boolean]>)
('merge %p statements: %p', (key, doMerge) => {
  assertMergedC(doMerge, [
    new iam.PolicyStatement({
      [key]: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      [key]: ['b'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: ['a', 'b'],
      Action: 'service:Action',
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

// Check that 'action' statements are merged, and that 'notAction' statements are not,
// if the statements are otherwise the same.
test.each([
  ['actions', true],
  ['notActions', false],
] as Array<['actions' | 'notActions', boolean]>)
('merge %p statements: %p', (key, doMerge) => {
  assertMergedC(doMerge, [
    new iam.PolicyStatement({
      resources: ['a'],
      [key]: ['service:Action1'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['a'],
      [key]: ['service:Action2'],
      principals: [principal1],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: 'a',
      Action: ['service:Action1', 'service:Action2'],
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

// Check that 'principal' statements are merged, and that 'notPrincipal' statements are not,
// if the statements are otherwise the same.
test.each([
  ['principals', true],
  ['notPrincipals', false],
] as Array<['principals' | 'notPrincipals', boolean]>)
('merge %p statements: %p', (key, doMerge) => {
  assertMergedC(doMerge, [
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      [key]: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      [key]: [principal2],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: 'a',
      Action: 'service:Action',
      Principal: { AWS: [PRINCIPAL_ARN1, PRINCIPAL_ARN2].sort() },
    },
  ]);
});

test('merge multiple types of principals', () => {
  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [new iam.ServicePrincipal('service.amazonaws.com')],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: 'a',
      Action: 'service:Action',
      Principal: {
        AWS: PRINCIPAL_ARN1,
        Service: 'service.amazonaws.com',
      },
    },
  ]);
});

test('multiple mergeable keys are not merged', () => {
  assertNoMerge([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action1'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['b'],
      actions: ['service:Action2'],
      principals: [principal1],
    }),
  ]);
});

test('can merge statements without principals', () => {
  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
    }),
    new iam.PolicyStatement({
      resources: ['b'],
      actions: ['service:Action'],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: ['a', 'b'],
      Action: 'service:Action',
    },
  ]);
});

test('if conditions are different, statements are not merged', () => {
  assertNoMerge([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
      conditions: {
        StringLike: {
          something: 'value',
        },
      },
    }),
    new iam.PolicyStatement({
      resources: ['b'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
  ]);
});

test('if conditions are the same, statements are merged', () => {
  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
      conditions: {
        StringLike: {
          something: 'value',
        },
      },
    }),
    new iam.PolicyStatement({
      resources: ['b'],
      actions: ['service:Action'],
      principals: [principal1],
      conditions: {
        StringLike: {
          something: 'value',
        },
      },
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: ['a', 'b'],
      Action: 'service:Action',
      Principal: { AWS: PRINCIPAL_ARN1 },
      Condition: {
        StringLike: {
          something: 'value',
        },
      },
    },
  ]);
});

test('also merge Deny statements', () => {
  assertMerged([
    new iam.PolicyStatement({
      effect: iam.Effect.DENY,
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      effect: iam.Effect.DENY,
      resources: ['b'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
  ], [
    {
      Effect: 'Deny',
      Resource: ['a', 'b'],
      Action: 'service:Action',
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

test('merges 3 statements in multiple steps', () => {
  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['b'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    // This can combine with the previous two once they have been merged
    new iam.PolicyStatement({
      resources: ['a', 'b'],
      actions: ['service:Action2'],
      principals: [principal1],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: ['a', 'b'],
      Action: ['service:Action', 'service:Action2'],
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

test('winnow down literal duplicates', () => {
  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['a', 'b'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: ['a', 'b'],
      Action: 'service:Action',
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

test('winnow down literal duplicates if they are Refs', () => {
  const stack = new Stack();
  const user1 = new iam.User(stack, 'User1');
  const user2 = new iam.User(stack, 'User2');

  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [user1],
    }),
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [user1, user2],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: 'a',
      Action: 'service:Action',
      Principal: {
        AWS: [
          { 'Fn::GetAtt': ['User1E278A736', 'Arn'] },
          { 'Fn::GetAtt': ['User21F1486D1', 'Arn'] },
        ],
      },
    },
  ]);
});

test('merges 2 pairs separately', () => {
  // Merges pairs (0,2) and (1,3)
  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['c'],
      actions: ['service:Action1'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['b'],
      actions: ['service:Action'],
      principals: [principal1],
    }),
    new iam.PolicyStatement({
      resources: ['c'],
      actions: ['service:Action2'],
      principals: [principal1],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: ['a', 'b'],
      Action: 'service:Action',
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
    {
      Effect: 'Allow',
      Resource: 'c',
      Action: ['service:Action1', 'service:Action2'],
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

test('do not deep-merge info Refs and GetAtts', () => {
  const stack = new Stack();
  const user1 = new iam.User(stack, 'User1');
  const user2 = new iam.User(stack, 'User2');

  assertMerged([
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [user1],
    }),
    new iam.PolicyStatement({
      resources: ['a'],
      actions: ['service:Action'],
      principals: [user2],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: 'a',
      Action: 'service:Action',
      Principal: {
        AWS: [
          { 'Fn::GetAtt': ['User1E278A736', 'Arn'] },
          { 'Fn::GetAtt': ['User21F1486D1', 'Arn'] },
        ],
      },
    },
  ]);
});

test('properly merge untyped principals (star)', () => {
  const statements = [
    PolicyStatement.fromJson({
      Action: ['service:Action1'],
      Effect: 'Allow',
      Resource: ['Resource'],
      Principal: '*',
    }),
    PolicyStatement.fromJson({
      Action: ['service:Action2'],
      Effect: 'Allow',
      Resource: ['Resource'],
      Principal: '*',
    }),
  ];

  assertMerged(statements, [
    {
      Action: ['service:Action1', 'service:Action2'],
      Effect: 'Allow',
      Resource: 'Resource',
      Principal: '*',
    },
  ]);
});

test('fail merging typed and untyped principals', () => {
  const statements = [
    PolicyStatement.fromJson({
      Action: ['service:Action'],
      Effect: 'Allow',
      Resource: ['Resource'],
      Principal: '*',
    }),
    PolicyStatement.fromJson({
      Action: ['service:Action'],
      Effect: 'Allow',
      Resource: ['Resource'],
      Principal: { AWS: PRINCIPAL_ARN1 },
    }),
  ];

  assertMerged(statements, [
    {
      Action: 'service:Action',
      Effect: 'Allow',
      Resource: 'Resource',
      Principal: '*',
    },
    {
      Action: 'service:Action',
      Effect: 'Allow',
      Resource: 'Resource',
      Principal: { AWS: PRINCIPAL_ARN1 },
    },
  ]);
});

test('keep merging even if it requires multiple passes', () => {
  // [A, R1], [B, R1], [A, R2], [B, R2]
  // -> [{A, B}, R1], [{A, B], R2]
  // -> [{A, B}, {R1, R2}]
  assertMerged([
    new iam.PolicyStatement({
      actions: ['service:A'],
      resources: ['R1'],
    }),
    new iam.PolicyStatement({
      actions: ['service:B'],
      resources: ['R1'],
    }),
    new iam.PolicyStatement({
      actions: ['service:A'],
      resources: ['R2'],
    }),
    new iam.PolicyStatement({
      actions: ['service:B'],
      resources: ['R2'],
    }),
  ], [
    {
      Effect: 'Allow',
      Action: ['service:A', 'service:B'],
      Resource: ['R1', 'R2'],
    },
  ]);
});

test('lazily generated statements are merged correctly', () => {
  assertMerged([
    new LazyStatement((s) => {
      s.addActions('service:A');
      s.addResources('R1');
    }),
    new LazyStatement((s) => {
      s.addActions('service:B');
      s.addResources('R1');
    }),
  ], [
    {
      Effect: 'Allow',
      Action: ['service:A', 'service:B'],
      Resource: 'R1',
    },
  ]);
});

test('merge statements if resource is a lazy', () => {
  const stack = new Stack();
  const user1 = new iam.User(stack, 'User1');
  const user2 = new iam.User(stack, 'User2');
  const resourceToken = Lazy.string({
    produce: () => 'a',
  });

  assertMerged([
    new iam.PolicyStatement({
      resources: [resourceToken],
      actions: ['service:Action'],
      principals: [user1],
    }),
    new iam.PolicyStatement({
      resources: [resourceToken],
      actions: ['service:Action'],
      principals: [user2],
    }),
  ], [
    {
      Effect: 'Allow',
      Resource: 'a',
      Action: 'service:Action',
      Principal: {
        AWS: [
          { 'Fn::GetAtt': ['User1E278A736', 'Arn'] },
          { 'Fn::GetAtt': ['User21F1486D1', 'Arn'] },
        ],
      },
    },
  ]);
});

function assertNoMerge(statements: iam.PolicyStatement[]) {
  const app = new App();
  const stack = new Stack(app, 'Stack');

  const regularResult = stack.resolve(new iam.PolicyDocument({ minimize: false, statements }));
  const minResult = stack.resolve(new iam.PolicyDocument({ minimize: true, statements }));

  expect(minResult).toEqual(regularResult);
}

function assertMerged(statements: iam.PolicyStatement[], expected: any[]) {
  const app = new App();
  const stack = new Stack(app, 'Stack');

  const minResult = stack.resolve(new iam.PolicyDocument({ minimize: true, statements }));

  expect(minResult.Statement).toEqual(expected);
}

/**
 * Assert Merged Conditional
 *
 * Based on a boolean, either call assertMerged or assertNoMerge. The 'expected'
 * argument only applies in the case where `doMerge` is true.
 */
function assertMergedC(doMerge: boolean, statements: iam.PolicyStatement[], expected: any[]) {
  return doMerge ? assertMerged(statements, expected) : assertNoMerge(statements);
}

/**
 * A statement that fills itself only when freeze() is called.
 */
class LazyStatement extends iam.PolicyStatement {
  constructor(private readonly modifyMe: (x: iam.PolicyStatement) => void) {
    super();
  }

  public freeze() {
    this.modifyMe(this);
    return super.freeze();
  }
}
