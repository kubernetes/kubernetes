import { testDeprecated } from '@aws-cdk/cdk-build-tools';
import type { Construct } from 'constructs';
import { Match, Template } from '../../assertions';
import type * as cloudwatch from '../../aws-cloudwatch';
import * as iam from '../../aws-iam';
import * as cdk from '../../core';
import * as stepfunctions from '../lib';

describe('State Machine Resources', () => {
  testDeprecated('Tasks can add permissions to the execution role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new stepfunctions.Task(stack, 'Task', {
      task: {
        bind: () => ({
          resourceArn: 'resource',
          policyStatements: [new iam.PolicyStatement({
            actions: ['resource:Everything'],
            resources: ['arn:aws:s3:::my-bucket/my-object'],
          })],
        }),
      },
    });

    // WHEN
    new stepfunctions.StateMachine(stack, 'SM', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 'resource:Everything',
            Effect: 'Allow',
            Resource: 'arn:aws:s3:::my-bucket/my-object',
          },
        ],
      },
    });
  }),

  test('Tasks hidden inside a Parallel state are also included', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task', {
      policies: [
        new iam.PolicyStatement({
          actions: ['resource:Everything'],
          resources: ['arn:aws:s3:::my-bucket/my-object'],
        }),
      ],
    });

    const para = new stepfunctions.Parallel(stack, 'Para');
    para.branch(task);

    // WHEN
    new stepfunctions.StateMachine(stack, 'SM', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(para),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Action: 'resource:Everything',
            Effect: 'Allow',
            Resource: 'arn:aws:s3:::my-bucket/my-object',
          },
        ],
      },
    });
  }),

  test('Fail should render ErrorPath / CausePath correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const fail = new stepfunctions.Fail(stack, 'Fail', {
      errorPath: stepfunctions.JsonPath.stringAt('$.error'),
      causePath: stepfunctions.JsonPath.stringAt('$.cause'),
    });

    // WHEN
    const failState = stack.resolve(fail.toStateJson());

    // THEN
    expect(failState).toStrictEqual({
      CausePath: '$.cause',
      ErrorPath: '$.error',
      Type: 'Fail',
    });
  }),

  test.each([
    [
      "States.Format('error: {}.', $.error)",
      "States.Format('cause: {}.', $.cause)",
    ],
    [
      stepfunctions.JsonPath.format('error: {}.', stepfunctions.JsonPath.stringAt('$.error')),
      stepfunctions.JsonPath.format('cause: {}.', stepfunctions.JsonPath.stringAt('$.cause')),
    ],
  ])('Fail should render ErrorPath / CausePath correctly when specifying ErrorPath / CausePath using intrinsics', (errorPath, causePath) => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);
    const fail = new stepfunctions.Fail(stack, 'Fail', {
      errorPath,
      causePath,
    });

    // WHEN
    const failState = stack.resolve(fail.toStateJson());

    // THEN
    expect(failState).toStrictEqual({
      CausePath: "States.Format('cause: {}.', $.cause)",
      ErrorPath: "States.Format('error: {}.', $.error)",
      Type: 'Fail',
    });
    expect(() => app.synth()).not.toThrow();
  }),

  test('fails in synthesis if error and errorPath are defined in Fail state', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);

    // WHEN
    new stepfunctions.Fail(stack, 'Fail', {
      error: 'error',
      errorPath: '$.error',
    });

    expect(() => app.synth()).toThrow(/Fail state cannot have both error and errorPath/);
  }),

  test('fails in synthesis if cause and causePath are defined in Fail state', () => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);

    // WHEN
    new stepfunctions.Fail(stack, 'Fail', {
      cause: 'cause',
      causePath: '$.cause',
    });

    expect(() => app.synth()).toThrow(/Fail state cannot have both cause and causePath/);
  }),

  test.each([
    'States.Array($.Id)',
    'States.ArrayPartition($.inputArray, 4)',
    'States.ArrayContains($.inputArray, $.lookingFor)',
    'States.ArrayRange(1, 9, 2)',
    'States.ArrayLength($.inputArray)',
    'States.JsonMerge($.json1, $.json2, false)',
    'States.StringToJson($.escapedJsonString)',
    'plainString',
  ])('fails in synthesis if specifying invalid intrinsic functions in the causePath and errorPath (%s)', (intrinsic) => {
    // GIVEN
    const app = new cdk.App();
    const stack = new cdk.Stack(app);

    // WHEN
    new stepfunctions.Fail(stack, 'Fail', {
      causePath: intrinsic,
      errorPath: intrinsic,
    });

    expect(() => app.synth()).toThrow(/You must specify a valid intrinsic function in causePath. Must be one of States.Format, States.JsonToString, States.ArrayGetItem, States.Base64Encode, States.Base64Decode, States.Hash, States.UUID/);
    expect(() => app.synth()).toThrow(/You must specify a valid intrinsic function in errorPath. Must be one of States.Format, States.JsonToString, States.ArrayGetItem, States.Base64Encode, States.Base64Decode, States.Hash, States.UUID/);
  }),

  testDeprecated('Task should render InputPath / Parameters / OutputPath correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new stepfunctions.Task(stack, 'Task', {
      inputPath: '$',
      outputPath: '$.state',
      task: {
        bind: () => ({
          resourceArn: 'resource',
          parameters: {
            'input.$': '$',
            'stringArgument': 'inital-task',
            'numberArgument': 123,
            'booleanArgument': true,
            'arrayArgument': ['a', 'b', 'c'],
          },
        }),
      },
    });

    // WHEN
    const taskState = task.toStateJson();

    // THEN
    expect(taskState).toStrictEqual({
      End: true,
      Retry: undefined,
      Catch: undefined,
      InputPath: '$',
      Parameters:
              {
                'input.$': '$',
                'stringArgument': 'inital-task',
                'numberArgument': 123,
                'booleanArgument': true,
                'arrayArgument': ['a', 'b', 'c'],
              },
      OutputPath: '$.state',
      Type: 'Task',
      Comment: undefined,
      Resource: 'resource',
      ResultPath: undefined,
      TimeoutSeconds: undefined,
      HeartbeatSeconds: undefined,
      Arguments: undefined,
      Output: undefined,
    });
  }),

  testDeprecated('Task combines taskobject parameters with direct parameters', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new stepfunctions.Task(stack, 'Task', {
      inputPath: '$',
      outputPath: '$.state',
      task: {
        bind: () => ({
          resourceArn: 'resource',
          parameters: {
            a: 'aa',
          },
        }),
      },
      parameters: {
        b: 'bb',
      },
    });

    // WHEN
    const taskState = task.toStateJson();

    // THEN
    expect(taskState).toStrictEqual({
      End: true,
      Retry: undefined,
      Catch: undefined,
      InputPath: '$',
      Parameters:
              {
                a: 'aa',
                b: 'bb',
              },
      OutputPath: '$.state',
      Type: 'Task',
      Comment: undefined,
      Resource: 'resource',
      ResultPath: undefined,
      TimeoutSeconds: undefined,
      HeartbeatSeconds: undefined,
      Arguments: undefined,
      Output: undefined,
    });
  }),

  test('Created state machine can grant start execution to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantStartExecution(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([Match.objectLike({
          Action: 'states:StartExecution',
          Effect: 'Allow',
          Resource: {
            Ref: 'StateMachine2E01A3A5',
          },
        })]),
      },
    });
  }),

  test('Created state machine can grant start sync execution to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
      stateMachineType: stepfunctions.StateMachineType.EXPRESS,
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantStartSyncExecution(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([Match.objectLike({
          Action: 'states:StartSyncExecution',
          Effect: 'Allow',
          Resource: {
            Ref: 'StateMachine2E01A3A5',
          },
        })]),
      },
    });
  }),

  test('Created state machine can grant read access to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantRead(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'states:ListExecutions',
              'states:ListStateMachines',
            ],
            Effect: 'Allow',
            Resource: {
              Ref: 'StateMachine2E01A3A5',
            },
          },
          {
            Action: [
              'states:DescribeExecution',
              'states:DescribeStateMachineForExecution',
              'states:GetExecutionHistory',
            ],
            Effect: 'Allow',
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':states:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':execution:',
                  {
                    'Fn::Select': [
                      6,
                      {
                        'Fn::Split': [
                          ':',
                          {
                            Ref: 'StateMachine2E01A3A5',
                          },
                        ],
                      },
                    ],
                  },
                  ':*',
                ],
              ],
            },
          },
          {
            Action: [
              'states:ListActivities',
              'states:DescribeStateMachine',
              'states:DescribeActivity',
            ],
            Effect: 'Allow',
            Resource: '*',
          },
        ],
      },
    },
    );
  }),

  test('Created state machine can grant task response actions to the state machine', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantTaskResponse(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'states:SendTaskSuccess',
              'states:SendTaskFailure',
              'states:SendTaskHeartbeat',
            ],
            Effect: 'Allow',
            Resource: {
              Ref: 'StateMachine2E01A3A5',
            },
          },
        ],
      },
    });
  }),

  test('Created state machine can grant redrive execution access to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantRedriveExecution(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([Match.objectLike({
          Action: 'states:RedriveExecution',
          Effect: 'Allow',
          Resource: {
            'Fn::Join': [
              '',
              [
                'arn:',
                {
                  Ref: 'AWS::Partition',
                },
                ':states:',
                {
                  Ref: 'AWS::Region',
                },
                ':',
                {
                  Ref: 'AWS::AccountId',
                },
                ':execution:',
                {
                  'Fn::Select': [
                    6,
                    {
                      'Fn::Split': [
                        ':',
                        {
                          Ref: 'StateMachine2E01A3A5',
                        },
                      ],
                    },
                  ],
                },
                ':*',
              ],
            ],
          },
        })]),
      },
    });
  }),

  test('Created state machine can grant actions to the executions', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantExecution(role, 'states:GetExecutionHistory');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'states:GetExecutionHistory',
            Effect: 'Allow',
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':states:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':execution:',
                  {
                    'Fn::Select': [
                      6,
                      {
                        'Fn::Split': [
                          ':',
                          {
                            Ref: 'StateMachine2E01A3A5',
                          },
                        ],
                      },
                    ],
                  },
                  ':*',
                ],
              ],
            },
          },
        ],
      },
    });
  }),

  test('Created state machine can grant actions to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task');
    const stateMachine = new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grant(role, 'states:ListExecution');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'states:ListExecution',
            Effect: 'Allow',
            Resource: {
              Ref: 'StateMachine2E01A3A5',
            },
          },
        ],
      },
    });
  }),

  test('Imported state machine can grant start execution to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const stateMachineArn = 'arn:aws:states:::my-state-machine';
    const stateMachine = stepfunctions.StateMachine.fromStateMachineArn(stack, 'StateMachine', stateMachineArn);
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantStartExecution(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'states:StartExecution',
            Effect: 'Allow',
            Resource: stateMachineArn,
          },
        ],
        Version: '2012-10-17',
      },
      PolicyName: 'RoleDefaultPolicy5FFB7DAB',
      Roles: [
        {
          Ref: 'Role1ABCC5F0',
        },
      ],
    });
  }),

  test('Imported state machine can grant read access to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const stateMachineArn = 'arn:aws:states:::my-state-machine';
    const stateMachine = stepfunctions.StateMachine.fromStateMachineArn(stack, 'StateMachine', stateMachineArn);
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantRead(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'states:ListExecutions',
              'states:ListStateMachines',
            ],
            Effect: 'Allow',
            Resource: stateMachineArn,
          },
          {
            Action: [
              'states:DescribeExecution',
              'states:DescribeStateMachineForExecution',
              'states:GetExecutionHistory',
            ],
            Effect: 'Allow',
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  {
                    Ref: 'AWS::Partition',
                  },
                  ':states:',
                  {
                    Ref: 'AWS::Region',
                  },
                  ':',
                  {
                    Ref: 'AWS::AccountId',
                  },
                  ':execution:*',
                ],
              ],
            },
          },
          {
            Action: [
              'states:ListActivities',
              'states:DescribeStateMachine',
              'states:DescribeActivity',
            ],
            Effect: 'Allow',
            Resource: '*',
          },
        ],
      },
    },
    );
  }),

  test('Imported state machine can task response permissions to the state machine', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const stateMachineArn = 'arn:aws:states:::my-state-machine';
    const stateMachine = stepfunctions.StateMachine.fromStateMachineArn(stack, 'StateMachine', stateMachineArn);
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grantTaskResponse(role);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              'states:SendTaskSuccess',
              'states:SendTaskFailure',
              'states:SendTaskHeartbeat',
            ],
            Effect: 'Allow',
            Resource: stateMachineArn,
          },
        ],
      },
    });
  }),

  test('Imported state machine can grant access to a role', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const stateMachineArn = 'arn:aws:states:::my-state-machine';
    const stateMachine = stepfunctions.StateMachine.fromStateMachineArn(stack, 'StateMachine', stateMachineArn);
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    // WHEN
    stateMachine.grant(role, 'states:ListExecution');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: 'states:ListExecution',
            Effect: 'Allow',
            Resource: stateMachine.stateMachineArn,
          },
        ],
      },
    });
  }),

  test('Imported state machine can provide metrics', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const stateMachineArn = 'arn:aws:states:us-east-1:123456789012:stateMachine:my-state-machine';
    const stateMachine = stepfunctions.StateMachine.fromStateMachineArn(stack, 'StateMachine', stateMachineArn);
    const color = '#00ff00';

    // WHEN
    const metrics = new Array<cloudwatch.Metric>();
    metrics.push(stateMachine.metricAborted({ color }));
    metrics.push(stateMachine.metricFailed({ color }));
    metrics.push(stateMachine.metricStarted({ color }));
    metrics.push(stateMachine.metricSucceeded({ color }));
    metrics.push(stateMachine.metricThrottled({ color }));
    metrics.push(stateMachine.metricTime({ color }));
    metrics.push(stateMachine.metricTimedOut({ color }));

    // THEN
    for (const metric of metrics) {
      expect(metric.namespace).toEqual('AWS/States');
      expect(metric.dimensions).toEqual({ StateMachineArn: stateMachineArn });
      expect(metric.color).toEqual(color);
    }
  }),

  test('Pass with JSONPath should render InputPath / Parameters / OutputPath correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = stepfunctions.Pass.jsonPath(stack, 'Pass', {
      stateName: 'my-pass-state',
      inputPath: '$',
      outputPath: '$.state',
      parameters: {
        'input.$': '$',
        'stringArgument': 'inital-task',
        'numberArgument': 123,
        'booleanArgument': true,
        'arrayArgument': ['a', 'b', 'c'],
      },
    });

    // WHEN
    const taskState = task.toStateJson();

    // THEN
    expect(taskState).toStrictEqual({
      End: true,
      InputPath: '$',
      OutputPath: '$.state',
      Parameters:
              {
                'input.$': '$',
                'stringArgument': 'inital-task',
                'numberArgument': 123,
                'booleanArgument': true,
                'arrayArgument': ['a', 'b', 'c'],
              },
      Type: 'Pass',
      QueryLanguage: undefined,
      Comment: undefined,
      Result: undefined,
      ResultPath: undefined,
      Arguments: undefined,
      Output: undefined,
      Assign: undefined,
    });
  }),

  test('Pass with JSONata should render Output correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const pass = stepfunctions.Pass.jsonata(stack, 'Pass', {
      outputs: {
        input: '{% $states.input %}',
        stringArgument: 'inital-task',
        numberArgument: 123,
        booleanArgument: true,
        arrayArgument: ['a', 'b', 'c'],
      },
    });

    // WHEN
    const passState = pass.toStateJson();

    // THEN
    expect(passState).toStrictEqual({
      Type: 'Pass',
      QueryLanguage: 'JSONata',
      Output: {
        input: '{% $states.input %}',
        stringArgument: 'inital-task',
        numberArgument: 123,
        booleanArgument: true,
        arrayArgument: ['a', 'b', 'c'],
      },
      End: true,
      Comment: undefined,
      InputPath: undefined,
      Result: undefined,
      ResultPath: undefined,
      Parameters: undefined,
      OutputPath: undefined,
      Arguments: undefined,
      Assign: undefined,
    });
  }),

  test('Pass with JSONata should render Assign as is', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = stepfunctions.Pass.jsonata(stack, 'Pass', {
      assign: {
        id: '{% $states.input.order.id %}',
        products: [
          {
            id: '{% $states.input.order.product.id %}',
            currentPrice: '{% $states.result.Payload.current_price %}',
            name: '{% $states.input.order.product.name %}',
          },
        ],
        count: 42,
        available: true,
      },
    });

    // WHEN
    const taskState = task.toStateJson();

    // THEN
    expect(taskState).toStrictEqual({
      Type: 'Pass',
      QueryLanguage: 'JSONata',
      Assign: {
        id: '{% $states.input.order.id %}',
        products: [
          {
            id: '{% $states.input.order.product.id %}',
            currentPrice: '{% $states.result.Payload.current_price %}',
            name: '{% $states.input.order.product.name %}',
          },
        ],
        count: 42,
        available: true,
      },
      End: true,
      Comment: undefined,
      InputPath: undefined,
      Result: undefined,
      ResultPath: undefined,
      Parameters: undefined,
      OutputPath: undefined,
      Arguments: undefined,
      Output: undefined,
    });
  }),

  test('Pass with JSONPath should render Assign correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = stepfunctions.Pass.jsonPath(stack, 'Pass', {
      assign: {
        details: {
          status: 'SUCCESS',
          lineItems: stepfunctions.JsonPath.stringAt('$.order.items'),
        },
        resultCode: stepfunctions.JsonPath.stringAt('$.result.code'),
        message: stepfunctions.JsonPath.format('Hello {}', stepfunctions.JsonPath.stringAt('$customer.name')),
        startTime: stepfunctions.JsonPath.stringAt('$$.Execution.StartTime'),
      },
    });

    // WHEN
    const taskState = task.toStateJson();

    // THEN
    expect(taskState).toStrictEqual({
      Type: 'Pass',
      Assign: {
        'details': {
          'status': 'SUCCESS',
          'lineItems.$': '$.order.items',
        },
        'resultCode.$': '$.result.code',
        'message.$': "States.Format('Hello {}', $customer.name)",
        'startTime.$': '$$.Execution.StartTime',
      },
      End: true,
      QueryLanguage: undefined,
      Comment: undefined,
      InputPath: undefined,
      Result: undefined,
      ResultPath: undefined,
      Parameters: undefined,
      OutputPath: undefined,
      Arguments: undefined,
      Output: undefined,
    });
  }),

  test('parameters can be selected from the input with a path', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new stepfunctions.Pass(stack, 'Pass', {
      parameters: {
        input: stepfunctions.JsonPath.stringAt('$.myField'),
      },
    });

    // WHEN
    const taskState = task.toStateJson();

    // THEN
    expect(taskState).toEqual({
      End: true,
      Parameters:
              { 'input.$': '$.myField' },
      Type: 'Pass',
    });
  }),

  test('State machines must depend on their roles', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const task = new FakeTask(stack, 'Task', {
      policies: [
        new iam.PolicyStatement({
          resources: ['arn:aws:s3:::my-bucket/my-object'],
          actions: ['lambda:InvokeFunction'],
        }),
      ],
    });
    new stepfunctions.StateMachine(stack, 'StateMachine', {
      definitionBody: stepfunctions.DefinitionBody.fromChainable(task),
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::StepFunctions::StateMachine', {
      DependsOn: [
        'StateMachineRoleDefaultPolicyDF1E6607',
        'StateMachineRoleB840431D',
      ],
    });
  }),

  test('Choice with JSONata should render Assign/Output correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const nextState = stepfunctions.Pass.jsonata(stack, 'Pass');
    const choice = stepfunctions.Choice.jsonata(stack, 'Choice', {
      assign: {
        a: '{% $default.a %}',
        b: 'default.b',
      },
      outputs: {
        a: '{% $states.input.default.a %}',
        b: 'input.default.b',
      },
    }).otherwise(nextState);
    choice.when(stepfunctions.Condition.jsonata('{% true %}'), nextState, {
      assign: {
        a: '{% $when.a %}',
        b: 'when.b',
      },
      outputs: {
        a: '{% $states.input.when.a %}',
        b: 'input.when.b',
      },
    });

    // WHEN
    const choiceState = choice.toStateJson();

    // THEN
    expect(choiceState).toStrictEqual({
      Type: 'Choice',
      Assign: {
        a: '{% $default.a %}',
        b: 'default.b',
      },
      Output: {
        a: '{% $states.input.default.a %}',
        b: 'input.default.b',
      },
      Choices: [
        {
          Condition: '{% true %}',
          Next: 'Pass',
          Assign: {
            a: '{% $when.a %}',
            b: 'when.b',
          },
          Output: {
            a: '{% $states.input.when.a %}',
            b: 'input.when.b',
          },
          Comment: undefined,
        },
      ],
      Default: 'Pass',
      QueryLanguage: 'JSONata',
      Comment: undefined,
      InputPath: undefined,
      Parameters: undefined,
      OutputPath: undefined,
      Arguments: undefined,
    });
  }),

  test('Choice with JSONPath should render Assign correctly', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const nextState = stepfunctions.Pass.jsonPath(stack, 'Pass');
    const choice = stepfunctions.Choice.jsonPath(stack, 'Choice', {
      assign: {
        a: stepfunctions.JsonPath.stringAt('$default.a'),
        b: 'default.b',
      },
    }).otherwise(nextState);
    choice.when(stepfunctions.Condition.stringEquals('$foo', 'bar'), nextState, {
      assign: {
        a: stepfunctions.JsonPath.stringAt('$when.a'),
        b: 'when.b',
      },
    });

    // WHEN
    const choiceState = choice.toStateJson();

    // THEN
    expect(choiceState).toStrictEqual({
      Type: 'Choice',
      Assign: {
        'a.$': '$default.a',
        'b': 'default.b',
      },
      Choices: [
        {
          StringEquals: 'bar',
          Variable: '$foo',
          Next: 'Pass',
          Assign: {
            'a.$': '$when.a',
            'b': 'when.b',
          },
          Comment: undefined,
        },
      ],
      Default: 'Pass',
      QueryLanguage: undefined,
      Comment: undefined,
      InputPath: undefined,
      Parameters: undefined,
      OutputPath: undefined,
      Arguments: undefined,
      Output: undefined,
    });
  });
});

interface FakeTaskProps extends stepfunctions.TaskStateBaseProps {
  readonly policies?: iam.PolicyStatement[];
}

class FakeTask extends stepfunctions.TaskStateBase {
  protected readonly taskMetrics?: stepfunctions.TaskMetricsConfig;
  protected readonly taskPolicies?: iam.PolicyStatement[];

  constructor(scope: Construct, id: string, props: FakeTaskProps = {}) {
    super(scope, id, props);
    this.taskPolicies = props.policies;
  }

  protected _renderTask(): any {
    return {
      Resource: 'my-resource',
      Parameters: stepfunctions.FieldUtils.renderObject({
        MyParameter: 'myParameter',
      }),
    };
  }
}
