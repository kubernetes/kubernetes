import { Template, Match } from '../../../assertions';
import * as codepipeline from '../../../aws-codepipeline';
import * as s3 from '../../../aws-s3';
import * as stepfunction from '../../../aws-stepfunctions';
import { Stack, Stage } from '../../../core';
import * as cpactions from '../../lib';

describe('StepFunctions Invoke Action', () => {
  test('Verify stepfunction configuration properties are set to specific values', () => {
    const stack = new Stack();

    // when
    minimalPipeline(stack);

    // then
    Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', Match.objectLike({
      Stages: [
        //  Must have a source stage
        {
          Actions: [
            {
              ActionTypeId: {
                Category: 'Source',
                Owner: 'AWS',
                Provider: 'S3',
                Version: '1',
              },
              Configuration: {
                S3Bucket: {
                  Ref: 'MyBucketF68F3FF0',
                },
                S3ObjectKey: 'some/path/to',
              },
            },
          ],
        },
        // Must have stepfunction invoke action configuration
        {
          Actions: [
            {
              ActionTypeId: {
                Category: 'Invoke',
                Owner: 'AWS',
                Provider: 'StepFunctions',
                Version: '1',
              },
              Configuration: {
                StateMachineArn: {
                  Ref: 'SimpleStateMachineE8E2CF40',
                },
                InputType: 'Literal',
                // JSON Stringified input when the input type is Literal
                Input: '{\"IsHelloWorldExample\":true}',
              },
            },
          ],
        },
      ],
    }));
  });

  test('Allows the pipeline to invoke this stepfunction', () => {
    const stack = new Stack();

    minimalPipeline(stack);

    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyName: 'MyPipelineInvokeCodePipelineActionRoleDefaultPolicy07A602B1',
      PolicyDocument: {
        Statement: Match.arrayWith([
          {
            Action: ['states:StartExecution', 'states:DescribeStateMachine'],
            Resource: {
              Ref: 'SimpleStateMachineE8E2CF40',
            },
            Effect: 'Allow',
          },
        ]),
      },
    });

    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 4);
  });

  test('Allows the pipeline to describe this stepfunction execution', () => {
    const stack = new Stack();

    minimalPipeline(stack);

    const arnParts = {
      'Fn::Split': [':', { Ref: 'SimpleStateMachineE8E2CF40' }],
    };
    Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {},
          {
            Action: 'states:DescribeExecution',
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { 'Fn::Select': [1, arnParts] },
                  ':states:',
                  { 'Fn::Select': [3, arnParts] },
                  ':',
                  { 'Fn::Select': [4, arnParts] },
                  ':execution:',
                  { 'Fn::Select': [6, arnParts] },
                  ':*',
                ],
              ],
            },
            Effect: 'Allow',
          },
        ],
      },
    });

    Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 4);
  });

  test('Allows the pipeline to describe this stepfunction execution (across accounts & regions)', () => {
    const stack = new Stack(undefined, undefined, { env: { account: '111111111111', region: 'us-east-1' } });

    minimalPipeline(stack, '999999999999', 'bermuda-triangle-1337');

    // The permissions are defined by the cross-account stack here...
    const cfnStack = Stage.of(stack)?.synth().stacks.find(({ stackName }) => stackName === 'Default-support-999999999999');
    expect(cfnStack).toBeDefined();

    Template.fromJSON(cfnStack!.template).hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {},
          {
            Action: 'states:DescribeExecution',
            Resource: 'arn:aws:states:bermuda-triangle-1337:999999999999:execution:SimpleStateMachine:*',
            Effect: 'Allow',
          },
        ],
      },
    });
  });
});

function minimalPipeline(stack: Stack, account?: string, region?: string): codepipeline.IStage {
  const sourceOutput = new codepipeline.Artifact();
  const simpleStateMachine = account || region
    ? stepfunction.StateMachine.fromStateMachineArn(stack, 'SimpleStateMachine', `arn:aws:states:${region}:${account}:stateMachine:SimpleStateMachine`)
    : new stepfunction.StateMachine(stack, 'SimpleStateMachine', {
      definitionBody: stepfunction.DefinitionBody.fromChainable(new stepfunction.Pass(stack, 'StartState')),
    });
  const pipeline = new codepipeline.Pipeline(stack, 'MyPipeline');
  const sourceStage = pipeline.addStage({
    stageName: 'Source',
    actions: [
      new cpactions.S3SourceAction({
        actionName: 'Source',
        bucket: new s3.Bucket(stack, 'MyBucket'),
        bucketKey: 'some/path/to',
        output: sourceOutput,
        trigger: cpactions.S3Trigger.POLL,
      }),
    ],
  });
  pipeline.addStage({
    stageName: 'Invoke',
    actions: [
      new cpactions.StepFunctionInvokeAction({
        actionName: 'Invoke',
        stateMachine: simpleStateMachine,
        stateMachineInput: cpactions.StateMachineInput.literal({ IsHelloWorldExample: true }),
      }),
    ],
  });

  return sourceStage;
}
