import { InputTransformation, Pipe } from '@aws-cdk/aws-pipes-alpha';
import { App, Stack, Resource } from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import * as iam from 'aws-cdk-lib/aws-iam';
import type { IPipeline, PipelineReference } from 'aws-cdk-lib/aws-sagemaker';
import { TestSource } from './test-classes';
import { SageMakerTarget } from '../lib';

describe('SageMaker', () => {
  let app: App;
  let stack: Stack;
  let pipeline: IPipeline;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app, 'TestStack');
    pipeline = new FakePipeline(stack, 'MyPipeline', { pipelineName: 'MyPipeline' });
  });

  it('should have only target arn', () => {
    // ARRANGE
    const target = new SageMakerTarget(pipeline);

    new Pipe(stack, 'MyPipe', {
      source: new TestSource(),
      target,
    });

    // ACT
    const template = Template.fromStack(stack);

    // ASSERT
    template.hasResourceProperties('AWS::Pipes::Pipe', {
      Target: pipeline.pipelineArn,
      TargetParameters: {},
    });
  });

  it('should have target parameters', () => {
    // ARRANGE
    const target = new SageMakerTarget(pipeline, {
      pipelineParameters: {
        foo: 'bar',
      },
    });

    new Pipe(stack, 'MyPipe', {
      source: new TestSource(),
      target,
    });

    // ACT
    const template = Template.fromStack(stack);

    // ASSERT
    template.hasResourceProperties('AWS::Pipes::Pipe', {
      TargetParameters: {
        SageMakerPipelineParameters: {
          PipelineParameterList: [
            {
              Name: 'foo',
              Value: 'bar',
            },
          ],
        },
      },
    });
  });

  it('should have input transformation', () => {
    // ARRANGE
    const inputTransformation = InputTransformation.fromObject({
      key: 'value',
    });
    const target = new SageMakerTarget(pipeline, {
      inputTransformation,
    });

    new Pipe(stack, 'MyPipe', {
      source: new TestSource(),
      target,
    });

    // ACT
    const template = Template.fromStack(stack);

    // ASSERT
    template.hasResourceProperties('AWS::Pipes::Pipe', {
      TargetParameters: {
        InputTemplate: '{"key":"value"}',
      },
    });
  });

  it('should grant pipe role push access', () => {
    // ARRANGE
    const target = new SageMakerTarget(pipeline);

    new Pipe(stack, 'MyPipe', {
      source: new TestSource(),
      target,
    });

    // ACT
    const template = Template.fromStack(stack);

    // ASSERT
    expect(template.findResources('AWS::IAM::Role')).toMatchSnapshot();
    expect(template.findResources('AWS::IAM::Policy')).toMatchSnapshot();
  });
});

interface FakePipelineProps {
  readonly pipelineName: string;
}

class FakePipeline extends Resource implements IPipeline {
  public readonly pipelineArn;

  public readonly pipelineName;
  constructor(scope: Stack, id: string, props: FakePipelineProps) {
    super(scope, id);
    this.pipelineArn = `arn:aws:sagemaker:us-east-1:111111111111:pipeline/${props.pipelineName}`;
    this.pipelineName = props.pipelineName;
  }

  public grantStartPipelineExecution(grantee: iam.IGrantable): iam.Grant {
    return iam.Grant.addToPrincipal({
      grantee,
      actions: ['sagemaker:StartPipelineExecution'],
      resourceArns: [this.pipelineArn],
    });
  }

  public get pipelineRef(): PipelineReference {
    return { pipelineName: this.pipelineName };
  }
}
