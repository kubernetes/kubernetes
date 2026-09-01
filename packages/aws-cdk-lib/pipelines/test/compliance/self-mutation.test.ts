import { Match, Template } from '../../../assertions';
import * as cb from '../../../aws-codebuild';
import { Stack, Stage } from '../../../core';
import { CDKP_DEFAULT_CODEBUILD_IMAGE } from '../../lib/private/default-codebuild-image';
import { PIPELINE_ENV, TestApp, ModernTestGitHubNpmPipeline } from '../testhelpers';

let app: TestApp;
let pipelineStack: Stack;

beforeEach(() => {
  app = new TestApp();
  pipelineStack = new Stack(app, 'PipelineStack', { env: PIPELINE_ENV });
});

afterEach(() => {
  app.cleanup();
});

test('CodePipeline has self-mutation stage', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.arrayWith([{
      Name: 'UpdatePipeline',
      Actions: [
        Match.objectLike({
          Name: 'SelfMutate',
          Configuration: Match.objectLike({
            ProjectName: { Ref: Match.anyValue() },
          }),
        }),
      ],
    }]),
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['npm install -g aws-cdk@2'],
          },
          build: {
            commands: Match.arrayWith(['cdk -a . deploy PipelineStack --require-approval=never --verbose']),
          },
        },
      })),
      Type: 'CODEPIPELINE',
    },
  });
});

test('selfmutation stage correctly identifies nested assembly of pipeline stack', () => {
  const pipelineStage = new Stage(app, 'PipelineStage');
  const nestedPipelineStack = new Stack(pipelineStage, 'PipelineStack', { env: PIPELINE_ENV });
  new ModernTestGitHubNpmPipeline(nestedPipelineStack, 'Cdk');

  Template.fromStack(nestedPipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          build: {
            commands: Match.arrayWith(['cdk -a assembly-PipelineStage deploy PipelineStage/PipelineStack --require-approval=never --verbose']),
          },
        },
      })),
    },
  });
});

test('selfmutation feature can be turned off', () => {
  // WHEN
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    selfMutation: false,
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
    Stages: Match.not(Match.arrayWith([{
      Name: 'UpdatePipeline',
      Actions: Match.anyValue(),
    }])),
  });
});

test('can control fix/CLI version used in pipeline selfupdate', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    pipelineName: 'vpipe',
    cliVersion: '1.2.3',
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Name: 'vpipe-selfupdate',
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['npm install -g aws-cdk@1.2.3'],
          },
        },
      })),
    },
  });
});

test('Pipeline stack itself can use assets (has implications for selfupdate)', () => {
  // WHEN
  new ModernTestGitHubNpmPipeline(pipelineStack, 'PrivilegedPipeline', {
    dockerEnabledForSelfMutation: true,
  });

  // THEN
  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      PrivilegedMode: true,
    },
  });
});

test('self-update project role uses tagged bootstrap-role permissions', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk');

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([
        {
          Action: 'sts:AssumeRole',
          Effect: 'Allow',
          Resource: 'arn:*:iam::123456789012:role/*',
          Condition: {
            'ForAnyValue:StringEquals': {
              'iam:ResourceTag/aws-cdk:bootstrap-role': ['image-publishing', 'file-publishing', 'deploy'],
            },
          },
        },
        {
          Action: 'cloudformation:DescribeStacks',
          Effect: 'Allow',
          Resource: '*',
        },
        {
          Action: 's3:ListBucket',
          Effect: 'Allow',
          Resource: '*',
        },
      ]),
    },
  });
});

test('self-mutation stage can be customized with BuildSpec', () => {
  new ModernTestGitHubNpmPipeline(pipelineStack, 'Cdk', {
    selfMutationCodeBuildDefaults: {
      partialBuildSpec: cb.BuildSpec.fromObject({
        phases: {
          install: {
            commands: ['npm config set registry example.com'],
          },
        },
        cache: {
          paths: ['node_modules'],
        },
      }),
    },
  });

  Template.fromStack(pipelineStack).hasResourceProperties('AWS::CodeBuild::Project', {
    Environment: {
      Image: CDKP_DEFAULT_CODEBUILD_IMAGE.imageId,
      PrivilegedMode: false,
    },
    Source: {
      BuildSpec: Match.serializedJson(Match.objectLike({
        phases: {
          install: {
            commands: ['npm config set registry example.com', 'npm install -g aws-cdk@2'],
          },
          build: {
            commands: Match.arrayWith(['cdk -a . deploy PipelineStack --require-approval=never --verbose']),
          },
        },
        cache: {
          paths: ['node_modules'],
        },
      })),
      Type: 'CODEPIPELINE',
    },
  });
});
