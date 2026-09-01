import { Template, Match } from '../../../assertions';
import * as codebuild from '../../../aws-codebuild';
import * as codecommit from '../../../aws-codecommit';
import * as codepipeline from '../../../aws-codepipeline';
import { LambdaFunction } from '../../../aws-events-targets';
import * as iam from '../../../aws-iam';
import * as kms from '../../../aws-kms';
import { Code, Function, Runtime } from '../../../aws-lambda';
import * as s3 from '../../../aws-s3';
import { Stack, Lazy, App } from '../../../core';
import { CODECOMMIT_SOURCE_ACTION_DEFAULT_BRANCH_NAME } from '../../../cx-api';
import * as cpactions from '../../lib';
import type { CodeCommitSourceActionProps } from '../../lib';

/* eslint-disable @stylistic/quote-props */

describe('CodeCommit Source Action', () => {
  describe('CodeCommit Source Action', () => {
    test('by default does not poll for source changes and uses Events', () => {
      const stack = new Stack();

      minimalPipeline(stack, undefined);

      Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Actions': [
              {
                'Configuration': {
                  'PollForSourceChanges': false,
                },
              },
            ],
          },
          {},
        ],
      });

      Template.fromStack(stack).resourceCountIs('AWS::Events::Rule', 1);
    });

    test('cross-account CodeCommit Repository Source does not use target role in source stack', () => {
      // Test for https://github.com/aws/aws-cdk/issues/15639
      const app = new App();
      const sourceStack = new Stack(app, 'SourceStack', { env: { account: '1234', region: 'north-pole' } });
      const targetStack = new Stack(app, 'TargetStack', { env: { account: '5678', region: 'north-pole' } });

      const repo = new codecommit.Repository(sourceStack, 'MyRepo', {
        repositoryName: 'my-repo',
      });

      const sourceOutput = new codepipeline.Artifact();
      new codepipeline.Pipeline(targetStack, 'MyPipeline', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({ actionName: 'Source', repository: repo, output: sourceOutput }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: new codebuild.PipelineProject(targetStack, 'MyProject'),
                input: sourceOutput,
              }),
            ],
          },
        ],
      });

      // THEN - creates a Rule in the source stack targeting the pipeline stack's event bus using a generated role
      Template.fromStack(sourceStack).hasResourceProperties('AWS::Events::Rule', {
        EventPattern: {
          source: ['aws.codecommit'],
          resources: [
            { 'Fn::GetAtt': ['MyRepoF4F48043', 'Arn'] },
          ],
        },
        Targets: [{
          RoleARN: Match.absent(),
          Arn: {
            'Fn::Join': ['', [
              'arn:',
              { 'Ref': 'AWS::Partition' },
              ':events:north-pole:5678:event-bus/default',
            ]],
          },
        }],
      });

      // THEN - creates a Rule in the pipeline stack using the role to start the pipeline
      Template.fromStack(targetStack).hasResourceProperties('AWS::Events::Rule', {
        'EventPattern': {
          'source': [
            'aws.codecommit',
          ],
          'resources': [
            {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { 'Ref': 'AWS::Partition' },
                  ':codecommit:north-pole:1234:my-repo',
                ],
              ],
            },
          ],
        },
        'Targets': [
          {
            'Arn': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':codepipeline:north-pole:5678:',
                { 'Ref': 'MyPipelineAED38ECF' },
              ]],
            },
            'RoleArn': { 'Fn::GetAtt': ['MyPipelineEventsRoleFAB99F32', 'Arn'] },
          },
        ],
      });
    });

    test('does not poll for source changes and uses Events for CodeCommitTrigger.EVENTS', () => {
      const stack = new Stack();

      minimalPipeline(stack, cpactions.CodeCommitTrigger.EVENTS);

      Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Actions': [
              {
                'Configuration': {
                  'PollForSourceChanges': false,
                },
              },
            ],
          },
          {},
        ],
      });

      Template.fromStack(stack).resourceCountIs('AWS::Events::Rule', 1);
    });

    test('polls for source changes and does not use Events for CodeCommitTrigger.POLL', () => {
      const stack = new Stack();

      minimalPipeline(stack, cpactions.CodeCommitTrigger.POLL);

      Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Actions': [
              {
                'Configuration': {
                  'PollForSourceChanges': true,
                },
              },
            ],
          },
          {},
        ],
      });

      Template.fromStack(stack).resourceCountIs('AWS::Events::Rule', 0);
    });

    test('does not poll for source changes and does not use Events for CodeCommitTrigger.NONE', () => {
      const stack = new Stack();

      minimalPipeline(stack, cpactions.CodeCommitTrigger.NONE);

      Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Actions': [
              {
                'Configuration': {
                  'PollForSourceChanges': false,
                },
              },
            ],
          },
          {},
        ],
      });

      Template.fromStack(stack).resourceCountIs('AWS::Events::Rule', 0);
    });

    test('check when a custom event is provided', () => {
      const stack = new Stack();

      const eventPattern
        = {
          'detail-type': ['CodeCommit Repository State Change'],
          'resources': ['foo'],
          'source': ['aws.codecommit'],
          'detail': {
            referenceType: ['branch'],
            event: ['referenceCreated', 'referenceUpdated'],
            referenceName: ['test-branch'],
          },
        };

      minimalPipeline(stack, cpactions.CodeCommitTrigger.EVENTS, {
        customEventRule: {
          eventPattern,
          target: new LambdaFunction(new Function(stack, 'TestFunction', {
            runtime: Runtime.NODEJS_LATEST,
            handler: 'index.handler',
            code: Code.fromInline('exports.handler = handler.toString()'),
          })),
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
        'EventPattern': {
          'source': [
            'aws.codecommit',
          ],
          'resources': ['foo', {
            'Fn::GetAtt': [
              'MyRepoF4F48043',
              'Arn',
            ],
          }],
          'detail-type': [
            'CodeCommit Repository State Change',
          ],
          'detail': {
            'referenceType': [
              'branch',
            ],
            'event': [
              'referenceCreated',
              'referenceUpdated',
            ],
            'referenceName': [
              'test-branch',
            ],
          },
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        'Handler': 'index.handler',
        'Code': {
          'ZipFile': 'exports.handler = handler.toString()',
        },
      });
    });

    test('cannot be created with an empty branch', () => {
      const stack = new Stack();
      const repo = new codecommit.Repository(stack, 'MyRepo', {
        repositoryName: 'my-repo',
      });

      expect(() => {
        new cpactions.CodeCommitSourceAction({
          actionName: 'Source2',
          repository: repo,
          output: new codepipeline.Artifact(),
          branch: '',
        });
      }).toThrow(/'branch' parameter cannot be an empty string/);
    });

    test('allows using the same repository multiple times with different branches when trigger=EVENTS', () => {
      const stack = new Stack();

      const repo = new codecommit.Repository(stack, 'MyRepo', {
        repositoryName: 'my-repo',
      });
      const sourceOutput1 = new codepipeline.Artifact();
      const sourceOutput2 = new codepipeline.Artifact();
      new codepipeline.Pipeline(stack, 'MyPipeline', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({
                actionName: 'Source1',
                repository: repo,
                output: sourceOutput1,
              }),
              new cpactions.CodeCommitSourceAction({
                actionName: 'Source2',
                repository: repo,
                output: sourceOutput2,
                branch: 'develop',
              }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: new codebuild.PipelineProject(stack, 'MyProject'),
                input: sourceOutput1,
              }),
            ],
          },
        ],
      });
    });

    test('exposes variables for other actions to consume', () => {
      const stack = new Stack();

      const sourceOutput = new codepipeline.Artifact();
      const codeCommitSourceAction = new cpactions.CodeCommitSourceAction({
        actionName: 'Source',
        repository: new codecommit.Repository(stack, 'MyRepo', {
          repositoryName: 'my-repo',
        }),
        output: sourceOutput,
      });
      new codepipeline.Pipeline(stack, 'Pipeline', {
        stages: [
          {
            stageName: 'Source',
            actions: [codeCommitSourceAction],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: new codebuild.PipelineProject(stack, 'MyProject'),
                input: sourceOutput,
                environmentVariables: {
                  AuthorDate: { value: codeCommitSourceAction.variables.authorDate },
                },
              }),
            ],
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Name': 'Source',
          },
          {
            'Name': 'Build',
            'Actions': [
              {
                'Name': 'Build',
                'Configuration': {
                  'EnvironmentVariables': '[{"name":"AuthorDate","type":"PLAINTEXT","value":"#{Source_Source_NS.AuthorDate}"}]',
                },
              },
            ],
          },
        ],
      });
    });

    test('allows using a Token for the branch name', () => {
      const stack = new Stack();

      const sourceOutput = new codepipeline.Artifact();
      new codepipeline.Pipeline(stack, 'P', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({
                actionName: 'CodeCommit',
                repository: new codecommit.Repository(stack, 'R', {
                  repositoryName: 'repository',
                }),
                branch: Lazy.string({ produce: () => 'my-branch' }),
                output: sourceOutput,
              }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: new codebuild.PipelineProject(stack, 'CodeBuild'),
                input: sourceOutput,
              }),
            ],
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
        EventPattern: {
          detail: {
            referenceName: ['my-branch'],
          },
        },
      });
    });

    test('allows to enable full clone', () => {
      const stack = new Stack();

      const sourceOutput = new codepipeline.Artifact();
      new codepipeline.Pipeline(stack, 'P', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({
                actionName: 'CodeCommit',
                repository: new codecommit.Repository(stack, 'R', {
                  repositoryName: 'repository',
                }),
                branch: Lazy.string({ produce: () => 'my-branch' }),
                output: sourceOutput,
                codeBuildCloneOutput: true,
              }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: new codebuild.PipelineProject(stack, 'CodeBuild'),
                input: sourceOutput,
              }),
            ],
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Name': 'Source',
            'Actions': [{
              'Configuration': {
                'OutputArtifactFormat': 'CODEBUILD_CLONE_REF',
              },
            }],
          },
          {
            'Name': 'Build',
            'Actions': [
              {
                'Name': 'Build',
              },
            ],
          },
        ],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([
            Match.objectLike({
              'Action': [
                'logs:CreateLogGroup',
                'logs:CreateLogStream',
                'logs:PutLogEvents',
              ],
            }),
            Match.objectLike({
              'Action': 'codecommit:GitPull',
              'Effect': 'Allow',
              'Resource': {
                'Fn::GetAtt': [
                  'RC21A1702',
                  'Arn',
                ],
              },
            }),
          ]),
        },
      });

      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
        'PolicyDocument': {
          'Statement': Match.arrayWith([
            Match.objectLike({
              'Action': [
                'codecommit:GetBranch',
                'codecommit:GetCommit',
                'codecommit:UploadArchive',
                'codecommit:GetUploadArchiveStatus',
                'codecommit:CancelUploadArchive',
                'codecommit:GetRepository',
              ],
              'Effect': 'Allow',
              'Resource': {
                'Fn::GetAtt': [
                  'RC21A1702',
                  'Arn',
                ],
              },
            }),
          ]),
        },
      });
    });

    test('uses the role when passed', () => {
      const stack = new Stack();

      const pipeline = new codepipeline.Pipeline(stack, 'P', {
        pipelineName: 'MyPipeline',
      });

      const triggerEventTestRole = new iam.Role(stack, 'Trigger-test-role', {
        assumedBy: new iam.ServicePrincipal('events.amazonaws.com'),
      });
      triggerEventTestRole.addToPolicy(new iam.PolicyStatement({
        actions: ['codepipeline:StartPipelineExecution'],
        resources: [pipeline.pipelineArn],
      }));

      const sourceOutput = new codepipeline.Artifact();

      const sourceAction = new cpactions.CodeCommitSourceAction({
        actionName: 'CodeCommit',
        repository: new codecommit.Repository(stack, 'R', {
          repositoryName: 'repository',
        }),
        branch: Lazy.string({ produce: () => 'my-branch' }),
        output: sourceOutput,
        eventRole: triggerEventTestRole,
      });

      pipeline.addStage({
        stageName: 'Source',
        actions: [sourceAction],
      });

      const buildAction = new cpactions.CodeBuildAction({
        actionName: 'Build',
        project: new codebuild.PipelineProject(stack, 'CodeBuild'),
        input: sourceOutput,
      });

      pipeline.addStage({
        stageName: 'build',
        actions: [buildAction],
      });

      Template.fromStack(stack).hasResourceProperties('AWS::Events::Rule', {
        Targets: [
          {
            Arn: stack.resolve(pipeline.pipelineArn),
            Id: 'Target0',
            RoleArn: stack.resolve(triggerEventTestRole.roleArn),
          },
        ],
      });
    });

    test('grants explicit s3:PutObjectAcl permissions when the Actions is cross-account', () => {
      const app = new App();

      const repoStack = new Stack(app, 'RepoStack', {
        env: { account: '123', region: 'us-east-1' },
      });
      const repoFomAnotherAccount = codecommit.Repository.fromRepositoryName(repoStack, 'Repo', 'my-repo');

      const pipelineStack = new Stack(app, 'PipelineStack', {
        env: { account: '456', region: 'us-east-1' },
      });
      new codepipeline.Pipeline(pipelineStack, 'Pipeline', {
        artifactBucket: s3.Bucket.fromBucketAttributes(pipelineStack, 'PipelineBucket', {
          bucketName: 'pipeline-bucket',
          encryptionKey: kms.Key.fromKeyArn(pipelineStack, 'PipelineKey',
            'arn:aws:kms:us-east-1:123456789012:key/my-key'),
        }),
        stages: [
          {
            stageName: 'Source',
            actions: [new cpactions.CodeCommitSourceAction({
              actionName: 'Source',
              output: new codepipeline.Artifact(),
              repository: repoFomAnotherAccount,
            })],
          },
          {
            stageName: 'Approve',
            actions: [new cpactions.ManualApprovalAction({
              actionName: 'Approve',
            })],
          },
        ],
      });

      Template.fromStack(repoStack).hasResourceProperties('AWS::IAM::Policy', {
        PolicyDocument: {
          Statement: Match.arrayWith([{
            'Action': [
              's3:PutObjectAcl',
              's3:PutObjectVersionAcl',
            ],
            'Effect': 'Allow',
            'Resource': {
              'Fn::Join': ['', [
                'arn:',
                { 'Ref': 'AWS::Partition' },
                ':s3:::pipeline-bucket/*',
              ]],
            },
          }]),
        },
      });
    });

    test('allows using a new Repository from another Stack as a source of the Pipeline, with Events', () => {
      const app = new App();

      const repoStack = new Stack(app, 'RepositoryStack');
      const repo = new codecommit.Repository(repoStack, 'Repository', {
        repositoryName: 'my-repo',
      });

      const pipelineStack = new Stack(app, 'PipelineStack');
      const sourceOutput = new codepipeline.Artifact();
      new codepipeline.Pipeline(pipelineStack, 'Pipeline', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({
                actionName: 'Source',
                repository: repo,
                output: sourceOutput,
              }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: codebuild.Project.fromProjectName(pipelineStack, 'Project', 'my-project'),
                input: sourceOutput,
              }),
            ],
          },
        ],
      });

      // If the Event Rule was created in the repo's Stack,
      // we would have a cycle
      // (the repo's Stack would need the name of the CodePipeline to trigger through the Rule,
      // while the pipeline's Stack would need the name of the Repository to use as a Source).
      // By moving the Rule to pipeline's Stack, we get rid of the cycle.
      Template.fromStack(pipelineStack).resourceCountIs('AWS::Events::Rule', 1);
    });

    test('using main as the default branch when feature flag is set', () => {
      const defaultBranchFeatureFlag = { [CODECOMMIT_SOURCE_ACTION_DEFAULT_BRANCH_NAME]: true };
      const app = new App({ context: defaultBranchFeatureFlag });

      const repoStack = new Stack(app, 'RepositoryStack');
      const repo = new codecommit.Repository(repoStack, 'Repository', {
        repositoryName: 'my-repo',
      });

      const pipelineStack = new Stack(app, 'PipelineStack');

      const sourceOutput = new codepipeline.Artifact();
      new codepipeline.Pipeline(pipelineStack, 'Pipeline', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({
                actionName: 'Source',
                repository: repo,
                output: sourceOutput,
              }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: codebuild.Project.fromProjectName(pipelineStack, 'Project', 'my-project'),
                input: sourceOutput,
              }),
            ],
          },
        ],
      });

      const template = Template.fromStack(pipelineStack);
      template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Actions': [
              {
                'Configuration': {
                  'BranchName': 'main',
                },
              },
            ],
          },
          {},
        ],
      });
      template.hasResourceProperties('AWS::Events::Rule', {
        'EventPattern': {
          'source': [
            'aws.codecommit',
          ],
          'resources': [
            {},
          ],
          'detail-type': [
            'CodeCommit Repository State Change',
          ],
          'detail': {
            'event': [
              'referenceCreated',
              'referenceUpdated',
            ],
            'referenceName': [
              'main',
            ],
          },
        },
      });
    });

    test('using master as the default branch when feature flag is not set', () => {
      const app = new App();

      const repoStack = new Stack(app, 'RepositoryStack');
      const repo = new codecommit.Repository(repoStack, 'Repository', {
        repositoryName: 'my-repo',
      });

      const pipelineStack = new Stack(app, 'PipelineStack');

      const sourceOutput = new codepipeline.Artifact();
      new codepipeline.Pipeline(pipelineStack, 'Pipeline', {
        stages: [
          {
            stageName: 'Source',
            actions: [
              new cpactions.CodeCommitSourceAction({
                actionName: 'Source',
                repository: repo,
                output: sourceOutput,
              }),
            ],
          },
          {
            stageName: 'Build',
            actions: [
              new cpactions.CodeBuildAction({
                actionName: 'Build',
                project: codebuild.Project.fromProjectName(pipelineStack, 'Project', 'my-project'),
                input: sourceOutput,
              }),
            ],
          },
        ],
      });

      const template = Template.fromStack(pipelineStack);
      template.hasResourceProperties('AWS::CodePipeline::Pipeline', {
        'Stages': [
          {
            'Actions': [
              {
                'Configuration': {
                  'BranchName': 'master',
                },
              },
            ],
          },
          {},
        ],
      });
      template.hasResourceProperties('AWS::Events::Rule', {
        'EventPattern': {
          'source': [
            'aws.codecommit',
          ],
          'resources': [
            {},
          ],
          'detail-type': [
            'CodeCommit Repository State Change',
          ],
          'detail': {
            'event': [
              'referenceCreated',
              'referenceUpdated',
            ],
            'referenceName': [
              'master',
            ],
          },
        },
      });
    });
  });
});

function minimalPipeline(stack: Stack, trigger?: cpactions.CodeCommitTrigger, customEventRule?: Pick<CodeCommitSourceActionProps, 'customEventRule'>): codepipeline.Pipeline {
  const sourceOutput = new codepipeline.Artifact();
  return new codepipeline.Pipeline(stack, 'MyPipeline', {
    stages: [
      {
        stageName: 'Source',
        actions: [
          new cpactions.CodeCommitSourceAction({
            actionName: 'Source',
            repository: new codecommit.Repository(stack, 'MyRepo', {
              repositoryName: 'my-repo',
            }),
            output: sourceOutput,
            trigger,
            customEventRule: customEventRule?.customEventRule,
          }),
        ],
      },
      {
        stageName: 'Build',
        actions: [
          new cpactions.CodeBuildAction({
            actionName: 'Build',
            project: new codebuild.PipelineProject(stack, 'MyProject'),
            input: sourceOutput,
          }),
        ],
      },
    ],
  });
}
